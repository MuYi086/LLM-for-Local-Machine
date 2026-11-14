"""提供单模型聊天路由，并以共享 GPU 租约管理 llama.cpp。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from llm_local_runtime.telemetry import TelemetryMiddleware, numeric_fields

from llm_local_runtime import (
    GpuLockTimeoutError,
    GpuRuntimeMetrics,
    gpu_quarantine_path,
    gpu_runtime_lock,
)

from .context_budget import ContextBudgetExceeded, ContextBudgetUnavailable, check_context_budget
from .runtime import (
    MODEL_ALIAS,
    create_quarantine_marker,
    remove_quarantine_marker,
)

API_ROUTE = "/v1/chat/completions"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
PROPS_PATH = "/props"
LOGGER = logging.getLogger(__name__)


def _positive_float(name: str, default: float) -> float:
    """读取有限正数配置。"""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} 必须为正数。") from exc
    if not 0 < value < float("inf"):
        raise ValueError(f"{name} 必须为有限正数。")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    """读取有限非负数配置。"""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} 必须为非负数。") from exc
    if not 0 <= value < float("inf"):
        raise ValueError(f"{name} 必须为有限非负数。")
    return value


def _optional_pid(name: str) -> int | None:
    """读取可选的正进程号配置。"""
    value = os.getenv(name)
    if value is None:
        return None
    try:
        pid = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数。") from exc
    if pid <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return pid


def _validate_loopback_url(value: str) -> str:
    """拒绝把模型推理请求转发到非 loopback 地址。"""
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("llama.cpp backend 必须使用本机 HTTP 地址。")
    if parsed.hostname != "localhost":
        try:
            if not ipaddress.ip_address(parsed.hostname).is_loopback:
                raise ValueError("llama.cpp backend 必须使用本机 loopback 地址。")
        except ValueError as exc:
            raise ValueError("llama.cpp backend 必须使用本机 loopback 地址。") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("llama.cpp backend URL 不得包含凭据、query 或 fragment。")
    if parsed.path not in {"", "/"}:
        raise ValueError("llama.cpp backend URL 必须指向根路径。")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("llama.cpp backend URL 端口无效。") from exc
    return value.rstrip("/")


def create_app(
    *,
    api_route: str = API_ROUTE,
    model_alias: str = MODEL_ALIAS,
    backend_base_url: str | None = None,
    backend_host: str | None = None,
    backend_port: int | None = None,
    backend_env_prefix: str = "LLAMA_CPP",
    backend_pid: int | None = None,
    backend_pgid: int | None = None,
    backend_executable: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    lock_path: str | Path | None = None,
    sleep_timeout: float | None = None,
    sleep_poll_interval: float | None = None,
    cuda_release_delay: float | None = None,
) -> FastAPI:
    """建立单模型 API，并为每次推理申请共享 GPU 租约。

    Args:
        backend_base_url: llama-server 的 loopback 根地址。
        api_route: 对外暴露的聊天路由。
        model_alias: 固定转发给 llama-server 的模型别名。
        backend_host: backend 绑定的 loopback 地址。
        backend_port: backend 监听端口。
        backend_env_prefix: backend 进程身份环境变量的前缀。
        backend_pid: backend 进程号，用于 quarantine 恢复检查。
        backend_pgid: backend 进程组号，用于 quarantine 恢复检查。
        backend_executable: backend 可执行文件路径，用于恢复时校验进程。
        transport: 测试时替换网络传输的 HTTPX transport。
        lock_path: 所有 GPU provider 共用的文件锁路径。
        sleep_timeout: 等待 llama-server 卸载模型的最长时间。
        sleep_poll_interval: 查询 llama-server sleep-idle 状态的间隔。
        cuda_release_delay: 确认 sleep-idle 后释放 GPU 锁前的缓冲时间。

    Returns:
        配置好的 FastAPI 应用。
    """
    base_url_env = (
        "LLAMA_CPP_BASE_URL"
        if backend_env_prefix == "LLAMA_CPP"
        else f"{backend_env_prefix}_LLAMA_BASE_URL"
    )
    base_url = _validate_loopback_url(
        backend_base_url or os.getenv(base_url_env, "http://127.0.0.1:8504")
    )
    request_timeout = _positive_float(f"{backend_env_prefix}_REQUEST_TIMEOUT", 1800.0)
    lock_timeout = _positive_float("GPU_LOCK_WAIT_TIMEOUT", 900.0)
    resolved_sleep_timeout = (
        sleep_timeout
        if sleep_timeout is not None
        else _positive_float(f"{backend_env_prefix}_SLEEP_TIMEOUT", 300.0)
    )
    resolved_poll_interval = (
        sleep_poll_interval
        if sleep_poll_interval is not None
        else _positive_float(f"{backend_env_prefix}_SLEEP_POLL_INTERVAL", 0.25)
    )
    resolved_cuda_release_delay = (
        cuda_release_delay
        if cuda_release_delay is not None
        else _nonnegative_float(
            f"{backend_env_prefix}_CUDA_RELEASE_DELAY",
            _nonnegative_float("CUDA_RELEASE_DELAY", 2.0),
        )
    )
    if resolved_cuda_release_delay < 0:
        raise ValueError("CUDA_RELEASE_DELAY 必须为非负数。")
    resolved_lock_path = Path(
        lock_path or os.getenv("GPU_LOCK_FILE", "storage/.cache/runtime/gpu-runtime.lock")
    )
    resolved_backend_host = backend_host or os.getenv(
        f"{backend_env_prefix}_BACKEND_HOST", "127.0.0.1"
    )
    default_backend_port_variable = (
        "LLAMA_CPP_PORT"
        if backend_env_prefix == "LLAMA_CPP"
        else f"{backend_env_prefix}_LLAMA_CPP_PORT"
    )
    resolved_backend_port = (
        backend_port
        if backend_port is not None
        else int(os.getenv(default_backend_port_variable, "8504"))
    )
    if (
        resolved_backend_host != "localhost"
        and not ipaddress.ip_address(resolved_backend_host).is_loopback
    ):
        raise ValueError("backend_host 必须是 loopback 地址。")
    if not 1 <= resolved_backend_port <= 65535:
        raise ValueError("backend_port 必须在 1 到 65535 之间。")
    parsed_backend_url = urlsplit(base_url)
    if (
        parsed_backend_url.hostname != resolved_backend_host
        or parsed_backend_url.port != resolved_backend_port
    ):
        raise ValueError("backend_base_url 必须与 backend_host 和 backend_port 一致。")
    resolved_backend_pid = (
        backend_pid
        if backend_pid is not None
        else _optional_pid(f"{backend_env_prefix}_BACKEND_PID")
    )
    resolved_backend_pgid = (
        backend_pgid
        if backend_pgid is not None
        else _optional_pid(f"{backend_env_prefix}_BACKEND_PGID")
    )
    resolved_backend_executable = backend_executable or os.getenv("LLAMA_SERVER_BIN")
    timeout = httpx.Timeout(request_timeout, connect=min(request_timeout, 5.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.llama_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )
        try:
            yield
        finally:
            await app.state.llama_client.aclose()

    app = FastAPI(title=f"{model_alias} API", lifespan=lifespan)
    app.add_middleware(TelemetryMiddleware, service=backend_env_prefix.lower(), paths={api_route})

    async def acquire_lease() -> AbstractContextManager[GpuRuntimeMetrics]:
        """在线程池等待共享 GPU 锁，避免阻塞异步事件循环。"""
        lease = gpu_runtime_lock(
            resolved_lock_path,
            f"llama.cpp/{model_alias}",
            timeout_seconds=lock_timeout,
        )
        try:
            await asyncio.to_thread(lease.__enter__)
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return lease

    async def release_lease(lease: AbstractContextManager[GpuRuntimeMetrics]) -> None:
        """在线程池释放共享 GPU 锁和显存采样器。"""
        await asyncio.to_thread(lease.__exit__, None, None, None)

    async def backend_is_sleeping() -> bool:
        """查询休眠状态；暂时不可用或响应格式无效时保留隔离状态。"""
        client: httpx.AsyncClient = app.state.llama_client
        try:
            response = await client.get(PROPS_PATH, timeout=5.0)
            props = response.json() if response.is_success else None
        except (httpx.RequestError, ValueError):
            return False
        return isinstance(props, dict) and props.get("is_sleeping") is True

    async def wait_for_sleep_and_clear(marker_content: bytes) -> bool:
        """确认模型休眠、等待 CUDA 缓冲并清除本次隔离标记。"""
        deadline = time.monotonic() + resolved_sleep_timeout
        while time.monotonic() < deadline:
            if await backend_is_sleeping():
                remaining = deadline - time.monotonic()
                if resolved_cuda_release_delay > remaining:
                    return False
                if resolved_cuda_release_delay:
                    await asyncio.sleep(resolved_cuda_release_delay)
                if await backend_is_sleeping() and remove_quarantine_marker(
                    resolved_lock_path,
                    expected_content=marker_content,
                ):
                    return True
            await asyncio.sleep(resolved_poll_interval)
        return False

    async def wait_until_sleeping() -> bool:
        """仅在 backend 稳定报告休眠后提交请求，避开 sleep-idle 竞态窗口。"""
        deadline = time.monotonic() + resolved_sleep_timeout
        while time.monotonic() < deadline:
            if await backend_is_sleeping():
                return True
            await asyncio.sleep(resolved_poll_interval)
        return False

    async def begin_request() -> tuple[
        AbstractContextManager[GpuRuntimeMetrics], bytes, httpx.AsyncClient
    ]:
        """获取 GPU 租约并创建可供恢复脚本识别的请求标记。"""
        lease = await acquire_lease()
        try:
            marker_content = create_quarantine_marker(
                resolved_lock_path,
                host=resolved_backend_host,
                port=resolved_backend_port,
                pid=resolved_backend_pid,
                pgid=resolved_backend_pgid,
                executable=resolved_backend_executable,
                model_alias=model_alias,
            )
        except OSError as exc:
            await release_lease(lease)
            raise HTTPException(status_code=503, detail="GPU 隔离标记写入失败。") from exc
        return lease, marker_content, app.state.llama_client

    @app.get("/v1/health")
    async def health(request: Request) -> dict[str, str]:
        """检查 llama-server 连通性，不触发推理或加载模型。"""
        try:
            response = await request.app.state.llama_client.get("/health")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="llama.cpp 服务暂不可用。") from exc
        if not response.is_success:
            raise HTTPException(status_code=503, detail="llama.cpp 服务尚未就绪。")
        return {"status": "ok", "backend": "llama.cpp"}

    @app.post(api_route)
    async def chat_completions(request: Request) -> Response:
        """在共享 GPU 租约内转发聊天请求并等待模型卸载。"""
        try:
            payload: Any = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="请求正文必须为有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="请求正文必须是 JSON 对象。")

        context_guard = payload.pop("llm_context_guard", False)
        if not isinstance(context_guard, bool):
            raise HTTPException(status_code=400, detail="llm_context_guard 必须为布尔值。")

        payload["model"] = model_alias
        trace = request.scope.get("model_telemetry")
        if trace:
            if payload.get("stream") is True:
                options = payload.get("stream_options")
                if options is None:
                    payload["stream_options"] = {"include_usage": True}
                elif isinstance(options, dict):
                    options.setdefault("include_usage", True)
            trace.metadata["generation"] = numeric_fields(
                payload, {"max_tokens", "max_completion_tokens", "temperature", "top_p", "seed"}
            )
            trace.mark("queue_enter")
            trace.metadata.update(
                {
                    "backend_pid": resolved_backend_pid,
                    "model": model_alias,
                    "stream": payload.get("stream") is True,
                }
            )
        lease, marker_content, client = await begin_request()
        if trace:
            trace.mark("lease_acquired")
        lease_released = False
        marker_cleanup_attempted = False
        stream_owns_lease = False

        async def clear_request_marker() -> bool:
            """客户端取消时也等到后端收尾，再按本次请求身份清除隔离标记。"""
            nonlocal marker_cleanup_attempted
            marker_cleanup_attempted = True
            if trace:
                trace.mark("cleanup_start")
            cleanup_task = asyncio.create_task(wait_for_sleep_and_clear(marker_content))
            cancellation_received = False
            while True:
                try:
                    sleeping = await asyncio.shield(cleanup_task)
                    break
                except asyncio.CancelledError:
                    cancellation_received = True
                    if cleanup_task.done():
                        sleeping = cleanup_task.result()
                        break
            if not sleeping:
                LOGGER.error(
                    "%s 模型未确认卸载，GPU quarantine 标记已保留。",
                    model_alias,
                )
            if cancellation_received:
                raise asyncio.CancelledError
            return sleeping

        async def release_once() -> None:
            """清理本次请求的隔离标记，并确保 GPU 租约只释放一次。"""
            nonlocal lease_released
            if not lease_released:
                lease_released = True
                cancellation_error: asyncio.CancelledError | None = None
                try:
                    marker_path = gpu_quarantine_path(resolved_lock_path)
                    if not marker_cleanup_attempted and marker_path.exists():
                        await clear_request_marker()
                except asyncio.CancelledError as exc:
                    cancellation_error = exc
                finally:
                    if trace:
                        trace.mark("cleanup_end")
                    release_task = asyncio.create_task(release_lease(lease))
                    while True:
                        try:
                            await asyncio.shield(release_task)
                            break
                        except asyncio.CancelledError as exc:
                            cancellation_error = cancellation_error or exc
                            if release_task.done():
                                release_task.result()
                                break
                if trace:
                    trace.mark("lease_released")
                if cancellation_error is not None:
                    raise cancellation_error

        try:
            if not await wait_until_sleeping():
                sleeping = await clear_request_marker()
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 请求前未确认模型休眠，GPU 已隔离，请运行恢复流程。",
                    )
                raise HTTPException(status_code=503, detail="llama.cpp 服务尚未进入 sleep-idle。")
            if context_guard:
                if trace:
                    trace.mark("context_guard_start")
                try:
                    budget = await check_context_budget(client, payload)
                except ContextBudgetExceeded as exc:
                    if trace:
                        trace.metadata["context_budget"] = asdict(exc.budget)
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                                "param": "messages",
                                "message": str(exc),
                            }
                        },
                    )
                except ContextBudgetUnavailable as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                finally:
                    if trace:
                        trace.mark("context_guard_end")
                if trace:
                    trace.metadata["context_budget"] = asdict(budget)
            if trace:
                trace.mark("backend_send")
            try:
                upstream = await client.send(
                    client.build_request("POST", CHAT_COMPLETIONS_PATH, json=payload),
                    stream=True,
                )
            except httpx.RequestError as exc:
                sleeping = await clear_request_marker()
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 无法确认模型已卸载，GPU 已隔离，请运行恢复流程。",
                    ) from exc
                raise HTTPException(status_code=503, detail="llama.cpp 服务暂不可用。") from exc

            if trace:
                trace.mark("backend_headers")
            response_headers = (
                {"content-type": upstream.headers["content-type"]}
                if "content-type" in upstream.headers
                else {}
            )

            if payload.get("stream") is True:

                async def stream_response() -> AsyncIterator[bytes]:
                    """逐块转发 SSE，并在关闭上游后确认后端 sleep-idle。"""
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    except asyncio.CancelledError:
                        LOGGER.warning(
                            "%s 流式请求被取消，仍将等待后端收尾后释放 GPU 租约。",
                            model_alias,
                        )
                        raise
                    finally:
                        try:
                            try:
                                await upstream.aclose()
                            except Exception:
                                LOGGER.exception("关闭 %s 上游流失败。", model_alias)
                            await clear_request_marker()
                        finally:
                            await release_once()

                response = StreamingResponse(
                    stream_response(),
                    status_code=upstream.status_code,
                    headers=response_headers,
                )
                stream_owns_lease = True
                return response

            try:
                content = await upstream.aread()
            except httpx.RequestError as exc:
                try:
                    await upstream.aclose()
                finally:
                    sleeping = await clear_request_marker()
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 无法确认模型已卸载，GPU 已隔离，请运行恢复流程。",
                    ) from exc
                raise HTTPException(status_code=503, detail="llama.cpp 服务暂不可用。") from exc
            await upstream.aclose()
            sleeping = await clear_request_marker()
            if not sleeping:
                raise HTTPException(
                    status_code=503,
                    detail="llama.cpp 无法确认模型已卸载，GPU 已隔离，请运行恢复流程。",
                )
            return Response(
                content,
                status_code=upstream.status_code,
                headers=response_headers,
            )
        finally:
            if not stream_owns_lease:
                await release_once()

    return app


app = create_app()
