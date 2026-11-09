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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from llm_local_runtime import GpuLockTimeoutError, GpuRuntimeMetrics, gpu_runtime_lock

from .runtime import (
    MODEL_ALIAS,
    create_quarantine_marker,
    remove_quarantine_marker,
)

API_ROUTE = "/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5"
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
    backend_env_prefix: str = "ORNITH",
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
    base_url = _validate_loopback_url(
        backend_base_url
        or os.getenv(f"{backend_env_prefix}_LLAMA_BASE_URL", "http://127.0.0.1:8402")
    )
    request_timeout = _positive_float("ORNITH_REQUEST_TIMEOUT", 1800.0)
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
        else _nonnegative_float("CUDA_RELEASE_DELAY", 2.0)
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
        if backend_env_prefix == "ORNITH"
        else f"{backend_env_prefix}_LLAMA_CPP_PORT"
    )
    resolved_backend_port = (
        backend_port
        if backend_port is not None
        else int(os.getenv(default_backend_port_variable, "8402"))
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

    async def wait_for_sleep_and_clear(marker_content: bytes) -> bool:
        """确认模型休眠、等待 CUDA 缓冲并清除本次隔离标记。"""
        client: httpx.AsyncClient = app.state.llama_client
        deadline = time.monotonic() + resolved_sleep_timeout
        props_sleeping = False
        while time.monotonic() < deadline:
            try:
                response = await client.get(PROPS_PATH, timeout=5.0)
                props = response.json() if response.is_success else None
                props_sleeping = isinstance(props, dict) and props.get("is_sleeping") is True
            except (httpx.RequestError, json.JSONDecodeError, ValueError):
                props_sleeping = False
            if props_sleeping:
                remaining = deadline - time.monotonic()
                if resolved_cuda_release_delay > remaining:
                    return False
                if resolved_cuda_release_delay:
                    await asyncio.sleep(resolved_cuda_release_delay)
                try:
                    confirmation = await client.get(PROPS_PATH, timeout=5.0)
                    confirmation_props = confirmation.json() if confirmation.is_success else None
                except (httpx.RequestError, json.JSONDecodeError, ValueError):
                    confirmation_props = None
                if (
                    isinstance(confirmation_props, dict)
                    and confirmation_props.get("is_sleeping") is True
                    and remove_quarantine_marker(
                        resolved_lock_path,
                        expected_content=marker_content,
                    )
                ):
                    return True
            await asyncio.sleep(resolved_poll_interval)
        return False

    async def wait_until_sleeping() -> bool:
        """仅在 backend 稳定报告休眠后提交请求，避开 sleep-idle 竞态窗口。"""
        client: httpx.AsyncClient = app.state.llama_client
        deadline = time.monotonic() + resolved_sleep_timeout
        while time.monotonic() < deadline:
            try:
                response = await client.get(PROPS_PATH, timeout=5.0)
                props = response.json() if response.is_success else None
            except (httpx.RequestError, json.JSONDecodeError, ValueError):
                props = None
            if isinstance(props, dict) and props.get("is_sleeping") is True:
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

        payload["model"] = model_alias
        lease, marker_content, client = await begin_request()
        lease_released = False
        stream_owns_lease = False

        async def release_once() -> None:
            """确保异常和取消路径也只释放一次文件锁。"""
            nonlocal lease_released
            if not lease_released:
                lease_released = True
                await release_lease(lease)

        try:
            if not await wait_until_sleeping():
                sleeping = await wait_for_sleep_and_clear(marker_content)
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 请求前未确认模型休眠，GPU 已隔离，请运行恢复流程。",
                    )
                raise HTTPException(status_code=503, detail="llama.cpp 服务尚未进入 sleep-idle。")
            try:
                upstream = await client.send(
                    client.build_request("POST", CHAT_COMPLETIONS_PATH, json=payload),
                    stream=True,
                )
            except httpx.RequestError as exc:
                sleeping = await wait_for_sleep_and_clear(marker_content)
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 无法确认模型已卸载，GPU 已隔离，请运行恢复流程。",
                    ) from exc
                raise HTTPException(status_code=503, detail="llama.cpp 服务暂不可用。") from exc

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
                    finally:
                        try:
                            await upstream.aclose()
                        except Exception:
                            LOGGER.exception("关闭 %s 上游流失败。", model_alias)
                        try:
                            sleeping = await wait_for_sleep_and_clear(marker_content)
                            if not sleeping:
                                LOGGER.error(
                                    "%s 模型未确认卸载，GPU quarantine 标记已保留。",
                                    model_alias,
                                )
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
                    sleeping = await wait_for_sleep_and_clear(marker_content)
                if not sleeping:
                    raise HTTPException(
                        status_code=503,
                        detail="llama.cpp 无法确认模型已卸载，GPU 已隔离，请运行恢复流程。",
                    ) from exc
                raise HTTPException(status_code=503, detail="llama.cpp 服务暂不可用。") from exc
            await upstream.aclose()
            sleeping = await wait_for_sleep_and_clear(marker_content)
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
