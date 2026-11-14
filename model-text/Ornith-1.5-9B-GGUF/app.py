"""提供 Ornith-1.5-9B-GGUF 的共享锁保护聊天 API。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from llama_cpp_runtime.app import create_app as create_shared_app

API_ROUTE = "/v1/ornith-ai/Ornith-1.5-9B-GGUF"
MODEL_ALIAS = "Ornith-1.5-9B-GGUF"


def create_app(
    *,
    backend_base_url: str | None = None,
    backend_host: str | None = None,
    backend_port: int | None = None,
    backend_pid: int | None = None,
    backend_pgid: int | None = None,
    backend_executable: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    lock_path: str | Path | None = None,
    sleep_timeout: float | None = None,
    sleep_poll_interval: float | None = None,
    cuda_release_delay: float | None = None,
) -> FastAPI:
    """创建固定路由和模型别名的 9B API。

    Args:
        backend_base_url: llama-server 的 loopback 根地址。
        backend_host: llama-server 绑定地址。
        backend_port: llama-server 端口。
        backend_pid: llama-server 进程号。
        backend_pgid: llama-server 进程组号。
        backend_executable: llama-server 可执行文件路径。
        transport: 测试时替换上游 HTTPX transport。
        lock_path: 所有 GPU provider 共用的文件锁路径。
        sleep_timeout: 等待模型卸载的最长时间。
        sleep_poll_interval: 查询模型休眠状态的间隔。
        cuda_release_delay: 确认休眠后释放 GPU 锁前的缓冲时间。

    Returns:
        配置好的 FastAPI 应用。
    """
    resolved_backend_port = (
        backend_port
        if backend_port is not None
        else int(os.getenv("ORNITH_9B_LLAMA_CPP_PORT", "8505"))
    )
    resolved_backend_base_url = backend_base_url or os.getenv(
        "ORNITH_9B_LLAMA_BASE_URL",
        f"http://127.0.0.1:{resolved_backend_port}",
    )
    return create_shared_app(
        api_route=API_ROUTE,
        model_alias=MODEL_ALIAS,
        backend_base_url=resolved_backend_base_url,
        backend_host=backend_host,
        backend_port=resolved_backend_port,
        backend_env_prefix="ORNITH_9B",
        backend_pid=backend_pid,
        backend_pgid=backend_pgid,
        backend_executable=backend_executable,
        transport=transport,
        lock_path=lock_path,
        sleep_timeout=sleep_timeout,
        sleep_poll_interval=sleep_poll_interval,
        cuda_release_delay=cuda_release_delay,
    )


app = create_app()
