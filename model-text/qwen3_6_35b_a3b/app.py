"""通过共享 llama.cpp 适配器提供 Qwen3.6 聊天 API。"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI
from ornith_1_5_35b_a3b.app import create_app as create_shared_app

API_ROUTE = "/v1/qwen/Qwen3.6-35B-A3B-MTP"
MODEL_ALIAS = "Qwen3.6-35B-A3B-MTP"


def _optional_pid(name: str) -> int | None:
    """读取可选的正 backend 进程号。"""
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


def create_qwen_app(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    lock_path: str | os.PathLike[str] | None = None,
    sleep_timeout: float | None = None,
    sleep_poll_interval: float | None = None,
    cuda_release_delay: float | None = None,
) -> FastAPI:
    """建立 Qwen3.6 API，并复用全局 GPU 锁和 llama.cpp quarantine。"""
    return create_shared_app(
        api_route=API_ROUTE,
        model_alias=MODEL_ALIAS,
        backend_base_url=os.getenv("QWEN36_LLAMA_BASE_URL", "http://127.0.0.2:8400"),
        backend_host=os.getenv("QWEN36_BACKEND_HOST", "127.0.0.2"),
        backend_port=int(os.getenv("QWEN36_LLAMA_CPP_PORT", "8400")),
        backend_env_prefix="QWEN36",
        backend_pid=_optional_pid("QWEN36_BACKEND_PID"),
        backend_pgid=_optional_pid("QWEN36_BACKEND_PGID"),
        backend_executable=os.getenv("LLAMA_SERVER_BIN"),
        transport=transport,
        lock_path=lock_path,
        sleep_timeout=sleep_timeout,
        sleep_poll_interval=sleep_poll_interval,
        cuda_release_delay=cuda_release_delay,
    )


app = create_qwen_app()
