"""在共享 GPU 锁下执行 LocalAI 请求并确认 backend 收尾。"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_runtime_lock

from .config import GatewayConfig
from .errors import BackendQuarantinedError
from .localai_client import LocalAIClient


@contextmanager
def _request_marker(lock_path: str, model: str):
    """在推理期间留下崩溃恢复标记，backend 确认停止后才移除。"""
    marker = Path(f"{lock_path}.quarantine")
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".gpu-request-", dir=marker.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": 1,
                    "request_id": uuid.uuid4().hex,
                    "model": model,
                    "started_at": time.time(),
                },
                file,
            )
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, marker)
        yield marker
    finally:
        temporary.unlink(missing_ok=True)


def execute_localai_operation[Result](
    *,
    config: GatewayConfig,
    operation: Callable[[LocalAIClient], Result],
    client: LocalAIClient | None = None,
) -> Result:
    """持有唯一 GPU 租约，执行一次 LocalAI 操作并确认 backend 收尾。"""
    localai = client or LocalAIClient(config)
    try:
        with gpu_runtime_lock(
            config.gpu_lock_file,
            f"localai/{config.model}",
            timeout_seconds=config.gpu_lock_wait_timeout,
        ):
            with _request_marker(config.gpu_lock_file, config.model) as marker:
                request_marker = marker.read_bytes()
                result: Result | None = None
                inference_error: BaseException | None = None
                try:
                    result = operation(localai)
                except BaseException as exc:
                    inference_error = exc

                try:
                    localai.shutdown_model()
                except Exception as cleanup_error:
                    raise BackendQuarantinedError(
                        f"LocalAI backend {config.model} 未确认停止，GPU 已隔离；标记文件保留在 {marker}。"
                    ) from cleanup_error
                # LocalAI 收尾期间若隔离标记变化，不能仅凭 HTTP 成功判断 backend 已退出。
                if marker.read_bytes() != request_marker:
                    raise BackendQuarantinedError(
                        f"LocalAI 收尾期间更新了 GPU 隔离标记，需人工恢复：{marker}。"
                    )
                if config.cuda_release_delay > 0:
                    time.sleep(config.cuda_release_delay)
                marker.unlink(missing_ok=True)
                if inference_error is not None:
                    raise inference_error
                if result is None:
                    raise RuntimeError("LocalAI 操作未返回结果。")
                return result
    except GpuQuarantinedError as exc:
        raise BackendQuarantinedError(str(exc)) from exc
    except GpuLockTimeoutError:
        raise
