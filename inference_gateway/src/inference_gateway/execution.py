"""在共享 GPU 锁下执行 LocalAI 请求并确认 backend 收尾。"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_runtime_lock

from .config import GatewayConfig
from .errors import BackendQuarantinedError, LocalAIRequestError
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


def execute_localai_worker(
    *,
    config: GatewayConfig,
    operation: str,
    payload: dict[str, object],
    client: LocalAIClient | None = None,
) -> tuple[dict[str, object], float]:
    """持有共享 GPU 租约并调用 LocalAI 外部 one-shot worker bridge。"""
    return execute_localai_operation(
        config=config,
        operation=lambda localai: localai.run_worker(operation=operation, payload=payload),
        client=client,
    )


def execute_localai_worker_file(
    *,
    config: GatewayConfig,
    operation: str,
    payload: dict[str, object],
    output_dir: Path,
    suffix: str,
    media_type: str,
    client: LocalAIClient | None = None,
) -> tuple[Path, float]:
    """调用 worker bridge、校验 base64 artifact 并原子暂存到本机。"""
    if suffix not in {".zip", ".json", ".npz", ".npy", ".wav", ".mp4"}:
        raise ValueError("LocalAI worker artifact 扩展名不受支持。")
    response, elapsed = execute_localai_worker(
        config=config,
        operation=operation,
        payload={**payload, "media_type": media_type},
        client=client,
    )
    if (
        response.get("kind") != "file"
        or response.get("media_type") != media_type
        or not isinstance(response.get("data_base64"), str)
        or not isinstance(response.get("size_bytes"), int)
    ):
        raise LocalAIRequestError("LocalAI worker bridge artifact 响应格式或 MIME 类型无效。")
    try:
        content = base64.b64decode(response["data_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise LocalAIRequestError("LocalAI worker bridge artifact 不是有效 base64。") from exc
    if not content or len(content) != response["size_bytes"] or len(content) > 128 * 1024 * 1024:
        raise LocalAIRequestError("LocalAI worker bridge artifact 大小校验失败。")

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".localai-worker-", suffix=suffix, dir=output_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        return temporary_path, elapsed
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def execute_localai_worker_file_reference(
    *,
    config: GatewayConfig,
    operation: str,
    payload: dict[str, object],
    output_dir: Path,
    artifact_dir: Path,
    suffix: str,
    media_type: str,
    client: LocalAIClient | None = None,
) -> tuple[Path, float]:
    """从受控共享目录取回大文件引用，并在读取后立即删除原 artifact。"""
    allowed_suffixes = {".zip", ".json", ".npz", ".npy", ".wav", ".mp4"}
    if suffix not in allowed_suffixes:
        raise ValueError("LocalAI worker artifact 扩展名不受支持。")
    response, elapsed = execute_localai_worker(
        config=config,
        operation=operation,
        payload={**payload, "media_type": media_type},
        client=client,
    )
    artifact_id = response.get("artifact_id")
    size_bytes = response.get("size_bytes")
    if (
        response.get("kind") != "file_ref"
        or response.get("media_type") != media_type
        or response.get("suffix") != suffix
        or not isinstance(artifact_id, str)
        or re.fullmatch(r"[a-f0-9]{32}", artifact_id) is None
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 0 < size_bytes <= 4 * 1024 * 1024 * 1024
    ):
        raise LocalAIRequestError("LocalAI worker bridge 大文件引用格式或大小无效。")

    if artifact_dir.exists() and artifact_dir.is_symlink():
        raise LocalAIRequestError("LocalAI worker artifact 目录不能是符号链接。")
    root = artifact_dir.resolve()
    source = root / f"{artifact_id}{suffix}"
    if source.resolve().parent != root or source.is_symlink() or not source.is_file():
        raise LocalAIRequestError("LocalAI worker artifact 引用不存在或越界。")
    if source.stat().st_size != size_bytes:
        source.unlink(missing_ok=True)
        raise LocalAIRequestError("LocalAI worker artifact 引用大小校验失败。")

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".localai-worker-ref-", suffix=suffix, dir=output_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        if temporary_path.stat().st_size != size_bytes:
            raise LocalAIRequestError("LocalAI worker artifact 复制后的大小校验失败。")
        return temporary_path, elapsed
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)


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
                # 固定 LocalAI 对外部 backend 的 Free 错误只记日志，HTTP 200 不能覆盖
                # bridge 已写下的回收失败标记；保留隔离，交给运维恢复流程确认进程退出。
                if marker.read_bytes() != request_marker:
                    raise BackendQuarantinedError(
                        f"LocalAI worker 收尾期间更新了 GPU 隔离标记，需人工恢复：{marker}。"
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
