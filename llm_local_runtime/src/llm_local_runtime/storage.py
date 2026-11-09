"""音频上传暂存、磁盘容量与生成文件保留工具。

共享模块只处理文件，不导入 FastAPI 或任何模型包，因此可以被独立 uv 项目复用。
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_ALLOWED_AUDIO_TYPES = frozenset(
    {
        "application/octet-stream",
        "audio/aac",
        "audio/flac",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/vnd.wave",
        "audio/wav",
        "audio/wave",
        "audio/webm",
        "audio/x-wav",
    }
)
DEFAULT_ALLOWED_AUDIO_SUFFIXES = frozenset(
    {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}
)


class _AsyncReadableUpload(Protocol):
    """避免为共享工具引入 FastAPI 依赖的最小上传对象协议。"""

    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...


class AudioUploadError(ValueError):
    """上传媒体类型、名称或大小不符合服务约束。"""

    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.status_code = status_code


@dataclass(frozen=True)
class UploadPolicy:
    """上传边界，默认限制 64 MiB，适合短参考音频。"""

    max_bytes: int = 64 * 1024 * 1024
    allowed_content_types: frozenset[str] = DEFAULT_ALLOWED_AUDIO_TYPES
    allowed_suffixes: frozenset[str] = DEFAULT_ALLOWED_AUDIO_SUFFIXES
    chunk_size: int = 1024 * 1024

    @classmethod
    def from_environment(cls) -> UploadPolicy:
        """从环境变量读取统一上限，阻止单个请求耗尽内存或磁盘。"""
        max_bytes = int(os.getenv("UPLOAD_MAX_BYTES", str(cls.max_bytes)))
        if max_bytes <= 0:
            raise ValueError("UPLOAD_MAX_BYTES 必须为正整数。")
        return cls(max_bytes=max_bytes)


@dataclass(frozen=True)
class StagedUpload:
    """写入暂存文件后的上传信息。"""

    path: Path
    sha256: str
    size_bytes: int
    suffix: str


async def stage_audio_upload(
    upload: _AsyncReadableUpload,
    staging_dir: str | Path,
    *,
    policy: UploadPolicy | None = None,
) -> StagedUpload:
    """流式读取上传内容、同步计算 SHA-256，并安全地写入暂存文件。

    暂存成功前不会碰业务目录。任何校验或 I/O 失败都会删除暂存文件，调用方
    因此可以安全地在随后线程池中执行去重与原子提交。
    """
    active_policy = policy or UploadPolicy.from_environment()
    filename = upload.filename or "upload.wav"
    suffix = Path(filename).suffix.lower() or ".wav"
    content_type = (upload.content_type or "").lower().split(";", 1)[0].strip()
    if suffix not in active_policy.allowed_suffixes:
        raise AudioUploadError("仅支持 WAV、MP3、OGG、FLAC、M4A、AAC 或 WebM 音频。")
    if content_type and content_type not in active_policy.allowed_content_types:
        raise AudioUploadError(f"不支持的音频 Content-Type: {content_type}")

    target_dir = Path(staging_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_dir,
        prefix="upload_",
        suffix=f"{suffix}.part",
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await upload.read(active_policy.chunk_size):
                size_bytes += len(chunk)
                if size_bytes > active_policy.max_bytes:
                    raise AudioUploadError(
                        f"上传音频超过 {active_policy.max_bytes // (1024 * 1024)} MiB 限制。",
                        status_code=413,
                    )
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if size_bytes == 0:
            raise AudioUploadError("上传音频不能为空。")
        return StagedUpload(
            path=temporary_path,
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            suffix=suffix,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def _path_lock(lock_path: Path) -> Iterator[None]:
    """为同一逻辑文件提供跨进程提交锁。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def commit_plain_upload(staged: StagedUpload, target_path: str | Path) -> None:
    """原子替换不需要音色去重的上传文件。"""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    with _path_lock(lock_path):
        backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
        had_original = target.exists()
        try:
            if had_original:
                os.replace(target, backup)
            os.replace(staged.path, target)
            backup.unlink(missing_ok=True)
        except Exception:
            target.unlink(missing_ok=True)
            if backup.exists():
                os.replace(backup, target)
            staged.path.unlink(missing_ok=True)
            raise


def storage_disk_status(storage_dir: str | Path) -> dict[str, int | bool]:
    """返回存储所在文件系统的容量与默认关闭的保留策略。"""
    usage = shutil.disk_usage(storage_dir)
    retention_hours = float(os.getenv("STORAGE_RETENTION_HOURS", "0"))
    retention_max_bytes = int(os.getenv("STORAGE_RETENTION_MAX_BYTES", "0"))
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "retention_hours": retention_hours,
        "retention_max_bytes": retention_max_bytes,
        "retention_enabled": retention_hours > 0 or retention_max_bytes > 0,
    }


def prune_generated_outputs(
    directories: dict[str | Path, tuple[str, ...]],
    *,
    retention_hours: float,
    retention_max_bytes: int,
    apply: bool = False,
) -> list[Path]:
    """按显式前缀清理旧生成结果，默认仅返回计划删除项。

    该函数从不匹配普通上传文件、隐藏文件或引用中的 timbre WAV。只有运维显式
    启用时才删除，因而默认策略不会破坏用户参考音频或当前克隆预览。
    """
    if retention_hours <= 0 and retention_max_bytes <= 0:
        return []

    now = time.time()
    candidates: list[Path] = []
    for directory_value, prefixes in directories.items():
        directory = Path(directory_value)
        if not directory.is_dir():
            continue
        for path in directory.glob("*.wav"):
            if not path.name.startswith(prefixes):
                continue
            candidates.append(path)

    candidates.sort(key=lambda path: path.stat().st_mtime)
    planned: list[Path] = []
    remaining_bytes = sum(path.stat().st_size for path in candidates)
    cutoff = now - retention_hours * 3600 if retention_hours > 0 else None
    for path in candidates:
        expired = cutoff is not None and path.stat().st_mtime < cutoff
        over_limit = retention_max_bytes > 0 and remaining_bytes > retention_max_bytes
        if not (expired or over_limit):
            continue
        planned.append(path)
        remaining_bytes -= path.stat().st_size

    if apply:
        for path in planned:
            path.unlink(missing_ok=True)
    return planned


async def close_upload_safely(upload: Any) -> None:
    """在框架提供 close 时尽力释放临时上传资源。"""
    close = getattr(upload, "close", None)
    if close is None:
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result
