"""参考音频上传、内容寻址和原子存储。

共享模块只处理文件和元数据，不导入 FastAPI 或任何模型包，因此可以被每个独立
uv 项目复用。设计音色 WAV 始终留在 ``timbre`` 目录；克隆预览只保存相对引用。
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
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


def hash_filename(filename: str) -> str:
    """将 WebUI 逻辑路径转为稳定、安全的本地文件名。"""
    extension = Path(filename).suffix.lower() or ".wav"
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{digest}{extension}"


def sha256_file(path: str | Path) -> str:
    """流式计算文件摘要，避免大音频被完整读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """用同目录临时文件提交 JSON，避免读者看到半写入元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, sort_keys=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本 sidecar。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


class AudioReferenceStore:
    """维护克隆上传、音色引用和内容摘要索引。"""

    def __init__(self, prompts_dir: str | Path, timbre_dir: str | Path) -> None:
        self.prompts_dir = Path(prompts_dir).resolve()
        self.timbre_dir = Path(timbre_dir).resolve()
        self.references_dir = self.timbre_dir / ".references"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.timbre_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)

    def clone_path(self, filename: str) -> Path:
        """返回普通克隆上传的目标路径。"""
        return self.prompts_dir / hash_filename(filename)

    def reference_path(self, filename: str) -> Path:
        """返回设计音色的轻量引用 sidecar 路径。"""
        return self.references_dir / f"{hash_filename(filename)}.json"

    def prompt_sidecar_path(self, filename: str) -> Path:
        """返回普通克隆上传的参考文本 sidecar 路径。"""
        return self.prompts_dir / f"{hash_filename(filename)}.prompt.txt"

    def _index_path(self) -> Path:
        return self.references_dir / "sha256-index.json"

    def _index_lock_path(self) -> Path:
        return self.references_dir / "sha256-index.lock"

    def _load_index(self) -> dict[str, str]:
        try:
            payload = json.loads(self._index_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        return {
            digest: name
            for digest, name in entries.items()
            if isinstance(digest, str) and isinstance(name, str)
        }

    def _write_index(self, entries: dict[str, str]) -> None:
        _atomic_write_json(self._index_path(), {"version": 1, "entries": entries})

    def _resolve_indexed_timbre(self, digest: str, entries: dict[str, str]) -> Path | None:
        filename = entries.get(digest)
        if not filename:
            return None
        candidate = (self.timbre_dir / filename).resolve()
        if candidate.parent != self.timbre_dir or not candidate.is_file():
            return None
        return candidate

    def _find_timbre_by_digest(self, digest: str) -> Path | None:
        """先查摘要索引，仅在首次缺失时扫描并回填历史音色。"""
        with _path_lock(self._index_lock_path()):
            entries = self._load_index()
            indexed = self._resolve_indexed_timbre(digest, entries)
            if indexed is not None:
                return indexed

            changed = False
            matched: Path | None = None
            for candidate in self.timbre_dir.glob("*.wav"):
                candidate_digest = sha256_file(candidate)
                entries[candidate_digest] = candidate.name
                changed = True
                if candidate_digest == digest:
                    matched = candidate
            if changed:
                self._write_index(entries)
            return matched

    def register_timbre_file(self, path: str | Path) -> str:
        """将新生成的音色注册进索引，之后上传不必重扫整个目录。"""
        candidate = Path(path).resolve()
        if candidate.parent != self.timbre_dir or not candidate.is_file():
            raise ValueError("只能登记 TIMBRE_STORAGE_DIR 中的音色 WAV。")
        digest = sha256_file(candidate)
        with _path_lock(self._index_lock_path()):
            entries = self._load_index()
            entries[digest] = candidate.name
            self._write_index(entries)
        return digest

    def _read_reference(self, filename: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.reference_path(filename).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def prompt_audio_path(self, filename: str) -> Path:
        """解析普通上传或设计音色的相对引用，拒绝目录逃逸。"""
        clone_path = self.clone_path(filename)
        if clone_path.is_file():
            return clone_path
        reference = self._read_reference(filename)
        relative_path = reference.get("relative_path") if reference else None
        if not isinstance(relative_path, str):
            return clone_path
        candidate = (self.references_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.timbre_dir)
        except ValueError:
            return clone_path
        return candidate if candidate.is_file() else clone_path

    def load_prompt_text(self, filename: str) -> str | None:
        """读取普通上传或设计音色映射中的参考文本。"""
        clone_path = self.clone_path(filename)
        if clone_path.is_file():
            try:
                value = self.prompt_sidecar_path(filename).read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                return None
            return value or None
        reference = self._read_reference(filename)
        value = reference.get("prompt_text") if reference else None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _backup_paths(self, paths: list[Path]) -> list[tuple[Path, Path]]:
        backups: list[tuple[Path, Path]] = []
        for path in paths:
            if not path.exists():
                continue
            backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")
            os.replace(path, backup)
            backups.append((path, backup))
        return backups

    @staticmethod
    def _restore_backups(backups: list[tuple[Path, Path]]) -> None:
        for original, backup in reversed(backups):
            if backup.exists():
                original.unlink(missing_ok=True)
                os.replace(backup, original)

    def commit_staged_upload(
        self,
        staged: StagedUpload,
        full_path: str,
        prompt_text: str | None = None,
    ) -> dict[str, object]:
        """去重后原子提交上传，并在失败时恢复原有引用与 sidecar。"""
        normalized_name = full_path.strip()
        if not normalized_name or len(normalized_name) > 1024:
            staged.path.unlink(missing_ok=True)
            raise AudioUploadError("full_path 不能为空且不得超过 1024 个字符。")
        if Path(normalized_name).suffix.lower() not in DEFAULT_ALLOWED_AUDIO_SUFFIXES:
            staged.path.unlink(missing_ok=True)
            raise AudioUploadError("full_path 必须带受支持的音频扩展名。")

        normalized_prompt = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        clone_path = self.clone_path(normalized_name)
        sidecar_path = self.prompt_sidecar_path(normalized_name)
        reference_path = self.reference_path(normalized_name)
        lock_path = self.references_dir / f"{hash_filename(normalized_name)}.lock"
        existing_paths = [clone_path, sidecar_path, reference_path]
        created_paths: list[Path] = []

        with _path_lock(lock_path):
            backups = self._backup_paths(existing_paths)
            try:
                timbre_path = self._find_timbre_by_digest(staged.sha256)
                if timbre_path is not None:
                    relative_path = os.path.relpath(timbre_path, self.references_dir)
                    _atomic_write_json(
                        reference_path,
                        {
                            "version": 1,
                            "kind": "timbre_reference",
                            "relative_path": relative_path,
                            "sha256": staged.sha256,
                            "prompt_text": normalized_prompt,
                        },
                    )
                    created_paths.append(reference_path)
                    staged.path.unlink(missing_ok=True)
                else:
                    clone_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged.path, clone_path)
                    created_paths.append(clone_path)
                    if normalized_prompt is not None:
                        _atomic_write_text(sidecar_path, normalized_prompt)
                        created_paths.append(sidecar_path)
                for _, backup in backups:
                    backup.unlink(missing_ok=True)
            except Exception:
                for path in created_paths:
                    path.unlink(missing_ok=True)
                self._restore_backups(backups)
                staged.path.unlink(missing_ok=True)
                raise

        return {
            "code": 200,
            "msg": "上传成功",
            "filename": normalized_name,
            "has_prompt_text": bool(normalized_prompt),
            "sha256": staged.sha256,
            "size_bytes": staged.size_bytes,
            "storage": "timbre_reference" if timbre_path is not None else "clone",
        }


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


def _referenced_timbre_paths(timbre_dir: Path) -> set[Path]:
    """收集仍被克隆预览引用的设计音色，保留策略绝不删除它们。"""
    references_dir = timbre_dir / ".references"
    referenced: set[Path] = set()
    for reference_path in references_dir.glob("*.json"):
        try:
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            relative_path = payload.get("relative_path")
            if not isinstance(relative_path, str):
                continue
            candidate = (references_dir / relative_path).resolve()
            candidate.relative_to(timbre_dir.resolve())
            referenced.add(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return referenced


def prune_generated_outputs(
    directories: dict[str | Path, tuple[str, ...]],
    *,
    retention_hours: float,
    retention_max_bytes: int,
    apply: bool = False,
    timbre_dir: str | Path | None = None,
) -> list[Path]:
    """按显式前缀清理旧生成结果，默认仅返回计划删除项。

    该函数从不匹配普通上传文件、隐藏文件或引用中的 timbre WAV。只有运维显式
    启用时才删除，因而默认策略不会破坏用户参考音频或当前克隆预览。
    """
    if retention_hours <= 0 and retention_max_bytes <= 0:
        return []

    protected = _referenced_timbre_paths(Path(timbre_dir)) if timbre_dir else set()
    now = time.time()
    candidates: list[Path] = []
    for directory_value, prefixes in directories.items():
        directory = Path(directory_value)
        if not directory.is_dir():
            continue
        for path in directory.glob("*.wav"):
            if path.resolve() in protected or not path.name.startswith(prefixes):
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
