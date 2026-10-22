"""原子保存成功生成的 TTS WAV，便于本地检查。"""

from __future__ import annotations

# 克隆和音色设计共用原子落盘工具，避免并发请求看到不完整 WAV。
import os
import shutil
import tempfile
import time
from pathlib import Path

PathLike = str | os.PathLike[str]


def persist_audio_bytes(audio_bytes: bytes, model_prefix: str, output_dir: PathLike) -> Path:
    """原子保存生成的 TTS WAV，便于本地检查且不暴露半成品。"""
    if not audio_bytes:
        raise ValueError("cannot persist empty audio")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_directory,
        prefix=f".{model_prefix}_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_directory / f"{temporary_file.name[1:-4]}.wav"
    try:
        with os.fdopen(output_fd, "wb") as destination:
            destination.write(audio_bytes)
        os.replace(temporary_file, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        try:
            temporary_file.unlink()
        except OSError:
            pass
        try:
            output_path.unlink()
        except OSError:
            pass
        raise


def persist_audio_file(source_path: PathLike, model_prefix: str, output_dir: PathLike) -> Path:
    """把 worker 已生成的 WAV 复制到业务输出目录并原子替换。"""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_directory,
        prefix=f".{model_prefix}_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_directory / f"{temporary_file.name[1:-4]}.wav"
    try:
        with open(source_path, "rb") as source, os.fdopen(output_fd, "wb") as destination:
            shutil.copyfileobj(source, destination)
        os.replace(temporary_file, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        try:
            temporary_file.unlink()
        except OSError:
            pass
        try:
            output_path.unlink()
        except OSError:
            pass
        raise
