"""原子保存 Confucius4-TTS 生成的 WAV。"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def persist_audio_bytes(audio_bytes: bytes, output_dir: str | os.PathLike[str]) -> Path:
    """将生成的 WAV 字节原子写入业务输出目录。"""
    if not audio_bytes:
        raise ValueError("cannot persist empty audio")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_directory,
        prefix=f".confucius4_tts_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_directory / f"{temporary_file.name[1:-4]}.wav"
    try:
        with os.fdopen(output_fd, "wb") as destination:
            destination.write(audio_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_file, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        temporary_file.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
