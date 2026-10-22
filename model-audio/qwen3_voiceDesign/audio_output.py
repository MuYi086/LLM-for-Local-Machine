"""原子保存 VoiceDesign 服务生成的音频。"""

from __future__ import annotations

# 音色设计结果先写临时文件，再替换到 timbre 目录，防止读取到半成品。
import os
import tempfile
import time
from pathlib import Path

PathLike = str | os.PathLike[str]


def persist_audio_bytes(audio_bytes: bytes, model_prefix: str, output_dir: PathLike) -> Path:
    """原子写入成功生成的音色 WAV。"""
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
        temporary_file.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
