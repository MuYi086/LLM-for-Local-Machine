"""Wan2.2-TI2V-5B-GGUF 的轻量预检与成品文件工具。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


def expand_path(path: str) -> str:
    """展开用户目录和环境变量并返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def positive_float(name: str, value: float) -> None:
    """校验运行时限是正的有限数。"""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正的有限数。")


def cuda_status() -> dict[str, Any]:
    """用 nvidia-smi 检查 GPU，不在 HTTP 进程导入 Torch。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "available": bool(result.stdout.strip()),
            "source": "nvidia-smi",
            "devices": result.stdout.splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "source": "nvidia-smi", "error": str(exc)}


def image_info(path: str | Path) -> tuple[int, int]:
    """解码图片并限制尺寸，避免坏数据进入 GPU 队列。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Wan2.2 图片预检依赖 Pillow 未安装。") from exc

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except OSError as exc:
        raise ValueError("无法解码参考图片。") from exc
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError("参考图片宽高必须在 1 到 8192 像素之间。")
    return width, height


def image_storage_name(full_path: str) -> str:
    """将上传逻辑标识转换成安全且稳定的图片文件名。"""
    suffix = Path(full_path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("图片 full_path 必须以 PNG、JPG、JPEG 或 WebP 扩展名结尾。")
    digest = hashlib.sha256(full_path.encode("utf-8")).hexdigest()
    return f"{digest}{suffix}"


def validate_video(path: Path, ffprobe_bin: str) -> None:
    """确认 worker 生成了含视频流的有效 MP4。"""
    if not path.is_file() or path.stat().st_size < 12:
        raise RuntimeError("Wan2.2 worker 未生成有效 MP4。")
    with path.open("rb") as source:
        if source.read(12)[4:8] != b"ftyp":
            raise RuntimeError("Wan2.2 worker 输出不是 MP4。")
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        metadata = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("无法读取 Wan2.2 MP4 元数据。") from exc
    if not any(stream.get("codec_type") == "video" for stream in metadata.get("streams", [])):
        raise RuntimeError("Wan2.2 MP4 不包含视频流。")


def persist_video(source: Path, output_dir: Path) -> Path:
    """流式复制并原子提交 MP4，避免 HTTP 进程读入整段视频。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"wan22_ti2v_{uuid.uuid4().hex}.mp4"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".wan22_ti2v_",
        suffix=".part",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as video:
            shutil.copyfileobj(video, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def terminate_process_group(process: Any, label: str) -> None:
    """停止 worker 进程组，包含其启动的 ComfyUI 子进程。"""
    if process is None:
        return
    pid = getattr(process, "pid", None)
    if pid is None:
        if process.poll() is None:
            process.terminate()
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if pid is None:
            process.kill()
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
