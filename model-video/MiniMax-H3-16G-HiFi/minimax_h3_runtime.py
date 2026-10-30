"""MiniMax H3 的 CPU 预检、GPU 状态和 MP4 文件管理工具。"""

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
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def image_storage_name(full_path: str) -> str:
    """把逻辑上传路径映射为摘要文件名，拒绝通过路径访问任意本地文件。"""
    suffix = Path(full_path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("图片 full_path 必须以 PNG、JPG、JPEG 或 WebP 扩展名结尾。")
    digest = hashlib.sha256(full_path.encode("utf-8")).hexdigest()
    return f"{digest}{suffix}"


def image_info(path: str | Path) -> tuple[int, int]:
    """解码并限制参考图像尺寸，避免无效输入进入 GPU 队列。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("MiniMax H3 图片预检需要 Pillow。") from exc

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


def cuda_status(device: str = "0") -> dict[str, Any]:
    """使用 nvidia-smi 检查可见 GPU，不在 HTTP 进程初始化 CUDA。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        selected = next(
            (
                line
                for line in devices
                if device in {field.strip() for field in line.split(",", 2)[:2]}
            ),
            None,
        )
        return {
            "available": selected is not None,
            "source": "nvidia-smi",
            "devices": devices,
            "selected_device": selected,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "source": "nvidia-smi", "error": str(exc)}


def query_gpu_memory(device: str = "0") -> dict[str, int]:
    """返回选定 GPU 当前空闲和总显存 MiB，供 OOM admission gate 使用。"""
    status = cuda_status(device)
    selected = status.get("selected_device")
    if not status.get("available") or not selected:
        raise RuntimeError("无法通过 nvidia-smi 读取 MiniMax H3 所选 GPU 显存。")
    fields = [field.strip() for field in selected.split(",")]
    try:
        return {"free_mib": int(fields[5]), "total_mib": int(fields[4])}
    except (IndexError, ValueError) as exc:
        raise RuntimeError("nvidia-smi 返回了无法识别的显存数据。") from exc


def validate_video(path: Path, ffprobe_bin: str, *, expect_audio: bool = False) -> None:
    """确认 MP4 的视频流有效，且音轨存在性符合本次请求。"""
    if not path.is_file() or path.stat().st_size < 12:
        raise RuntimeError("MiniMax H3 worker 未生成有效 MP4。")
    with path.open("rb") as source:
        if source.read(12)[4:8] != b"ftyp":
            raise RuntimeError("MiniMax H3 worker 输出不是 MP4。")
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
        raise RuntimeError("无法读取 MiniMax H3 MP4 元数据。") from exc
    stream_types = {stream.get("codec_type") for stream in metadata.get("streams", [])}
    if "video" not in stream_types:
        raise RuntimeError("MiniMax H3 MP4 不包含视频流。")
    has_audio = "audio" in stream_types
    if expect_audio and not has_audio:
        raise RuntimeError("MiniMax H3 请求启用原生音频，但输出 MP4 不包含音轨。")
    if not expect_audio and has_audio:
        raise RuntimeError("MiniMax H3 请求关闭原生音频，但 MP4 意外包含音轨。")


def persist_video(source: Path, output_dir: Path) -> Path:
    """把 MP4 原子保存到结果目录，不将成片读入 HTTP 进程内存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"minimax_h3_{uuid.uuid4().hex}.mp4"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".minimax_h3_",
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
    """结束一次性 worker 的整个进程组，超时后升级为 SIGKILL。"""
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


def positive_float(name: str, value: float) -> None:
    """拒绝非有限数或非正超时，避免 GPU 锁或 worker 无限等待。"""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正的有限数。")
