"""LongCat Avatar 请求的轻量文件、GPU 与视频预检工具。"""

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
    """展开环境变量与用户目录，并返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def logical_storage_name(full_path: str, allowed_suffixes: set[str]) -> str:
    """将安全的逻辑文件名转换为摘要文件名。"""
    candidate = Path(full_path)
    if (
        not full_path
        or candidate.name != full_path
        or full_path in {".", ".."}
        or "\x00" in full_path
    ):
        raise ValueError("full_path 必须是单个文件名，不能包含目录或路径跳转。")
    suffix = candidate.suffix.lower()
    if suffix not in allowed_suffixes:
        raise ValueError(f"文件扩展名必须属于：{', '.join(sorted(allowed_suffixes))}。")
    digest = hashlib.sha256(full_path.encode("utf-8")).hexdigest()
    return f"{digest}{suffix}"


def audio_storage_name(full_path: str) -> str:
    """验证音频逻辑标识并返回安全文件名。"""
    return logical_storage_name(
        full_path, {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"}
    )


def image_storage_name(full_path: str) -> str:
    """验证图片逻辑标识并返回安全文件名。"""
    return logical_storage_name(full_path, {".png", ".jpg", ".jpeg", ".webp"})


def image_info(path: str | Path) -> tuple[int, int]:
    """解码参考图并限制图像尺寸。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("LongCat Avatar 图片预检需要 Pillow。") from exc
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


def cuda_status(visible_devices: str) -> dict[str, Any]:
    """通过 nvidia-smi 检查所选 GPU，不在 API 进程初始化 CUDA。"""
    requested = [device.strip() for device in visible_devices.split(",") if device.strip()]
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
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "source": "nvidia-smi", "error": str(exc), "devices": []}

    rows: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            rows[fields[0]] = {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "used_mib": int(fields[3]),
                "total_mib": int(fields[4]),
                "free_mib": int(fields[5]),
            }
            rows[fields[1]] = rows[fields[0]]
        except ValueError:
            continue
    selected = [rows[device] for device in requested if device in rows]
    return {
        "available": len(selected) == len(requested) and bool(requested),
        "source": "nvidia-smi",
        "requested_devices": requested,
        "devices": selected,
    }


def query_gpu_memory(visible_devices: str) -> dict[str, Any]:
    """返回选中 GPU 的显存容量。"""
    status = cuda_status(visible_devices)
    if not status["available"]:
        raise RuntimeError("无法读取 LongCat Avatar 所选 GPU 的显存状态。")
    return {"devices": status["devices"]}


def audio_duration(path: Path, ffprobe_bin: str) -> float:
    """读取音频时长，优先使用 ffprobe，并以 WAV 解析作为本地退路。"""
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        duration = float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("无法通过 ffprobe 读取输入音频时长。") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("输入音频时长必须是有限正数。")
    return duration


def validate_video(path: Path, ffprobe_bin: str) -> None:
    """确认生成 MP4 同时包含视频流与音频流。"""
    if not path.is_file() or path.stat().st_size < 12:
        raise RuntimeError("LongCat Avatar worker 未生成有效 MP4。")
    with path.open("rb") as source:
        if source.read(12)[4:8] != b"ftyp":
            raise RuntimeError("LongCat Avatar worker 输出不是 MP4。")
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_streams",
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
        raise RuntimeError("无法读取 LongCat Avatar MP4 元数据。") from exc
    stream_types = {stream.get("codec_type") for stream in metadata.get("streams", [])}
    if "video" not in stream_types:
        raise RuntimeError("LongCat Avatar MP4 不包含视频流。")
    if "audio" not in stream_types:
        raise RuntimeError("LongCat Avatar MP4 不包含输入音频轨。")


def persist_video(source: Path, output_dir: Path) -> Path:
    """将经验证的 MP4 流式原子保存到结果目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"longcat_avatar_{uuid.uuid4().hex}.mp4"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".longcat_avatar_", suffix=".part"
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
    """终止 worker 及 torchrun 启动的所有 rank，并在超时后发送 SIGKILL。"""
    del label
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
    """拒绝非有限数或非正超时，避免 GPU 锁无限等待。"""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数。")
