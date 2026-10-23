"""SoulX-FlashHead 的无模型媒体预检与进程组清理工具。"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any


def cuda_status() -> dict[str, Any]:
    """通过 nvidia-smi 查询设备，避免 HTTP 进程初始化 CUDA。"""
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


def probe_media(path: str | Path, ffprobe_bin: str) -> dict[str, Any]:
    """在有界 CPU 子进程中读取媒体元数据，不加载模型。"""
    try:
        result = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取媒体文件: {Path(path).name}") from exc


def validate_inputs(image: Path, audio: Path, ffprobe_bin: str, max_duration: float) -> None:
    """在获取 GPU 锁前检查图片、音频流及驱动音频时长。"""
    image_info = probe_media(image, ffprobe_bin)
    images = image_info.get("streams", [])
    if not any(
        stream.get("codec_name") in {"png", "mjpeg", "webp"}
        and 0 < stream.get("width", 0) <= 8192
        and 0 < stream.get("height", 0) <= 8192
        for stream in images
    ):
        raise ValueError("参考图片必须是可解码的 PNG、JPEG 或 WebP，边长不超过 8192。")
    audio_info = probe_media(audio, ffprobe_bin)
    if not any(stream.get("codec_type") == "audio" for stream in audio_info.get("streams", [])):
        raise ValueError("驱动文件不包含可解码的音频流。")
    try:
        duration = float(audio_info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("无法确认驱动音频时长。") from exc
    if not math.isfinite(duration) or not 0 < duration <= max_duration:
        raise ValueError(f"驱动音频时长必须大于 0 且不超过 {max_duration:g} 秒。")


def validate_video(path: Path, ffprobe_bin: str) -> None:
    """确认 worker 生成包含音视频流的 MP4，防止编码失败后发布空文件。"""
    if not path.is_file() or path.stat().st_size < 12:
        raise RuntimeError("SoulX-FlashHead worker 未生成有效 MP4。")
    with path.open("rb") as source:
        if source.read(12)[4:8] != b"ftyp":
            raise RuntimeError("SoulX-FlashHead worker 输出不是 MP4。")
    info = probe_media(path, ffprobe_bin)
    kinds = {stream.get("codec_type") for stream in info.get("streams", [])}
    if not {"video", "audio"}.issubset(kinds):
        raise RuntimeError("SoulX-FlashHead MP4 缺少音频或视频流。")


def persist_video(source: Path, output_dir: Path) -> Path:
    """流式复制并原子提交视频，避免把整个成品读进 HTTP 内存。"""
    import tempfile
    import uuid

    target = output_dir / f"soulx_flashhead_{uuid.uuid4().hex}.mp4"
    descriptor, name = tempfile.mkstemp(dir=output_dir, prefix=".flashhead_", suffix=".part")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as audio_video:
            shutil.copyfileobj(audio_video, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def terminate_process_group(process: Any, label: str) -> None:
    """清理整个 worker 进程组，主进程成功退出后也回收其编码子进程。"""
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
            # 清理失败不得覆盖推理/超时的原始错误；SIGKILL 后由操作系统继续回收。
            print(f"[{label}] SIGKILL 后 worker 尚未完成回收")
    if pid is not None:
        # 父进程 wait 完成不代表后代进程已退出；SIGKILL 确保锁释放时无 GPU 子进程残留。
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
