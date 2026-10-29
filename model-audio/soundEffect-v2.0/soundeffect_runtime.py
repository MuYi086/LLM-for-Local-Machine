"""MOSS-SoundEffect 服务的轻量进程与音频文件辅助函数。"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any


def _process_group_exists(process_id: int) -> bool:
    """检查 worker leader 对应的进程组是否仍有成员。"""
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    process: Any,
    label: str,
    terminate_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """清理一次性 worker 进程组，避免超时后 CUDA 子进程继续占用设备。"""
    if process is None:
        return

    process_id = getattr(process, "pid", None)
    if process_id is None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=terminate_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=kill_timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        return

    try:
        # 不能只看 leader 是否已退出：它退出后，同组后代仍可能占用 GPU 或持有管道。
        os.killpg(process_id, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            try:
                process.wait(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
        return
    except OSError as exc:
        print(f"[{label}] 终止 worker 进程组失败，改为终止主进程: {exc}")
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + terminate_timeout
    while _process_group_exists(process_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[{label}] worker 进程组未及时退出，强制终止")
            break
        if process.poll() is None:
            try:
                process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(remaining, 0.05))

    if _process_group_exists(process_id):
        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            print(f"[{label}] 强制终止 worker 进程组失败: {exc}")
            if process.poll() is None:
                process.kill()

    kill_deadline = time.monotonic() + kill_timeout
    while _process_group_exists(process_id):
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            print(f"[{label}] worker 进程组在 SIGKILL 后仍未退出")
            break
        if process.poll() is None:
            try:
                process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(remaining, 0.05))
    try:
        process.wait(timeout=max(kill_deadline - time.monotonic(), 0))
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def persist_audio_bytes(audio_bytes: bytes, output_dir: str | os.PathLike[str]) -> Path:
    """将生成 WAV 以临时文件加原子重命名的方式保存。"""
    if not audio_bytes:
        raise ValueError("不能保存空音频。")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_fd, temporary_name = tempfile.mkstemp(
        dir=output_directory,
        prefix=".moss_soundeffect_v2_",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    output_path = output_directory / f"soundeffect_{time.time_ns()}.wav"
    try:
        with os.fdopen(output_fd, "wb") as destination:
            destination.write(audio_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise


def validate_generated_wav(audio_bytes: bytes) -> float:
    """验证 worker 输出为非空、48 kHz 单声道且不超过 30 秒的 WAV。"""
    try:
        import io

        with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            sample_width = audio.getsampwidth()
            frame_data = audio.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("SoundEffect worker 输出的文件不是有效 WAV。") from exc

    if channels != 1 or sample_rate != 48_000 or frames <= 0:
        raise ValueError("SoundEffect worker 必须输出非空 48 kHz 单声道 WAV。")
    if len(frame_data) != frames * channels * sample_width:
        raise ValueError("SoundEffect worker 输出的 WAV 帧数据不完整。")
    duration = frames / sample_rate
    if duration > 30.0:
        raise ValueError("SoundEffect worker 输出超过 30 秒。")
    return duration
