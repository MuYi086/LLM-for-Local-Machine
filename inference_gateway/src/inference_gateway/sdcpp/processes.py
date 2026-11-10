"""sd.cpp Linux 进程身份、原子隔离标记和运维恢复。"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from pathlib import Path

from inference_gateway.errors import BackendQuarantinedError

from .config import positive_number


def sync_directory(path: Path) -> None:
    """同步目录项，避免标记或产物发布仅停留在页缓存。"""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_marker(path: Path, record: dict) -> bytes:
    """通过临时文件和 rename 持久化本次请求的隔离身份。"""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode()
    descriptor, name = tempfile.mkstemp(prefix=".sdcpp-marker-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def clear_marker(path: Path, expected: bytes) -> None:
    """仅在当前标记仍属于本次请求时解除隔离。"""
    if path.read_bytes() != expected:
        raise BackendQuarantinedError("sd.cpp 隔离标记发生变化，需通过恢复脚本复核。")
    path.unlink()
    sync_directory(path.parent)


def process_identity(pid: int) -> tuple[str, int, str] | None:
    """读取 Linux starttime、进程组和状态；权限错误不能视作已退出。"""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[19], int(fields[2]), fields[0]


def group_active(pgid: int) -> bool:
    """确认进程组是否仍有可运行成员，已死亡的 zombie 不持有 CUDA 资源。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # killpg(0) 也会看到被容器 PID 1 延迟回收的 zombie，因此需复核状态。
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = process_identity(int(entry.name))
        except (OSError, ValueError, IndexError):
            return True
        if identity is not None and identity[1] == pgid and identity[2] not in {"Z", "X"}:
            return True
    return False


def stop_process_group(pgid: int, timeout: float) -> None:
    """先 TERM 后 KILL，并在时限内确认整个独立进程组停止。"""
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError("拒绝停止调用方进程组。")
    timeout = positive_number(timeout, "cleanup_timeout")
    deadline = time.monotonic() + timeout
    for sig, until in (
        (signal.SIGTERM, time.monotonic() + min(2.0, timeout / 2)),
        (signal.SIGKILL, deadline),
    ):
        if not group_active(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        while time.monotonic() < until:
            if not group_active(pgid):
                return
            time.sleep(min(0.05, max(0, until - time.monotonic())))
    if group_active(pgid):
        raise BackendQuarantinedError("sd.cpp 进程组停止无法确认，保留 GPU 隔离。")


def recover_sdcpp(marker_path: Path, marker: dict, timeout: float) -> None:
    """在调用方已持有共享锁时验证身份并恢复 sd.cpp 隔离。

    Args:
        marker_path: 与全局 GPU 锁配套的标记。
        marker: 持锁后读取的标记内容。
        timeout: 停止进程组的正数时限。

    Raises:
        BackendQuarantinedError: 身份或停止状态无法确认。
    """
    positive_number(timeout, "cleanup_timeout")
    release_delay = positive_number(
        os.getenv("CUDA_RELEASE_DELAY", "2"), "CUDA_RELEASE_DELAY", allow_zero=True
    )
    expected = marker_path.read_bytes()
    if (
        json.loads(expected) != marker
        or marker.get("provider") != "stable-diffusion.cpp"
        or type(marker.get("schema_version")) is not int
        or marker["schema_version"] != 1
    ):
        raise BackendQuarantinedError("sd.cpp 隔离标记无效或已经变化。")
    if (
        marker.get("phase") == "prepared"
        and marker.get("pid") is None
        and marker.get("pgid") is None
    ):
        # 未记录身份前不发送启动许可；旧父进程失去锁后不能再让这个 worker 加载权重。
        clear_marker(marker_path, expected)
        return
    pid, pgid, start = marker.get("pid"), marker.get("pgid"), marker.get("pid_start_time")
    executable = marker.get("executable")
    if (
        marker.get("phase") != "registered"
        or type(pid) is not int
        or pid <= 1
        or pgid != pid
        or pgid == os.getpgrp()
        or not isinstance(start, str)
        or not start.isdigit()
        or not isinstance(executable, str)
        or not Path(executable).is_absolute()
    ):
        raise BackendQuarantinedError("sd.cpp 隔离标记缺少可核对的进程身份。")
    identity = process_identity(pid)
    if identity is not None and identity[0] == start and identity[1] == pgid:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        command = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        worker = str(Path(__file__).with_name("worker.py"))
        is_waiting_worker = len(command) > 3 and command[1:4] == ["-I", worker, executable]
        is_backend = any(
            arg.startswith("/") and Path(arg).resolve() == Path(executable).resolve()
            for arg in command[:2]
        )
        if identity[2] not in {"Z", "X"} and not (is_waiting_worker or is_backend):
            raise BackendQuarantinedError("PID 与 sd.cpp 命令不匹配，拒绝终止未知进程。")
        stop_process_group(pgid, timeout)
    elif group_active(pgid):
        # 根 PID 被复用或已退出但后代仍在时，缺少证据证明整个组属于旧请求。
        raise BackendQuarantinedError("无法核对 sd.cpp 遗留进程组身份，保留隔离。")
    if release_delay:
        time.sleep(release_delay)
    clear_marker(marker_path, expected)
