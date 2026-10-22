"""TIGER-DnR 服务的无模型运行时辅助函数。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from typing import Any


def expand_path(path: str) -> str:
    """展开用户目录和环境变量，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def cuda_status() -> dict[str, Any]:
    """不导入 Torch 或创建 CUDA 上下文地读取 GPU 状态。"""
    status: dict[str, Any] = {"available": False, "source": "nvidia-smi"}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        status["error"] = "nvidia-smi not found"
        return status
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows:
            status["error"] = "nvidia-smi returned no GPU"
            return status
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 5:
            status["error"] = f"unexpected nvidia-smi output: {rows[0]}"
            return status
        _, device_name, free_mib, total_mib, used_mib = fields
        status.update(
            {
                "available": True,
                "device_count": len(rows),
                "device_name": device_name,
                "memory": {
                    "free_mib": float(free_mib),
                    "total_mib": float(total_mib),
                    "used_mib": float(used_mib),
                    "allocated_mib": None,
                    "reserved_mib": None,
                },
            }
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        status["error"] = str(exc)
    return status


def wait_after_cuda_release(delay: float, label: str = "") -> None:
    """在 worker 退出后留出 CUDA 驱动归还显存的窗口。"""
    if delay <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {delay:.1f}s 释放显存: {label}")
    time.sleep(delay)


def _process_is_running(process: Any) -> bool:
    """兼容 subprocess 和无模型测试内的伪进程对象。"""
    if process is None:
        return False
    poll = getattr(process, "poll", None)
    if callable(poll):
        return poll() is None
    return getattr(process, "returncode", None) is None


def terminate_process_group(
    process: Any,
    label: str,
    terminate_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """终止 worker 进程组，防止超时请求遗留子进程与 CUDA 上下文。"""
    if not _process_is_running(process):
        return
    pid: int | None = getattr(process, "pid", None)
    if pid is None:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            print(f"[{label}] 终止 worker 进程组失败，改为终止主进程: {exc}")
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()

    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=terminate_timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[{label}] worker 未及时退出，强制终止进程组")

    if pid is None:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
    try:
        wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        pass
