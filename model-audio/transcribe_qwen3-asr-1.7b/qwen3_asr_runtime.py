"""Qwen3-ASR 服务不依赖模型包的运行时辅助函数。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from typing import Any


def cuda_status() -> dict[str, Any]:
    """通过 nvidia-smi 检查 GPU，避免 HTTP 进程创建 CUDA 上下文。"""
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


def process_is_running(process: Any) -> bool:
    """判断 worker 是否仍在运行，兼容 subprocess 和测试替身。"""
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
    """先终止 worker 进程组，再强杀超时进程，避免推理子进程残留。"""
    if not process_is_running(process):
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
            wait = getattr(process, "wait", None)
            if callable(wait):
                try:
                    wait(timeout=kill_timeout)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
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
        pass

    if pid is None:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            print(f"[{label}] 强制终止 worker 进程组失败: {exc}")
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
    try:
        wait(timeout=kill_timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
