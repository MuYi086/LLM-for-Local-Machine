"""Index-Echo GGUF 服务的轻量运行时辅助函数。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from typing import Any


def resolve_executable(executable: str) -> str | None:
    """解析 PATH 命令或显式可执行文件路径。"""
    expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(executable)))
    if os.path.dirname(executable):
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    return shutil.which(executable)


def terminate_process_group(
    process: Any,
    label: str,
    terminate_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """终止一次性 worker 及其 llama-server 子进程。"""
    if process is None or process.poll() is not None:
        return

    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        print(f"[{label}] 终止 worker 进程组失败，改为终止主进程: {exc}")
        process.terminate()

    try:
        process.wait(timeout=terminate_timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[{label}] worker 未及时退出，强制终止进程组")

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()

    try:
        process.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        pass
