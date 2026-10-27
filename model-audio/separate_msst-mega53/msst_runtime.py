"""MSST Mega 53 HTTP 服务的轻量依赖检查与 GPU 状态查询。"""

from __future__ import annotations

# API 进程不导入 Torch；GPU 状态由标准库查询。
import importlib.util
import shutil
import subprocess
from typing import Any


def module_available(module_name: str) -> bool:
    """检查依赖模块规格，不导入 Torch 或初始化 CUDA。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def cuda_status() -> dict[str, Any]:
    """通过 nvidia-smi 查询 GPU 状态，不在 API 进程创建 CUDA 上下文。"""
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
                },
            }
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        status["error"] = str(exc)
    return status
