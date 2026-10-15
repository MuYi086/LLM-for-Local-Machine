"""Index-Echo 原生 服务的轻量运行时辅助函数。"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import signal
import subprocess
from pathlib import Path
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
    """终止一次性 worker 及其子进程。"""
    if process is None:
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
        # 主进程退出后子进程仍可能存活，继续回收整个进程组。
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


NATIVE_PACKAGES = {
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "transformers": "5.6.0",
    "tokenizers": "0.22.2",
    "librosa": "1.0.0",
    "onnxruntime": "1.30.0",
    "wetext": "0.0.4",
    "kaldifst": "1.8.0",
    "conformer": "0.3.2",
    "hydra-core": "1.3.7",
    "HyperPyYAML": "1.2.3",
    "lightning": "2.6.6",
    "x-transformers": "2.11.24",
    "einx": "0.4.3",
    "frozendict": "2.4.7",
    "pyworld": "0.3.5",
    "soundfile": "0.13.1",
    "modelscope": "1.37.1",
    "safetensors": "0.7.0",
    "numpy": "2.4.6",
}


def native_runtime_status(model_dir: Path, device: str) -> dict:
    """只检查完整包、官方依赖版本和显存必要条件，不导入模型或 Torch。"""
    components = (
        "modeling_dubbing.py",
        "config.json",
        "stlm_llm",
        "stlm_ckpt",
        "stlm_omni",
        "cosyvoice3",
        "bridge",
        "code",
    )
    missing_components = [name for name in components if not (model_dir / name).exists()]
    packages = {}
    for name, expected in NATIVE_PACKAGES.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        packages[name] = {
            "installed": installed,
            "expected": expected,
            "available": installed is not None and installed.split("+")[0] == expected,
        }
    gpu_ready = False
    gpu_detail = "官方仅验证 CUDA 单卡至少 24 GB 路径。"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        capacities = [int(value) for value in result.stdout.splitlines()]
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            # 非数字 UUID 配置无法仅用此查询映射，明确报告不满足预检。
            indices = [int(value.strip()) for value in visible.split(",") if value.strip()]
            capacities = [capacities[index] for index in indices]
        device_index = int(device.split(":", 1)[1]) if ":" in device else 0
        gpu_ready = device.startswith("cuda") and capacities[device_index] >= 24_000
        gpu_detail = f"可见 GPU 总显存 {capacities} MiB；所选设备 {device}，要求单卡至少 24 GB。"
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        gpu_detail = f"{gpu_detail} 检查失败：{exc}"
    available = {
        "native_model": not missing_components,
        "native_dependencies": all(value["available"] for value in packages.values()),
        "gpu_memory": gpu_ready,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    return {
        "available": available,
        "ready": all(available.values()),
        "missing_components": missing_components,
        "packages": packages,
        "gpu": gpu_detail,
    }
