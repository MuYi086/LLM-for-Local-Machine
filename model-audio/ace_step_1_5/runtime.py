"""ACE-Step 1.5 一次性 worker 服务的运行时辅助函数。

API 进程不会导入 torch 或 diffusers。每次请求都由全新的 worker 进程独占
模型和 CUDA 上下文。
"""

from __future__ import annotations

# ACE-Step API 只编排请求；模型导入和 CUDA 上下文由一次性 worker 独占。
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_local_runtime import gpu_runtime_lock as shared_gpu_runtime_lock


@dataclass(frozen=True)
class WorkerConfig:
    """一次 worker 调用所需的脚本、临时目录和超时配置。"""

    worker_script: Path
    temp_dir: Path
    timeout: float
    label: str
    file_prefix: str


@dataclass(frozen=True)
class WorkerResult:
    """记录 worker 输出 WAV 及生成元数据，供 HTTP 响应头复用。"""

    audio: bytes
    metadata: dict[str, Any]


def module_available(module_name: str) -> bool:
    """在不实际导入模块的情况下判断其是否可查找到。"""
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def cuda_status() -> dict[str, Any]:
    """使用 nvidia-smi 读取 GPU 状态，不创建 CUDA 上下文。"""
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
    """兼容 subprocess.Popen 和测试替身的进程状态检查。"""
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
    """终止 worker 及其全部子进程，必要时升级为 SIGKILL。"""
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
        print(f"[{label}] worker 未及时退出，强制终止进程组")

    if pid is None:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                wait(timeout=kill_timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            return
        except OSError:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()

    try:
        wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        # 清理阶段不能覆盖 worker 原始异常；SIGKILL 后仍未回收时交给系统继续回收。
        pass


def worker_error_excerpt(output: str, label: str) -> str:
    """保留 worker traceback 的末尾关键信息，作为 HTTP 错误内容。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return f"{label} worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def run_local_worker(payload: dict[str, Any], config: WorkerConfig) -> WorkerResult:
    """使用本项目的 Python 启动 worker，并验证 WAV 与元数据输出。"""
    if not config.worker_script.is_file():
        raise RuntimeError(f"{config.label} worker 脚本不存在: {config.worker_script}")

    config.temp_dir.mkdir(parents=True, exist_ok=True)
    request_fd, request_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_req_",
        suffix=".json",
    )
    output_fd, output_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_out_",
        suffix=".wav",
    )
    metadata_fd, metadata_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_meta_",
        suffix=".json",
    )
    for file_descriptor in (request_fd, output_fd, metadata_fd):
        os.close(file_descriptor)

    process: subprocess.Popen[str] | None = None
    paths = (request_path, output_path, metadata_path)

    try:
        with open(request_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        command = [
            os.environ.get("ACESTEP_PYTHON", sys.executable),
            str(config.worker_script),
            "--input-json",
            request_path,
            "--output-wav",
            output_path,
            "--metadata-json",
            metadata_path,
        ]
        print(f"[{config.label}] 启动 uv worker: python={command[0]}")
        worker_env = os.environ.copy()
        worker_env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=worker_env,
        )
        try:
            stdout, stderr = process.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_group(process, config.label)
            process.communicate()
            raise RuntimeError(f"{config.label} worker 超时（>{config.timeout:.0f}s）") from exc

        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        if process.returncode != 0:
            raise RuntimeError(worker_error_excerpt(stderr or stdout, config.label))
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"{config.label} worker 未生成音频文件。")

        metadata: dict[str, Any] = {}
        if os.path.isfile(metadata_path) and os.path.getsize(metadata_path) > 0:
            with open(metadata_path, encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                metadata = loaded

        with open(output_path, "rb") as file:
            return WorkerResult(audio=file.read(), metadata=metadata)
    finally:
        terminate_process_group(process, config.label)
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def gpu_runtime_lock(lock_path: Path, label: str):
    """获取带等待上限和显存采样的仓库级 GPU 队列。"""
    return shared_gpu_runtime_lock(lock_path, label)


def persist_audio_bytes(audio_bytes: bytes, model_prefix: str, output_dir: Path) -> Path:
    """原子保存成功的 WAV 响应，便于本地检查。"""
    if not audio_bytes:
        raise ValueError("cannot persist empty audio")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{model_prefix}_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_dir / f"{temporary_file.name[1:-4]}.wav"
    try:
        with os.fdopen(output_fd, "wb") as destination:
            destination.write(audio_bytes)
        os.replace(temporary_file, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        temporary_file.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
