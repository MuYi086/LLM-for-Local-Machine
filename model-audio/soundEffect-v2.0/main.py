#!/usr/bin/env python3
"""MOSS-SoundEffect v2.0 文本音效生成 HTTP 服务。"""

from __future__ import annotations

# HTTP 进程保持轻量；Torch 与官方模型包只在每次请求的一次性 worker 中导入。
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from llm_local_runtime import (
    GpuLockTimeoutError,
    GpuQuarantinedError,
)
from llm_local_runtime import (
    gpu_runtime_lock as shared_gpu_runtime_lock,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from soundeffect_runtime import (
    persist_audio_bytes,
    terminate_process_group,
    validate_generated_wav,
)

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent


def expand_path(value: str) -> Path:
    """展开模型、存储和缓存配置路径。"""
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
MODEL_DIR = expand_path(
    os.getenv(
        "MOSS_SOUNDEFFECT_V2_MODEL_DIR",
        str(HF_MIRROR_DIR / "OpenMOSS/MOSS-SoundEffect-v2.0"),
    )
)
CODE_PATH = expand_path(
    os.getenv(
        "MOSS_SOUNDEFFECT_V2_CODE_PATH",
        str(Path.home() / "LLM-depency/MOSS-TTS/moss_soundeffect_v2"),
    )
)
OUTPUT_DIR = expand_path(
    os.getenv("MOSS_SOUNDEFFECT_V2_OUTPUT_DIR", str(STORAGE_DIR / "sound_effect_v2"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv("MOSS_SOUNDEFFECT_V2_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "soundeffect_v2"))
)
GPU_LOCK_FILE = str(
    expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
API_HOST = os.getenv("MOSS_SOUNDEFFECT_V2_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MOSS_SOUNDEFFECT_V2_PORT", os.getenv("PORT", "8362")))
DEVICE = os.getenv("MOSS_SOUNDEFFECT_V2_DEVICE", "cuda")
REQUEST_TIMEOUT = float(os.getenv("MOSS_SOUNDEFFECT_V2_REQUEST_TIMEOUT", "900"))
WORKER_PIPE_DRAIN_TIMEOUT = 5.0
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2"))
MODEL_FILE_SIZES = {
    "text_encoder/model-00001-of-00002.safetensors": 3_441_185_608,
    "text_encoder/model-00002-of-00002.safetensors": 622_329_984,
    "transformer/diffusion_pytorch_model.safetensors": 5_664_274_560,
    "vae/vae_128d_48k.pth": 1_486_465_965,
}
MODEL_CONFIG_FILES = (
    "model_index.json",
    "configuration.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "transformer/config.json",
    "vae/config.json",
)
MODEL_REQUIRED_FILES = (*MODEL_FILE_SIZES, *MODEL_CONFIG_FILES)
MAX_PROMPT_LENGTH = 8_000
DEFAULT_SECONDS = 10.0
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CFG_SCALE = 4.0
DEFAULT_SIGMA_SHIFT = 5.0
DEFAULT_SEED = 0

for name, value in (
    ("MOSS_SOUNDEFFECT_V2_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正的有限数。")
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
WORKER_TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(CODE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(CODE_PATH.parent))


class MossSoundEffectRequest(BaseModel):
    """MOSS-SoundEffect 文本提示词与采样参数。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    seconds: float = Field(default=DEFAULT_SECONDS, ge=0.1, le=30, allow_inf_nan=False)
    num_inference_steps: int = Field(default=DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=200)
    cfg_scale: float = Field(default=DEFAULT_CFG_SCALE, ge=0, le=20, allow_inf_nan=False)
    sigma_shift: float = Field(default=DEFAULT_SIGMA_SHIFT, ge=0, le=20, allow_inf_nan=False)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=4_294_967_295)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        """拒绝纯空白提示词，并去除提示词首尾空格。"""
        value = value.strip()
        if not value:
            raise ValueError("prompt 不能只包含空白字符。")
        return value


def module_available(module_name: str) -> bool:
    """检查模块规格，不在 API 进程导入模型或 CUDA。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def model_status() -> dict[str, Any]:
    """通过权重大小和配置文件检查本地快照完整性，不读取权重内容。"""
    errors: list[str] = []
    for relative_path, expected_size in MODEL_FILE_SIZES.items():
        path = MODEL_DIR / relative_path
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{relative_path}: {exc}")
            continue
        if actual_size != expected_size:
            errors.append(
                f"{relative_path}: 大小为 {actual_size} bytes，预期 {expected_size} bytes"
            )

    for relative_path in MODEL_CONFIG_FILES:
        path = MODEL_DIR / relative_path
        try:
            if path.stat().st_size == 0:
                errors.append(f"{relative_path}: 文件为空")
        except OSError as exc:
            errors.append(f"{relative_path}: {exc}")
    return {"path": str(MODEL_DIR), "complete": not errors, "errors": errors}


def source_status() -> dict[str, Any]:
    """检查固定 revision 的仓库外官方源码入口，不导入模型依赖。"""
    required_files = (
        CODE_PATH / "__init__.py",
        CODE_PATH / "pipeline_moss_soundeffect.py",
        CODE_PATH / "diffsynth/pipelines/wan_audio.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    return {"path": str(CODE_PATH), "complete": not missing, "missing": missing}


def cuda_status() -> dict[str, Any]:
    """读取 GPU 可见性，健康检查不初始化 CUDA 上下文。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {"available": False, "error": "nvidia-smi 没有报告 GPU。"}
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5:
        return {"available": False, "error": f"无法解析 nvidia-smi 输出: {rows[0]}"}
    _, name, free_mib, total_mib, used_mib = fields
    try:
        memory = {
            "free_mib": float(free_mib),
            "total_mib": float(total_mib),
            "used_mib": float(used_mib),
        }
    except ValueError as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, "device_name": name, "memory": memory, "device_count": len(rows)}


def runtime_status() -> dict[str, Any]:
    """汇总模型、推理依赖和 GPU 状态；不执行模型推理。"""
    model = model_status()
    source = source_status()
    dependencies = {
        name: module_available(name)
        for name in ("torch", "torchaudio", "transformers", "diffusers", "safetensors")
    }
    cuda = cuda_status()
    worker = WORKER_SCRIPT.is_file()
    ready = bool(
        model["complete"]
        and source["complete"]
        and all(dependencies.values())
        and cuda["available"]
        and worker
    )
    return {
        "ready": ready,
        "model": model,
        "source": source,
        "dependencies": dependencies,
        "cuda": cuda,
        "worker": worker,
        "device": DEVICE,
        "local_files_only": LOCAL_FILES_ONLY,
    }


def ensure_runtime_ready() -> None:
    """在获取 GPU 锁前拒绝未准备好的模型或运行环境。"""
    status = runtime_status()
    if not status["ready"]:
        raise FileNotFoundError(f"MOSS-SoundEffect v2.0 本地运行环境未就绪: {status}")


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 SoundEffect GPU 推理。"""
    return shared_gpu_runtime_lock(
        GPU_LOCK_FILE,
        label,
        timeout_seconds=GPU_LOCK_WAIT_TIMEOUT,
    )


def wait_after_cuda_release() -> None:
    """worker 退出后在持锁状态下等待 CUDA 资源归还。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 末尾错误行，限制日志返回长度。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[-4_000:] or "SoundEffect worker 未输出错误信息。"


class MossSoundEffectWorkerManager:
    """为每个生成请求启动隔离 worker，并清理临时 JSON/WAV。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """使用当前服务 uv 解释器调用官方 pipeline 并返回 WAV 字节。"""
        if not sys.executable or not Path(sys.executable).is_file():
            raise RuntimeError("没有找到 MOSS-SoundEffect uv 环境的 Python 解释器。")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"SoundEffect worker 不存在: {WORKER_SCRIPT}")

        request_fd, request_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="moss_soundeffect_req_",
            suffix=".json",
        )
        output_fd, output_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="moss_soundeffect_out_",
            suffix=".wav",
        )
        os.close(output_fd)
        request_path = Path(request_name)
        output_path = Path(output_name)
        process: subprocess.Popen[str] | None = None
        process_group_cleanup_attempted = False
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as request_file:
                json.dump(payload, request_file, ensure_ascii=False)
                request_file.flush()
                os.fsync(request_file.fileno())

            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(WORKER_SCRIPT),
                    "--input-json",
                    str(request_path),
                    "--output-wav",
                    str(output_path),
                ],
                cwd=PROJECT_DIR,
                env=worker_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "MOSS-SoundEffect v2.0")
                process_group_cleanup_attempted = True
                try:
                    process.communicate(timeout=WORKER_PIPE_DRAIN_TIMEOUT)
                except (subprocess.TimeoutExpired, OSError) as drain_error:
                    LOGGER.error(
                        "MOSS-SoundEffect worker 输出管道在进程组清理后仍未关闭: %s",
                        drain_error,
                    )
                    for stream in (process.stdout, process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except (OSError, ValueError):
                                pass
                raise TimeoutError(
                    f"MOSS-SoundEffect worker 超时（>{REQUEST_TIMEOUT:g}s）。"
                ) from exc

            if stdout.strip():
                LOGGER.info("MOSS-SoundEffect worker: %s", stdout.rstrip())
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("MOSS-SoundEffect worker 未生成 WAV。")
            audio_bytes = output_path.read_bytes()
            validate_generated_wav(audio_bytes)
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            if not process_group_cleanup_attempted:
                terminate_process_group(process, "MOSS-SoundEffect v2.0")
            request_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


app = FastAPI(title="LLM Local Backend MOSS-SoundEffect v2.0 API")
manager = MossSoundEffectWorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """返回本地模型、运行依赖和 GPU 状态，不加载权重。"""
    status = runtime_status()
    return {
        "code": 200,
        "available": status,
        "paths": {
            "model_dir": str(MODEL_DIR),
            "code_path": str(CODE_PATH),
            "output_dir": str(OUTPUT_DIR),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": str(WORKER_SCRIPT),
        },
        "runtime": {
            "ready": status["ready"],
            "worker_python": sys.executable,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "gpu_scheduling": "shared exclusive file lock",
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "sample_rate": 48_000,
            "max_seconds": 30,
        },
        "last_error": manager.last_error,
    }


@app.post("/v1/moss/soundEffect-v2.0")
def generate(request: MossSoundEffectRequest) -> Response:
    """按提示词生成音效，返回 WAV 并原子保存一份成品。"""
    try:
        ensure_runtime_ready()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = request.model_dump()
    payload.update({"model_dir": str(MODEL_DIR), "code_path": str(CODE_PATH), "device": DEVICE})
    try:
        with gpu_runtime_lock("moss_soundeffect_v2/generate"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    duration = validate_generated_wav(audio_bytes)
                    output_path = persist_audio_bytes(audio_bytes, OUTPUT_DIR)
                    LOGGER.info("已保存 SoundEffect 输出: %s", output_path)
                    return Response(
                        content=audio_bytes,
                        media_type="audio/wav",
                        headers={
                            "X-Audio-Sample-Rate": "48000",
                            "X-Audio-Duration-Seconds": f"{duration:.3f}",
                        },
                    )
                finally:
                    wait_after_cuda_release()
    except (GpuLockTimeoutError, GpuQuarantinedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("MOSS-SoundEffect generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
