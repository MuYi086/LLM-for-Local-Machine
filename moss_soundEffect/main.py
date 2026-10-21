#!/usr/bin/env python3
"""独立 MOSS-SoundEffect v2 uv 服务的 HTTP 封装。"""

from __future__ import annotations

# 学习入口：API 进程只处理参数和生命周期，MOSS 模型在一次性 uv worker 中运行。
import importlib.util
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    execute_localai_worker_file,
    load_gateway_config,
)
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import GpuLockTimeoutError
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from audio_output import persist_audio_bytes
from runtime import UvWorkerConfig, cuda_status, run_uv_worker

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(value: str) -> Path:
    """展开环境变量和用户目录，得到绝对路径。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    """把环境变量解析为统一的布尔值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
SOUNDEFFECT_STORAGE_DIR = expand_path(
    os.getenv("SOUNDEFFECT_STORAGE_DIR", str(STORAGE_DIR / "soundEffect"))
)


def resolve_conda_executable() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and expand_path(conda_exe).is_file():
        return str(expand_path(conda_exe))
    return shutil.which("conda")


def source_package_dir() -> Path:
    if (MOSS_SOUNDEFFECT_CODE_PATH / "moss_soundeffect_v2").is_dir():
        return MOSS_SOUNDEFFECT_CODE_PATH / "moss_soundeffect_v2"
    return MOSS_SOUNDEFFECT_CODE_PATH


def source_package_is_available() -> bool:
    return (source_package_dir() / "__init__.py").is_file()


def local_model_is_complete(model_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            model_dir / "model_index.json",
            model_dir / "transformer" / "diffusion_pytorch_model.safetensors",
            model_dir / "vae" / "vae_128d_48k.pth",
        )
    )


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "$HOME/hf-mirror"))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
API_HOST = os.getenv("MOSS_SOUNDEFFECT_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MOSS_SOUNDEFFECT_PORT", os.getenv("PORT", "8312")))

MOSS_SOUNDEFFECT_MODEL_DIR = expand_path(
    os.getenv(
        "MOSS_SOUNDEFFECT_MODEL_DIR",
        str(HF_MIRROR_DIR / "OpenMOSS-Team/MOSS-SoundEffect-v2.0"),
    )
)
MOSS_SOUNDEFFECT_CODE_PATH = expand_path(
    os.getenv(
        "MOSS_SOUNDEFFECT_CODE_PATH",
        str(expand_path("$HOME/tts-depency") / "MOSS-TTS"),
    )
)
MOSS_SOUNDEFFECT_DEVICE = os.getenv("MOSS_SOUNDEFFECT_DEVICE", "cuda")
MOSS_SOUNDEFFECT_DTYPE = os.getenv("MOSS_SOUNDEFFECT_DTYPE", "bfloat16")
MOSS_SOUNDEFFECT_DEFAULT_SECONDS = float(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SECONDS", "10"))
MOSS_SOUNDEFFECT_DEFAULT_STEPS = int(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_STEPS", "100"))
MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE = float(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE", "4.0"))
MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT = float(
    os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT", "5.0")
)
MOSS_SOUNDEFFECT_DEFAULT_SEED = int(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SEED", "0"))
MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO = env_bool("MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO", True)
MOSS_SOUNDEFFECT_REQUEST_TIMEOUT = float(os.getenv("MOSS_SOUNDEFFECT_REQUEST_TIMEOUT", "600"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="UNITALE_MOSS_SOUNDEFFECT_PROVIDER",
    default_model="unitale.moss-soundeffect-v2.worker.v1",
    request_timeout=MOSS_SOUNDEFFECT_REQUEST_TIMEOUT,
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))

WORKER_SCRIPT = PROJECT_DIR / "worker.py"
WORKER_TMP_DIR = RUNTIME_CACHE_DIR / "soundeffect_worker"
for directory in (
    RUNTIME_CACHE_DIR,
    WORKER_TMP_DIR,
    SOUNDEFFECT_STORAGE_DIR,
    GPU_LOCK_FILE.parent,
):
    directory.mkdir(parents=True, exist_ok=True)


class ForceCORS(BaseHTTPMiddleware):
    """为本地 WebUI 提供跨域响应和 OPTIONS 预检支持。"""

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "false",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response


class SoundEffectGenerateRequest(BaseModel):
    """MOSS-SoundEffect v2 的文本、时长和扩散参数请求模型。"""

    prompt: str = Field(min_length=1, max_length=2_000)
    seconds: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SECONDS, gt=0, le=30)
    num_inference_steps: int = Field(default=MOSS_SOUNDEFFECT_DEFAULT_STEPS, gt=0, le=500)
    cfg_scale: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE, ge=0, le=100)
    sigma_shift: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT, gt=0, le=100)
    seed: int = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SEED)
    device: str | None = Field(default=None, min_length=1)
    torch_dtype: str | None = Field(default=None, min_length=1)

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt 不能为空")
        return normalized


def gpu_runtime_lock(label: str):
    """使用仓库级 GPU 文件锁串行化声效生成。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release() -> None:
    if CUDA_RELEASE_DELAY > 0:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s，确保 worker 显存已释放")
        time.sleep(CUDA_RELEASE_DELAY)


class SoundEffectWorkerManager:
    """把声效请求转换为 worker JSON，并集中记录 worker 错误。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: SoundEffectGenerateRequest) -> dict[str, Any]:
        """合并请求参数与环境默认值，生成 worker 输入。"""
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "num_inference_steps": request.num_inference_steps,
            "cfg_scale": request.cfg_scale,
            "sigma_shift": request.sigma_shift,
            "seed": request.seed,
            "device": request.device or MOSS_SOUNDEFFECT_DEVICE,
            "torch_dtype": request.torch_dtype or MOSS_SOUNDEFFECT_DTYPE,
            "model_path": str(MOSS_SOUNDEFFECT_MODEL_DIR),
            "code_path": str(MOSS_SOUNDEFFECT_CODE_PATH),
            "local_files_only": LOCAL_FILES_ONLY,
            "disable_torchdynamo": MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO,
        }

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """在当前 uv 环境启动一次 MOSS 声效推理并读取 WAV。"""
        config = UvWorkerConfig(
            python_executable=sys.executable,
            worker_script=str(WORKER_SCRIPT),
            model_dir=str(MOSS_SOUNDEFFECT_MODEL_DIR),
            code_path=str(MOSS_SOUNDEFFECT_CODE_PATH),
            temp_dir=str(WORKER_TMP_DIR),
            timeout=MOSS_SOUNDEFFECT_REQUEST_TIMEOUT,
            label="SoundEffect",
            file_prefix="soundeffect",
        )
        try:
            audio = run_uv_worker(payload, config)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def run_localai_worker(self, payload: dict[str, Any]) -> bytes:
        """通过固定 worker bridge 执行声效请求并读取 WAV 成品。"""
        artifact, _ = execute_localai_worker_file(
            config=GATEWAY_CONFIG,
            operation="run",
            payload={"worker_input": payload},
            output_dir=WORKER_TMP_DIR,
            suffix=".wav",
            media_type="audio/wav",
        )
        try:
            audio = artifact.read_bytes()
            if not audio:
                raise RuntimeError("LocalAI MOSS-SoundEffect worker 返回空 WAV。")
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            artifact.unlink(missing_ok=True)


app = FastAPI(title="Unitale MOSS-SoundEffect v2 API")
app.add_middleware(ForceCORS)
manager = SoundEffectWorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """在不加载模型的情况下报告源码、权重和 GPU 状态。"""
    cuda = cuda_status()
    flash_attn_available = any(
        importlib.util.find_spec(name) is not None
        for name in ("flash_attn", "flash_attn_interface", "sageattention")
    )
    source_available = source_package_is_available()
    return {
        "code": 200,
        "paths": {
            "project_dir": str(PROJECT_DIR),
            "model_dir": str(MOSS_SOUNDEFFECT_MODEL_DIR),
            "code_path": str(MOSS_SOUNDEFFECT_CODE_PATH),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
            "soundeffect_storage_dir": str(SOUNDEFFECT_STORAGE_DIR),
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "uv": True,
            "python": sys.executable,
            "model_dir": MOSS_SOUNDEFFECT_MODEL_DIR.is_dir(),
            "model_weights_complete": local_model_is_complete(MOSS_SOUNDEFFECT_MODEL_DIR),
            "source_repo": MOSS_SOUNDEFFECT_CODE_PATH.is_dir(),
            "moss_package": source_available,
            "worker_script": WORKER_SCRIPT.is_file(),
            "cuda": cuda["available"],
            "flash_attn": flash_attn_available,
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": MOSS_SOUNDEFFECT_REQUEST_TIMEOUT,
            "device": MOSS_SOUNDEFFECT_DEVICE,
            "torch_dtype": MOSS_SOUNDEFFECT_DTYPE,
            "default_seconds": MOSS_SOUNDEFFECT_DEFAULT_SECONDS,
            "default_num_inference_steps": MOSS_SOUNDEFFECT_DEFAULT_STEPS,
            "default_cfg_scale": MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE,
            "default_sigma_shift": MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT,
            "disable_torchdynamo": MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO,
            "flash_attention_policy": (
                "optional; using PyTorch SDPA when unavailable"
                if flash_attn_available
                else "not required; PyTorch SDPA fallback"
            ),
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
        },
        "last_errors": {"soundeffect": manager.last_error},
    }


@app.post("/v1/moss/soundEffect")
def generate(request: SoundEffectGenerateRequest) -> Response:
    """串行调用 worker、保存生成音效并返回 WAV 响应。"""
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        if GATEWAY_CONFIG.provider == "localai":
            with manager.lock:
                try:
                    audio = manager.run_localai_worker(payload)
                    if not audio:
                        raise RuntimeError("SoundEffect worker 返回空音频。")
                    saved_output_path = persist_audio_bytes(
                        audio,
                        "moss_soundeffect",
                        SOUNDEFFECT_STORAGE_DIR,
                    )
                    print(f"[MOSS-SoundEffect] 已保存音效: {saved_output_path}")
                    return Response(content=audio, media_type="audio/wav")
                except HTTPException:
                    raise
                except (BackendQuarantinedError, LocalAIUnavailableError) as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except LocalAIRequestError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
        else:
            with gpu_runtime_lock("soundeffect/generate"):
                with manager.lock:
                    try:
                        audio = manager.run_worker(payload)
                        if not audio:
                            raise RuntimeError("SoundEffect worker 返回空音频。")
                        saved_output_path = persist_audio_bytes(
                            audio,
                            "moss_soundeffect",
                            SOUNDEFFECT_STORAGE_DIR,
                        )
                        print(f"[MOSS-SoundEffect] 已保存音效: {saved_output_path}")
                        return Response(content=audio, media_type="audio/wav")
                    except HTTPException:
                        raise
                    except Exception as exc:
                        raise HTTPException(status_code=500, detail=str(exc)) from exc
                    finally:
                        wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 MOSS-SoundEffect v2 (uv)")
    print("==================================================")
    print(f"[配置] uv Python: {sys.executable}")
    print(f"[配置] 模型目录: {MOSS_SOUNDEFFECT_MODEL_DIR}")
    print(f"[配置] 上游源码: {MOSS_SOUNDEFFECT_CODE_PATH}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {WORKER_SCRIPT}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, timeout={MOSS_SOUNDEFFECT_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
