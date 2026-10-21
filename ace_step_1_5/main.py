#!/usr/bin/env python3
"""Unitale ACE-Step 1.5 XL Turbo BGM HTTP 服务。

API 进程保持轻量。每次生成都会获取共享 GPU 锁、启动一个 worker、返回 WAV，
并且只在 worker 进程退出后释放锁。
"""

from __future__ import annotations

import io

# 学习入口：BGM 请求走“校验 -> GPU 文件锁 -> 一次性 worker -> WAV 落盘”流程。
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    UnsupportedCapabilityError,
    load_gateway_config,
)
from inference_gateway.execution import execute_localai_operation
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import GpuLockTimeoutError

from runtime import (
    WorkerConfig,
    WorkerResult,
    cuda_status,
    gpu_runtime_lock,
    module_available,
    persist_audio_bytes,
    run_local_worker,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
MIN_SECONDS = 10.0
MAX_SECONDS = 600.0
SAMPLE_RATE = 48_000
CHANNELS = 2
MODEL_NAME = "acestep-v15-xl-turbo-diffusers"
REQUIRED_MODEL_PATHS = (
    "model_index.json",
    "condition_encoder",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)


def expand_path(value: str) -> Path:
    """展开环境变量与用户目录，统一得到绝对路径。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    """读取常见形式的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage")))
BGM_STORAGE_DIR = expand_path(os.getenv("BGM_STORAGE_DIR", str(STORAGE_DIR / "bgm")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
ACESTEP_MODEL_DIR = expand_path(
    os.getenv(
        "ACESTEP_MODEL_DIR",
        str(HF_MIRROR_DIR / "ACE-Step/acestep-v15-xl-turbo-diffusers"),
    )
)
ACESTEP_OUTPUT_DIR = expand_path(os.getenv("ACESTEP_OUTPUT_DIR", str(BGM_STORAGE_DIR)))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
ACESTEP_DEVICE = os.getenv("ACESTEP_DEVICE", "cuda")
ACESTEP_DTYPE = os.getenv("ACESTEP_DTYPE", "bfloat16")
ACESTEP_OFFLOAD = os.getenv("ACESTEP_OFFLOAD", "model")
ACESTEP_VAE_TILING = env_bool("ACESTEP_VAE_TILING", True)
ACESTEP_DEFAULT_SECONDS = float(os.getenv("ACESTEP_DEFAULT_SECONDS", "60"))
ACESTEP_DEFAULT_STEPS = int(os.getenv("ACESTEP_DEFAULT_STEPS", "8"))
ACESTEP_DEFAULT_SEED = int(os.getenv("ACESTEP_DEFAULT_SEED", "-1"))
ACESTEP_REQUEST_TIMEOUT = float(os.getenv("ACESTEP_REQUEST_TIMEOUT", "1800"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
LOCALAI_PROVIDER_ENV = "UNITALE_ACESTEP_PROVIDER"
LOCALAI_MODEL_ID = "unitale.ace-step15.xl-turbo.q8dit"
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env=LOCALAI_PROVIDER_ENV,
    default_model=LOCALAI_MODEL_ID,
    request_timeout=ACESTEP_REQUEST_TIMEOUT,
)
LAST_LOCALAI_ERROR: str | None = None
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8313"))
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
WORKER_TMP_DIR = expand_path(
    os.getenv("ACESTEP_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "ace_step_1_5_worker"))
)

for directory in (RUNTIME_CACHE_DIR, WORKER_TMP_DIR, ACESTEP_OUTPUT_DIR, GPU_LOCK_FILE.parent):
    directory.mkdir(parents=True, exist_ok=True)

if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ForceCORS(BaseHTTPMiddleware):
    """保留本地 WebUI 所需的宽松跨域行为。"""

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


class AceStepBgmRequest(BaseModel):
    """有声内容 BGM 工作流使用的文本生成音乐参数。"""

    prompt: str = Field(min_length=1, max_length=2_000)
    seconds: float = Field(default=ACESTEP_DEFAULT_SECONDS, ge=MIN_SECONDS, le=MAX_SECONDS)
    steps: int = Field(default=ACESTEP_DEFAULT_STEPS, ge=1, le=20)
    bpm: int | None = Field(default=None, ge=30, le=240)
    keyscale: str | None = Field(default=None, max_length=64)
    timesignature: str | None = Field(default=None, max_length=16)
    seed: int = Field(default=ACESTEP_DEFAULT_SEED)

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt 不能为空")
        return normalized

    @field_validator("keyscale", "timesignature")
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


def model_status() -> dict[str, object]:
    """检查模型目录、分片和运行依赖，不触发模型加载。"""
    required = {
        name: (
            (ACESTEP_MODEL_DIR / name).is_file()
            if name.endswith(".json")
            else (ACESTEP_MODEL_DIR / name).is_dir()
        )
        for name in REQUIRED_MODEL_PATHS
    }
    transformer_weights = list((ACESTEP_MODEL_DIR / "transformer").glob("*.safetensors"))
    return {
        "required": required,
        "transformer_weights": bool(transformer_weights),
        "transformer_weight_files": len(transformer_weights),
        "complete": all(required.values()) and bool(transformer_weights),
    }


def wait_after_cuda_release() -> None:
    """等待已终止 worker 的 CUDA 上下文消失后再释放 GPU 锁。"""
    if CUDA_RELEASE_DELAY > 0:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s，确保 ACE-Step worker 显存已释放")
        time.sleep(CUDA_RELEASE_DELAY)


def validate_localai_request(request: AceStepBgmRequest) -> None:
    """拒绝 SoundGeneration HTTP 合同当前无法传递的 ACE-Step 参数。"""
    if request.steps != ACESTEP_DEFAULT_STEPS:
        raise UnsupportedCapabilityError(
            "LocalAI SoundGeneration 当前没有逐请求 num_inference_steps 字段；"
            f"ACE-Step XL Turbo 只能使用当前服务默认 steps={ACESTEP_DEFAULT_STEPS}。"
        )
    if request.seed >= 0:
        raise UnsupportedCapabilityError(
            "LocalAI SoundGeneration HTTP 合同当前不能传递 ACE-Step seed；"
            "请使用 seed=-1 进行随机生成，或切回 native provider。"
        )


def validate_localai_wav(audio_bytes: bytes) -> tuple[int, int]:
    """完整解码返回音频并检查 Unitale 的 48 kHz 双声道 WAV 合同。"""
    try:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(audio_bytes), mode="r") as wav_file:
            sample_rate = wav_file.samplerate
            channels = wav_file.channels
            frame_count = wav_file.frames
            if frame_count <= 0:
                raise LocalAIRequestError("LocalAI ACE-Step 返回了空 WAV。")
            if sample_rate != SAMPLE_RATE or channels != CHANNELS:
                raise LocalAIRequestError(
                    "LocalAI ACE-Step WAV 格式与当前接口不同："
                    f"期望 {SAMPLE_RATE} Hz/{CHANNELS} 声道，得到 {sample_rate} Hz/{channels} 声道。"
                )
            decoded_frames = sum(
                len(block)
                for block in wav_file.blocks(blocksize=65_536, dtype="float32", always_2d=True)
            )
    except LocalAIRequestError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise LocalAIRequestError(f"LocalAI ACE-Step 返回的 WAV 无效：{exc}") from exc
    if decoded_frames != frame_count:
        raise LocalAIRequestError(
            f"LocalAI ACE-Step WAV 不完整：头部声明 {frame_count} 帧，实际解码 {decoded_frames} 帧。"
        )
    return sample_rate, channels


def generate_localai(request: AceStepBgmRequest) -> tuple[bytes, float]:
    """在共享 LocalAI GPU 租约内执行不含歌词字段的 BGM 请求。"""
    audio_bytes, elapsed = execute_localai_operation(
        config=GATEWAY_CONFIG,
        operation=lambda client: client.generate_sound(
            text=request.prompt,
            duration_seconds=request.seconds,
            bpm=request.bpm,
            keyscale=request.keyscale,
            timesignature=request.timesignature,
        ),
    )
    validate_localai_wav(audio_bytes)
    return audio_bytes, elapsed


ACESTEP_WORKER = WorkerConfig(
    worker_script=WORKER_SCRIPT,
    temp_dir=WORKER_TMP_DIR,
    timeout=ACESTEP_REQUEST_TIMEOUT,
    label="ACE-Step 1.5",
    file_prefix="ace_step_1_5",
)


class AceStepWorkerManager:
    """负责组装请求、启动一次 worker 并保存最近一次错误。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: AceStepBgmRequest) -> dict[str, object]:
        """将经过校验的 BGM 参数和环境默认值序列化给 worker。"""
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "steps": request.steps,
            "bpm": request.bpm,
            "keyscale": request.keyscale,
            "timesignature": request.timesignature,
            "seed": request.seed,
            "model_path": str(ACESTEP_MODEL_DIR),
            "device": ACESTEP_DEVICE,
            "dtype": ACESTEP_DTYPE,
            "offload": ACESTEP_OFFLOAD,
            "vae_tiling": ACESTEP_VAE_TILING,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": str(RUNTIME_CACHE_DIR),
            "hf_mirror_dir": str(HF_MIRROR_DIR),
        }

    def run_worker(self, payload: dict[str, object]) -> WorkerResult:
        """在当前 ACE-Step uv 项目中执行一次隔离生成。"""
        status = model_status()
        if not status["complete"]:
            missing = [name for name, present in status["required"].items() if not present]
            weights = "transformer/*.safetensors" if not status["transformer_weights"] else ""
            details = ", ".join(filter(None, [*missing, weights]))
            raise RuntimeError(
                f"ACE-Step 本地权重不完整或不存在: {ACESTEP_MODEL_DIR}; 缺少: {details}"
            )

        try:
            result = run_local_worker(payload, ACESTEP_WORKER)
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = str(exc)
            raise


app = FastAPI(title="Unitale ACE-Step 1.5 BGM API")
app.add_middleware(ForceCORS)
manager = AceStepWorkerManager()


@app.get("/v1/health")
def health() -> dict[str, object]:
    """返回 BGM 服务的依赖、模型、GPU 和运行参数状态。"""
    cuda = cuda_status()
    localai_client = LocalAIClient(GATEWAY_CONFIG) if GATEWAY_CONFIG.provider == "localai" else None
    localai_healthy = bool(localai_client and localai_client.health())
    localai_model_registered = bool(
        localai_client and localai_healthy and localai_client.model_ready()
    )
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "model_dir": str(ACESTEP_MODEL_DIR),
            "output_dir": str(ACESTEP_OUTPUT_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
            "project_dir": str(PROJECT_DIR),
            "worker_python": os.environ.get("ACESTEP_PYTHON", os.sys.executable),
        },
        "available": {
            "uv": bool(shutil.which("uv")) or Path(os.sys.executable).is_file(),
            "diffusers": module_available("diffusers"),
            "accelerate": module_available("accelerate"),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
            "model_complete": model_status()["complete"],
            "worker_script": WORKER_SCRIPT.is_file(),
            "selected_provider": GATEWAY_CONFIG.provider,
            "selected_provider_ready": (
                localai_model_registered if localai_client else model_status()["complete"]
            ),
            "localai_healthy": localai_healthy,
            "localai_model_registered": localai_model_registered,
        },
        "model": model_status(),
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "model": f"ACE-Step/{MODEL_NAME}",
            "device": ACESTEP_DEVICE,
            "dtype": ACESTEP_DTYPE,
            "offload": ACESTEP_OFFLOAD,
            "vae_tiling": ACESTEP_VAE_TILING,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "min_seconds": MIN_SECONDS,
            "max_seconds": MAX_SECONDS,
            "default_seconds": ACESTEP_DEFAULT_SECONDS,
            "default_steps": ACESTEP_DEFAULT_STEPS,
            "default_seed": ACESTEP_DEFAULT_SEED,
            "request_timeout": ACESTEP_REQUEST_TIMEOUT,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if localai_client else None,
            "contract": "BGM request without lyrics; backend receives its empty-lyrics default",
        },
        "last_errors": {
            "ace_step_1_5": manager.last_error,
            "localai": LAST_LOCALAI_ERROR,
        },
    }


@app.post("/v1/aceStep/bgm")
def generate_bgm(request: AceStepBgmRequest) -> Response:
    """串行执行一次 ACE-Step 生成，并将 WAV 保存到 BGM 目录。"""
    global LAST_LOCALAI_ERROR
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("ACE-Step request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if GATEWAY_CONFIG.provider == "localai":
        try:
            validate_localai_request(request)
            audio, _elapsed = generate_localai(request)
            saved_output_path = persist_audio_bytes(audio, "ace_step_1_5", ACESTEP_OUTPUT_DIR)
            manager.last_error = None
            LAST_LOCALAI_ERROR = None
            print(f"[ACE-Step] 已保存 LocalAI BGM: {saved_output_path}")
            headers = {
                "X-ACE-Step-Model": MODEL_NAME,
                # SoundGeneration 不返回随机 seed；-1 表示后端负责生成随机值。
                "X-ACE-Step-Seed": str(request.seed),
                "X-ACE-Step-Sample-Rate": str(SAMPLE_RATE),
            }
            return Response(content=audio, media_type="audio/wav", headers=headers)
        except UnsupportedCapabilityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except BackendQuarantinedError as exc:
            LAST_LOCALAI_ERROR = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (LocalAIRequestError, LocalAIUnavailableError) as exc:
            LAST_LOCALAI_ERROR = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LAST_LOCALAI_ERROR = str(exc)
            LOGGER.exception("LocalAI ACE-Step request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock(GPU_LOCK_FILE, "ace_step_1_5/generate"):
            with manager.lock:
                try:
                    result = manager.run_worker(payload)
                    if isinstance(result, WorkerResult):
                        audio = result.audio
                        metadata = result.metadata
                    else:
                        audio = result
                        metadata = {}
                    saved_output_path = persist_audio_bytes(
                        audio, "ace_step_1_5", ACESTEP_OUTPUT_DIR
                    )
                    print(f"[ACE-Step] 已保存 BGM: {saved_output_path}")
                    headers = {
                        "X-ACE-Step-Model": MODEL_NAME,
                        "X-ACE-Step-Seed": str(metadata.get("seed", request.seed)),
                        "X-ACE-Step-Sample-Rate": str(metadata.get("sample_rate", SAMPLE_RATE)),
                    }
                    return Response(content=audio, media_type="audio/wav", headers=headers)
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("ACE-Step request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale ACE-Step 1.5 XL Turbo BGM API")
    print("==================================================")
    print(f"[配置] uv project: {PROJECT_DIR}")
    print(f"[配置] 模型目录: {ACESTEP_MODEL_DIR}")
    print(f"[配置] BGM 输出目录: {ACESTEP_OUTPUT_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, dtype={ACESTEP_DTYPE}, "
        f"offload={ACESTEP_OFFLOAD}, timeout={ACESTEP_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
