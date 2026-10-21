#!/usr/bin/env python3
"""Unitale Stable Audio 3 Medium HTTP 服务。

API 进程负责校验请求并持有 GPU 锁；每个请求由全新的 worker 进程加载重型模型，
并在返回响应前退出。
"""

from __future__ import annotations

import io

# 学习入口：请求先经过 Pydantic 校验，再持有共享 GPU 锁，最后交给一次性 worker。
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal

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
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import GpuLockTimeoutError

from runtime import (
    WorkerConfig,
    cuda_status,
    gpu_runtime_lock,
    module_available,
    persist_audio_bytes,
    run_local_worker,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
REQUIRED_MODEL_FILES = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
)
MAX_SECONDS = 380.0
SAMPLE_RATE = 44_100
CHANNELS = 2
LOCALAI_PROVIDER_ENV = "UNITALE_STABLE_AUDIO_3_MEDIUM_PROVIDER"
LOCALAI_MODEL_ID = "unitale.stable-audio3-medium.q8-0"


def expand_path(value: str) -> Path:
    """展开环境变量与 ``~``，把配置路径统一转换为绝对路径。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    """读取常见形式的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def local_model_is_complete(model_dir: Path) -> bool:
    """只检查 worker 必需的最小权重文件集合，不在健康检查时加载模型。"""
    return all((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES)


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage")))
SOUNDEFFECT_STORAGE_DIR = expand_path(
    os.getenv("SOUNDEFFECT_STORAGE_DIR", str(STORAGE_DIR / "soundEffect"))
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(PROJECT_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8311"))

STABLE_AUDIO_3_MEDIUM_MODEL_DIR = expand_path(
    os.getenv(
        "STABLE_AUDIO_3_MEDIUM_MODEL_DIR",
        str(HF_MIRROR_DIR / "stabilityai/stable-audio-3-medium"),
    )
)
STABLE_AUDIO_3_REPO_PATH = expand_path(
    os.getenv("STABLE_AUDIO_3_REPO_PATH", "~/tts-depency/stable-audio-3")
)
STABLE_AUDIO_3_MEDIUM_DEVICE = os.getenv("STABLE_AUDIO_3_MEDIUM_DEVICE", "cuda")
STABLE_AUDIO_3_MEDIUM_DTYPE = os.getenv("STABLE_AUDIO_3_MEDIUM_DTYPE", "float16")
STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS", "7")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS = int(os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS", "8"))
STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE", "1.0")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED = int(os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED", "-1"))
STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT", "900")
)
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env=LOCALAI_PROVIDER_ENV,
    default_model=LOCALAI_MODEL_ID,
    request_timeout=STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
)
LAST_LOCALAI_ERROR: str | None = None
STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR = expand_path(
    os.getenv("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", str(SOUNDEFFECT_STORAGE_DIR))
)
# 上游运行时已经验证过 SDPA/flex-attention 回退方案。由于旧的 cp310
# FlashAttention wheel 不能用于 Python 3.12，这里将回退方案作为 uv 默认值；
# 部署环境可以设为 1，强制使用官方路径。
STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN = env_bool(
    "STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN", False
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
WORKER_TMP_DIR = RUNTIME_CACHE_DIR / "stable_audio_3_medium_worker"

for directory in (
    RUNTIME_CACHE_DIR,
    WORKER_TMP_DIR,
    STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR,
    GPU_LOCK_FILE.parent,
):
    directory.mkdir(parents=True, exist_ok=True)

if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ForceCORS(BaseHTTPMiddleware):
    """保留本地 WebUI 所需的宽松跨域行为，并处理 OPTIONS 预检。"""

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


class StableAudio3MediumGenerateRequest(BaseModel):
    """Stable Audio 3 Medium 文本生成音频的请求参数。"""

    prompt: str = Field(min_length=1, max_length=2_000)
    seconds: float | None = Field(default=None, gt=0, le=MAX_SECONDS)
    duration: float | None = Field(default=None, gt=0, le=MAX_SECONDS)
    steps: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS, ge=1, le=100)
    cfg_scale: float = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE, ge=0, le=100)
    seed: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED)
    device: Literal["cuda"] | None = None
    dtype: Literal["float16"] | None = None

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt 不能为空")
        return normalized

    @model_validator(mode="after")
    def normalize_duration(self):
        if self.seconds is None and self.duration is None:
            self.seconds = STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS
        elif self.seconds is None:
            self.seconds = self.duration
        elif self.duration is not None and self.seconds != self.duration:
            raise ValueError("seconds 与 duration 必须相同")
        return self


def wait_after_cuda_release() -> None:
    """worker 退出后短暂等待 CUDA 上下文消失，再释放共享 GPU 锁。"""
    if CUDA_RELEASE_DELAY > 0:
        print(
            f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s，确保 Stable Audio 3 Medium worker 显存已释放"
        )
        time.sleep(CUDA_RELEASE_DELAY)


def flash_attention_status() -> dict[str, object]:
    """不导入编译扩展，直接报告 FlashAttention 与回退实现的可用性。"""
    available = module_available("flash_attn")
    return {
        "available": available,
        "required_by_default": STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN,
        "mode": "flash-attn" if available else "upstream SDPA/flex fallback",
        "note": (
            "worker imports flash_attn when available; upstream stable-audio-3 "
            "falls back to flex_attention/SDPA when it is absent"
        ),
    }


STABLE_AUDIO_3_MEDIUM_WORKER = WorkerConfig(
    worker_script=WORKER_SCRIPT,
    temp_dir=WORKER_TMP_DIR,
    timeout=STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
    label="Stable Audio 3 Medium",
    file_prefix="stable_audio_3_medium",
)


def validate_localai_request(request: StableAudio3MediumGenerateRequest) -> None:
    """拒绝 SoundGeneration HTTP 合同无法逐请求表达的原生参数。"""
    if request.dtype is not None:
        raise UnsupportedCapabilityError(
            "LocalAI Stable Audio 候选固定使用 Q8 GGUF；不能满足请求中的 PyTorch dtype 覆盖。"
        )
    if request.steps != STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS:
        raise UnsupportedCapabilityError(
            "LocalAI SoundGeneration 当前没有逐请求 num_inference_steps 字段；"
            f"仅接受当前服务默认 steps={STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS}。"
        )
    if request.cfg_scale != STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE:
        raise UnsupportedCapabilityError(
            "LocalAI SoundGeneration 当前没有逐请求 Stable Audio guidance_scale 字段；"
            f"仅接受当前服务默认 cfg_scale={STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE}。"
        )
    if request.seed >= 0:
        raise UnsupportedCapabilityError(
            "LocalAI SoundGeneration 当前不能传递 Stable Audio seed；"
            "请使用 seed=-1 进行随机生成，或切回 native provider。"
        )


def validate_localai_wav(audio_bytes: bytes) -> tuple[int, int]:
    """完整解码返回音频并检查 Stable Audio 3 Medium WAV 合同。"""
    try:
        import soundfile as sf

        with sf.SoundFile(io.BytesIO(audio_bytes), mode="r") as wav_file:
            sample_rate = wav_file.samplerate
            channels = wav_file.channels
            frame_count = wav_file.frames
            if frame_count <= 0:
                raise LocalAIRequestError("LocalAI Stable Audio 返回了空 WAV。")
            if sample_rate != SAMPLE_RATE or channels != CHANNELS:
                raise LocalAIRequestError(
                    "LocalAI Stable Audio WAV 格式与当前接口不同："
                    f"期望 {SAMPLE_RATE} Hz/{CHANNELS} 声道，得到 {sample_rate} Hz/{channels} 声道。"
                )
            decoded_frames = sum(
                len(block)
                for block in wav_file.blocks(blocksize=65_536, dtype="float32", always_2d=True)
            )
    except LocalAIRequestError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise LocalAIRequestError(f"LocalAI Stable Audio 返回的 WAV 无效：{exc}") from exc
    if decoded_frames != frame_count:
        raise LocalAIRequestError(
            f"LocalAI Stable Audio WAV 不完整：头部声明 {frame_count} 帧，实际解码 {decoded_frames} 帧。"
        )
    return sample_rate, channels


def generate_localai(request: StableAudio3MediumGenerateRequest) -> tuple[bytes, float]:
    """在共享 LocalAI GPU 租约内执行一次纯文本 Stable Audio 生成。"""
    audio_bytes, elapsed = execute_localai_operation(
        config=GATEWAY_CONFIG,
        operation=lambda client: client.generate_sound(
            text=request.prompt,
            duration_seconds=request.seconds or STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS,
            bpm=None,
            keyscale=None,
            timesignature=None,
        ),
    )
    validate_localai_wav(audio_bytes)
    return audio_bytes, elapsed


class StableAudio3MediumWorkerManager:
    """校验本地资源、启动一次 worker，并记录最近一次错误。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: StableAudio3MediumGenerateRequest) -> dict:
        """把 HTTP 请求与服务级默认配置合并成 worker JSON。"""
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "seed": request.seed,
            "device": request.device or STABLE_AUDIO_3_MEDIUM_DEVICE,
            "dtype": request.dtype or STABLE_AUDIO_3_MEDIUM_DTYPE,
            "model_path": str(STABLE_AUDIO_3_MEDIUM_MODEL_DIR),
            "upstream_path": str(STABLE_AUDIO_3_REPO_PATH),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": str(RUNTIME_CACHE_DIR),
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "require_flash_attn": STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN,
        }

    def run_worker(self, payload: dict) -> bytes:
        """确认本地权重和上游源码完整后执行一次隔离推理。"""
        if not local_model_is_complete(STABLE_AUDIO_3_MEDIUM_MODEL_DIR):
            missing = [
                name
                for name in REQUIRED_MODEL_FILES
                if not (STABLE_AUDIO_3_MEDIUM_MODEL_DIR / name).is_file()
            ]
            raise RuntimeError(
                "Stable Audio 3 Medium 本地权重不完整或不存在: "
                f"{STABLE_AUDIO_3_MEDIUM_MODEL_DIR}; 缺少: {', '.join(missing)}"
            )
        if not (STABLE_AUDIO_3_REPO_PATH / "stable_audio_3").is_dir():
            raise RuntimeError(
                f"Stable Audio 3 官方源码目录不存在或不完整: {STABLE_AUDIO_3_REPO_PATH}"
            )

        try:
            audio = run_local_worker(payload, STABLE_AUDIO_3_MEDIUM_WORKER)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


app = FastAPI(title="Unitale Stable Audio 3 Medium API")
app.add_middleware(ForceCORS)
manager = StableAudio3MediumWorkerManager()


@app.get("/v1/health")
def health() -> dict:
    """在不加载权重的前提下返回服务、模型文件和 GPU 就绪状态。"""
    localai_client = LocalAIClient(GATEWAY_CONFIG) if GATEWAY_CONFIG.provider == "localai" else None
    localai_healthy = bool(localai_client and localai_client.health())
    localai_model_registered = bool(
        localai_client and localai_healthy and localai_client.model_ready()
    )
    cuda = cuda_status()
    required_files = {
        name: (STABLE_AUDIO_3_MEDIUM_MODEL_DIR / name).is_file() for name in REQUIRED_MODEL_FILES
    }
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "model_dir": str(STABLE_AUDIO_3_MEDIUM_MODEL_DIR),
            "upstream_path": str(STABLE_AUDIO_3_REPO_PATH),
            "tts_output_dir": str(STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
            "project_dir": str(PROJECT_DIR),
            "worker_python": os.environ.get("STABLE_AUDIO_3_MEDIUM_PYTHON", os.sys.executable),
        },
        "available": {
            # 保留此字段是为了兼容健康检查响应格式；当前不再要求 Conda。
            "conda": bool(shutil.which("conda")),
            "uv": bool(shutil.which("uv")) or Path(os.sys.executable).is_file(),
            "model_dir": STABLE_AUDIO_3_MEDIUM_MODEL_DIR.is_dir(),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "upstream_source": (STABLE_AUDIO_3_REPO_PATH / "stable_audio_3").is_dir(),
            "worker_script": WORKER_SCRIPT.is_file(),
            "torch_in_api_env": module_available("torch"),
            "flash_attn_in_api_env": module_available("flash_attn"),
            "cuda": cuda["available"],
            "selected_provider": GATEWAY_CONFIG.provider,
            "selected_provider_ready": (
                localai_model_registered
                if localai_client
                else local_model_is_complete(STABLE_AUDIO_3_MEDIUM_MODEL_DIR)
            ),
            "localai_healthy": localai_healthy,
            "localai_model_registered": localai_model_registered,
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_env": "stable_audio_3_medium/.venv",
            "worker_runtime": "same uv project",
            "model": "stabilityai/stable-audio-3-medium",
            "model_lifecycle": "one request -> one uv worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
            "device": STABLE_AUDIO_3_MEDIUM_DEVICE,
            "dtype": STABLE_AUDIO_3_MEDIUM_DTYPE,
            "default_seconds": STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS,
            "default_steps": STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS,
            "default_cfg_scale": STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE,
            "default_seed": STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED,
            "sample_rate": 44100,
            "channels": 2,
            "max_seconds": MAX_SECONDS,
            "prompt_language": "English recommended; other languages may underperform",
            "hardware": "CUDA GPU with Ampere-or-newer compute capability",
            "flash_attention": flash_attention_status(),
            "contract": "supports music and sound effects; not designed for speech or voice generation",
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if localai_client else None,
        },
        "last_errors": {
            "stable_audio_3_medium": manager.last_error,
            "localai": LAST_LOCALAI_ERROR,
        },
    }


@app.post("/v1/stableAudio/soundEffect")
def generate(request: StableAudio3MediumGenerateRequest) -> Response:
    """串行持有共享 GPU 锁，生成 WAV 并写入 soundEffect 存储目录。"""
    global LAST_LOCALAI_ERROR
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Stable Audio 3 Medium request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if GATEWAY_CONFIG.provider == "localai":
        try:
            validate_localai_request(request)
            audio, _elapsed = generate_localai(request)
            saved_output_path = persist_audio_bytes(
                audio,
                "stable_audio_3_medium",
                STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR,
            )
            manager.last_error = None
            LAST_LOCALAI_ERROR = None
            print(f"[Stable Audio 3 Medium] 已保存 LocalAI 音频: {saved_output_path}")
            return Response(content=audio, media_type="audio/wav")
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
            LOGGER.exception("LocalAI Stable Audio request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock(GPU_LOCK_FILE, "stable_audio_3_medium/generate"):
            with manager.lock:
                try:
                    audio = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio,
                        "stable_audio_3_medium",
                        STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR,
                    )
                    print(f"[Stable Audio 3 Medium] 已保存生成音频: {saved_output_path}")
                    return Response(content=audio, media_type="audio/wav")
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("Stable Audio 3 Medium request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale Stable Audio 3 Medium uv API")
    print("==================================================")
    print(f"[配置] uv project: {PROJECT_DIR}")
    print(f"[配置] 模型目录: {STABLE_AUDIO_3_MEDIUM_MODEL_DIR}")
    print(f"[配置] 官方源码: {STABLE_AUDIO_3_REPO_PATH}")
    print(f"[配置] 输出目录: {STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        "[配置] "
        f"local_files_only={LOCAL_FILES_ONLY}, timeout={STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT}, "
        f"require_flash_attn={STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
