#!/usr/bin/env python3
"""独立 Seed-VC 音色转换 HTTP 服务。"""

from __future__ import annotations

# HTTP 进程仅校验文件、处理上传和编排 worker，不导入 Torch 或官方模型代码。
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    stage_audio_upload,
    storage_disk_status,
)
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from seed_vc_runtime import persist_audio_bytes, read_inference_config, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(path: str) -> Path:
    """展开环境变量和用户目录，返回绝对路径。"""
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def env_bool(name: str, default: bool) -> bool:
    """读取布尔环境配置。"""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", str(STORAGE_DIR / "clone")))
TIMBRE_STORAGE_DIR = expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre")))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", str(CLONE_STORAGE_DIR)))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(os.getenv("SEED_VC_MODEL_DIR", str(HF_MIRROR_DIR / "Plachta/Seed-VC")))
CODE_PATH = expand_path(os.getenv("SEED_VC_CODE_PATH", "~/tts-depency/seed-vc"))
WHISPER_MODEL_DIR = expand_path(
    os.getenv("SEED_VC_WHISPER_MODEL_DIR", str(HF_MIRROR_DIR / "openai/whisper-small"))
)
VOCODER_MODEL_DIR = expand_path(
    os.getenv(
        "SEED_VC_VOCODER_MODEL_DIR",
        str(HF_MIRROR_DIR / "netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x"),
    )
)
F0_VOCODER_MODEL_DIR = expand_path(
    os.getenv(
        "SEED_VC_F0_VOCODER_MODEL_DIR",
        str(HF_MIRROR_DIR / "nvidia/bigvgan_v2_44khz_128band_512x"),
    )
)
STYLE_ENCODER_CHECKPOINT = expand_path(
    os.getenv(
        "SEED_VC_STYLE_ENCODER_CHECKPOINT",
        str(HF_MIRROR_DIR / "netease-youdao/funasr/campplus/campplus_cn_common.bin"),
    )
)
RMVPE_CHECKPOINT = expand_path(
    os.getenv(
        "SEED_VC_RMVPE_CHECKPOINT", str(HF_MIRROR_DIR / "lj1995/VoiceConversionWebUI/rmvpe.pt")
    )
)
CHECKPOINT_PATH = expand_path(
    os.getenv(
        "SEED_VC_CHECKPOINT_PATH",
        str(MODEL_DIR / "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
    )
)
CONFIG_PATH = expand_path(
    os.getenv(
        "SEED_VC_CONFIG_PATH", str(MODEL_DIR / "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    )
)
F0_CHECKPOINT_PATH = expand_path(
    os.getenv(
        "SEED_VC_F0_CHECKPOINT_PATH",
        str(MODEL_DIR / "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
    )
)
F0_CONFIG_PATH = expand_path(
    os.getenv(
        "SEED_VC_F0_CONFIG_PATH",
        str(MODEL_DIR / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
    )
)
WORKER_TMP_DIR = expand_path(
    os.getenv("SEED_VC_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "seed_vc_worker"))
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
OUTPUT_DIR = expand_path(os.getenv("SEED_VC_OUTPUT_DIR", str(CLONE_STORAGE_DIR)))
API_HOST = os.getenv("SEED_VC_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("SEED_VC_PORT", os.getenv("PORT", "8381")))
REQUEST_TIMEOUT = float(os.getenv("SEED_VC_REQUEST_TIMEOUT", "900"))
DEVICE = os.getenv("SEED_VC_DEVICE", "cuda:0")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
if not all(
    math.isfinite(value) and value > 0 for value in (REQUEST_TIMEOUT, GPU_LOCK_WAIT_TIMEOUT)
):
    raise ValueError("Seed-VC 的请求超时与 GPU 队列等待超时必须为正数。")

DEFAULT_DIFFUSION_STEPS = int(os.getenv("SEED_VC_DIFFUSION_STEPS", "30"))
DEFAULT_LENGTH_ADJUST = float(os.getenv("SEED_VC_LENGTH_ADJUST", "1.0"))
DEFAULT_INFERENCE_CFG_RATE = float(os.getenv("SEED_VC_INFERENCE_CFG_RATE", "0.7"))
DEFAULT_F0_CONDITION = env_bool("SEED_VC_F0_CONDITION", False)
DEFAULT_AUTO_F0_ADJUST = env_bool("SEED_VC_AUTO_F0_ADJUST", False)
DEFAULT_SEMI_TONE_SHIFT = int(os.getenv("SEED_VC_SEMI_TONE_SHIFT", "0"))
DEFAULT_FP16 = env_bool("SEED_VC_FP16", True)

for name, path in {
    "HF_HOME": HF_MIRROR_DIR,
    "HF_MODULES_CACHE": RUNTIME_CACHE_DIR / "hf_modules",
    "NUMBA_CACHE_DIR": RUNTIME_CACHE_DIR / "numba",
    "MPLCONFIGDIR": RUNTIME_CACHE_DIR / "matplotlib",
    "XDG_CACHE_HOME": RUNTIME_CACHE_DIR / "xdg",
}.items():
    os.environ.setdefault(name, str(path))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
for directory in (
    PROMPTS_DIR,
    TIMBRE_STORAGE_DIR,
    WORKER_TMP_DIR,
    OUTPUT_DIR,
    RUNTIME_CACHE_DIR / "uploads",
):
    directory.mkdir(parents=True, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title="Unitale Seed-VC API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SeedVcRequest(BaseModel):
    """原始语音与参考音色的逻辑路径及官方推理参数。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)

    source_audio_path: str = Field(min_length=1, max_length=1_024)
    reference_audio_path: str = Field(min_length=1, max_length=1_024)
    diffusion_steps: int = Field(default=DEFAULT_DIFFUSION_STEPS, ge=1, le=200)
    length_adjust: float = Field(default=DEFAULT_LENGTH_ADJUST, ge=0.5, le=2)
    inference_cfg_rate: float = Field(default=DEFAULT_INFERENCE_CFG_RATE, ge=0, le=1)
    f0_condition: bool = DEFAULT_F0_CONDITION
    auto_f0_adjust: bool = DEFAULT_AUTO_F0_ADJUST
    semi_tone_shift: int = Field(default=DEFAULT_SEMI_TONE_SHIFT, ge=-24, le=24)
    fp16: bool = DEFAULT_FP16

    @model_validator(mode="after")
    def validate_pitch_controls(self) -> SeedVcRequest:
        """防止在未加载 F0 模型时静默忽略用户的音高控制。"""
        if not self.f0_condition and (self.auto_f0_adjust or self.semi_tone_shift):
            raise ValueError("auto_f0_adjust 和 semi_tone_shift 需要 f0_condition=true。")
        return self


def required_files(request: SeedVcRequest) -> dict[str, Path]:
    """列出当前模式必需的本地模型和官方源码文件。"""
    vocoder = F0_VOCODER_MODEL_DIR if request.f0_condition else VOCODER_MODEL_DIR
    paths = {
        "checkpoint_path": F0_CHECKPOINT_PATH if request.f0_condition else CHECKPOINT_PATH,
        "config_path": F0_CONFIG_PATH if request.f0_condition else CONFIG_PATH,
        "style_encoder_checkpoint": STYLE_ENCODER_CHECKPOINT,
        "whisper_config": WHISPER_MODEL_DIR / "config.json",
        "whisper_preprocessor": WHISPER_MODEL_DIR / "preprocessor_config.json",
        "vocoder_config": vocoder / "config.json",
        "vocoder_checkpoint": vocoder / "bigvgan_generator.pt",
        "inference_code": CODE_PATH / "inference.py",
        "commons_code": CODE_PATH / "modules/commons.py",
        "campplus_code": CODE_PATH / "modules/campplus/DTDNN.py",
        "vocoder_code": CODE_PATH / "modules/bigvgan/bigvgan.py",
        "hf_utils_code": CODE_PATH / "hf_utils.py",
        "worker_script": Path(WORKER_SCRIPT),
    }
    if request.f0_condition:
        paths["rmvpe_checkpoint"] = RMVPE_CHECKPOINT
        paths["rmvpe_code"] = CODE_PATH / "modules/rmvpe.py"
    return paths


def whisper_weights_ready() -> bool:
    """检查 Whisper 本地权重，不将目录存在误判为模型就绪。"""
    return any(
        (WHISPER_MODEL_DIR / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def module_available(name: str) -> bool:
    """只检查安装信息，不触发模型依赖导入。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def gpu_runtime_lock(label: str):
    """使用有限等待的共享 GPU 队列并记录请求指标。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """保持 GPU 锁直到 worker 显存释放等待结束。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def store_uploaded_audio(staged: StagedUpload, full_path: str, prompt_text: str | None):
    """在线程池中原子提交音频，复用内容寻址和设计音色引用。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


class SeedVcWorkerManager:
    """完成 CPU 文件预检并管理每次请求的单个隔离 worker。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None
        self.last_metrics: dict[str, Any] | None = None

    def build_worker_payload(self, request: SeedVcRequest) -> dict[str, Any]:
        """在取得 GPU 锁前解析两路音频并检查所有本地依赖。"""
        payload = request.model_dump()
        for field in ("source_audio_path", "reference_audio_path"):
            logical_path = payload[field]
            audio_path = reference_store.prompt_audio_path(logical_path)
            if not audio_path.is_file():
                raise HTTPException(status_code=404, detail=f"音频不存在: {logical_path}")
            payload[field] = str(audio_path)
        missing = [str(path) for path in required_files(request).values() if not path.is_file()]
        if not whisper_weights_ready():
            missing.append(f"{WHISPER_MODEL_DIR}/model.safetensors 或 pytorch_model.bin")
        if missing:
            raise FileNotFoundError("Seed-VC 本地文件不完整: " + ", ".join(missing))
        read_inference_config(
            F0_CONFIG_PATH if request.f0_condition else CONFIG_PATH, request.f0_condition
        )
        payload.update(
            code_path=str(CODE_PATH),
            checkpoint_path=str(F0_CHECKPOINT_PATH if request.f0_condition else CHECKPOINT_PATH),
            config_path=str(F0_CONFIG_PATH if request.f0_condition else CONFIG_PATH),
            whisper_model_dir=str(WHISPER_MODEL_DIR),
            vocoder_model_dir=str(
                F0_VOCODER_MODEL_DIR if request.f0_condition else VOCODER_MODEL_DIR
            ),
            style_encoder_checkpoint=str(STYLE_ENCODER_CHECKPOINT),
            rmvpe_checkpoint=str(RMVPE_CHECKPOINT),
            device=DEVICE,
            hf_mirror_dir=str(HF_MIRROR_DIR),
            local_files_only=LOCAL_FILES_ONLY,
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """使用当前服务的 uv 解释器执行一次转换，并清理所有临时文件。"""
        process = None
        with tempfile.TemporaryDirectory(dir=WORKER_TMP_DIR, prefix="seed_vc_") as job_dir:
            request_path = Path(job_dir) / "request.json"
            output_path = Path(job_dir) / "output.wav"
            try:
                with request_path.open("w", encoding="utf-8") as destination:
                    json.dump(payload, destination, ensure_ascii=False)
                worker_env = os.environ.copy()
                if LOCAL_FILES_ONLY:
                    worker_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        WORKER_SCRIPT,
                        "--input-json",
                        str(request_path),
                        "--output-wav",
                        str(output_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                    cwd=job_dir,
                    env=worker_env,
                )
                try:
                    stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"Seed-VC worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc
                if stdout.strip():
                    LOGGER.info("Seed-VC worker: %s", stdout.rstrip())
                if process.returncode != 0:
                    lines = [
                        line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                    ]
                    raise RuntimeError(" | ".join(lines[-8:]) or "Seed-VC worker 执行失败。")
                audio_bytes = output_path.read_bytes()
                if not audio_bytes.startswith(b"RIFF") or audio_bytes[8:12] != b"WAVE":
                    raise RuntimeError("Seed-VC worker 未生成有效 WAV。")
                self.last_error = None
                return audio_bytes
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                terminate_process_group(process)


manager = SeedVcWorkerManager()


@app.get("/v1/health")
def health():
    """报告两种模型模式的文件就绪状态和存储容量，不加载模型。"""
    availability = {}
    for mode, f0 in (("voice_conversion", False), ("f0_conversion", True)):
        request = SeedVcRequest(
            source_audio_path="health.wav",
            reference_audio_path="health.wav",
            f0_condition=f0,
            auto_f0_adjust=False,
            semi_tone_shift=0,
        )
        files = required_files(request)
        missing = [str(path) for path in files.values() if not path.is_file()]
        if not whisper_weights_ready():
            missing.append(str(WHISPER_MODEL_DIR / "model.safetensors"))
        availability[mode] = {"ready": not missing, "missing_files": missing}
    return {
        "code": 200,
        "paths": {
            "model_dir": str(MODEL_DIR),
            "code_path": str(CODE_PATH),
            "prompts_dir": str(PROMPTS_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
        },
        "available": {
            **availability,
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
        },
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "device": DEVICE,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "last_metrics": manager.last_metrics,
        },
        "storage": storage_disk_status(STORAGE_DIR),
        "last_errors": {"seed_vc": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...), full_path: str = Form(...), prompt_text: str | None = Form(None)
):
    """流式暂存原始音频或参考音色，在线程池中原子提交。"""
    try:
        staged = await stage_audio_upload(audio, RUNTIME_CACHE_DIR / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查上传音频或设计音色引用。"""
    exists = reference_store.prompt_audio_path(file_name).is_file()
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(reference_store.load_prompt_text(file_name)),
    }


@app.post("/v1/seedVc/voiceConversion")
def voice_conversion(request: SeedVcRequest):
    """串行执行一次 Seed-VC 转换，原子保存结果并返回 WAV。"""
    try:
        payload = manager.build_worker_payload(request)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    metrics = None
    try:
        with gpu_runtime_lock("seed_vc/voiceConversion") as metrics, manager.lock:
            try:
                audio_bytes = manager.run_worker(payload)
                manager.last_output_path = str(persist_audio_bytes(audio_bytes, OUTPUT_DIR))
                return Response(content=audio_bytes, media_type="audio/wav")
            finally:
                wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        manager.last_error = str(exc)
        LOGGER.exception("Seed-VC 转换失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if metrics is not None:
            manager.last_metrics = metrics.as_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
