"""dots.tts-soar 一次性语音克隆的 HTTP 封装。"""

from __future__ import annotations

# 学习入口：dots 服务将每个请求序列化给当前 uv 环境的 worker，父进程不加载重型依赖。
import importlib.util
import logging
import os
import re
import sys
import threading
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    stage_audio_upload,
)
from unitale_runtime import (
    gpu_runtime_lock as shared_gpu_runtime_lock,
)
from unitale_runtime.storage import sha256_file

from runtime import UvWorkerConfig, cuda_status, persist_audio_bytes, run_uv_worker
from synthesis_request import CloneSynthesisRequest

LOGGER = logging.getLogger(__name__)

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动配置中的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: str | None) -> str | None:
    """把可选文本清理成字符串或 ``None``。"""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def normalize_synthesis_text(text: str) -> str:
    """去除 Markdown 标题标记，避免额外格式被模型朗读。"""
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", os.path.join(PROJECT_DIR, "storage")))
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", os.path.join(STORAGE_DIR, "clone")))
TIMBRE_STORAGE_DIR = expand_path(
    os.getenv("TIMBRE_STORAGE_DIR", os.path.join(STORAGE_DIR, "timbre"))
)
TIMBRE_REFERENCE_DIR = os.path.join(TIMBRE_STORAGE_DIR, ".references")
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", os.path.join(STORAGE_DIR, ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8324"))

DOTS_TTS_SOAR_MODEL_DIR = expand_path(
    os.getenv("DOTS_TTS_SOAR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "rednote-hilab/dots.tts-soar"))
)
# ============================================================================
# dots.tts-soar 克隆调试默认值
#
# 调试参考音频克隆效果时，优先直接修改下面带 _DEFAULT 后缀的值，然后
# 重启服务即可生效。环境变量仍可覆盖默认值，方便部署时统一配置。
# ============================================================================
# 推理精度：bfloat16 通常更省显存；遇到硬件不兼容时可改为 float16 或 float32。
DOTS_TTS_SOAR_PRECISION_DEFAULT = "bfloat16"
# 语言标识：官方示例使用 chinese；多语言场景可按模型支持范围调整。
DOTS_TTS_SOAR_LANGUAGE_DEFAULT = "chinese"
# ODE 求解器：euler 速度较快；具体可选值取决于当前 dots.tts-soar 版本。
DOTS_TTS_SOAR_ODE_METHOD_DEFAULT = "euler"
# 推理步数：步数越高通常越稳定，但会增加生成耗时。
DOTS_TTS_SOAR_NUM_STEPS_DEFAULT = 10
# 文本引导强度：提高可强化内容约束，过高可能影响自然度。
DOTS_TTS_SOAR_GUIDANCE_SCALE_DEFAULT = 1.2
# 说话人条件强度：提高可强化参考音色，过高可能带来音色或韵律失真。
DOTS_TTS_SOAR_SPEAKER_SCALE_DEFAULT = 1.5
# 单段最大生成长度：用于限制模型一次生成的 token/帧长度，具体含义由官方 runtime 决定。
DOTS_TTS_SOAR_MAX_GENERATE_LENGTH_DEFAULT = 500
# 文本分片字符数：0 表示不分片；分片可以降低显存压力，但会插入停顿。
DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK_DEFAULT = 120
# 分片之间的停顿时长，单位为毫秒；仅在发生分片时生效。
DOTS_TTS_SOAR_PAUSE_MS_DEFAULT = 250
# 随机种子：固定整数便于复现；如需随机行为请按当前官方 runtime 支持范围调整。
DOTS_TTS_SOAR_SEED_DEFAULT = 42
# 是否让官方 runtime 规范化输入文本；保持 False 可最大程度保留原始文本。
DOTS_TTS_SOAR_NORMALIZE_TEXT_DEFAULT = False
# 是否开启推理性能分析；仅调试性能时开启，避免额外开销。
DOTS_TTS_SOAR_PROFILE_INFERENCE_DEFAULT = False
# 单次请求超时时间，单位为秒；包含 worker 启动和完整合成。
DOTS_TTS_SOAR_REQUEST_TIMEOUT_DEFAULT = 900.0

DOTS_TTS_SOAR_PRECISION = os.getenv("DOTS_TTS_SOAR_PRECISION", DOTS_TTS_SOAR_PRECISION_DEFAULT)
DOTS_TTS_SOAR_LANGUAGE = normalize_optional_text(
    os.getenv("DOTS_TTS_SOAR_LANGUAGE", DOTS_TTS_SOAR_LANGUAGE_DEFAULT)
)
DOTS_TTS_SOAR_ODE_METHOD = os.getenv("DOTS_TTS_SOAR_ODE_METHOD", DOTS_TTS_SOAR_ODE_METHOD_DEFAULT)
DOTS_TTS_SOAR_NUM_STEPS = int(
    os.getenv("DOTS_TTS_SOAR_NUM_STEPS", str(DOTS_TTS_SOAR_NUM_STEPS_DEFAULT))
)
DOTS_TTS_SOAR_GUIDANCE_SCALE = float(
    os.getenv("DOTS_TTS_SOAR_GUIDANCE_SCALE", str(DOTS_TTS_SOAR_GUIDANCE_SCALE_DEFAULT))
)
DOTS_TTS_SOAR_SPEAKER_SCALE = float(
    os.getenv("DOTS_TTS_SOAR_SPEAKER_SCALE", str(DOTS_TTS_SOAR_SPEAKER_SCALE_DEFAULT))
)
DOTS_TTS_SOAR_MAX_GENERATE_LENGTH = int(
    os.getenv(
        "DOTS_TTS_SOAR_MAX_GENERATE_LENGTH",
        str(DOTS_TTS_SOAR_MAX_GENERATE_LENGTH_DEFAULT),
    )
)
DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK = int(
    os.getenv(
        "DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK",
        str(DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK_DEFAULT),
    )
)
DOTS_TTS_SOAR_PAUSE_MS = int(
    os.getenv("DOTS_TTS_SOAR_PAUSE_MS", str(DOTS_TTS_SOAR_PAUSE_MS_DEFAULT))
)
DOTS_TTS_SOAR_SEED = int(os.getenv("DOTS_TTS_SOAR_SEED", str(DOTS_TTS_SOAR_SEED_DEFAULT)))
DOTS_TTS_SOAR_NORMALIZE_TEXT = env_bool(
    "DOTS_TTS_SOAR_NORMALIZE_TEXT", DOTS_TTS_SOAR_NORMALIZE_TEXT_DEFAULT
)
DOTS_TTS_SOAR_PROFILE_INFERENCE = env_bool(
    "DOTS_TTS_SOAR_PROFILE_INFERENCE", DOTS_TTS_SOAR_PROFILE_INFERENCE_DEFAULT
)
DOTS_TTS_SOAR_REQUEST_TIMEOUT = float(
    os.getenv("DOTS_TTS_SOAR_REQUEST_TIMEOUT", str(DOTS_TTS_SOAR_REQUEST_TIMEOUT_DEFAULT))
)
DOTS_TTS_SOAR_WORKER_SCRIPT = os.path.join(SERVICE_DIR, "worker.py")
DOTS_TTS_SOAR_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "dots_tts_soar_worker")
DOTS_TTS_SOAR_OUTPUT_DIR = expand_path(
    os.getenv(
        "DOTS_TTS_SOAR_OUTPUT_DIR",
        CLONE_STORAGE_DIR,
    )
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(TIMBRE_STORAGE_DIR, exist_ok=True)
os.makedirs(TIMBRE_REFERENCE_DIR, exist_ok=True)
os.makedirs(DOTS_TTS_SOAR_WORKER_TMP_DIR, exist_ok=True)
os.makedirs(DOTS_TTS_SOAR_OUTPUT_DIR, exist_ok=True)
for cache_key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    os.makedirs(os.environ[cache_key], exist_ok=True)
lock_dir = os.path.dirname(GPU_LOCK_FILE)
if lock_dir:
    os.makedirs(lock_dir, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)

app = FastAPI(title="Unitale dots.tts-soar Voice Clone API")


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


app.add_middleware(ForceCORS)


def hash_filename(filename: str) -> str:
    return reference_store.clone_path(filename).name


def clone_prompt_audio_path(filename: str) -> str:
    return str(reference_store.clone_path(filename))


def timbre_reference_map_path(filename: str) -> str:
    return str(reference_store.reference_path(filename))


def prompt_audio_path(filename: str) -> str:
    """解析克隆上传，或解析只保存在音色目录中的设计音频。"""
    return str(reference_store.prompt_audio_path(filename))


def prompt_text_sidecar_path(filename: str) -> str:
    return str(reference_store.prompt_sidecar_path(filename))


def load_prompt_text_sidecar(filename: str) -> str | None:
    return reference_store.load_prompt_text(filename)


def store_uploaded_audio(
    staged: StagedUpload,
    full_path: str,
    prompt_text: str | None,
) -> dict[str, object]:
    """原子提交流式上传，避免并发请求互相覆盖。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


def gpu_runtime_lock(label: str):
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str) -> None:
    if CUDA_RELEASE_DELAY > 0:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
        import time

        time.sleep(CUDA_RELEASE_DELAY)


class DotsTtsSoarSynthesizeRequest(CloneSynthesisRequest):
    """dots.tts-soar 官方克隆参数及本项目兼容字段。"""

    text: str = Field(min_length=1, max_length=12_000)
    audio_path: str = Field(min_length=1, max_length=1_024)
    backend: Literal["dots-tts-soar"] | None = None
    prompt_text: str | None = Field(default=None, max_length=12_000)
    language: str | None = Field(default=None, max_length=32)
    template_name: Literal["tts", "instruction_tts", "text_to_audio", "tts_interleave"] | None = (
        None
    )
    precision: str | None = Field(default=None, max_length=32)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    ode_method: str | None = Field(default=None, max_length=32)
    num_steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    speaker_scale: float | None = Field(default=None, ge=0, le=20)
    max_generate_length: int | None = Field(default=None, ge=1, le=16_384)
    max_chars_per_chunk: int | None = Field(default=None, ge=0, le=2_000)
    pause_ms: int | None = Field(default=None, ge=0, le=10_000)
    normalize_text: bool | None = None
    profile_inference: bool | None = None


DOTS_TTS_SOAR_WORKER = UvWorkerConfig(
    python_executable=sys.executable,
    worker_script=DOTS_TTS_SOAR_WORKER_SCRIPT,
    model_dir=DOTS_TTS_SOAR_MODEL_DIR,
    temp_dir=DOTS_TTS_SOAR_WORKER_TMP_DIR,
    timeout=DOTS_TTS_SOAR_REQUEST_TIMEOUT,
    label="dots.tts-soar",
    file_prefix="dots_tts_soar",
)


class DotsTtsSoarWorkerManager:
    """管理 dots 参考音频解析、worker JSON 和进程清理。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: DotsTtsSoarSynthesizeRequest) -> dict:
        """把 HTTP 请求转换为 dots 一次性 worker 使用的参数字典。"""
        ref_audio_path = prompt_audio_path(request.audio_path)
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = normalize_optional_text(request.prompt_text)
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)

        return {
            "operation": "clone",
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": DOTS_TTS_SOAR_MODEL_DIR,
            "language": (
                normalize_optional_text(request.language)
                if request.language is not None
                else DOTS_TTS_SOAR_LANGUAGE
            ),
            "template_name": request.template_name,
            "precision": request.precision or DOTS_TTS_SOAR_PRECISION,
            "seed": request.seed if request.seed is not None else DOTS_TTS_SOAR_SEED,
            "ode_method": request.ode_method or DOTS_TTS_SOAR_ODE_METHOD,
            "num_steps": request.num_steps
            if request.num_steps is not None
            else DOTS_TTS_SOAR_NUM_STEPS,
            "guidance_scale": (
                request.guidance_scale
                if request.guidance_scale is not None
                else DOTS_TTS_SOAR_GUIDANCE_SCALE
            ),
            "speaker_scale": (
                request.speaker_scale
                if request.speaker_scale is not None
                else DOTS_TTS_SOAR_SPEAKER_SCALE
            ),
            "max_generate_length": (
                request.max_generate_length
                if request.max_generate_length is not None
                else DOTS_TTS_SOAR_MAX_GENERATE_LENGTH
            ),
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms
            if request.pause_ms is not None
            else DOTS_TTS_SOAR_PAUSE_MS,
            "normalize_text": (
                request.normalize_text
                if request.normalize_text is not None
                else DOTS_TTS_SOAR_NORMALIZE_TEXT
            ),
            "profile_inference": (
                request.profile_inference
                if request.profile_inference is not None
                else DOTS_TTS_SOAR_PROFILE_INFERENCE
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        """调用当前 uv 环境的 dots worker，并验证返回 WAV。"""
        try:
            audio = run_uv_worker(payload, DOTS_TTS_SOAR_WORKER)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = DotsTtsSoarWorkerManager()


@app.get("/v1/health")
def health():
    """返回 dots 模型、源码、worker 和 GPU 的就绪状态。"""
    cuda = cuda_status()
    required_files = {
        name: os.path.isfile(os.path.join(DOTS_TTS_SOAR_MODEL_DIR, name))
        for name in (
            "config.json",
            "model.safetensors",
            "speaker_encoder.safetensors",
            "vocoder.safetensors",
        )
    }
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "model_dir": DOTS_TTS_SOAR_MODEL_DIR,
            "prompts_dir": PROMPTS_DIR,
            "tts_output_dir": DOTS_TTS_SOAR_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": DOTS_TTS_SOAR_WORKER_SCRIPT,
            "worker_tmp_dir": DOTS_TTS_SOAR_WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_runtime": "uv",
            "worker_script": os.path.isfile(DOTS_TTS_SOAR_WORKER_SCRIPT),
            "model_dir": os.path.isdir(DOTS_TTS_SOAR_MODEL_DIR),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "torch": module_available("torch"),
            "cuda": cuda["available"],
            "flash_attn": module_available("flash_attn"),
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model": "rednote-hilab/dots.tts-soar",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": DOTS_TTS_SOAR_REQUEST_TIMEOUT,
            "sample_rate": 48000,
            "precision": DOTS_TTS_SOAR_PRECISION,
            "language": DOTS_TTS_SOAR_LANGUAGE,
            "ode_method": DOTS_TTS_SOAR_ODE_METHOD,
            "seed": DOTS_TTS_SOAR_SEED,
            "num_steps": DOTS_TTS_SOAR_NUM_STEPS,
            "guidance_scale": DOTS_TTS_SOAR_GUIDANCE_SCALE,
            "speaker_scale": DOTS_TTS_SOAR_SPEAKER_SCALE,
            "max_generate_length": DOTS_TTS_SOAR_MAX_GENERATE_LENGTH,
            "max_chars_per_chunk": DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK,
            "pause_ms": DOTS_TTS_SOAR_PAUSE_MS,
            "normalize_text": DOTS_TTS_SOAR_NORMALIZE_TEXT,
            "profile_inference": DOTS_TTS_SOAR_PROFILE_INFERENCE,
            "flash_attention_policy": (
                "not required; dots.tts uses native PyTorch flex_attention and does not import flash_attn"
            ),
            "clone_contract": "prompt_audio + optional exact prompt_text; 48 kHz mono output",
        },
        "last_errors": {"dots_tts_soar": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    """在线程池中保存参考音频和可选的文本 sidecar。"""
    try:
        staged = await stage_audio_upload(audio, os.path.join(RUNTIME_CACHE_DIR, "uploads"))
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查逻辑参考路径是否对应有效的本地音频。"""
    audio_path = prompt_audio_path(file_name)
    exists = os.path.isfile(audio_path)
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "size_bytes": os.path.getsize(audio_path) if exists else None,
        "sha256": sha256_file(audio_path) if exists else None,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v1/dotsTTS/clone")
def synthesize(request: DotsTtsSoarSynthesizeRequest):
    """串行执行 dots.tts-soar 克隆并返回生成 WAV。"""
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("dots.tts-soar request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("dots_tts_soar/synthesize"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "dots_tts_soar",
                        DOTS_TTS_SOAR_OUTPUT_DIR,
                    )
                    print(f"[dots.tts-soar] 已保存生成音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("dots.tts-soar request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after dots.tts-soar worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 dots.tts-soar Voice Clone")
    print("==================================================")
    print("[配置] worker runtime: uv")
    print(f"[配置] worker python: {sys.executable}")
    print(f"[配置] model: {DOTS_TTS_SOAR_MODEL_DIR}")
    print(f"[配置] port: {API_PORT}")
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={DOTS_TTS_SOAR_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
