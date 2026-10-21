#!/usr/bin/env python3
"""FireRedTTS3 的独立 HTTP 服务。

8304 以 ``timbre`` 模式提供 Instruct Voice Design，8325 以 ``clone`` 模式提供
Base zero-shot cloning。HTTP 父进程不导入 FireRed、Torch 或 Transformers；每次请求
只在独立 worker 中加载一个变体，避免 Base 与 Instruct 同时占用显存。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    UnsupportedCapabilityError,
    load_gateway_config,
)
from inference_gateway.execution import execute_localai_operation
from pydantic import BaseModel, ConfigDict, Field, model_validator
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

from audio_output import persist_audio_bytes
from runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    """解析环境变量中的布尔值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str | os.PathLike[str]) -> str:
    """展开用户目录和环境变量，统一返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def module_available(module_name: str) -> bool:
    """只检查依赖是否可发现，不在 HTTP 父进程导入重型模型。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def normalize_optional_text(value: str | None) -> str | None:
    """把可选文本清理为空值或去除首尾空白的字符串。"""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


DEFAULT_PORT = int(os.getenv("FIRERED_TTS3_PORT", os.getenv("PORT", "8304")))
SERVICE_MODE = (
    os.getenv(
        "FIRERED_TTS3_MODE",
        "timbre" if DEFAULT_PORT == 8304 else "clone",
    )
    .strip()
    .lower()
)
if SERVICE_MODE not in {"clone", "timbre"}:
    raise ValueError("FIRERED_TTS3_MODE 必须是 clone 或 timbre。")

STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
CLONE_STORAGE_DIR = Path(expand_path(os.getenv("CLONE_STORAGE_DIR", str(STORAGE_DIR / "clone"))))
PROMPTS_DIR = Path(expand_path(os.getenv("PROMPTS_DIR", str(CLONE_STORAGE_DIR))))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", str(Path(RUNTIME_CACHE_DIR) / "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "FIRERED_TTS3_WORKER_TMP_DIR",
        str(Path(RUNTIME_CACHE_DIR) / f"firered_tts3_{SERVICE_MODE}_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv(
        "FIRERED_TTS3_MODEL_DIR",
        str(Path(HF_MIRROR_DIR) / "drbaph/FireRedTTS3-bf16"),
    )
)
CODE_PATH = expand_path(os.getenv("FIRERED_TTS3_CODE_PATH", "~/tts-depency/FireRedTTS3"))
OUTPUT_DIR = expand_path(
    os.getenv(
        "FIRERED_TTS3_OUTPUT_DIR",
        str(TIMBRE_STORAGE_DIR if SERVICE_MODE == "timbre" else CLONE_STORAGE_DIR),
    )
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("FIRERED_TTS3_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("FIRERED_TTS3_PORT", os.getenv("PORT", str(DEFAULT_PORT))))
REQUEST_TIMEOUT = float(os.getenv("FIRERED_TTS3_REQUEST_TIMEOUT", "900"))
LANGUAGE_DEFAULT = os.getenv("FIRERED_TTS3_LANGUAGE", "Chinese")
USE_FASTTEXT = env_bool("FIRERED_TTS3_USE_FASTTEXT", False)
USE_WETEXT = env_bool("FIRERED_TTS3_USE_WETEXT", True)
DO_TN = env_bool("FIRERED_TTS3_DO_TN", True)
N_TIMESTEPS_DEFAULT = int(os.getenv("FIRERED_TTS3_N_TIMESTEPS", "10"))
CLONE_INFERENCE_CFG_DEFAULT = float(os.getenv("FIRERED_TTS3_CLONE_INFERENCE_CFG", "2.0"))
TIMBRE_INFERENCE_CFG_DEFAULT = float(os.getenv("FIRERED_TTS3_TIMBRE_INFERENCE_CFG", "1.2"))
CLONE_SEED_DEFAULT = int(os.getenv("FIRERED_TTS3_CLONE_SEED", "1234"))
TIMBRE_SEED_DEFAULT = int(os.getenv("FIRERED_TTS3_TIMBRE_SEED", "2"))
STOP_THRESHOLD_DEFAULT = float(os.getenv("FIRERED_TTS3_STOP_THRESHOLD", "0.5"))
FIRERED_TTS3_ATTN_IMPLEMENTATION = (
    os.getenv("FIRERED_TTS3_ATTN_IMPLEMENTATION", "auto").strip().lower()
)
if FIRERED_TTS3_ATTN_IMPLEMENTATION not in {"auto", "flash_attention_2", "sdpa", "eager"}:
    raise ValueError(
        "FIRERED_TTS3_ATTN_IMPLEMENTATION 必须是 auto、flash_attention_2、sdpa 或 eager。"
    )
LOCALAI_PROVIDER_ENV = f"UNITALE_FIRERED_TTS3_{SERVICE_MODE.upper()}_PROVIDER"
LOCALAI_MODEL_ID = (
    "unitale.fireredtts3.instruct.q8-0"
    if SERVICE_MODE == "timbre"
    else "unitale.fireredtts3.base.q8-0"
)
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env=LOCALAI_PROVIDER_ENV,
    default_model=LOCALAI_MODEL_ID,
    request_timeout=REQUEST_TIMEOUT,
)
LAST_LOCALAI_ERROR: str | None = None

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(Path(RUNTIME_CACHE_DIR) / "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(RUNTIME_CACHE_DIR) / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(RUNTIME_CACHE_DIR) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(RUNTIME_CACHE_DIR) / "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (
    TIMBRE_STORAGE_DIR,
    CLONE_STORAGE_DIR,
    PROMPTS_DIR,
    OUTPUT_DIR,
    WORKER_TMP_DIR,
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.path.dirname(GPU_LOCK_FILE),
):
    if directory:
        os.makedirs(directory, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title=f"Unitale FireRedTTS3 {SERVICE_MODE} API")


class ForceCORS(BaseHTTPMiddleware):
    """为本地 WebUI 请求提供跨域响应和预检处理。"""

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


class CloneRequest(BaseModel):
    """FireRedTTS3 Base 克隆请求。"""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value):
        if isinstance(value, dict) and "style_prompt" in value:
            raise ValueError("style_prompt 不适用于 FireRedTTS3 克隆接口。")
        return value

    text: str = Field(min_length=1, max_length=12_000)
    audio_path: str = Field(min_length=1, max_length=1_024)
    backend: Literal["fireredtts3", "firered-tts3"] | None = None
    prompt_text: str | None = Field(default=None, max_length=12_000)
    language: str | None = Field(default=None, max_length=64)
    n_timesteps: int | None = Field(default=None, ge=1, le=100)
    inference_cfg: float | None = Field(default=None, ge=0, le=20)
    stop_threshold: float | None = Field(default=None, ge=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)


class TimbreRequest(BaseModel):
    """FireRedTTS3 Instruct 音色设计请求。"""

    model_config = ConfigDict(extra="forbid")

    voice_description: str = Field(min_length=1, max_length=2_000)
    text: str = Field(default="这是生成的参考音频预览。", min_length=1, max_length=12_000)
    language: str | None = Field(default=None, max_length=64)
    n_timesteps: int | None = Field(default=None, ge=1, le=100)
    inference_cfg: float | None = Field(default=None, ge=0, le=20)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    stop_threshold: float | None = Field(default=None, ge=0, le=1)


def gpu_runtime_lock(label: str):
    """使用共享 GPU 文件锁串行化所有本地模型 worker。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 回收，再释放共享锁。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def normalize_synthesis_text(text: str) -> str:
    """清理 WebUI 可能携带的 Markdown 标记，避免模型误读标题。"""
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


def required_model_files(model_dir: str | os.PathLike[str]) -> tuple[str, ...]:
    """返回 FireRedTTS3 权重镜像必须包含的核心文件。"""
    return (
        "fireredtts3_base/config.json",
        "fireredtts3_base/model.safetensors",
        "fireredtts3_instruct/config.json",
        "fireredtts3_instruct/model.safetensors",
        "redae/config.json",
        "redae/model.safetensors",
        "campp/campplus_voxceleb.bin",
        "text_tokenizer/tokenizer.json",
        "text_tokenizer/tokenizer_config.json",
        "text_tokenizer/vocab.json",
    )


def model_is_ready() -> bool:
    """检查权重结构，不加载模型。"""
    model_path = Path(MODEL_DIR)
    return model_path.is_dir() and all(
        (model_path / name).is_file() for name in required_model_files(MODEL_DIR)
    )


def hash_filename(filename: str) -> str:
    """将 WebUI 逻辑路径映射为安全的稳定本地文件名。"""
    return reference_store.clone_path(filename).name


def prompt_audio_path(filename: str) -> str:
    """解析普通克隆上传或音色目录中的设计音频引用。"""
    return str(reference_store.prompt_audio_path(filename))


def load_prompt_text_sidecar(filename: str) -> str | None:
    """读取普通上传或音色引用中的参考文本。"""
    return reference_store.load_prompt_text(filename)


def store_uploaded_audio(
    staged: StagedUpload,
    full_path: str,
    prompt_text: str | None,
) -> dict[str, object]:
    """原子提交流式上传，并保存 FireRed 克隆所需的参考文本 sidecar。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


class FireRedWorkerManager:
    """构造一次 FireRed 请求、启动 worker 并清理临时文件。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_payload(self, request: CloneRequest | TimbreRequest) -> dict[str, object]:
        """把 HTTP 请求合并为 worker 可直接消费的 JSON。"""
        payload = request.model_dump()
        payload.update(
            {
                "operation": SERVICE_MODE,
                "model_path": MODEL_DIR,
                "code_path": CODE_PATH,
                "local_files_only": LOCAL_FILES_ONLY,
                "language": request.language or LANGUAGE_DEFAULT,
                "use_fasttext": USE_FASTTEXT,
                "use_wetext": USE_WETEXT,
                "do_tn": DO_TN,
                "n_timesteps": request.n_timesteps or N_TIMESTEPS_DEFAULT,
                "stop_threshold": request.stop_threshold
                if request.stop_threshold is not None
                else STOP_THRESHOLD_DEFAULT,
                "runtime_cache_dir": RUNTIME_CACHE_DIR,
                "attn_implementation": FIRERED_TTS3_ATTN_IMPLEMENTATION,
            }
        )
        if isinstance(request, CloneRequest):
            reference_audio = prompt_audio_path(request.audio_path)
            if not os.path.isfile(reference_audio):
                raise HTTPException(status_code=404, detail="参考音频不存在。")
            prompt_text = normalize_optional_text(request.prompt_text) or load_prompt_text_sidecar(
                request.audio_path
            )
            if not prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="FireRedTTS3 克隆需要参考音频对应的准确 prompt_text。",
                )
            payload.update(
                {
                    "text": normalize_synthesis_text(request.text),
                    "ref_audio_path": reference_audio,
                    "prompt_text": prompt_text,
                    "inference_cfg": request.inference_cfg
                    if request.inference_cfg is not None
                    else CLONE_INFERENCE_CFG_DEFAULT,
                    "seed": request.seed if request.seed is not None else CLONE_SEED_DEFAULT,
                }
            )
        else:
            payload.update(
                {
                    "instruction": request.voice_description.strip(),
                    "text": normalize_synthesis_text(request.text),
                    "inference_cfg": request.inference_cfg
                    if request.inference_cfg is not None
                    else TIMBRE_INFERENCE_CFG_DEFAULT,
                    "seed": request.seed if request.seed is not None else TIMBRE_SEED_DEFAULT,
                }
            )
        return payload

    def run_worker(self, payload: dict[str, object]) -> bytes:
        """使用当前 FireRed uv 环境执行一次隔离推理。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 FireRedTTS3 uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"FireRedTTS3 worker 脚本不存在: {WORKER_SCRIPT}")
        if not os.path.isdir(CODE_PATH):
            raise RuntimeError(
                f"FireRedTTS3 官方源码目录不存在: {CODE_PATH}；请先准备 FireRedTTS3 源码。"
            )
        if not model_is_ready():
            raise RuntimeError(f"FireRedTTS3 模型目录不完整: {MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix=f"firered_tts3_{SERVICE_MODE}_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix=f"firered_tts3_{SERVICE_MODE}_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        process: subprocess.Popen | None = None
        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
            command = [
                python_executable,
                WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[FireRedTTS3/{SERVICE_MODE}] 启动 worker: python={python_executable}")
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, f"FireRedTTS3/{SERVICE_MODE}")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"FireRedTTS3/{SERVICE_MODE} worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(
                f"[FireRedTTS3/{SERVICE_MODE}] worker 退出码={process.returncode}，"
                f"耗时 {time.perf_counter() - started:.2f}s"
            )
            if process.returncode != 0:
                lines = [line.strip() for line in (stderr or stdout).splitlines() if line.strip()]
                raise RuntimeError(" | ".join(lines[-8:]) or "worker 未输出错误信息")
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FireRedTTS3 worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes:
                raise RuntimeError("FireRedTTS3 worker 返回空 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, f"FireRedTTS3/{SERVICE_MODE}")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = FireRedWorkerManager()


@app.get("/v1/health")
def health() -> dict[str, object]:
    """返回当前模式、权重、源码、worker 和 GPU 就绪状态。"""
    cuda = cuda_status()
    localai_client = LocalAIClient(GATEWAY_CONFIG) if GATEWAY_CONFIG.provider == "localai" else None
    localai_healthy = bool(localai_client and localai_client.health())
    localai_model_registered = bool(
        localai_client and localai_healthy and localai_client.model_ready()
    )
    native_ready = model_is_ready() and os.path.isdir(CODE_PATH) and cuda["available"]
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "code_path": CODE_PATH,
            "output_dir": OUTPUT_DIR,
            "prompts_dir": str(PROMPTS_DIR),
            "timbre_storage_dir": str(TIMBRE_STORAGE_DIR),
            "clone_storage_dir": str(CLONE_STORAGE_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "conda": bool(shutil.which("conda")),
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "model_dir": os.path.isdir(MODEL_DIR),
            "model_required_files": model_is_ready(),
            "code_path": os.path.isdir(CODE_PATH),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "transformers": module_available("transformers"),
            "flash_attn": module_available("flash_attn"),
            "fasttext": module_available("fasttext"),
            "cuda": cuda["available"],
            "selected_provider": GATEWAY_CONFIG.provider,
            "selected_provider_ready": (
                localai_model_registered if localai_client else native_ready
            ),
            "localai_healthy": localai_healthy,
            "localai_model_registered": localai_model_registered,
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "service_mode": SERVICE_MODE,
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if localai_client else None,
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model": "FireRedTTS3-Instruct" if SERVICE_MODE == "timbre" else "FireRedTTS3-Base",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "language_default": LANGUAGE_DEFAULT,
            "use_fasttext": USE_FASTTEXT,
            "use_wetext": USE_WETEXT,
            "do_tn": DO_TN,
            "attn_implementation": FIRERED_TTS3_ATTN_IMPLEMENTATION,
            "n_timesteps": N_TIMESTEPS_DEFAULT,
            "flash_attention_policy": (
                "optional; worker probes the extension and falls back to native PyTorch SDPA"
            ),
            "route": "/v1/FireRedTTS3/timbre"
            if SERVICE_MODE == "timbre"
            else "/v1/FireRedTTS3/clone",
        },
        "last_errors": {"firered_tts3": manager.last_error, "localai": LAST_LOCALAI_ERROR},
    }


def build_localai_request(payload: dict[str, object]) -> dict[str, object]:
    """把当前 FireRed 模式映射到固定 audio.cpp 请求字段。"""
    if USE_FASTTEXT or not USE_WETEXT or not DO_TN:
        raise UnsupportedCapabilityError(
            "固定 audio.cpp FireRedTTS3 未暴露 fasttext/WeText/DO_TN 开关；"
            "LocalAI 仅支持服务默认的文本规范化配置。"
        )
    if FIRERED_TTS3_ATTN_IMPLEMENTATION != "auto":
        raise UnsupportedCapabilityError(
            "LocalAI audio.cpp 不接受 FireRedTTS3 的 PyTorch attn_implementation 覆盖。"
        )

    request: dict[str, object] = {
        "text": str(payload["text"]),
        "language": str(payload["language"]),
        "n_timesteps": int(payload["n_timesteps"]),
        "inference_cfg": float(payload["inference_cfg"]),
        "stop_threshold": float(payload["stop_threshold"]),
        "seed": int(payload["seed"]),
        "template_name": "voice_design" if SERVICE_MODE == "timbre" else None,
    }
    if SERVICE_MODE == "clone":
        request.update(
            {
                "voice_path": Path(str(payload["ref_audio_path"])),
                "reference_text": str(payload["prompt_text"]),
                "instruction": None,
            }
        )
    else:
        request.update(
            {
                "voice_path": None,
                "reference_text": None,
                "instruction": str(payload["instruction"]),
            }
        )
    return request


def run_localai(payload: dict[str, object]) -> bytes:
    """在共享 GPU 租约内执行请求，并在释放租约前确认 backend 停止。"""
    localai_request = build_localai_request(payload)
    audio_bytes, _elapsed = execute_localai_operation(
        config=GATEWAY_CONFIG,
        operation=lambda client: client.synthesize_fireredtts3(**localai_request),
    )
    return audio_bytes


if SERVICE_MODE == "clone":

    @app.post("/v1/upload_audio")
    async def upload_audio(
        audio: UploadFile = File(...),
        full_path: str = Form(...),
        prompt_text: str | None = Form(None),
    ):
        """流式暂存并原子提交 FireRed 克隆参考音频。"""
        try:
            staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
            return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
        except AudioUploadError as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/v1/check/audio")
    def check_audio_exists(file_name: str):
        """检查逻辑参考路径并返回 SHA-256 与参考文本状态。"""
        audio_path = prompt_audio_path(file_name)
        exists = os.path.isfile(audio_path)
        return {
            "code": 200 if exists else 404,
            "exists": exists,
            "size_bytes": os.path.getsize(audio_path) if exists else None,
            "sha256": sha256_file(audio_path) if exists else None,
            "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
        }

    @app.post("/v1/FireRedTTS3/clone")
    def clone(request: CloneRequest):
        """串行运行 FireRedTTS3 Base 克隆并保存到 clone 目录。"""
        if GATEWAY_CONFIG.provider == "localai":
            global LAST_LOCALAI_ERROR
            try:
                payload = manager.build_payload(request)
                audio_bytes = run_localai(payload)
                output_path = persist_audio_bytes(audio_bytes, "fireredtts3_clone", OUTPUT_DIR)
                manager.last_error = None
                LAST_LOCALAI_ERROR = None
                print(f"[FireRedTTS3] 已保存 LocalAI 克隆音频: {output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
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
            except HTTPException:
                raise
            except Exception as exc:
                LAST_LOCALAI_ERROR = str(exc)
                LOGGER.exception("LocalAI FireRedTTS3 clone request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            payload = manager.build_payload(request)
            with gpu_runtime_lock("firered_tts3/clone"):
                with manager.lock:
                    try:
                        audio_bytes = manager.run_worker(payload)
                        output_path = persist_audio_bytes(
                            audio_bytes, "fireredtts3_clone", OUTPUT_DIR
                        )
                        print(f"[FireRedTTS3] 已保存克隆音频: {output_path}")
                        return Response(content=audio_bytes, media_type="audio/wav")
                    finally:
                        wait_after_cuda_release("after FireRedTTS3 Base worker")
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("FireRedTTS3 clone request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

else:

    @app.post("/v1/FireRedTTS3/timbre")
    def timbre(request: TimbreRequest):
        """串行运行 FireRedTTS3 Instruct 音色设计并只保存到 timbre 目录。"""
        if GATEWAY_CONFIG.provider == "localai":
            global LAST_LOCALAI_ERROR
            try:
                payload = manager.build_payload(request)
                audio_bytes = run_localai(payload)
                output_path = persist_audio_bytes(audio_bytes, "fireredtts3_timbre", OUTPUT_DIR)
                reference_store.register_timbre_file(output_path)
                manager.last_error = None
                LAST_LOCALAI_ERROR = None
                print(f"[FireRedTTS3] 已保存 LocalAI 设计音色: {output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
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
            except HTTPException:
                raise
            except Exception as exc:
                LAST_LOCALAI_ERROR = str(exc)
                LOGGER.exception("LocalAI FireRedTTS3 voice design request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            payload = manager.build_payload(request)
            with gpu_runtime_lock("firered_tts3/timbre"):
                with manager.lock:
                    try:
                        audio_bytes = manager.run_worker(payload)
                        output_path = persist_audio_bytes(
                            audio_bytes, "fireredtts3_timbre", OUTPUT_DIR
                        )
                        reference_store.register_timbre_file(output_path)
                        print(f"[FireRedTTS3] 已保存设计音色: {output_path}")
                        return Response(content=audio_bytes, media_type="audio/wav")
                    finally:
                        wait_after_cuda_release("after FireRedTTS3 Instruct worker")
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("FireRedTTS3 timbre request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI FireRedTTS3 uv service")
    print("==================================================")
    print(f"[配置] mode={SERVICE_MODE}, model={MODEL_DIR}")
    print(f"[配置] official code={CODE_PATH}")
    print(f"[配置] worker={WORKER_SCRIPT}, port={API_PORT}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, timeout={REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
