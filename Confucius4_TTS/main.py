#!/usr/bin/env python3
"""独立 Confucius4-TTS 零样本语音克隆 HTTP 服务。"""

from __future__ import annotations

# API 进程只处理上传、校验、GPU 队列与 worker 编排，绝不导入模型或 Torch。
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    stage_audio_upload,
)
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from confucius_audio_output import persist_audio_bytes
from confucius_runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动配置中的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
CLONE_STORAGE_DIR = Path(expand_path(os.getenv("CLONE_STORAGE_DIR", str(STORAGE_DIR / "clone"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
PROMPTS_DIR = Path(expand_path(os.getenv("PROMPTS_DIR", str(CLONE_STORAGE_DIR))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "netease-youdao/Confucius4-TTS"),
    )
)
CODE_PATH = expand_path(os.getenv("CONFUCIUS4_TTS_CODE_PATH", "~/tts-depency/Confucius4-TTS"))
W2V_BERT_MODEL_DIR = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "netease-youdao/facebook/w2v-bert-2.0"),
    )
)
VOCODER_MODEL_DIR = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_VOCODER_MODEL_DIR",
        os.path.join(
            HF_MIRROR_DIR,
            "netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x",
        ),
    )
)
STYLE_ENCODER_CHECKPOINT = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT",
        os.path.join(
            HF_MIRROR_DIR,
            "netease-youdao/funasr/campplus/campplus_cn_common.bin",
        ),
    )
)
CONFIG_TEMPLATE_PATH = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_CONFIG_PATH",
        str(PROJECT_DIR / "config" / "inference_config.yaml"),
    )
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "CONFUCIUS4_TTS_WORKER_TMP_DIR",
        str(RUNTIME_CACHE_DIR / "confucius4_tts_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("CONFUCIUS4_TTS_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("CONFUCIUS4_TTS_PORT", os.getenv("PORT", "8361")))
REQUEST_TIMEOUT = float(os.getenv("CONFUCIUS4_TTS_REQUEST_TIMEOUT", "900"))
DEVICE = os.getenv("CONFUCIUS4_TTS_DEVICE", "cuda:0")
OUTPUT_DIR = Path(expand_path(os.getenv("CONFUCIUS4_TTS_OUTPUT_DIR", str(CLONE_STORAGE_DIR))))

DEFAULT_TEMPERATURE = float(os.getenv("CONFUCIUS4_TTS_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("CONFUCIUS4_TTS_TOP_P", "0.8"))
DEFAULT_TOP_K = int(os.getenv("CONFUCIUS4_TTS_TOP_K", "30"))
DEFAULT_NUM_BEAMS = int(os.getenv("CONFUCIUS4_TTS_NUM_BEAMS", "3"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("CONFUCIUS4_TTS_REPETITION_PENALTY", "10.0"))
DEFAULT_MAX_LENGTH = int(os.getenv("CONFUCIUS4_TTS_MAX_LENGTH", "1520"))
DEFAULT_N_TIMESTEPS = int(os.getenv("CONFUCIUS4_TTS_N_TIMESTEPS", "25"))
DEFAULT_INFERENCE_CFG_RATE = float(os.getenv("CONFUCIUS4_TTS_INFERENCE_CFG_RATE", "0.7"))
DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT = int(
    os.getenv("CONFUCIUS4_TTS_MAX_TEXT_TOKENS_PER_SEGMENT", "80")
)
DEFAULT_CROSS_FADE_DURATION = float(os.getenv("CONFUCIUS4_TTS_CROSS_FADE_DURATION", "0.3"))
DEFAULT_EDGE_FADE_DURATION = float(os.getenv("CONFUCIUS4_TTS_EDGE_FADE_DURATION", "0.1"))
DEFAULT_EDGE_PAD_DURATION = float(os.getenv("CONFUCIUS4_TTS_EDGE_PAD_DURATION", "0.1"))

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for path in (
    PROMPTS_DIR,
    TIMBRE_STORAGE_DIR,
    Path(WORKER_TMP_DIR),
    RUNTIME_CACHE_DIR / "uploads",
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(os.environ["NUMBA_CACHE_DIR"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(GPU_LOCK_FILE).parent,
    OUTPUT_DIR,
):
    path.mkdir(parents=True, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title="Unitale Confucius4-TTS API")


class ForceCORS(BaseHTTPMiddleware):
    """给本地 WebUI 请求补充跨域响应头。"""

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


class Confucius4TtsRequest(BaseModel):
    """Confucius4-TTS 参考音频克隆请求。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=12_000)
    lang: str = Field(default="zh", min_length=1, max_length=32)
    audio_path: str = Field(min_length=1, max_length=1_024)
    raw: bool = False
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0, le=5)
    top_p: float = Field(default=DEFAULT_TOP_P, gt=0, le=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=500)
    num_beams: int = Field(default=DEFAULT_NUM_BEAMS, ge=1, le=32)
    repetition_penalty: float = Field(default=DEFAULT_REPETITION_PENALTY, gt=0, le=100)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=1, le=8_192)
    n_timesteps: int = Field(default=DEFAULT_N_TIMESTEPS, ge=1, le=200)
    inference_cfg_rate: float = Field(default=DEFAULT_INFERENCE_CFG_RATE, ge=0, le=10)
    max_text_tokens_per_segment: int = Field(
        default=DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT,
        ge=1,
        le=1_000,
    )
    cross_fade_duration: float = Field(default=DEFAULT_CROSS_FADE_DURATION, ge=0, le=10)
    edge_fade_duration: float = Field(default=DEFAULT_EDGE_FADE_DURATION, ge=0, le=10)
    edge_pad_duration: float = Field(default=DEFAULT_EDGE_PAD_DURATION, ge=0, le=10)
    vocalization_duration_seconds: float | None = Field(
        default=None, gt=0, le=2, allow_inf_nan=False
    )
    verbose: bool = False


def module_available(module_name: str) -> bool:
    """只检查模块规格，不导入可能初始化 CUDA 的重型依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def required_model_files_ready() -> bool:
    """检查从模型目录加载所需的权重和 Tokenizer 文件。"""
    required_files = (
        "t2s_model.safetensors",
        "s2a_model.pt",
        "wav2vec2bert_stats.pt",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    return all((Path(MODEL_DIR) / name).is_file() for name in required_files)


def confucius_code_is_ready() -> bool:
    """检查官方 Confucius4-TTS 源码结构，不在 API 进程导入模型代码。"""
    code_root = Path(CODE_PATH)
    required_files = (
        code_root / "confuciustts" / "cli" / "inference.py",
        code_root / "confuciustts" / "flow" / "flow.py",
        code_root / "external" / "bigvgan" / "bigvgan.py",
        code_root / "external" / "campplus" / "__init__.py",
    )
    return all(path.is_file() for path in required_files)


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 Confucius4-TTS GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一项任务进入队列。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def prompt_audio_path(filename: str) -> str:
    """解析普通上传或音色设计引用对应的参考音频路径。"""
    return str(reference_store.prompt_audio_path(filename))


def load_prompt_text_sidecar(filename: str) -> str | None:
    """读取参考音频的文本 sidecar，供检查接口报告状态。"""
    return reference_store.load_prompt_text(filename)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 最后的可读错误行，避免回传过长 traceback。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Confucius4-TTS worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def store_uploaded_audio(staged, full_path: str, prompt_text: str | None):
    """在线程池中提交上传参考音频，复用共享原子存储策略。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


class Confucius4TtsWorkerManager:
    """组装一次性 worker 请求，并负责临时 JSON/WAV 生命周期。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: Confucius4TtsRequest) -> dict[str, Any]:
        """完成纯 CPU/文件预检，并将请求字段合成为 worker JSON。"""
        reference_path = prompt_audio_path(request.audio_path)
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(f"参考音频不存在: {request.audio_path}")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Confucius4-TTS worker 脚本不存在: {WORKER_SCRIPT}")
        if not required_model_files_ready():
            raise FileNotFoundError(f"Confucius4-TTS 模型文件不完整: {MODEL_DIR}")
        if not confucius_code_is_ready():
            raise FileNotFoundError(f"Confucius4-TTS 源码不完整: {CODE_PATH}")
        if not os.path.isdir(W2V_BERT_MODEL_DIR):
            raise FileNotFoundError(f"Wav2Vec2-BERT 模型目录不存在: {W2V_BERT_MODEL_DIR}")
        if not os.path.isdir(VOCODER_MODEL_DIR):
            raise FileNotFoundError(f"BigVGAN 模型目录不存在: {VOCODER_MODEL_DIR}")
        if not os.path.isfile(STYLE_ENCODER_CHECKPOINT):
            raise FileNotFoundError(f"CAMPPlus 权重不存在: {STYLE_ENCODER_CHECKPOINT}")
        if not os.path.isfile(CONFIG_TEMPLATE_PATH):
            raise FileNotFoundError(f"Confucius4-TTS 配置模板不存在: {CONFIG_TEMPLATE_PATH}")

        payload = request.model_dump()
        payload.update(
            {
                "reference_audio_path": reference_path,
                "model_dir": MODEL_DIR,
                "code_path": CODE_PATH,
                "w2v_bert_model_dir": W2V_BERT_MODEL_DIR,
                "vocoder_model_dir": VOCODER_MODEL_DIR,
                "style_encoder_checkpoint": STYLE_ENCODER_CHECKPOINT,
                "config_template_path": CONFIG_TEMPLATE_PATH,
                "device": DEVICE,
                "local_files_only": LOCAL_FILES_ONLY,
                "hf_mirror_dir": HF_MIRROR_DIR,
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """在本服务的 uv 解释器中执行一次隔离 Confucius4-TTS 合成任务。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 Confucius4_TTS uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Confucius4-TTS worker 脚本不存在: {WORKER_SCRIPT}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="confucius4_tts_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="confucius4_tts_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        process: subprocess.Popen | None = None
        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())

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
            print(f"[Confucius4-TTS] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, "Confucius4-TTS")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Confucius4-TTS worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[Confucius4-TTS] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Confucius4-TTS worker 未生成 WAV 文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes.startswith(b"RIFF"):
                raise RuntimeError("Confucius4-TTS worker 生成的文件不是 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Confucius4-TTS")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = Confucius4TtsWorkerManager()


@app.get("/v1/health")
def health():
    """返回服务、模型文件、源码和 GPU 状态，不加载任何权重。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "confucius4_tts_model_dir": MODEL_DIR,
            "confucius4_tts_code_path": CODE_PATH,
            "w2v_bert_model_dir": W2V_BERT_MODEL_DIR,
            "vocoder_model_dir": VOCODER_MODEL_DIR,
            "style_encoder_checkpoint": STYLE_ENCODER_CHECKPOINT,
            "prompts_dir": str(PROMPTS_DIR),
            "tts_output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "confucius4_tts_model": required_model_files_ready(),
            "confucius4_tts_code": confucius_code_is_ready(),
            "w2v_bert_model": os.path.isdir(W2V_BERT_MODEL_DIR),
            "vocoder_model": os.path.isdir(VOCODER_MODEL_DIR),
            "style_encoder_checkpoint": os.path.isfile(STYLE_ENCODER_CHECKPOINT),
            "config_template": os.path.isfile(CONFIG_TEMPLATE_PATH),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "transformers": module_available("transformers"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "native_vocalization_duration": True,
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device": DEVICE,
            "output_dir": str(OUTPUT_DIR),
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"confucius4_tts": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    """在线程池中保存上传参考音频，避免阻塞 FastAPI 事件循环。"""
    try:
        staged = await stage_audio_upload(audio, RUNTIME_CACHE_DIR / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查逻辑路径对应的参考音频或音色引用是否存在。"""
    exists = os.path.isfile(prompt_audio_path(file_name))
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v1/confucius4TTS/generate")
def generate(request: Confucius4TtsRequest):
    """串行执行一次 Confucius4-TTS 零样本克隆，并返回生成 WAV。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Confucius4-TTS request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("confucius4_tts/generate"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(audio_bytes, OUTPUT_DIR)
                    print(f"[Confucius4-TTS] 已保存生成音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except Exception as exc:
                    LOGGER.exception("Confucius4-TTS request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after Confucius4-TTS worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Confucius4-TTS")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] 官方源码: {CODE_PATH}")
    print(f"[配置] Wav2Vec2-BERT: {W2V_BERT_MODEL_DIR}")
    print(f"[配置] BigVGAN: {VOCODER_MODEL_DIR}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device={DEVICE}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
