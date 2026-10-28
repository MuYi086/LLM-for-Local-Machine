#!/usr/bin/env python3
"""IndexTTS-2.5 零样本语音克隆 HTTP 服务。"""

from __future__ import annotations

# API 进程只负责 HTTP、文件预检、GPU 排队和 worker 编排，不导入模型或 Torch。
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from llm_local_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    stage_audio_upload,
)
from llm_local_runtime import gpu_runtime_lock as shared_gpu_runtime_lock
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from index_tts25_audio_output import persist_audio_bytes
from index_tts25_model_resources import build_model_view, missing_model_resources
from index_tts25_runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def env_bool(name: str, default: bool = False) -> bool:
    """解析环境变量中的布尔值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        "INDEXTTS25_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "IndexTeam/IndexTTS-2.5"),
    )
)
CODE_PATH = expand_path(os.getenv("INDEXTTS25_CODE_PATH", "~/LLM-depency/index-tts-2.5.0"))
W2V_BERT_MODEL_DIR = expand_path(
    os.getenv(
        "INDEXTTS25_W2V_BERT_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "netease-youdao/facebook/w2v-bert-2.0"),
    )
)
CAMPPLUS_PATH = expand_path(
    os.getenv(
        "INDEXTTS25_CAMPPLUS_PATH",
        os.path.join(
            HF_MIRROR_DIR,
            "netease-youdao/funasr/campplus/campplus_cn_common.bin",
        ),
    )
)
BIGVGAN_MODEL_DIR = expand_path(
    os.getenv(
        "INDEXTTS25_BIGVGAN_MODEL_DIR",
        os.path.join(
            HF_MIRROR_DIR,
            "netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x",
        ),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "INDEXTTS25_WORKER_TMP_DIR",
        str(RUNTIME_CACHE_DIR / "index_tts25_worker"),
    )
)
OUTPUT_DIR = Path(expand_path(os.getenv("INDEXTTS25_OUTPUT_DIR", str(CLONE_STORAGE_DIR))))
LOCAL_FILES_ONLY = True
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("INDEXTTS25_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("INDEXTTS25_PORT", os.getenv("PORT", "8323")))
REQUEST_TIMEOUT = float(os.getenv("INDEXTTS25_REQUEST_TIMEOUT", "900"))
DEVICE = os.getenv("INDEXTTS25_DEVICE", "cuda:0")
USE_BF16 = env_bool("INDEXTTS25_USE_BF16", True)

os.environ.setdefault("HF_HOME", str(RUNTIME_CACHE_DIR / "index_tts25_hf"))
os.environ.setdefault("MODELSCOPE_CACHE", str(RUNTIME_CACHE_DIR / "index_tts25_modelscope"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for path in (
    PROMPTS_DIR,
    TIMBRE_STORAGE_DIR,
    Path(WORKER_TMP_DIR),
    RUNTIME_CACHE_DIR / "uploads",
    RUNTIME_CACHE_DIR / "index_tts25_hf",
    RUNTIME_CACHE_DIR / "index_tts25_modelscope",
    Path(GPU_LOCK_FILE).parent,
    OUTPUT_DIR,
):
    path.mkdir(parents=True, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title="LLM Local Backend IndexTTS-2.5 API")


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


class IndexTTS25Request(BaseModel):
    """IndexTTS-2.5 参考音频克隆请求。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=12_000)
    audio_path: str = Field(min_length=1, max_length=1_024)
    lang: Literal["ZH", "EN", "JA", "ES", "AR"] = "ZH"
    duration_factor: float = Field(default=1.0, ge=0.5, le=2.0, allow_inf_nan=False)
    emo_vector: list[float] | None = Field(default=None, min_length=8, max_length=8)
    emo_alpha: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    verbose: bool = False

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """拒绝全空白文本。"""
        if not value.strip():
            raise ValueError("text 不能为空。")
        return value

    @field_validator("lang", mode="before")
    @classmethod
    def normalize_language(cls, value: Any) -> Any:
        """将语言代码规范为 IndexTTS 使用的大写形式。"""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("emo_vector")
    @classmethod
    def validate_emotion_vector(cls, value: list[float] | None) -> list[float] | None:
        """验证情感向量每项都是有限的 [0, 1.2] 强度值。"""
        if value is None:
            return None
        if any(not math.isfinite(item) or not 0 <= item <= 1.2 for item in value):
            raise ValueError("emo_vector 每项必须是 [0, 1.2] 内的有限数值。")
        return value


class ReferenceAudioNotFoundError(FileNotFoundError):
    """逻辑引用未解析到已保存的参考音频。"""


class ModelResourcesNotReadyError(RuntimeError):
    """本地 IndexTTS-2.5 源码、模型文件或依赖尚未就绪。"""


def module_available(module_name: str) -> bool:
    """只检查模块规格，不导入可能初始化 CUDA 的依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def model_resources_missing() -> list[str]:
    """返回缺少的主模型和辅助模型文件。"""
    return missing_model_resources(
        MODEL_DIR,
        W2V_BERT_MODEL_DIR,
        CAMPPLUS_PATH,
        BIGVGAN_MODEL_DIR,
    )


def index_tts_code_is_ready() -> bool:
    """检查仓库外官方 IndexTTS-2.5 推理入口是否存在。"""
    return Path(CODE_PATH, "indextts", "infer_v2_5.py").is_file()


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 IndexTTS-2.5 GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一项任务进入队列。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def prompt_audio_path(filename: str) -> str:
    """解析普通上传或设计音色引用对应的参考音频。"""
    return str(reference_store.prompt_audio_path(filename))


def load_prompt_text_sidecar(filename: str) -> str | None:
    """读取参考音频文本 sidecar，仅用于检查接口报告状态。"""
    return reference_store.load_prompt_text(filename)


def store_uploaded_audio(staged, full_path: str, prompt_text: str | None):
    """在线程池中提交上传文件并原子保存逻辑引用。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 末尾错误行，避免向 HTTP 客户端回传长 traceback。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "IndexTTS-2.5 worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def is_worker_tmp_dir(path: Path) -> bool:
    """确认路径是 worker 临时根目录下的子目录。"""
    root = Path(WORKER_TMP_DIR).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return False
    return relative != Path(".")


class IndexTTS25WorkerManager:
    """准备只读模型视图并管理一次性 IndexTTS-2.5 worker。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: IndexTTS25Request) -> dict[str, Any]:
        """解析参考音频、检查模型并创建 worker 专用本地模型视图。"""
        reference_path = prompt_audio_path(request.audio_path)
        if not os.path.isfile(reference_path):
            raise ReferenceAudioNotFoundError(f"参考音频不存在: {request.audio_path}")
        if not os.path.isfile(WORKER_SCRIPT):
            raise ModelResourcesNotReadyError(f"IndexTTS-2.5 worker 脚本不存在: {WORKER_SCRIPT}")
        if not index_tts_code_is_ready():
            raise ModelResourcesNotReadyError(f"IndexTTS-2.5 官方源码不完整: {CODE_PATH}")

        missing = model_resources_missing()
        if missing:
            raise ModelResourcesNotReadyError(f"IndexTTS-2.5 模型资源不完整: {', '.join(missing)}")

        worker_tmp_dir = Path(tempfile.mkdtemp(prefix="index_tts25_", dir=WORKER_TMP_DIR))
        try:
            model_view_dir = build_model_view(
                model_dir=Path(MODEL_DIR),
                view_dir=worker_tmp_dir / "model-view",
                w2v_bert_dir=Path(W2V_BERT_MODEL_DIR),
                campplus_path=Path(CAMPPLUS_PATH),
                bigvgan_dir=Path(BIGVGAN_MODEL_DIR),
            )
        except Exception:
            shutil.rmtree(worker_tmp_dir, ignore_errors=True)
            raise

        payload = request.model_dump()
        payload.update(
            {
                "model_dir": str(model_view_dir),
                "code_path": CODE_PATH,
                "reference_audio_path": reference_path,
                "device": DEVICE,
                "use_bf16": USE_BF16,
                "local_files_only": LOCAL_FILES_ONLY,
                "worker_tmp_dir": str(worker_tmp_dir),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """在当前 uv 环境中运行一次推理并读取生成 WAV。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 IndexTTS-2.5 uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"IndexTTS-2.5 worker 脚本不存在: {WORKER_SCRIPT}")

        worker_tmp_dir = Path(str(payload.get("worker_tmp_dir") or "")).resolve()
        if not worker_tmp_dir.is_dir() or not is_worker_tmp_dir(worker_tmp_dir):
            raise RuntimeError("IndexTTS-2.5 worker 临时目录无效。")
        request_path = worker_tmp_dir / "request.json"
        output_path = worker_tmp_dir / "output.wav"
        process: subprocess.Popen | None = None

        worker_request = {key: value for key, value in payload.items() if key != "worker_tmp_dir"}
        try:
            with request_path.open("w", encoding="utf-8") as request_file:
                json.dump(worker_request, request_file, ensure_ascii=False)
                request_file.flush()
                os.fsync(request_file.fileno())

            command = [
                python_executable,
                WORKER_SCRIPT,
                "--input-json",
                str(request_path),
                "--output-wav",
                str(output_path),
            ]
            worker_env = os.environ.copy()
            worker_env["HF_HUB_OFFLINE"] = "1"
            worker_env["TRANSFORMERS_OFFLINE"] = "1"
            worker_env["HF_DATASETS_OFFLINE"] = "1"
            worker_env["HF_HOME"] = str(RUNTIME_CACHE_DIR / "index_tts25_hf")
            worker_env["MODELSCOPE_CACHE"] = str(RUNTIME_CACHE_DIR / "index_tts25_modelscope")

            print(f"[IndexTTS-2.5] 启动 worker: python={python_executable}")
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=worker_tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "IndexTTS-2.5")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"IndexTTS-2.5 worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[IndexTTS-2.5] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("IndexTTS-2.5 worker 未生成 WAV 文件。")
            audio_bytes = output_path.read_bytes()
            if not audio_bytes.startswith(b"RIFF"):
                raise RuntimeError("IndexTTS-2.5 worker 生成的文件不是 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "IndexTTS-2.5")
            request_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def cleanup_payload(self, payload: dict[str, Any]) -> None:
        """删除本次请求临时 JSON、WAV 和模型符号链接视图。"""
        worker_tmp_dir = payload.get("worker_tmp_dir")
        if not isinstance(worker_tmp_dir, str):
            return
        candidate = Path(worker_tmp_dir).resolve()
        if is_worker_tmp_dir(candidate):
            shutil.rmtree(candidate, ignore_errors=True)


manager = IndexTTS25WorkerManager()


@app.get("/v1/health")
def health():
    """报告服务依赖、外部源码、模型文件及 GPU 状态，不加载模型。"""
    cuda = cuda_status()
    missing = model_resources_missing()
    worker_ready = os.path.isfile(WORKER_SCRIPT)
    code_ready = index_tts_code_is_ready()
    model_ready = not missing
    modules = {
        name: module_available(name)
        for name in ("torch", "torchaudio", "transformers", "modelscope", "omegaconf", "librosa")
    }
    dependencies_ready = all(modules.values())
    device_ready = not DEVICE.startswith("cuda") or cuda["available"]
    selected_ready = (
        worker_ready and code_ready and model_ready and dependencies_ready and device_ready
    )
    return {
        "code": 200,
        "paths": {
            "index_tts25_model_dir": MODEL_DIR,
            "index_tts25_code_path": CODE_PATH,
            "w2v_bert_model_dir": W2V_BERT_MODEL_DIR,
            "campplus_path": CAMPPLUS_PATH,
            "bigvgan_model_dir": BIGVGAN_MODEL_DIR,
            "prompts_dir": str(PROMPTS_DIR),
            "tts_output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_script": worker_ready,
            "model_resources": model_ready,
            "model_resources_missing": missing,
            "index_tts_code": code_ready,
            **modules,
            "cuda": cuda["available"],
            "selected_provider": "native",
            "selected_provider_ready": selected_ready,
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device": DEVICE,
            "use_bf16": USE_BF16,
            "output_dir": str(OUTPUT_DIR),
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"index_tts25": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    """流式暂存并在共享克隆目录中提交参考音频。"""
    try:
        staged = await stage_audio_upload(audio, RUNTIME_CACHE_DIR / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查逻辑路径对应的参考音频或音色引用。"""
    exists = os.path.isfile(prompt_audio_path(file_name))
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v1/index-team/indexTTS-2.5")
def generate(request: IndexTTS25Request):
    """串行运行 IndexTTS-2.5 零样本克隆并返回 WAV。"""
    try:
        payload = manager.build_worker_payload(request)
    except ReferenceAudioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelResourcesNotReadyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("IndexTTS-2.5 request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("index_tts25/generate"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(audio_bytes, OUTPUT_DIR)
                    manager.last_output_path = str(saved_output_path)
                    print(f"[IndexTTS-2.5] 已保存生成音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except Exception as exc:
                    LOGGER.exception("IndexTTS-2.5 request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after IndexTTS-2.5 worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        manager.cleanup_payload(payload)


if __name__ == "__main__":
    print("==================================================")
    print("   LLM 本地后端 IndexTTS-2.5")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] 官方源码: {CODE_PATH}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device={DEVICE}, "
        f"use_bf16={USE_BF16}, request_timeout={REQUEST_TIMEOUT:.0f}s"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
