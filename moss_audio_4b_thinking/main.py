#!/usr/bin/env python3
"""独立 MOSS-Audio-4B 音频理解 HTTP 服务。"""

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
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    execute_localai_worker,
    load_gateway_config,
)
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import AudioUploadError, GpuLockTimeoutError, stage_audio_upload
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from moss_audio_thinking_runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
requested_variant = os.getenv("MOSS_AUDIO_4B_VARIANT")
if requested_variant is None:
    requested_variant = (
        "instruct"
        if any(key.startswith("MOSS_AUDIO_4B_INSTRUCT_") for key in os.environ)
        else "thinking"
    )
MODEL_VARIANT = requested_variant.strip().lower()
if MODEL_VARIANT not in {"thinking", "instruct"}:
    raise ValueError("MOSS_AUDIO_4B_VARIANT 仅支持 thinking 或 instruct。")

MODEL_VARIANT_LABEL = {
    "thinking": "Thinking",
    "instruct": "Instruct",
}[MODEL_VARIANT]
MODEL_ENV_PREFIX = f"MOSS_AUDIO_4B_{MODEL_VARIANT.upper()}"
MODEL_SLUG = f"moss_audio_4b_{MODEL_VARIANT}"
MODEL_SERVICE_NAME = f"MOSS-Audio-4B-{MODEL_VARIANT_LABEL}"


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动配置中的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def variant_env(name: str, default: str) -> str:
    """读取当前 MOSS-Audio 变体的专用环境变量。"""
    return os.getenv(f"{MODEL_ENV_PREFIX}_{name}", default)


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
DEFAULT_MODEL_DIR = os.path.join(
    HF_MIRROR_DIR,
    "OpenMOSS-Team",
    f"MOSS-Audio-4B-{MODEL_VARIANT_LABEL}",
)
MOSS_AUDIO_MODEL_DIR = expand_path(variant_env("MODEL_DIR", DEFAULT_MODEL_DIR))
MOSS_AUDIO_DEPENDENCY_PATH = expand_path(variant_env("DEPENDENCY_PATH", "~/tts-depency/MOSS-Audio"))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        f"{MODEL_ENV_PREFIX}_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, f"{MODEL_SLUG}_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = variant_env("HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(variant_env("PORT", os.getenv("PORT", "8341")))
REQUEST_TIMEOUT = float(variant_env("REQUEST_TIMEOUT", "900"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env=f"UNITALE_MOSS_AUDIO_4B_{MODEL_VARIANT.upper()}_PROVIDER",
    default_model=f"unitale.moss-audio-4b.{MODEL_VARIANT}.worker.v1",
    request_timeout=REQUEST_TIMEOUT,
)
DEVICE = variant_env("DEVICE", "cuda:0")
DTYPE = variant_env("DTYPE", "auto")
MAX_NEW_TOKENS = int(variant_env("MAX_NEW_TOKENS", "1024"))
ENABLE_TIME_MARKER = env_bool(f"{MODEL_ENV_PREFIX}_ENABLE_TIME_MARKER", True)
DEFAULT_DO_SAMPLE = env_bool(
    f"{MODEL_ENV_PREFIX}_DO_SAMPLE",
    MODEL_VARIANT == "instruct",
)
DEFAULT_PROMPT = (
    "Describe this audio."
    if MODEL_VARIANT == "instruct"
    else "请准确转写这段音频，仅输出转写文本。"
)

# 保留原模块变量名，兼容同目录已有的无模型集成测试；实际值由变体配置决定。
MOSS_AUDIO_4B_THINKING_MODEL_DIR = MOSS_AUDIO_MODEL_DIR
MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH = MOSS_AUDIO_DEPENDENCY_PATH

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for path in (
    WORKER_TMP_DIR,
    os.path.join(RUNTIME_CACHE_DIR, "uploads"),
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.path.dirname(GPU_LOCK_FILE) or ".",
):
    os.makedirs(path, exist_ok=True)

app = FastAPI(title=f"Unitale {MODEL_SERVICE_NAME} API")


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


class MossAudioThinkingRequest(BaseModel):
    """MOSS-Audio 音频理解请求字段。"""

    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=4_000)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=4_096)
    do_sample: bool = DEFAULT_DO_SAMPLE
    temperature: float = Field(default=1.0, gt=0, le=5)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=50, ge=1, le=500)
    enable_time_marker: bool = ENABLE_TIME_MARKER
    strip_thinking: bool = False
    device: str = Field(default=DEVICE, min_length=1, max_length=128)
    dtype: Literal["auto", "bfloat16", "float16"] = DTYPE


def module_available(module_name: str) -> bool:
    """只检查模块规格，不导入可能初始化 CUDA 的重型依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def moss_audio_source_is_ready() -> bool:
    """检查外部 MOSS-Audio 源码结构，不在 API 进程导入上游模块。"""
    source_path = Path(MOSS_AUDIO_DEPENDENCY_PATH) / "src"
    return all(
        (source_path / name).is_file()
        for name in ("audio_io.py", "modeling_moss_audio.py", "processing_moss_audio.py")
    )


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 MOSS-Audio GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一项任务进入队列。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 最后的可读错误行，避免将过长回溯直接回传客户端。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return f"{MODEL_SERVICE_NAME} worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class MossAudioThinkingWorkerManager:
    """组装一次性 worker 请求，并负责临时 JSON 生命周期。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(
        self,
        request: MossAudioThinkingRequest,
        audio_path: Path,
    ) -> dict[str, Any]:
        """完成纯 CPU/文件预检，并将请求字段合成为 worker JSON。"""
        if not audio_path.is_file():
            raise FileNotFoundError(f"待理解音频不存在: {audio_path}")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"{MODEL_SERVICE_NAME} worker 脚本不存在: {WORKER_SCRIPT}")
        if not os.path.isdir(MOSS_AUDIO_MODEL_DIR):
            raise FileNotFoundError(f"{MODEL_SERVICE_NAME} 模型目录不存在: {MOSS_AUDIO_MODEL_DIR}")
        if not moss_audio_source_is_ready():
            raise FileNotFoundError(f"MOSS-Audio 依赖源码不完整: {MOSS_AUDIO_DEPENDENCY_PATH}/src")

        payload = request.model_dump()
        payload.update(
            {
                "audio_path": str(audio_path),
                "model_path": MOSS_AUDIO_MODEL_DIR,
                "dependency_path": MOSS_AUDIO_DEPENDENCY_PATH,
                "local_files_only": LOCAL_FILES_ONLY,
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在本服务的 uv 解释器中执行一次隔离音频理解任务。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError(f"未找到 {MODEL_SLUG} uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"{MODEL_SERVICE_NAME} worker 脚本不存在: {WORKER_SCRIPT}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix=f"{MODEL_SLUG}_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix=f"{MODEL_SLUG}_out_",
            suffix=".json",
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
                "--output-json",
                output_path,
            ]
            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[{MODEL_SERVICE_NAME}] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, MODEL_SERVICE_NAME)
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"{MODEL_SERVICE_NAME} worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[{MODEL_SERVICE_NAME}] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"{MODEL_SERVICE_NAME} worker 未生成结果 JSON。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise RuntimeError(f"{MODEL_SERVICE_NAME} worker 返回了非法结果 JSON。")
            if not result["text"].strip():
                raise RuntimeError(f"{MODEL_SERVICE_NAME} worker 返回空文本。")
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, MODEL_SERVICE_NAME)
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = MossAudioThinkingWorkerManager()


def execute_understanding_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """持有共享 GPU 锁执行 worker；此函数在线程池中调用。"""
    if GATEWAY_CONFIG.provider == "localai":
        result, _ = execute_localai_worker(
            config=GATEWAY_CONFIG,
            operation="run",
            payload={"worker_input": payload},
        )
        value = result.get("value")
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise LocalAIRequestError("LocalAI MOSS-Audio worker 返回了非法文本 JSON。")
        if not value["text"].strip():
            raise LocalAIRequestError("LocalAI MOSS-Audio worker 返回空文本。")
        return value
    with gpu_runtime_lock(f"{MODEL_SLUG}/understand"):
        with manager.lock:
            try:
                return manager.run_worker(payload)
            finally:
                wait_after_cuda_release(f"after {MODEL_SERVICE_NAME} worker")


@app.get("/v1/health")
def health():
    """返回模型、源码、worker 与 GPU 的就绪状态，不加载任何权重。"""
    cuda = cuda_status()
    model_path_key = f"{MODEL_SLUG}_model_dir"
    dependency_path_key = f"{MODEL_SLUG}_dependency_path"
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            model_path_key: MOSS_AUDIO_MODEL_DIR,
            dependency_path_key: MOSS_AUDIO_DEPENDENCY_PATH,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            model_path_key: os.path.isdir(MOSS_AUDIO_MODEL_DIR),
            "moss_audio_source": moss_audio_source_is_ready(),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "transformers": module_available("transformers"),
            "soundfile": module_available("soundfile"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_variant": MODEL_VARIANT,
            "model_name": MODEL_SERVICE_NAME,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device": DEVICE,
            "dtype": DTYPE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DEFAULT_DO_SAMPLE,
            "enable_time_marker": ENABLE_TIME_MARKER,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {MODEL_SLUG: manager.last_error},
    }


@app.post("/v1/mossAudioThinking/understand")
async def understand(
    audio: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    max_new_tokens: int = Form(MAX_NEW_TOKENS),
    do_sample: bool = Form(DEFAULT_DO_SAMPLE),
    temperature: float = Form(1.0),
    top_p: float = Form(1.0),
    top_k: int = Form(50),
    enable_time_marker: bool = Form(ENABLE_TIME_MARKER),
    strip_thinking: bool = Form(False),
    device: str = Form(DEVICE),
    dtype: str = Form(DTYPE),
):
    """流式接收一个音频文件，交给一次性 worker 返回 MOSS 文本理解结果。"""
    try:
        request = MossAudioThinkingRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_time_marker=enable_time_marker,
            strip_thinking=strip_thinking,
            device=device,
            dtype=dtype,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    try:
        payload = manager.build_worker_payload(request, staged.path)
        result = await run_in_threadpool(execute_understanding_payload, payload)
        return {
            "code": 200,
            "text": result["text"].strip(),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "audio": {
                "sha256": staged.sha256,
                "size_bytes": staged.size_bytes,
                "suffix": staged.suffix,
            },
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (BackendQuarantinedError, LocalAIUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LocalAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("%s request failed", MODEL_SERVICE_NAME)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        staged.path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("==================================================")
    print(f"   Unitale AI 本地后端 {MODEL_SERVICE_NAME}")
    print("==================================================")
    print(f"[配置] 模型目录: {MOSS_AUDIO_MODEL_DIR}")
    print(f"[配置] MOSS-Audio 源码: {MOSS_AUDIO_DEPENDENCY_PATH}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device={DEVICE}, dtype={DTYPE}, "
        f"max_new_tokens={MAX_NEW_TOKENS}"
    )
    print(
        f"[配置] variant={MODEL_VARIANT}, local_files_only={LOCAL_FILES_ONLY}, "
        f"request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
