#!/usr/bin/env python3
"""Qwen3-ASR-1.7B 独立语音识别 HTTP 服务。"""

from __future__ import annotations

# API 进程只处理上传、校验、GPU 队列和 worker 生命周期，不导入模型或 Torch。
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
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    UnsupportedCapabilityError,
    load_gateway_config,
)
from inference_gateway.execution import execute_localai_transcription
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    GpuQuarantinedError,
    gpu_quarantine_path,
    stage_audio_upload,
)
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from qwen3_asr_runtime import cuda_status, terminate_process_group

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
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv("QWEN3_ASR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-ASR-1.7B"))
)
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
GPU_QUARANTINE_FILE = str(gpu_quarantine_path(GPU_LOCK_FILE))
WORKER_TMP_DIR = expand_path(
    os.getenv("QWEN3_ASR_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "qwen3_asr_worker"))
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("QWEN3_ASR_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("QWEN3_ASR_PORT", os.getenv("PORT", "8371")))
REQUEST_TIMEOUT = float(os.getenv("QWEN3_ASR_REQUEST_TIMEOUT", "900"))
DEVICE = os.getenv("QWEN3_ASR_DEVICE", "cuda:0")
DTYPE = os.getenv("QWEN3_ASR_DTYPE", "bfloat16")
MAX_INFERENCE_BATCH_SIZE = int(os.getenv("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", "1"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "256"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
    default_model="unitale.qwen3-asr.1p7b",
    request_timeout=REQUEST_TIMEOUT,
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for path in (
    Path(WORKER_TMP_DIR),
    UPLOAD_STAGING_DIR,
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(os.environ["NUMBA_CACHE_DIR"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(GPU_LOCK_FILE).parent,
):
    path.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Unitale Qwen3-ASR-1.7B API")


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


class Qwen3AsrRequest(BaseModel):
    """Qwen3-ASR 单音频转写参数。"""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(default=None, max_length=64)
    context: str = Field(default="", max_length=2_000)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=4_096)


def module_available(module_name: str) -> bool:
    """只检查模块规格，不导入可能初始化 CUDA 的模型依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def required_model_files_ready() -> bool:
    """检查配置、Tokenizer 和权重索引及索引中列出的所有权重分片。"""
    model_dir = Path(MODEL_DIR)
    required_files = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "merges.txt",
        "vocab.json",
        "model.safetensors.index.json",
    )
    if not all((model_dir / filename).is_file() for filename in required_files):
        return False
    try:
        with (model_dir / "model.safetensors.index.json").open(encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        return all((model_dir / filename).is_file() for filename in set(weight_map.values()))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行执行 Qwen3-ASR GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一项任务进入队列。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 最后的可读错误行，避免回传过长 traceback。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Qwen3-ASR worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class Qwen3AsrWorkerManager:
    """组装一次性 worker 请求，并负责临时 JSON 生命周期。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(
        self,
        request: Qwen3AsrRequest,
        audio_path: Path,
    ) -> dict[str, Any]:
        """完成纯 CPU/文件预检，并将请求字段合成为 worker JSON。"""
        if not audio_path.is_file():
            raise FileNotFoundError(f"待识别音频不存在: {audio_path}")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Qwen3-ASR worker 脚本不存在: {WORKER_SCRIPT}")
        if not required_model_files_ready():
            raise FileNotFoundError(f"Qwen3-ASR 模型文件不完整: {MODEL_DIR}")
        if not module_available("qwen_asr"):
            raise FileNotFoundError("qwen-asr 未安装，请先同步 Qwen3_ASR_1.7B 项目的锁定依赖。")
        if DTYPE not in {"bfloat16", "float16", "float32"}:
            raise ValueError("QWEN3_ASR_DTYPE 仅支持 bfloat16、float16 或 float32。")
        if MAX_INFERENCE_BATCH_SIZE < 1:
            raise ValueError("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE 必须为正整数。")

        payload = request.model_dump()
        payload.update(
            {
                "audio_path": str(audio_path),
                "model_dir": MODEL_DIR,
                "device": DEVICE,
                "dtype": DTYPE,
                "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
                "local_files_only": LOCAL_FILES_ONLY,
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        """在本服务 uv 解释器中执行一次隔离的 Qwen3-ASR 转写任务。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 Qwen3_ASR_1.7B uv 环境的 Python 解释器。")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen3_asr_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen3_asr_out_",
            suffix=".json",
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
                "--output-json",
                output_path,
            ]
            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[Qwen3-ASR] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, "Qwen3-ASR")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Qwen3-ASR worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[Qwen3-ASR] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Qwen3-ASR worker 未生成结果 JSON。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("text"), str)
                or (result.get("language") is not None and not isinstance(result["language"], str))
            ):
                raise RuntimeError("Qwen3-ASR worker 返回了非法结果 JSON。")
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Qwen3-ASR")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = Qwen3AsrWorkerManager()
LAST_LOCALAI_ERROR: str | None = None


def set_last_localai_error(error: str | None) -> None:
    """更新健康检查暴露的最近一次 LocalAI 错误。"""
    global LAST_LOCALAI_ERROR
    LAST_LOCALAI_ERROR = error


def execute_transcription_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """持有共享 GPU 锁执行一次转写，并在 worker 退出后等待显存释放。"""
    with gpu_runtime_lock("qwen3_asr/transcribe"):
        with manager.lock:
            try:
                return manager.run_worker(payload)
            finally:
                wait_after_cuda_release("after Qwen3-ASR worker")


@app.get("/v1/health")
def health():
    """返回模型文件、运行环境与 GPU 状态，不加载权重。"""
    cuda = cuda_status()
    gpu_quarantined = Path(GPU_QUARANTINE_FILE).is_file()
    localai_client = LocalAIClient(GATEWAY_CONFIG) if GATEWAY_CONFIG.provider == "localai" else None
    localai_healthy = bool(localai_client and localai_client.health())
    localai_model_registered = bool(
        localai_client and localai_healthy and localai_client.model_ready()
    )
    localai_ready = localai_model_registered and not gpu_quarantined
    native_ready = (
        os.path.isfile(WORKER_SCRIPT)
        and required_model_files_ready()
        and module_available("qwen_asr")
        and module_available("torch")
        and not gpu_quarantined
    )
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen3_asr_model_dir": MODEL_DIR,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "upload_staging_dir": str(UPLOAD_STAGING_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "gpu_quarantine_file": GPU_QUARANTINE_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "qwen3_asr_model": required_model_files_ready(),
            "qwen_asr": module_available("qwen_asr"),
            "torch": module_available("torch"),
            "transformers": module_available("transformers"),
            "selected_provider": GATEWAY_CONFIG.provider,
            "selected_provider_ready": localai_ready if localai_client else native_ready,
            "localai_healthy": localai_healthy,
            "localai_model_registered": localai_model_registered,
            "gpu_quarantined": gpu_quarantined,
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": (
                "LocalAI backend shutdown confirmation before GPU lock release"
                if localai_client
                else "one request -> one native worker -> process exit releases VRAM"
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device": DEVICE,
            "dtype": DTYPE,
            "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
            "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "gpu_scheduling": "shared exclusive file lock",
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if localai_client else None,
        },
        "last_errors": {"qwen3_asr": manager.last_error, "localai": LAST_LOCALAI_ERROR},
    }


@app.post("/v1/qwen3/asr")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    context: str = Form(""),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
):
    """流式接收音频，交给一次性 worker 返回 Qwen3-ASR 转写文本。"""
    try:
        request = Qwen3AsrRequest(
            language=language.strip() or None if language is not None else None,
            context=context,
            max_new_tokens=max_new_tokens,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    audio_path: Path | None = None
    try:
        # 暂存器以 .part 结尾；恢复真实后缀，供 Transformers 音频解码器识别格式。
        audio_path = staged.path.with_suffix("")
        os.replace(staged.path, audio_path)
        if GATEWAY_CONFIG.provider == "localai":
            validate_localai_request(request)
            result, elapsed_seconds = await run_in_threadpool(
                execute_localai_transcription,
                config=GATEWAY_CONFIG,
                audio_path=audio_path,
                filename=audio.filename or f"audio{staged.suffix}",
                content_type=audio.content_type or "application/octet-stream",
                language=request.language,
            )
            if request.language is None and not result.get("language"):
                raise LocalAIRequestError("LocalAI 未返回自动识别语种，不能满足当前 ASR 响应合同。")
            set_last_localai_error(None)
        else:
            payload = manager.build_worker_payload(request, audio_path)
            result = await run_in_threadpool(execute_transcription_payload, payload)
            elapsed_seconds = result.get("elapsed_seconds")
        return {
            "code": 200,
            "text": result["text"].strip(),
            "language": result.get("language"),
            "elapsed_seconds": elapsed_seconds,
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
    except (BackendQuarantinedError, GpuQuarantinedError) as exc:
        set_last_localai_error(str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnsupportedCapabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LocalAIRequestError, LocalAIUnavailableError) as exc:
        set_last_localai_error(str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if GATEWAY_CONFIG.provider == "localai":
            set_last_localai_error(str(exc))
        LOGGER.exception("Qwen3-ASR request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        staged.path.unlink(missing_ok=True)
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


def validate_localai_request(request: Qwen3AsrRequest) -> None:
    """拒绝当前 OpenAI 音频接口无法完整保留的 Qwen3-ASR 参数。"""
    if request.context:
        raise UnsupportedCapabilityError(
            "当前 audio-cpp transcription API 没有已验证的 Qwen3-ASR context 映射。"
        )
    if request.max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise UnsupportedCapabilityError(
            "当前 audio-cpp transcription API 没有已验证的 max_new_tokens 映射。"
        )


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Qwen3-ASR-1.7B")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device={DEVICE}, dtype={DTYPE}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
