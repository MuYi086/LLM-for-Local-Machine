#!/usr/bin/env python3
"""独立 Step-Audio-EditX HTTP 服务。

API 进程刻意不导入 Torch、vLLM、ONNX Runtime 或上游模型代码。重型依赖只
由 ``worker.py`` 在一次性子进程中加载，因此没有模型环境的机器也能执行健康检查。
"""

from __future__ import annotations

# 学习入口：EditX API 只负责参考音频、参数校验和 worker 生命周期，不导入模型依赖。
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
    execute_localai_worker_file,
    load_gateway_config,
)
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    commit_plain_upload,
    stage_audio_upload,
)
from unitale_runtime.storage import hash_filename

from audio_output import persist_audio_bytes
from step_audio_editx_runtime import (
    cuda_status,
    env_bool,
    expand_path,
    gpu_runtime_lock,
    terminate_process_group,
    wait_after_cuda_release,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", str(Path(STORAGE_DIR) / "clone")))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(Path(STORAGE_DIR) / ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "STEP_AUDIO_EDITX_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, "step_audio_editx_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("STEP_AUDIO_EDITX_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("STEP_AUDIO_EDITX_PORT", os.getenv("PORT", "8331")))
REQUEST_TIMEOUT = float(os.getenv("STEP_AUDIO_EDITX_REQUEST_TIMEOUT", "900"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="UNITALE_STEP_AUDIO_EDITX_PROVIDER",
    default_model="unitale.step-audio-editx.worker.v1",
    request_timeout=REQUEST_TIMEOUT,
)
MODEL_DIR = expand_path(
    os.getenv(
        "STEP_AUDIO_EDITX_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "stepfun-ai/Step-Audio-EditX"),
    )
)
TOKENIZER_PATH = expand_path(
    os.getenv(
        "STEP_AUDIO_TOKENIZER_PATH",
        os.path.join(HF_MIRROR_DIR, "stepfun-ai/Step-Audio-Tokenizer"),
    )
)
CODE_PATH = expand_path(os.getenv("STEP_AUDIO_EDITX_CODE_PATH", "~/tts-depency/Step-Audio-EditX"))
STEP_AUDIO_EDITX_OUTPUT_DIR = expand_path(
    os.getenv("STEP_AUDIO_EDITX_OUTPUT_DIR", CLONE_STORAGE_DIR)
)
DTYPE = os.getenv("STEP_AUDIO_EDITX_DTYPE", "bfloat16")
MAX_MODEL_LEN = int(os.getenv("STEP_AUDIO_EDITX_MAX_MODEL_LEN", "3072"))
GPU_MEMORY_UTILIZATION = float(os.getenv("STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION", "0.5"))
MAX_NUM_SEQS = int(os.getenv("STEP_AUDIO_EDITX_MAX_NUM_SEQS", "1"))
COSYVOICE_DTYPE = os.getenv("STEP_AUDIO_EDITX_COSYVOICE_DTYPE", "bfloat16")
ENFORCE_EAGER = env_bool("STEP_AUDIO_EDITX_ENFORCE_EAGER", True)
COSYVOICE_CUDA_GRAPH = env_bool("STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH", False)
EDIT_TYPES = frozenset({"emotion", "style", "paralinguistic", "denoise", "vad", "speed"})

for directory in (PROMPTS_DIR, WORKER_TMP_DIR, STEP_AUDIO_EDITX_OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)
os.makedirs(os.path.dirname(GPU_LOCK_FILE) or ".", exist_ok=True)

app = FastAPI(title="Unitale Step-Audio-EditX API")


class ForceCORS(BaseHTTPMiddleware):
    """为本地 WebUI 提供跨域响应和 OPTIONS 预检处理。"""

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


class StepAudioEditXEditRequest(BaseModel):
    """Step-Audio-EditX 编辑接口的请求参数与输入约束。"""

    prompt_text: str | None = Field(default=None, max_length=12_000)
    prompt_audio: str = Field(
        min_length=1, max_length=1_024, description="经 /v1/upload_audio 上传后的 prompt 音频路径"
    )
    generated_text: str | None = Field(default=None, max_length=12_000)
    edit_type: Literal["emotion", "style", "paralinguistic", "denoise", "vad", "speed"]
    edit_info: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def validate_edit_contract(self):
        self.prompt_audio = self.prompt_audio.strip()
        self.prompt_text = self.prompt_text.strip() if self.prompt_text else None
        self.generated_text = self.generated_text.strip() if self.generated_text else None
        self.edit_info = self.edit_info.strip()
        if not self.prompt_audio:
            raise ValueError("prompt_audio 不能为空。")
        if self.edit_type not in {"denoise", "vad"} and not self.prompt_text:
            raise ValueError(f"edit_type={self.edit_type} 需要与 prompt_audio 一致的 prompt_text。")
        if self.edit_type not in {"denoise", "vad"} and not self.generated_text:
            self.generated_text = self.prompt_text
        if self.edit_type in {"emotion", "style", "speed"} and not self.edit_info:
            raise ValueError(f"edit_type={self.edit_type} 需要 edit_info。")
        return self


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def prompt_audio_path(filename: str) -> Path:
    """把 WebUI 逻辑路径解析为 clone 存储目录中的本地文件。"""
    return Path(PROMPTS_DIR) / hash_filename(filename)


def store_uploaded_audio(staged: StagedUpload, full_path: str) -> dict[str, object]:
    """同步保存 prompt 音频，供异步上传路由在线程池中调用。"""
    if not full_path.strip() or len(full_path) > 1_024:
        staged.path.unlink(missing_ok=True)
        raise AudioUploadError("full_path 不能为空且不得超过 1024 个字符。")
    save_path = prompt_audio_path(full_path)
    commit_plain_upload(staged, save_path)
    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "sha256": staged.sha256,
        "size_bytes": staged.size_bytes,
    }


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:]) if lines else "Step-Audio-EditX worker 未输出错误信息。"


class StepAudioEditXWorkerManager:
    """使用当前 uv 解释器为每个请求启动一个上游 EditX worker。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    @staticmethod
    def _value(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def build_worker_payload(
        self,
        request: StepAudioEditXEditRequest,
        prompt_wav_path: Path,
    ) -> dict[str, Any]:
        """合并编辑请求与环境默认值，生成 worker JSON。"""
        if not prompt_wav_path.is_file():
            raise FileNotFoundError(f"Step-Audio-EditX prompt 音频不存在：{prompt_wav_path}")
        return {
            "prompt_wav_path": str(prompt_wav_path),
            "prompt_text": request.prompt_text,
            "generated_text": request.generated_text,
            "edit_type": request.edit_type,
            "edit_info": request.edit_info,
            "model_path": MODEL_DIR,
            "tokenizer_path": TOKENIZER_PATH,
            "code_path": CODE_PATH,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": MAX_NUM_SEQS,
            "cosyvoice_dtype": COSYVOICE_DTYPE,
            "enforce_eager": ENFORCE_EAGER,
            "cosyvoice_cuda_graph": COSYVOICE_CUDA_GRAPH,
            "local_files_only": LOCAL_FILES_ONLY,
        }

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """启动一次 EditX 推理，校验输出并清理临时文件。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 Step_Audio_EditX uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Step-Audio-EditX worker 脚本不存在：{WORKER_SCRIPT}")
        for path, label in (
            (MODEL_DIR, "Step-Audio-EditX 模型目录"),
            (TOKENIZER_PATH, "Step-Audio-Tokenizer 模型目录"),
            (CODE_PATH, "Step-Audio-EditX 源码目录"),
        ):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{label}不存在：{path}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="step_audio_editx_req_", suffix=".json"
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="step_audio_editx_out_", suffix=".wav"
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
            worker_env.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
            print(f"[Step-Audio-EditX] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, "Step-Audio-EditX")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Step-Audio-EditX worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc
            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[Step-Audio-EditX] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Step-Audio-EditX worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes.startswith(b"RIFF"):
                raise RuntimeError("Step-Audio-EditX worker 返回的不是 WAV 文件。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Step-Audio-EditX")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def run_localai_worker(self, payload: dict[str, Any]) -> bytes:
        """通过固定 worker bridge 执行一次编辑并读取 WAV 成品。"""
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
            if not audio.startswith(b"RIFF"):
                raise LocalAIRequestError("LocalAI Step-Audio-EditX worker 返回了非法 WAV。")
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            artifact.unlink(missing_ok=True)


manager = StepAudioEditXWorkerManager()


def step_audio_editx_is_ready() -> bool:
    return (
        os.path.isdir(MODEL_DIR)
        and os.path.isdir(TOKENIZER_PATH)
        and os.path.isfile(os.path.join(CODE_PATH, "tts.py"))
        and os.path.isfile(os.path.join(CODE_PATH, "tokenizer.py"))
        and os.path.isfile(WORKER_SCRIPT)
    )


@app.get("/v1/health")
def health():
    """返回 EditX 源码、模型、worker 和 GPU 的就绪状态。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "step_audio_editx_model_dir": MODEL_DIR,
            "step_audio_tokenizer_path": TOKENIZER_PATH,
            "step_audio_editx_code_path": CODE_PATH,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "prompts_dir": PROMPTS_DIR,
            "clone_storage_dir": STEP_AUDIO_EDITX_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "step_audio_editx": step_audio_editx_is_ready(),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "vllm": module_available("vllm"),
            "onnxruntime": module_available("onnxruntime"),
            "flash_attn": module_available("flash_attn"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": MAX_NUM_SEQS,
            "enforce_eager": ENFORCE_EAGER,
            "cosyvoice_dtype": COSYVOICE_DTYPE,
            "cosyvoice_cuda_graph": COSYVOICE_CUDA_GRAPH,
            "flash_attention_policy": "not required; VLLM_ATTENTION_BACKEND=TRITON_ATTN",
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"step_audio_editx": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    """在线程池中保存编辑用参考音频，避免阻塞事件循环。"""
    try:
        staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查编辑请求引用的逻辑音频路径是否存在。"""
    exists = prompt_audio_path(file_name).is_file()
    return {"code": 200 if exists else 404, "exists": exists}


@app.post("/v1/stepAudioEditx/edit")
def step_audio_editx_edit(request: StepAudioEditXEditRequest):
    """串行执行一次语音编辑，并返回保存后的 WAV。"""
    prompt_path = prompt_audio_path(request.prompt_audio)
    try:
        payload = manager.build_worker_payload(request, prompt_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Step-Audio-EditX request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        if GATEWAY_CONFIG.provider == "localai":
            with manager.lock:
                audio_bytes = manager.run_localai_worker(payload)
                saved_output_path = persist_audio_bytes(
                    audio_bytes,
                    "step_audio_editx",
                    STEP_AUDIO_EDITX_OUTPUT_DIR,
                )
                LOGGER.info("[Step-Audio-EditX] 已保存语音音频: %s", saved_output_path)
                return Response(content=audio_bytes, media_type="audio/wav")
        with gpu_runtime_lock(GPU_LOCK_FILE, "step-audio-editx/edit"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "step_audio_editx",
                        STEP_AUDIO_EDITX_OUTPUT_DIR,
                    )
                    print(f"[Step-Audio-EditX] 已保存语音音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("Step-Audio-EditX request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release(CUDA_RELEASE_DELAY, "after Step-Audio-EditX worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (BackendQuarantinedError, LocalAIUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LocalAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Step-Audio-EditX (uv)")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] Tokenizer 目录: {TOKENIZER_PATH}")
    print(f"[配置] 上游源码: {CODE_PATH}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] host={API_HOST}, port={API_PORT}, local_files_only={LOCAL_FILES_ONLY}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
