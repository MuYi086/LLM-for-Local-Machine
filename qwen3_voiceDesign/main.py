#!/usr/bin/env python3
"""独立 Qwen3-TTS VoiceDesign HTTP 服务。"""

from __future__ import annotations

# 学习入口：VoiceDesign API 不导入模型，只负责请求校验、锁、worker 和音频存储。
import importlib.util
import io
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
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioReferenceStore,
    GpuLockTimeoutError,
)
from unitale_runtime import (
    gpu_runtime_lock as shared_gpu_runtime_lock,
)

from audio_output import persist_audio_bytes
from voicedesign_runtime import cuda_status, split_text, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动脚本传入的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开用户目录和环境变量，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
QWEN_VOICEDESIGN_MODEL_DIR = expand_path(
    os.getenv(
        "QWEN_VOICEDESIGN_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "QWEN_VOICEDESIGN_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, "qwen_voicedesign_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("QWEN_VOICEDESIGN_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("QWEN_VOICEDESIGN_PORT", os.getenv("PORT", "8301")))
REQUEST_TIMEOUT = float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="UNITALE_QWEN3_VOICEDESIGN_PROVIDER",
    default_model="unitale.qwen3-tts.1p7b-voicedesign",
    request_timeout=REQUEST_TIMEOUT,
)
DEVICE_MAP = os.getenv("QWEN_VOICEDESIGN_DEVICE_MAP", "cuda:0")
DTYPE = os.getenv("QWEN_VOICEDESIGN_DTYPE", "auto")
ATTN_IMPLEMENTATION = os.getenv(
    "QWEN_VOICEDESIGN_ATTN_IMPLEMENTATION",
    "auto",
)
LANGUAGE = os.getenv("QWEN_VOICEDESIGN_LANGUAGE", "Chinese")
MAX_CHARS_PER_CHUNK = int(os.getenv("QWEN_VOICEDESIGN_MAX_CHARS_PER_CHUNK", "0"))
PAUSE_MS = int(os.getenv("QWEN_VOICEDESIGN_PAUSE_MS", "250"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_VOICEDESIGN_MAX_NEW_TOKENS", "2048"))

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
    TIMBRE_STORAGE_DIR,
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
):
    os.makedirs(path, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.path.dirname(GPU_LOCK_FILE) or ".", exist_ok=True)

reference_store = AudioReferenceStore(STORAGE_DIR / "clone", TIMBRE_STORAGE_DIR)

app = FastAPI(title="Unitale Qwen3-TTS VoiceDesign API")
LAST_LOCALAI_ERROR: str | None = None


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


class QwenDesignRequest(BaseModel):
    """独立 VoiceDesign 服务的请求参数，文本和音色描述均由 WebUI 提供。"""

    voice_description: str = Field(min_length=1, max_length=2_000)
    text: str = Field(default="这是生成的参考音频预览。", min_length=1, max_length=12_000)
    language: str | None = Field(default="Chinese", max_length=32)
    max_chars_per_chunk: int | None = Field(default=0, ge=0, le=2_000)
    pause_ms: int | None = Field(default=250, ge=0, le=10_000)
    max_new_tokens: int | None = Field(default=2048, ge=1, le=4_096)
    top_p: float | None = Field(default=None, gt=0, le=1)
    temperature: float | None = Field(default=None, ge=0, le=5)
    dtype: str | None = Field(default="auto", max_length=32)
    attn_implementation: str | None = Field(default="auto", max_length=64)
    device_map: str | None = Field(default="cuda:0", max_length=128)


def module_available(module_name: str) -> bool:
    """只检查模块规格，不实际导入重型依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def gpu_runtime_lock(label: str):
    """通过共享文件锁保证多个本地 GPU 服务不同时推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一个请求进入。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Qwen3-TTS VoiceDesign worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def validate_localai_request(request: QwenDesignRequest, payload: dict[str, Any]) -> list[str]:
    """在 GPU 排队前拒绝 LocalAI 无法保留的 PyTorch 配置并生成分段。"""
    hardware_defaults = {
        "device_map": DEVICE_MAP,
        "dtype": DTYPE,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    requested_hardware_fields = [
        name for name, default in hardware_defaults.items() if getattr(request, name) != default
    ]
    if requested_hardware_fields:
        fields = ", ".join(requested_hardware_fields)
        raise UnsupportedCapabilityError(
            f"LocalAI audio-cpp 不支持逐请求设置 PyTorch 参数：{fields}。"
        )

    chunks = split_text(str(payload["text"]), int(payload["max_chars_per_chunk"]))
    if len(chunks) > 32:
        raise UnsupportedCapabilityError(
            "LocalAI VoiceDesign 单次请求最多支持 32 个文本片段；请增大 max_chars_per_chunk。"
        )
    if payload["temperature"] is not None and float(payload["temperature"]) <= 0:
        raise UnsupportedCapabilityError(
            "audio.cpp Qwen3 VoiceDesign 在采样开启时要求 temperature 大于 0。"
        )
    return chunks


def join_localai_wavs(wav_chunks: list[bytes], pause_ms: int) -> bytes:
    """验证 LocalAI 分段 WAV 并按原 VoiceDesign worker 规则插入静音合并。"""
    if not wav_chunks:
        raise LocalAIRequestError("LocalAI VoiceDesign 未返回音频片段。")
    if sum(map(len, wav_chunks)) > 128 * 1024 * 1024:
        raise LocalAIRequestError("LocalAI VoiceDesign 分段音频超过 128 MiB 响应上限。")
    try:
        import numpy as np
        import soundfile as sf

        waveforms: list[Any] = []
        sample_rate: int | None = None
        for wav_chunk in wav_chunks:
            waveform, rate = sf.read(io.BytesIO(wav_chunk), dtype="float32", always_2d=True)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("WAV 片段为空或包含非有限样本。")
            if sample_rate is not None and rate != sample_rate:
                raise ValueError("WAV 分段采样率不一致。")
            sample_rate = int(rate)
            waveforms.append(waveform.mean(axis=1, dtype=np.float32))

        assert sample_rate is not None
        pause = np.zeros(int(sample_rate * max(pause_ms, 0) / 1000), dtype=np.float32)
        joined: list[Any] = []
        for index, waveform in enumerate(waveforms):
            joined.append(waveform)
            if index + 1 < len(waveforms) and len(pause):
                joined.append(pause)
        output = io.BytesIO()
        sf.write(output, np.concatenate(joined), sample_rate, format="WAV")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise LocalAIRequestError(f"LocalAI VoiceDesign 返回的 WAV 无效：{exc}") from exc
    return output.getvalue()


def synthesize_localai(request: QwenDesignRequest, payload: dict[str, Any]) -> tuple[bytes, float]:
    """在一次共享 GPU 租约内逐段设计音色，并确认 backend 停止。"""
    chunks = validate_localai_request(request, payload)

    def run(client: LocalAIClient) -> tuple[bytes, float]:
        wav_chunks: list[bytes] = []
        elapsed_seconds = 0.0
        for text_chunk in chunks:
            audio_bytes, elapsed = client.design_voice(
                text=text_chunk,
                instruction=str(payload["voice_description"]),
                language=payload["language"],
                max_new_tokens=int(payload["max_new_tokens"]),
                top_p=payload["top_p"],
                temperature=payload["temperature"],
            )
            wav_chunks.append(audio_bytes)
            elapsed_seconds += elapsed
        return join_localai_wavs(wav_chunks, int(payload["pause_ms"])), elapsed_seconds

    return execute_localai_operation(config=GATEWAY_CONFIG, operation=run)


class QwenVoiceDesignWorkerManager:
    """组装 VoiceDesign worker 请求并管理临时 JSON/WAV 文件。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    @staticmethod
    def _value(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def build_worker_payload(self, request: QwenDesignRequest) -> dict[str, Any]:
        """将请求字段与环境默认值合并为 worker 输入字典。"""
        payload = request.model_dump()
        payload.update(
            {
                "model_path": QWEN_VOICEDESIGN_MODEL_DIR,
                "local_files_only": LOCAL_FILES_ONLY,
                "language": self._value(request.language, LANGUAGE),
                "max_chars_per_chunk": self._value(
                    request.max_chars_per_chunk, MAX_CHARS_PER_CHUNK
                ),
                "pause_ms": self._value(request.pause_ms, PAUSE_MS),
                "max_new_tokens": self._value(request.max_new_tokens, MAX_NEW_TOKENS),
                "dtype": self._value(request.dtype, DTYPE),
                "attn_implementation": self._value(
                    request.attn_implementation, ATTN_IMPLEMENTATION
                ),
                "device_map": self._value(request.device_map, DEVICE_MAP),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """在 qwen3_voiceDesign uv 解释器中执行一次隔离推理。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 qwen3_voiceDesign uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Qwen3-TTS VoiceDesign worker 脚本不存在: {WORKER_SCRIPT}")
        if not os.path.isdir(QWEN_VOICEDESIGN_MODEL_DIR):
            raise RuntimeError(
                f"Qwen3-TTS VoiceDesign 模型目录不存在: {QWEN_VOICEDESIGN_MODEL_DIR}"
            )

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen_voicedesign_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen_voicedesign_out_",
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
            print(f"[Qwen3-TTS VoiceDesign] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, "Qwen3-TTS VoiceDesign")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Qwen3-TTS VoiceDesign worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(
                f"[Qwen3-TTS VoiceDesign] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s"
            )
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Qwen3-TTS VoiceDesign worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes:
                raise RuntimeError("Qwen3-TTS VoiceDesign worker 返回空 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Qwen3-TTS VoiceDesign")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = QwenVoiceDesignWorkerManager()


@app.get("/v1/health")
def health():
    """返回 VoiceDesign 依赖和模型状态，不加载任何模型权重。"""
    cuda = cuda_status()
    localai_client = LocalAIClient(GATEWAY_CONFIG) if GATEWAY_CONFIG.provider == "localai" else None
    localai_healthy = bool(localai_client and localai_client.health())
    localai_model_registered = bool(
        localai_client and localai_healthy and localai_client.model_ready()
    )
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen_voicedesign_model_dir": QWEN_VOICEDESIGN_MODEL_DIR,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "timbre_storage_dir": str(TIMBRE_STORAGE_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "qwen_voicedesign_model_dir": os.path.isdir(QWEN_VOICEDESIGN_MODEL_DIR),
            "torch": module_available("torch"),
            "qwen_tts": module_available("qwen_tts"),
            "soundfile": module_available("soundfile"),
            "flash_attn": module_available("flash_attn"),
            "cuda": cuda["available"],
            "selected_provider": GATEWAY_CONFIG.provider,
            "selected_provider_ready": (
                localai_model_registered
                if localai_client
                else os.path.isdir(QWEN_VOICEDESIGN_MODEL_DIR)
            ),
            "localai_healthy": localai_healthy,
            "localai_model_registered": localai_model_registered,
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if localai_client else None,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device_map": DEVICE_MAP,
            "dtype": DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "flash_attention_policy": "optional; auto falls back to sdpa when unavailable",
            "language": LANGUAGE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
            "pause_ms": PAUSE_MS,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {
            "qwen_voicedesign": manager.last_error,
            "localai": LAST_LOCALAI_ERROR,
        },
    }


@app.post("/v1/qwen/timbre")
def qwen_design(request: QwenDesignRequest):
    """串行执行一次音色设计，并将 WAV 保存到 timbre 目录。"""
    global LAST_LOCALAI_ERROR
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Qwen3 VoiceDesign request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if GATEWAY_CONFIG.provider == "localai":
        try:
            validate_localai_request(request, payload)
            with manager.lock:
                audio_bytes, _elapsed = synthesize_localai(request, payload)
                saved_output_path = persist_audio_bytes(
                    audio_bytes,
                    "qwen_voicedesign",
                    TIMBRE_STORAGE_DIR,
                )
                reference_store.register_timbre_file(saved_output_path)
                manager.last_error = None
                LAST_LOCALAI_ERROR = None
                print(f"[Qwen3-TTS VoiceDesign LocalAI] 已保存音色音频: {saved_output_path}")
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
        except Exception as exc:
            LAST_LOCALAI_ERROR = str(exc)
            LOGGER.exception("LocalAI Qwen3 VoiceDesign request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("qwen/design"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "qwen_voicedesign",
                        TIMBRE_STORAGE_DIR,
                    )
                    reference_store.register_timbre_file(saved_output_path)
                    print(f"[Qwen3-TTS VoiceDesign] 已保存音色音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("Qwen3 VoiceDesign request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after Qwen VoiceDesign worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Qwen3-TTS VoiceDesign")
    print("==================================================")
    print(f"[配置] 模型目录: {QWEN_VOICEDESIGN_MODEL_DIR}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device_map={DEVICE_MAP}, "
        f"dtype={DTYPE}, attn_implementation={ATTN_IMPLEMENTATION}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
