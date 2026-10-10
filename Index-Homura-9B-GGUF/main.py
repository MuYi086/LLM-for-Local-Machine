#!/usr/bin/env python3
"""Index-Homura 音节约束语音翻译 HTTP 服务。"""

from __future__ import annotations

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
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from unitale_runtime import AudioReferenceStore, GpuLockTimeoutError
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from index_homura_runtime import resolve_executable, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
SUPPORTED_TARGET_LANGUAGES = (
    "zh",
    "en",
    "ja",
    "ko",
    "de",
    "fr",
    "th",
    "id",
    "vi",
    "es",
    "pt",
    "it",
    "ru",
    "ms",
)
LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
    "th": "泰语",
    "id": "印尼语",
    "vi": "越南语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "ru": "俄语",
    "ms": "马来语",
}
ASR_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "th": "Thai",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ms": "Malay",
}


def expand_path(path: str) -> str:
    """展开用户目录和环境变量，统一使用绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
PROMPTS_DIR = Path(
    expand_path(
        os.getenv("PROMPTS_DIR", os.getenv("CLONE_STORAGE_DIR", str(STORAGE_DIR / "clone")))
    )
)
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
WORKER_TMP_DIR = Path(
    expand_path(os.getenv("INDEX_HOMURA_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "index_homura")))
)
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "INDEX_HOMURA_MODEL_DIR",
            str(Path(HF_MIRROR_DIR) / "IndexTeam/Index-Homura-9B-GGUF"),
        )
    )
)
MODEL_PATH = Path(
    expand_path(os.getenv("INDEX_HOMURA_MODEL_PATH", str(MODEL_DIR / "Index-Homura-9B.Q8_0.gguf")))
)
LLAMA_SERVER_CONFIG = os.getenv("INDEX_HOMURA_LLAMA_SERVER_BIN", "llama-server")
ASR_URL = os.getenv("INDEX_HOMURA_ASR_URL", "http://127.0.0.1:8371/v1/qwen3/asr").rstrip("/")
TTS_URL = os.getenv(
    "INDEX_HOMURA_TTS_URL", "http://127.0.0.1:8361/v1/confucius4TTS/generate"
).rstrip("/")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUEST_TIMEOUT = float(os.getenv("INDEX_HOMURA_REQUEST_TIMEOUT", "900"))
STARTUP_TIMEOUT = float(os.getenv("INDEX_HOMURA_STARTUP_TIMEOUT", "300"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
N_GPU_LAYERS = int(os.getenv("INDEX_HOMURA_N_GPU_LAYERS", "99"))
CONTEXT_SIZE = int(os.getenv("INDEX_HOMURA_CONTEXT_SIZE", "4096"))
MAX_NEW_TOKENS = int(os.getenv("INDEX_HOMURA_MAX_NEW_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("INDEX_HOMURA_TEMPERATURE", "0.3"))
API_HOST = os.getenv("INDEX_HOMURA_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("INDEX_HOMURA_PORT", os.getenv("PORT", "8383")))
WORKER_SCRIPT = PROJECT_DIR / "worker.py"

if REQUEST_TIMEOUT <= 0 or STARTUP_TIMEOUT <= 0:
    raise ValueError("Index-Homura 请求与模型启动超时必须大于零。")
if CONTEXT_SIZE < 512 or MAX_NEW_TOKENS < 512 or not 0 <= TEMPERATURE <= 2:
    raise ValueError("Index-Homura 上下文长度、最大生成 token 或温度配置无效。")

for directory in (WORKER_TMP_DIR, RUNTIME_CACHE_DIR, Path(GPU_LOCK_FILE).parent):
    directory.mkdir(parents=True, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title="Unitale Index-Homura Syllable-Controlled Speech Translation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Target-Language", "X-Target-Syllables"],
)


class HomuraRequest(BaseModel):
    """待翻译音频、目标语言和目标音节数。"""

    model_config = ConfigDict(extra="forbid")

    audio_path: str = Field(min_length=1, max_length=1024)
    target_language: str = Field(
        default="en", pattern="^(zh|en|ja|ko|de|fr|th|id|vi|es|pt|it|ru|ms)$"
    )
    target_syllables: int = Field(ge=1, le=1024)
    source_language: str | None = Field(default=None, min_length=1, max_length=64)
    glossary: dict[str, str] = Field(default_factory=dict, max_length=128)
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE, ge=0, le=2)

    @field_validator("glossary")
    @classmethod
    def normalize_glossary(cls, glossary: dict[str, str]) -> dict[str, str]:
        """在 ASR 获取 GPU 前验证术语表，避免无效请求消耗推理资源。"""
        normalized = {}
        for term, translation in glossary.items():
            source_term = term.strip()
            target_term = translation.strip()
            if (
                not source_term
                or not target_term
                or len(source_term) > 256
                or len(target_term) > 256
            ):
                raise ValueError("glossary 的术语和译名必须为 1–256 个字符。")
            normalized[source_term] = target_term
        return normalized


class IndexHomuraWorkerManager:
    """管理一次性 llama worker 及其临时 JSON 生命周期。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def check_runtime_ready(self) -> None:
        """在调用其他 GPU 服务前完成 GGUF、worker 和 llama.cpp 文件预检。"""
        executable = resolve_executable(LLAMA_SERVER_CONFIG)
        if executable is None:
            raise FileNotFoundError(
                "未找到 llama-server，请安装 llama.cpp 并配置 INDEX_HOMURA_LLAMA_SERVER_BIN。"
            )
        if not MODEL_PATH.is_file() or MODEL_PATH.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Index-Homura GGUF 权重不存在: {MODEL_PATH}")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"Index-Homura worker 脚本不存在: {WORKER_SCRIPT}")

    def build_worker_payload(self, request: HomuraRequest, text: str) -> dict:
        """检查推理文件并构造符合 Homura 音节控制模板的输入。"""
        executable = resolve_executable(LLAMA_SERVER_CONFIG)
        if executable is None:
            raise FileNotFoundError(
                "未找到 llama-server，请安装 llama.cpp 并配置 INDEX_HOMURA_LLAMA_SERVER_BIN。"
            )
        if not MODEL_PATH.is_file() or MODEL_PATH.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Index-Homura GGUF 权重不存在: {MODEL_PATH}")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"Index-Homura worker 脚本不存在: {WORKER_SCRIPT}")

        max_new_tokens = request.max_new_tokens or min(MAX_NEW_TOKENS, max(512, len(text) * 3))
        return {
            "text": text,
            "target_language": request.target_language,
            "target_language_name": LANGUAGE_NAMES[request.target_language],
            "target_syllables": request.target_syllables,
            "glossary": request.glossary,
            "max_new_tokens": max_new_tokens,
            "temperature": request.temperature,
            "model_path": str(MODEL_PATH),
            "llama_server_bin": executable,
            "n_gpu_layers": N_GPU_LAYERS,
            "context_size": CONTEXT_SIZE,
            "startup_timeout_seconds": STARTUP_TIMEOUT,
            "inference_timeout_seconds": REQUEST_TIMEOUT,
        }

    def run_worker(self, payload: dict) -> str:
        """在独立进程组中运行 llama-server，超时或失败时清理整个进程组。"""
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_homura_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_homura_out_",
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
                sys.executable,
                str(WORKER_SCRIPT),
                "--input-json",
                request_path,
                "--output-json",
                output_path,
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "Index-Homura")
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"Index-Homura llama worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4_000:] or "Index-Homura llama worker 执行失败。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            translated_text = result.get("translated_text") if isinstance(result, dict) else None
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise RuntimeError("Index-Homura llama worker 未返回有效译文。")
            self.last_error = None
            return translated_text.strip()
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Index-Homura")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = IndexHomuraWorkerManager()


def gpu_runtime_lock(label: str):
    """使用仓库共享的有界 GPU 队列执行 GGUF 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release() -> None:
    """worker 退出后短暂保留 GPU 锁，让 llama.cpp 释放 CUDA 显存。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def service_health_url(endpoint: str) -> str:
    """从业务接口 URL 推导同服务的健康检查 URL。"""
    parsed = urlsplit(endpoint)
    prefix = parsed.path.split("/v1/", 1)[0]
    return f"{parsed.scheme}://{parsed.netloc}{prefix}/v1/health"


def remote_service_available(endpoint: str, required_fields: tuple[str, ...]) -> bool:
    """检查依赖健康响应中的模型和运行时状态，而不只看 HTTP 状态码。"""
    try:
        response = httpx.get(service_health_url(endpoint), timeout=2)
        if not response.is_success:
            return False
        payload = response.json()
        available = payload.get("available") if isinstance(payload, dict) else None
        return isinstance(available, dict) and all(
            available.get(field) is True for field in required_fields
        )
    except (httpx.HTTPError, ValueError):
        return False


def audio_upload_metadata(audio_path: str) -> tuple[str, str]:
    """从逻辑路径后缀推导上传给 ASR 的文件名和 MIME。"""
    suffix = Path(audio_path).suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }
    if suffix not in media_types:
        raise ValueError("audio_path 必须带受支持的音频扩展名。")
    return f"source{suffix}", media_types[suffix]


def remote_error_detail(response: httpx.Response) -> str:
    """提取依赖服务的紧凑错误信息。"""
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)[:1_000]
    except (ValueError, json.JSONDecodeError):
        pass
    return response.text.strip()[:1_000] or f"HTTP {response.status_code}"


def transcribe_audio(audio_path: Path, logical_path: str, source_language: str | None) -> dict:
    """把本地参考音频流式转发给 Qwen3-ASR。"""
    filename, mime_type = audio_upload_metadata(logical_path)
    language = source_language.strip() if source_language else None
    data = {"language": ASR_LANGUAGE_NAMES.get(language.lower(), language)} if language else {}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client, audio_path.open("rb") as audio_file:
            response = client.post(
                ASR_URL,
                data=data,
                files={"audio": (filename, audio_file, mime_type)},
            )
        if not response.is_success:
            status_code = 422 if response.status_code in {400, 422} else 503
            raise HTTPException(
                status_code=status_code,
                detail=f"Qwen3-ASR 返回 HTTP {response.status_code}: {remote_error_detail(response)}",
            )
        result = response.json()
    except (httpx.HTTPError, OSError) as exc:
        raise ConnectionError(f"无法调用 Qwen3-ASR 服务: {exc}") from exc
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or not result["text"].strip()
    ):
        raise ValueError("Qwen3-ASR 未识别出有效文本。")
    return result


def synthesize_audio(text: str, target_language: str, audio_path: str) -> bytes:
    """调用 Confucius4-TTS，以原始音频作为音色参考合成译文。"""
    try:
        response = httpx.post(
            TTS_URL,
            json={"text": text, "lang": target_language, "audio_path": audio_path},
            timeout=REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        raise ConnectionError(f"无法调用 Confucius4-TTS 服务: {exc}") from exc
    if not response.is_success:
        status_code = 422 if response.status_code in {400, 422} else 503
        raise HTTPException(
            status_code=status_code,
            detail=f"Confucius4-TTS 返回 HTTP {response.status_code}: {remote_error_detail(response)}",
        )
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError("Confucius4-TTS 返回的内容不是 WAV 音频。")
    return response.content


@app.get("/v1/health")
def health() -> dict:
    """报告本地 GGUF/llama.cpp 与 ASR、TTS 编排依赖状态。"""
    llama_server_bin = resolve_executable(LLAMA_SERVER_CONFIG)
    return {
        "code": 200,
        "paths": {
            "model_dir": str(MODEL_DIR),
            "model_path": str(MODEL_PATH),
            "llama_server_bin": llama_server_bin or LLAMA_SERVER_CONFIG,
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "model": MODEL_PATH.is_file() and MODEL_PATH.suffix.lower() == ".gguf",
            "llama_server": llama_server_bin is not None,
            "worker_script": WORKER_SCRIPT.is_file(),
            "httpx": importlib.util.find_spec("httpx") is not None,
            "qwen3_asr": remote_service_available(
                ASR_URL, ("qwen3_asr_model", "qwen_asr", "torch", "transformers", "cuda")
            ),
            "confucius4_tts": remote_service_available(
                TTS_URL,
                (
                    "confucius4_tts_model",
                    "confucius4_tts_code",
                    "w2v_bert_model",
                    "vocoder_model",
                    "style_encoder_checkpoint",
                    "torch",
                    "torchaudio",
                    "transformers",
                    "cuda",
                ),
            ),
        },
        "runtime": {
            "backend": "llama.cpp Homura GGUF text backbone + Qwen3-ASR + Confucius4-TTS",
            "target_languages": SUPPORTED_TARGET_LANGUAGES,
            "request_timeout": REQUEST_TIMEOUT,
            "startup_timeout": STARTUP_TIMEOUT,
            "n_gpu_layers": N_GPU_LAYERS,
            "context_size": CONTEXT_SIZE,
            "temperature": TEMPERATURE,
            "local_files_only": LOCAL_FILES_ONLY,
            "asr_url": ASR_URL,
            "tts_url": TTS_URL,
            "model_lifecycle": "one request -> one llama-server worker -> process exit releases VRAM",
            "gpu_scheduling": "shared exclusive file lock for GGUF inference",
        },
        "last_errors": {"index_homura": manager.last_error},
    }


@app.post("/v1/indexTranslate/homura")
def translate_speech(request: HomuraRequest):
    """转写输入音频、按目标音节数翻译并以源音频音色合成 WAV。"""
    try:
        reference_audio_path = reference_store.prompt_audio_path(request.audio_path)
        if not reference_audio_path.is_file():
            raise HTTPException(status_code=404, detail=f"参考音频不存在: {request.audio_path}")
        audio_upload_metadata(request.audio_path)
        try:
            manager.check_runtime_ready()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        asr_result = transcribe_audio(
            reference_audio_path,
            request.audio_path,
            request.source_language,
        )
        source_text = asr_result["text"].strip()
        try:
            payload = manager.build_worker_payload(request, source_text)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            with gpu_runtime_lock("index_homura/translate"):
                try:
                    with manager.lock:
                        translated_text = manager.run_worker(payload)
                finally:
                    wait_after_cuda_release()
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Index-Homura GGUF translation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        wav_bytes = synthesize_audio(translated_text, request.target_language, request.audio_path)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Target-Language": request.target_language,
                "X-Target-Syllables": str(request.target_syllables),
            },
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Index-Homura speech translation request failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI local backend Index-Homura")
    print("==================================================")
    print(f"[配置] 模型: {MODEL_PATH}")
    print(f"[配置] llama-server: {resolve_executable(LLAMA_SERVER_CONFIG) or LLAMA_SERVER_CONFIG}")
    print(f"[配置] ASR: {ASR_URL}")
    print(f"[配置] TTS: {TTS_URL}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, n_gpu_layers={N_GPU_LAYERS}, "
        f"context_size={CONTEXT_SIZE}, request_timeout={REQUEST_TIMEOUT}, "
        f"temperature={TEMPERATURE}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
