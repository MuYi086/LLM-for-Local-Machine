#!/usr/bin/env python3
"""Index-Translate 原生文本翻译 HTTP 服务，仅启动该模型的一次性 worker。"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from unitale_runtime import GpuLockTimeoutError
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from index_translate_runtime import resolve_executable, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(path: str) -> str:
    """展开配置路径，所有本地模型文件都由运维配置。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv("INDEX_TRANSLATE_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "index_translate"))
    )
)
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "INDEX_TRANSLATE_MODEL_DIR",
            str(Path(HF_MIRROR_DIR) / "IndexTeam/Index-Translate-9B-GGUF"),
        )
    )
)
MODEL_PATH = Path(
    expand_path(
        os.getenv("INDEX_TRANSLATE_MODEL_PATH", str(MODEL_DIR / "Index-Translate-9B.Q8_0.gguf"))
    )
)
LLAMA_SERVER_CONFIG = os.getenv("INDEX_TRANSLATE_LLAMA_SERVER_BIN", "llama-server")
REQUEST_TIMEOUT = float(os.getenv("INDEX_TRANSLATE_REQUEST_TIMEOUT", "900"))
STARTUP_TIMEOUT = float(os.getenv("INDEX_TRANSLATE_STARTUP_TIMEOUT", "300"))
WORKER_TIMEOUT = STARTUP_TIMEOUT + REQUEST_TIMEOUT
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
N_GPU_LAYERS = int(os.getenv("INDEX_TRANSLATE_N_GPU_LAYERS", "99"))
CONTEXT_SIZE = int(os.getenv("INDEX_TRANSLATE_CONTEXT_SIZE", "32768"))
MAX_NEW_TOKENS = int(os.getenv("INDEX_TRANSLATE_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("INDEX_TRANSLATE_TEMPERATURE", "0"))
API_HOST = os.getenv("INDEX_TRANSLATE_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("INDEX_TRANSLATE_PORT", os.getenv("PORT", "8385")))
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
MMPROJ_PATH = Path(
    expand_path(
        os.getenv(
            "INDEX_TRANSLATE_MMPROJ_PATH", str(MODEL_DIR / "Index-Translate-9B.mmproj-Q8_0.gguf")
        )
    )
)
IMAGE_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(64 * 1024 * 1024)))

if (
    REQUEST_TIMEOUT <= 0
    or STARTUP_TIMEOUT <= 0
    or float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900")) <= 0
):
    raise ValueError("请求、启动及 GPU 排队超时必须大于零。")
if CONTEXT_SIZE < 512:
    raise ValueError("上下文长度必须至少为 512。")
for directory in (WORKER_TMP_DIR, Path(GPU_LOCK_FILE).parent):
    directory.mkdir(parents=True, exist_ok=True)

LANGUAGE_NAMES = {
    "en": "英语",
    "zh": "中文",
    "de": "德语",
    "fr": "法语",
    "es": "西班牙语",
    "ja": "日语",
    "ko": "韩语",
    "pt": "葡萄牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "it": "意大利语",
    "nl": "荷兰语",
    "pl": "波兰语",
    "ro": "罗马尼亚语",
    "sv": "瑞典语",
    "tr": "土耳其语",
    "hi": "印地语",
    "vi": "越南语",
    "th": "泰语",
    "id": "印尼语",
    "ms": "马来语",
    "fil": "菲律宾语",
    "ukr_cyrl": "乌克兰语",
    "fas_arab": "波斯语",
    "ces_latn": "捷克语",
    "ell_grek": "希腊语",
    "dan_latn": "丹麦语",
    "hun_latn": "匈牙利语",
    "fin_latn": "芬兰语",
    "nob_latn": "书面挪威语",
    "slk_latn": "斯洛伐克语",
    "bul_cyrl": "保加利亚语",
}

app = FastAPI(title="Unitale Index-Translate Native Translation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TranslateRequest(BaseModel):
    """Index-Translate 官方原生任务参数；不接受语音引用或 TTS 参数。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", max_length=1_000_000)
    image: str | None = Field(default=None, max_length=IMAGE_MAX_BYTES * 4 // 3 + 128)
    source_language: str = Field(default="auto", min_length=1, max_length=128)
    target_language: str = Field(default="en", min_length=1, max_length=128)
    glossary: dict[str, str] = Field(default_factory=dict, max_length=128)
    hard_constraints: list[str] = Field(default_factory=list, max_length=128)
    soft_constraints: list[str] = Field(default_factory=list, max_length=128)
    genre: str = Field(default="文本", min_length=1, max_length=128)
    instruction: str = Field(default="", max_length=32_768)
    raw_prompt: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=TEMPERATURE, ge=0, le=2)

    @field_validator("glossary")
    @classmethod
    def validate_glossary(cls, values: dict[str, str]) -> dict[str, str]:
        """预先拒绝空术语，避免无效约束占用 GPU。"""
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("术语和译名不能为空。")
        return {key.strip(): value.strip() for key, value in values.items()}

    @field_validator("hard_constraints", "soft_constraints")
    @classmethod
    def validate_constraints(cls, values: list[str]) -> list[str]:
        """拒绝空约束，完整保留多行约束内容。"""
        if any(not value.strip() for value in values):
            raise ValueError("翻译约束不能为空白字符串。")
        return values

    @field_validator("target_language", "source_language", "genre")
    @classmethod
    def validate_label(cls, value: str) -> str:
        """校验官方模板使用的语言和文体标签。"""
        if not value.strip():
            raise ValueError("语言和文体不能为空白字符串。")
        return value.strip()

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str | None) -> str | None:
        """验证原生多模态 data URL，禁止用客户端字符串读取服务端文件。"""
        if value is None:
            return None
        header, separator, encoded = value.partition(",")
        if not separator or header not in {
            "data:image/png;base64",
            "data:image/jpeg;base64",
            "data:image/webp;base64",
            "data:image/gif;base64",
            "data:image/bmp;base64",
        }:
            raise ValueError("image 必须为 PNG/JPEG/WebP/GIF/BMP 的 base64 data URL。")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("图片 base64 无效。") from exc
        if not image_bytes or len(image_bytes) > IMAGE_MAX_BYTES:
            raise ValueError("图片为空或超过 UPLOAD_MAX_BYTES 限制。")
        return value

    @model_validator(mode="after")
    def validate_input(self):
        """图片翻译可不含参考文字；普通翻译必须有非空文本。"""
        if not self.text.strip() and self.image is None:
            raise ValueError("请提供非空 text 或 image。")
        return self


class IndexTranslateWorkerManager:
    """管理该模型独立 worker、临时 JSON 和子进程组。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: TranslateRequest) -> dict:
        """在获取 GPU 锁前检查本地依赖并保留模型原生输入。"""
        executable = resolve_executable(LLAMA_SERVER_CONFIG)
        if executable is None:
            raise FileNotFoundError(
                "未找到 llama-server，请安装 llama.cpp 并配置 INDEX_TRANSLATE_LLAMA_SERVER_BIN。"
            )
        if not MODEL_PATH.is_file() or MODEL_PATH.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Index-Translate GGUF 权重不存在: {MODEL_PATH}")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"worker 脚本不存在: {WORKER_SCRIPT}")
        if request.image and not MMPROJ_PATH.is_file():
            raise FileNotFoundError(f"图片翻译需要官方 mmproj：{MMPROJ_PATH}")
        return {
            "text": request.text.strip(),
            "image": request.image,
            "source_language_name": ""
            if request.source_language.lower() == "auto"
            else LANGUAGE_NAMES.get(request.source_language.lower(), request.source_language),
            "target_language_name": LANGUAGE_NAMES.get(
                request.target_language.lower(), request.target_language
            ),
            "glossary": request.glossary,
            "hard_constraints": request.hard_constraints,
            "soft_constraints": request.soft_constraints,
            "genre": request.genre,
            "instruction": request.instruction,
            "raw_prompt": request.raw_prompt,
            "max_new_tokens": request.max_new_tokens or MAX_NEW_TOKENS,
            "temperature": request.temperature,
            "mmproj_path": str(MMPROJ_PATH) if request.image else None,
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
            prefix="index_translate_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_translate_out_",
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
                stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "Index-Translate")
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"Index-Translate llama worker 超时（>{WORKER_TIMEOUT:.0f}s）"
                ) from exc

            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4_000:] or "Index-Translate llama worker 执行失败。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            translated_text = result.get("translated_text") if isinstance(result, dict) else None
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise RuntimeError("Index-Translate llama worker 未返回有效译文。")
            self.last_error = None
            return translated_text.strip()
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Index-Translate")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = IndexTranslateWorkerManager()


def gpu_runtime_lock(label: str):
    """只在该 Index 模型执行期间获取共享 GPU 锁。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release() -> None:
    """持锁等待一次性 worker 释放 CUDA 资源。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


@app.get("/v1/health")
def health() -> dict:
    """仅检查本地文件与可执行文件，不导入模型或访问其他服务。"""
    available = {
        "model": MODEL_PATH.is_file(),
        "llama_server": resolve_executable(LLAMA_SERVER_CONFIG) is not None,
        "worker_script": WORKER_SCRIPT.is_file(),
    }
    ready = all(available.values())
    return {
        "code": 200,
        "ready": ready,
        "available": available,
        "image_ready": ready and MMPROJ_PATH.is_file(),
        "paths": {
            "model_path": str(MODEL_PATH),
            "worker_script": str(WORKER_SCRIPT),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "runtime": {
            "backend": "Index-Translate GGUF native translation",
            "output": "text/plain",
            "context_size": CONTEXT_SIZE,
            "request_timeout": REQUEST_TIMEOUT,
            "startup_timeout": STARTUP_TIMEOUT,
            "model_lifecycle": "one request -> one worker -> process exit",
        },
        "last_errors": {"worker": manager.last_error},
    }


@app.post("/v1/indexTranslate/translate", response_class=PlainTextResponse)
def translate_text(request: TranslateRequest):
    """直接翻译文本，原生译文返回 UTF-8；不调用 ASR 或 TTS。"""
    try:
        payload = manager.build_worker_payload(request)
        with gpu_runtime_lock("index_translate/translate"):
            try:
                with manager.lock:
                    translated = manager.run_worker(payload)
            finally:
                wait_after_cuda_release()
        return PlainTextResponse(translated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Index-Translate 原生翻译失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
