#!/usr/bin/env python3
"""Index-Nailong 原生文本翻译 HTTP 服务，仅启动该模型的一次性 worker。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from unitale_runtime import GpuLockTimeoutError
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from index_nailong_runtime import resolve_executable, terminate_process_group

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
    expand_path(os.getenv("INDEX_NAILONG_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "index_nailong")))
)
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "INDEX_NAILONG_MODEL_DIR", str(Path(HF_MIRROR_DIR) / "IndexTeam/Index-Nailong-9B-GGUF")
        )
    )
)
MODEL_PATH = Path(
    expand_path(
        os.getenv("INDEX_NAILONG_MODEL_PATH", str(MODEL_DIR / "Index-Nailong-9B.Q8_0.gguf"))
    )
)
LLAMA_SERVER_CONFIG = os.getenv("INDEX_NAILONG_LLAMA_SERVER_BIN", "llama-server")
REQUEST_TIMEOUT = float(os.getenv("INDEX_NAILONG_REQUEST_TIMEOUT", "900"))
STARTUP_TIMEOUT = float(os.getenv("INDEX_NAILONG_STARTUP_TIMEOUT", "300"))
WORKER_TIMEOUT = STARTUP_TIMEOUT + REQUEST_TIMEOUT
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
N_GPU_LAYERS = int(os.getenv("INDEX_NAILONG_N_GPU_LAYERS", "99"))
CONTEXT_SIZE = int(os.getenv("INDEX_NAILONG_CONTEXT_SIZE", "229376"))
API_HOST = os.getenv("INDEX_NAILONG_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("INDEX_NAILONG_PORT", os.getenv("PORT", "8384")))
WORKER_SCRIPT = PROJECT_DIR / "worker.py"

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

app = FastAPI(title="Unitale Index-Nailong Native Translation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NailongRequest(BaseModel):
    """Index-Nailong 官方原生任务参数；不接受语音引用或 TTS 参数。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1_000_000)
    direction: Literal["zh-en", "en-zh", "zh-ja", "ja-zh"] = "zh-en"
    max_new_tokens: int | None = Field(default=None, ge=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        """保留段落内容，拒绝仅含空白的文本。"""
        if not text.strip():
            raise ValueError("text 不能为空白字符串。")
        return text


class IndexNailongWorkerManager:
    """管理该模型独立 worker、临时 JSON 和子进程组。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: NailongRequest) -> dict:
        """在获取 GPU 锁前检查本地依赖并保留模型原生输入。"""
        executable = resolve_executable(LLAMA_SERVER_CONFIG)
        if executable is None:
            raise FileNotFoundError(
                "未找到 llama-server，请安装 llama.cpp 并配置 INDEX_NAILONG_LLAMA_SERVER_BIN。"
            )
        if not MODEL_PATH.is_file() or MODEL_PATH.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"Index-Nailong GGUF 权重不存在: {MODEL_PATH}")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"worker 脚本不存在: {WORKER_SCRIPT}")
        return {
            "text": request.text.replace("\r\n", "\n").strip(),
            "direction": request.direction,
            "max_new_tokens": request.max_new_tokens,
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
            prefix="index_nailong_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_nailong_out_",
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
                terminate_process_group(process, "Index-Nailong")
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"Index-Nailong llama worker 超时（>{WORKER_TIMEOUT:.0f}s）"
                ) from exc

            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4_000:] or "Index-Nailong llama worker 执行失败。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            translated_text = result.get("translated_text") if isinstance(result, dict) else None
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise RuntimeError("Index-Nailong llama worker 未返回有效译文。")
            self.last_error = None
            return translated_text
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Index-Nailong")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = IndexNailongWorkerManager()


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
        "directions": ["zh-en", "en-zh", "zh-ja", "ja-zh"],
        "paths": {
            "model_path": str(MODEL_PATH),
            "worker_script": str(WORKER_SCRIPT),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "runtime": {
            "backend": "Index-Nailong GGUF native translation",
            "output": "text/plain",
            "context_size": CONTEXT_SIZE,
            "request_timeout": REQUEST_TIMEOUT,
            "startup_timeout": STARTUP_TIMEOUT,
            "model_lifecycle": "one request -> one worker -> process exit",
        },
        "last_errors": {"worker": manager.last_error},
    }


@app.post("/v1/indexTranslate/naiLong", response_class=PlainTextResponse)
def translate_text(request: NailongRequest):
    """直接翻译文本，原生译文返回 UTF-8；不调用 ASR 或 TTS。"""
    try:
        payload = manager.build_worker_payload(request)
        with gpu_runtime_lock("index_nailong/translate"):
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
        LOGGER.exception("Index-Nailong 原生翻译失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
