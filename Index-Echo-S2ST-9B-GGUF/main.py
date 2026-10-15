#!/usr/bin/env python3
"""Index-Echo-S2ST 原生音频输入和语音输出 HTTP 服务。"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zipfile
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from unitale_runtime import AudioUploadError, GpuLockTimeoutError, stage_audio_upload
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from index_echo_runtime import native_runtime_status, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(path: str) -> str:
    """展开配置中的本地路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv("INDEX_ECHO_S2ST_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "index_echo_s2st"))
    )
)
UPLOAD_STAGING_DIR = WORKER_TMP_DIR / "uploads"
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "INDEX_ECHO_S2ST_MODEL_DIR", str(Path(HF_MIRROR_DIR) / "IndexTeam/Index-Echo-S2ST-9B")
        )
    )
)
DEVICE = os.getenv("INDEX_ECHO_S2ST_DEVICE", "cuda")
REQUEST_TIMEOUT = float(os.getenv("INDEX_ECHO_S2ST_REQUEST_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("INDEX_ECHO_S2ST_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("INDEX_ECHO_S2ST_PORT", os.getenv("PORT", "8382")))
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
if REQUEST_TIMEOUT <= 0 or float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900")) <= 0:
    raise ValueError("Echo 请求和 GPU 排队超时必须大于零。")
for directory in (WORKER_TMP_DIR, UPLOAD_STAGING_DIR, Path(GPU_LOCK_FILE).parent):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Unitale Index-Echo-S2ST Native API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Audio-Sample-Rate"],
)


class EchoS2stRequest(BaseModel):
    """官方 DubbingBridgeModel.dub 的已支持参数。"""

    model_config = ConfigDict(extra="forbid")
    target_language: Literal["en", "es", "ja", "zh"] = "en"
    mode: Literal["h2cv"] = "h2cv"
    chunk: Literal[False] = False
    return_info: bool = False


class IndexEchoWorkerManager:
    """只启动 Echo 完整原生包的一次性 Python worker。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: EchoS2stRequest, audio_path: Path) -> dict:
        """在获取 GPU 锁前检查完整原生包和官方运行条件。"""
        status = native_runtime_status(MODEL_DIR, DEVICE)
        if not status["ready"]:
            missing = [name for name, ready in status["available"].items() if not ready]
            raise FileNotFoundError(
                f"Echo 原生环境未就绪：{missing}；{status['gpu']}；需要完整 Index-Echo-S2ST-9B 包，文本 GGUF 不能代替。"
            )
        if not audio_path.is_file() or not WORKER_SCRIPT.is_file():
            raise FileNotFoundError("Echo 输入或 worker 文件不存在。")
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("输入音频无法解码或缺少有效时长。") from exc
        if probe.returncode != 0 or not 0 < duration < float("inf"):
            raise ValueError("Echo 输入音频必须有有限的正数时长。")
        return {
            **request.model_dump(),
            "audio_path": str(audio_path),
            "model_dir": str(MODEL_DIR),
            "device": DEVICE,
        }

    def run_worker(self, payload: dict) -> tuple[bytes, dict]:
        """在独立进程组中运行 原生语音模型，超时或失败时清理整个进程组。"""
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_echo_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="index_echo_out_",
            suffix=".json",
        )
        os.close(request_fd)
        os.close(output_fd)
        wav_fd, wav_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="index_echo_audio_", suffix=".wav"
        )
        os.close(wav_fd)
        payload = {**payload, "output_wav": wav_path}
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
                terminate_process_group(process, "Index-Echo")
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"Index-Echo 原生 worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4_000:] or "Index-Echo 原生 worker 执行失败。")
            with open(output_path, encoding="utf-8") as file:
                result = json.load(file)
            if not isinstance(result, dict):
                raise RuntimeError("Echo 原生 worker 未返回有效元数据。")
            with wave.open(wav_path, "rb") as audio:
                if audio.getnframes() == 0 or audio.getframerate() != 24_000:
                    raise RuntimeError("Echo 原生 worker 未生成有效的 24 kHz WAV。")
            wav_bytes = Path(wav_path).read_bytes()
            self.last_error = None
            return wav_bytes, result
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Index-Echo")
            for path in (request_path, output_path, wav_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = IndexEchoWorkerManager()


def gpu_runtime_lock(label: str):
    """对完整 Echo 推理只获取一次共享 GPU 锁。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release() -> None:
    """在锁释放前等待原生 worker 回收 CUDA。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def execute_payload(payload: dict) -> tuple[bytes, dict]:
    """在线程池中执行阻塞锁和一次性 worker。"""
    with gpu_runtime_lock("index_echo_s2st/native"):
        try:
            with manager.lock:
                return manager.run_worker(payload)
        finally:
            wait_after_cuda_release()


@app.get("/v1/health")
def health() -> dict:
    """报告原生环境状态，不加载模型，也不访问 ASR/TTS 服务。"""
    status = native_runtime_status(MODEL_DIR, DEVICE)
    return {
        "code": 200,
        **status,
        "paths": {"model_dir": str(MODEL_DIR), "worker_script": str(WORKER_SCRIPT)},
        "runtime": {
            "backend": "Index-Echo-S2ST DubbingBridgeModel",
            "output": "audio/wav",
            "device": DEVICE,
            "model_lifecycle": "one request -> one native worker -> process exit",
        },
        "last_errors": {"worker": manager.last_error},
    }


@app.post(
    "/v1/indexTranslate/echoS2ST",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}, "application/zip": {}}}},
)
async def echo_s2st(
    audio: UploadFile = File(...),
    target_language: str = Form("en"),
    mode: str = Form("h2cv"),
    chunk: bool = Form(False),
    return_info: bool = Form(False),
):
    """接收源语音并返回原生 24 kHz WAV；请求诊断时返回 WAV 与原生 info 的 ZIP。"""
    try:
        request = EchoS2stRequest(
            target_language=target_language, mode=mode, chunk=chunk, return_info=return_info
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR)
    except AudioUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    audio_path = staged.path.with_suffix("")
    try:
        os.replace(staged.path, audio_path)
        payload = await run_in_threadpool(manager.build_worker_payload, request, audio_path)
        wav_bytes, result = await run_in_threadpool(execute_payload, payload)
        headers = {"X-Audio-Sample-Rate": str(result["sample_rate"])}
        if return_info:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("audio.wav", wav_bytes)
                output.writestr("info.json", json.dumps(result.get("info", {}), ensure_ascii=False))
            return Response(archive.getvalue(), media_type="application/zip", headers=headers)
        return Response(wav_bytes, media_type="audio/wav", headers=headers)
    except (FileNotFoundError, GpuLockTimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Echo 原生语音翻译失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        staged.path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
