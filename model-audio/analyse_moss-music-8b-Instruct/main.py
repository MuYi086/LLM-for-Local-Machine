#!/usr/bin/env python3
"""MOSS-Music-8B-Instruct 音乐理解 API；模型仅在一次性 GGUF worker 中加载。"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    GpuQuarantinedError,
    UploadPolicy,
    gpu_runtime_lock,
    stage_audio_upload,
)
from starlette.concurrency import run_in_threadpool

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent


def expand_path(value: str) -> Path:
    """展开部署路径中的用户目录和环境变量。"""
    return Path(os.path.expandvars(value)).expanduser().resolve()


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
MODEL_DIR = expand_path(
    os.getenv("MOSS_MUSIC_MODEL_DIR", str(HF_MIRROR_DIR / "OpenMOSS/moss-music"))
)
CAPTION_BIN = expand_path(
    os.getenv("MOSS_MUSIC_CAPTION_BIN", "~/.local/share/moss-music/build/ace-caption")
)
WORKER_TMP_DIR = expand_path(
    os.getenv("MOSS_MUSIC_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "moss_music"))
)
GPU_LOCK_FILE = str(
    expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
API_HOST = os.getenv("MOSS_MUSIC_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MOSS_MUSIC_PORT", os.getenv("PORT", "8310")))
BACKEND = os.getenv("MOSS_MUSIC_BACKEND", "CUDA0")
FFMPEG_BIN = os.getenv("MOSS_MUSIC_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("MOSS_MUSIC_FFPROBE_BIN", "ffprobe")
REQUEST_TIMEOUT = float(os.getenv("MOSS_MUSIC_REQUEST_TIMEOUT", "900"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2"))
MAX_AUDIO_SECONDS = float(os.getenv("MOSS_MUSIC_MAX_AUDIO_SECONDS", "420"))
MAX_UPLOAD_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(64 * 1024 * 1024)))
MAX_OUTPUT_BYTES = 1024 * 1024
DEFAULT_PROMPT = "请详细分析这段音乐的风格、乐器、人声、情绪、节奏、编曲与制作特点。"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
# 本地完整 GGUF 制品的大小，阻止下载未完成的文件进入 GPU 队列；不在健康检查读取数 GB 权重。
MODEL_FILES = {"moss-lm-q8_0.gguf": 8_709_513_792, "moss-aud-f16.gguf": 1_724_480_640}

for name, value in (
    ("MOSS_MUSIC_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
    ("MOSS_MUSIC_MAX_AUDIO_SECONDS", MAX_AUDIO_SECONDS),
):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正的有限数。")
if MAX_AUDIO_SECONDS > 420:
    raise ValueError("MOSS_MUSIC_MAX_AUDIO_SECONDS 不能超过 GGUF 入口的 420 秒上限。")
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
if MAX_UPLOAD_BYTES <= 0:
    raise ValueError("UPLOAD_MAX_BYTES 必须是正整数。")
if BACKEND != "CPU" and not (BACKEND.startswith("CUDA") and BACKEND[4:].isdigit()):
    raise ValueError("MOSS_MUSIC_BACKEND 必须是 CPU 或 CUDA 加设备序号，例如 CUDA0。")
WORKER_TMP_DIR.mkdir(parents=True, exist_ok=True)


def model_status() -> dict[str, object]:
    """检查完整 GGUF 文件大小和文件头，不加载权重或初始化 GPU。"""
    errors: list[str] = []
    for name, expected_size in MODEL_FILES.items():
        path = MODEL_DIR / name
        try:
            if path.stat().st_size != expected_size:
                errors.append(f"{name}: 大小不匹配，预期 {expected_size} bytes")
            with path.open("rb") as source:
                if source.read(8) != b"GGUF\x03\x00\x00\x00":
                    errors.append(f"{name}: 不是 GGUF v3")
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    return {"path": str(MODEL_DIR), "complete": not errors, "errors": errors}


def backend_available() -> bool:
    """通过 nvidia-smi 检查请求的 GPU 序号，不导入 CUDA 或模型运行时。"""
    if BACKEND == "CPU":
        return True
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(BACKEND[4:]) < len(result.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        return False


def runtime_status() -> dict[str, object]:
    """汇总权重、依赖及本地二进制的可用性，健康检查不执行真实模型。"""
    model = model_status()
    dependencies = all(importlib.util.find_spec(name) for name in ("multipart", "uvicorn"))
    binary = CAPTION_BIN.is_file() and os.access(CAPTION_BIN, os.X_OK)
    ffmpeg, ffprobe = shutil.which(FFMPEG_BIN), shutil.which(FFPROBE_BIN)
    backend = backend_available()
    available = bool(
        model["complete"]
        and dependencies
        and binary
        and ffmpeg
        and ffprobe
        and WORKER_SCRIPT.is_file()
        and backend
    )
    return {
        "ready": available,
        "model": model,
        "caption_bin": str(CAPTION_BIN),
        "binary": binary,
        "dependencies": bool(dependencies),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "worker": WORKER_SCRIPT.is_file(),
        "backend": backend,
    }


def ensure_runtime_ready() -> None:
    """在音频处理及 GPU 排队前拒绝缺失的本地运行依赖。"""
    status = runtime_status()
    if not status["ready"]:
        raise FileNotFoundError(f"MOSS-Music 本地运行依赖未就绪: {status}")


def normalize_audio(source: Path, destination: Path) -> float:
    """先校验时长，再转成 16 kHz 单声道 PCM16；不静默截断超长音频。"""
    probe = subprocess.run(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    duration = float(probe.stdout.strip())
    if not math.isfinite(duration) or not 0 < duration <= MAX_AUDIO_SECONDS:
        raise ValueError(f"音乐时长必须大于 0 且不超过 {MAX_AUDIO_SECONDS:g} 秒。")
    subprocess.run(
        [
            FFMPEG_BIN,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-t",
            str(MAX_AUDIO_SECONDS + 1),
            str(destination),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    # 再核对解码长度，防止伪造容器时长绕过上限或导致 CLI 静默截断。
    import wave

    with wave.open(str(destination), "rb") as audio:
        decoded_duration = audio.getnframes() / audio.getframerate()
        if (
            audio.getframerate() != 16000
            or audio.getnchannels() != 1
            or not 0 < decoded_duration <= MAX_AUDIO_SECONDS
        ):
            raise ValueError("解码后的音频格式或时长不符合 MOSS-Music 输入要求。")
    return decoded_duration


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    """即使组长已退出仍回收后代，防止子进程跨请求持有 GPU。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # 进程组可能仍有忽略 SIGTERM 的后代，不能根据组长 poll() 提前返回。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_worker(payload: dict[str, object], request_dir: Path) -> str:
    """在当前 uv 解释器中启动一次性 worker，并核验文本输出。"""
    request = request_dir / "request.json"
    output = request_dir / "answer.txt"
    request.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            str(WORKER_SCRIPT),
            "--input-json",
            str(request),
            "--output-file",
            str(output),
        ],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"MOSS-Music worker 超过 {REQUEST_TIMEOUT:g} 秒时限。") from exc
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout)[-2000:] or "MOSS-Music worker 失败。")
        if not output.is_file() or not 0 < output.stat().st_size <= MAX_OUTPUT_BYTES:
            raise RuntimeError("MOSS-Music worker 未生成有效文本或输出超过 1 MiB。")
        answer = output.read_text(encoding="utf-8").strip()
        if not answer:
            raise RuntimeError("MOSS-Music worker 返回空文本。")
        return answer
    finally:
        terminate_process_group(process)


def execute_analysis(payload: dict[str, object], request_dir: Path) -> str:
    """串行执行一次推理，并在持锁期间等待 CUDA 资源回收。"""
    with gpu_runtime_lock(
        GPU_LOCK_FILE, "moss-music/analyse", timeout_seconds=GPU_LOCK_WAIT_TIMEOUT
    ):
        try:
            return run_worker(payload, request_dir)
        finally:
            if CUDA_RELEASE_DELAY > 0:
                time.sleep(CUDA_RELEASE_DELAY)


app = FastAPI(title="MOSS-Music-8B-Instruct 音乐理解 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/v1/health")
def health() -> dict[str, object]:
    """返回轻量运行依赖检查结果；ready 不代替真实模型推理验收。"""
    return {
        "status": "ok",
        "service": "moss-music",
        "available": runtime_status(),
        "runtime": {
            "worker_python": sys.executable,
            "backend": BACKEND,
            "model_lifecycle": "one request -> one worker -> process exit",
            "sampling_rate": 16000,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "provider": "native_gguf",
        },
    }


@app.post("/v1/openMoss/moss-music")
async def analyse_music(
    audio: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT, min_length=1, max_length=16000),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096),
    temperature: float = Form(DEFAULT_TEMPERATURE, ge=0, le=2, allow_inf_nan=False),
) -> dict[str, object]:
    """对上传音乐进行文字分析或按 prompt 回答音乐相关问题。"""
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt 不能为空白。")
    try:
        await run_in_threadpool(ensure_runtime_ready)
        with tempfile.TemporaryDirectory(prefix="request_", dir=WORKER_TMP_DIR) as temporary:
            request_dir = Path(temporary)
            staged = await stage_audio_upload(
                audio, request_dir, policy=UploadPolicy(max_bytes=MAX_UPLOAD_BYTES)
            )
            normalized = request_dir / "audio.wav"
            duration = await run_in_threadpool(normalize_audio, staged.path, normalized)
            payload = {
                "audio_path": str(normalized),
                "model_dir": str(MODEL_DIR),
                "caption_bin": str(CAPTION_BIN),
                "backend": BACKEND,
                "prompt": prompt.strip(),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "max_audio_seconds": MAX_AUDIO_SECONDS,
            }
            answer = await run_in_threadpool(execute_analysis, payload, request_dir)
            return {
                "model": "OpenMOSS-Team/MOSS-Music-8B-Instruct",
                "runtime": "hot-step-cpp-gguf",
                "text": answer,
                "duration_seconds": duration,
                "audio_sha256": staged.sha256,
            }
    except AudioUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except (GpuLockTimeoutError, GpuQuarantinedError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        LOGGER.exception("MOSS-Music 请求超时")
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(status_code=422, detail="音频无效或预检失败: " + str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("MOSS-Music 推理失败")
        raise HTTPException(status_code=500, detail=str(exc)[-2000:]) from exc
    finally:
        await audio.close()


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
