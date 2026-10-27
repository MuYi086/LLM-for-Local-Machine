#!/usr/bin/env python3
"""MSST Mega 53 重点声部分离服务：流式上传、CPU 预检、一次性 GPU worker 和 ZIP 成品。"""

from __future__ import annotations

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
import uuid
import zipfile
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    GpuQuarantinedError,
    UploadPolicy,
    gpu_runtime_lock,
    stage_audio_upload,
)
from starlette.concurrency import run_in_threadpool

from contract import (
    CONFIG_FILENAME,
    SAMPLE_RATE,
    WEIGHT_FILENAME,
    parse_stems,
    supported_stems,
)
from msst_runtime import cuda_status, module_available

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent


def expand_path(value: str) -> Path:
    """展开用户目录和环境变量，保持服务配置与统一启动一致。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def positive_setting(name: str, default: str) -> float:
    """拒绝无限排队、NaN 和无效时限。"""
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须为正的有限数。")
    return value


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
MODEL_DIR = expand_path(os.getenv("MSST_MODEL_DIR", str(HF_MIRROR_DIR / "oulianov/mvsep_mega_53")))
SOURCE_DIR = expand_path(
    os.getenv("MSST_SOURCE_DIR", "~/LLM-depency/Music-Source-Separation-Training")
)
OUTPUT_DIR = expand_path(os.getenv("MSST_OUTPUT_DIR", str(STORAGE_DIR / "msst")))
WORKER_TMP_DIR = expand_path(os.getenv("MSST_WORKER_TMP_DIR", str(CACHE_DIR / "msst_worker")))
UPLOAD_DIR = expand_path(os.getenv("MSST_UPLOAD_DIR", str(CACHE_DIR / "msst_uploads")))
GPU_LOCK_FILE = str(expand_path(os.getenv("GPU_LOCK_FILE", str(CACHE_DIR / "gpu-runtime.lock"))))
REQUEST_TIMEOUT = positive_setting("MSST_REQUEST_TIMEOUT", "900")
GPU_LOCK_WAIT_TIMEOUT = positive_setting("GPU_LOCK_WAIT_TIMEOUT", "900")
MAX_AUDIO_SECONDS = positive_setting("MSST_MAX_AUDIO_SECONDS", "600")
MAX_OUTPUT_BYTES = int(positive_setting("MSST_MAX_OUTPUT_BYTES", str(2 * 1024**3)))
MAX_UPLOAD_BYTES = int(positive_setting("UPLOAD_MAX_BYTES", str(64 * 1024 * 1024)))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2"))
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
DEVICE = os.getenv("MSST_DEVICE", "cuda:0")
if DEVICE != "cpu" and not (DEVICE.startswith("cuda:") and DEVICE[5:].isdigit()):
    raise ValueError("MSST_DEVICE 必须是 cpu 或 cuda:N。")
FFMPEG_BIN = os.getenv("MSST_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("MSST_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("MSST_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MSST_PORT", "8380"))
DEFAULT_CHUNK_SECONDS = 10
REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "numpy",
    "soundfile",
    "librosa",
    "ml_collections",
    "omegaconf",
    "beartype",
    "einops",
    "rotary_embedding_torch",
    "PoPE_pytorch",
    "matplotlib",
    "packaging",
    "tqdm",
    "yaml",
)
REQUIRED_SOURCE_FILES = (
    "utils/settings.py",
    "utils/model_utils.py",
    "models/bs_roformer/bs_roformer.py",
)
for directory in (OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend MSST Mega 53 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-MSST-Batch-ID", "X-MSST-Audio-SHA256"],
)


def runtime_status() -> dict:
    """检查模块规格、模型和外部源码，不导入 Torch 或加载权重。"""
    missing_model = [
        name
        for name in (WEIGHT_FILENAME, CONFIG_FILENAME)
        if not (MODEL_DIR / name).is_file() or (MODEL_DIR / name).stat().st_size == 0
    ]
    missing_source = [name for name in REQUIRED_SOURCE_FILES if not (SOURCE_DIR / name).is_file()]
    missing_dependencies = [name for name in REQUIRED_MODULES if not module_available(name)]
    missing_tools = [name for name in (FFMPEG_BIN, FFPROBE_BIN) if shutil.which(name) is None]
    config_error = None
    supported = []
    if not missing_model:
        try:
            supported = supported_stems(MODEL_DIR / CONFIG_FILENAME)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            config_error = str(exc)
    gpu = cuda_status()
    ready = not (
        missing_model or missing_source or missing_dependencies or missing_tools or config_error
    )
    device_available = DEVICE == "cpu" or (
        gpu["available"] and int(DEVICE[5:]) < gpu.get("device_count", 1)
    )
    worker_available = (PROJECT_DIR / "worker.py").is_file()
    ready = ready and device_available and worker_available
    return {
        "ready": ready,
        "worker_available": worker_available,
        "device_available": device_available,
        "missing_model": missing_model,
        "config_error": config_error,
        "supported_stems": supported,
        "missing_source": missing_source,
        "missing_dependencies": missing_dependencies,
        "missing_tools": missing_tools,
        "cuda": gpu,
    }


@app.get("/v1/health")
def health() -> dict:
    """返回服务可用性和重点声部合同；ready 不代表真实推理验收通过。"""
    return {
        **runtime_status(),
        "port": API_PORT,
        "model_dir": str(MODEL_DIR),
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "device": DEVICE,
        "worker_python": sys.executable,
        "model_lifecycle": "one-shot",
    }


def normalize_audio(source: Path, destination: Path) -> None:
    """GPU 排队前用 FFmpeg 解码到 44.1 kHz 双声道 FLOAT WAV，且不改变增益。"""
    probe = subprocess.run(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    info = json.loads(probe.stdout)
    duration = float(info.get("format", {}).get("duration", "nan"))
    if (
        not info.get("streams")
        or not math.isfinite(duration)
        or not 0 < duration <= MAX_AUDIO_SECONDS
    ):
        raise ValueError(f"必须包含音频，时长须为 0–{MAX_AUDIO_SECONDS:g} 秒。")
    subprocess.run(
        [
            FFMPEG_BIN,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-t",
            str(MAX_AUDIO_SECONDS + 1),
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=REQUEST_TIMEOUT,
    )
    # ffprobe 容器时长可能不可靠；再检查实际解码长度，避免伪造时长绕过资源限制。
    if destination.stat().st_size > int(MAX_AUDIO_SECONDS * SAMPLE_RATE * 2 * 4) + 4096:
        raise ValueError("解码后的音频超过时长限制。")


def stop_worker(process: subprocess.Popen | None) -> None:
    """即使 worker 主进程已退出也回收进程组，避免子进程残留 CUDA 上下文。"""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def execute_worker(payload: dict, job_dir: Path) -> None:
    """持共享 GPU 锁执行一个 worker，在所有退出路径回收进程组并延迟解锁。"""
    request = job_dir / "request.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    with gpu_runtime_lock(GPU_LOCK_FILE, "msst/separate", timeout_seconds=GPU_LOCK_WAIT_TIMEOUT):
        process = None
        try:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            # 日志写临时文件而非 PIPE，避免整首歌的 worker 日志无限积压到父进程内存。
            with (job_dir / "worker.log").open("w+") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(PROJECT_DIR / "worker.py"),
                        "--input-json",
                        str(request),
                        "--output-dir",
                        str(job_dir / "result"),
                    ],
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=REQUEST_TIMEOUT)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError("MSST worker 推理超时。") from exc
                if process.returncode:
                    log.flush()
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 4000))
                    raise RuntimeError("MSST worker 失败：" + log.read()[-2000:])
        finally:
            try:
                stop_worker(process)
            finally:
                if CUDA_RELEASE_DELAY:
                    time.sleep(CUDA_RELEASE_DELAY)


def persist_archive(result_dir: Path, stems: list[str], batch_id: str) -> Path:
    """只归档预期的普通文件，完整写入后原子发布 ZIP。"""
    names = [f"{stem}.wav" for stem in [*stems, "residual"]] + ["manifest.json"]
    for name in names:
        path = result_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"worker 缺少有效结果：{name}")
    destination = OUTPUT_DIR / f"msst_{batch_id}.zip"
    temporary = OUTPUT_DIR / f".{batch_id}.tmp"
    try:
        # WAV FLOAT 不作 ZIP 压缩，避免长音频压缩额外占用 CPU。
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                archive.write(result_dir / name, arcname=name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def process_audio(source: Path, stems: list[str], chunk_seconds: int, digest: str) -> Path:
    """同步完成 CPU 预检、分轨和提交；所有临时文件在成功或异常后删除。"""
    with tempfile.TemporaryDirectory(dir=WORKER_TMP_DIR, prefix="msst_") as directory:
        job_dir = Path(directory)
        normalized = job_dir / "input.wav"
        normalize_audio(source, normalized)
        estimated_bytes = normalized.stat().st_size * (len(stems) + 1)
        if estimated_bytes > MAX_OUTPUT_BYTES:
            raise ValueError(
                "预计分轨结果超过 MSST_MAX_OUTPUT_BYTES，请减少声部、缩短输入或调整容量配置。"
            )
        (job_dir / "result").mkdir()
        payload = {
            "audio_path": str(normalized),
            "source_dir": str(SOURCE_DIR),
            "model_dir": str(MODEL_DIR),
            "device": DEVICE,
            "stems": stems,
            "chunk_seconds": chunk_seconds,
            "input_sha256": digest,
        }
        execute_worker(payload, job_dir)
        return persist_archive(job_dir / "result", stems, uuid.uuid4().hex)


@app.post("/v1/msst/separate")
async def separate_route(
    audio: UploadFile = File(...),
    stems: str = Form(...),
    chunk_seconds: int = Form(DEFAULT_CHUNK_SECONDS),
) -> FileResponse:
    """上传歌曲，返回选定声部、residual 和元数据 ZIP，不暴露服务器任意路径。"""
    staged = None
    try:
        if chunk_seconds not in (5, 10, 20):
            raise ValueError("chunk_seconds 仅支持 5、10、20，默认 10 秒。")
        status = await run_in_threadpool(runtime_status)
        if not status["ready"]:
            raise HTTPException(status_code=503, detail=status)
        selected = parse_stems(stems, status["supported_stems"])
        staged = await stage_audio_upload(
            audio, UPLOAD_DIR, policy=UploadPolicy(max_bytes=MAX_UPLOAD_BYTES)
        )
        archive = await run_in_threadpool(
            process_audio, staged.path, selected, chunk_seconds, staged.sha256
        )
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=archive.name,
            headers={"X-MSST-Batch-ID": archive.stem[5:], "X-MSST-Audio-SHA256": staged.sha256},
        )
    except HTTPException:
        raise
    except AudioUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except (GpuLockTimeoutError, GpuQuarantinedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ValueError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("MSST 分轨请求失败")
        raise HTTPException(status_code=500, detail="MSST 分轨失败，请检查服务日志。") from exc
    finally:
        await audio.close()
        if staged:
            staged.path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
