#!/usr/bin/env python3
"""MERT-v2-FullSong 音乐表征提取 HTTP 服务。"""

from __future__ import annotations

# API 进程只处理请求、CPU 音频预检和 worker 生命周期，不导入 Torch 或模型代码。
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    execute_localai_worker_file,
    load_gateway_config,
)
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import AudioUploadError, GpuLockTimeoutError, stage_audio_upload
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from runtime import cuda_status, module_available, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(value: str) -> Path:
    """展开环境变量和用户目录，并返回绝对路径。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    """解析常见形式的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = str(
    expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
)
MODEL_DIR = expand_path(
    os.getenv("MERT_V2_FULLSONG_MODEL_DIR", str(HF_MIRROR_DIR / "m-a-p/MERT-v2-FullSong"))
)
OUTPUT_DIR = expand_path(
    os.getenv("MERT_V2_FULLSONG_OUTPUT_DIR", str(STORAGE_DIR / "mert_v2_fullsong"))
)
UPLOAD_STAGING_DIR = expand_path(
    os.getenv(
        "MERT_V2_FULLSONG_UPLOAD_STAGING_DIR",
        str(RUNTIME_CACHE_DIR / "mert_v2_fullsong_uploads"),
    )
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "MERT_V2_FULLSONG_WORKER_TMP_DIR",
        str(RUNTIME_CACHE_DIR / "mert_v2_fullsong_worker"),
    )
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
DEVICE = os.getenv("MERT_V2_FULLSONG_DEVICE", "cuda:0")
REQUEST_TIMEOUT = float(os.getenv("MERT_V2_FULLSONG_REQUEST_TIMEOUT", "900"))
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="UNITALE_MERT_V2_FULLSONG_PROVIDER",
    default_model="unitale.mert-v2-fullsong.worker.v1",
    request_timeout=REQUEST_TIMEOUT,
)
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
MIN_AUDIO_SECONDS = float(os.getenv("MERT_V2_FULLSONG_MIN_AUDIO_SECONDS", "30"))
MAX_AUDIO_SECONDS = float(os.getenv("MERT_V2_FULLSONG_MAX_AUDIO_SECONDS", "360"))
MAX_UPLOAD_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(64 * 1024 * 1024)))
FFMPEG_BIN = os.getenv("MERT_V2_FULLSONG_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("MERT_V2_FULLSONG_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("MERT_V2_FULLSONG_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MERT_V2_FULLSONG_PORT", os.getenv("PORT", "8317")))

for label, value in (
    ("MERT_V2_FULLSONG_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
    ("MERT_V2_FULLSONG_MIN_AUDIO_SECONDS", MIN_AUDIO_SECONDS),
    ("MERT_V2_FULLSONG_MAX_AUDIO_SECONDS", MAX_AUDIO_SECONDS),
):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} 必须是正的有限数。")
if MIN_AUDIO_SECONDS >= MAX_AUDIO_SECONDS:
    raise ValueError("MERT-v2-FullSong 音频时长范围配置无效。")
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
if MAX_UPLOAD_BYTES <= 0:
    raise ValueError("UPLOAD_MAX_BYTES 必须为正整数。")

os.environ.setdefault("HF_HOME", str(HF_MIRROR_DIR))
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "mert_v2_hf_modules"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (
    OUTPUT_DIR,
    UPLOAD_STAGING_DIR,
    WORKER_TMP_DIR,
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(GPU_LOCK_FILE).parent,
):
    directory.mkdir(parents=True, exist_ok=True)

REQUIRED_MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "configuration_mert2.py",
    "modeling_mert2.py",
    "model.safetensors",
    "weights_manifest.json",
)
REQUIRED_DEPENDENCIES = (
    "fastapi",
    "multipart",
    "numpy",
    "safetensors",
    "soundfile",
    "torch",
    "torchaudio",
    "transformers",
)


def model_status() -> dict[str, object]:
    """检查本地 MERT 配置、自定义代码和权重清单，不读取完整权重。"""
    missing = [name for name in REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    manifest_error: str | None = None
    manifest_path = MODEL_DIR / "weights_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            filename = manifest.get("filename")
            weight_path = MODEL_DIR / str(filename or "")
            if not filename or not weight_path.is_file():
                missing.append(str(filename or "weights_manifest.json:filename"))
            elif weight_path.stat().st_size != manifest.get("bytes"):
                manifest_error = f"权重文件大小与 manifest 不匹配: {filename}"
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            manifest_error = f"无法读取 weights_manifest.json: {exc}"

    return {
        "path": str(MODEL_DIR),
        "complete": not missing and manifest_error is None,
        "missing": sorted(set(missing)),
        "error": manifest_error,
    }


def missing_dependencies() -> list[str]:
    """列出当前 uv 项目环境中未安装的运行依赖。"""
    return [name for name in REQUIRED_DEPENDENCIES if not module_available(name)]


def ensure_runtime_ready() -> None:
    """在音频转换和 GPU 排队前检查模型、依赖、worker 与设备。"""
    if not WORKER_SCRIPT.is_file():
        raise FileNotFoundError(f"MERT-v2-FullSong worker 不存在: {WORKER_SCRIPT}")
    model = model_status()
    if not model["complete"]:
        raise FileNotFoundError(f"MERT-v2-FullSong 本地模型不完整: {model}")
    missing = missing_dependencies()
    if missing:
        raise FileNotFoundError(
            "MERT-v2-FullSong 依赖未安装: "
            f"{', '.join(missing)}。请先 uv sync --project MERT-v2-FullSong --locked。"
        )
    if DEVICE.startswith("cuda") and not cuda_status()["available"]:
        raise FileNotFoundError("MERT_V2_FULLSONG_DEVICE 使用 CUDA，但未检测到可用 NVIDIA GPU。")


def inspect_audio_duration(audio_path: Path) -> float:
    """使用 ffprobe 在 GPU 排队前校验完整歌曲的时长范围。"""
    executable = shutil.which(FFPROBE_BIN)
    if not executable:
        raise FileNotFoundError(f"未找到 ffprobe: {FFPROBE_BIN}")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise ValueError("ffprobe 未能读取有效音频时长。") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("音频时长必须是正的有限数。")
    if duration < MIN_AUDIO_SECONDS or duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"MERT-v2-FullSong 仅接受 {MIN_AUDIO_SECONDS:g}–{MAX_AUDIO_SECONDS:g} 秒完整歌曲。"
        )
    return duration


def normalize_audio(audio_path: Path) -> Path:
    """使用 FFmpeg 转换为官方模型要求的 24 kHz 单声道 float32 WAV。"""
    executable = shutil.which(FFMPEG_BIN)
    if not executable:
        raise FileNotFoundError(f"未找到 ffmpeg: {FFMPEG_BIN}")
    descriptor, name = tempfile.mkstemp(
        dir=WORKER_TMP_DIR,
        prefix="mert_v2_fullsong_audio_",
        suffix=".wav",
    )
    os.close(descriptor)
    normalized_path = Path(name)
    try:
        subprocess.run(
            [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_f32le",
                str(normalized_path),
            ],
            check=True,
            capture_output=True,
            timeout=REQUEST_TIMEOUT,
        )
        return normalized_path
    except Exception:
        normalized_path.unlink(missing_ok=True)
        raise


def persist_output(source: Path, output: str) -> Path:
    """将 worker 结果原子提交到服务输出目录。"""
    suffix = ".json" if output == "embedding" else ".npz"
    destination = OUTPUT_DIR / f"mert_v2_fullsong_{output}_{uuid.uuid4().hex}{suffix}"
    temporary = OUTPUT_DIR / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)


def worker_error_excerpt(message: str) -> str:
    """保留 worker 最后的可读错误行，避免 API 返回完整 traceback。"""
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return "\n".join(lines[-8:])[-2_000:] or "MERT-v2-FullSong worker 失败。"


class MertV2FullSongWorkerManager:
    """在本项目 uv 环境中启动并清理一次性特征提取 worker。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def build_payload(self, audio_path: Path, output: str, duration: float) -> dict[str, object]:
        """构造仅包含本地路径和已校验选项的离线 worker 请求。"""
        ensure_runtime_ready()
        return {
            "audio_path": str(audio_path),
            "device": DEVICE,
            "duration_seconds": duration,
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "model_dir": str(MODEL_DIR),
            "output": output,
        }

    def run_worker(self, payload: dict[str, object]) -> Path:
        """运行一次 worker，成功、失败、超时均清理进程组与请求文件。"""
        request_fd, request_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="mert_v2_fullsong_request_",
            suffix=".json",
        )
        suffix = ".json" if payload["output"] == "embedding" else ".npz"
        output_fd, output_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="mert_v2_fullsong_output_",
            suffix=suffix,
        )
        os.close(output_fd)
        request_path = Path(request_name)
        output_path = Path(output_name)
        process: subprocess.Popen[str] | None = None
        succeeded = False
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as request_file:
                json.dump(payload, request_file, ensure_ascii=False)
                request_file.flush()
                os.fsync(request_file.fileno())

            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(WORKER_SCRIPT),
                    "--input-json",
                    str(request_path),
                    "--output-file",
                    str(output_path),
                ],
                cwd=PROJECT_DIR,
                env=worker_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"MERT-v2-FullSong worker 超过 {REQUEST_TIMEOUT:g} 秒时限。"
                ) from exc

            if stdout.strip():
                LOGGER.info("MERT-v2-FullSong worker: %s", stdout.rstrip())
            if process.returncode != 0:
                excerpt = (stderr or stdout).strip()[-4_000:]
                raise RuntimeError(excerpt or "MERT-v2-FullSong worker 失败。")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("MERT-v2-FullSong worker 未生成输出。")
            succeeded = True
            self.last_error = None
            return output_path
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "MERT-v2-FullSong")
            request_path.unlink(missing_ok=True)
            if not succeeded:
                output_path.unlink(missing_ok=True)


def execute_worker(payload: dict[str, object]) -> Path:
    """持有共享 GPU 锁执行 worker，并在解锁前等待 CUDA 资源归还。"""
    if GATEWAY_CONFIG.provider == "localai":
        output = str(payload["output"])
        suffix, media_type = (
            (".json", "application/json")
            if output == "embedding"
            else (".npz", "application/x-npz")
        )
        try:
            artifact, _ = execute_localai_worker_file(
                config=GATEWAY_CONFIG,
                operation="run",
                payload={"worker_input": payload},
                output_dir=Path(WORKER_TMP_DIR),
                suffix=suffix,
                media_type=media_type,
            )
            manager.last_error = None
            return artifact
        except Exception as exc:
            manager.last_error = str(exc)
            raise
    with shared_gpu_runtime_lock(
        GPU_LOCK_FILE,
        "mert_v2_fullsong/extract",
        timeout_seconds=GPU_LOCK_WAIT_TIMEOUT,
    ):
        with manager.lock:
            try:
                return manager.run_worker(payload)
            finally:
                if CUDA_RELEASE_DELAY > 0:
                    LOGGER.info("等待 %.1fs 释放 CUDA 资源", CUDA_RELEASE_DELAY)
                    time.sleep(CUDA_RELEASE_DELAY)


app = FastAPI(title="Unitale MERT-v2-FullSong Feature API")
manager = MertV2FullSongWorkerManager()


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


@app.get("/v1/health")
def health() -> dict[str, object]:
    """返回模型、依赖、媒体工具与 GPU 状态，不加载权重。"""
    model = model_status()
    dependencies = {name: module_available(name) for name in REQUIRED_DEPENDENCIES}
    ffmpeg = shutil.which(FFMPEG_BIN) is not None
    ffprobe = shutil.which(FFPROBE_BIN) is not None
    cuda = cuda_status()
    ready = (
        WORKER_SCRIPT.is_file()
        and model["complete"]
        and all(dependencies.values())
        and ffmpeg
        and ffprobe
        and (cuda["available"] or not DEVICE.startswith("cuda"))
    )
    return {
        "code": 200,
        "ready": ready,
        "paths": {
            "model_dir": str(MODEL_DIR),
            "output_dir": str(OUTPUT_DIR),
            "upload_staging_dir": str(UPLOAD_STAGING_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "worker_script": WORKER_SCRIPT.is_file(),
            "model": model,
            "dependencies": dependencies,
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "device": DEVICE,
            "local_files_only": LOCAL_FILES_ONLY,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "sampling_rate": 24_000,
            "frame_rate": 25,
            "embedding_dimension": 1_024,
            "min_audio_seconds": MIN_AUDIO_SECONDS,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
        },
        "last_errors": {"mert_v2_fullsong": manager.last_error},
        "inference": {
            "provider": GATEWAY_CONFIG.provider,
            "localai_model": GATEWAY_CONFIG.model if GATEWAY_CONFIG.provider == "localai" else None,
        },
    }


@app.post("/v1/m-a-p/MERT-v2-FullSong")
async def extract_features_route(
    audio: UploadFile = File(...),
    output: Literal["embedding", "frames"] = Form("embedding"),
) -> Response:
    """提取完整歌曲的 1024 维 mean embedding 或 25 Hz 逐帧表征。"""
    try:
        await run_in_threadpool(ensure_runtime_ready)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    staged_path: Path | None = None
    normalized_path: Path | None = None
    try:
        try:
            staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR)
            staged_path = staged.path
        except AudioUploadError as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

        try:
            duration = await run_in_threadpool(inspect_audio_duration, staged_path)
            normalized_path = await run_in_threadpool(normalize_audio, staged_path)
            payload = await run_in_threadpool(
                manager.build_payload,
                normalized_path,
                output,
                duration,
            )
            result_path = await run_in_threadpool(execute_worker, payload)
            saved_path = await run_in_threadpool(persist_output, result_path, output)
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (BackendQuarantinedError, LocalAIUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LocalAIRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except TimeoutError as exc:
            LOGGER.exception("MERT-v2-FullSong request timed out")
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except (ValueError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("MERT-v2-FullSong inference failed")
            raise HTTPException(status_code=500, detail=worker_error_excerpt(str(exc))) from exc

        headers = {
            "X-MERT-v2-FullSong-Model": "m-a-p/MERT-v2-FullSong",
            "X-MERT-v2-FullSong-Output": output,
            "X-MERT-v2-FullSong-Audio-SHA256": staged.sha256,
            "X-MERT-v2-FullSong-Sampling-Rate": "24000",
            "X-MERT-v2-FullSong-Frame-Rate": "25",
        }
        if output == "embedding":
            return Response(
                content=saved_path.read_bytes(),
                media_type="application/json",
                headers=headers,
            )
        return FileResponse(
            saved_path,
            media_type="application/octet-stream",
            filename=saved_path.name,
            headers=headers,
        )
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        if normalized_path is not None:
            normalized_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
