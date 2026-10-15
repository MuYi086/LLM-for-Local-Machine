#!/usr/bin/env python3
"""YuE2-Vae 独立 latent 编码与音频解码 HTTP 服务。"""

from __future__ import annotations

# API 进程只处理文件、预检和 worker 生命周期，不导入 Torch 或自定义模型代码。
import hashlib
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
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
MODEL_DIR = expand_path(os.getenv("YUE2_VAE_MODEL_DIR", str(HF_MIRROR_DIR / "m-a-p/YuE2-Vae")))
OUTPUT_DIR = expand_path(os.getenv("YUE2_VAE_OUTPUT_DIR", str(STORAGE_DIR / "yue2_vae")))
UPLOAD_STAGING_DIR = expand_path(
    os.getenv("YUE2_VAE_UPLOAD_STAGING_DIR", str(RUNTIME_CACHE_DIR / "yue2_vae_uploads"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv("YUE2_VAE_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "yue2_vae_worker"))
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
DEVICE = os.getenv("YUE2_VAE_DEVICE", "cuda")
REQUEST_TIMEOUT = float(os.getenv("YUE2_VAE_REQUEST_TIMEOUT", "900"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
MAX_AUDIO_SECONDS = float(os.getenv("YUE2_VAE_MAX_AUDIO_SECONDS", "600"))
MAX_LATENT_FRAMES = int(os.getenv("YUE2_VAE_MAX_LATENT_FRAMES", "15000"))
MAX_UPLOAD_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(64 * 1024 * 1024)))
FFMPEG_BIN = os.getenv("YUE2_VAE_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("YUE2_VAE_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("YUE2_VAE_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("YUE2_VAE_PORT", os.getenv("PORT", "8316")))

for label, value in (
    ("YUE2_VAE_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
    ("YUE2_VAE_MAX_AUDIO_SECONDS", MAX_AUDIO_SECONDS),
):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} 必须是正的有限数。")
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
if MAX_LATENT_FRAMES < 1 or MAX_UPLOAD_BYTES < 1:
    raise ValueError("YUE2_VAE_MAX_LATENT_FRAMES 和 UPLOAD_MAX_BYTES 必须为正整数。")

os.environ.setdefault("HF_HOME", str(HF_MIRROR_DIR))
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "hf_modules"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (
    OUTPUT_DIR,
    UPLOAD_STAGING_DIR,
    WORKER_TMP_DIR,
    RUNTIME_CACHE_DIR / "hf_modules",
    Path(GPU_LOCK_FILE).parent,
):
    directory.mkdir(parents=True, exist_ok=True)

REQUIRED_MODEL_FILES = (
    "config.json",
    "modeling_vae.py",
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
    "transformers",
)


@dataclass(frozen=True)
class StagedLatent:
    """安全暂存后的 NPY latent 上传信息。"""

    path: Path
    sha256: str
    size_bytes: int


async def stage_latent_upload(upload: UploadFile) -> StagedLatent:
    """按块暂存仅支持的 NPY latent，不将完整上传读入内存。"""
    filename = upload.filename or ""
    if Path(filename).suffix.lower() != ".npy":
        raise AudioUploadError("decode 操作只接受 .npy latent 文件。")
    content_type = (upload.content_type or "").lower().split(";", 1)[0].strip()
    if content_type and content_type not in {
        "application/octet-stream",
        "application/x-npy",
        "binary/octet-stream",
    }:
        raise AudioUploadError(f"不支持的 latent Content-Type: {content_type}")

    descriptor, name = tempfile.mkstemp(
        dir=UPLOAD_STAGING_DIR,
        prefix="latent_",
        suffix=".npy.part",
    )
    path = Path(name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_BYTES:
                    raise AudioUploadError(
                        f"latent 上传超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB 限制。",
                        status_code=413,
                    )
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if size_bytes == 0:
            raise AudioUploadError("latent 文件不能为空。")
        return StagedLatent(path=path, sha256=digest.hexdigest(), size_bytes=size_bytes)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def model_status() -> dict[str, object]:
    """检查本地模型配置、代码和权重清单，不加载模型。"""
    missing = [name for name in REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    manifest_error: str | None = None
    manifest_path = MODEL_DIR / "weights_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                manifest_error = "weights_manifest.json 缺少 files 映射"
            else:
                for name, metadata in files.items():
                    weight_path = MODEL_DIR / name
                    if not weight_path.is_file():
                        missing.append(name)
                    elif isinstance(metadata, dict) and weight_path.stat().st_size != metadata.get(
                        "bytes"
                    ):
                        manifest_error = f"权重文件大小与清单不匹配: {name}"
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            manifest_error = f"无法读取 weights_manifest.json: {exc}"

    return {
        "path": str(MODEL_DIR),
        "complete": not missing and manifest_error is None,
        "missing": sorted(set(missing)),
        "error": manifest_error,
    }


def validate_latent(path: Path) -> int:
    """在排队 GPU 前验证 NPY 类型、形状、有限值和时长上限。"""
    import numpy as np

    try:
        latent = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"无法读取 NPY latent 文件: {exc}") from exc
    if (
        latent.ndim != 2
        or latent.shape[1] != 64
        or latent.shape[0] < 1
        or latent.shape[0] > MAX_LATENT_FRAMES
    ):
        raise ValueError(f"latent 必须为 [T, 64]，且 1 <= T <= {MAX_LATENT_FRAMES}。")
    if not np.issubdtype(latent.dtype, np.floating):
        raise ValueError("latent 数组必须使用浮点类型。")
    if not np.isfinite(latent).all():
        raise ValueError("latent 数组包含 NaN 或无穷值。")
    return int(latent.shape[0])


def inspect_audio_duration(audio_path: Path) -> float:
    """使用 ffprobe 在 CPU 预检音频时长，避免超长请求占用 GPU。"""
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
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"音频时长超过 {MAX_AUDIO_SECONDS:g} 秒限制。")
    if duration * 48_000 < 1_920:
        raise ValueError("音频至少需要 0.04 秒才能编码为 latent。")
    return duration


def normalize_audio(audio_path: Path) -> Path:
    """用 FFmpeg 转成模型要求的 48 kHz 双声道 float32 WAV。"""
    executable = shutil.which(FFMPEG_BIN)
    if not executable:
        raise FileNotFoundError(f"未找到 ffmpeg: {FFMPEG_BIN}")
    descriptor, name = tempfile.mkstemp(
        dir=WORKER_TMP_DIR,
        prefix="yue2_vae_audio_",
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
                "2",
                "-ar",
                "48000",
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


def atomic_persist_output(source: Path, operation: str) -> Path:
    """将 worker 输出原子写入服务目录，并清理临时结果。"""
    suffix = ".wav" if operation == "decode" else ".npy"
    destination = OUTPUT_DIR / f"yue2_vae_{operation}_{uuid.uuid4().hex}{suffix}"
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


class YuE2VaeWorkerManager:
    """在本项目 uv 环境中启动并清理一次性 VAE worker。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def build_payload(
        self,
        operation: str,
        *,
        audio_path: Path | None = None,
        latent_path: Path | None = None,
        sample: bool = False,
        seed: int | None = None,
    ) -> dict[str, object]:
        """完成模型与依赖预检，构造本地离线 worker 请求。"""
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"YuE2-Vae worker 不存在: {WORKER_SCRIPT}")
        model = model_status()
        if not model["complete"]:
            raise FileNotFoundError(f"YuE2-Vae 本地模型不完整: {model}")
        missing = [name for name in REQUIRED_DEPENDENCIES if not module_available(name)]
        if missing:
            raise FileNotFoundError(
                "YuE2-Vae 依赖未安装: "
                f"{', '.join(missing)}。请先 uv sync --project YuE2-Vae --locked。"
            )
        if DEVICE.startswith("cuda") and not cuda_status()["available"]:
            raise FileNotFoundError("YUE2_VAE_DEVICE 使用 CUDA，但未检测到可用 NVIDIA GPU。")

        payload: dict[str, object] = {
            "device": DEVICE,
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "max_latent_frames": MAX_LATENT_FRAMES,
            "model_dir": str(MODEL_DIR),
            "operation": operation,
        }
        if operation == "decode" and latent_path is not None:
            payload["latent_path"] = str(latent_path)
        elif operation == "encode" and audio_path is not None:
            payload["audio_path"] = str(audio_path)
            payload["sample"] = sample
            payload["seed"] = seed
        else:
            raise ValueError("operation 与输入文件不匹配。")
        return payload

    def run_worker(self, payload: dict[str, object]) -> Path:
        """运行一次 worker，确保成功、失败和超时都会终止进程组。"""
        request_fd, request_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="yue2_vae_request_",
            suffix=".json",
        )
        suffix = ".wav" if payload["operation"] == "decode" else ".npy"
        output_fd, output_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="yue2_vae_output_",
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
                raise TimeoutError(f"YuE2-Vae worker 超过 {REQUEST_TIMEOUT:g} 秒时限。") from exc

            if stdout.strip():
                LOGGER.info("YuE2-Vae worker: %s", stdout.rstrip())
            if process.returncode != 0:
                excerpt = (stderr or stdout).strip()[-4_000:]
                raise RuntimeError(excerpt or "YuE2-Vae worker 失败。")
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("YuE2-Vae worker 未生成输出文件。")
            succeeded = True
            self.last_error = None
            return output_path
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "YuE2-Vae")
            request_path.unlink(missing_ok=True)
            if not succeeded:
                output_path.unlink(missing_ok=True)


def execute_worker(payload: dict[str, object]) -> Path:
    """在共享 GPU 锁内运行 worker，并在释放锁前等待显存归还。"""
    with shared_gpu_runtime_lock(
        GPU_LOCK_FILE,
        f"yue2_vae/{payload['operation']}",
        timeout_seconds=GPU_LOCK_WAIT_TIMEOUT,
    ):
        with manager.lock:
            try:
                return manager.run_worker(payload)
            finally:
                if CUDA_RELEASE_DELAY > 0:
                    time.sleep(CUDA_RELEASE_DELAY)


app = FastAPI(title="Unitale YuE2-Vae API")
manager = YuE2VaeWorkerManager()


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
    """报告模型、依赖、FFmpeg 和 GPU 状态，不加载权重。"""
    model = model_status()
    dependencies = {name: module_available(name) for name in REQUIRED_DEPENDENCIES}
    ffmpeg = shutil.which(FFMPEG_BIN) is not None
    ffprobe = shutil.which(FFPROBE_BIN) is not None
    cuda = cuda_status()
    ready = (
        model["complete"]
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
            "model": model,
            "dependencies": dependencies,
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "device": DEVICE,
            "local_files_only": LOCAL_FILES_ONLY,
            "worker_runtime": "uv",
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "decode_output": "48 kHz stereo PCM_24 WAV",
            "encode_output": "float32 NPY with shape [T, 64]",
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "max_latent_frames": MAX_LATENT_FRAMES,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "request_timeout": REQUEST_TIMEOUT,
        },
        "last_errors": {"yue2_vae": manager.last_error},
    }


@app.post("/v1/m-a-p/YuE2-Vae")
async def process_vae(
    operation: Literal["encode", "decode"] = Form(...),
    audio: UploadFile | None = File(default=None),
    latent: UploadFile | None = File(default=None),
    sample: bool = Form(False),
    seed: int | None = Form(None),
) -> FileResponse:
    """编码 48 kHz 音频为 latent，或将 YuE2 acoustic latent 解码为 WAV。"""
    if operation == "decode" and (latent is None or audio is not None):
        raise HTTPException(status_code=422, detail="decode 需要且只接受 latent .npy 文件。")
    if operation == "encode" and (audio is None or latent is not None):
        raise HTTPException(status_code=422, detail="encode 需要且只接受 audio 文件。")

    staged_path: Path | None = None
    normalized_path: Path | None = None
    staged_sha256: str | None = None
    latent_frames: int | None = None
    try:
        if operation == "decode":
            try:
                staged = await stage_latent_upload(latent)
                staged_path = staged.path
                staged_sha256 = staged.sha256
                latent_frames = await run_in_threadpool(validate_latent, staged.path)
                payload = await run_in_threadpool(
                    manager.build_payload,
                    operation,
                    latent_path=staged.path,
                )
            except AudioUploadError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        else:
            try:
                staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR)
                staged_path = staged.path
                staged_sha256 = staged.sha256
                await run_in_threadpool(inspect_audio_duration, staged.path)
                normalized_path = await run_in_threadpool(normalize_audio, staged.path)
                payload = await run_in_threadpool(
                    manager.build_payload,
                    operation,
                    audio_path=normalized_path,
                    sample=sample,
                    seed=seed,
                )
            except AudioUploadError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except (ValueError, subprocess.SubprocessError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        output_path = await run_in_threadpool(execute_worker, payload)
        saved_path = await run_in_threadpool(atomic_persist_output, output_path, operation)
        headers = {
            "X-YuE2-Vae-Operation": operation,
            "X-YuE2-Vae-Input-SHA256": staged_sha256 or "",
        }
        if operation == "decode":
            headers["X-YuE2-Vae-Sample-Rate"] = "48000"
            headers["X-YuE2-Vae-Latent-Frames"] = str(latent_frames)
            media_type = "audio/wav"
        else:
            import numpy as np

            latent_result = np.load(saved_path, mmap_mode="r", allow_pickle=False)
            headers["X-YuE2-Vae-Latent-Frames"] = str(latent_result.shape[0])
            media_type = "application/octet-stream"
        return FileResponse(
            saved_path,
            media_type=media_type,
            filename=saved_path.name,
            headers=headers,
        )
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("YuE2-Vae worker failed")
        manager.last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for upload in (audio, latent):
            if upload is not None:
                await upload.close()
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        if normalized_path is not None:
            normalized_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, workers=1)
