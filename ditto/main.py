#!/usr/bin/env python3
"""独立 Ditto 音频驱动人像视频 HTTP 服务。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    UploadPolicy,
    commit_plain_upload,
    stage_audio_upload,
    storage_disk_status,
)
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from ditto_runtime import (
    cuda_status,
    persist_video,
    terminate_process_group,
    validate_inputs,
    validate_video,
)

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


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
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv("DITTO_MODEL_DIR", str(Path(HF_MIRROR_DIR) / "thewintersun/ditto-talkinghead"))
)
CODE_PATH = expand_path(os.getenv("DITTO_CODE_PATH", "~/tts-depency/ditto-talkinghead"))
DATA_ROOT = expand_path(os.getenv("DITTO_DATA_ROOT", str(Path(MODEL_DIR) / "ditto_onnx")))
CONFIG_PATH = expand_path(
    os.getenv("DITTO_CONFIG_PATH", str(Path(MODEL_DIR) / "ditto_cfg/v0.4_hubert_cfg_trt.pkl"))
)
IMAGE_DIR = Path(expand_path(os.getenv("DITTO_IMAGE_DIR", str(STORAGE_DIR / "ditto/images"))))
OUTPUT_DIR = Path(expand_path(os.getenv("DITTO_OUTPUT_DIR", str(STORAGE_DIR / "video"))))
WORKER_TMP_DIR = expand_path(
    os.getenv("DITTO_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "ditto_worker"))
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
FFMPEG_BIN = os.getenv("DITTO_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("DITTO_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("DITTO_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("DITTO_PORT", os.getenv("PORT", "8392")))
REQUEST_TIMEOUT = float(os.getenv("DITTO_REQUEST_TIMEOUT", "1800"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
CUDA_VISIBLE_DEVICES = os.getenv("DITTO_CUDA_VISIBLE_DEVICES", "0")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}
MAX_AUDIO_DURATION = float(os.getenv("DITTO_MAX_AUDIO_DURATION", "300"))
DEFAULT_SEED = int(os.getenv("DITTO_SEED", "42"))
DEFAULT_SAMPLING_TIMESTEPS = int(os.getenv("DITTO_SAMPLING_TIMESTEPS", "50"))
DEFAULT_MAX_SIZE = int(os.getenv("DITTO_MAX_SIZE", "1920"))

if REQUEST_TIMEOUT <= 0 or MAX_AUDIO_DURATION <= 0:
    raise ValueError("Ditto 超时和音频时长上限必须大于 0。")
if not CUDA_VISIBLE_DEVICES or "," in CUDA_VISIBLE_DEVICES:
    raise ValueError("Ditto 必须指定单个可见 CUDA 设备。")

for directory in (IMAGE_DIR, OUTPUT_DIR, Path(WORKER_TMP_DIR), RUNTIME_CACHE_DIR / "uploads"):
    directory.mkdir(parents=True, exist_ok=True)
reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
app = FastAPI(title="Unitale Ditto API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TalkingHeadRequest(BaseModel):
    """参考图片与驱动音频使用上传时的 WebUI full_path 标识。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    image_path: str = Field(min_length=1, max_length=1024)
    audio_path: str = Field(min_length=1, max_length=1024)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**32 - 1)
    sampling_timesteps: int = Field(default=DEFAULT_SAMPLING_TIMESTEPS, ge=1, le=100)
    max_size: int = Field(default=DEFAULT_MAX_SIZE, ge=256, le=4096)


def image_path(full_path: str) -> Path:
    """用逻辑路径摘要隔离图片存储，禁止请求直接读取服务器任意路径。"""
    suffix = Path(full_path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("图片 full_path 必须以 PNG、JPG、JPEG 或 WebP 扩展名结尾。")
    digest = hashlib.sha256(full_path.encode("utf-8")).hexdigest()
    return IMAGE_DIR / f"{digest}{suffix}"


# 官方配置中的组件名称与当前镜像的 ONNX 文件一一对应；未使用的 wavlm 不需要加载。
ONNX_COMPONENTS = {
    "insightface_det_cfg": "insightface_det.onnx",
    "landmark106_cfg": "landmark106.onnx",
    "landmark203_cfg": "landmark203.onnx",
    "appearance_extractor_cfg": "appearance_extractor.onnx",
    "motion_extractor_cfg": "motion_extractor.onnx",
    "stitch_network_cfg": "stitch_network.onnx",
    "warp_network_cfg": "warp_network_ori.onnx",
    "decoder_cfg": "decoder.onnx",
    "hubert_cfg": "hubert.onnx",
}


def required_files() -> dict[str, Path]:
    """只做文件预检，HTTP 进程不读取 pickle 或加载 ONNX 模型。"""
    return {
        "worker": Path(WORKER_SCRIPT),
        "config": Path(CONFIG_PATH),
        "inference": Path(CODE_PATH) / "inference.py",
        "pipeline": Path(CODE_PATH) / "stream_pipeline_offline.py",
        "loader": Path(CODE_PATH) / "core/utils/load_model.py",
        **{name: Path(DATA_ROOT) / filename for name, filename in ONNX_COMPONENTS.items()},
        "blaze_face": Path(DATA_ROOT) / "blaze_face.onnx",
        "face_mesh": Path(DATA_ROOT) / "face_mesh.onnx",
        "audio2motion": Path(DATA_ROOT) / "lmdm_v0.4_hubert.onnx",
    }


def gpu_runtime_lock(label: str):
    """复用所有本地模型服务共享的有界 GPU 队列与显存采样。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release() -> None:
    """worker 退出后等待驱动归还显存，等待期间继续持有 GPU 锁。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


class TalkingHeadWorkerManager:
    """执行单次 worker，并在所有退出分支清理请求与中间视频。"""

    def __init__(self) -> None:
        self.last_error: str | None = None
        self.last_output_path: str | None = None
        self.last_metrics: dict[str, Any] | None = None

    def build_worker_payload(self, request: TalkingHeadRequest) -> dict[str, Any]:
        """先完成输入和依赖的 CPU/文件预检，再允许进入 GPU 队列。"""
        image = image_path(request.image_path)
        audio = reference_store.prompt_audio_path(request.audio_path)
        for path in (image, audio):
            if not path.is_file():
                raise FileNotFoundError(f"请先上传参考图片和驱动音频: {path.name}")
        missing = [str(path) for path in required_files().values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Ditto 本地源码/权重不完整: {', '.join(missing)}")
        if not shutil.which(FFMPEG_BIN) or not shutil.which(FFPROBE_BIN):
            raise RuntimeError("Ditto 需要可用的 ffmpeg 和 ffprobe。")
        validate_inputs(image, audio, FFPROBE_BIN, MAX_AUDIO_DURATION)
        return {
            **request.model_dump(),
            "reference_image_path": str(image),
            "reference_audio_path": str(audio),
            "model_dir": MODEL_DIR,
            "code_path": CODE_PATH,
            "data_root": DATA_ROOT,
            "config_path": CONFIG_PATH,
            "onnx_components": ONNX_COMPONENTS,
            "ffmpeg_bin": str(Path(shutil.which(FFMPEG_BIN)).resolve()),
            "local_files_only": LOCAL_FILES_ONLY,
        }

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """用本服务解释器启动一个隔离 worker，验证并原子保存成品。"""
        process: subprocess.Popen | None = None
        with tempfile.TemporaryDirectory(dir=WORKER_TMP_DIR, prefix="ditto_") as job:
            request_path = Path(job) / "request.json"
            output_path = Path(job) / "output.mp4"
            try:
                request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                worker_env = os.environ.copy()
                # 服务固定单 GPU，不能继承调用方 torchrun 的分布式配置。
                for key in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE"):
                    worker_env.pop(key, None)
                worker_env["WORLD_SIZE"] = "1"
                worker_env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
                worker_env["HF_HOME"] = HF_MIRROR_DIR
                worker_env["NUMBA_CACHE_DIR"] = str(RUNTIME_CACHE_DIR / "ditto_numba")
                if LOCAL_FILES_ONLY:
                    worker_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        WORKER_SCRIPT,
                        "--input-json",
                        str(request_path),
                        "--output-video",
                        str(output_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                    env=worker_env,
                )
                try:
                    stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(f"Ditto worker 超时（>{REQUEST_TIMEOUT:g}s）") from exc
                if process.returncode != 0:
                    excerpt = " | ".join((stderr or stdout).strip().splitlines()[-8:])
                    raise RuntimeError(excerpt or "Ditto worker 执行失败。")
                validate_video(output_path, FFPROBE_BIN)
                saved = persist_video(output_path, OUTPUT_DIR)
                self.last_output_path = str(saved)
                self.last_error = None
                return saved
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                terminate_process_group(process, "Ditto")


manager = TalkingHeadWorkerManager()


def module_available(name: str) -> bool:
    """只查询模块规格，HTTP 进程不导入 Torch 或官方模型包。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@app.get("/v1/health")
def health():
    """报告本地权重、源码、依赖和存储状态，不加载模型。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "code_path": CODE_PATH,
            "data_root": DATA_ROOT,
            "config_path": CONFIG_PATH,
            "onnx_components": ONNX_COMPONENTS,
            "image_dir": str(IMAGE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "model": all(
                path.is_file()
                for name, path in required_files().items()
                if name not in {"worker", "inference", "pipeline", "loader"}
            ),
            "code": all(
                required_files()[name].is_file() for name in ("inference", "pipeline", "loader")
            ),
            "worker": Path(WORKER_SCRIPT).is_file(),
            "onnxruntime": module_available("onnxruntime"),
            "ffmpeg": bool(shutil.which(FFMPEG_BIN)),
            "ffprobe": bool(shutil.which(FFPROBE_BIN)),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "max_audio_duration": MAX_AUDIO_DURATION,
            "backend": "onnxruntime-cuda",
            "default_sampling_timesteps": DEFAULT_SAMPLING_TIMESTEPS,
            "default_max_size": DEFAULT_MAX_SIZE,
            "default_seed": DEFAULT_SEED,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        },
        "last_errors": {"ditto": manager.last_error},
        "last_metrics": manager.last_metrics,
    }


def store_uploaded_audio(staged, full_path: str, prompt_text: str | None):
    """复用音色去重与原子上传提交规则。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...), full_path: str = Form(...), prompt_text: str | None = Form(None)
):
    """按块暂存驱动音频，在线程池中提交上传。"""
    try:
        staged = await stage_audio_upload(audio, RUNTIME_CACHE_DIR / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


def store_uploaded_image(staged, full_path: str):
    """原子保存图片，失败时删除暂存文件。"""
    try:
        target = image_path(full_path)
        commit_plain_upload(staged, target)
        return {
            "code": 200,
            "filename": full_path,
            "sha256": staged.sha256,
            "size_bytes": staged.size_bytes,
        }
    finally:
        staged.path.unlink(missing_ok=True)


@app.post("/v1/upload_image")
async def upload_image(image: UploadFile = File(...), full_path: str = Form(...)):
    """复用无模型运行时的流式暂存，为图片提供独立媒体策略。"""
    try:
        image_path(full_path)
        policy = UploadPolicy(
            max_bytes=UploadPolicy.from_environment().max_bytes,
            allowed_suffixes=frozenset({".png", ".jpg", ".jpeg", ".webp"}),
            allowed_content_types=frozenset(
                {"image/png", "image/jpeg", "image/webp", "application/octet-stream"}
            ),
        )
        staged = await stage_audio_upload(image, RUNTIME_CACHE_DIR / "uploads", policy=policy)
        return await run_in_threadpool(store_uploaded_image, staged, full_path)
    except (AudioUploadError, ValueError) as exc:
        return JSONResponse(
            status_code=getattr(exc, "status_code", 422), content={"detail": str(exc)}
        )


@app.get("/v1/check/image")
def check_image_exists(file_name: str):
    """检查 WebUI 逻辑路径对应的参考图片。"""
    try:
        exists = image_path(file_name).is_file()
    except ValueError:
        exists = False
    return {"code": 200 if exists else 404, "exists": exists}


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查驱动音频及参考文本 sidecar。"""
    exists = reference_store.prompt_audio_path(file_name).is_file()
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(reference_store.load_prompt_text(file_name)),
    }


@app.post("/v1/ditto/talkingHead")
def generate(request: TalkingHeadRequest):
    """串行生成音频驱动人像视频，保存成品并以流式文件响应返回 MP4。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        with gpu_runtime_lock("ditto/talkingHead") as metrics:
            try:
                started = time.perf_counter()
                saved = manager.run_worker(payload)
                return FileResponse(saved, media_type="video/mp4", filename=saved.name)
            finally:
                metrics.worker_seconds = time.perf_counter() - started
                wait_after_cuda_release()
                manager.last_metrics = metrics.as_dict()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Ditto generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
