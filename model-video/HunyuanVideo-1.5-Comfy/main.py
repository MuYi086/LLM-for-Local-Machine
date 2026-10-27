#!/usr/bin/env python3
"""独立 HunyuanVideo-1.5 ComfyUI 图生视频 HTTP 服务。"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    UploadPolicy,
    commit_plain_upload,
    stage_audio_upload,
    storage_disk_status,
)
from llm_local_runtime import gpu_runtime_lock as shared_gpu_runtime_lock
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from hunyuan_runtime import (
    cuda_status,
    expand_path,
    image_info,
    image_storage_name,
    persist_video,
    positive_float,
    terminate_process_group,
    validate_video,
)

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]

STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv("HUNYUAN_VIDEO_COMFY_MODEL_DIR", str(Path(HF_MIRROR_DIR) / "HunyuanVideo-1.5-Comfy"))
)
COMFYUI_CODE_PATH = expand_path(os.getenv("HUNYUAN_VIDEO_COMFY_CODE_PATH", "~/LLM-depency/ComfyUI"))
IMAGE_DIR = Path(
    expand_path(os.getenv("HUNYUAN_VIDEO_COMFY_IMAGE_DIR", str(STORAGE_DIR / "hunyuan/images")))
)
OUTPUT_DIR = Path(
    expand_path(os.getenv("HUNYUAN_VIDEO_COMFY_OUTPUT_DIR", str(STORAGE_DIR / "video")))
)
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv(
            "HUNYUAN_VIDEO_COMFY_WORKER_TMP_DIR",
            str(RUNTIME_CACHE_DIR / "hunyuan_video_comfy_worker"),
        )
    )
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
FFPROBE_BIN = os.getenv("HUNYUAN_VIDEO_COMFY_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("HUNYUAN_VIDEO_COMFY_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("HUNYUAN_VIDEO_COMFY_PORT", os.getenv("PORT", "8393")))
REQUEST_TIMEOUT = float(os.getenv("HUNYUAN_VIDEO_COMFY_REQUEST_TIMEOUT", "2400"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
CUDA_VISIBLE_DEVICES = os.getenv("HUNYUAN_VIDEO_COMFY_CUDA_VISIBLE_DEVICES", "0")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}
DEFAULT_STEPS = int(os.getenv("HUNYUAN_VIDEO_COMFY_STEPS", "8"))
DEFAULT_FRAMES = int(os.getenv("HUNYUAN_VIDEO_COMFY_FRAMES", "121"))
DEFAULT_SEED = int(os.getenv("HUNYUAN_VIDEO_COMFY_SEED", "42"))

REQUIRED_MODEL_FILES = {
    "diffusion_model": Path(MODEL_DIR)
    / "split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
    "qwen_encoder": Path(MODEL_DIR)
    / "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "byt5_encoder": Path(MODEL_DIR)
    / "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
    "clip_vision": Path(MODEL_DIR)
    / "split_files/clip_vision/sigclip_vision_patch14_384.safetensors",
    "vae": Path(MODEL_DIR) / "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
}
REQUIRED_COMFYUI_FILES = (
    Path(COMFYUI_CODE_PATH) / "main.py",
    Path(COMFYUI_CODE_PATH) / "nodes.py",
    Path(COMFYUI_CODE_PATH) / "comfy" / "model_management.py",
    Path(COMFYUI_CODE_PATH) / "comfy_extras" / "nodes_video.py",
)
REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "einops",
    "transformers",
    "safetensors",
    "sentencepiece",
    "aiohttp",
    "yaml",
    "PIL",
    "av",
    "torchsde",
)

for name, value in (
    ("HUNYUAN_VIDEO_COMFY_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
):
    positive_float(name, value)
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 不能为负数。")
if not CUDA_VISIBLE_DEVICES or "," in CUDA_VISIBLE_DEVICES:
    raise ValueError("HunyuanVideo-1.5 Comfy 必须指定单个可见 CUDA 设备。")
if DEFAULT_STEPS not in {8, 12} or DEFAULT_FRAMES not in {49, 73, 97, 121}:
    raise ValueError("默认采样步数必须是 8/12，帧数必须是 49/73/97/121。")
if not 0 <= DEFAULT_SEED <= 2**64 - 1:
    raise ValueError("HUNYUAN_VIDEO_COMFY_SEED 必须在 uint64 范围内。")

for directory in (IMAGE_DIR, OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_STAGING_DIR, RUNTIME_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend HunyuanVideo-1.5 Comfy API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HunyuanVideoRequest(BaseModel):
    """使用上传时逻辑路径的 HunyuanVideo-1.5 图生视频请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12_000)
    image_path: str = Field(min_length=1, max_length=1024)
    negative_prompt: str = Field(default="", max_length=4_000)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**64 - 1)
    steps: Literal[8, 12] = DEFAULT_STEPS
    num_frames: Literal[49, 73, 97, 121] = DEFAULT_FRAMES


def module_available(module_name: str) -> bool:
    """检查 Python 模块规格，不导入可能初始化 CUDA 的运行时。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def model_files_ready() -> bool:
    """确认 480p I2V 模型及全部必需辅助权重均已在本地准备。"""
    return all(path.is_file() for path in REQUIRED_MODEL_FILES.values())


def comfyui_code_ready() -> bool:
    """检查 ComfyUI 核心与视频保存节点源码。"""
    return all(path.is_file() for path in REQUIRED_COMFYUI_FILES)


def image_path(full_path: str) -> Path:
    """将 WebUI 逻辑路径映射到受控上传目录，拒绝服务器任意路径读取。"""
    name = image_storage_name(full_path)
    return IMAGE_DIR / name


def gpu_runtime_lock(label: str):
    """复用共享的有界 GPU 队列，避免与其他本地模型同时推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """在释放共享 GPU 锁前等待 ComfyUI 退出并归还显存。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def store_uploaded_image(staged, full_path: str) -> dict[str, Any]:
    """原子提交按块暂存的图片。"""
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


class HunyuanVideoWorkerManager:
    """启动一次性 ComfyUI worker 并提交最终 MP4。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: HunyuanVideoRequest) -> dict[str, Any]:
        """完成图片、ComfyUI、权重及运行依赖预检。"""
        reference_image = image_path(request.image_path)
        if not reference_image.is_file():
            raise FileNotFoundError(f"请先上传参考图片: {request.image_path}")
        missing = [str(path) for path in REQUIRED_MODEL_FILES.values() if not path.is_file()]
        missing.extend(str(path) for path in REQUIRED_COMFYUI_FILES if not path.is_file())
        missing_modules = [name for name in REQUIRED_MODULES if not module_available(name)]
        if missing or missing_modules:
            detail = []
            if missing:
                detail.append("缺少文件: " + ", ".join(missing))
            if missing_modules:
                detail.append("缺少依赖: " + ", ".join(missing_modules))
            raise RuntimeError("HunyuanVideo-1.5 Comfy 本地运行环境不完整；" + "；".join(detail))
        image_info(reference_image)
        if not Path(WORKER_SCRIPT).is_file():
            raise RuntimeError(f"HunyuanVideo-1.5 worker 脚本不存在: {WORKER_SCRIPT}")
        if not shutil_which(FFPROBE_BIN):
            raise RuntimeError("HunyuanVideo-1.5 Comfy 需要可用的 ffprobe。")
        return {
            **request.model_dump(),
            "reference_image_path": str(reference_image),
            "model_dir": MODEL_DIR,
            "comfyui_code_path": COMFYUI_CODE_PATH,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "request_timeout": REQUEST_TIMEOUT,
            "ffprobe_bin": shutil_which(FFPROBE_BIN),
        }

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """在隔离进程组内执行 ComfyUI，校验后原子保存 MP4。"""
        python_executable = sys.executable
        if not python_executable or not Path(python_executable).is_file():
            raise RuntimeError("未找到 HunyuanVideo-1.5 uv 环境的 Python 解释器。")
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="hunyuanvideo15_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="hunyuanvideo15_out_",
            suffix=".mp4",
        )
        os.close(request_fd)
        os.close(output_fd)
        process: subprocess.Popen | None = None
        try:
            with open(request_path, "w", encoding="utf-8") as request_file:
                json.dump(payload, request_file, ensure_ascii=False)
                request_file.flush()
                os.fsync(request_file.fileno())
            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(
                [
                    python_executable,
                    WORKER_SCRIPT,
                    "--input-json",
                    request_path,
                    "--output-video",
                    output_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            started = time.perf_counter()
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "HunyuanVideo-1.5 Comfy")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"HunyuanVideo-1.5 Comfy worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc
            if stdout.strip():
                LOGGER.info("HunyuanVideo-1.5 worker: %s", stdout.rstrip())
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "HunyuanVideo-1.5 worker 退出码=%s，耗时 %.2fs",
                process.returncode,
                elapsed,
            )
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4000:] or "HunyuanVideo-1.5 worker 执行失败。")
            temporary_video = Path(output_path)
            validate_video(temporary_video, str(payload["ffprobe_bin"]))
            saved = persist_video(temporary_video, OUTPUT_DIR)
            self.last_output_path = str(saved)
            self.last_error = None
            return saved
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "HunyuanVideo-1.5 Comfy")
            Path(request_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


def shutil_which(program: str) -> str | None:
    """返回可执行文件的绝对路径。"""
    import shutil

    resolved = shutil.which(program)
    return str(Path(resolved).resolve()) if resolved else None


manager = HunyuanVideoWorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """返回 API、模型、ComfyUI、依赖及 GPU 状态，不加载模型。"""
    cuda = cuda_status()
    available_modules = {name: module_available(name) for name in REQUIRED_MODULES}
    model_files = {name: path.is_file() for name, path in REQUIRED_MODEL_FILES.items()}
    native_ready = (
        all(model_files.values())
        and comfyui_code_ready()
        and all(available_modules.values())
        and Path(WORKER_SCRIPT).is_file()
        and bool(shutil_which(FFPROBE_BIN))
        and cuda["available"]
    )
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "comfyui_code_path": COMFYUI_CODE_PATH,
            "image_dir": str(IMAGE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
        },
        "available": {
            "worker_script": Path(WORKER_SCRIPT).is_file(),
            "model_files": model_files,
            "comfyui_code": comfyui_code_ready(),
            "dependencies": available_modules,
            "ffprobe": bool(shutil_which(FFPROBE_BIN)),
            "cuda": cuda["available"],
            "selected_provider": "comfyui",
            "selected_provider_ready": native_ready,
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> one ComfyUI process -> exit",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "model": "480p image-to-video step-distilled FP8",
            "default_steps": DEFAULT_STEPS,
            "supported_steps": [8, 12],
            "default_frames": DEFAULT_FRAMES,
            "supported_frames": [49, 73, 97, 121],
            "frame_rate": 24,
            "resolution": "848x480",
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"hunyuan_video_comfy": manager.last_error},
    }


@app.post("/v1/upload_image")
async def upload_image(image: UploadFile = File(...), full_path: str = Form(...)):
    """按块暂存并原子保存参考图片，full_path 只作为逻辑标识。"""
    try:
        image_path(full_path)
        base_policy = UploadPolicy.from_environment()
        policy = UploadPolicy(
            max_bytes=base_policy.max_bytes,
            allowed_suffixes=frozenset({".png", ".jpg", ".jpeg", ".webp"}),
            allowed_content_types=frozenset(
                {"image/png", "image/jpeg", "image/webp", "application/octet-stream"}
            ),
        )
        staged = await stage_audio_upload(image, UPLOAD_STAGING_DIR, policy=policy)
        return await run_in_threadpool(store_uploaded_image, staged, full_path)
    except (AudioUploadError, ValueError) as exc:
        return JSONResponse(
            status_code=getattr(exc, "status_code", 422), content={"detail": str(exc)}
        )


@app.get("/v1/check/image")
def check_image_exists(file_name: str) -> dict[str, Any]:
    """检查逻辑路径对应的上传图片。"""
    try:
        exists = image_path(file_name).is_file()
    except ValueError:
        exists = False
    return {"code": 200 if exists else 404, "exists": exists}


@app.post("/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy")
def generate(request: HunyuanVideoRequest):
    """串行调用 ComfyUI 生成 480p 视频并返回 MP4。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("hunyuan_video_comfy/generate") as metrics:
            try:
                started = time.perf_counter()
                saved = manager.run_worker(payload)
                metrics.worker_seconds = time.perf_counter() - started
                return FileResponse(saved, media_type="video/mp4", filename=saved.name)
            finally:
                wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("HunyuanVideo-1.5 generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   LLM Local Backend HunyuanVideo-1.5 Comfy")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] ComfyUI 源码: {COMFYUI_CODE_PATH}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, cuda_visible_devices={CUDA_VISIBLE_DEVICES}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
