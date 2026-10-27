#!/usr/bin/env python3
"""Wan2.2 TI2V 5B GGUF ComfyUI HTTP 服务；HTTP 进程不导入 Torch。"""

from __future__ import annotations

import importlib.util
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from wan22_runtime import (
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
    os.getenv(
        "WAN22_TI2V_MODEL_DIR",
        str(Path(HF_MIRROR_DIR) / "QuantStack/Wan2.2-TI2V-5B-GGUF"),
    )
)
COMFYUI_CODE_PATH = expand_path(os.getenv("WAN22_TI2V_COMFYUI_CODE_PATH", "~/LLM-depency/ComfyUI"))
GGUF_NODE_PATH = expand_path(os.getenv("WAN22_TI2V_GGUF_NODE_PATH", "~/LLM-depency/ComfyUI-GGUF"))
IMAGE_DIR = Path(expand_path(os.getenv("WAN22_TI2V_IMAGE_DIR", str(STORAGE_DIR / "wan22/images"))))
OUTPUT_DIR = Path(expand_path(os.getenv("WAN22_TI2V_OUTPUT_DIR", str(STORAGE_DIR / "video"))))
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv(
            "WAN22_TI2V_WORKER_TMP_DIR",
            str(RUNTIME_CACHE_DIR / "wan22_ti2v_worker"),
        )
    )
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
FFPROBE_BIN = os.getenv("WAN22_TI2V_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("WAN22_TI2V_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("WAN22_TI2V_PORT", os.getenv("PORT", "8392")))
REQUEST_TIMEOUT = float(os.getenv("WAN22_TI2V_REQUEST_TIMEOUT", "3600"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
CUDA_VISIBLE_DEVICES = os.getenv("WAN22_TI2V_CUDA_VISIBLE_DEVICES", "0")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}
DEFAULT_WIDTH = int(os.getenv("WAN22_TI2V_WIDTH", "832"))
DEFAULT_HEIGHT = int(os.getenv("WAN22_TI2V_HEIGHT", "480"))
DEFAULT_LENGTH = int(os.getenv("WAN22_TI2V_LENGTH", "49"))
DEFAULT_STEPS = int(os.getenv("WAN22_TI2V_STEPS", "20"))
DEFAULT_SEED = int(os.getenv("WAN22_TI2V_SEED", "42"))
DEFAULT_FPS = int(os.getenv("WAN22_TI2V_FPS", "24"))

MODEL_FILES = {
    "diffusion_model": Path(MODEL_DIR) / "Wan2.2-TI2V-5B-Q8_0.gguf",
    "text_encoder": Path(MODEL_DIR) / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": Path(MODEL_DIR) / "wan2.2_vae.safetensors",
}
REQUIRED_COMFYUI_FILES = (
    Path(COMFYUI_CODE_PATH) / "main.py",
    Path(COMFYUI_CODE_PATH) / "nodes.py",
    Path(COMFYUI_CODE_PATH) / "comfy" / "model_management.py",
    Path(COMFYUI_CODE_PATH) / "comfy_extras" / "nodes_wan.py",
    Path(COMFYUI_CODE_PATH) / "comfy_extras" / "nodes_video.py",
)
REQUIRED_GGUF_NODE_FILES = (
    Path(GGUF_NODE_PATH) / "__init__.py",
    Path(GGUF_NODE_PATH) / "nodes.py",
    Path(GGUF_NODE_PATH) / "loader.py",
)
REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "torchaudio",
    "torchsde",
    "numpy",
    "einops",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "safetensors",
    "aiohttp",
    "yaml",
    "PIL",
    "av",
    "gguf",
    "google.protobuf",
    "comfy_aimdo",
    "comfy_kitchen",
)

for name, value in (
    ("WAN22_TI2V_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
):
    positive_float(name, value)
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 不能为负数。")
if not CUDA_VISIBLE_DEVICES or "," in CUDA_VISIBLE_DEVICES:
    raise ValueError("Wan2.2 TI2V 服务必须指定单个可见 CUDA 设备。")
if DEFAULT_WIDTH % 32 or DEFAULT_HEIGHT % 32:
    raise ValueError("WAN22_TI2V_WIDTH 和 WAN22_TI2V_HEIGHT 必须是 32 的倍数。")
if not (256 <= DEFAULT_WIDTH <= 1280 and 256 <= DEFAULT_HEIGHT <= 1280):
    raise ValueError("Wan2.2 默认宽高必须在 256 到 1280 之间。")
if DEFAULT_WIDTH * DEFAULT_HEIGHT > 1280 * 704:
    raise ValueError("Wan2.2 默认分辨率面积不能超过 1280×704。")
if DEFAULT_LENGTH not in {49, 81, 121}:
    raise ValueError("WAN22_TI2V_LENGTH 仅支持 49、81 或 121 帧。")
if not 1 <= DEFAULT_STEPS <= 50:
    raise ValueError("WAN22_TI2V_STEPS 必须在 1 到 50 之间。")
if not 0 <= DEFAULT_SEED <= 2**64 - 1:
    raise ValueError("WAN22_TI2V_SEED 必须在 uint64 范围内。")
if not 1 <= DEFAULT_FPS <= 60:
    raise ValueError("WAN22_TI2V_FPS 必须在 1 到 60 之间。")

for directory in (IMAGE_DIR, OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_STAGING_DIR, RUNTIME_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend Wan2.2 TI2V 5B GGUF API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Wan22Request(BaseModel):
    """Wan2.2 文生视频或参考图生视频请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12_000)
    negative_prompt: str = Field(default="", max_length=4_000)
    image_path: str | None = Field(default=None, max_length=1_024)
    width: int = Field(default=DEFAULT_WIDTH, ge=256, le=1280)
    height: int = Field(default=DEFAULT_HEIGHT, ge=256, le=1280)
    num_frames: Literal[49, 81, 121] = DEFAULT_LENGTH
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=50)
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**64 - 1)
    fps: int = Field(default=DEFAULT_FPS, ge=1, le=60)

    @field_validator("width", "height")
    @classmethod
    def dimensions_are_aligned(cls, value: int) -> int:
        """确保宽高符合 Wan latent 下采样要求。"""
        if value % 32:
            raise ValueError("width 和 height 必须是 32 的倍数。")
        return value

    @field_validator("image_path")
    @classmethod
    def image_path_is_nonempty(cls, value: str | None) -> str | None:
        """拒绝空的参考图片逻辑标识。"""
        if value is not None and not value.strip():
            raise ValueError("image_path 不能为空字符串。")
        return value

    @model_validator(mode="after")
    def resolution_area_is_bounded(self) -> Wan22Request:
        """限制总像素量，避免单请求超出 5B 服务的预设显存预算。"""
        if self.width * self.height > 1280 * 704:
            raise ValueError("宽高面积不能超过 1280×704。")
        return self


def module_available(module_name: str) -> bool:
    """检查依赖规格而不导入 Torch 或初始化 CUDA。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def image_path(full_path: str) -> Path:
    """将客户端逻辑图片路径映射到受控目录，拒绝任意服务器路径。"""
    return IMAGE_DIR / image_storage_name(full_path)


def gpu_runtime_lock(label: str):
    """通过共享 GPU 文件锁串行运行本地生成任务。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """ComfyUI 完全退出后再释放 GPU 锁。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def store_uploaded_image(staged, full_path: str) -> dict[str, Any]:
    """原子提交已分块暂存的参考图。"""
    try:
        image_info(staged.path)
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


class Wan22WorkerManager:
    """启动一次性 ComfyUI worker、校验输出并保存视频。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: Wan22Request) -> dict[str, Any]:
        """在 GPU 排队前检查模型、ComfyUI、GGUF 节点及请求图片。"""
        reference_image: Path | None = None
        if request.image_path:
            reference_image = image_path(request.image_path)
            if not reference_image.is_file():
                raise FileNotFoundError(f"请先上传参考图片: {request.image_path}")
            image_info(reference_image)
        missing = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
        missing.extend(str(path) for path in REQUIRED_COMFYUI_FILES if not path.is_file())
        missing.extend(str(path) for path in REQUIRED_GGUF_NODE_FILES if not path.is_file())
        missing_modules = [name for name in REQUIRED_MODULES if not module_available(name)]
        if missing or missing_modules:
            detail = []
            if missing:
                detail.append("缺少文件: " + ", ".join(missing))
            if missing_modules:
                detail.append("缺少依赖: " + ", ".join(missing_modules))
            raise RuntimeError("Wan2.2 TI2V 本地运行环境不完整；" + "；".join(detail))
        if not Path(WORKER_SCRIPT).is_file():
            raise RuntimeError(f"Wan2.2 worker 脚本不存在: {WORKER_SCRIPT}")
        ffprobe = shutil.which(FFPROBE_BIN)
        if not ffprobe:
            raise RuntimeError("Wan2.2 服务需要可用的 ffprobe。")
        return {
            **request.model_dump(),
            "reference_image_path": str(reference_image) if reference_image else None,
            "model_dir": MODEL_DIR,
            "comfyui_code_path": COMFYUI_CODE_PATH,
            "gguf_node_path": GGUF_NODE_PATH,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "request_timeout": REQUEST_TIMEOUT,
            "ffprobe_bin": str(Path(ffprobe).resolve()),
        }

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """在独立进程组中运行生成任务，校验后原子保存 MP4。"""
        python_executable = sys.executable
        if not python_executable or not Path(python_executable).is_file():
            raise RuntimeError("未找到 Wan2.2 uv 项目的 Python 解释器。")
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="wan22_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="wan22_out_",
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
                terminate_process_group(process, "Wan2.2 TI2V")
                stdout, stderr = process.communicate()
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(
                    f"Wan2.2 worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                    + (f": {excerpt[-1500:]}" if excerpt else "")
                ) from exc
            if stdout.strip():
                LOGGER.info("Wan2.2 worker: %s", stdout.rstrip())
            LOGGER.info(
                "Wan2.2 worker 退出码=%s，耗时 %.2fs",
                process.returncode,
                time.perf_counter() - started,
            )
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4000:] or "Wan2.2 worker 执行失败。")
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
            terminate_process_group(process, "Wan2.2 TI2V")
            Path(request_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


manager = Wan22WorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """检查本地权重、ComfyUI/GGUF 源码、依赖和 GPU，不加载模型。"""
    cuda = cuda_status()
    available_modules = {name: module_available(name) for name in REQUIRED_MODULES}
    model_files = {name: path.is_file() for name, path in MODEL_FILES.items()}
    native_ready = (
        all(model_files.values())
        and all(path.is_file() for path in REQUIRED_COMFYUI_FILES)
        and all(path.is_file() for path in REQUIRED_GGUF_NODE_FILES)
        and all(available_modules.values())
        and Path(WORKER_SCRIPT).is_file()
        and bool(shutil.which(FFPROBE_BIN))
        and cuda["available"]
    )
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "comfyui_code_path": COMFYUI_CODE_PATH,
            "gguf_node_path": GGUF_NODE_PATH,
            "image_dir": str(IMAGE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
        },
        "available": {
            "worker_script": Path(WORKER_SCRIPT).is_file(),
            "model_files": model_files,
            "comfyui_code": all(path.is_file() for path in REQUIRED_COMFYUI_FILES),
            "gguf_custom_node": all(path.is_file() for path in REQUIRED_GGUF_NODE_FILES),
            "dependencies": available_modules,
            "ffprobe": bool(shutil.which(FFPROBE_BIN)),
            "cuda": cuda["available"],
            "selected_provider": "comfyui_gguf",
            "selected_provider_ready": native_ready,
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> one whitelisted ComfyUI -> exit",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "default_size": [DEFAULT_WIDTH, DEFAULT_HEIGHT],
            "max_area": 1280 * 704,
            "default_frames": DEFAULT_LENGTH,
            "supported_frames": [49, 81, 121],
            "default_steps": DEFAULT_STEPS,
            "fps": DEFAULT_FPS,
            "gpu_scheduling": "shared exclusive file lock",
            "text_encoder_device": "cpu",
            "vae_decode": "tiled",
        },
        "last_errors": {"wan22_ti2v": manager.last_error},
    }


@app.post("/v1/upload_image")
async def upload_image(image: UploadFile = File(...), full_path: str = Form(...)):
    """分块暂存并原子保存图片；full_path 只作为客户端逻辑标识。"""
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
    """检查逻辑图片标识是否已上传。"""
    try:
        exists = image_path(file_name).is_file()
    except ValueError:
        exists = False
    return {"code": 200 if exists else 404, "exists": exists}


@app.post("/v1/QuantStack/Wan2.2-TI2V-5B-GGUF")
def generate(request: Wan22Request):
    """串行调用 ComfyUI 完成文生视频或参考图生视频。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("wan22_ti2v/generate") as metrics:
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
        LOGGER.exception("Wan2.2 TI2V generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   LLM Local Backend Wan2.2 TI2V 5B GGUF")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] ComfyUI 源码: {COMFYUI_CODE_PATH}")
    print(f"[配置] ComfyUI-GGUF 节点: {GGUF_NODE_PATH}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, cuda_visible_devices={CUDA_VISIBLE_DEVICES}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
