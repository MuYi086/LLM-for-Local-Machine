#!/usr/bin/env python3
"""独立 Kandinsky 6 Lite 蒸馏版图生视频 HTTP 服务。"""

from __future__ import annotations

# API 进程只做输入校验、上传、GPU 排队与一次性 worker 编排，不导入 Torch/Diffusers。
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
from typing import Any

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
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from kandinsky_runtime import (
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
    os.getenv("KANDINSKY6_MODEL_DIR", str(Path(HF_MIRROR_DIR) / "kandinskylab"))
)
IMAGE_DIR = Path(
    expand_path(os.getenv("KANDINSKY6_IMAGE_DIR", str(STORAGE_DIR / "kandinsky/images")))
)
OUTPUT_DIR = Path(expand_path(os.getenv("KANDINSKY6_OUTPUT_DIR", str(STORAGE_DIR / "video"))))
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv("KANDINSKY6_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "kandinsky6_worker"))
    )
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
FFPROBE_BIN = os.getenv("KANDINSKY6_FFPROBE_BIN", "ffprobe")
API_HOST = os.getenv("KANDINSKY6_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("KANDINSKY6_PORT", os.getenv("PORT", "8369")))
REQUEST_TIMEOUT = float(os.getenv("KANDINSKY6_REQUEST_TIMEOUT", "2400"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
CUDA_VISIBLE_DEVICES = os.getenv("KANDINSKY6_CUDA_VISIBLE_DEVICES", "0")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_FRAMES = 121
DEFAULT_FRAME_RATE = 24
INFERENCE_STEPS = 10
GUIDANCE_SCALE = 1.0

REQUIRED_MODEL_FILES = tuple(
    Path(MODEL_DIR) / relative_path
    for relative_path in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00005.safetensors",
        "text_encoder/model-00002-of-00005.safetensors",
        "text_encoder/model-00003-of-00005.safetensors",
        "text_encoder/model-00004-of-00005.safetensors",
        "text_encoder/model-00005-of-00005.safetensors",
        "text_encoder/preprocessor_config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/tokenizer_config.json",
        "text_encoder/merges.txt",
        "text_encoder/vocab.json",
        "tokenizer/processor_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors",
        "tokenizer_2/tokenizer.json",
        "tokenizer_2/tokenizer_config.json",
        "audio_vae/config.json",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "vocoder/config.json",
        "vocoder/diffusion_pytorch_model.safetensors",
    )
)
REQUIRED_MODULES = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "PIL", "av")

for name, value in (
    ("KANDINSKY6_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
):
    positive_float(name, value)
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 不能为负数。")
if not CUDA_VISIBLE_DEVICES or "," in CUDA_VISIBLE_DEVICES:
    raise ValueError("Kandinsky 6 必须指定单个可见 CUDA 设备。")

for directory in (IMAGE_DIR, OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_STAGING_DIR, RUNTIME_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend Kandinsky 6 Lite Distilled API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Kandinsky6Request(BaseModel):
    """使用受控上传图片和文本提示词发起 5 秒图生视频请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12_000)
    image_path: str = Field(min_length=1, max_length=1024)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def prompt_is_not_blank(self) -> Kandinsky6Request:
        """在 GPU 排队前拒绝空白提示词和无效图片标识。"""
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空。")
        image_storage_name(self.image_path)
        return self


def module_available(module_name: str) -> bool:
    """检查依赖模块规格，不导入可能初始化 CUDA 的运行时。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def model_files_ready() -> bool:
    """检查 Diffusers pipeline 使用的完整本地模型快照。"""
    return all(path.is_file() for path in REQUIRED_MODEL_FILES)


def image_path(full_path: str) -> Path:
    """把 WebUI 逻辑路径映射到上传目录，拒绝读取服务器任意路径。"""
    return IMAGE_DIR / image_storage_name(full_path)


def gpu_runtime_lock(label: str):
    """使用共享有界文件锁串行调度所有本地 GPU worker。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """worker 退出后等待显存归还，再释放共享 GPU 锁。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def shutil_which(program: str) -> str | None:
    """返回可执行文件绝对路径。"""
    resolved = shutil.which(program)
    return str(Path(resolved).resolve()) if resolved else None


def store_uploaded_image(staged, full_path: str) -> dict[str, Any]:
    """原子提交按块暂存的参考图片。"""
    try:
        commit_plain_upload(staged, image_path(full_path))
        return {
            "code": 200,
            "filename": full_path,
            "sha256": staged.sha256,
            "size_bytes": staged.size_bytes,
        }
    finally:
        staged.path.unlink(missing_ok=True)


class Kandinsky6WorkerManager:
    """启动一次性 Diffusers worker，验证并原子保存生成视频。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: Kandinsky6Request) -> dict[str, Any]:
        """在排队前检查图片、完整权重、依赖和输出校验工具。"""
        reference_image = image_path(request.image_path)
        if not reference_image.is_file():
            raise FileNotFoundError(f"请先上传参考图片: {request.image_path}")
        missing_files = [str(path) for path in REQUIRED_MODEL_FILES if not path.is_file()]
        missing_modules = [name for name in REQUIRED_MODULES if not module_available(name)]
        if missing_files or missing_modules:
            detail = []
            if missing_files:
                detail.append("缺少文件: " + ", ".join(missing_files))
            if missing_modules:
                detail.append("缺少依赖: " + ", ".join(missing_modules))
            raise RuntimeError(
                "Kandinsky 6 本地模型文件不完整或运行环境不完整；" + "；".join(detail)
            )
        image_info(reference_image)
        if not Path(WORKER_SCRIPT).is_file():
            raise RuntimeError(f"Kandinsky 6 worker 脚本不存在: {WORKER_SCRIPT}")
        ffprobe_bin = shutil_which(FFPROBE_BIN)
        if not ffprobe_bin:
            raise RuntimeError("Kandinsky 6 服务需要可用的 ffprobe。")
        return {
            **request.model_dump(),
            "reference_image_path": str(reference_image),
            "model_dir": MODEL_DIR,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "request_timeout": REQUEST_TIMEOUT,
            "ffprobe_bin": ffprobe_bin,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "num_frames": DEFAULT_FRAMES,
            "frame_rate": DEFAULT_FRAME_RATE,
            "num_inference_steps": INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "sample_audio": False,
        }

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """隔离 worker 进程组并在验证后原子保存 MP4。"""
        python_executable = sys.executable
        if not python_executable or not Path(python_executable).is_file():
            raise RuntimeError("未找到 Kandinsky 6 uv 环境的 Python 解释器。")
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="kandinsky6_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="kandinsky6_out_",
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
                terminate_process_group(process, "Kandinsky 6")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Kandinsky 6 worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc
            if stdout.strip():
                LOGGER.info("Kandinsky 6 worker: %s", stdout.rstrip())
            LOGGER.info(
                "Kandinsky 6 worker 退出码=%s，耗时 %.2fs",
                process.returncode,
                time.perf_counter() - started,
            )
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-4000:] or "Kandinsky 6 worker 执行失败。")
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
            terminate_process_group(process, "Kandinsky 6")
            Path(request_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


manager = Kandinsky6WorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """报告本地模型文件、依赖和 GPU 状态，不加载权重。"""
    cuda = cuda_status()
    dependencies = {name: module_available(name) for name in REQUIRED_MODULES}
    model_files = {
        str(path.relative_to(MODEL_DIR)): path.is_file() for path in REQUIRED_MODEL_FILES
    }
    native_ready = (
        all(model_files.values())
        and all(dependencies.values())
        and Path(WORKER_SCRIPT).is_file()
        and bool(shutil_which(FFPROBE_BIN))
        and cuda["available"]
    )
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "image_dir": str(IMAGE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
        },
        "available": {
            "worker_script": Path(WORKER_SCRIPT).is_file(),
            "model_files": model_files,
            "dependencies": dependencies,
            "ffprobe": bool(shutil_which(FFPROBE_BIN)),
            "cuda": cuda["available"],
            "selected_provider": "diffusers",
            "selected_provider_ready": native_ready,
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> exit",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "precision": "bfloat16",
            "cpu_offload": "sequential",
            "inference_steps": INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "resolution": f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
            "num_frames": DEFAULT_FRAMES,
            "frame_rate": DEFAULT_FRAME_RATE,
            "sample_audio": False,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"kandinsky6": manager.last_error},
    }


@app.post("/v1/upload_image")
async def upload_image(image: UploadFile = File(...), full_path: str = Form(...)):
    """分块接收参考图并原子保存；full_path 只作为逻辑标识。"""
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
    """检查逻辑标识对应的已上传图片。"""
    try:
        exists = image_path(file_name).is_file()
    except ValueError:
        exists = False
    return {"code": 200 if exists else 404, "exists": exists}


@app.post("/v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16")
def generate(request: Kandinsky6Request):
    """串行调用本地 BF16 pipeline 生成 864×480、5 秒 MP4。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("kandinsky6_lite_distill/generate") as metrics:
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
        LOGGER.exception("Kandinsky 6 generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   LLM Local Backend Kandinsky 6 Lite Distilled")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, cuda_visible_devices={CUDA_VISIBLE_DEVICES}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
