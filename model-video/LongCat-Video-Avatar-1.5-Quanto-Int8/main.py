#!/usr/bin/env python3
"""LongCat Avatar 1.5 Quanto-Int8 独立 HTTP 服务。"""

from __future__ import annotations

import hashlib

# HTTP 进程只处理上传、预检、GPU 排队与 worker 编排，不导入 Torch 或模型代码。
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    UploadPolicy,
    commit_plain_upload,
    stage_audio_upload,
)
from llm_local_runtime import gpu_runtime_lock as shared_gpu_runtime_lock
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from longcat_runtime import (
    audio_duration,
    audio_storage_name,
    cuda_status,
    expand_path,
    image_info,
    image_storage_name,
    persist_video,
    positive_float,
    query_gpu_memory,
    terminate_process_group,
    validate_video,
)
from wangp_runtime import DEFAULT_MEMORY_PROFILE, link_status, validate_memory_profile

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]

STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "LONGCAT_AVATAR_MODEL_DIR",
            os.path.join(HF_MIRROR_DIR, "meituan/LongCat-Video-Avatar-1.5"),
        )
    )
)
BASE_MODEL_DIR = Path(
    expand_path(
        os.getenv(
            "LONGCAT_AVATAR_BASE_MODEL_DIR",
            os.path.join(HF_MIRROR_DIR, "meituan/LongCat-Video"),
        )
    )
)
WANGP_DIR = Path(expand_path(os.getenv("LONGCAT_AVATAR_WANGP_DIR", "~/LLM-depency/WanGP")))
WANGP_PYTHON = Path(
    expand_path(os.getenv("LONGCAT_AVATAR_WANGP_PYTHON", str(WANGP_DIR / ".venv/bin/python")))
)
WANGP_CONFIG_FILE = WANGP_DIR / "wgp_config.json"
AUDIO_DIR = Path(
    expand_path(os.getenv("LONGCAT_AVATAR_AUDIO_DIR", str(STORAGE_DIR / "longcat_avatar/audio")))
)
IMAGE_DIR = Path(
    expand_path(os.getenv("LONGCAT_AVATAR_IMAGE_DIR", str(STORAGE_DIR / "longcat_avatar/images")))
)
OUTPUT_DIR = Path(
    expand_path(os.getenv("LONGCAT_AVATAR_OUTPUT_DIR", str(STORAGE_DIR / "video/longcat_avatar")))
)
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv(
            "LONGCAT_AVATAR_WORKER_TMP_DIR",
            str(RUNTIME_CACHE_DIR / "longcat_avatar_worker"),
        )
    )
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("LONGCAT_AVATAR_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("LONGCAT_AVATAR_PORT", os.getenv("PORT", "8390")))
REQUEST_TIMEOUT = float(os.getenv("LONGCAT_AVATAR_REQUEST_TIMEOUT", "1800"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_VISIBLE_DEVICES = os.getenv("LONGCAT_AVATAR_CUDA_VISIBLE_DEVICES", "0").strip()
MEMORY_PROFILE = validate_memory_profile(
    os.getenv("LONGCAT_AVATAR_MEMORY_PROFILE", str(DEFAULT_MEMORY_PROFILE))
)
MIN_FREE_VRAM_MIB = int(os.getenv("LONGCAT_AVATAR_MIN_FREE_VRAM_MIB", "10000"))
MAX_UPLOAD_IMAGE_BYTES = int(os.getenv("LONGCAT_AVATAR_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
MAX_SEGMENTS = 5
MAX_AUDIO_SECONDS = 16.52
FIRST_SEGMENT_AUDIO_SECONDS = 93 / 25
CONTINUATION_AUDIO_SECONDS = (93 - 13) / 25
REQUIRED_MODULES = (
    "fastapi",
    "uvicorn",
    "multipart",
    "PIL",
)
WANGP_MODULES = (
    "torch",
    "triton",
    "gradio",
    "diffusers",
    "transformers",
    "optimum.quanto",
    "safetensors",
    "easydict",
    "smplfitter",
    "mmgp",
)

REQUIRED_MODEL_FILES = {
    "quanto_dit": MODEL_DIR / "longcat_avatar_v1_5_quanto_bf16_int8.safetensors",
    "quanto_t5": MODEL_DIR / "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
    "tokenizer": MODEL_DIR / "umt5-xxl" / "tokenizer.json",
    "tokenizer_model": MODEL_DIR / "umt5-xxl" / "spiece.model",
    "tokenizer_config": MODEL_DIR / "umt5-xxl" / "tokenizer_config.json",
    "tokenizer_special_tokens": MODEL_DIR / "umt5-xxl" / "special_tokens_map.json",
    "distillation_lora": MODEL_DIR / "lora" / "dmd_lora.safetensors",
    "whisper_config": MODEL_DIR / "whisper-large-v3" / "config.json",
    "whisper_generation_config": MODEL_DIR / "whisper-large-v3" / "generation_config.json",
    "whisper_weights": MODEL_DIR / "whisper-large-v3" / "model.safetensors",
    "whisper_preprocessor": MODEL_DIR / "whisper-large-v3" / "preprocessor_config.json",
    "base_vae_weights": BASE_MODEL_DIR / "vae" / "diffusion_pytorch_model.safetensors",
}
REQUIRED_WANGP_FILES = (
    WANGP_DIR / "wgp.py",
    WANGP_DIR / "shared" / "api.py",
    WANGP_DIR / "mmgp" / "__init__.py",
    WANGP_DIR / "models" / "longcat" / "longcat_handler.py",
    WANGP_DIR / "models" / "longcat" / "longcat_main.py",
    WANGP_DIR / "models" / "longcat" / "configs" / "longcat_avatar_v1_5.json",
    WANGP_DIR / "models" / "longcat" / "configs" / "longcat_avatar_v1_5_scheduler.json",
    WANGP_DIR / "models" / "longcat" / "configs" / "longcat_vae.json",
)

for name, value in (
    ("LONGCAT_AVATAR_REQUEST_TIMEOUT", REQUEST_TIMEOUT),
    ("GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT),
):
    positive_float(name, value)
if CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 不能为负数。")
if not CUDA_VISIBLE_DEVICES or len(CUDA_VISIBLE_DEVICES.split(",")) != 1:
    raise ValueError("LongCat Avatar WanGP worker 固定使用一张可见 CUDA GPU。")
if MIN_FREE_VRAM_MIB < 1 or MAX_UPLOAD_IMAGE_BYTES < 1:
    raise ValueError("LongCat Avatar 显存门槛和图片上传上限必须为正数。")

for directory in (AUDIO_DIR, IMAGE_DIR, OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_STAGING_DIR):
    directory.mkdir(parents=True, exist_ok=True)
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

app = FastAPI(title="LLM Local Backend LongCat Avatar 1.5 Quanto-Int8 API")


class LongCatAvatarRequest(BaseModel):
    """LongCat Avatar 单人音频驱动视频请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=12_000)
    audio_path: str = Field(min_length=1, max_length=1_024)
    image_path: str = Field(min_length=1, max_length=1_024)
    resolution: Literal["480p", "720p"] = "480p"
    num_segments: int | None = Field(default=None, ge=1, le=MAX_SEGMENTS)

    @model_validator(mode="after")
    def validate_identifiers(self) -> LongCatAvatarRequest:
        """校验文本与逻辑文件名，阻止路径穿越。"""
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空。")
        audio_storage_name(self.audio_path)
        image_storage_name(self.image_path)
        return self


def module_available(module_name: str) -> bool:
    """只检查 Python 包规格，不导入 Torch 或初始化 CUDA。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def audio_path(full_path: str) -> Path:
    """将音频逻辑标识映射到受控上传目录。"""
    return AUDIO_DIR / audio_storage_name(full_path)


def image_path(full_path: str) -> Path:
    """将图片逻辑标识映射到受控上传目录。"""
    return IMAGE_DIR / image_storage_name(full_path)


def nonempty_file(path: Path) -> bool:
    """只将文件系统中非空的普通文件视为已准备资源。"""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def model_file_status() -> dict[str, bool]:
    """返回 WanGP LongCat 推理所需的 Quanto 权重和辅助文件状态。"""
    return {name: nonempty_file(path) for name, path in REQUIRED_MODEL_FILES.items()}


def required_model_files_ready() -> dict[str, bool]:
    """提供 health 兼容字段，并包含所有模型文件状态。"""
    return model_file_status()


def wangp_source_status() -> dict[str, bool]:
    """检查 WanGP API、MMGP 和 LongCat Avatar 模型实现文件。"""
    return {
        path.relative_to(WANGP_DIR).as_posix(): nonempty_file(path) for path in REQUIRED_WANGP_FILES
    }


def runtime_modules_ready() -> dict[str, bool]:
    """只检查轻量 HTTP 服务自身的模块，不导入模型依赖。"""
    return {name: module_available(name) for name in REQUIRED_MODULES}


def wangp_runtime_modules_ready() -> dict[str, bool]:
    """通过 WanGP Python 检查模型运行时模块规格，不导入 Torch。"""
    status = {name: False for name in WANGP_MODULES}
    if not WANGP_PYTHON.is_file() or not WANGP_DIR.is_dir():
        return status
    script = f"""
import importlib.util
import json

result = {{}}
for name in {WANGP_MODULES!r}:
    try:
        result[name] = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        result[name] = False
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [str(WANGP_PYTHON), "-c", script],
            cwd=WANGP_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES},
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if isinstance(payload, dict):
            return {name: bool(payload.get(name, False)) for name in WANGP_MODULES}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError):
        LOGGER.exception("WanGP Python 模块预检失败")
    return status


def wangp_config_status() -> dict[str, bool]:
    """确认 WanGP 配置指向本地检查点和当前服务显存档。"""
    checks = {
        "config_file": False,
        "checkpoints_paths": False,
        "loras_root": False,
        "profile_matches_runtime": False,
        "attention_sdpa": False,
    }
    try:
        config = json.loads(WANGP_CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return checks
        checks.update(
            {
                "config_file": True,
                "checkpoints_paths": config.get("checkpoints_paths") == ["ckpts"],
                "loras_root": config.get("loras_root") == "loras",
                "profile_matches_runtime": (
                    config.get("video_profile") == MEMORY_PROFILE
                    and config.get("profile") == MEMORY_PROFILE
                ),
                "attention_sdpa": config.get("attention_mode") == "sdpa",
            }
        )
    except (OSError, json.JSONDecodeError):
        return checks
    return checks


def cuda_status_for_service() -> dict[str, Any]:
    """查询 WanGP worker 可见的单张 GPU 是否可用。"""
    return cuda_status(CUDA_VISIBLE_DEVICES)


def resolve_num_segments(duration: float, requested: int | None) -> int:
    """依据音频时长确定延续段数，拒绝不完整覆盖或静默截断。"""
    if duration <= 0:
        raise ValueError("音频时长必须大于零。")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"音频超过最多生成 5 段的覆盖范围（最长 {MAX_AUDIO_SECONDS:.2f} 秒），超出时返回 422。"
        )
    if duration <= FIRST_SEGMENT_AUDIO_SECONDS:
        required = 1
    else:
        continuations = (duration - FIRST_SEGMENT_AUDIO_SECONDS) / CONTINUATION_AUDIO_SECONDS
        required = 1 + math.ceil(continuations - 1e-9)
    if requested is not None and requested < required:
        raise ValueError("num_segments 无法覆盖完整音频，不能静默截断。")
    if requested is not None:
        if requested > required:
            raise ValueError("num_segments 超过音频实际需要的延续段数。")
    if required > MAX_SEGMENTS:
        raise ValueError("LongCat Avatar 最多生成 5 段。")
    return required


def gpu_runtime_lock(label: str):
    """使用共享有界 GPU 锁串行运行模型 worker。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """worker 退出后等待 CUDA 资源归还，再释放共享锁。"""
    if CUDA_RELEASE_DELAY > 0:
        time.sleep(CUDA_RELEASE_DELAY)


def store_uploaded_audio(staged, full_path: str) -> dict[str, Any]:
    """原子保存按块暂存的参考音频。"""
    try:
        commit_plain_upload(staged, audio_path(full_path))
        return {
            "code": 200,
            "filename": full_path,
            "sha256": staged.sha256,
            "size_bytes": staged.size_bytes,
        }
    finally:
        staged.path.unlink(missing_ok=True)


def store_uploaded_image(
    staged_path: Path, full_path: str, sha256: str, size_bytes: int
) -> dict[str, Any]:
    """校验并原子提交已暂存的图片。"""
    try:
        image_info(staged_path)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=IMAGE_DIR, prefix=".upload_", suffix=Path(full_path).suffix.lower()
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(staged_path, temporary)
            from llm_local_runtime.storage import StagedUpload

            commit_plain_upload(
                StagedUpload(temporary, sha256, size_bytes, Path(full_path).suffix.lower()),
                image_path(full_path),
            )
        finally:
            temporary.unlink(missing_ok=True)
        return {"code": 200, "filename": full_path, "sha256": sha256, "size_bytes": size_bytes}
    finally:
        staged_path.unlink(missing_ok=True)


class LongCatAvatarWorkerManager:
    """编排 WanGP 一次性 worker，验证并原子保存最终视频。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: LongCatAvatarRequest) -> dict[str, Any]:
        """在 GPU 队列前完成 WanGP、输入、时长与单卡显存预检。"""
        source_audio = audio_path(request.audio_path)
        if not source_audio.is_file():
            raise FileNotFoundError("请先上传请求使用的音频。")
        source_image = image_path(request.image_path)
        if not source_image.is_file():
            raise FileNotFoundError("请先上传请求使用的参考图片。")
        image_info(source_image)

        missing_files = [name for name, ready in model_file_status().items() if not ready]
        missing_code = [name for name, ready in wangp_source_status().items() if not ready]
        modules = runtime_modules_ready()
        missing_modules = [name for name, ready in modules.items() if not ready]
        wangp_modules = wangp_runtime_modules_ready()
        missing_wangp_modules = [name for name, ready in wangp_modules.items() if not ready]
        links = link_status(WANGP_DIR, MODEL_DIR, BASE_MODEL_DIR)
        missing_links = [name for name, ready in links.items() if not ready]
        config = wangp_config_status()
        missing_config = [name for name, ready in config.items() if not ready]
        if (
            missing_files
            or missing_code
            or missing_modules
            or missing_wangp_modules
            or missing_links
            or missing_config
        ):
            details = []
            if missing_files:
                details.append(f"缺少模型文件: {', '.join(missing_files)}")
            if missing_code:
                details.append(f"缺少 WanGP 源码: {', '.join(missing_code)}")
            if missing_modules:
                details.append(f"缺少 HTTP 服务依赖: {', '.join(missing_modules)}")
            if missing_wangp_modules:
                details.append(f"缺少 WanGP 运行依赖: {', '.join(missing_wangp_modules)}")
            if missing_links:
                details.append(f"WanGP 权重别名未就绪: {', '.join(missing_links)}")
            if missing_config:
                details.append(f"WanGP 配置不匹配: {', '.join(missing_config)}")
            raise RuntimeError("；".join(details))

        ffprobe = shutil.which(FFPROBE_BIN)
        if not ffprobe:
            raise RuntimeError("LongCat Avatar 服务需要可用的 ffprobe。")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("LongCat Avatar 服务需要可用的 ffmpeg。")
        duration = audio_duration(source_audio, ffprobe)
        segments = resolve_num_segments(duration, request.num_segments)

        gpu_status = cuda_status_for_service()
        if not gpu_status.get("available"):
            raise RuntimeError("LongCat Avatar 需要一张可用 CUDA GPU。")
        memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
        devices = memory.get("devices", [])
        if len(devices) != 1:
            raise RuntimeError("LongCat Avatar WanGP worker 需要一张可用 CUDA GPU。")
        if any(int(device.get("free_mib", 0)) < MIN_FREE_VRAM_MIB for device in devices):
            raise RuntimeError(f"LongCat Avatar 至少需要 {MIN_FREE_VRAM_MIB} MiB 空闲显存。")

        return {
            "wangp_dir": str(WANGP_DIR),
            "model_dir": str(MODEL_DIR),
            "base_model_dir": str(BASE_MODEL_DIR),
            "audio_file": str(source_audio),
            "image_file": str(source_image),
            "prompt": request.prompt.strip(),
            "resolution": "832x480" if request.resolution == "480p" else "1280x720",
            "duration_seconds": duration,
            "num_segments": segments,
            "memory_profile": MEMORY_PROFILE,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "ffprobe_bin": ffprobe,
            "request_timeout": max(1.0, REQUEST_TIMEOUT - 30.0),
        }

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """启动 worker 进程组并始终清理工作区。"""
        with self.lock:
            workspace = Path(tempfile.mkdtemp(prefix="request_", dir=WORKER_TMP_DIR))
            output_video = workspace / "result.mp4"
            request_path = workspace / "request.json"
            worker_payload = {
                **payload,
                "workspace_dir": str(workspace),
                "output_video": str(output_video),
            }
            request_path.write_text(
                json.dumps(worker_payload, ensure_ascii=False), encoding="utf-8"
            )
            process = None
            try:
                process = subprocess.Popen(
                    [str(WANGP_PYTHON), WORKER_SCRIPT, "--input-json", str(request_path)],
                    cwd=WANGP_DIR,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
                except subprocess.TimeoutExpired as exc:
                    terminate_process_group(process, "LongCat Avatar worker")
                    raise RuntimeError("LongCat Avatar worker 超时，已结束整个进程组。") from exc
                if process.returncode != 0:
                    raise RuntimeError(
                        f"LongCat Avatar worker 失败（{process.returncode}）："
                        f"{stderr[-4000:] or stdout[-4000:]}"
                    )
                validate_video(output_video, payload["ffprobe_bin"])
                saved = persist_video(output_video, OUTPUT_DIR)
                self.last_output_path = str(saved)
                self.last_error = None
                return saved
            except Exception as exc:
                self.last_error = str(exc)
                raise
            finally:
                if process is not None:
                    terminate_process_group(process, "LongCat Avatar worker")
                shutil.rmtree(workspace, ignore_errors=True)


manager = LongCatAvatarWorkerManager()


@app.get("/v1/health")
def health() -> JSONResponse:
    """返回不加载模型的本地权重、WanGP 依赖与单卡状态。"""
    files = model_file_status()
    code = wangp_source_status()
    modules = runtime_modules_ready()
    wangp_modules = wangp_runtime_modules_ready()
    wangp_links = link_status(WANGP_DIR, MODEL_DIR, BASE_MODEL_DIR)
    wangp_config = wangp_config_status()
    gpu = cuda_status_for_service()
    binaries = {name: shutil.which(name) is not None for name in (FFPROBE_BIN, "ffmpeg")}
    try:
        memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
        enough_memory = len(memory.get("devices", [])) == 1 and all(
            int(device.get("free_mib", 0)) >= MIN_FREE_VRAM_MIB
            for device in memory.get("devices", [])
        )
    except RuntimeError:
        memory = {"devices": []}
        enough_memory = False
    selected_ready = (
        all(files.values())
        and all(code.values())
        and all(modules.values())
        and all(wangp_modules.values())
        and all(wangp_links.values())
        and all(wangp_config.values())
        and all(binaries.values())
    )
    selected_ready = selected_ready and bool(gpu.get("available")) and enough_memory
    return JSONResponse(
        {
            "status": "ok" if selected_ready else "unavailable",
            "service": "longcat-video-avatar-1.5-quanto-int8",
            "port": API_PORT,
            "available": {
                "selected_provider_ready": selected_ready,
                "model_files": files,
                "wangp_source": code,
                "http_modules": modules,
                "wangp_modules": wangp_modules,
                "wangp_weights": wangp_links,
                "wangp_config": wangp_config,
                "binaries": binaries,
                "cuda": gpu,
                "gpu_memory": memory,
            },
            "runtime": {
                "last_error": manager.last_error,
                "last_output_path": manager.last_output_path,
            },
        }
    )


@app.post("/v1/upload_audio")
async def upload_audio(full_path: str = Form(...), audio: UploadFile = File(...)) -> dict[str, Any]:
    """流式暂存并原子保存参考音频。"""
    try:
        audio_storage_name(full_path)
        staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR, policy=UploadPolicy())
        return await run_in_threadpool(store_uploaded_audio, staged, full_path)
    except (AudioUploadError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/upload_image")
async def upload_image(full_path: str = Form(...), image: UploadFile = File(...)) -> dict[str, Any]:
    """分块暂存并校验参考图片，再原子提交。"""
    try:
        image_storage_name(full_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    descriptor, staged_name = tempfile.mkstemp(
        dir=UPLOAD_STAGING_DIR, prefix="image_", suffix=".part"
    )
    staged = Path(staged_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            while chunk := await image.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="参考图片超过上传大小限制。")
                digest.update(chunk)
                destination.write(chunk)
        return await run_in_threadpool(
            store_uploaded_image, staged, full_path, digest.hexdigest(), size_bytes
        )
    except HTTPException:
        staged.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await image.close()


@app.get("/v1/check/audio")
def check_audio(file_name: str) -> dict[str, Any]:
    """检查逻辑音频标识对应的上传文件。"""
    try:
        path = audio_path(file_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"exists": path.is_file(), "filename": file_name}


@app.get("/v1/check/image")
def check_image(file_name: str) -> dict[str, Any]:
    """检查逻辑图片标识对应的上传文件。"""
    try:
        path = image_path(file_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"exists": path.is_file(), "filename": file_name}


@app.post("/v1/meituan/LongCat-Video-Avatar-1.5-Quanto-Int8")
def generate_video(request: LongCatAvatarRequest) -> FileResponse:
    """排队运行 WanGP Avatar worker 并返回已验证的 MP4。"""
    try:
        payload = manager.build_worker_payload(request)
        with gpu_runtime_lock("LongCat-Avatar-Quanto-Int8"):
            gpu_status = cuda_status_for_service()
            if not gpu_status.get("available"):
                raise HTTPException(
                    status_code=503, detail="LongCat Avatar 需要一张可用 CUDA GPU。"
                )
            memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
            if any(
                int(device.get("free_mib", 0)) < MIN_FREE_VRAM_MIB
                for device in memory.get("devices", [])
            ):
                raise HTTPException(status_code=503, detail="LongCat Avatar GPU 空闲显存不足。")
            try:
                output = manager.run_worker(payload)
            finally:
                wait_after_cuda_release()
        return FileResponse(output, media_type="video/mp4", filename=output.name)
    except HTTPException:
        raise
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("LongCat Avatar generation failed")
        raise HTTPException(status_code=500, detail="LongCat Avatar 生成失败。") from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
