#!/usr/bin/env python3
"""MiniMax H3 16G HiFi 本地图生视频 HTTP 服务。"""

from __future__ import annotations

# HTTP 进程仅处理上传、预检、GPU 队列和一次性 worker 编排，不导入 Torch/ComfyUI。
import importlib.util
import json
import logging
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
from starlette.middleware.cors import CORSMiddleware

from minimax_h3_runtime import (
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

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动配置中的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv(
        "MINIMAX_H3_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "minimax/MiniMax-H3-16G-HiFi"),
    )
)
COMFYUI_CODE_PATH = expand_path(
    os.getenv("MINIMAX_H3_COMFYUI_CODE_PATH", "~/LLM-depency/MiniMax-H3-ComfyUI")
)
COMFYUI_PYTHON = expand_path(
    os.getenv("MINIMAX_H3_COMFYUI_PYTHON", os.path.join(COMFYUI_CODE_PATH, ".venv/bin/python"))
)
COMFYUI_PLUGIN_DIR = os.path.join(COMFYUI_CODE_PATH, "custom_nodes/ComfyUI-sol-attn-main")
IMAGE_DIR = Path(
    expand_path(os.getenv("MINIMAX_H3_IMAGE_DIR", str(STORAGE_DIR / "minimax_h3/images")))
)
OUTPUT_DIR = Path(
    expand_path(os.getenv("MINIMAX_H3_OUTPUT_DIR", str(STORAGE_DIR / "video/minimax_h3")))
)
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv(
            "MINIMAX_H3_WORKER_TMP_DIR",
            str(RUNTIME_CACHE_DIR / "minimax_h3_worker"),
        )
    )
)
UPLOAD_STAGING_DIR = RUNTIME_CACHE_DIR / "uploads"
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("MINIMAX_H3_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MINIMAX_H3_PORT", "8392"))
REQUEST_TIMEOUT = float(os.getenv("MINIMAX_H3_REQUEST_TIMEOUT", "1800"))
COMFYUI_STARTUP_TIMEOUT = float(os.getenv("MINIMAX_H3_COMFYUI_STARTUP_TIMEOUT", "180"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
CUDA_VISIBLE_DEVICES = os.getenv("MINIMAX_H3_CUDA_VISIBLE_DEVICES", "0").strip()
MIN_FREE_VRAM_MIB = int(os.getenv("MINIMAX_H3_MIN_FREE_VRAM_MIB", "13000"))
RESERVED_VRAM_GIB = 2.5
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 544
MAX_PIXELS = 960 * 544
DEFAULT_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 5.0
DEFAULT_STEPS = 20
DEFAULT_FPS = 24
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

REQUIRED_MODEL_FILES = {
    "fl2va_dit": Path(MODEL_DIR) / "diffusion_models/minimax_h3_fl2va_dit_16g.safetensors",
    "ref2va_dit": Path(MODEL_DIR) / "diffusion_models/minimax_h3_ref2va_dit_16g.safetensors",
    "text_encoder": Path(MODEL_DIR) / "text_encoders/qwen3vl_text_encoder.safetensors",
    "video_vae": Path(MODEL_DIR) / "vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": Path(MODEL_DIR) / "vae/minimax_h3_audio_vae_fp32.safetensors",
}
REQUIRED_MODULES = ("fastapi", "uvicorn", "multipart", "PIL")
COMFYUI_REQUIRED_MODULES = ("torch", "triton", "pydantic", "pydantic_settings")

for value, name in (
    (REQUEST_TIMEOUT, "MINIMAX_H3_REQUEST_TIMEOUT"),
    (COMFYUI_STARTUP_TIMEOUT, "MINIMAX_H3_COMFYUI_STARTUP_TIMEOUT"),
    (GPU_LOCK_WAIT_TIMEOUT, "GPU_LOCK_WAIT_TIMEOUT"),
):
    positive_float(name, value)
if CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 不能为负数。")
if not CUDA_VISIBLE_DEVICES or "," in CUDA_VISIBLE_DEVICES:
    raise ValueError("MiniMax H3 必须指定单个可见 CUDA 设备。")
if MIN_FREE_VRAM_MIB < 1:
    raise ValueError("MINIMAX_H3_MIN_FREE_VRAM_MIB 必须是正整数。")
if MIN_FREE_VRAM_MIB < 13_000:
    raise ValueError("MINIMAX_H3_MIN_FREE_VRAM_MIB 不能低于 13000，避免降低 H3 显存门槛。")

for directory in (IMAGE_DIR, OUTPUT_DIR, WORKER_TMP_DIR, UPLOAD_STAGING_DIR, RUNTIME_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend MiniMax H3 16G HiFi API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MiniMaxH3Request(BaseModel):
    """MiniMax H3 本地图生视频请求，支持首尾帧与参考图两种模式。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fl2va", "ref2va"] = "fl2va"
    prompt: str = Field(min_length=1, max_length=12_000)
    first_frame_path: str | None = Field(default=None, min_length=1, max_length=1_024)
    last_frame_path: str | None = Field(default=None, min_length=1, max_length=1_024)
    reference_image_paths: list[str] = Field(default_factory=list, max_length=9)
    include_audio: bool = False
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    width: int = Field(default=DEFAULT_WIDTH, ge=64, le=960)
    height: int = Field(default=DEFAULT_HEIGHT, ge=64, le=960)
    duration_seconds: float = Field(
        default=DEFAULT_DURATION_SECONDS,
        gt=0,
        le=MAX_DURATION_SECONDS,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_request_contract(self) -> MiniMaxH3Request:
        """校验提示词、分辨率预算、逻辑图片路径和任务模式。"""
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空。")
        if self.width % 32 or self.height % 32:
            raise ValueError("width 和 height 必须是 32 的倍数。")
        if self.width * self.height > MAX_PIXELS:
            raise ValueError("分辨率超过 MiniMax H3 服务的安全预算 960×544。")
        paths = [path for path in (self.first_frame_path, self.last_frame_path) if path]
        paths.extend(self.reference_image_paths)
        for path in paths:
            image_storage_name(path)
        if self.mode == "fl2va" and self.reference_image_paths:
            raise ValueError(
                "fl2va 使用 first_frame_path/last_frame_path，不能传 reference_image_paths。"
            )
        if self.mode == "ref2va":
            if not self.reference_image_paths:
                raise ValueError("ref2va 至少需要一张 reference_image_paths。")
            if self.first_frame_path or self.last_frame_path:
                raise ValueError("ref2va 使用 reference_image_paths，不能传首尾帧路径。")
        return self


def module_available(module_name: str) -> bool:
    """只检查模块规格，不导入可能初始化 CUDA 的运行时。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def comfyui_modules_available() -> dict[str, bool]:
    """在 ComfyUI 独立解释器中检查模块元数据，不导入 Torch 或初始化 CUDA。"""
    if not Path(COMFYUI_PYTHON).is_file() or not os.access(COMFYUI_PYTHON, os.X_OK):
        return {name: False for name in COMFYUI_REQUIRED_MODULES}
    script = (
        "import importlib.util, json; "
        f"names = {COMFYUI_REQUIRED_MODULES!r}; "
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))"
    )
    try:
        result = subprocess.run(
            [COMFYUI_PYTHON, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        available = json.loads(result.stdout)
        return {name: bool(available.get(name)) for name in COMFYUI_REQUIRED_MODULES}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        LOGGER.warning("检查 MiniMax H3 ComfyUI Python 包失败: %s", exc)
        return {name: False for name in COMFYUI_REQUIRED_MODULES}


def image_path(full_path: str) -> Path:
    """将上传逻辑标识映射到图片目录，避免读取服务器任意路径。"""
    return IMAGE_DIR / image_storage_name(full_path)


def gpu_runtime_lock(label: str):
    """使用共享有界 GPU 锁串行调度重型视频 worker。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label, timeout_seconds=GPU_LOCK_WAIT_TIMEOUT)


def wait_after_cuda_release() -> None:
    """worker 退出后等待 CUDA 资源归还，再释放共享 GPU 锁。"""
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


class MiniMaxH3WorkerManager:
    """编排一次性 H3 worker，检查并原子保存结果。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: MiniMaxH3Request) -> dict[str, Any]:
        """预检权重、ComfyUI、GPU 余量和上传图片，再建立 worker 输入。"""
        image_paths: dict[str, Path] = {}
        if request.first_frame_path:
            image_paths["first"] = image_path(request.first_frame_path)
        if request.last_frame_path:
            image_paths["last"] = image_path(request.last_frame_path)
        for index, full_path in enumerate(request.reference_image_paths):
            image_paths[f"ref_{index}"] = image_path(full_path)
        for path in image_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"请先上传请求使用的参考图片: {path.name}")
            image_info(path)

        missing_files = [
            f"{name}: {path}"
            for name, path in REQUIRED_MODEL_FILES.items()
            if (request.include_audio or name != "audio_vae") and not path.is_file()
        ]
        comfy_root = Path(COMFYUI_CODE_PATH)
        required_comfy_paths = (
            comfy_root / "main.py",
            comfy_root / "comfy_extras/nodes_minimax_h3.py",
            Path(COMFYUI_PYTHON),
            Path(COMFYUI_PLUGIN_DIR) / "__init__.py",
        )
        missing_comfy = [str(path) for path in required_comfy_paths if not path.is_file()]
        missing_modules = [name for name in REQUIRED_MODULES if not module_available(name)]
        comfy_modules = comfyui_modules_available()
        missing_comfy_modules = [name for name, available in comfy_modules.items() if not available]
        if missing_files or missing_comfy or missing_modules or missing_comfy_modules:
            details = []
            if missing_files:
                details.append("缺少模型文件: " + ", ".join(missing_files))
            if missing_comfy:
                details.append("缺少 ComfyUI 运行文件: " + ", ".join(missing_comfy))
            if missing_modules:
                details.append("缺少 HTTP/图片依赖: " + ", ".join(missing_modules))
            if missing_comfy_modules:
                details.append("缺少 ComfyUI Python 依赖: " + ", ".join(missing_comfy_modules))
            raise RuntimeError("MiniMax H3 运行环境不完整；" + "；".join(details))
        if not os.access(COMFYUI_PYTHON, os.X_OK):
            raise RuntimeError(f"ComfyUI Python 解释器不可执行: {COMFYUI_PYTHON}")
        if not Path(WORKER_SCRIPT).is_file():
            raise RuntimeError(f"MiniMax H3 worker 脚本不存在: {WORKER_SCRIPT}")

        memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
        if memory["free_mib"] < MIN_FREE_VRAM_MIB:
            raise RuntimeError(
                "MiniMax H3 空闲显存不足："
                f"当前 {memory['free_mib']} MiB，要求至少 {MIN_FREE_VRAM_MIB} MiB。"
            )
        ffprobe_bin = shutil_which(FFPROBE_BIN)
        if not ffprobe_bin:
            raise RuntimeError("MiniMax H3 服务需要可用的 ffprobe。")

        dit_filename = (
            "minimax_h3_fl2va_dit_16g.safetensors"
            if request.mode == "fl2va"
            else "minimax_h3_ref2va_dit_16g.safetensors"
        )
        frame_count = max(5, round(request.duration_seconds * DEFAULT_FPS))
        frame_count += (5 - frame_count % 17) % 17
        payload = request.model_dump()
        payload.update(
            {
                "image_paths": {key: str(path) for key, path in image_paths.items()},
                "model_dir": MODEL_DIR,
                "dit_filename": dit_filename,
                "comfyui_code_path": COMFYUI_CODE_PATH,
                "comfyui_python": COMFYUI_PYTHON,
                "comfyui_plugin_dir": COMFYUI_PLUGIN_DIR,
                "worker_tmp_dir": str(WORKER_TMP_DIR),
                "local_files_only": LOCAL_FILES_ONLY,
                "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                "request_timeout": REQUEST_TIMEOUT,
                "comfyui_startup_timeout": COMFYUI_STARTUP_TIMEOUT,
                "reserve_vram_gib": RESERVED_VRAM_GIB,
                "frame_count": frame_count,
                "frame_rate": DEFAULT_FPS,
                "steps": DEFAULT_STEPS,
                "ffprobe_bin": ffprobe_bin,
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """启动隔离 worker，校验视频/音频流后原子保存 MP4。"""
        python_executable = sys.executable
        if not python_executable or not Path(python_executable).is_file():
            raise RuntimeError("未找到 MiniMax H3 服务 uv 环境的 Python 解释器。")
        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="minimax_h3_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="minimax_h3_out_",
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
                terminate_process_group(process, "MiniMax H3")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"MiniMax H3 worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc
            if stdout.strip():
                LOGGER.info("MiniMax H3 worker: %s", stdout.rstrip())
            LOGGER.info(
                "MiniMax H3 worker 退出码=%s，耗时 %.2fs",
                process.returncode,
                time.perf_counter() - started,
            )
            if process.returncode != 0:
                excerpt = " | ".join(
                    line.strip() for line in (stderr or stdout).splitlines() if line.strip()
                )
                raise RuntimeError(excerpt[-6000:] or "MiniMax H3 worker 执行失败。")
            temporary_video = Path(output_path)
            validate_video(
                temporary_video,
                str(payload["ffprobe_bin"]),
                expect_audio=bool(payload.get("include_audio", False)),
            )
            saved = persist_video(temporary_video, OUTPUT_DIR)
            self.last_output_path = str(saved)
            self.last_error = None
            return saved
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "MiniMax H3")
            Path(request_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


manager = MiniMaxH3WorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """报告权重、ComfyUI、自定义节点、依赖和 GPU 状态，不加载模型。"""
    cuda = cuda_status(CUDA_VISIBLE_DEVICES)
    dependencies = {name: module_available(name) for name in REQUIRED_MODULES}
    comfy_dependencies = comfyui_modules_available()
    model_files = {name: path.is_file() for name, path in REQUIRED_MODEL_FILES.items()}
    comfy_paths = {
        "source": Path(COMFYUI_CODE_PATH) / "main.py",
        "h3_core_node": Path(COMFYUI_CODE_PATH) / "comfy_extras/nodes_minimax_h3.py",
        "python": Path(COMFYUI_PYTHON),
        "sol_plugin": Path(COMFYUI_PLUGIN_DIR) / "__init__.py",
    }
    comfy_files = {name: path.is_file() for name, path in comfy_paths.items()}
    ffprobe_ready = bool(shutil_which(FFPROBE_BIN))
    runtime_ready = (
        all(available for name, available in model_files.items() if name != "audio_vae")
        and all(comfy_files.values())
        and all(dependencies.values())
        and all(comfy_dependencies.values())
        and ffprobe_ready
        and cuda["available"]
        and Path(WORKER_SCRIPT).is_file()
    )
    native_audio_ready = runtime_ready and model_files["audio_vae"]
    try:
        gpu_memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
    except RuntimeError as exc:
        gpu_memory = {"error": str(exc)}
    return {
        "code": 200,
        "paths": {
            "model_dir": MODEL_DIR,
            "comfyui_code_path": COMFYUI_CODE_PATH,
            "comfyui_python": COMFYUI_PYTHON,
            "image_dir": str(IMAGE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
        },
        "available": {
            "worker_script": Path(WORKER_SCRIPT).is_file(),
            "model_files": model_files,
            "comfyui_files": comfy_files,
            "dependencies": dependencies,
            "comfyui_dependencies": comfy_dependencies,
            "ffprobe": ffprobe_ready,
            "cuda": cuda["available"],
            "selected_provider": "comfyui_local",
            "selected_provider_ready": runtime_ready,
            "native_audio_ready": native_audio_ready,
        },
        "cuda": cuda,
        "gpu_memory": gpu_memory,
        "runtime": {
            "worker_runtime": "uv",
            "comfyui_runtime": COMFYUI_PYTHON,
            "model_lifecycle": "one request -> one worker -> one ComfyUI child -> exit",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "min_free_vram_mib": MIN_FREE_VRAM_MIB,
            "reserved_vram_gib": RESERVED_VRAM_GIB,
            "precision": "repository HiFi ConvRot weights with ComfyUI native H3",
            "sampler": "res_multistep/simple",
            "steps": DEFAULT_STEPS,
            "resolution": f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
            "duration_seconds": DEFAULT_DURATION_SECONDS,
            "frame_rate": DEFAULT_FPS,
            "native_audio": True,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"minimax_h3": manager.last_error},
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


@app.post("/v1/minimax/H3-16G-HiFi")
def generate(request: MiniMaxH3Request):
    """按安全分辨率/时长和显存门槛运行 H3，按请求决定是否封装原生音轨。"""
    try:
        payload = manager.build_worker_payload(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("minimax_h3_16g_hifi/generate") as metrics:
            try:
                memory = query_gpu_memory(CUDA_VISIBLE_DEVICES)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            if memory["free_mib"] < MIN_FREE_VRAM_MIB:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "MiniMax H3 空闲显存不足："
                        f"当前 {memory['free_mib']} MiB，要求至少 {MIN_FREE_VRAM_MIB} MiB。"
                    ),
                )
            try:
                started = time.perf_counter()
                saved = manager.run_worker(payload)
                metrics.worker_seconds = time.perf_counter() - started
                return FileResponse(saved, media_type="video/mp4", filename=saved.name)
            finally:
                wait_after_cuda_release()
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("MiniMax H3 generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   LLM Local Backend MiniMax H3 16G HiFi")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] ComfyUI: {COMFYUI_CODE_PATH}")
    print(f"[配置] ComfyUI Python: {COMFYUI_PYTHON}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, cuda_visible_devices={CUDA_VISIBLE_DEVICES}, "
        f"local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
