#!/usr/bin/env python3
"""SheetSage2 音频转乐谱 HTTP 服务；重型依赖只在一次性 worker 中导入。"""

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
import uuid
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import AudioUploadError, GpuLockTimeoutError, stage_audio_upload
from unitale_runtime import gpu_runtime_lock as shared_gpu_runtime_lock

from runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    """解析布尔环境变量。"""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
)
UPLOAD_STAGING_DIR = Path(
    expand_path(
        os.getenv("SHEETSAGE2_UPLOAD_STAGING_DIR", str(RUNTIME_CACHE_DIR / "sheetsage2_uploads"))
    )
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MODEL_DIR = expand_path(
    os.getenv("SHEETSAGE2_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "m-a-p/SheetSage2"))
)
BASE_MODEL_DIR = expand_path(
    os.getenv(
        "SHEETSAGE2_BASE_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "m-a-p/MERT-v2-FullSong"),
    )
)
WORKER_TMP_DIR = Path(
    expand_path(
        os.getenv("SHEETSAGE2_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "sheetsage2_worker"))
    )
)
OUTPUT_DIR = Path(expand_path(os.getenv("SHEETSAGE2_OUTPUT_DIR", str(STORAGE_DIR / "sheetsage2"))))
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("SHEETSAGE2_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("SHEETSAGE2_PORT", os.getenv("PORT", "8315")))
REQUEST_TIMEOUT = float(os.getenv("SHEETSAGE2_REQUEST_TIMEOUT", "1800"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
DEVICE = os.getenv("SHEETSAGE2_DEVICE", "cuda:0")
DEFAULT_DTYPE = os.getenv("SHEETSAGE2_DTYPE", "bf16")
DEFAULT_PRESET = os.getenv("SHEETSAGE2_PRESET", "default")

if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
if not math.isfinite(REQUEST_TIMEOUT) or REQUEST_TIMEOUT <= 0:
    raise ValueError("SHEETSAGE2_REQUEST_TIMEOUT 必须是正数。")
if not math.isfinite(GPU_LOCK_WAIT_TIMEOUT) or GPU_LOCK_WAIT_TIMEOUT <= 0:
    raise ValueError("GPU_LOCK_WAIT_TIMEOUT 必须是正数，避免 GPU 队列无限等待。")
if DEFAULT_DTYPE not in {"bf16", "fp32"}:
    raise ValueError("SHEETSAGE2_DTYPE 仅支持 bf16 或 fp32。")
if DEFAULT_PRESET not in {"default", "paper"}:
    raise ValueError("SHEETSAGE2_PRESET 仅支持 default 或 paper。")

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "sheetsage2_hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_DIR / "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for path in (
    UPLOAD_STAGING_DIR,
    WORKER_TMP_DIR,
    OUTPUT_DIR,
    Path(GPU_LOCK_FILE).parent,
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(os.environ["NUMBA_CACHE_DIR"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["XDG_CACHE_HOME"]),
):
    path.mkdir(parents=True, exist_ok=True)

REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "configuration_mert2.py",
    "configuration_sheetsage2.py",
    "modeling_mert2.py",
    "modeling_sheetsage2.py",
    "tokenization_sheetsage2.py",
    "processing_sheetsage2.py",
    "audio_sheetsage2.py",
    "chord_spelling_sheetsage2.py",
    "durations_sheetsage2.py",
    "exports_sheetsage2.py",
    "generation_sheetsage2.py",
    "io_sheetsage2.py",
    "labels_sheetsage2.py",
    "midi_sheetsage2.py",
    "notation_sheetsage2.py",
    "pipeline_sheetsage2.py",
    "rendering_sheetsage2.py",
    "schema_sheetsage2.py",
    "tensors_sheetsage2.py",
)
REQUIRED_BASE_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "configuration_mert2.py",
    "modeling_mert2.py",
)
REQUIRED_DEPENDENCIES = (
    "mir_eval",
    "mido",
    "numpy",
    # pretty_midi 的传递导入依赖，不能仅检查 pretty_midi 自身是否已安装。
    "pkg_resources",
    "pretty_midi",
    "safetensors",
    "scipy",
    "torch",
    "torchaudio",
    "transformers",
)


class SheetSage2Request(BaseModel):
    """SheetSage2 原生转谱选项。"""

    model_config = ConfigDict(extra="forbid")

    dtype: Literal["bf16", "fp32"] = DEFAULT_DTYPE  # type: ignore[assignment]
    preset: Literal["default", "paper"] = DEFAULT_PRESET  # type: ignore[assignment]
    max_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    overlap_seconds: float | None = Field(default=None, ge=0, lt=300, allow_inf_nan=False)
    lookahead_seconds: float | None = Field(default=None, ge=0, lt=300, allow_inf_nan=False)
    prompts: list[str] | None = Field(default=None, min_length=1, max_length=32)
    melody_only: bool = False
    export_logits: bool = False
    export_scores: bool = False
    export_embeddings: bool = False
    output_hidden_states: bool = False

    @model_validator(mode="after")
    def validate_window_options(self) -> SheetSage2Request:
        """在排队前校验模型的窗口与提示词约束。"""
        if self.prompts is not None and any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("prompts 中不能包含空字符串。")
        if self.prompts is not None and "timestamp" not in self.prompts:
            raise ValueError("prompts 必须包含 timestamp 才能生成时间标注。")

        default_overlap, default_lookahead = (
            (100.0, 0.0) if self.preset == "paper" else (200.0, 100.0)
        )
        overlap = default_overlap if self.overlap_seconds is None else self.overlap_seconds
        lookahead = default_lookahead if self.lookahead_seconds is None else self.lookahead_seconds
        if not 0 <= lookahead <= overlap < 300:
            raise ValueError("窗口参数必须满足 0 <= lookahead_seconds <= overlap_seconds < 300。")
        if self.preset == "paper" and (overlap, lookahead) != (100.0, 0.0):
            raise ValueError("paper preset 固定 overlap_seconds=100 和 lookahead_seconds=0。")
        return self


def module_available(module_name: str) -> bool:
    """只检查模块规格，不在 API 进程导入 Torch 或模型代码。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def directory_status(directory: str, required_files: tuple[str, ...]) -> dict[str, Any]:
    """检查本地模型目录中所需文件是否齐全。"""
    root = Path(directory)
    missing = [name for name in required_files if not (root / name).is_file()]
    return {
        "path": str(root),
        "complete": root.is_dir() and not missing,
        "missing": missing if root.is_dir() else ["<directory>", *missing],
    }


def model_status() -> dict[str, dict[str, Any]]:
    """检查 SheetSage2 与其固定 MERT-v2 父模型，不加载权重。"""
    return {
        "sheetsage2": directory_status(MODEL_DIR, REQUIRED_MODEL_FILES),
        "mert_v2_fullsong": directory_status(BASE_MODEL_DIR, REQUIRED_BASE_MODEL_FILES),
    }


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行执行 SheetSage2 GPU 推理。"""
    return shared_gpu_runtime_lock(
        GPU_LOCK_FILE,
        label,
        timeout_seconds=GPU_LOCK_WAIT_TIMEOUT,
    )


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 资源归还，再允许下一任务进入队列。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        LOGGER.info("等待 %.1fs 释放显存: %s", CUDA_RELEASE_DELAY, label)
    time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(message: str) -> str:
    """缩短 worker traceback，避免异常响应携带过长的日志。"""
    excerpt = message.strip()
    return excerpt[-4_000:] if excerpt else "SheetSage2 worker 失败。"


def persist_archive_file(source: Path, output_dir: Path) -> Path:
    """将 ZIP 流式复制到输出目录，并用原子替换提交成品。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sheetsage2_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:12]}.zip"
    destination = output_dir / filename
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".sheetsage2_",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination_file:
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.replace(temporary_path, destination)
        return destination
    finally:
        temporary_path.unlink(missing_ok=True)
        source.unlink(missing_ok=True)


class SheetSage2WorkerManager:
    """管理单请求 worker，避免模型权重与 CUDA 上下文跨请求常驻。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def build_worker_payload(
        self,
        request: SheetSage2Request,
        audio_path: Path,
    ) -> dict[str, Any]:
        """完成 CPU/文件预检，并构造一次性 worker 请求。"""
        if not audio_path.is_file():
            raise FileNotFoundError(f"待转写音频不存在: {audio_path}")
        if not os.path.isfile(WORKER_SCRIPT):
            raise FileNotFoundError(f"SheetSage2 worker 不存在: {WORKER_SCRIPT}")
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("未找到 ffmpeg；SheetSage2 需要 FFmpeg 解码音频。")

        models = model_status()
        incomplete = [name for name, status in models.items() if not status["complete"]]
        if incomplete:
            details = {name: models[name]["missing"] for name in incomplete}
            raise FileNotFoundError(f"SheetSage2 本地模型文件不完整: {details}")

        missing_dependencies = [
            name for name in REQUIRED_DEPENDENCIES if not module_available(name)
        ]
        if missing_dependencies:
            raise FileNotFoundError(
                "SheetSage2 运行依赖未安装: "
                f"{', '.join(missing_dependencies)}。请先 uv sync --project SheetSage2 --locked。"
            )

        payload = request.model_dump(exclude_none=True)
        payload.update(
            {
                "audio_path": str(audio_path),
                "base_model_dir": BASE_MODEL_DIR,
                "device": DEVICE,
                "local_files_only": LOCAL_FILES_ONLY,
                "model_dir": MODEL_DIR,
                "worker_tmp_dir": str(WORKER_TMP_DIR),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> Path:
        """在本服务 uv 环境中执行隔离转写，并返回临时 ZIP 路径。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 SheetSage2 uv 环境的 Python 解释器。")

        request_fd, request_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="sheetsage2_req_",
            suffix=".json",
        )
        output_fd, output_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="sheetsage2_out_",
            suffix=".zip",
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
                    python_executable,
                    WORKER_SCRIPT,
                    "--input-json",
                    str(request_path),
                    "--output-zip",
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
                raise TimeoutError(f"SheetSage2 推理超过 {REQUEST_TIMEOUT:.0f} 秒时限。") from exc

            if stdout.strip():
                LOGGER.info("SheetSage2 worker: %s", stdout.rstrip())
            if stderr.strip():
                LOGGER.warning("SheetSage2 worker stderr: %s", worker_error_excerpt(stderr))
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(f"{stdout}\n{stderr}"))
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("SheetSage2 worker 成功退出，但没有生成结果 ZIP。")

            succeeded = True
            self.last_error = None
            return output_path
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "SheetSage2")
            request_path.unlink(missing_ok=True)
            if not succeeded:
                output_path.unlink(missing_ok=True)


manager = SheetSage2WorkerManager()
app = FastAPI(title="Unitale SheetSage2 Music Transcription API")


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


def execute_transcription_payload(payload: dict[str, Any]) -> Path:
    """持有共享 GPU 锁执行转写并原子保存 ZIP 成品。"""
    with gpu_runtime_lock("sheetsage2/transcribe"):
        with manager.lock:
            try:
                temporary_archive = manager.run_worker(payload)
            finally:
                wait_after_cuda_release("after SheetSage2 worker")
    return persist_archive_file(temporary_archive, OUTPUT_DIR)


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """返回本地模型、依賴、FFmpeg 和 GPU 状态，不加载权重。"""
    gpu = cuda_status()
    models = model_status()
    dependencies = {name: module_available(name) for name in REQUIRED_DEPENDENCIES}
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ready = (
        os.path.isfile(WORKER_SCRIPT)
        and all(dependencies.values())
        and all(model["complete"] for model in models.values())
        and ffmpeg_available
        and gpu["available"]
    )
    return {
        "code": 200,
        "ready": ready,
        "paths": {
            "sheetsage2_model_dir": MODEL_DIR,
            "mert_v2_fullsong_model_dir": BASE_MODEL_DIR,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "upload_staging_dir": str(UPLOAD_STAGING_DIR),
            "output_dir": str(OUTPUT_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "ffmpeg": ffmpeg_available,
            "dependencies": dependencies,
            "models": models,
            "cuda": gpu["available"],
        },
        "cuda": gpu,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "gpu_lock_wait_timeout": GPU_LOCK_WAIT_TIMEOUT,
            "device": DEVICE,
            "dtype": DEFAULT_DTYPE,
            "preset": DEFAULT_PRESET,
            "gpu_scheduling": "shared exclusive file lock",
            "response": "application/zip with native ABC, MIDI and annotation files",
        },
        "last_errors": {"sheetsage2": manager.last_error},
    }


def build_request(
    *,
    dtype: str,
    preset: str,
    max_seconds: float | None,
    overlap_seconds: float | None,
    lookahead_seconds: float | None,
    prompts: list[str] | None,
    melody_only: bool,
    export_logits: bool,
    export_scores: bool,
    export_embeddings: bool,
    output_hidden_states: bool,
) -> SheetSage2Request:
    """通过同一 Pydantic schema 校验 multipart 字段。"""
    try:
        return SheetSage2Request(
            dtype=dtype,
            preset=preset,
            max_seconds=max_seconds,
            overlap_seconds=overlap_seconds,
            lookahead_seconds=lookahead_seconds,
            prompts=prompts,
            melody_only=melody_only,
            export_logits=export_logits,
            export_scores=export_scores,
            export_embeddings=export_embeddings,
            output_hidden_states=output_hidden_states,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False, include_input=False),
        ) from exc


@app.post("/v1/m-a-p/SheetSage2")
async def transcribe(
    audio: UploadFile = File(...),
    dtype: str = Form(DEFAULT_DTYPE),
    preset: str = Form(DEFAULT_PRESET),
    max_seconds: float | None = Form(None),
    overlap_seconds: float | None = Form(None),
    lookahead_seconds: float | None = Form(None),
    prompts: list[str] | None = Form(None),
    melody_only: bool = Form(False),
    export_logits: bool = Form(False),
    export_scores: bool = Form(False),
    export_embeddings: bool = Form(False),
    output_hidden_states: bool = Form(False),
) -> Response:
    """上传音乐并返回 SheetSage2 生成的 ABC、MIDI 与时间标注 ZIP。"""
    request = build_request(
        dtype=dtype,
        preset=preset,
        max_seconds=max_seconds,
        overlap_seconds=overlap_seconds,
        lookahead_seconds=lookahead_seconds,
        prompts=prompts,
        melody_only=melody_only,
        export_logits=export_logits,
        export_scores=export_scores,
        export_embeddings=export_embeddings,
        output_hidden_states=output_hidden_states,
    )
    try:
        staged = await stage_audio_upload(audio, UPLOAD_STAGING_DIR)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    audio_path = staged.path.with_suffix("")
    try:
        os.replace(staged.path, audio_path)
        try:
            payload = manager.build_worker_payload(request, audio_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if not cuda_status()["available"]:
            raise HTTPException(status_code=503, detail="SheetSage2 需要可用的 NVIDIA CUDA GPU。")

        try:
            archive_path = await run_in_threadpool(execute_transcription_payload, payload)
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TimeoutError as exc:
            LOGGER.exception("SheetSage2 request timed out")
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("SheetSage2 transcription failed")
            raise HTTPException(status_code=500, detail=worker_error_excerpt(str(exc))) from exc

        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=archive_path.name,
            headers={
                "X-SheetSage2-Audio-SHA256": staged.sha256,
                "X-SheetSage2-Model": "m-a-p/SheetSage2",
                "X-SheetSage2-Preset": request.preset,
            },
        )
    finally:
        staged.path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
