#!/usr/bin/env python3
"""用于共享存储和运行时诊断的轻量控制面 API。

模型推理不在这里执行。各模型服务自行管理推理生命周期；本进程只保留
周边 WebUI 使用的 8300 端口健康检查、上传、文件检查和 CPU 音频导出工具。
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from gpu_runtime import cuda_status
from render_progress import RenderProgressError, RenderProgressRegistry
from spatial_audio import (
    SPATIAL_EXPORT_FORMATS,
    SPATIAL_EXPORT_PROFILES,
    SpatialAudioExportError,
    SpatialAudioProcessor,
)
from spatial_schema import MANIFEST_VERSION, SpatialRenderManifest
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from steam_audio_render import SteamAudioRenderError, SteamAudioRenderProcessor

from llm_local_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    StagedUpload,
    UploadPolicy,
    stage_audio_upload,
    storage_disk_status,
)

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MAIN_DIR.parent


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，统一得到可用于存储配置的绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(PROJECT_DIR / "storage")))
TIMBRE_STORAGE_DIR = expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(Path(STORAGE_DIR) / "timbre")))
TIMBRE_REFERENCE_DIR = str(Path(TIMBRE_STORAGE_DIR) / ".references")
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", str(Path(STORAGE_DIR) / "clone")))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(Path(STORAGE_DIR) / ".cache/runtime"))
)
SPATIAL_EXPORT_CACHE_DIR = expand_path(
    os.getenv(
        "SPATIAL_EXPORT_CACHE_DIR",
        str(Path(RUNTIME_CACHE_DIR) / "spatial_exports"),
    )
)
SPATIAL_EXPORT_MAX_BYTES = int(os.getenv("SPATIAL_EXPORT_MAX_BYTES", str(512 * 1024 * 1024)))
SPATIAL_EXPORT_TIMEOUT = float(os.getenv("SPATIAL_EXPORT_TIMEOUT", "600"))
SPATIAL_EXPORT_FFMPEG_BIN = os.getenv("SPATIAL_EXPORT_FFMPEG_BIN", "ffmpeg")
STEAM_AUDIO_RENDER_CACHE_DIR = expand_path(
    os.getenv(
        "STEAM_AUDIO_RENDER_CACHE_DIR",
        str(Path(RUNTIME_CACHE_DIR) / "steam_audio_renders"),
    )
)
_default_renderer_path = PROJECT_DIR / "model-audio" / "steam_audio_renderer" / "build"
if os.name == "nt":
    _default_renderer_path = _default_renderer_path / "Release" / "steam-audio-render.exe"
else:
    _default_renderer_path = _default_renderer_path / "steam-audio-render"
STEAM_AUDIO_RENDERER_BIN = os.getenv(
    "STEAM_AUDIO_RENDERER_BIN",
    str(_default_renderer_path),
)
STEAM_AUDIO_SDK_DIR = os.getenv("STEAM_AUDIO_SDK_DIR") or None
STEAM_AUDIO_HRTF_PATH = os.getenv("STEAM_AUDIO_HRTF_PATH") or None
STEAM_AUDIO_RENDER_TIMEOUT = float(os.getenv("STEAM_AUDIO_RENDER_TIMEOUT", "900"))
STEAM_AUDIO_RENDER_MAX_ASSETS = int(os.getenv("STEAM_AUDIO_RENDER_MAX_ASSETS", "500"))
STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES = int(
    os.getenv("STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES", str(8 * 1024 * 1024))
)
STEAM_AUDIO_RENDER_MAX_BYTES = int(
    os.getenv("STEAM_AUDIO_RENDER_MAX_BYTES", str(2 * 1024 * 1024 * 1024))
)
STEAM_AUDIO_RENDER_THREADS = int(os.getenv("STEAM_AUDIO_RENDER_THREADS", "4"))
STEAM_AUDIO_PROGRESS_RETENTION_SECONDS = float(
    os.getenv("STEAM_AUDIO_PROGRESS_RETENTION_SECONDS", "3600")
)
if SPATIAL_EXPORT_MAX_BYTES <= 0:
    raise ValueError("SPATIAL_EXPORT_MAX_BYTES 必须为正整数。")
if SPATIAL_EXPORT_TIMEOUT <= 0:
    raise ValueError("SPATIAL_EXPORT_TIMEOUT 必须大于 0。")
if STEAM_AUDIO_RENDER_TIMEOUT <= 0:
    raise ValueError("STEAM_AUDIO_RENDER_TIMEOUT 必须大于 0。")
if STEAM_AUDIO_RENDER_MAX_ASSETS <= 0:
    raise ValueError("STEAM_AUDIO_RENDER_MAX_ASSETS 必须为正整数。")
if STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES <= 0:
    raise ValueError("STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES 必须为正整数。")
if STEAM_AUDIO_RENDER_MAX_BYTES <= 0:
    raise ValueError("STEAM_AUDIO_RENDER_MAX_BYTES 必须为正整数。")
if STEAM_AUDIO_RENDER_THREADS <= 0:
    raise ValueError("STEAM_AUDIO_RENDER_THREADS 必须为正整数。")
if STEAM_AUDIO_PROGRESS_RETENTION_SECONDS <= 0:
    raise ValueError("STEAM_AUDIO_PROGRESS_RETENTION_SECONDS 必须大于 0。")
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", str(Path(RUNTIME_CACHE_DIR) / "gpu-runtime.lock"))
)
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8300"))
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

for directory in (
    TIMBRE_STORAGE_DIR,
    TIMBRE_REFERENCE_DIR,
    CLONE_STORAGE_DIR,
    PROMPTS_DIR,
    RUNTIME_CACHE_DIR,
    SPATIAL_EXPORT_CACHE_DIR,
    STEAM_AUDIO_RENDER_CACHE_DIR,
):
    os.makedirs(directory, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
spatial_audio_processor = SpatialAudioProcessor(
    SPATIAL_EXPORT_CACHE_DIR,
    ffmpeg_bin=SPATIAL_EXPORT_FFMPEG_BIN,
    timeout_seconds=SPATIAL_EXPORT_TIMEOUT,
)
steam_audio_processor = SteamAudioRenderProcessor(
    STEAM_AUDIO_RENDER_CACHE_DIR,
    renderer_bin=STEAM_AUDIO_RENDERER_BIN,
    ffmpeg_bin=SPATIAL_EXPORT_FFMPEG_BIN,
    timeout_seconds=STEAM_AUDIO_RENDER_TIMEOUT,
    mastering_timeout_seconds=SPATIAL_EXPORT_TIMEOUT,
    threads=STEAM_AUDIO_RENDER_THREADS,
    sdk_dir=STEAM_AUDIO_SDK_DIR,
    hrtf_path=STEAM_AUDIO_HRTF_PATH,
)
steam_audio_progress = RenderProgressRegistry(
    retention_seconds=STEAM_AUDIO_PROGRESS_RETENTION_SECONDS
)


def spatial_export_available() -> bool:
    """检查配置的 FFmpeg 命令是否能从当前控制面环境解析。"""
    return shutil.which(SPATIAL_EXPORT_FFMPEG_BIN) is not None


app = FastAPI(title="LLM Local Backend Control Plane")


class ForceCORS(BaseHTTPMiddleware):
    """为 WebUI 请求补充宽松的跨域响应头，并快速处理预检请求。"""

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
        response.headers["Access-Control-Expose-Headers"] = (
            "Content-Disposition, X-Audio-Sample-Rate, X-Audio-Profile, X-Audio-Format, "
            "X-Spatial-Engine, X-Spatial-Manifest-Version, X-Spatial-Job-ID"
        )
        return response


app.add_middleware(ForceCORS)


def hash_filename(filename: str) -> str:
    """将 WebUI 的逻辑路径映射为稳定文件名，避免直接使用用户输入作路径。"""
    return reference_store.clone_path(filename).name


def prompt_audio_path(filename: str) -> Path:
    """解析普通克隆上传，或解析预览时引用的音色设计音频。"""
    return reference_store.prompt_audio_path(filename)


def store_uploaded_audio(staged: StagedUpload, full_path: str) -> dict[str, object]:
    """在线程池中原子提交流式暂存的参考音频。"""
    return reference_store.commit_staged_upload(staged, full_path)


@app.get("/v1/health")
@app.get("/v1/control")
def health():
    """返回控制面、存储目录和 GPU 可见性的诊断信息。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "storage_dir": STORAGE_DIR,
            "timbre_storage_dir": TIMBRE_STORAGE_DIR,
            "clone_storage_dir": CLONE_STORAGE_DIR,
            "prompts_dir": PROMPTS_DIR,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "spatial_export_cache_dir": SPATIAL_EXPORT_CACHE_DIR,
            "steam_audio_render_cache_dir": STEAM_AUDIO_RENDER_CACHE_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "last_errors": {},
        "offline": {
            "local_files_only": LOCAL_FILES_ONLY,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
            "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
        },
        "runtime": {
            "service_role": "control_plane",
            "model_inference": "delegated to standalone services",
        },
        "audio_export": {
            "available": spatial_export_available(),
            "ffmpeg_bin": SPATIAL_EXPORT_FFMPEG_BIN,
            "sample_rate": 48000,
            "profiles": list(SPATIAL_EXPORT_PROFILES),
            "formats": list(SPATIAL_EXPORT_FORMATS),
            "max_bytes": SPATIAL_EXPORT_MAX_BYTES,
            "timeout_seconds": SPATIAL_EXPORT_TIMEOUT,
        },
        "spatial_renderer": {
            **steam_audio_processor.diagnostics(),
            "manifest_version": MANIFEST_VERSION,
            "max_assets": STEAM_AUDIO_RENDER_MAX_ASSETS,
            "max_manifest_bytes": STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES,
            "max_bytes": STEAM_AUDIO_RENDER_MAX_BYTES,
        },
    }


@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    """上传克隆参考音频，但不复制已有的音色设计资产。"""
    try:
        staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.post("/v1/audio/export")
async def export_audio(
    audio: UploadFile = File(...),
    profile: Literal["standard", "balanced", "immersive"] = Form("balanced"),
    output_format: Literal["wav", "mp3"] = Form("wav"),
):
    """流式接收 WebUI 混音，经 CPU FFmpeg 处理后返回 48 kHz 成品。"""
    upload_policy = UploadPolicy(max_bytes=SPATIAL_EXPORT_MAX_BYTES)
    try:
        staged = await stage_audio_upload(
            audio,
            Path(SPATIAL_EXPORT_CACHE_DIR) / "uploads",
            policy=upload_policy,
        )
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    try:
        result = await run_in_threadpool(
            spatial_audio_processor.export,
            staged,
            profile=profile,
            output_format=output_format,
        )
    except SpatialAudioExportError as exc:
        staged.path.unlink(missing_ok=True)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except Exception:
        staged.path.unlink(missing_ok=True)
        raise

    return FileResponse(
        path=result.path,
        media_type=result.media_type,
        filename=result.download_name,
        headers={
            "X-Audio-Sample-Rate": "48000",
            "X-Audio-Profile": profile,
            "X-Audio-Format": output_format,
        },
        background=BackgroundTask(result.cleanup),
    )


@app.get("/v1/audio/spatial/render/progress/{job_id}")
def get_spatial_render_progress(job_id: str):
    """返回当前进程内空间音频任务的阶段、百分比和终态诊断。"""
    snapshot = steam_audio_progress.get(job_id)
    if snapshot is None:
        return JSONResponse(status_code=404, content={"detail": "未找到该空间音频任务。"})
    return snapshot


@app.post("/v1/audio/spatial/render")
async def render_spatial_audio(
    manifest: str = Form(...),
    assets: list[UploadFile] = File(...),
    profile: Literal["standard", "balanced", "immersive"] = Form("balanced"),
    output_format: Literal["wav", "mp3"] = Form("wav"),
    job_id: str | None = Form(None),
):
    """接收对象级资产并调用 CPU Steam Audio renderer 生成正式双耳成品。"""
    render_job_id = job_id or f"server-{uuid.uuid4().hex}"
    try:
        steam_audio_progress.start(render_job_id, "请求已接收，正在校验 Manifest 与资产")
    except RenderProgressError as exc:
        status_code = 409 if "已存在" in str(exc) else 422
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    def fail_response(status_code: int, detail: str, **extra: object) -> JSONResponse:
        """记录任务失败终态并构造包含 job ID 的错误响应。"""
        steam_audio_progress.fail(render_job_id, detail)
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail, "job_id": render_job_id, **extra},
            headers={"X-Spatial-Job-ID": render_job_id},
        )

    if len(manifest.encode("utf-8")) > STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES:
        return fail_response(413, "Manifest 超过大小限制。")
    try:
        render_manifest = SpatialRenderManifest.model_validate_json(manifest)
    except ValueError as exc:
        return fail_response(422, f"Manifest 无效: {exc}")

    if len(assets) > STEAM_AUDIO_RENDER_MAX_ASSETS:
        return fail_response(
            413,
            f"上传资产数量超过 {STEAM_AUDIO_RENDER_MAX_ASSETS} 个限制。",
        )

    filenames = [upload.filename or "" for upload in assets]
    if not all(filenames):
        return fail_response(422, "每个上传资产都必须有文件名。")
    if len(set(filenames)) != len(filenames):
        return fail_response(422, "上传资产文件名不能重复。")
    referenced_filenames = render_manifest.referenced_filenames()
    uploaded_filenames = set(filenames)
    if uploaded_filenames != referenced_filenames:
        missing = sorted(referenced_filenames - uploaded_filenames)
        unused = sorted(uploaded_filenames - referenced_filenames)
        return fail_response(
            422,
            "上传资产必须与 Manifest 引用一一对应。",
            missing=missing,
            unused=unused,
        )

    staged_assets: dict[str, StagedUpload] = {}
    upload_policy = UploadPolicy(max_bytes=STEAM_AUDIO_RENDER_MAX_BYTES)
    total_bytes = 0
    steam_audio_progress.update(
        render_job_id,
        stage="staging",
        progress=8,
        message=f"正在暂存 {len(assets)} 个上传资产",
    )
    try:
        for index, upload in enumerate(assets, start=1):
            staged = await stage_audio_upload(
                upload,
                Path(STEAM_AUDIO_RENDER_CACHE_DIR) / "uploads",
                policy=upload_policy,
            )
            staged_assets[upload.filename or ""] = staged
            total_bytes += staged.size_bytes
            if total_bytes > STEAM_AUDIO_RENDER_MAX_BYTES:
                raise AudioUploadError(
                    "空间音频上传总量超过限制。",
                    status_code=413,
                )
            steam_audio_progress.update(
                render_job_id,
                stage="staging",
                progress=8 + round(10 * index / len(assets)),
                message=f"已暂存上传资产 {index}/{len(assets)}",
            )
    except AudioUploadError as exc:
        for staged in staged_assets.values():
            staged.path.unlink(missing_ok=True)
        return fail_response(exc.status_code, str(exc))
    except Exception as exc:
        for staged in staged_assets.values():
            staged.path.unlink(missing_ok=True)
        steam_audio_progress.fail(render_job_id, f"暂存上传资产时发生异常: {exc}")
        raise

    def report_progress(stage: str, progress: int, message: str) -> None:
        steam_audio_progress.update(
            render_job_id,
            stage=stage,
            progress=progress,
            message=message,
        )

    try:
        result = await run_in_threadpool(
            steam_audio_processor.render,
            render_manifest,
            staged_assets,
            profile=profile,
            output_format=output_format,
            progress_callback=report_progress,
        )
    except SteamAudioRenderError as exc:
        for staged in staged_assets.values():
            staged.path.unlink(missing_ok=True)
        return fail_response(503, str(exc))
    except Exception as exc:
        for staged in staged_assets.values():
            staged.path.unlink(missing_ok=True)
        steam_audio_progress.fail(render_job_id, f"空间音频渲染发生异常: {exc}")
        raise

    steam_audio_progress.succeed(render_job_id, "渲染完成，文件已可下载")
    return FileResponse(
        path=result.path,
        media_type=result.media_type,
        filename=result.download_name,
        headers={
            "X-Audio-Sample-Rate": "48000",
            "X-Audio-Profile": profile,
            "X-Audio-Format": output_format,
            "X-Spatial-Engine": "steam-audio",
            "X-Spatial-Manifest-Version": MANIFEST_VERSION,
            "X-Spatial-Job-ID": render_job_id,
        },
        background=BackgroundTask(result.cleanup),
    )


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    exists = prompt_audio_path(file_name).is_file()
    return {"code": 200 if exists else 404, "exists": exists}


if __name__ == "__main__":
    print("==================================================")
    print("   LLM Local Backend control plane (no model inference)")
    print("==================================================")
    print(f"[配置] storage: {STORAGE_DIR}")
    print(f"[配置] prompts: {PROMPTS_DIR}")
    print(f"[配置] host={API_HOST}, port={API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
