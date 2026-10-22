#!/usr/bin/env python3
"""独立 TIGER-DnR HTTP 服务。

API 进程只处理流式上传、GPU 排队、worker 生命周期及三 Stem 成品归档；Torch 和
上游 TIGER 代码仅在一次性 worker 中导入，以便健康检查不占用 GPU 或模型内存。
"""

from __future__ import annotations

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
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from inference_gateway import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    execute_localai_worker_file_reference,
    load_gateway_config,
)
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    gpu_runtime_lock,
    stage_audio_upload,
)
from llm_local_runtime.storage import close_upload_safely

from tiger_dnr_runtime import (
    cuda_status,
    expand_path,
    terminate_process_group,
    wait_after_cuda_release,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent.parent
STEM_NAMES = ("dialog", "effects", "music")
MODEL_SAMPLE_RATE = 44_100


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动脚本传入的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
TIGER_DNR_MODEL_DIR = Path(
    expand_path(
        os.getenv("TIGER_DNR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "JusperLee", "TIGER-DnR"))
    )
)
TIGER_DNR_SOURCE_DIR = Path(
    expand_path(os.getenv("TIGER_DNR_SOURCE_DIR", "~/.local/share/tiger-dnr/TIGER"))
)
TIGER_DNR_OUTPUT_DIR = Path(
    expand_path(os.getenv("TIGER_DNR_OUTPUT_DIR", str(STORAGE_DIR / "separation")))
)
RUNTIME_CACHE_DIR = Path(
    expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache" / "runtime")))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
WORKER_TMP_DIR = Path(
    expand_path(os.getenv("TIGER_DNR_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "tiger_dnr_worker")))
)
UPLOAD_TMP_DIR = Path(
    expand_path(os.getenv("TIGER_DNR_UPLOAD_TMP_DIR", str(RUNTIME_CACHE_DIR / "tiger_dnr_uploads")))
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("TIGER_DNR_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("TIGER_DNR_PORT", os.getenv("PORT", "8351")))
REQUEST_TIMEOUT = float(os.getenv("TIGER_DNR_REQUEST_TIMEOUT", "900"))
DEFAULT_DEVICE = os.getenv("TIGER_DNR_DEVICE", "cuda")
GATEWAY_CONFIG = load_gateway_config(
    service_provider_env="LLM_LOCAL_TIGER_DNR_PROVIDER",
    default_model="llm-local.tiger-dnr.worker.v1",
    request_timeout=REQUEST_TIMEOUT,
)
LOCALAI_ARTIFACT_DIR = Path(
    expand_path(
        os.getenv(
            "LLM_LOCALAI_ARTIFACT_DIR",
            str(RUNTIME_CACHE_DIR / "localai_artifacts"),
        )
    )
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "hf_modules"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (
    TIGER_DNR_OUTPUT_DIR,
    WORKER_TMP_DIR,
    UPLOAD_TMP_DIR,
    Path(os.environ["HF_MODULES_CACHE"]),
    Path(GPU_LOCK_FILE).parent,
):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Local Backend TIGER-DnR API")


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


@dataclass(frozen=True)
class SeparationResult:
    """已原子提交的一次三 Stem 分离结果。"""

    archive_path: Path
    batch_dir: Path


def module_available(module_name: str) -> bool:
    """只检查模块规格，不实际导入重型依赖。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def tiger_source_is_ready() -> bool:
    """确认上游调用所需的最小源码模块都在本地。"""
    required_paths = (
        TIGER_DNR_SOURCE_DIR / "look2hear" / "models" / "base_model.py",
        TIGER_DNR_SOURCE_DIR / "look2hear" / "models" / "tiger_dnr.py",
        TIGER_DNR_SOURCE_DIR / "look2hear" / "layers" / "activations.py",
        TIGER_DNR_SOURCE_DIR / "look2hear" / "layers" / "normalizations.py",
    )
    return all(path.is_file() for path in required_paths)


def tiger_model_is_ready() -> bool:
    """确认镜像中的 config 和 Safetensors 均存在且非空。"""
    required_paths = (
        TIGER_DNR_MODEL_DIR / "config.json",
        TIGER_DNR_MODEL_DIR / "model.safetensors",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required_paths)


def worker_error_excerpt(output: str) -> str:
    """提取 worker 末尾错误，避免 HTTP 响应包含无界日志。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "TIGER-DnR worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class TigerDnrWorkerManager:
    """组装 TIGER-DnR worker 请求并管理临时 JSON 与三 Stem 文件。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_archive_path: Path | None = None

    def build_worker_payload(self, staged: StagedUpload, device: str) -> dict[str, Any]:
        """完成纯文件预检后构造一次性 worker 的离线调用参数。"""
        if device not in {"cuda", "cpu"}:
            raise ValueError("device 仅支持 cuda 或 cpu。")
        if not tiger_model_is_ready():
            raise FileNotFoundError(f"TIGER-DnR 模型目录不完整：{TIGER_DNR_MODEL_DIR}")
        if not tiger_source_is_ready():
            raise FileNotFoundError(f"TIGER-DnR 上游源码目录不完整：{TIGER_DNR_SOURCE_DIR}")
        if not staged.path.is_file():
            raise FileNotFoundError(f"暂存音频不存在：{staged.path}")
        return {
            "audio_path": str(staged.path),
            "model_dir": str(TIGER_DNR_MODEL_DIR),
            "source_dir": str(TIGER_DNR_SOURCE_DIR),
            "device": device,
            "local_files_only": LOCAL_FILES_ONLY,
        }

    def run_worker(self, payload: dict[str, Any]) -> dict[str, Path]:
        """在 TIGER-DnR 自身 uv 解释器中运行一次隔离分离。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 TIGER-DnR uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"TIGER-DnR worker 脚本不存在：{WORKER_SCRIPT}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="tiger_dnr_req_",
            suffix=".json",
        )
        os.close(request_fd)
        output_dir = Path(tempfile.mkdtemp(dir=WORKER_TMP_DIR, prefix="tiger_dnr_out_"))
        process: subprocess.Popen[str] | None = None
        succeeded = False
        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)

            command = [
                python_executable,
                WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-dir",
                str(output_dir),
            ]
            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[TIGER-DnR] 启动 worker: python={python_executable}")
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "TIGER-DnR")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"TIGER-DnR worker 超时（>{REQUEST_TIMEOUT:.0f}s）") from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[TIGER-DnR] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))

            outputs = {stem: output_dir / f"{stem}.wav" for stem in STEM_NAMES}
            invalid_stems = [
                stem
                for stem, path in outputs.items()
                if not path.is_file() or path.stat().st_size == 0
            ]
            if invalid_stems:
                raise RuntimeError(f"TIGER-DnR worker 未生成有效 Stem：{', '.join(invalid_stems)}")
            succeeded = True
            self.last_error = None
            return outputs
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "TIGER-DnR")
            Path(request_path).unlink(missing_ok=True)
            if not succeeded:
                shutil.rmtree(output_dir, ignore_errors=True)

    def run_localai_worker(self, payload: dict[str, Any]) -> dict[str, Path]:
        """经 file_ref 传输取回三 Stem ZIP，并校验后解包到私有任务目录。"""
        archive, _ = execute_localai_worker_file_reference(
            config=GATEWAY_CONFIG,
            operation="run",
            payload={"worker_input": payload},
            output_dir=WORKER_TMP_DIR,
            artifact_dir=LOCALAI_ARTIFACT_DIR,
            suffix=".zip",
            media_type="application/zip",
        )
        output_dir = Path(tempfile.mkdtemp(dir=WORKER_TMP_DIR, prefix="tiger_dnr_ref_"))
        try:
            with zipfile.ZipFile(archive) as worker_archive:
                expected = {f"{stem}.wav" for stem in STEM_NAMES}
                infos = worker_archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(expected) or set(names) != expected:
                    raise LocalAIRequestError("TIGER-DnR worker ZIP 必须且只能包含三个命名 Stem。")
                if any(info.is_dir() or info.file_size <= 0 for info in infos):
                    raise LocalAIRequestError("TIGER-DnR worker ZIP 含空 Stem 或目录项。")
                if sum(info.file_size for info in infos) > 4 * 1024 * 1024 * 1024:
                    raise LocalAIRequestError("TIGER-DnR worker ZIP 解压后超过 4 GiB。")
                for info in infos:
                    destination = output_dir / info.filename
                    with worker_archive.open(info) as source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    if destination.stat().st_size != info.file_size:
                        raise LocalAIRequestError("TIGER-DnR worker Stem 解压后大小不一致。")
            outputs = {stem: output_dir / f"{stem}.wav" for stem in STEM_NAMES}
            self.last_error = None
            return outputs
        except Exception as exc:
            self.last_error = str(exc)
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        finally:
            archive.unlink(missing_ok=True)

    @staticmethod
    def cleanup_worker_outputs(outputs: dict[str, Path]) -> None:
        """在生成内容完成原子提交后删除一次性 worker 输出目录。"""
        if not outputs:
            return
        output_dir = next(iter(outputs.values())).parent.resolve()
        worker_root = WORKER_TMP_DIR.resolve()
        try:
            output_dir.relative_to(worker_root)
        except ValueError:
            return
        shutil.rmtree(output_dir, ignore_errors=True)


manager = TigerDnrWorkerManager()


def persist_stem_outputs(outputs: dict[str, Path]) -> SeparationResult:
    """原子提交完整的三 Stem 批次，并创建可流式下载的 ZIP。

    每个 Stem 先复制到同一目标文件系统的隐藏批次目录，最后以 ``os.replace``
    公开整个目录，确保客户端和运维脚本不会看到半成品集合。
    """
    batch_name = f"tiger_dnr_{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"
    TIGER_DNR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(dir=TIGER_DNR_OUTPUT_DIR, prefix=f".{batch_name}_"))
    batch_dir = TIGER_DNR_OUTPUT_DIR / batch_name
    try:
        for stem in STEM_NAMES:
            source = outputs.get(stem)
            if source is None or not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"TIGER-DnR 缺少可保存的 {stem}.wav。")
            destination = temporary_dir / f"{stem}.wav"
            with source.open("rb") as input_file, destination.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
                output_file.flush()
                os.fsync(output_file.fileno())

        archive_path = temporary_dir / "tiger_dnr_stems.zip"
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for stem in STEM_NAMES:
                archive.write(temporary_dir / f"{stem}.wav", arcname=f"{stem}.wav")
        os.replace(temporary_dir, batch_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    result = SeparationResult(
        archive_path=batch_dir / "tiger_dnr_stems.zip",
        batch_dir=batch_dir,
    )
    manager.last_archive_path = result.archive_path
    return result


def execute_separation_payload(payload: dict[str, Any]) -> SeparationResult:
    """持有共享 GPU 锁运行 worker 并持久化三 Stem；在线程池中调用。"""
    outputs: dict[str, Path] = {}
    if GATEWAY_CONFIG.provider == "localai":
        try:
            with manager.lock:
                try:
                    outputs = manager.run_localai_worker(payload)
                    return persist_stem_outputs(outputs)
                finally:
                    manager.cleanup_worker_outputs(outputs)
        except Exception as exc:
            manager.last_error = str(exc)
            raise
    try:
        with gpu_runtime_lock(GPU_LOCK_FILE, "tiger-dnr/separate"):
            with manager.lock:
                try:
                    outputs = manager.run_worker(payload)
                    return persist_stem_outputs(outputs)
                finally:
                    manager.cleanup_worker_outputs(outputs)
                    wait_after_cuda_release(CUDA_RELEASE_DELAY, "after TIGER-DnR worker")
    except Exception as exc:
        manager.last_error = str(exc)
        raise


@app.get("/v1/health")
def health():
    """返回模型、上游源码、worker 与 GPU 的就绪状态，不加载模型。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "tiger_dnr_model_dir": str(TIGER_DNR_MODEL_DIR),
            "tiger_dnr_source_dir": str(TIGER_DNR_SOURCE_DIR),
            "tiger_dnr_output_dir": str(TIGER_DNR_OUTPUT_DIR),
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "upload_tmp_dir": str(UPLOAD_TMP_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "tiger_dnr_model_dir": tiger_model_is_ready(),
            "tiger_dnr_source_dir": tiger_source_is_ready(),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "huggingface_hub": module_available("huggingface_hub"),
            "safetensors": module_available("safetensors"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "default_device": DEFAULT_DEVICE,
            "model_sample_rate": MODEL_SAMPLE_RATE,
            "stems": list(STEM_NAMES),
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"tiger_dnr": manager.last_error},
    }


@app.post("/v1/tigerDnr/separate")
async def separate(
    audio: UploadFile = File(...),
    device: Literal["cuda", "cpu"] = Form(DEFAULT_DEVICE),
):
    """上传混音音频，返回 dialog、effects、music 三 Stem 的 ZIP 成品。"""
    staged: StagedUpload | None = None
    try:
        staged = await stage_audio_upload(audio, UPLOAD_TMP_DIR)
        payload = manager.build_worker_payload(staged, device)
        result = await run_in_threadpool(execute_separation_payload, payload)
        return FileResponse(
            result.archive_path,
            media_type="application/zip",
            filename="tiger_dnr_stems.zip",
        )
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (BackendQuarantinedError, LocalAIUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LocalAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("TIGER-DnR separation request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if staged is not None:
            staged.path.unlink(missing_ok=True)
        await close_upload_safely(audio)


if __name__ == "__main__":
    print("==================================================")
    print("   LLM 本地后端 TIGER-DnR")
    print("==================================================")
    print(f"[配置] 模型目录: {TIGER_DNR_MODEL_DIR}")
    print(f"[配置] 上游源码: {TIGER_DNR_SOURCE_DIR}")
    print(f"[配置] 输出目录: {TIGER_DNR_OUTPUT_DIR}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] host={API_HOST}, port={API_PORT}, device={DEFAULT_DEVICE}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
