#!/usr/bin/env python3
"""YuE2-3B 本地歌曲生成 HTTP 服务。"""

from __future__ import annotations

# API 进程只负责请求校验、GPU 队列和 worker 管理，不导入模型或 Torch。
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import GpuLockTimeoutError

from runtime import (
    cuda_status,
    gpu_runtime_lock,
    module_available,
    persist_audio_bytes,
    terminate_process_group,
    wait_after_cuda_release,
)

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
BGM_STORAGE_DIR = expand_path(os.getenv("BGM_STORAGE_DIR", str(STORAGE_DIR / "bgm")))
OUTPUT_DIR = expand_path(os.getenv("YUE2_3B_OUTPUT_DIR", str(BGM_STORAGE_DIR)))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = str(
    expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
)
MODEL_DIR = expand_path(os.getenv("YUE2_3B_MODEL_DIR", str(HF_MIRROR_DIR / "m-a-p/YuE2-3B")))
INFERENCE_WHEEL = MODEL_DIR / "yue2_infer-0.1.5-py3-none-any.whl"
VAE_MODEL_DIR = expand_path(os.getenv("YUE2_VAE_MODEL_DIR", str(HF_MIRROR_DIR / "m-a-p/YuE2-Vae")))
WORKER_TMP_DIR = expand_path(
    os.getenv("YUE2_3B_WORKER_TMP_DIR", str(RUNTIME_CACHE_DIR / "yue2_3b_worker"))
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
REQUEST_TIMEOUT = float(os.getenv("YUE2_3B_REQUEST_TIMEOUT", "1800"))
GPU_LOCK_WAIT_TIMEOUT = float(os.getenv("GPU_LOCK_WAIT_TIMEOUT", "900"))
DEVICE = os.getenv("YUE2_3B_DEVICE", "cuda")
MEMORY_BUDGET_GIB = float(os.getenv("YUE2_3B_MEMORY_BUDGET_GIB", "24"))
BACKEND = os.getenv("YUE2_3B_BACKEND", "torch")
QUANTIZATION = os.getenv("YUE2_3B_QUANTIZATION", "none")
OFFLOAD_AR = env_bool("YUE2_3B_OFFLOAD_AR", False)
API_HOST = os.getenv("YUE2_3B_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("YUE2_3B_PORT", os.getenv("PORT", "8314")))

if not math.isfinite(REQUEST_TIMEOUT) or REQUEST_TIMEOUT <= 0:
    raise ValueError("YUE2_3B_REQUEST_TIMEOUT 必须大于 0。")
if not math.isfinite(CUDA_RELEASE_DELAY) or CUDA_RELEASE_DELAY < 0:
    raise ValueError("CUDA_RELEASE_DELAY 必须是非负有限数。")
if not math.isfinite(GPU_LOCK_WAIT_TIMEOUT) or GPU_LOCK_WAIT_TIMEOUT <= 0:
    raise ValueError("GPU_LOCK_WAIT_TIMEOUT 必须是正数，避免 GPU 队列无限等待。")
if not math.isfinite(MEMORY_BUDGET_GIB) or MEMORY_BUDGET_GIB <= 2:
    raise ValueError("YUE2_3B_MEMORY_BUDGET_GIB 必须大于 2 GiB。")

os.environ.setdefault("HF_HOME", str(HF_MIRROR_DIR))
os.environ.setdefault("HF_MODULES_CACHE", str(RUNTIME_CACHE_DIR / "hf_modules"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(RUNTIME_CACHE_DIR / "torchinductor"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (OUTPUT_DIR, RUNTIME_CACHE_DIR, WORKER_TMP_DIR, Path(GPU_LOCK_FILE).parent):
    directory.mkdir(parents=True, exist_ok=True)

if INFERENCE_WHEEL.is_file():
    sys.path.insert(0, str(INFERENCE_WHEEL))


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


class SamplingRequest(BaseModel):
    """YuE2 原生 ABC 或 semantic token 采样配置。"""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    penalty_window: int | None = Field(default=None, ge=1, le=100)
    min_tokens: int | None = Field(default=None, ge=0, le=24_576)
    max_tokens: int | None = Field(default=None, ge=1, le=24_576)

    @model_validator(mode="after")
    def validate_token_range(self) -> SamplingRequest:
        """确保最小生成 token 数不大于最大生成 token 数。"""
        if (
            self.min_tokens is not None
            and self.max_tokens is not None
            and self.min_tokens > self.max_tokens
        ):
            raise ValueError("min_tokens 不能大于 max_tokens。")
        return self


class YuE2MusicRequest(BaseModel):
    """YuE2 原生文本到音乐请求。"""

    model_config = ConfigDict(extra="forbid")

    style: str | None = Field(default=None, min_length=1, max_length=2_000)
    tags: str | None = Field(default=None, min_length=1, max_length=2_000)
    lyrics: str = Field(min_length=1, max_length=100_000)
    cot: Literal["off", "melody", "full"] = "full"
    seed: int = Field(default=831_001, ge=0, lt=2**63)
    abc: str | None = Field(default=None, min_length=1, max_length=100_000)
    cfg_scale: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False)
    id: str = Field(default="song", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")
    abc_sampling: SamplingRequest | None = None
    semantic_sampling: SamplingRequest | None = None

    @model_validator(mode="after")
    def validate_native_request(self) -> YuE2MusicRequest:
        """检查风格别名、文本空白和外部乐谱的原生组合约束。"""
        if self.style is None and self.tags is None:
            raise ValueError("style 或 tags 至少提供一项。")
        if (
            self.style is not None
            and self.tags is not None
            and self.style.strip() != self.tags.strip()
        ):
            raise ValueError("style 与 tags 同时提供时必须内容一致。")
        if self.style is not None and not self.style.strip():
            raise ValueError("style 不能为空。")
        if self.tags is not None and not self.tags.strip():
            raise ValueError("tags 不能为空。")
        if not self.lyrics.strip():
            raise ValueError("lyrics 不能为空。")
        if self.abc is not None and (not self.abc.strip() or self.cot == "off"):
            raise ValueError("abc 必须是非空文本，且 cot 不能为 off。")
        return self


def model_directory_status(model_dir: Path) -> dict[str, Any]:
    """检查 YuE2 MOT 目录及 manifest 声明的文件，不读取大权重内容。"""
    required = ("config.json", "modeling_yue2.py", "qwen.tiktoken", "weights_manifest.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    weight_files: list[str] = []
    manifest_error: str | None = None
    manifest_path = model_dir / "weights_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                manifest_error = "weights_manifest.json 缺少 files 映射"
            else:
                for name, metadata in files.items():
                    path = model_dir / name
                    weight_files.append(name)
                    if not path.is_file():
                        missing.append(name)
                    elif isinstance(metadata, dict) and path.stat().st_size != metadata.get(
                        "bytes"
                    ):
                        manifest_error = f"权重文件大小与 manifest 不匹配: {name}"
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            manifest_error = f"无法读取 weights_manifest.json: {exc}"

    return {
        "path": str(model_dir),
        "complete": not missing and manifest_error is None,
        "missing": sorted(set(missing)),
        "weight_files": weight_files,
        "error": manifest_error,
    }


def vae_directory_status(model_dir: Path) -> dict[str, Any]:
    """检查 YuE2 VAE 配置和完整 safetensors 权重集合。"""
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")

    weight_index_path = model_dir / "model.safetensors.index.json"
    if weight_index_path.is_file():
        try:
            index = json.loads(weight_index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                missing.append("model.safetensors.index.json:weight_map")
            else:
                missing.extend(
                    name
                    for name in sorted(set(weight_map.values()))
                    if not (model_dir / name).is_file()
                )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            missing.append(f"model.safetensors.index.json: {exc}")
    elif not (model_dir / "model.safetensors").is_file():
        missing.append("model.safetensors 或 model.safetensors.index.json")

    return {"path": str(model_dir), "complete": not missing, "missing": missing}


def model_status() -> dict[str, Any]:
    """返回权重和推理包状态，不触发模型加载。"""
    return {
        "inference_package": {
            "path": str(INFERENCE_WHEEL),
            "complete": INFERENCE_WHEEL.is_file(),
            "missing": [] if INFERENCE_WHEEL.is_file() else [INFERENCE_WHEEL.name],
        },
        "mot": model_directory_status(MODEL_DIR),
        "vae": vae_directory_status(VAE_MODEL_DIR),
    }


def worker_error_excerpt(output: str) -> str:
    """提取 worker 最后的可读错误行，避免回传完整 traceback。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "YuE2-3B worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class YuE2WorkerManager:
    """把每次生成放进独立进程，并保存最后一次错误供健康检查查看。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: YuE2MusicRequest) -> dict[str, Any]:
        """附加本机模型、运行时和离线配置。"""
        payload = request.model_dump(mode="json", exclude_none=True)
        payload.update(
            {
                "model_dir": str(MODEL_DIR),
                "vae_model_dir": str(VAE_MODEL_DIR),
                "device": DEVICE,
                "memory_budget_gib": MEMORY_BUDGET_GIB,
                "backend": BACKEND,
                "quantization": QUANTIZATION,
                "offload_ar": OFFLOAD_AR,
                "local_files_only": LOCAL_FILES_ONLY,
                "hf_mirror_dir": str(HF_MIRROR_DIR),
                "inference_wheel": str(INFERENCE_WHEEL),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """使用本服务 uv 解释器启动一次 YuE2 推理并读取 WAV。"""
        if not sys.executable or not Path(sys.executable).is_file():
            raise RuntimeError("未找到 YuE2-3B uv 环境的 Python 解释器。")
        if not WORKER_SCRIPT.is_file():
            raise RuntimeError(f"YuE2-3B worker 不存在: {WORKER_SCRIPT}")

        request_fd, request_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="yue2_3b_request_",
            suffix=".json",
        )
        output_fd, output_name = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="yue2_3b_output_",
            suffix=".wav",
        )
        os.close(output_fd)
        request_path = Path(request_name)
        output_path = Path(output_name)
        process: subprocess.Popen[str] | None = None
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as request_file:
                json.dump(payload, request_file, ensure_ascii=False)
                request_file.flush()
                os.fsync(request_file.fileno())

            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            command = [
                sys.executable,
                str(WORKER_SCRIPT),
                "--input-json",
                str(request_path),
                "--output-wav",
                str(output_path),
            ]
            process = subprocess.Popen(
                command,
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
                raise TimeoutError(f"YuE2-3B 推理超过 {REQUEST_TIMEOUT:.0f} 秒时限。") from exc

            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(f"{stdout}\n{stderr}"))

            audio_bytes = output_path.read_bytes()
            if not audio_bytes:
                raise RuntimeError("YuE2-3B worker 成功退出，但没有生成 WAV 文件。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "YuE2-3B")
            for path in (request_path, output_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    LOGGER.warning("清理 YuE2-3B worker 文件失败 %s: %s", path, exc)


app = FastAPI(title="Unitale YuE2-3B Music API")
app.add_middleware(ForceCORS)
manager = YuE2WorkerManager()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    """返回服务依赖、模型文件和 GPU 状态，不加载模型权重。"""
    gpu = cuda_status()
    models = model_status()
    dependencies = {
        name: module_available(name)
        for name in ("yue2", "torch", "transformers", "soundfile", "safetensors")
    }
    ready = (
        all(dependencies.values())
        and gpu["available"]
        and all(item["complete"] for item in models.values())
    )
    return {
        "code": 200,
        "ready": ready,
        "paths": {
            "project_dir": str(PROJECT_DIR),
            "model_dir": str(MODEL_DIR),
            "inference_wheel": str(INFERENCE_WHEEL),
            "vae_model_dir": str(VAE_MODEL_DIR),
            "output_dir": str(OUTPUT_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_python": sys.executable,
        },
        "available": {
            "dependencies": dependencies,
            "cuda": gpu["available"],
            "model_complete": models["mot"]["complete"],
            "vae_complete": models["vae"]["complete"],
        },
        "models": models,
        "gpu": gpu,
        "recommended_gpu_memory_gib": 24,
        "last_errors": {"yue2_3b": manager.last_error},
    }


@app.post("/v1/m-a-p/YuE2")
def generate_music(request: YuE2MusicRequest) -> Response:
    """串行生成一首歌曲，返回 48 kHz WAV 并原子保存到 BGM 目录。"""
    dependencies = {
        name: module_available(name)
        for name in ("yue2", "torch", "transformers", "soundfile", "safetensors")
    }
    missing_dependencies = [name for name, available in dependencies.items() if not available]
    if missing_dependencies:
        raise HTTPException(
            status_code=503,
            detail=f"YuE2-3B 运行依赖未安装: {', '.join(missing_dependencies)}。请先 uv sync。",
        )

    models = model_status()
    incomplete_models = [name for name, status in models.items() if not status["complete"]]
    if incomplete_models:
        details = {
            name: models[name]["missing"] or models[name].get("error") for name in incomplete_models
        }
        raise HTTPException(
            status_code=503,
            detail={"message": "YuE2-3B 本地权重未准备完整。", "models": details},
        )

    if not cuda_status()["available"]:
        raise HTTPException(status_code=503, detail="YuE2-3B 需要可用的 NVIDIA CUDA GPU。")

    payload = manager.build_worker_payload(request)
    try:
        with gpu_runtime_lock(GPU_LOCK_FILE, "yue2_3b/music"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_path = persist_audio_bytes(audio_bytes, OUTPUT_DIR)
                    LOGGER.info("YuE2-3B 生成 WAV 已保存: %s", saved_path)
                    return Response(
                        content=audio_bytes,
                        media_type="audio/wav",
                        headers={
                            "X-YuE2-3B-Seed": str(request.seed),
                            "X-YuE2-3B-Sample-Rate": "48000",
                            "X-YuE2-3B-Model": "m-a-p/YuE2-3B",
                        },
                    )
                except TimeoutError as exc:
                    LOGGER.exception("YuE2-3B request timed out")
                    raise HTTPException(status_code=504, detail=str(exc)) from exc
                except Exception as exc:
                    LOGGER.exception("YuE2-3B request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release(CUDA_RELEASE_DELAY, "after YuE2-3B worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
