"""提供共享 GPU 锁保护的 FLUX.3 Klein 图片生成和编辑 API。"""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from inference_gateway.errors import BackendQuarantinedError
from inference_gateway.sdcpp import GenerationRequest, ModelSpec, SdcppConfig, load_models
from inference_gateway.sdcpp.config import local_path
from inference_gateway.sdcpp.execution import GenerationResult, generate
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_quarantine_path

MODEL_ID = "FLUX-3-Image-Klein-GGUF-Q8"
IMAGE_ROUTE = "/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8"
MODEL_MANIFEST = Path(__file__).resolve().parents[2] / "sdcpp/manifests/models.json"
DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
LOGGER = logging.getLogger(__name__)
ImageGenerator = Callable[[SdcppConfig, ModelSpec, GenerationRequest], GenerationResult]


class ImageGenerationRequest(BaseModel):
    """OpenAI Images 风格的 FLUX 单张图片生成请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=65_536)
    model: str | None = Field(default=None, max_length=128)
    size: str = Field(default="1024x1024", pattern=r"^[0-9]{2,4}x[0-9]{2,4}$")
    n: Literal[1] = 1
    steps: int = Field(default=16, ge=1, le=50)
    cfg_scale: float = Field(default=1.0, ge=0, le=30)
    seed: int = Field(default=-1, ge=-1, le=2_147_483_647)
    response_format: Literal["b64_json"] = "b64_json"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """拒绝空白提示和 CLI 无法安全处理的空字符。"""
        if not value.strip() or "\0" in value:
            raise ValueError("prompt 必须包含可见文本且不能包含空字符。")
        return value


class ImageData(BaseModel):
    """OpenAI Images 风格的 Base64 图片数据。"""

    b64_json: str


class ImageGenerationResponse(BaseModel):
    """单张图片生成响应。"""

    created: int
    data: list[ImageData]


def _load_runtime() -> tuple[SdcppConfig, ModelSpec]:
    """读取固定 FLUX 清单项与 sd.cpp 配置，不加载模型权重。"""
    config = SdcppConfig.from_env()
    manifest = local_path(os.getenv("SDCPP_MODEL_MANIFEST", str(MODEL_MANIFEST)))
    models = load_models(manifest)
    try:
        model = models[MODEL_ID]
    except KeyError as exc:
        raise ValueError(f"sd.cpp 清单未配置模型：{MODEL_ID}") from exc
    if not {"text-to-image", "image-to-image"}.issubset(model.tasks):
        raise ValueError(f"sd.cpp 清单必须声明文生图和图像编辑任务：{MODEL_ID}")
    return config, model


def _dimensions(size: str) -> tuple[int, int]:
    """把 OpenAI Images 尺寸字符串解析为宽高。"""
    match = re.fullmatch(r"([0-9]{2,4})x([0-9]{2,4})", size)
    if match is None:
        raise HTTPException(status_code=422, detail="size 必须使用 WIDTHxHEIGHT 格式。")
    return int(match.group(1)), int(match.group(2))


def _stage_image(
    upload: UploadFile,
    upload_dir: Path,
    max_bytes: int,
) -> Path:
    """以有界分块方式暂存 PNG/JPEG 上传，并校验实际文件签名。"""
    content_type = upload.content_type
    if content_type is not None and content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="参考图只接受 PNG 或 JPEG。")

    upload_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="flux3-edit-", dir=upload_dir)
    staged_path = Path(name)
    total = 0
    header = bytearray()
    try:
        with os.fdopen(descriptor, "wb") as staged_file:
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="参考图超过上传大小限制。")
                if len(header) < len(PNG_SIGNATURE):
                    header.extend(chunk[: len(PNG_SIGNATURE) - len(header)])
                staged_file.write(chunk)
            staged_file.flush()
            os.fsync(staged_file.fileno())

        if bytes(header).startswith(PNG_SIGNATURE):
            extension, expected_type = ".png", "image/png"
        elif bytes(header).startswith(JPEG_SIGNATURE):
            extension, expected_type = ".jpg", {"image/jpeg", "image/jpg"}
        else:
            raise HTTPException(status_code=415, detail="参考图内容不是有效的 PNG 或 JPEG 文件。")
        if content_type is not None and content_type not in (
            {expected_type} if isinstance(expected_type, str) else expected_type
        ):
            raise HTTPException(status_code=415, detail="参考图 MIME 类型与文件内容不匹配。")
        published_path = staged_path.with_suffix(extension)
        staged_path.replace(published_path)
        return published_path
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise


def create_app(
    config: SdcppConfig | None = None,
    model: ModelSpec | None = None,
    *,
    generator: ImageGenerator = generate,
    upload_dir: Path | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    """创建 FLUX.3 Klein 图片生成和编辑 API。

    Args:
        config: sd.cpp 二进制、GPU 锁和输出目录配置。
        model: 已解析的 FLUX.3 Klein 模型清单项。
        generator: 负责校验并执行模型请求的共享 sd.cpp provider。
        upload_dir: 编辑参考图的临时暂存目录。
        max_upload_bytes: 单个参考图的最大字节数。

    Returns:
        注册健康检查、图片生成和图片编辑路由的 FastAPI 应用。
    """
    if (config is None) != (model is None):
        raise ValueError("config 和 model 必须同时提供或同时省略。")
    if config is None or model is None:
        config, model = _load_runtime()
    if model.id != MODEL_ID or not {"text-to-image", "image-to-image"}.issubset(model.tasks):
        raise ValueError(f"API 仅支持清单中声明文生图和图像编辑的 {MODEL_ID}。")

    if upload_dir is None:
        runtime_cache = os.getenv("RUNTIME_CACHE_DIR", "storage/.cache/runtime")
        upload_dir = local_path(
            os.getenv("FLUX3_IMAGE_UPLOAD_DIR", str(Path(runtime_cache) / "sdcpp_uploads"))
        )
    if max_upload_bytes is None:
        try:
            max_upload_bytes = int(
                os.getenv("FLUX3_IMAGE_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
            )
        except ValueError as exc:
            raise ValueError("FLUX3_IMAGE_MAX_UPLOAD_BYTES 必须是正整数。") from exc
    if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
        raise ValueError("FLUX3_IMAGE_MAX_UPLOAD_BYTES 必须是正整数。")

    app = FastAPI(title=f"{MODEL_ID} Image API")
    app.state.sdcpp_config = config
    app.state.model_spec = model
    app.state.upload_dir = upload_dir
    app.state.max_upload_bytes = max_upload_bytes

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        """检查二进制、模型组件和隔离状态，不计算摘要或加载模型。"""
        if not config.binary.is_file() or not os.access(config.binary, os.X_OK):
            raise HTTPException(status_code=503, detail="sd.cpp 二进制未就绪。")
        if any(
            not path.is_file() or path.stat().st_size == 0 for path in model.components.values()
        ):
            raise HTTPException(status_code=503, detail="FLUX.3 Klein 模型组件未就绪。")
        if gpu_quarantine_path(config.lock_path).exists():
            raise HTTPException(status_code=503, detail="GPU 处于隔离状态，请运行恢复流程。")
        return {"status": "ok", "backend": "stable-diffusion.cpp", "model": MODEL_ID}

    def execute(
        payload: ImageGenerationRequest,
        task: str,
        upload: UploadFile | None = None,
    ) -> ImageGenerationResponse:
        """验证参数、暂存可选参考图并执行一次共享 sd.cpp 请求。"""
        if payload.model is not None and payload.model != MODEL_ID:
            raise HTTPException(status_code=422, detail=f"model 必须是 {MODEL_ID}。")
        width, height = _dimensions(payload.size)
        staged_path: Path | None = None
        try:
            if upload is not None:
                staged_path = _stage_image(upload, upload_dir, max_upload_bytes)
            request = GenerationRequest(
                task=task,
                prompt=payload.prompt,
                width=width,
                height=height,
                steps=payload.steps,
                cfg_scale=payload.cfg_scale,
                seed=payload.seed,
                init_image=staged_path,
            )
            try:
                model.arguments(request, Path("result.png"))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            result = generator(config, model, request)
            image_bytes = result.output.read_bytes()
        except GpuQuarantinedError as exc:
            raise HTTPException(
                status_code=503,
                detail="GPU 处于隔离状态，请确认推理进程退出并运行恢复流程。",
            ) from exc
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except BackendQuarantinedError as exc:
            LOGGER.exception("sd.cpp 推理后未确认 GPU 释放。")
            raise HTTPException(
                status_code=503,
                detail="GPU 处于隔离状态，请确认 sd.cpp 进程退出并运行恢复流程。",
            ) from exc
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("sd.cpp FLUX.3 Klein 图片任务失败。")
            raise HTTPException(
                status_code=503, detail="sd.cpp 图片任务失败，请检查服务日志。"
            ) from exc
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

        LOGGER.info(
            "sd.cpp FLUX.3 Klein 图片任务完成 task=%s request_id=%s metrics=%s",
            task,
            result.request_id,
            result.metrics,
        )
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=base64.b64encode(image_bytes).decode("ascii"))],
        )

    @app.post(IMAGE_ROUTE, response_model=ImageGenerationResponse)
    def generate_image(payload: ImageGenerationRequest) -> ImageGenerationResponse:
        """生成单张 PNG 图片并以 Base64 返回。"""
        return execute(payload, "text-to-image")

    @app.post(f"{IMAGE_ROUTE}/edit", response_model=ImageGenerationResponse)
    def edit_image(
        prompt: str = Form(..., min_length=1, max_length=65_536),
        image: UploadFile = File(...),
        size: str = Form("1024x1024", pattern=r"^[0-9]{2,4}x[0-9]{2,4}$"),
        steps: int = Form(16, ge=1, le=50),
        cfg_scale: float = Form(1.0, ge=0, le=30),
        seed: int = Form(-1, ge=-1, le=2_147_483_647),
    ) -> ImageGenerationResponse:
        """上传参考图并根据提示词执行图像编辑。"""
        try:
            payload = ImageGenerationRequest(
                prompt=prompt,
                size=size,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return execute(payload, "image-to-image", image)

    return app


def run_service() -> None:
    """启动图片 API；阻塞推理由 FastAPI 在线程池执行。"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8502")))
