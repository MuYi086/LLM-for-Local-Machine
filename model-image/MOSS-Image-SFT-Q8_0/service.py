"""提供共享 GPU 锁保护的 MOSS-Image-SFT 图片生成 API。"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from inference_gateway.errors import BackendQuarantinedError
from inference_gateway.sdcpp import GenerationRequest, ModelSpec, SdcppConfig, load_models
from inference_gateway.sdcpp.config import local_path
from inference_gateway.sdcpp.execution import GenerationResult, generate
from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_quarantine_path

MODEL_ID = "MOSS-Image-SFT-Q8_0"
IMAGE_ROUTE = "/v1/openmoss/MOSS-Image-SFT-Q8_0"
MODEL_MANIFEST = Path(__file__).resolve().parents[2] / "sdcpp/manifests/models.json"
LOGGER = logging.getLogger(__name__)
ImageGenerator = Callable[[SdcppConfig, ModelSpec, GenerationRequest], GenerationResult]


class ImageGenerationRequest(BaseModel):
    """OpenAI Images 风格的单张 MOSS 图片生成请求。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=65_536)
    model: str | None = Field(default=None, max_length=128)
    negative_prompt: str = Field(default="", max_length=65_536)
    size: str = Field(default="1024x1024", pattern=r"^[0-9]{2,4}x[0-9]{2,4}$")
    n: Literal[1] = 1
    steps: int = Field(default=20, ge=1, le=50)
    cfg_scale: float = Field(default=3.0, ge=0, le=30)
    seed: int = Field(default=-1, ge=-1, le=2_147_483_647)
    response_format: Literal["b64_json"] = "b64_json"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """拒绝空白提示和 CLI 无法安全处理的空字符。"""
        if not value.strip() or "\0" in value:
            raise ValueError("prompt 必须包含可见文本且不能包含空字符。")
        return value

    @field_validator("negative_prompt")
    @classmethod
    def validate_negative_prompt(cls, value: str) -> str:
        """拒绝 CLI 无法安全处理的空字符。"""
        if "\0" in value:
            raise ValueError("negative_prompt 不能包含空字符。")
        return value


class ImageData(BaseModel):
    """OpenAI Images 风格的 Base64 图片数据。"""

    b64_json: str


class ImageGenerationResponse(BaseModel):
    """单张图片生成响应。"""

    created: int
    data: list[ImageData]


def _load_runtime() -> tuple[SdcppConfig, ModelSpec]:
    """读取固定模型清单与 sd.cpp 配置，不加载模型权重。"""
    config = SdcppConfig.from_env()
    manifest = local_path(os.getenv("SDCPP_MODEL_MANIFEST", str(MODEL_MANIFEST)))
    models = load_models(manifest)
    try:
        model = models[MODEL_ID]
    except KeyError as exc:
        raise ValueError(f"sd.cpp 清单未配置模型：{MODEL_ID}") from exc
    if "text-to-image" not in model.tasks:
        raise ValueError(f"sd.cpp 清单未声明图片生成任务：{MODEL_ID}")
    return config, model


def _dimensions(size: str) -> tuple[int, int]:
    """把 OpenAI Images 尺寸字符串解析为宽高。"""
    match = re.fullmatch(r"([0-9]{2,4})x([0-9]{2,4})", size)
    if match is None:
        raise HTTPException(status_code=422, detail="size 必须使用 WIDTHxHEIGHT 格式。")
    return int(match.group(1)), int(match.group(2))


def create_app(
    config: SdcppConfig | None = None,
    model: ModelSpec | None = None,
    *,
    generator: ImageGenerator = generate,
) -> FastAPI:
    """创建 MOSS-Image-SFT 单模型图片 API。

    Args:
        config: sd.cpp 二进制、GPU 锁和输出目录配置。
        model: 已解析的 MOSS-Image-SFT 模型清单项。
        generator: 负责校验并执行模型请求的共享 sd.cpp provider。

    Returns:
        注册了健康检查和图片生成路由的 FastAPI 应用。
    """
    if (config is None) != (model is None):
        raise ValueError("config 和 model 必须同时提供或同时省略。")
    if config is None or model is None:
        config, model = _load_runtime()
    if model.id != MODEL_ID or "text-to-image" not in model.tasks:
        raise ValueError(f"API 仅支持清单中声明 text-to-image 的 {MODEL_ID}。")

    app = FastAPI(title=f"{MODEL_ID} Image API")
    app.state.sdcpp_config = config
    app.state.model_spec = model

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        """检查二进制、模型组件和隔离状态，不计算摘要或加载模型。"""
        if not config.binary.is_file() or not os.access(config.binary, os.X_OK):
            raise HTTPException(status_code=503, detail="sd.cpp 二进制未就绪。")
        if any(
            not path.is_file() or path.stat().st_size == 0 for path in model.components.values()
        ):
            raise HTTPException(status_code=503, detail="MOSS 模型组件未就绪。")
        if gpu_quarantine_path(config.lock_path).exists():
            raise HTTPException(status_code=503, detail="GPU 处于隔离状态，请运行恢复流程。")
        return {"status": "ok", "backend": "stable-diffusion.cpp", "model": MODEL_ID}

    @app.post(IMAGE_ROUTE, response_model=ImageGenerationResponse)
    def generate_image(payload: ImageGenerationRequest) -> ImageGenerationResponse:
        """通过固定 sd.cpp provider 生成 PNG，并以 Base64 返回。

        Args:
            payload: 图片提示、尺寸和有界采样参数。

        Returns:
            OpenAI Images 风格的单张 Base64 PNG 响应。
        """
        if payload.model is not None and payload.model != MODEL_ID:
            raise HTTPException(status_code=422, detail=f"model 必须是 {MODEL_ID}。")
        width, height = _dimensions(payload.size)
        request = GenerationRequest(
            task="text-to-image",
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=width,
            height=height,
            steps=payload.steps,
            cfg_scale=payload.cfg_scale,
            seed=payload.seed,
        )
        try:
            model.arguments(request, Path("result.png"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
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
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("sd.cpp MOSS 图片生成失败。")
            raise HTTPException(
                status_code=503, detail="sd.cpp 图片生成失败，请检查服务日志。"
            ) from exc

        LOGGER.info(
            "sd.cpp MOSS 图片生成完成 request_id=%s metrics=%s",
            result.request_id,
            result.metrics,
        )
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=base64.b64encode(image_bytes).decode("ascii"))],
        )

    return app


def run_service() -> None:
    """启动图片 API，阻塞推理由 FastAPI 在线程池执行。"""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8501")))
