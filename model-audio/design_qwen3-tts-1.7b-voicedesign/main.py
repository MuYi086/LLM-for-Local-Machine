#!/usr/bin/env python3
"""通过固定 LocalAI/audio.cpp 后端提供 Qwen3-TTS VoiceDesign API。"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import uuid
import wave
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from inference_gateway import (
    BackendQuarantinedError,
    GatewayConfig,
    GatewayConfigurationError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    load_gateway_config,
)
from inference_gateway.execution import execute_localai_operation
from llm_local_runtime import GpuLockTimeoutError
from pydantic import BaseModel, ConfigDict, Field, field_validator

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_MODEL = "llm-local.qwen3-tts.1p7b-voicedesign"
SERVICE_PROVIDER_ENV = "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER"
MAX_SEGMENTS = 32


class VoiceDesignRequest(BaseModel):
    """Qwen3-TTS 音色设计请求。"""

    model_config = ConfigDict(extra="forbid")

    voice_description: str = Field(min_length=1, max_length=2000)
    text: str = Field(default="这是生成的参考音频预览。", min_length=1, max_length=12000)
    language: str | None = Field(default="Chinese", max_length=32)
    max_chars_per_chunk: int = Field(default=0, ge=0, le=2000)
    pause_ms: int = Field(default=250, ge=0, le=10000)
    max_new_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float | None = Field(default=None, gt=0, le=1)
    temperature: float | None = Field(default=None, gt=0, le=5)

    @field_validator("voice_description")
    @classmethod
    def validate_voice_description(cls, value: str) -> str:
        """拒绝空白音色指令，并去除首尾空格。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("voice_description 不能只包含空白字符。")
        return normalized

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """拒绝空白朗读文本，并去除首尾空格。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("text 不能只包含空白字符。")
        return normalized


class InvalidAudioError(ValueError):
    """后端返回的音频不是有效 PCM WAV。"""


def _load_gateway_config() -> GatewayConfig:
    """读取服务专属 LocalAI 配置和共享 GPU 生命周期参数。"""
    request_timeout = float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900"))
    config = load_gateway_config(
        service_provider_env=SERVICE_PROVIDER_ENV,
        default_model=DEFAULT_MODEL,
        request_timeout=request_timeout,
    )
    if config.provider != "localai":
        raise GatewayConfigurationError(
            f"{SERVICE_PROVIDER_ENV} 必须为 localai；VoiceDesign GGUF 仅通过 audio.cpp 运行。"
        )
    if config.model != DEFAULT_MODEL:
        raise GatewayConfigurationError(f"VoiceDesign GGUF 仅支持固定模型 ID {DEFAULT_MODEL}。")
    return config


def _split_text(text: str, max_chars_per_chunk: int) -> list[str]:
    """按 Unicode 字符数切分文本，并限制单次请求的后端调用数。"""
    if max_chars_per_chunk == 0:
        return [text]
    chunks = [
        text[index : index + max_chars_per_chunk]
        for index in range(0, len(text), max_chars_per_chunk)
    ]
    if len(chunks) > MAX_SEGMENTS:
        raise HTTPException(status_code=422, detail=f"文本最多可拆分为 {MAX_SEGMENTS} 段。")
    return chunks


def _decode_wav(audio_data: bytes) -> tuple[wave._wave_params, bytes]:
    """验证后端 WAV 为带有效采样帧的未压缩 PCM。"""
    try:
        with wave.open(io.BytesIO(audio_data), "rb") as audio_file:
            params = audio_file.getparams()
            frames = audio_file.readframes(params.nframes)
    except (EOFError, OSError, wave.Error) as exc:
        raise InvalidAudioError("LocalAI 未返回有效 PCM WAV。") from exc
    if (
        params.comptype != "NONE"
        or params.nchannels not in {1, 2}
        or params.sampwidth not in {1, 2, 3, 4}
        or params.framerate <= 0
        or params.nframes <= 0
        or len(frames) != params.nframes * params.nchannels * params.sampwidth
    ):
        raise InvalidAudioError("LocalAI WAV 编码或采样帧无效。")
    return params, frames


def _combine_wav_segments(segments: list[bytes], pause_ms: int) -> tuple[bytes, int]:
    """校验多个 PCM WAV 的参数一致后，在段间插入静音。"""
    decoded = [_decode_wav(segment) for segment in segments]
    reference_params = decoded[0][0]
    if any(params != reference_params for params, _ in decoded[1:]):
        raise InvalidAudioError("LocalAI 分段 WAV 的采样参数不一致。")
    pause_frames = reference_params.framerate * pause_ms // 1000
    silence_value = b"\x80" if reference_params.sampwidth == 1 else b"\x00"
    silence = silence_value * pause_frames * reference_params.nchannels * reference_params.sampwidth
    output = io.BytesIO()
    with wave.open(output, "wb") as audio_file:
        audio_file.setparams(reference_params)
        for index, (_, frames) in enumerate(decoded):
            audio_file.writeframes(frames)
            if index < len(decoded) - 1 and silence:
                audio_file.writeframes(silence)
    return output.getvalue(), reference_params.framerate


def _persist_timbre(audio_data: bytes, timbre_dir: Path) -> Path:
    """原子保存 VoiceDesign WAV，失败时清理半成品。"""
    _decode_wav(audio_data)
    timbre_dir.mkdir(parents=True, exist_ok=True)
    output_path = timbre_dir / f"qwen3-tts-voicedesign-{uuid.uuid4().hex}.wav"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".qwen3-tts-voicedesign-", suffix=".tmp", dir=timbre_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as audio_file:
            audio_file.write(audio_data)
            audio_file.flush()
            os.fsync(audio_file.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(timbre_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def _generate_segments(
    config: GatewayConfig,
    request: VoiceDesignRequest,
    text_chunks: list[str],
) -> list[bytes]:
    """在同一 GPU 租约内生成并检查请求的全部音频段。"""

    def generate(client: LocalAIClient) -> list[bytes]:
        segments = []
        for text_chunk in text_chunks:
            audio_data, _ = client.design_voice(
                text=text_chunk,
                instruction=request.voice_description,
                language=request.language,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                temperature=request.temperature,
            )
            _decode_wav(audio_data)
            segments.append(audio_data)
        return segments

    return execute_localai_operation(config=config, operation=generate)


def create_app(
    *,
    timbre_dir: str | Path | None = None,
    gateway_config: GatewayConfig | None = None,
) -> FastAPI:
    """创建 VoiceDesign HTTP 服务。"""
    config = gateway_config or _load_gateway_config()
    if config.provider != "localai":
        raise GatewayConfigurationError("Qwen3 VoiceDesign 服务只接受 LocalAI/audio.cpp provider。")
    if config.model != DEFAULT_MODEL:
        raise GatewayConfigurationError(f"Qwen3 VoiceDesign 只接受固定模型 ID {DEFAULT_MODEL}。")
    resolved_timbre_dir = (
        Path(timbre_dir or os.getenv("TIMBRE_STORAGE_DIR", REPOSITORY_DIR / "storage/timbre"))
        .expanduser()
        .resolve()
    )
    app = FastAPI(title="Qwen3-TTS VoiceDesign")
    app.state.gateway_config = config

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        """检查 LocalAI readiness 和 VoiceDesign 候选注册，不加载模型。"""
        client = LocalAIClient(config)
        localai_ready = client.health()
        model_ready = localai_ready and client.model_ready()
        result = {
            "status": "ok" if model_ready else "unavailable",
            "dependencies": {"localai": localai_ready, "model_registered": model_ready},
        }
        if not model_ready:
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.post("/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign")
    def design_voice(request: VoiceDesignRequest) -> FileResponse:
        """通过 audio.cpp 合成并保存音色设计 WAV。"""
        chunks = _split_text(request.text, request.max_chars_per_chunk)
        try:
            segment_wavs = _generate_segments(config, request, chunks)
        except GpuLockTimeoutError as exc:
            raise HTTPException(status_code=503, detail="GPU 推理队列等待超时。") from exc
        except BackendQuarantinedError as exc:
            raise HTTPException(status_code=503, detail="LocalAI backend 已隔离。") from exc
        except LocalAIUnavailableError as exc:
            raise HTTPException(status_code=503, detail="LocalAI 暂不可用。") from exc
        except (InvalidAudioError, LocalAIRequestError) as exc:
            LOGGER.warning("VoiceDesign backend 返回无效音频：%s", exc)
            raise HTTPException(status_code=502, detail="LocalAI 未返回有效 WAV 音频。") from exc
        except Exception as exc:
            LOGGER.exception("Qwen3 VoiceDesign 推理失败。")
            raise HTTPException(status_code=500, detail="Qwen3 VoiceDesign 推理失败。") from exc

        try:
            if len(segment_wavs) == 1:
                output_data = segment_wavs[0]
                params, _ = _decode_wav(output_data)
                sample_rate = params.framerate
            else:
                output_data, sample_rate = _combine_wav_segments(segment_wavs, request.pause_ms)
            output_path = _persist_timbre(output_data, resolved_timbre_dir)
        except InvalidAudioError as exc:
            raise HTTPException(status_code=502, detail="LocalAI 未返回兼容的 PCM WAV。") from exc
        except OSError as exc:
            LOGGER.exception("保存 VoiceDesign 输出失败。")
            raise HTTPException(status_code=500, detail="VoiceDesign WAV 保存失败。") from exc

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_path.name,
            headers={"X-Audio-Sample-Rate": str(sample_rate)},
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("QWEN_VOICEDESIGN_HOST", os.getenv("HOST", "0.0.0.0")),
        port=int(os.getenv("QWEN_VOICEDESIGN_PORT", os.getenv("PORT", "8301"))),
    )
