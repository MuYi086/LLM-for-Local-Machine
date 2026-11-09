"""共享八个 audio.cpp 模型 API 的请求校验、GPU 排队和 LocalAI 编排。"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
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
from llm_local_runtime import (
    AudioUploadError,
    GpuLockTimeoutError,
    UploadPolicy,
    stage_audio_upload,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .specs import MODEL_SPECS, ModelSpec

_ALLOWED_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}


class SoundGenerationRequest(BaseModel):
    """MiDashengLM-Gen 或 Stable Audio SFX 生成请求。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2_000)
    duration_seconds: float = Field(default=8, gt=0, le=30)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """拒绝仅含空格的提示词。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("text 不能只包含空白字符。")
        return normalized


def _load_config(spec: ModelSpec) -> GatewayConfig:
    """读取 LocalAI、共享锁与请求超时配置，并固定服务模型 ID。"""
    try:
        timeout = float(os.getenv("AUDIO_CPP_REQUEST_TIMEOUT", "900"))
    except ValueError as exc:
        raise GatewayConfigurationError("AUDIO_CPP_REQUEST_TIMEOUT 必须是正数。") from exc
    values = dict(os.environ)
    values.pop("LLM_LOCAL_AUDIO_CPP_MODEL", None)
    config = load_gateway_config(
        service_provider_env="LLM_LOCAL_AUDIO_CPP_PROVIDER",
        default_model=spec.model_id,
        request_timeout=timeout,
        environ=values,
    )
    if config.provider != "localai":
        raise GatewayConfigurationError("audio.cpp 模型服务固定使用 LocalAI provider。")
    if config.model != spec.model_id:
        raise GatewayConfigurationError(f"模型 ID 必须固定为 {spec.model_id}。")
    return config


def _require_nonempty_text(name: str, value: str, max_length: int) -> str:
    """规范化并限制请求文本。"""
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{name} 不能为空。")
    if len(normalized) > max_length:
        raise HTTPException(status_code=422, detail=f"{name} 长度不能超过 {max_length} 字符。")
    return normalized


def _stage_upload(upload: UploadFile, *, require_wav: bool = False) -> Path:
    """将单个有界音频上传暂存到唯一文件，供后端安全读取。"""
    original_name = Path(upload.filename or "audio.wav").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in _ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=415, detail="只接受 WAV、MP3、FLAC、OGG、M4A、AAC 或 WebM 音频。"
        )
    if require_wav and suffix != ".wav":
        raise HTTPException(status_code=415, detail="该任务的参考音频必须为 WAV。")

    temporary_dir = Path(
        os.getenv(
            "AUDIO_CPP_TEMP_DIR",
            str(
                Path(os.getenv("GPU_LOCK_FILE", "storage/.cache/runtime/gpu-runtime.lock")).parent
                / "audio_cpp_uploads"
            ),
        )
    ).expanduser()
    try:
        staged = asyncio.run(
            stage_audio_upload(
                upload,
                temporary_dir,
                policy=UploadPolicy.from_environment(),
            )
        )
    except AudioUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    staged_path = staged.path
    temporary_path = staged_path.with_suffix("")
    try:
        if require_wav:
            with staged_path.open("rb") as source:
                header = source.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise HTTPException(status_code=415, detail="参考音频不是有效 RIFF/WAVE 文件。")
        os.replace(staged_path, temporary_path)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _valid_wav(audio: bytes) -> bytes:
    """确认 LocalAI 响应包含可解码 PCM WAV 音频。"""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)
    except (EOFError, OSError, wave.Error) as exc:
        raise HTTPException(
            status_code=502, detail="LocalAI/audio.cpp 未返回有效 PCM WAV。"
        ) from exc
    if (
        params.comptype != "NONE"
        or params.nchannels not in {1, 2}
        or params.sampwidth not in {1, 2, 3, 4}
        or params.framerate <= 0
        or params.nframes <= 0
        or len(frames) != params.nframes * params.nchannels * params.sampwidth
    ):
        raise HTTPException(status_code=502, detail="LocalAI/audio.cpp WAV 采样数据无效。")
    return audio


def _audio_response(spec: ModelSpec, audio: bytes) -> Response:
    """返回经过验证的 WAV 下载响应。"""
    _valid_wav(audio)
    filename = f"{spec.key}-{uuid.uuid4().hex[:12]}.wav"
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _run_locked(config: GatewayConfig, operation: Callable[[LocalAIClient], Any]) -> Any:
    """在共享 GPU 锁下运行 LocalAI 操作，并把队列/后端错误映射为 HTTP。"""
    try:
        return execute_localai_operation(config=config, operation=operation)
    except (GpuLockTimeoutError, BackendQuarantinedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LocalAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LocalAIUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _generate_tts(
    spec: ModelSpec,
    config: GatewayConfig,
    *,
    text: str,
    reference_audio: UploadFile | None,
    reference_text: str | None,
    language: str | None,
    instruction: str | None = None,
) -> Response:
    """准备 TTS 请求和参考音频路径，在全局 GPU 锁下合成语音。"""
    normalized_text = _require_nonempty_text("text", text, 12_000)
    normalized_instruction = (
        _require_nonempty_text("instruction", instruction, 1_000) if instruction else None
    )
    normalized_reference_text = (
        _require_nonempty_text("reference_text", reference_text, 12_000) if reference_text else None
    )
    if spec.key == "qwen3_tts_base" and normalized_reference_text is None:
        raise HTTPException(status_code=422, detail="Qwen3-TTS Base 必须提供 reference_text。")
    reference_path = _stage_upload(reference_audio, require_wav=True) if reference_audio else None
    request_text = (
        f"({normalized_instruction}) {normalized_text}"
        if normalized_instruction
        else normalized_text
    )
    request: dict[str, object] = {
        "model": spec.model_id,
        "input": request_text,
        "response_format": "wav",
    }
    if reference_path is not None:
        request["voice"] = str(reference_path.resolve())
    if language:
        normalized_language = _require_nonempty_text("language", language, 32)
        if spec.key == "qwen3_tts_base" and normalized_language.lower() in {"zh", "zh-cn"}:
            normalized_language = "Chinese"
        request["language"] = normalized_language
    if normalized_reference_text:
        option_name = "reference_text" if spec.key == "qwen3_tts_base" else "ref_text"
        request["params"] = {option_name: normalized_reference_text}
    try:
        audio = _run_locked(config, lambda client: client.synthesize_speech(request))
        return _audio_response(spec, audio)
    finally:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)


def _transcribe(
    spec: ModelSpec,
    config: GatewayConfig,
    *,
    audio_upload: UploadFile,
    language: str | None,
    prompt: str | None,
) -> JSONResponse:
    """转写上传音频并返回 OpenAI transcription JSON。"""
    normalized_language = _require_nonempty_text("language", language, 32) if language else None
    normalized_prompt = _require_nonempty_text("prompt", prompt, 2_000) if prompt else None
    audio_path = _stage_upload(audio_upload)
    try:
        result = _run_locked(
            config,
            lambda client: client.transcribe(
                filename=Path(audio_upload.filename or "audio.wav").name,
                audio_path=audio_path,
                language=normalized_language,
                prompt=normalized_prompt,
            ),
        )
    finally:
        audio_path.unlink(missing_ok=True)
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise HTTPException(status_code=502, detail="LocalAI/audio.cpp 未返回 transcription JSON。")
    return JSONResponse(result)


def _generate_sound(
    spec: ModelSpec,
    config: GatewayConfig,
    payload: SoundGenerationRequest,
) -> Response:
    """根据文本提示生成音效或结构化混合音频。"""
    request = {
        "model_id": spec.model_id,
        "text": payload.text,
    }
    if spec.key == "midashenglm_gen":
        # audio.cpp 将 MiDasheng 时长作为模型专属 duration_sec 选项读取。
        request["params"] = {"duration_sec": str(payload.duration_seconds)}
    else:
        request["duration_seconds"] = payload.duration_seconds
    audio = _run_locked(config, lambda client: client.generate_sound(request))
    return _audio_response(spec, audio)


def _convert_voice(
    spec: ModelSpec,
    config: GatewayConfig,
    *,
    audio_upload: UploadFile,
    reference_upload: UploadFile,
) -> Response:
    """通过 SeedVC v2_vc 路径转换音色，输入和目标参考均只暂存。"""
    source_path = _stage_upload(audio_upload, require_wav=True)
    reference_path: Path | None = None
    try:
        reference_path = _stage_upload(reference_upload, require_wav=True)
        audio = _run_locked(
            config,
            lambda client: client.transform_audio(
                fields={"model": spec.model_id, "response_format": "wav"},
                files={
                    "audio": (
                        source_path.name,
                        source_path,
                        audio_upload.content_type or "audio/wav",
                    ),
                    "reference": (
                        reference_path.name,
                        reference_path,
                        reference_upload.content_type or "audio/wav",
                    ),
                },
            ),
        )
        return _audio_response(spec, audio)
    finally:
        source_path.unlink(missing_ok=True)
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)


def create_app(model_key: str, gateway_config: GatewayConfig | None = None) -> FastAPI:
    """创建单模型 FastAPI 服务，并使其请求进入共享 GPU 队列。"""
    try:
        spec = MODEL_SPECS[model_key]
    except KeyError as exc:
        raise ValueError(f"未知 audio.cpp 服务 key：{model_key}") from exc
    config = gateway_config or _load_config(spec)
    if config.provider != "localai" or config.model != spec.model_id:
        raise GatewayConfigurationError("audio.cpp 服务必须绑定其清单中的固定 LocalAI 模型 ID。")

    app = FastAPI(title=spec.description)
    app.state.gateway_config = config
    app.state.model_spec = spec

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        """检查 LocalAI 和已注册模型，不触发模型加载。"""
        client = LocalAIClient(config)
        if not client.health():
            raise HTTPException(status_code=503, detail="LocalAI/audio.cpp 未就绪。")
        if not client.model_ready():
            raise HTTPException(status_code=503, detail=f"模型尚未注册：{spec.model_id}")
        return {"status": "ok", "model": spec.model_id, "backend": "LocalAI/audio.cpp"}

    if spec.kind == "tts_clone":

        @app.post(spec.route)
        def clone_tts(
            text: Annotated[str, Form(min_length=1, max_length=12_000)],
            reference_audio: Annotated[UploadFile, File(...)],
            reference_text: Annotated[str | None, Form(max_length=12_000)] = None,
            language: Annotated[str | None, Form(max_length=32)] = None,
        ) -> Response:
            """使用 WAV 参考音频生成克隆语音。"""
            return _generate_tts(
                spec,
                config,
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                language=language,
            )

    elif spec.kind == "tts_optional_reference":

        @app.post(spec.route)
        def tts_optional_reference(
            text: Annotated[str, Form(min_length=1, max_length=12_000)],
            reference_audio: Annotated[UploadFile | None, File()] = None,
            reference_text: Annotated[str | None, Form(max_length=12_000)] = None,
            language: Annotated[str | None, Form(max_length=32)] = None,
            instruction: Annotated[str | None, Form(max_length=1_000)] = None,
        ) -> Response:
            """生成 VoxCPM2 语音，可选提供参考音频或前置音色指令。"""
            return _generate_tts(
                spec,
                config,
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                language=language,
                instruction=instruction,
            )

    elif spec.kind == "sound_generation":

        @app.post(spec.route)
        def sound_generation(payload: SoundGenerationRequest) -> Response:
            """根据文本提示生成音频。"""
            return _generate_sound(spec, config, payload)

    elif spec.kind == "asr":

        @app.post(spec.route)
        def asr(
            audio: Annotated[UploadFile, File(...)],
            language: Annotated[str | None, Form(max_length=32)] = None,
            prompt: Annotated[str | None, Form(max_length=2_000)] = None,
        ) -> JSONResponse:
            """转写上传音频并返回 JSON 文本。"""
            return _transcribe(
                spec,
                config,
                audio_upload=audio,
                language=language,
                prompt=prompt,
            )

    elif spec.kind == "voice_conversion":

        @app.post(spec.route)
        def voice_conversion(
            audio: Annotated[UploadFile, File(...)],
            reference: Annotated[UploadFile, File(...)],
        ) -> Response:
            """将源音频转换为目标参考音色。"""
            return _convert_voice(
                spec,
                config,
                audio_upload=audio,
                reference_upload=reference,
            )

    else:
        raise RuntimeError(f"未实现 audio.cpp 服务类型：{spec.kind}")
    return app


def run_service(model_key: str) -> None:
    """以共享 uv 项目启动一个模型 HTTP 服务。"""
    spec = MODEL_SPECS[model_key]
    app = create_app(model_key)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", str(spec.port)))
    uvicorn.run(app, host=host, port=port)
