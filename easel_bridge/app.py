"""为 Easel 提供带 Bearer 鉴权的本地 OpenAI-compatible API。"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import subprocess
import urllib.parse
import wave
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from llm_local_runtime.telemetry import TelemetryMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CHAT_MODEL_ID = "Ornith-1.5-9B-GGUF"
MOSS_MODEL_ID = "MOSS-Image-SFT-Q8_0"
VOXCPM_MODEL_ID = "OpenBMB/VoxCPM2-GGUF"

DEFAULT_CHAT_URL = "http://127.0.0.1:8503/v1/ornith-ai/Ornith-1.5-9B-GGUF"
DEFAULT_MOSS_URL = "http://127.0.0.1:8501/v1/openmoss/MOSS-Image-SFT-Q8_0"
DEFAULT_VOXCPM_URL = "http://127.0.0.1:8313/v1/OpenBMB/VoxCPM2-GGUF"

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class BridgeSettings:
    """网关鉴权、上游地址和资源上限。"""

    api_key: str
    chat_url: str
    moss_url: str
    voxcpm_url: str
    timeout_seconds: float
    max_request_bytes: int
    max_response_bytes: int
    ffmpeg_bin: str
    ffmpeg_timeout_seconds: float


def _positive_float(name: str, value: str, default: float) -> float:
    """读取正浮点配置，错误值在启动时明确报出。"""
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正数。") from exc
    if result <= 0:
        raise ValueError(f"{name} 必须是正数。")
    return result if value else default


def _positive_int(name: str, value: str, default: int) -> int:
    """读取正整数配置。"""
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数。") from exc
    if result <= 0:
        raise ValueError(f"{name} 必须是正整数。")
    return result if value else default


def _validate_upstream_url(name: str, value: str) -> str:
    """限制上游地址为无凭据的 HTTP(S) URL。"""
    try:
        parsed = urllib.parse.urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.port != 0
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须是有效的 http(s) URL。")
    return value.rstrip("/")


def _settings_from_environment(
    *,
    api_key: str | None,
    chat_url: str | None,
    moss_url: str | None,
    voxcpm_url: str | None,
    timeout_seconds: float | None,
    max_request_bytes: int | None,
    max_response_bytes: int | None,
    ffmpeg_bin: str | None,
    ffmpeg_timeout_seconds: float | None,
) -> BridgeSettings:
    """合并环境配置和 app factory 覆盖项。"""
    env = os.environ
    timeout_text = env.get("EASEL_BRIDGE_TIMEOUT", "900")
    max_request_text = env.get("EASEL_BRIDGE_MAX_REQUEST_BYTES", str(12 * 1024 * 1024))
    max_response_text = env.get("EASEL_BRIDGE_MAX_RESPONSE_BYTES", str(64 * 1024 * 1024))
    ffmpeg_timeout_text = env.get("EASEL_BRIDGE_FFMPEG_TIMEOUT", "120")
    timeout = (
        _positive_float("EASEL_BRIDGE_TIMEOUT", timeout_text, 900)
        if timeout_seconds is None
        else timeout_seconds
    )
    request_limit = (
        _positive_int("EASEL_BRIDGE_MAX_REQUEST_BYTES", max_request_text, 12 * 1024 * 1024)
        if max_request_bytes is None
        else max_request_bytes
    )
    response_limit = (
        _positive_int("EASEL_BRIDGE_MAX_RESPONSE_BYTES", max_response_text, 64 * 1024 * 1024)
        if max_response_bytes is None
        else max_response_bytes
    )
    ffmpeg_timeout = (
        _positive_float("EASEL_BRIDGE_FFMPEG_TIMEOUT", ffmpeg_timeout_text, 120)
        if ffmpeg_timeout_seconds is None
        else ffmpeg_timeout_seconds
    )
    if timeout <= 0 or ffmpeg_timeout <= 0:
        raise ValueError("超时配置必须是正数。")
    if request_limit <= 0 or response_limit <= 0:
        raise ValueError("请求和响应大小限制必须是正整数。")
    return BridgeSettings(
        api_key=(env.get("EASEL_BRIDGE_API_KEY", "") if api_key is None else api_key).strip(),
        chat_url=_validate_upstream_url(
            "EASEL_BRIDGE_CHAT_URL", chat_url or env.get("EASEL_BRIDGE_CHAT_URL", DEFAULT_CHAT_URL)
        ),
        moss_url=_validate_upstream_url(
            "EASEL_BRIDGE_MOSS_URL", moss_url or env.get("EASEL_BRIDGE_MOSS_URL", DEFAULT_MOSS_URL)
        ),
        voxcpm_url=_validate_upstream_url(
            "EASEL_BRIDGE_VOXCPM_URL",
            voxcpm_url or env.get("EASEL_BRIDGE_VOXCPM_URL", DEFAULT_VOXCPM_URL),
        ),
        timeout_seconds=timeout,
        max_request_bytes=request_limit,
        max_response_bytes=response_limit,
        ffmpeg_bin=ffmpeg_bin or env.get("EASEL_BRIDGE_FFMPEG_BIN", "ffmpeg"),
        ffmpeg_timeout_seconds=ffmpeg_timeout,
    )


class RequestBodyLimitMiddleware:
    """缓冲并限制 JSON 请求体，避免 chunked 请求绕过 Content-Length 检查。"""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > self.max_bytes:
                    await self._too_large(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse({"detail": "Content-Length 无效。"}, status_code=400)
                await response(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered
            if delivered:
                # 请求体结束不等于连接断开；流式响应需监听真实客户端事件。
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"请求体超过 {self.max_bytes} 字节限制。"}, status_code=413
        )
        await response(scope, receive, send)


class ImageGenerationRequest(BaseModel):
    """Easel 使用的单张 MOSS 文生图请求。"""

    model_config = ConfigDict(extra="ignore")

    model: str = MOSS_MODEL_ID
    prompt: str = Field(min_length=1, max_length=65_536)
    size: str = Field(default="1024x1024", pattern=r"^[0-9]{2,4}x[0-9]{2,4}$")
    n: int = Field(default=1, ge=1, le=4)
    response_format: Literal["b64_json"] = "b64_json"
    negative_prompt: str | None = Field(default=None, max_length=65_536)
    steps: int | None = Field(default=None, ge=1, le=50)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    seed: int | None = Field(default=None, ge=-1, le=2_147_483_647)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """拒绝空白提示词和空字符。"""
        if not value.strip() or "\0" in value:
            raise ValueError("prompt 必须包含可见文本且不能包含空字符。")
        return value


class SpeechRequest(BaseModel):
    """OpenAI TTS 风格的 VoxCPM2 请求子集。"""

    model_config = ConfigDict(extra="ignore")

    model: str = VOXCPM_MODEL_ID
    input: str = Field(min_length=1, max_length=12_000)
    voice: str = "alloy"
    response_format: Literal["mp3", "wav"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(default=None, max_length=1_000)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        """拒绝空白语音输入和空字符。"""
        if not value.strip() or "\0" in value:
            raise ValueError("input 必须包含可见文本且不能包含空字符。")
        return value

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str) -> str:
        """目前只接受 Easel 默认兼容音色名；实际音色通过 instructions 指定。"""
        if value != "alloy":
            raise ValueError("VoxCPM2 网关目前只接受 voice=alloy；请用 instructions 描述音色。")
        return value

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, value: float) -> float:
        """VoxCPM2 服务合同没有 speed 参数，因此仅接受默认速度。"""
        if value != 1.0:
            raise ValueError("当前 VoxCPM2 接口不支持调整 speed。")
        return value


def _error(
    status_code: int, message: str, error_type: str = "invalid_request_error"
) -> JSONResponse:
    """生成 OpenAI 风格错误响应。"""
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": status_code,
            }
        },
        status_code=status_code,
    )


def _upstream_headers(headers: httpx.Headers) -> dict[str, str]:
    """仅透传内容类型及少量安全响应头。"""
    allowed = {"content-type", "cache-control", "retry-after", "x-request-id"}
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in allowed and name.lower() not in _HOP_BY_HOP_HEADERS
    }


async def _read_limited(response: httpx.Response, max_bytes: int) -> bytes:
    """按块读取上游响应，并拒绝超出内存上限的内容。"""
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > max_bytes:
            raise HTTPException(status_code=502, detail="上游响应超过网关大小限制。")
        content.extend(chunk)
    return bytes(content)


def _transcode_wav_to_mp3(audio: bytes, *, binary: str, timeout: float) -> bytes:
    """通过 ffmpeg stdin/stdout 转换音频，不写入临时文件。"""
    if shutil.which(binary) is None:
        raise FileNotFoundError("找不到 ffmpeg。")
    result = subprocess.run(
        [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=audio,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0 or not result.stdout.startswith(b"ID3"):
        raise RuntimeError("ffmpeg 未能生成有效 MP3。")
    return result.stdout


def _is_wav(audio: bytes) -> bool:
    """检查 VoxCPM2 响应是否包含 RIFF/WAVE 数据。"""
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return False
    try:
        with wave.open(BytesIO(audio), "rb") as wav_file:
            return wav_file.getnframes() > 0
    except (EOFError, OSError, wave.Error):
        return False


def create_app(
    *,
    api_key: str | None = None,
    chat_url: str | None = None,
    moss_url: str | None = None,
    voxcpm_url: str | None = None,
    timeout_seconds: float | None = None,
    max_request_bytes: int | None = None,
    max_response_bytes: int | None = None,
    ffmpeg_bin: str | None = None,
    ffmpeg_timeout_seconds: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    wav_to_mp3: Callable[[bytes], bytes] | None = None,
) -> FastAPI:
    """创建 Easel 网关，保留上游服务自身的 GPU 锁和模型生命周期。"""
    settings = _settings_from_environment(
        api_key=api_key,
        chat_url=chat_url,
        moss_url=moss_url,
        voxcpm_url=voxcpm_url,
        timeout_seconds=timeout_seconds,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not settings.api_key:
            raise RuntimeError("必须设置 EASEL_BRIDGE_API_KEY；网关不会以匿名模式启动。")
        if len(settings.api_key.encode("utf-8")) < 32 or settings.api_key.lower().startswith(
            "replace-with-"
        ):
            raise RuntimeError(
                "EASEL_BRIDGE_API_KEY 必须是至少 32 字节的随机值，不能使用模板占位符。"
            )
        yield

    app = FastAPI(title="Easel Local Model Bridge", lifespan=lifespan)
    app.add_middleware(TelemetryMiddleware, service="bridge", paths={"/v1/chat/completions"})
    app.state.bridge_settings = settings
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    def verify_key(authorization: Annotated[str | None, Header()] = None) -> None:
        """验证 Bearer Key，不把客户端密钥转发给上游。"""
        expected = f"Bearer {settings.api_key}"
        if (
            not settings.api_key
            or authorization is None
            or not hmac.compare_digest(authorization.encode("utf-8"), expected.encode("utf-8"))
        ):
            raise HTTPException(
                status_code=401,
                detail="缺少或无效的 Bearer Key。",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/v1/models")
    def models(_auth: None = Depends(verify_key)) -> dict[str, Any]:
        """返回网关支持的聊天、图片和语音模型。"""
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "llm-local"}
                for model_id in (CHAT_MODEL_ID, MOSS_MODEL_ID, VOXCPM_MODEL_ID)
            ],
        }

    async def post_json(url: str, payload: dict[str, Any]) -> Response:
        """把 JSON 请求转发到服务端配置的固定上游地址。"""
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=settings.timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                async with client.stream("POST", url, json=payload) as upstream:
                    body = await _read_limited(upstream, settings.max_response_bytes)
                    return Response(
                        content=body,
                        status_code=upstream.status_code,
                        headers=_upstream_headers(upstream.headers),
                    )
        except httpx.TimeoutException:
            return _error(504, "等待本地模型响应超时。", "upstream_timeout")
        except httpx.RequestError:
            return _error(502, "无法连接本地模型服务。", "upstream_error")
        except HTTPException as exc:
            return _error(exc.status_code, str(exc.detail), "upstream_error")

    @app.post("/v1/images/generations")
    async def image_generation(
        payload: ImageGenerationRequest,
        _auth: None = Depends(verify_key),
    ) -> Response:
        """把 OpenAI Images 请求映射到现有 MOSS 单图 API。"""
        if payload.model != MOSS_MODEL_ID:
            return _error(404, f"未知图片模型：{payload.model}", "model_not_found")
        forwarded = payload.model_dump(exclude_unset=True)
        forwarded["model"] = MOSS_MODEL_ID
        forwarded["n"] = 1
        forwarded["response_format"] = "b64_json"
        if payload.n == 1:
            return await post_json(settings.moss_url, forwarded)

        data: list[dict[str, Any]] = []
        created: int | None = None
        for _ in range(payload.n):
            response = await post_json(settings.moss_url, forwarded)
            if response.status_code >= 400:
                return response
            try:
                result = json.loads(response.body)
                items = result["data"]
                if not isinstance(items, list) or not items:
                    raise ValueError("上游图片响应缺少 data。")
                data.extend(items)
                if created is None:
                    created = result.get("created")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return _error(502, "MOSS 返回了无效的图片响应。", "upstream_error")
        return JSONResponse({"created": created, "data": data})

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        _auth: None = Depends(verify_key),
    ) -> Response:
        """把 OpenAI Chat Completions 请求转发给已有 Ornith 适配器。"""
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return _error(400, "请求体不是有效 JSON。")
        if not isinstance(payload, dict):
            return _error(400, "请求体必须是 JSON 对象。")
        if payload.get("model") != CHAT_MODEL_ID:
            return _error(404, f"未知聊天模型：{payload.get('model', '')}", "model_not_found")

        client = httpx.AsyncClient(
            transport=transport,
            timeout=settings.timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        try:
            trace = request.scope.get("model_telemetry")
            headers = {"X-Model-Request-ID": trace.request_id} if trace else {}
            upstream_request = client.build_request(
                "POST", settings.chat_url, json=payload, headers=headers
            )
            upstream = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            await client.aclose()
            return _error(504, "等待本地模型响应超时。", "upstream_timeout")
        except httpx.RequestError:
            await client.aclose()
            return _error(502, "无法连接本地模型服务。", "upstream_error")

        if upstream.status_code >= 400 or not payload.get("stream"):
            try:
                body = await _read_limited(upstream, settings.max_response_bytes)
                headers = _upstream_headers(upstream.headers)
                return Response(content=body, status_code=upstream.status_code, headers=headers)
            except HTTPException as exc:
                return _error(exc.status_code, str(exc.detail), "upstream_error")
            finally:
                await upstream.aclose()
                await client.aclose()

        async def stream_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=_upstream_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    @app.post("/v1/audio/speech")
    async def speech(
        payload: SpeechRequest,
        _auth: None = Depends(verify_key),
    ) -> Response:
        """把 OpenAI TTS JSON 请求映射到 VoxCPM2 表单，并可转成 MP3。"""
        if payload.model != VOXCPM_MODEL_ID:
            return _error(404, f"未知语音模型：{payload.model}", "model_not_found")
        form: dict[str, str] = {"text": payload.input}
        if payload.instructions:
            form["instruction"] = payload.instructions

        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=settings.timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                multipart = {name: (None, value) for name, value in form.items()}
                async with client.stream("POST", settings.voxcpm_url, files=multipart) as upstream:
                    audio = await _read_limited(upstream, settings.max_response_bytes)
                    if upstream.status_code >= 400:
                        return Response(
                            content=audio,
                            status_code=upstream.status_code,
                            headers=_upstream_headers(upstream.headers),
                        )
                    if not _is_wav(audio):
                        return _error(502, "VoxCPM2 服务未返回有效 WAV。", "upstream_error")
        except httpx.TimeoutException:
            return _error(504, "等待 VoxCPM2 响应超时。", "upstream_timeout")
        except httpx.RequestError:
            return _error(502, "无法连接 VoxCPM2 服务。", "upstream_error")
        except HTTPException as exc:
            return _error(exc.status_code, str(exc.detail), "upstream_error")

        if payload.response_format == "wav":
            return Response(content=audio, media_type="audio/wav")
        try:
            converted = (
                wav_to_mp3(audio)
                if wav_to_mp3 is not None
                else await asyncio.to_thread(
                    _transcode_wav_to_mp3,
                    audio,
                    binary=settings.ffmpeg_bin,
                    timeout=settings.ffmpeg_timeout_seconds,
                )
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
            return _error(503, f"无法将 WAV 转为 MP3：{exc}", "transcoding_error")
        return Response(content=converted, media_type="audio/mpeg")

    return app


app = create_app()
