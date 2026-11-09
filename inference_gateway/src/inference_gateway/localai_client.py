"""使用标准库向私有 LocalAI 发送流式 multipart 请求。"""

from __future__ import annotations

import http.client
import json
import mimetypes
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

from .config import GatewayConfig
from .errors import LocalAIRequestError, LocalAIUnavailableError

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_AUDIO_BYTES = 512 * 1024 * 1024


class LocalAIClient:
    """LocalAI OpenAI-compatible API 与 backend 生命周期客户端。"""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        parsed = urlsplit(config.base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._base_path = parsed.path.rstrip("/")
        self._headers = {"Accept": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    def design_voice(
        self,
        *,
        text: str,
        instruction: str,
        language: str | None,
        max_new_tokens: int,
        top_p: float | None,
        temperature: float | None,
    ) -> tuple[bytes, float]:
        """用 LocalAI Qwen3 VoiceDesign 模型根据指令生成单段语音。"""
        params = {"max_tokens": str(max_new_tokens)}
        if top_p is not None:
            params["top_p"] = str(top_p)
        if temperature is not None:
            params["temperature"] = str(temperature)
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "instructions": instruction,
            "response_format": "wav",
            "params": params,
        }
        if language:
            request_body["language"] = language
        return self._post_speech(request_body)

    def _post_speech(self, request_body: dict[str, object]) -> tuple[bytes, float]:
        """发送 TTS JSON 请求并检查大小受限的 RIFF/WAVE 响应。"""

        started = time.perf_counter()
        response = self._request(
            "POST",
            "/v1/audio/speech",
            body=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={**self._headers, "Content-Type": "application/json"},
            timeout=self._config.request_timeout,
            response_limit=_MAX_AUDIO_BYTES,
            expect_json=False,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(response, tuple) or len(response) != 2:
            raise LocalAIRequestError("LocalAI TTS 响应格式无效。")
        audio_bytes, content_type = response
        if not isinstance(audio_bytes, bytes) or len(audio_bytes) < 12:
            raise LocalAIRequestError("LocalAI TTS 未返回有效 WAV 音频。")
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise LocalAIRequestError("LocalAI TTS 响应不是 RIFF/WAVE 音频。")
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            raise LocalAIRequestError("LocalAI TTS 响应 Content-Type 不是 WAV。")
        return audio_bytes, elapsed

    def synthesize_speech(self, request_body: dict[str, object]) -> bytes:
        """通过 LocalAI speech API 合成并返回 WAV。"""
        audio_bytes, _ = self._post_speech(request_body)
        return audio_bytes

    def transcribe(
        self,
        *,
        filename: str,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, object]:
        """将有界上传文件流式转交 LocalAI transcription API。"""
        fields = {"model": self._config.model, "response_format": "json"}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt
        result = self._post_multipart(
            "/v1/audio/transcriptions",
            fields=fields,
            files={
                "file": (
                    filename,
                    audio_path,
                    mimetypes.guess_type(filename)[0] or "application/octet-stream",
                )
            },
            expect_json=True,
        )
        if not isinstance(result, dict):
            raise LocalAIRequestError("LocalAI transcription 响应必须是 JSON object。")
        return result

    def generate_sound(self, request_body: dict[str, object]) -> bytes:
        """通过 LocalAI sound-generation API 生成 WAV。"""
        body = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode()
        response = self._request(
            "POST",
            "/v1/sound-generation",
            body=body,
            headers={**self._headers, "Content-Type": "application/json"},
            timeout=self._config.request_timeout,
            response_limit=_MAX_AUDIO_BYTES,
            expect_json=False,
        )
        if not isinstance(response, tuple) or not isinstance(response[0], bytes):
            raise LocalAIRequestError("LocalAI sound-generation 响应格式无效。")
        audio_bytes = response[0]
        if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise LocalAIRequestError("LocalAI sound-generation 未返回 RIFF/WAVE 音频。")
        return audio_bytes

    def transform_audio(
        self,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, Path, str]],
    ) -> bytes:
        """流式提交源音频和参考音频，调用 LocalAI audio-transform API。"""
        response = self._post_multipart(
            "/audio/transform",
            fields=fields,
            files=files,
            expect_json=False,
        )
        if not isinstance(response, tuple) or not isinstance(response[0], bytes):
            raise LocalAIRequestError("LocalAI audio-transform 响应格式无效。")
        audio_bytes = response[0]
        if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise LocalAIRequestError("LocalAI audio-transform 未返回 RIFF/WAVE 音频。")
        return audio_bytes

    def _post_multipart(
        self,
        path: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, Path, str]],
        expect_json: bool,
    ) -> object:
        """以固定 Content-Length 分块发送 multipart，避免复制整段输入音频。"""
        boundary = f"llm-local-{time.time_ns():x}"
        body_parts: list[bytes | Path] = []
        for name, value in fields.items():
            body_parts.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                )
            )
        for name, (filename, file_path, content_type) in files.items():
            safe_filename = Path(filename).name.replace('"', "")
            if any(character in name + safe_filename for character in "\r\n"):
                raise LocalAIRequestError("multipart 字段名或文件名含有无效换行。")
            body_parts.extend(
                (
                    f"--{boundary}\r\n".encode(),
                    (
                        f'Content-Disposition: form-data; name="{name}"; '
                        f'filename="{safe_filename}"\r\n'
                    ).encode(),
                    f"Content-Type: {content_type}\r\n\r\n".encode(),
                    file_path,
                    b"\r\n",
                )
            )
        body_parts.append(f"--{boundary}--\r\n".encode())
        content_length = sum(
            part.stat().st_size if isinstance(part, Path) else len(part) for part in body_parts
        )

        def iter_body() -> Iterable[bytes]:
            for part in body_parts:
                if isinstance(part, bytes):
                    yield part
                    continue
                with part.open("rb") as audio_file:
                    while chunk := audio_file.read(64 * 1024):
                        yield chunk

        return self._request(
            "POST",
            path,
            body=iter_body(),
            headers={
                **self._headers,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(content_length),
            },
            timeout=self._config.request_timeout,
            response_limit=_MAX_JSON_BYTES if expect_json else _MAX_AUDIO_BYTES,
            expect_json=expect_json,
        )

    def shutdown_model(self) -> None:
        """调用 LocalAI backend/shutdown 并要求得到停止确认。"""
        body = json.dumps({"model": self._config.model}, separators=(",", ":")).encode()
        self._request(
            "POST",
            "/backend/shutdown",
            body=body,
            headers={**self._headers, "Content-Type": "application/json"},
            timeout=self._config.cleanup_timeout,
        )

    def health(self) -> bool:
        """查询 LocalAI 健康端点，不触发模型加载。"""
        try:
            self._request("GET", "/readyz", timeout=self._config.connect_timeout)
        except (LocalAIRequestError, LocalAIUnavailableError):
            return False
        return True

    def model_ready(self) -> bool:
        """确认 OpenAI 模型目录中已注册所选模型，但不加载模型。"""
        try:
            payload = self._request("GET", "/v1/models", timeout=self._config.connect_timeout)
        except (LocalAIRequestError, LocalAIUnavailableError):
            return False
        if not isinstance(payload, dict):
            return False
        models = payload.get("data")
        if not isinstance(models, list):
            return False
        return any(
            isinstance(item, dict)
            and (item.get("id") == self._config.model or item.get("name") == self._config.model)
            for item in models
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        body: bytes | Iterable[bytes] | None = None,
        headers: dict[str, str] | None = None,
        response_limit: int = _MAX_JSON_BYTES,
        expect_json: bool = True,
    ) -> object:
        """执行有界 HTTP 请求并限制 JSON 或音频响应大小。"""
        connection = http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=min(self._config.connect_timeout, timeout),
        )
        request_path = f"{self._base_path}{path}" or path
        deadline = time.monotonic() + timeout

        def set_remaining_timeout() -> None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalAIUnavailableError("LocalAI 请求超过总超时。")
            if connection.sock is not None:
                connection.sock.settimeout(remaining)

        request_body = body
        if body is not None and not isinstance(body, bytes):

            def bounded_body() -> Iterable[bytes]:
                for chunk in body:
                    set_remaining_timeout()
                    yield chunk

            request_body = bounded_body()

        try:
            connection.connect()
            set_remaining_timeout()
            connection.request(
                method,
                request_path,
                body=request_body,
                headers=headers or self._headers,
            )
            set_remaining_timeout()
            response = connection.getresponse()
            response_buffer = bytearray()
            while len(response_buffer) <= response_limit:
                set_remaining_timeout()
                remaining_bytes = response_limit + 1 - len(response_buffer)
                chunk = response.read1(min(64 * 1024, remaining_bytes))
                if not chunk:
                    break
                response_buffer.extend(chunk)
            response_body = bytes(response_buffer)
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            raise LocalAIUnavailableError("LocalAI 请求失败或超时。") from exc
        finally:
            connection.close()

        if len(response_body) > response_limit:
            raise LocalAIRequestError(
                f"LocalAI 响应超过 {response_limit // (1024 * 1024)} MiB 上限。"
            )
        if response.status < 200 or response.status >= 300:
            detail = response_body.decode("utf-8", errors="replace")[:2_000]
            raise LocalAIRequestError(f"LocalAI HTTP {response.status}: {detail}")
        if not expect_json:
            return response_body, response.getheader("Content-Type", "")
        if not response_body:
            return {}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError:
            return response_body.decode("utf-8", errors="replace")
