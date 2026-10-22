"""使用标准库向私有 LocalAI 发送流式 multipart 请求。"""

from __future__ import annotations

import http.client
import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

from .config import GatewayConfig
from .errors import LocalAIRequestError, LocalAIUnavailableError
from .worker_protocol import WorkerProtocolError, decode_worker_response, encode_worker_request

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_AUDIO_BYTES = 512 * 1024 * 1024
_MAX_WORKER_RESPONSE_BYTES = 180 * 1024 * 1024


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

    def run_worker(
        self,
        *,
        operation: str,
        payload: dict[str, object],
    ) -> tuple[dict[str, object], float]:
        """经 LocalAI completion RPC 调用版本化的一次性 LLM Local Backend worker。"""
        request_id = uuid.uuid4().hex
        try:
            prompt = encode_worker_request(
                model=self._config.model,
                operation=operation,
                request_id=request_id,
                payload=payload,
            )
        except WorkerProtocolError as exc:
            raise LocalAIRequestError(str(exc)) from exc
        request_body = json.dumps(
            {
                "model": self._config.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 1,
                "stream": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        started = time.perf_counter()
        response = self._request(
            "POST",
            "/v1/completions",
            body=request_body,
            headers={**self._headers, "Content-Type": "application/json"},
            timeout=self._config.request_timeout,
            response_limit=_MAX_WORKER_RESPONSE_BYTES,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(response, dict):
            raise LocalAIRequestError("LocalAI worker completion 响应必须是 JSON 对象。")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LocalAIRequestError("LocalAI worker completion 响应缺少 choices。")
        try:
            result = decode_worker_response(
                choices[0].get("text"),
                model=self._config.model,
                operation=operation,
                request_id=request_id,
            )
        except WorkerProtocolError as exc:
            raise LocalAIRequestError(str(exc)) from exc
        return result, elapsed

    def synthesize_speech(
        self,
        *,
        text: str,
        voice_path: Path,
        language: str | None,
        reference_text: str | None,
        x_vector_only: bool,
        max_new_tokens: int,
        top_p: float | None,
        temperature: float | None,
    ) -> tuple[bytes, float]:
        """调用 LocalAI TTS 并将参考音频及 Qwen3 克隆参数传入后端。"""
        params = {
            "max_tokens": str(max_new_tokens),
            "x_vector_only_mode": str(x_vector_only).lower(),
        }
        if reference_text:
            params["reference_text"] = reference_text
        if top_p is not None:
            params["top_p"] = str(top_p)
        if temperature is not None:
            params["temperature"] = str(temperature)
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "voice": str(voice_path),
            "response_format": "wav",
            "params": params,
        }
        if language:
            request_body["language"] = language

        return self._post_speech(request_body)

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
