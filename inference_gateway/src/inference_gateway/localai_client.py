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
_FILE_CHUNK_BYTES = 1024 * 1024
_AUDIO_CONTENT_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def _safe_multipart_filename(filename: str) -> str:
    """将客户端文件名限制为安全的 ASCII multipart 参数。"""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    safe_name = "".join(
        character
        if character.isascii() and (character.isalnum() or character in {".", "_", "-"})
        else "_"
        for character in basename
    )
    safe_name = safe_name.strip("._-")[:255]
    return safe_name or "audio"


def _validate_multipart_content_type(content_type: str) -> None:
    """拒绝会破坏 multipart 头边界的 MIME 值。"""
    if (
        not content_type
        or not content_type.isascii()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in content_type)
    ):
        raise LocalAIRequestError("上传音频 Content-Type 含无效字符。")


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

    def transcribe(
        self,
        *,
        audio_path: Path,
        filename: str,
        content_type: str,
        language: str | None,
    ) -> tuple[dict[str, object], float]:
        """流式上传单个音频文件并返回 JSON 转写及调用耗时。"""
        _validate_multipart_content_type(content_type)
        boundary = f"unitale-{uuid.uuid4().hex}"
        fields = {
            "model": self._config.model,
            "response_format": "verbose_json",
        }
        if language:
            fields["language"] = language
        prefix = bytearray()
        for name, value in fields.items():
            prefix.extend(f"--{boundary}\r\n".encode())
            prefix.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        safe_filename = _safe_multipart_filename(filename)
        prefix.extend(f"--{boundary}\r\n".encode())
        prefix.extend(
            (
                f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
        )
        suffix = f"--{boundary}--\r\n".encode()
        file_size = audio_path.stat().st_size
        body_length = len(prefix) + file_size + len(suffix)

        def body() -> Iterable[bytes]:
            yield bytes(prefix)
            with audio_path.open("rb") as audio_file:
                while chunk := audio_file.read(_FILE_CHUNK_BYTES):
                    yield chunk
            yield suffix

        headers = {
            **self._headers,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(body_length),
        }
        started = time.perf_counter()
        payload = self._request(
            "POST",
            "/v1/audio/transcriptions",
            body=body(),
            headers=headers,
            timeout=self._config.request_timeout,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise LocalAIRequestError("LocalAI 转写响应缺少 text 字符串。")
        language_result = payload.get("language")
        if language_result is not None and not isinstance(language_result, str):
            raise LocalAIRequestError("LocalAI 转写响应的 language 类型无效。")
        return payload, elapsed

    def run_worker(
        self,
        *,
        operation: str,
        payload: dict[str, object],
    ) -> tuple[dict[str, object], float]:
        """经 LocalAI completion RPC 调用版本化的一次性 Unitale worker。"""
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

    def generate_sound(
        self,
        *,
        text: str,
        duration_seconds: float,
        bpm: int | None,
        keyscale: str | None,
        timesignature: str | None,
        lyrics: str | None = None,
        think: bool | None = None,
    ) -> tuple[bytes, float]:
        """通过 LocalAI SoundGeneration RPC 生成一段音乐 WAV。"""
        request_body: dict[str, object] = {
            "model_id": self._config.model,
            "text": text,
            "duration_seconds": duration_seconds,
        }
        if lyrics is not None:
            request_body["lyrics"] = lyrics
        if think is not None:
            request_body["think"] = think
        if bpm is not None:
            request_body["bpm"] = bpm
        if keyscale:
            request_body["keyscale"] = keyscale
        if timesignature:
            request_body["timesignature"] = timesignature

        started = time.perf_counter()
        response = self._request(
            "POST",
            "/v1/sound-generation",
            body=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={**self._headers, "Content-Type": "application/json"},
            timeout=self._config.request_timeout,
            response_limit=_MAX_AUDIO_BYTES,
            expect_json=False,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(response, tuple) or len(response) != 2:
            raise LocalAIRequestError("LocalAI SoundGeneration 响应格式无效。")
        audio_bytes, content_type = response
        if not isinstance(audio_bytes, bytes) or len(audio_bytes) < 12:
            raise LocalAIRequestError("LocalAI SoundGeneration 未返回有效 WAV 音频。")
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise LocalAIRequestError("LocalAI SoundGeneration 响应不是 RIFF/WAVE 音频。")
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            raise LocalAIRequestError("LocalAI SoundGeneration 响应 Content-Type 不是 WAV。")
        return audio_bytes, elapsed

    def transform_audio(
        self,
        *,
        audio_path: Path,
        reference_path: Path,
        params: dict[str, str],
    ) -> tuple[bytes, float]:
        """将源音频和目标参考音频流式发送到 LocalAI AudioTransform。"""
        boundary = f"unitale-{uuid.uuid4().hex}"
        fields = {"model": self._config.model, "response_format": "wav"}
        for key, value in params.items():
            if not key or any(
                not (char.isascii() and (char.isalnum() or char in "_.-")) for char in key
            ):
                raise LocalAIRequestError("LocalAI AudioTransform 参数名无效。")
            if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
                raise LocalAIRequestError("LocalAI AudioTransform 参数值含无效字符。")
            fields[f"params[{key}]"] = value

        prefix = bytearray()
        for name, value in fields.items():
            prefix.extend(f"--{boundary}\r\n".encode())
            prefix.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )

        file_parts: list[tuple[bytes, Path]] = []
        total_audio_bytes = 0
        for field_name, path in (("audio", audio_path), ("reference", reference_path)):
            file_size = path.stat().st_size
            total_audio_bytes += file_size
            if total_audio_bytes > _MAX_AUDIO_BYTES:
                raise LocalAIRequestError("LocalAI AudioTransform 输入超过 512 MiB 上限。")
            filename = _safe_multipart_filename(path.name)
            content_type = _AUDIO_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            _validate_multipart_content_type(content_type)
            file_header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            file_parts.append((file_header, path))

        suffix = f"\r\n--{boundary}--\r\n".encode()
        body_length = (
            len(prefix)
            + len(suffix)
            + total_audio_bytes
            + sum(len(header) for header, _path in file_parts)
            + len(file_parts) * 2
        )

        def body() -> Iterable[bytes]:
            yield bytes(prefix)
            for header, path in file_parts:
                yield header
                with path.open("rb") as audio_file:
                    while chunk := audio_file.read(_FILE_CHUNK_BYTES):
                        yield chunk
                yield b"\r\n"
            yield suffix

        headers = {
            **self._headers,
            "Accept": "audio/wav",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(body_length),
        }
        started = time.perf_counter()
        response = self._request(
            "POST",
            "/v1/audio/transform",
            body=body(),
            headers=headers,
            timeout=self._config.request_timeout,
            response_limit=_MAX_AUDIO_BYTES,
            expect_json=False,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(response, tuple) or len(response) != 2:
            raise LocalAIRequestError("LocalAI AudioTransform 响应格式无效。")
        audio_bytes, content_type = response
        if not isinstance(audio_bytes, bytes) or len(audio_bytes) < 12:
            raise LocalAIRequestError("LocalAI AudioTransform 未返回有效 WAV 音频。")
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise LocalAIRequestError("LocalAI AudioTransform 响应不是 RIFF/WAVE 音频。")
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            raise LocalAIRequestError("LocalAI AudioTransform 响应 Content-Type 不是 WAV。")
        return audio_bytes, elapsed

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

    def design_moss_voice(
        self,
        *,
        text: str,
        instruction: str,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int,
        text_chunk_size: int,
    ) -> tuple[bytes, float]:
        """调用固定 audio.cpp MOSS-VoiceGenerator 设计参数生成单段 WAV。"""
        params = {
            "temperature": str(audio_temperature),
            "top_p": str(audio_top_p),
            "top_k": str(audio_top_k),
            "repetition_penalty": str(audio_repetition_penalty),
            "seed": str(seed),
            "text_chunk_size": str(text_chunk_size),
        }
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "instructions": instruction,
            "response_format": "wav",
            "params": params,
        }
        return self._post_speech(request_body)

    def synthesize_fireredtts3(
        self,
        *,
        text: str,
        language: str,
        n_timesteps: int,
        inference_cfg: float,
        stop_threshold: float,
        seed: int,
        template_name: str | None,
        voice_path: Path | None,
        reference_text: str | None,
        instruction: str | None,
    ) -> tuple[bytes, float]:
        """调用 FireRedTTS3 Base clone 或 Instruct voice-design 请求。"""
        params = {
            "num_inference_steps": str(n_timesteps),
            "guidance_scale": str(inference_cfg),
            "stop_threshold": str(stop_threshold),
            "seed": str(seed),
        }
        if template_name:
            params["template_name"] = template_name
        if reference_text:
            params["reference_text"] = reference_text
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "language": language,
            "response_format": "wav",
            "params": params,
        }
        if voice_path is not None:
            request_body["voice"] = str(voice_path)
        if instruction:
            request_body["instructions"] = instruction
        return self._post_speech(request_body)

    def synthesize_dots_speech(
        self,
        *,
        text: str,
        voice_path: Path,
        language: str | None,
        reference_text: str | None,
        max_tokens: int,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        sampler_mode: str,
        speaker_scale: float,
        template_name: str | None,
        text_chunk_size: int,
        sample_rate: int = 48_000,
    ) -> tuple[bytes, float]:
        """调用固定 audio.cpp DotTTS 请求选项生成克隆语音。"""
        params = {
            "max_tokens": str(max_tokens),
            "seed": str(seed),
            "guidance_scale": str(guidance_scale),
            "num_inference_steps": str(num_inference_steps),
            "sampler_mode": sampler_mode,
            "speaker_scale": str(speaker_scale),
            "text_chunk_size": str(text_chunk_size),
        }
        if reference_text:
            params["reference_text"] = reference_text
        if template_name:
            params["template_name"] = template_name
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "voice": str(voice_path),
            "response_format": "wav",
            "sample_rate": sample_rate,
            "params": params,
        }
        if language:
            request_body["language"] = language
        return self._post_speech(request_body)

    def synthesize_voxcpm2_speech(
        self,
        *,
        text: str,
        voice_path: Path,
        reference_text: str | None,
        guidance_scale: float,
        num_inference_steps: int,
        text_chunk_size: int,
    ) -> tuple[bytes, float]:
        """调用固定 audio.cpp VoxCPM2 请求选项生成参考音色语音。"""
        params = {
            "guidance_scale": str(guidance_scale),
            "num_inference_steps": str(num_inference_steps),
            "text_chunk_size": str(text_chunk_size),
        }
        if reference_text:
            params["reference_text"] = reference_text
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "voice": str(voice_path),
            "response_format": "wav",
            "params": params,
        }
        return self._post_speech(request_body)

    def synthesize_confucius4_speech(
        self,
        *,
        text: str,
        voice_path: Path,
        language: str,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        repetition_penalty: float,
        max_tokens: int,
        num_inference_steps: int,
        guidance_scale: float,
        text_chunk_size: int,
        cross_fade_duration_sec: float,
        edge_fade_duration_sec: float,
        edge_pad_duration_sec: float,
    ) -> tuple[bytes, float]:
        """调用固定 audio.cpp Confucius4-TTS 克隆参数生成 WAV。"""
        params = {
            "temperature": str(temperature),
            "top_p": str(top_p),
            "top_k": str(top_k),
            "num_beams": str(num_beams),
            "repetition_penalty": str(repetition_penalty),
            "max_tokens": str(max_tokens),
            "num_inference_steps": str(num_inference_steps),
            "guidance_scale": str(guidance_scale),
            "text_chunk_size": str(text_chunk_size),
            "cross_fade_duration_sec": str(cross_fade_duration_sec),
            "edge_fade_duration_sec": str(edge_fade_duration_sec),
            "edge_pad_duration_sec": str(edge_pad_duration_sec),
        }
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": text,
            "voice": str(voice_path),
            "language": language,
            "response_format": "wav",
            "params": params,
        }
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
