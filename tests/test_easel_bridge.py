"""验证 Easel 兼容网关的鉴权和本地模型协议映射。"""

from __future__ import annotations

import asyncio
import email
import io
import json
import tempfile
import unittest
import wave
from email import policy
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from easel_bridge.app import (
    CHAT_MODEL_ID,
    MOSS_MODEL_ID,
    VOXCPM_MODEL_ID,
    create_app,
)

API_KEY = "bridge-test-secret-00000000000000000000000000000000"
CHAT_URL = "http://ornith.test/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8"
MOSS_URL = "http://moss.test/v1/openmoss/MOSS-Image-SFT-Q8_0"
VOXCPM_URL = "http://voice.test/v1/OpenBMB/VoxCPM2-GGUF"
PNG_B64 = "iVBORw0KGgo="


def _wav_bytes() -> bytes:
    """生成有效的静音 PCM WAV 夹具。"""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * 24)
    return output.getvalue()


def _multipart_fields(request: httpx.Request) -> dict[str, str]:
    """读取 HTTPX 请求中的 multipart 文本字段。"""
    envelope = (
        f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n"
    ).encode() + request.content
    message = email.message_from_bytes(envelope, policy=policy.default)
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode(
            "utf-8"
        )
        for part in message.iter_parts()
    }


class _AsyncBytesStream(httpx.AsyncByteStream):
    """为 MockTransport 提供一个可逐块读取的 SSE 响应。"""

    async def __aiter__(self):
        yield b'data: {"choices":[]}'
        yield b"\n\ndata: [DONE]\n\n"

    async def aclose(self) -> None:
        return None


class EaselBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(
            patch.dict(
                "os.environ", {"MODEL_TELEMETRY_DIR": directory, "MODEL_TELEMETRY_ENABLED": "true"}
            )
        )
        self.requests: list[httpx.Request] = []
        self.wav = _wav_bytes()

        def upstream(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url == httpx.URL(CHAT_URL):
                if json.loads(request.content).get("stream"):
                    return httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=_AsyncBytesStream(),
                    )
                return httpx.Response(
                    200,
                    json={"id": "chatcmpl-test", "choices": [{"message": {"content": "ok"}}]},
                )
            if request.url == httpx.URL(MOSS_URL):
                return httpx.Response(
                    200,
                    json={"created": 1, "data": [{"b64_json": PNG_B64}]},
                )
            if request.url == httpx.URL(VOXCPM_URL):
                return httpx.Response(
                    200,
                    headers={"content-type": "audio/wav"},
                    content=self.wav,
                )
            return httpx.Response(404, json={"detail": "unknown upstream route"})

        self.transport = httpx.MockTransport(upstream)
        self.app = create_app(
            api_key=API_KEY,
            chat_url=CHAT_URL,
            moss_url=MOSS_URL,
            voxcpm_url=VOXCPM_URL,
            transport=self.transport,
        )
        self.client = TestClient(self.app)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {API_KEY}"}

    def test_retired_chat_model_is_rejected_without_upstream_call(self) -> None:
        """旧模型 ID 不得静默映射至 MTP 或继续调用已退役服务。"""
        response = self.client.post(
            "/v1/chat/completions",
            headers=self._headers(),
            json={"model": "Ornith-1.5-9B-GGUF", "messages": []},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.requests, [])

    def test_delayed_stream_completes_without_false_disconnect(self) -> None:
        """真实 ASGI 接收通道在请求结束后仍等待客户端事件。"""

        async def exercise() -> None:
            class DelayedStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    await asyncio.sleep(0.02)
                    yield b"data: [DONE]\n\n"

            app = create_app(
                api_key=API_KEY,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        stream=DelayedStream(),
                    )
                ),
            )
            incoming = asyncio.Queue()
            await incoming.put(
                {
                    "type": "http.request",
                    "body": json.dumps({"model": CHAT_MODEL_ID, "stream": True}).encode(),
                    "more_body": False,
                }
            )
            messages = []

            async def send(message):
                messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "query_string": b"",
                "headers": [
                    (b"authorization", f"Bearer {API_KEY}".encode()),
                    (b"content-type", b"application/json"),
                ],
                "server": ("127.0.0.1", 8500),
                "client": ("127.0.0.1", 12345),
            }
            await asyncio.wait_for(app(scope, incoming.get, send), timeout=2)
            bodies = [m for m in messages if m["type"] == "http.response.body"]
            self.assertEqual(b"".join(m.get("body", b"") for m in bodies), b"data: [DONE]\n\n")
            self.assertFalse(bodies[-1].get("more_body", False))

        asyncio.run(exercise())

    def test_model_list_requires_key_and_lists_the_supported_models(self) -> None:
        denied = self.client.get("/v1/models")
        allowed = self.client.get("/v1/models", headers=self._headers())

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            {model["id"] for model in allowed.json()["data"]},
            {CHAT_MODEL_ID, MOSS_MODEL_ID, VOXCPM_MODEL_ID},
        )

    def test_image_generation_maps_to_moss_route_and_returns_openai_image_shape(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers=self._headers(),
            json={
                "model": MOSS_MODEL_ID,
                "prompt": "a quiet street at night",
                "size": "1024x1024",
                "n": 1,
                "quality": "high",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["b64_json"], PNG_B64)
        request = self.requests[-1]
        self.assertEqual(request.url, httpx.URL(MOSS_URL))
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(
            json.loads(request.content),
            {
                "model": MOSS_MODEL_ID,
                "prompt": "a quiet street at night",
                "size": "1024x1024",
                "n": 1,
                "response_format": "b64_json",
            },
        )

    def test_image_generation_expands_batch_into_serial_single_image_calls(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers=self._headers(),
            json={"model": MOSS_MODEL_ID, "prompt": "cat", "n": 2},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 2)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(json.loads(request.content)["n"] == 1 for request in self.requests))

    def test_image_generation_rejects_batch_size_above_four(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            headers=self._headers(),
            json={"model": MOSS_MODEL_ID, "prompt": "cat", "n": 5},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_chat_completion_forwards_openai_request_and_streams_sse(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": CHAT_MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn(b"data: [DONE]", response.content)
        self.assertEqual(self.requests[-1].url, httpx.URL(CHAT_URL))
        self.assertEqual(json.loads(self.requests[-1].content)["model"], CHAT_MODEL_ID)
        self.assertEqual(
            response.headers["x-model-request-id"], self.requests[-1].headers["x-model-request-id"]
        )
        self.assertNotIn("authorization", self.requests[-1].headers)

    def test_speech_maps_openai_json_to_voxcpm_multipart_and_returns_wav(self) -> None:
        response = self.client.post(
            "/v1/audio/speech",
            headers=self._headers(),
            json={
                "model": VOXCPM_MODEL_ID,
                "input": "午夜十二点，走廊里传来了脚步声。",
                "voice": "alloy",
                "response_format": "wav",
                "instructions": "低沉、克制的旁白。",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, self.wav)
        request = self.requests[-1]
        self.assertEqual(request.url, httpx.URL(VOXCPM_URL))
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(
            _multipart_fields(request),
            {"text": "午夜十二点，走廊里传来了脚步声。", "instruction": "低沉、克制的旁白。"},
        )

    def test_speech_converts_wav_to_mp3_when_requested(self) -> None:
        app = create_app(
            api_key=API_KEY,
            chat_url=CHAT_URL,
            moss_url=MOSS_URL,
            voxcpm_url=VOXCPM_URL,
            transport=self.transport,
            wav_to_mp3=lambda audio: b"ID3-mp3-data" if audio == self.wav else b"",
        )

        response = TestClient(app).post(
            "/v1/audio/speech",
            headers=self._headers(),
            json={"model": VOXCPM_MODEL_ID, "input": "你好", "response_format": "mp3"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.content, b"ID3-mp3-data")

    def test_speech_rejects_unsupported_voice_or_speed(self) -> None:
        for payload in (
            {"model": VOXCPM_MODEL_ID, "input": "你好", "voice": "nova"},
            {"model": VOXCPM_MODEL_ID, "input": "你好", "speed": 1.25},
        ):
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/v1/audio/speech", headers=self._headers(), json=payload
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.requests, [])

    def test_upstream_service_unavailable_status_is_preserved(self) -> None:
        def unavailable(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(503, json={"detail": "GPU 队列繁忙"})

        app = create_app(
            api_key=API_KEY,
            chat_url=CHAT_URL,
            moss_url=MOSS_URL,
            voxcpm_url=VOXCPM_URL,
            transport=httpx.MockTransport(unavailable),
        )

        response = TestClient(app).post(
            "/v1/images/generations",
            headers=self._headers(),
            json={"model": MOSS_MODEL_ID, "prompt": "cat"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "GPU 队列繁忙")

    def test_large_request_body_is_rejected_before_model_dispatch(self) -> None:
        app = create_app(
            api_key=API_KEY,
            chat_url=CHAT_URL,
            moss_url=MOSS_URL,
            voxcpm_url=VOXCPM_URL,
            transport=self.transport,
            max_request_bytes=128,
        )

        response = TestClient(app).post(
            "/v1/images/generations",
            headers=self._headers(),
            json={"model": MOSS_MODEL_ID, "prompt": "x" * 256},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.requests, [])

    def test_application_refuses_to_start_without_a_bearer_key(self) -> None:
        app = create_app(
            api_key="",
            chat_url=CHAT_URL,
            moss_url=MOSS_URL,
            voxcpm_url=VOXCPM_URL,
            transport=self.transport,
        )

        with self.assertRaisesRegex(RuntimeError, "EASEL_BRIDGE_API_KEY"):
            with TestClient(app):
                pass

    def test_application_rejects_a_short_or_template_api_key(self) -> None:
        for key in ("short", "replace-with-a-random-secret"):
            with self.subTest(key=key):
                app = create_app(
                    api_key=key,
                    chat_url=CHAT_URL,
                    moss_url=MOSS_URL,
                    voxcpm_url=VOXCPM_URL,
                    transport=self.transport,
                )
                with self.assertRaisesRegex(RuntimeError, "至少 32 字节"):
                    with TestClient(app):
                        pass


if __name__ == "__main__":
    unittest.main()
