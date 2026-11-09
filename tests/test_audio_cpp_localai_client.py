"""验证共享 LocalAI audio.cpp 客户端的请求合同与有界流式上传。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from inference_gateway import GatewayConfig, LocalAIClient, LocalAIRequestError


def _gateway_config() -> GatewayConfig:
    """创建指向测试 loopback 地址的 LocalAI 配置。"""
    return GatewayConfig(
        provider="localai",
        localai_enabled=True,
        base_url="http://127.0.0.1:8400",
        model="llm-local.test.audio-cpp",
        api_key=None,
        connect_timeout=1,
        request_timeout=5,
        cleanup_timeout=1,
        gpu_lock_file="/tmp/test-gpu-runtime.lock",
        gpu_lock_wait_timeout=5,
        cuda_release_delay=0,
    )


class AudioCppLocalAIClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LocalAIClient(_gateway_config())

    def test_speech_request_uses_openai_audio_speech_contract(self) -> None:
        wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 16)
        request = Mock(return_value=(wav, "audio/wav"))
        self.client._request = request  # type: ignore[method-assign]
        payload = {"model": _gateway_config().model, "input": "你好", "response_format": "wav"}

        self.assertEqual(self.client.synthesize_speech(payload), wav)

        self.assertEqual(request.call_args.args, ("POST", "/v1/audio/speech"))
        self.assertEqual(json.loads(request.call_args.kwargs["body"]), payload)

    def test_transcription_streams_audio_and_maps_language_prompt(self) -> None:
        requests: list[dict[str, object]] = []

        def capture_request(*args: object, **kwargs: object) -> dict[str, str]:
            kwargs["body"] = b"".join(kwargs["body"])
            requests.append({"args": args, **kwargs})
            return {"text": "转写"}

        request = Mock(side_effect=capture_request)
        self.client._request = request  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "spoken.wav"
            audio_path.write_bytes(b"input-audio")

            result = self.client.transcribe(
                filename="spoken.wav",
                audio_path=audio_path,
                language="Chinese",
                prompt="人名提示",
            )

        self.assertEqual(result, {"text": "转写"})
        self.assertEqual(requests[0]["args"], ("POST", "/v1/audio/transcriptions"))
        body = requests[0]["body"]
        self.assertIn(b'name="model"', body)
        self.assertIn(b'name="language"', body)
        self.assertIn(b"Chinese", body)
        self.assertIn(b'name="prompt"', body)
        self.assertIn(b"input-audio", body)
        self.assertEqual(int(requests[0]["headers"]["Content-Length"]), len(body))

    def test_sound_generation_uses_localai_sound_generation_payload(self) -> None:
        wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 16)
        request = Mock(return_value=(wav, "audio/wav"))
        self.client._request = request  # type: ignore[method-assign]
        payload = {
            "model_id": _gateway_config().model,
            "text": "footsteps on gravel",
            "params": {"duration_sec": "4"},
        }

        self.assertEqual(self.client.generate_sound(payload), wav)

        self.assertEqual(request.call_args.args, ("POST", "/v1/sound-generation"))
        self.assertEqual(json.loads(request.call_args.kwargs["body"]), payload)
        self.assertNotIn("src", json.loads(request.call_args.kwargs["body"]))

    def test_transform_streams_both_audio_files_as_multipart(self) -> None:
        wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 16)
        requests: list[dict[str, object]] = []

        def capture_request(*args: object, **kwargs: object) -> tuple[bytes, str]:
            kwargs["body"] = b"".join(kwargs["body"])
            requests.append({"args": args, **kwargs})
            return wav, "audio/wav"

        request = Mock(side_effect=capture_request)
        self.client._request = request  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            reference = Path(temporary) / "reference.wav"
            source.write_bytes(b"source-audio")
            reference.write_bytes(b"reference-audio")

            result = self.client.transform_audio(
                fields={"model": _gateway_config().model, "response_format": "wav"},
                files={
                    "audio": (source.name, source, "audio/wav"),
                    "reference": (reference.name, reference, "audio/wav"),
                },
            )

        self.assertEqual(result, wav)
        self.assertEqual(requests[0]["args"], ("POST", "/audio/transform"))
        body = requests[0]["body"]
        self.assertIn(b'name="audio"', body)
        self.assertIn(b'name="reference"', body)
        self.assertIn(b"source-audio", body)
        self.assertIn(b"reference-audio", body)
        self.assertEqual(int(requests[0]["headers"]["Content-Length"]), len(body))

    def test_audio_endpoints_reject_non_wav_responses(self) -> None:
        request = Mock(return_value=(b"not-a-wav", "application/octet-stream"))
        self.client._request = request  # type: ignore[method-assign]

        with self.assertRaises(LocalAIRequestError):
            self.client.synthesize_speech({"model": _gateway_config().model, "input": "hello"})

        with self.assertRaises(LocalAIRequestError):
            self.client.generate_sound({"model_id": _gateway_config().model, "text": "rain"})


if __name__ == "__main__":
    unittest.main()
