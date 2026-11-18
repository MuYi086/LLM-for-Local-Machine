"""验证共享 LocalAI audio.cpp 客户端的请求合同与有界流式上传。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import wave
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

    def test_voice_design_forwards_seed_to_localai_params(self) -> None:
        wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 16)
        request = Mock(return_value=(wav, "audio/wav"))
        self.client._request = request
        self.client.design_voice(
            text="你好",
            instruction="自然",
            language="Chinese",
            max_new_tokens=2048,
            top_p=None,
            temperature=None,
            seed=42,
        )
        self.assertEqual(json.loads(request.call_args.kwargs["body"])["params"]["seed"], "42")

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

    def test_transcription_can_request_verbose_json_for_forced_alignment(self) -> None:
        requests: list[dict[str, object]] = []

        def capture_request(*args: object, **kwargs: object) -> dict[str, object]:
            kwargs["body"] = b"".join(kwargs["body"])
            requests.append({"args": args, **kwargs})
            return {"text": "hello", "words": [{"text": "hello", "start": 0, "end": 1}]}

        self.client._request = Mock(side_effect=capture_request)  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "spoken.wav"
            audio_path.write_bytes(b"input-audio")
            result = self.client.transcribe(
                filename="spoken.wav",
                audio_path=audio_path,
                language="English",
                prompt="hello",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        self.assertIn("words", result)
        self.assertIn(b'name="response_format"', requests[0]["body"])
        self.assertIn(b"verbose_json", requests[0]["body"])
        self.assertIn(b'name="timestamp_granularities[]"', requests[0]["body"])
        self.assertIn(b"word", requests[0]["body"])
        self.assertIn(b"hello", requests[0]["body"])

    def test_transform_audio_stems_reads_and_saves_all_named_wavs(self) -> None:
        wav_output = io.BytesIO()
        with wave.open(wav_output, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(44_100)
            output.writeframes(b"\x00\x00" * 4)
        wav = wav_output.getvalue()
        requests: list[tuple[str, str, dict[str, object]]] = []

        def capture_request(method: str, path: str, **kwargs: object) -> object:
            requests.append((method, path, kwargs))
            if method == "POST":
                kwargs["body"] = b"".join(kwargs["body"])
                return (
                    wav,
                    "audio/wav",
                    {
                        "X-Audio-Stems": json.dumps(
                            [
                                {"name": "vocals", "url": "/generated-audio/run.vocals.wav"},
                                {
                                    "name": "instrumental",
                                    "url": "/generated-audio/run.instrumental.wav",
                                },
                            ]
                        )
                    },
                )
            return wav, "audio/wav"

        self.client._request = Mock(side_effect=capture_request)  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.write_bytes(wav)
            output_dir = Path(temporary) / "stems"
            output_dir.mkdir()
            stems = self.client.transform_audio_stems(
                model=_gateway_config().model,
                audio_path=source,
                output_dir=output_dir,
            )
            self.assertEqual(set(stems), {"vocals", "instrumental"})
            self.assertEqual(stems["vocals"].read_bytes(), wav)
            self.assertEqual(stems["instrumental"].read_bytes(), wav)
            self.assertEqual(requests[0][0:2], ("POST", "/audio/transform"))
            self.assertIn(b"response_format", requests[0][2]["body"])
            self.assertIn(
                ("GET", "/generated-audio/run.vocals.wav"),
                [item[:2] for item in requests],
            )

    def test_transform_audio_stems_rejects_urls_outside_generated_audio(self) -> None:
        wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 16)
        request = Mock(
            return_value=(
                wav,
                "audio/wav",
                {"X-Audio-Stems": '[{"name":"vocals","url":"/v1/models"}]'},
            )
        )
        self.client._request = request  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "song.wav"
            source.write_bytes(wav)
            output_dir = Path(temporary) / "stems"
            output_dir.mkdir()
            with self.assertRaises(LocalAIRequestError):
                self.client.transform_audio_stems(
                    model=_gateway_config().model,
                    audio_path=source,
                    output_dir=output_dir,
                )

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
