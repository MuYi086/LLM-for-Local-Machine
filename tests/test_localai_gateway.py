"""LocalAI 配置、HTTP 传输和后端生命周期的无模型测试。"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
sys.path.insert(0, str(REPOSITORY_DIR / "unitale_runtime/src"))

from inference_gateway.execution import (  # noqa: E402
    execute_localai_operation,
    execute_localai_transcription,
    execute_localai_worker_file,
    execute_localai_worker_file_reference,
)
from inference_gateway.worker_protocol import (  # noqa: E402
    decode_worker_request,
    encode_worker_response,
)

from inference_gateway import (  # noqa: E402
    BackendQuarantinedError,
    GatewayConfigurationError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    load_gateway_config,
)
from unitale_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_runtime_lock  # noqa: E402


def config(**overrides: str):
    """生成隔离测试使用的 LocalAI 配置。"""
    environment = {
        "UNITALE_LOCALAI_ENABLED": "true",
        "UNITALE_QWEN3_ASR_PROVIDER": "localai",
        "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
        "GPU_LOCK_FILE": "/tmp/unitale-localai-test.lock",
        "CUDA_RELEASE_DELAY": "0",
        **overrides,
    }
    return load_gateway_config(
        service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
        default_model="unitale.qwen3-asr.1p7b",
        request_timeout=10,
        environ=environment,
    )


class LocalAIGatewayTests(unittest.TestCase):
    """验证网关拒绝不安全配置并且总在推理后尝试停止 backend。"""

    def test_worker_client_uses_versioned_completion_envelope(self) -> None:
        client = LocalAIClient(config(UNITALE_QWEN3_ASR_MODEL="unitale.bridge.smoke"))
        captured: dict[str, object] = {}

        def capture_request(method: str, path: str, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            captured["method"] = method
            captured["path"] = path
            request = json.loads(kwargs["body"])
            decoded = decode_worker_request(
                request["prompt"], model_identity="unitale.bridge.smoke"
            )
            captured["decoded"] = decoded
            result = encode_worker_response(
                model=decoded.model,
                operation=decoded.operation,
                request_id=decoded.request_id,
                result={"kind": "json", "value": {"text": "ok"}},
            ).decode("utf-8")
            return {"choices": [{"text": result}]}

        with patch.object(client, "_request", side_effect=capture_request):
            result, elapsed = client.run_worker(
                operation="run",
                payload={"worker_input": {"text": "hello"}},
            )

        request = json.loads(captured["body"])
        decoded = captured["decoded"]
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/v1/completions")
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(decoded.payload["worker_input"]["text"], "hello")
        self.assertEqual(result["kind"], "json")
        self.assertEqual(result["value"]["text"], "ok")
        self.assertGreaterEqual(elapsed, 0)

    def test_shared_file_reference_is_copied_from_scoped_id_and_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "shared"
            artifact_root.mkdir()
            artifact_id = "a" * 32
            shared_file = artifact_root / f"{artifact_id}.zip"
            shared_file.write_bytes(b"large archive payload")
            with patch(
                "inference_gateway.execution.execute_localai_worker",
                return_value=(
                    {
                        "kind": "file_ref",
                        "artifact_id": artifact_id,
                        "suffix": ".zip",
                        "media_type": "application/zip",
                        "size_bytes": len(b"large archive payload"),
                    },
                    1.0,
                ),
            ):
                copied, elapsed = execute_localai_worker_file_reference(
                    config=config(),
                    operation="run",
                    payload={"worker_input": {}},
                    output_dir=root / "service-output",
                    artifact_dir=artifact_root,
                    suffix=".zip",
                    media_type="application/zip",
                )

            self.assertEqual(copied.read_bytes(), b"large archive payload")
            self.assertGreaterEqual(elapsed, 0)
            self.assertFalse(shared_file.exists())
            copied.unlink()

    def test_shared_file_reference_rejects_invalid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "inference_gateway.execution.execute_localai_worker",
                return_value=(
                    {
                        "kind": "file_ref",
                        "artifact_id": "../outside",
                        "suffix": ".zip",
                        "media_type": "application/zip",
                        "size_bytes": 1,
                    },
                    0.1,
                ),
            ):
                with self.assertRaisesRegex(LocalAIRequestError, "引用格式"):
                    execute_localai_worker_file_reference(
                        config=config(),
                        operation="run",
                        payload={"worker_input": {}},
                        output_dir=Path(temporary) / "out",
                        artifact_dir=Path(temporary) / "shared",
                        suffix=".zip",
                        media_type="application/zip",
                    )

    def test_worker_file_helper_validates_and_stages_artifact(self) -> None:
        class FakeLocalAIClient:
            def run_worker(self, **kwargs: object) -> tuple[dict[str, object], float]:
                self.payload = kwargs["payload"]
                content = b"test artifact"
                return (
                    {
                        "kind": "file",
                        "media_type": "application/zip",
                        "data_base64": base64.b64encode(content).decode("ascii"),
                        "size_bytes": len(content),
                    },
                    0.25,
                )

            def shutdown_model(self) -> None:
                self.shutdown = True

        client = FakeLocalAIClient()
        with tempfile.TemporaryDirectory() as temporary:
            output_path, elapsed = execute_localai_worker_file(
                config=config(GPU_LOCK_FILE=f"{temporary}/gpu.lock"),
                operation="infer",
                payload={"worker_input": {"text": "hello"}},
                output_dir=Path(temporary),
                suffix=".zip",
                media_type="application/zip",
                client=client,
            )
            self.assertEqual(output_path.read_bytes(), b"test artifact")
            self.assertEqual(elapsed, 0.25)
            self.assertEqual(client.payload["media_type"], "application/zip")
            self.assertTrue(client.shutdown)

    def test_sound_generation_client_maps_ace_step_bgm_fields(self) -> None:
        client = LocalAIClient(config(UNITALE_QWEN3_ASR_MODEL="unitale.ace-step15.xl-turbo.q8dit"))
        captured: dict[str, object] = {}
        wav = b"RIFF" + b"\0\0\0\0WAVE" + b"test"

        def capture_request(method: str, path: str, **kwargs: object) -> tuple[bytes, str]:
            captured["method"] = method
            captured["path"] = path
            captured["body"] = kwargs["body"]
            return wav, "audio/wav"

        with patch.object(client, "_request", side_effect=capture_request):
            audio_bytes, elapsed = client.generate_sound(
                text="warm instrumental lo-fi",
                duration_seconds=30,
                bpm=82,
                keyscale="D minor",
                timesignature="4",
                lyrics="",
            )

        self.assertEqual(audio_bytes, wav)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/v1/sound-generation")
        self.assertEqual(
            json.loads(captured["body"]),
            {
                "model_id": "unitale.ace-step15.xl-turbo.q8dit",
                "text": "warm instrumental lo-fi",
                "duration_seconds": 30,
                "bpm": 82,
                "keyscale": "D minor",
                "timesignature": "4",
                "lyrics": "",
            },
        )

    def test_audio_transform_streams_source_and_reference_with_backend_params(self) -> None:
        client = LocalAIClient(config(UNITALE_QWEN3_ASR_MODEL="unitale.seed-vc.vc.f16"))
        captured: dict[str, object] = {}
        wav = b"RIFF" + b"\0\0\0\0WAVE" + b"test"

        def capture_request(method: str, path: str, **kwargs: object) -> tuple[bytes, str]:
            captured["method"] = method
            captured["path"] = path
            captured["headers"] = kwargs["headers"]
            captured["body"] = b"".join(kwargs["body"])
            return wav, "audio/wav"

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            reference = Path(temporary) / "target voice.wav"
            source.write_bytes(b"source bytes")
            reference.write_bytes(b"reference bytes")
            with patch.object(client, "_request", side_effect=capture_request):
                audio_bytes, elapsed = client.transform_audio(
                    audio_path=source,
                    reference_path=reference,
                    params={
                        "route": "v2_vc",
                        "num_inference_steps": "30",
                        "similarity_guidance_scale": "0.7",
                    },
                )

        self.assertEqual(audio_bytes, wav)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/v1/audio/transform")
        headers = captured["headers"]
        body = captured["body"]
        self.assertEqual(headers["Accept"], "audio/wav")
        self.assertEqual(int(headers["Content-Length"]), len(body))
        self.assertIn(b'name="audio"; filename="source.wav"', body)
        self.assertIn(b'name="reference"; filename="target_voice.wav"', body)
        self.assertIn(b'name="params[route]"\r\n\r\nv2_vc', body)
        self.assertIn(b'name="params[num_inference_steps]"\r\n\r\n30', body)
        self.assertIn(b'name="params[similarity_guidance_scale]"\r\n\r\n0.7', body)

    def test_audio_transform_rejects_invalid_parameter_header_values(self) -> None:
        client = LocalAIClient(config())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.wav"
            reference = Path(temporary) / "reference.wav"
            source.write_bytes(b"source")
            reference.write_bytes(b"reference")
            with self.assertRaises(LocalAIRequestError):
                client.transform_audio(
                    audio_path=source,
                    reference_path=reference,
                    params={"route\r\nBad": "v2_vc"},
                )
            with self.assertRaises(LocalAIRequestError):
                client.transform_audio(
                    audio_path=source,
                    reference_path=reference,
                    params={"route": "v2_vc\r\nBad"},
                )

    def test_native_provider_is_default_and_external_url_is_rejected(self) -> None:
        native = load_gateway_config(
            service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
            default_model="unitale.qwen3-asr.1p7b",
            request_timeout=10,
            environ={},
        )
        self.assertEqual(native.provider, "native")
        with self.assertRaises(GatewayConfigurationError):
            config(UNITALE_LOCALAI_BASE_URL="http://192.168.1.10:8400")
        with self.assertRaises(GatewayConfigurationError):
            load_gateway_config(
                service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
                default_model="unitale.qwen3-asr.1p7b",
                request_timeout=10,
                environ={"UNITALE_QWEN3_ASR_PROVIDER": "localai"},
            )

    def test_nonpositive_gpu_wait_is_rejected(self) -> None:
        with self.assertRaises(GatewayConfigurationError):
            config(GPU_LOCK_WAIT_TIMEOUT="0")

    def test_cuda_release_delay_must_be_finite_and_nonnegative(self) -> None:
        for value in ("-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(GatewayConfigurationError):
                config(CUDA_RELEASE_DELAY=value)

    def test_client_streams_file_and_sends_auth_and_language(self) -> None:
        captured: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                captured.append(
                    {
                        "path": self.path,
                        "body": body,
                        "authorization": self.headers.get("Authorization"),
                    }
                )
                if self.path.endswith("transcriptions"):
                    encoded = json.dumps({"text": "recognized", "language": "English"}).encode()
                    content_type = "application/json"
                else:
                    encoded = b"backend stopped"
                    content_type = "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "speech.wav"
            audio_path.write_bytes(b"RIFF\x00audio-bytes")
            client_config = load_gateway_config(
                service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
                default_model="unitale.qwen3-asr.1p7b",
                request_timeout=3,
                environ={
                    "UNITALE_LOCALAI_ENABLED": "true",
                    "UNITALE_QWEN3_ASR_PROVIDER": "localai",
                    "UNITALE_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    "UNITALE_LOCALAI_API_KEY": "test-secret",
                    "GPU_LOCK_FILE": f"{temporary}/gpu.lock",
                },
            )
            client = LocalAIClient(client_config)
            result, elapsed = client.transcribe(
                audio_path=audio_path,
                filename="speech.wav",
                content_type="audio/wav",
                language="English",
            )
            client.shutdown_model()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

        self.assertEqual(result["text"], "recognized")
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(captured[0]["authorization"], "Bearer test-secret")
        self.assertIn(b'name="language"', captured[0]["body"])
        self.assertIn(b"audio-bytes", captured[0]["body"])
        self.assertEqual(captured[1]["path"], "/backend/shutdown")

    def test_client_sanitizes_untrusted_filename_before_building_multipart_headers(self) -> None:
        client = LocalAIClient(config())
        captured: list[bytes] = []

        def capture_request(method: str, path: str, **kwargs: object) -> dict[str, str]:
            body = kwargs["body"]
            assert not isinstance(body, bytes)
            captured.append(b"".join(body))
            return {"text": "recognized", "language": "English"}

        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "speech.wav"
            audio_path.write_bytes(b"audio")
            with patch.object(client, "_request", side_effect=capture_request):
                client.transcribe(
                    audio_path=audio_path,
                    filename="speech.wav\r\nX-Injected: true",
                    content_type="audio/wav",
                    language=None,
                )

        self.assertEqual(len(captured), 1)
        self.assertNotIn(b"\r\nX-Injected:", captured[0])
        self.assertIn(b'filename="speech.wav__X-Injected__true"', captured[0])

    def test_tts_client_maps_qwen3_clone_fields_and_validates_wav_response(self) -> None:
        captured: list[dict[str, object]] = []
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                captured.append({"path": self.path, "body": body})
                if self.path == "/v1/audio/speech":
                    encoded = response_wav
                    content_type = "audio/wav"
                else:
                    encoded = b"stopped"
                    content_type = "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="UNITALE_QWEN3_TTS_PROVIDER",
                default_model="unitale.qwen3-tts.1p7b-base",
                request_timeout=3,
                environ={
                    "UNITALE_LOCALAI_ENABLED": "true",
                    "UNITALE_QWEN3_TTS_PROVIDER": "localai",
                    "UNITALE_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    "GPU_LOCK_FILE": "/tmp/qwen3-tts-test.lock",
                },
            )
            client = LocalAIClient(client_config)
            with tempfile.TemporaryDirectory() as temporary:
                reference = Path(temporary) / "reference.wav"
                reference.write_bytes(b"ref")
                audio_bytes, elapsed = client.synthesize_speech(
                    text="你好。",
                    voice_path=reference,
                    language="Chinese",
                    reference_text="参考文本。",
                    x_vector_only=False,
                    max_new_tokens=512,
                    top_p=0.8,
                    temperature=0.7,
                )
                client.shutdown_model()
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        body = json.loads(captured[0]["body"])
        self.assertEqual(captured[0]["path"], "/v1/audio/speech")
        self.assertEqual(body["model"], "unitale.qwen3-tts.1p7b-base")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["voice"], str(reference))
        self.assertEqual(body["language"], "Chinese")
        self.assertEqual(
            body["params"],
            {
                "max_tokens": "512",
                "x_vector_only_mode": "false",
                "reference_text": "参考文本。",
                "top_p": "0.8",
                "temperature": "0.7",
            },
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(captured[1]["path"], "/backend/shutdown")

    def test_voice_design_client_maps_instruction_to_localai_speech_request(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="UNITALE_QWEN3_VOICEDESIGN_PROVIDER",
            default_model="unitale.qwen3-tts.1p7b-voicedesign",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_QWEN3_VOICEDESIGN_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        with patch.object(client, "_request", return_value=(response_wav, "audio/wav")) as request:
            audio_bytes, elapsed = client.design_voice(
                text="你好。",
                instruction="成年女性，温柔清晰。",
                language="Chinese",
                max_new_tokens=768,
                top_p=0.85,
                temperature=0.75,
            )

        body = json.loads(request.call_args.kwargs["body"])
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "unitale.qwen3-tts.1p7b-voicedesign")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["instructions"], "成年女性，温柔清晰。")
        self.assertEqual(body["language"], "Chinese")
        self.assertEqual(
            body["params"],
            {"max_tokens": "768", "top_p": "0.85", "temperature": "0.75"},
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_dots_tts_client_maps_reference_and_flow_options(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="UNITALE_DOTS_TTS_SOAR_PROVIDER",
            default_model="unitale.dots-tts-soar.q8-0",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_DOTS_TTS_SOAR_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        reference = Path("/tmp/validated-reference.wav")
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        with patch.object(client, "_request", return_value=(response_wav, "audio/wav")) as call:
            audio_bytes, elapsed = client.synthesize_dots_speech(
                text="你好。",
                voice_path=reference,
                language="chinese",
                reference_text="参考文本。",
                max_tokens=500,
                seed=42,
                guidance_scale=1.2,
                num_inference_steps=10,
                sampler_mode="euler",
                speaker_scale=1.5,
                template_name="tts",
                text_chunk_size=4,
            )

        body = json.loads(call.call_args.kwargs["body"])
        self.assertEqual(call.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "unitale.dots-tts-soar.q8-0")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["voice"], str(reference))
        self.assertEqual(body["language"], "chinese")
        self.assertEqual(body["sample_rate"], 48_000)
        self.assertEqual(
            body["params"],
            {
                "max_tokens": "500",
                "seed": "42",
                "guidance_scale": "1.2",
                "num_inference_steps": "10",
                "sampler_mode": "euler",
                "speaker_scale": "1.5",
                "text_chunk_size": "4",
                "reference_text": "参考文本。",
                "template_name": "tts",
            },
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_voxcpm2_client_maps_reference_and_clone_controls(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="UNITALE_VOXCPM2_PROVIDER",
            default_model="unitale.voxcpm2.q8-0",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_VOXCPM2_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        reference = Path("/tmp/validated-voxcpm2-reference.wav")
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        with patch.object(client, "_request", return_value=(response_wav, "audio/wav")) as call:
            audio_bytes, elapsed = client.synthesize_voxcpm2_speech(
                text="(轻快地说)你好。",
                voice_path=reference,
                reference_text="参考转写。",
                guidance_scale=2.5,
                num_inference_steps=12,
                text_chunk_size=20,
            )

        body = json.loads(call.call_args.kwargs["body"])
        self.assertEqual(call.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "unitale.voxcpm2.q8-0")
        self.assertEqual(body["input"], "(轻快地说)你好。")
        self.assertEqual(body["voice"], str(reference))
        self.assertEqual(
            body["params"],
            {
                "guidance_scale": "2.5",
                "num_inference_steps": "12",
                "text_chunk_size": "20",
                "reference_text": "参考转写。",
            },
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_confucius4_client_maps_language_and_generation_options(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="UNITALE_CONFUCIUS4_TTS_PROVIDER",
            default_model="unitale.confucius4-tts.orig",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_CONFUCIUS4_TTS_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        reference = Path("/tmp/validated-confucius4-reference.wav")
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        with patch.object(client, "_request", return_value=(response_wav, "audio/wav")) as call:
            audio_bytes, elapsed = client.synthesize_confucius4_speech(
                text="你好。",
                voice_path=reference,
                language="zh",
                temperature=0.8,
                top_p=0.8,
                top_k=30,
                num_beams=3,
                repetition_penalty=10,
                max_tokens=1520,
                num_inference_steps=25,
                guidance_scale=0.7,
                text_chunk_size=80,
                cross_fade_duration_sec=0.3,
                edge_fade_duration_sec=0.1,
                edge_pad_duration_sec=0.1,
            )

        body = json.loads(call.call_args.kwargs["body"])
        self.assertEqual(call.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "unitale.confucius4-tts.orig")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["voice"], str(reference))
        self.assertEqual(body["language"], "zh")
        self.assertEqual(
            body["params"],
            {
                "temperature": "0.8",
                "top_p": "0.8",
                "top_k": "30",
                "num_beams": "3",
                "repetition_penalty": "10",
                "max_tokens": "1520",
                "num_inference_steps": "25",
                "guidance_scale": "0.7",
                "text_chunk_size": "80",
                "cross_fade_duration_sec": "0.3",
                "edge_fade_duration_sec": "0.1",
                "edge_pad_duration_sec": "0.1",
            },
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_moss_voicegenerator_client_maps_instruction_and_audio_sampling(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="UNITALE_MOSS_VOICEGENERATOR_PROVIDER",
            default_model="unitale.moss-voicegenerator.bf16",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_MOSS_VOICEGENERATOR_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        wav = b"RIFF\x04\x00\x00\x00WAVE"
        with patch.object(client, "_request", return_value=(wav, "audio/wav")) as call:
            audio_bytes, elapsed = client.design_moss_voice(
                text="你好。",
                instruction="成年女性，清晰自然。",
                audio_temperature=1.5,
                audio_top_p=0.6,
                audio_top_k=50,
                audio_repetition_penalty=1.1,
                seed=123,
                text_chunk_size=4,
            )

        body = json.loads(call.call_args.kwargs["body"])
        self.assertEqual(call.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "unitale.moss-voicegenerator.bf16")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["instructions"], "成年女性，清晰自然。")
        self.assertEqual(
            body["params"],
            {
                "temperature": "1.5",
                "top_p": "0.6",
                "top_k": "50",
                "repetition_penalty": "1.1",
                "seed": "123",
                "text_chunk_size": "4",
            },
        )
        self.assertEqual(audio_bytes, wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_fireredtts3_client_maps_base_clone_and_instruct_voice_design(self) -> None:
        wav = b"RIFF\x04\x00\x00\x00WAVE"
        base_config = load_gateway_config(
            service_provider_env="UNITALE_FIRERED_TTS3_CLONE_PROVIDER",
            default_model="unitale.fireredtts3.base.q8-0",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_FIRERED_TTS3_CLONE_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        base_client = LocalAIClient(base_config)
        reference = Path("/tmp/validated-firered-reference.wav")
        with patch.object(base_client, "_request", return_value=(wav, "audio/wav")) as base_call:
            base_client.synthesize_fireredtts3(
                text="目标台词",
                language="Chinese",
                n_timesteps=4,
                inference_cfg=2.0,
                stop_threshold=0.5,
                seed=1234,
                template_name=None,
                voice_path=reference,
                reference_text="准确参考转写。",
                instruction=None,
            )

        base_body = json.loads(base_call.call_args.kwargs["body"])
        self.assertEqual(base_body["model"], "unitale.fireredtts3.base.q8-0")
        self.assertEqual(base_body["voice"], str(reference))
        self.assertEqual(base_body["language"], "Chinese")
        self.assertNotIn("instructions", base_body)
        self.assertEqual(
            base_body["params"],
            {
                "num_inference_steps": "4",
                "guidance_scale": "2.0",
                "stop_threshold": "0.5",
                "seed": "1234",
                "reference_text": "准确参考转写。",
            },
        )

        instruct_config = load_gateway_config(
            service_provider_env="UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER",
            default_model="unitale.fireredtts3.instruct.q8-0",
            request_timeout=10,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        instruct_client = LocalAIClient(instruct_config)
        with patch.object(
            instruct_client, "_request", return_value=(wav, "audio/wav")
        ) as instruct_call:
            instruct_client.synthesize_fireredtts3(
                text="你好。",
                language="Chinese",
                n_timesteps=10,
                inference_cfg=1.2,
                stop_threshold=0.5,
                seed=2,
                template_name="voice_design",
                voice_path=None,
                reference_text=None,
                instruction="年轻女性，温柔清晰。",
            )

        instruct_body = json.loads(instruct_call.call_args.kwargs["body"])
        self.assertEqual(instruct_body["model"], "unitale.fireredtts3.instruct.q8-0")
        self.assertEqual(instruct_body["instructions"], "年轻女性，温柔清晰。")
        self.assertNotIn("voice", instruct_body)
        self.assertEqual(instruct_body["params"]["template_name"], "voice_design")

    def test_client_rejects_control_characters_in_content_type(self) -> None:
        client = LocalAIClient(config())
        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "speech.wav"
            audio_path.write_bytes(b"audio")
            with (
                patch.object(client, "_request") as request,
                self.assertRaises(LocalAIRequestError),
            ):
                client.transcribe(
                    audio_path=audio_path,
                    filename="speech.wav",
                    content_type="audio/wav\r\nX-Injected: true",
                    language=None,
                )
        request.assert_not_called()

    def test_model_ready_checks_exact_registered_model_without_loading(self) -> None:
        registered_models = [{"id": "unitale.qwen3-asr.1p7b"}]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                encoded = json.dumps({"object": "list", "data": registered_models}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
                default_model="unitale.qwen3-asr.1p7b",
                request_timeout=3,
                environ={
                    "UNITALE_LOCALAI_ENABLED": "true",
                    "UNITALE_QWEN3_ASR_PROVIDER": "localai",
                    "UNITALE_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                },
            )
            client = LocalAIClient(client_config)
            self.assertTrue(client.model_ready())
            registered_models[:] = [{"id": "different-model"}]
            self.assertFalse(client.model_ready())
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_response_trickle_cannot_extend_total_request_timeout(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = b"slow-response"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    try:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                    time.sleep(0.04)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="UNITALE_QWEN3_ASR_PROVIDER",
                default_model="unitale.qwen3-asr.1p7b",
                request_timeout=2,
                environ={
                    "UNITALE_LOCALAI_ENABLED": "true",
                    "UNITALE_QWEN3_ASR_PROVIDER": "localai",
                    "UNITALE_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                },
            )
            client = LocalAIClient(client_config)
            started = time.monotonic()
            with self.assertRaises(LocalAIUnavailableError):
                client._request("GET", "/slow", timeout=0.15)
            self.assertLess(time.monotonic() - started, 0.4)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_success_clears_crash_marker_after_backend_shutdown(self) -> None:
        class Client:
            def transcribe(self, **kwargs: object):
                marker = Path(f"{kwargs['audio_path'].parent}/gpu.lock.quarantine")
                assert marker.is_file()
                return {"text": "ok", "language": None}, 0.4

            def shutdown_model(self) -> None:
                return

        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "audio.wav"
            audio_path.write_bytes(b"audio")
            lock_path = str(Path(temporary) / "gpu.lock")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            with patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock):
                payload, elapsed = execute_localai_transcription(
                    config=replace(config(), gpu_lock_file=lock_path),
                    audio_path=audio_path,
                    filename="audio.wav",
                    content_type="audio/wav",
                    language=None,
                    client=Client(),
                )

            self.assertEqual(payload["text"], "ok")
            self.assertEqual(elapsed, 0.4)
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_cuda_release_delay_runs_after_shutdown_while_marker_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")
            marker = Path(f"{lock_path}.quarantine")
            events: list[str] = []

            class Client:
                def shutdown_model(self) -> None:
                    self.assert_marker()
                    events.append("shutdown")

                @staticmethod
                def assert_marker() -> None:
                    assert marker.is_file()

            def fake_sleep(delay: float) -> None:
                self.assertEqual(delay, 0.25)
                self.assertTrue(marker.is_file())
                events.append("release-delay")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            with (
                patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock),
                patch("inference_gateway.execution.time.sleep", side_effect=fake_sleep),
            ):
                result = execute_localai_operation(
                    config=replace(
                        config(),
                        gpu_lock_file=lock_path,
                        cuda_release_delay=0.25,
                    ),
                    operation=lambda localai: "complete",
                    client=Client(),
                )

            self.assertEqual(result, "complete")
            self.assertEqual(events, ["shutdown", "release-delay"])
            self.assertFalse(marker.exists())

    def test_failed_backend_shutdown_preserves_quarantine_marker(self) -> None:
        class Client:
            def transcribe(self, **kwargs: object):
                return {"text": "possibly-running", "language": None}, 0.1

            def shutdown_model(self) -> None:
                raise RuntimeError("backend still running")

        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "audio.wav"
            audio_path.write_bytes(b"audio")
            lock_path = str(Path(temporary) / "gpu.lock")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            with patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock):
                with self.assertRaises(BackendQuarantinedError):
                    execute_localai_transcription(
                        config=replace(config(), gpu_lock_file=lock_path),
                        audio_path=audio_path,
                        filename="audio.wav",
                        content_type="audio/wav",
                        language=None,
                        client=Client(),
                    )

            marker_path = Path(f"{lock_path}.quarantine")
            self.assertTrue(marker_path.is_file())
            with self.assertRaises(GpuQuarantinedError):
                with gpu_runtime_lock(lock_path, "native-after-failure", timeout_seconds=0.1):
                    self.fail("隔离状态下不应获取 GPU 租约")

    def test_native_request_cannot_enter_during_localai_gpu_lease(self) -> None:
        entered_inference = threading.Event()
        finish_inference = threading.Event()
        finished_request = threading.Event()
        errors: list[BaseException] = []

        class Client:
            def transcribe(self, **kwargs: object):
                entered_inference.set()
                if not finish_inference.wait(timeout=2):
                    raise TimeoutError("测试未及时释放 LocalAI 请求。")
                return {"text": "ok", "language": "English"}, 0.1

            def shutdown_model(self) -> None:
                return

        with tempfile.TemporaryDirectory() as temporary:
            audio_path = Path(temporary) / "audio.wav"
            audio_path.write_bytes(b"audio")
            lock_path = str(Path(temporary) / "gpu.lock")

            def localai_request() -> None:
                try:
                    execute_localai_transcription(
                        config=replace(config(), gpu_lock_file=lock_path),
                        audio_path=audio_path,
                        filename="audio.wav",
                        content_type="audio/wav",
                        language=None,
                        client=Client(),
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    finished_request.set()

            thread = threading.Thread(target=localai_request, daemon=True)
            thread.start()
            self.assertTrue(entered_inference.wait(timeout=2))
            try:
                with self.assertRaises(GpuLockTimeoutError):
                    with gpu_runtime_lock(lock_path, "native-request", timeout_seconds=0.05):
                        self.fail("LocalAI 持有 GPU 租约时，native 请求不应进入。")
            finally:
                finish_inference.set()
            self.assertTrue(finished_request.wait(timeout=2))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
