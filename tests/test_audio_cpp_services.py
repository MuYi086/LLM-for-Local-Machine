"""验证共享 audio.cpp 服务合同，不加载真实模型。"""

from __future__ import annotations

import io
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-audio/audio_cpp_runtime/src"))

from audio_cpp_runtime.service import MODEL_SPECS, create_app  # noqa: E402

from inference_gateway import GatewayConfig  # noqa: E402


def _wav_bytes() -> bytes:
    """构造一个极短 PCM WAV 响应。"""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


def _gateway_config(model: str) -> GatewayConfig:
    """创建测试用 LocalAI 配置并固定共享锁路径。"""
    return GatewayConfig(
        provider="localai",
        localai_enabled=True,
        base_url="http://127.0.0.1:8400",
        model=model,
        api_key=None,
        connect_timeout=1,
        request_timeout=5,
        cleanup_timeout=1,
        gpu_lock_file="/tmp/test-gpu-runtime.lock",
        gpu_lock_wait_timeout=5,
        cuda_release_delay=0,
    )


class FakeLocalAIClient:
    """记录共享服务发出的任务请求。"""

    def __init__(self) -> None:
        self.speech_request: dict[str, object] | None = None
        self.transcription: tuple[str, bytes, str | None] | None = None
        self.transformation: tuple[dict[str, str], dict[str, tuple[str, bytes, str]]] | None = None

    def synthesize_speech(self, request: dict[str, object]) -> bytes:
        self.speech_request = request
        voice_path = request.get("voice")
        if isinstance(voice_path, str):
            assert Path(voice_path).read_bytes() == _wav_bytes()
        return _wav_bytes()

    def transcribe(
        self,
        *,
        filename: str,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, str]:
        self.transcription = filename, audio_path.read_bytes(), language
        return {"text": "测试转写"}

    def transform_audio(
        self,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, Path, str]],
    ) -> bytes:
        self.transformation = (
            fields,
            {
                name: (filename, path.read_bytes(), content_type)
                for name, (filename, path, content_type) in files.items()
            },
        )
        return _wav_bytes()

    def generate_sound(self, request: dict[str, object]) -> bytes:
        self.speech_request = request
        return _wav_bytes()


class MockLocalAIHealthClient:
    """记录不加载模型的 LocalAI readiness 查询。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def health(self) -> bool:
        self.calls.append("health")
        return True

    def model_ready(self) -> bool:
        self.calls.append("model_ready")
        return True


class AudioCppServiceTests(unittest.TestCase):
    def test_registry_declares_all_eight_fixed_audio_cpp_models(self) -> None:
        self.assertEqual(
            set(MODEL_SPECS),
            {
                "confucius4_tts",
                "index_tts25",
                "midashenglm_gen",
                "qwen3_asr",
                "qwen3_tts_base",
                "seed_vc",
                "stable_audio_sfx",
                "voxcpm2",
            },
        )
        self.assertEqual(len({spec.route for spec in MODEL_SPECS.values()}), 8)
        self.assertEqual(len({spec.port for spec in MODEL_SPECS.values()}), 8)

    def test_tts_reference_audio_runs_under_the_configured_shared_gpu_lock(self) -> None:
        spec = MODEL_SPECS["confucius4_tts"]
        fake_client = FakeLocalAIClient()

        def execute(*, config: GatewayConfig, operation: object) -> bytes:
            self.assertEqual(config.gpu_lock_file, "/tmp/test-gpu-runtime.lock")
            return operation(fake_client)  # type: ignore[operator]

        with patch("audio_cpp_runtime.service.execute_localai_operation", side_effect=execute):
            response = TestClient(
                create_app("confucius4_tts", _gateway_config(spec.model_id))
            ).post(
                spec.route,
                data={"text": "你好。", "language": "zh", "reference_text": "参考音频。"},
                files={
                    "reference_audio": (
                        "reference.wav",
                        _wav_bytes(),
                        "audio/wav",
                    )
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual(fake_client.speech_request["model"], spec.model_id)
        self.assertEqual(fake_client.speech_request["params"]["ref_text"], "参考音频。")

    def test_index_and_qwen_base_routes_forward_reference_audio(self) -> None:
        for model_key in ("index_tts25", "qwen3_tts_base"):
            with self.subTest(model=model_key):
                spec = MODEL_SPECS[model_key]
                fake_client = FakeLocalAIClient()

                def execute(
                    *,
                    config: GatewayConfig,
                    operation: object,
                    fake_client: FakeLocalAIClient = fake_client,
                ) -> bytes:
                    self.assertEqual(config.gpu_lock_file, "/tmp/test-gpu-runtime.lock")
                    return operation(fake_client)  # type: ignore[operator]

                with patch(
                    "audio_cpp_runtime.service.execute_localai_operation",
                    side_effect=execute,
                ):
                    fields = {"text": "今天阳光很好。", "language": "Chinese"}
                    if model_key == "qwen3_tts_base":
                        fields["language"] = "zh"
                        fields["reference_text"] = "你好，欢迎使用 Qwen3-TTS 音色设计。"
                    response = TestClient(
                        create_app(model_key, _gateway_config(spec.model_id))
                    ).post(
                        spec.route,
                        data=fields,
                        files={
                            "reference_audio": (
                                "reference.wav",
                                _wav_bytes(),
                                "audio/wav",
                            )
                        },
                    )

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(fake_client.speech_request["model"], spec.model_id)
                self.assertEqual(Path(fake_client.speech_request["voice"]).suffix, ".wav")
                if model_key == "qwen3_tts_base":
                    self.assertEqual(
                        fake_client.speech_request["params"]["reference_text"],
                        "你好，欢迎使用 Qwen3-TTS 音色设计。",
                    )
                    self.assertEqual(fake_client.speech_request["language"], "Chinese")

    def test_qwen3_tts_base_requires_reference_text(self) -> None:
        spec = MODEL_SPECS["qwen3_tts_base"]
        with patch("audio_cpp_runtime.service.execute_localai_operation") as execute:
            response = TestClient(
                create_app("qwen3_tts_base", _gateway_config(spec.model_id))
            ).post(
                spec.route,
                data={"text": "今天阳光很好。", "language": "Chinese"},
                files={
                    "reference_audio": (
                        "reference.wav",
                        _wav_bytes(),
                        "audio/wav",
                    )
                },
            )

        self.assertEqual(response.status_code, 422, response.text)
        execute.assert_not_called()

    def test_asr_upload_uses_the_shared_lock_and_returns_transcript_json(self) -> None:
        spec = MODEL_SPECS["qwen3_asr"]
        fake_client = FakeLocalAIClient()

        def execute(*, config: GatewayConfig, operation: object) -> object:
            self.assertEqual(config.model, spec.model_id)
            return operation(fake_client)  # type: ignore[operator]

        with patch("audio_cpp_runtime.service.execute_localai_operation", side_effect=execute):
            response = TestClient(create_app("qwen3_asr", _gateway_config(spec.model_id))).post(
                spec.route,
                data={"language": "Chinese"},
                files={"audio": ("speech.wav", _wav_bytes(), "audio/wav")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"text": "测试转写"})
        self.assertEqual(fake_client.transcription, ("speech.wav", _wav_bytes(), "Chinese"))

    def test_seed_vc_requires_source_and_reference_and_returns_wav(self) -> None:
        spec = MODEL_SPECS["seed_vc"]
        fake_client = FakeLocalAIClient()

        def execute(*, config: GatewayConfig, operation: object) -> bytes:
            return operation(fake_client)  # type: ignore[operator]

        with patch("audio_cpp_runtime.service.execute_localai_operation", side_effect=execute):
            response = TestClient(create_app("seed_vc", _gateway_config(spec.model_id))).post(
                spec.route,
                files={
                    "audio": ("source.wav", _wav_bytes(), "audio/wav"),
                    "reference": ("target.wav", _wav_bytes(), "audio/wav"),
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual(fake_client.transformation[0]["model"], spec.model_id)
        self.assertEqual(fake_client.transformation[1]["audio"][1], _wav_bytes())
        self.assertEqual(fake_client.transformation[1]["reference"][1], _wav_bytes())

    def test_both_generation_models_forward_prompt_and_duration_under_gpu_lock(self) -> None:
        for model_key in ("midashenglm_gen", "stable_audio_sfx"):
            with self.subTest(model=model_key):
                spec = MODEL_SPECS[model_key]
                fake_client = FakeLocalAIClient()

                def execute(
                    *,
                    config: GatewayConfig,
                    operation: object,
                    fake_client: FakeLocalAIClient = fake_client,
                ) -> bytes:
                    self.assertEqual(config.gpu_lock_file, "/tmp/test-gpu-runtime.lock")
                    return operation(fake_client)  # type: ignore[operator]

                with patch(
                    "audio_cpp_runtime.service.execute_localai_operation",
                    side_effect=execute,
                ):
                    response = TestClient(
                        create_app(model_key, _gateway_config(spec.model_id))
                    ).post(
                        spec.route,
                        json={
                            "text": "<|caption|>雨滴落在树叶上。<|sfx|>轻柔的雨声。",
                            "duration_seconds": 3,
                        },
                    )

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(fake_client.speech_request["model_id"], spec.model_id)
                if model_key == "midashenglm_gen":
                    self.assertEqual(fake_client.speech_request["params"], {"duration_sec": "3.0"})
                    self.assertNotIn("duration_seconds", fake_client.speech_request)
                else:
                    self.assertEqual(fake_client.speech_request["duration_seconds"], 3)

    def test_voxcpm2_accepts_voice_design_text_without_reference_audio(self) -> None:
        spec = MODEL_SPECS["voxcpm2"]
        fake_client = FakeLocalAIClient()

        def execute(*, config: GatewayConfig, operation: object) -> bytes:
            return operation(fake_client)  # type: ignore[operator]

        with patch("audio_cpp_runtime.service.execute_localai_operation", side_effect=execute):
            response = TestClient(create_app("voxcpm2", _gateway_config(spec.model_id))).post(
                spec.route,
                data={"text": "你好。", "instruction": "温柔、清晰的成年女声。"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            fake_client.speech_request["input"],
            "(温柔、清晰的成年女声。) 你好。",
        )
        self.assertNotIn("voice", fake_client.speech_request)

    def test_health_checks_registered_candidate_without_gpu_inference(self) -> None:
        spec = MODEL_SPECS["qwen3_asr"]
        fake_client = MockLocalAIHealthClient()
        with patch("audio_cpp_runtime.service.LocalAIClient", return_value=fake_client):
            response = TestClient(create_app("qwen3_asr", _gateway_config(spec.model_id))).get(
                "/v1/health"
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"], spec.model_id)
        self.assertEqual(fake_client.calls, ["health", "model_ready"])


if __name__ == "__main__":
    unittest.main()
