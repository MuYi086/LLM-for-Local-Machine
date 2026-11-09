"""覆盖 Qwen3-TTS VoiceDesign HTTP 合同，不加载模型或访问 LocalAI。"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "model-audio/design_qwen3-tts-1.7b-voicedesign"
sys.path[:0] = [
    str(SERVICE_DIR),
    str(ROOT / "inference_gateway/src"),
    str(ROOT / "llm_local_runtime/src"),
]
os.environ["LLM_LOCALAI_ENABLED"] = "true"
os.environ["LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER"] = "localai"

from main import create_app  # noqa: E402

ROUTE = "/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign"


def make_wav(*, frames: int = 2400, sample_rate: int = 24000) -> bytes:
    """构造小型有效 PCM WAV 测试数据。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


class FakeLocalAIClient:
    """替代实际 LocalAI 网络边界，保留请求字段供断言。"""

    def __init__(self, responses: list[bytes] | None = None) -> None:
        self.responses = list(responses or [make_wav()])
        self.requests: list[dict[str, object]] = []

    def design_voice(self, **kwargs: object) -> tuple[bytes, float]:
        self.requests.append(kwargs)
        return self.responses.pop(0), 0.01


class VoiceDesignServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.timbre_dir = self.root / "timbre"
        self.app = create_app(timbre_dir=self.timbre_dir)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        if not hasattr(self, "client"):
            return
        self.client.close()
        self.temporary.cleanup()

    def test_design_route_returns_wav_and_registers_generated_timbre(self) -> None:
        fake = FakeLocalAIClient()

        def execute(*, config: object, operation):
            self.assertEqual(config.model, "llm-local.qwen3-tts.1p7b-voicedesign")
            return operation(fake)

        with patch("main.execute_localai_operation", side_effect=execute):
            response = self.client.post(
                ROUTE,
                json={
                    "voice_description": "成年女性，声音清晰自然，语速中等。",
                    "text": "你好。",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, make_wav())
        self.assertEqual(response.headers["x-audio-sample-rate"], "24000")
        self.assertEqual(fake.requests[0]["instruction"], "成年女性，声音清晰自然，语速中等。")
        files = list(self.timbre_dir.glob("qwen3-tts-voicedesign-*.wav"))
        self.assertEqual(len(files), 1)
        self.assertFalse((self.timbre_dir / ".references").exists())

    def test_blank_voice_description_is_rejected_before_inference(self) -> None:
        with patch("main.execute_localai_operation") as execute:
            response = self.client.post(
                ROUTE,
                json={"voice_description": "   ", "text": "你好。"},
            )

        self.assertEqual(response.status_code, 422)
        execute.assert_not_called()

    def test_invalid_wav_response_does_not_leave_generated_files(self) -> None:
        fake = FakeLocalAIClient([b"not a wave file"])

        with (
            self.assertLogs("main", level="WARNING"),
            patch(
                "main.execute_localai_operation",
                side_effect=lambda *, config, operation: operation(fake),
            ),
        ):
            response = self.client.post(
                ROUTE,
                json={"voice_description": "自然、平静", "text": "你好。"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(list(self.timbre_dir.glob("*.wav")), [])
        self.assertEqual(list(self.timbre_dir.glob(".*.tmp")), [])

    def test_long_text_is_split_and_pause_is_inserted_between_wav_segments(self) -> None:
        fake = FakeLocalAIClient([make_wav(), make_wav()])

        with patch(
            "main.execute_localai_operation",
            side_effect=lambda *, config, operation: operation(fake),
        ) as execute:
            response = self.client.post(
                ROUTE,
                json={
                    "voice_description": "自然、平静",
                    "text": "你好世界",
                    "max_chars_per_chunk": 2,
                    "pause_ms": 10,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(fake.requests), 2)
        execute.assert_called_once()
        output = io.BytesIO(response.content)
        with wave.open(output, "rb") as wav_file:
            self.assertEqual(wav_file.getnframes(), 2400 + 240 + 2400)


if __name__ == "__main__":
    unittest.main()
