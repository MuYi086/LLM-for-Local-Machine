"""Stable Audio 3 Medium uv 服务的无模型契约测试。"""

from __future__ import annotations

# Stable Audio 测试 mock worker，验证本地权重检查、FlashAttention 策略和路由契约。
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from fastapi.testclient import TestClient

TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="stable-audio-3-medium-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
REPOSITORY_DIR = Path(__file__).resolve().parents[2]
os.environ.update(
    {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "SOUNDEFFECT_STORAGE_DIR": str(TEST_ROOT / "storage" / "soundEffect"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "UNITALE_LOCALAI_ENABLED": "false",
        "UNITALE_STABLE_AUDIO_3_MEDIUM_PROVIDER": "native",
        "UNITALE_STABLE_AUDIO_3_MEDIUM_MODEL": "unitale.stable-audio3-medium.q8-0",
        "CUDA_RELEASE_DELAY": "0",
        "STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT": "5",
    }
)
os.environ.pop("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", None)
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))

import main
import runtime
import worker

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class StableAudioApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_health_keeps_compatibility_sections_without_loading_model(self):
        with patch.object(main, "cuda_status", return_value={"available": False}):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], 200)
        for key in ("paths", "available", "cuda", "runtime", "last_errors"):
            self.assertIn(key, body)
        self.assertIn("model_required_files_detail", body["available"])
        self.assertIn("flash_attention", body["runtime"])

    def test_duration_alias_and_defaults_are_normalized(self):
        request = main.StableAudio3MediumGenerateRequest(
            prompt="  glass shattering  ", duration=1.5
        )
        payload = main.manager.build_worker_payload(request)
        self.assertEqual(request.prompt, "glass shattering")
        self.assertEqual(request.seconds, 1.5)
        self.assertEqual(payload["seconds"], 1.5)
        self.assertEqual(payload["device"], main.STABLE_AUDIO_3_MEDIUM_DEVICE)
        self.assertIn("require_flash_attn", payload)

    def test_mismatched_seconds_and_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            main.StableAudio3MediumGenerateRequest(prompt="sound", seconds=1, duration=2)

    def test_generate_routes_return_audio_wav(self):
        self.assertEqual(main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR, main.SOUNDEFFECT_STORAGE_DIR)
        saved_before = len(
            list(main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR.glob("stable_audio_3_medium_*.wav"))
        )
        with (
            patch.object(main.manager, "run_worker", return_value=FAKE_WAV) as run_worker,
            patch.object(main, "wait_after_cuda_release"),
        ):
            response = self.client.post(
                "/v1/stableAudio/soundEffect",
                json={"prompt": "a short glass shatter", "seconds": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, FAKE_WAV)
        self.assertEqual(run_worker.call_count, 1)
        self.assertEqual(run_worker.call_args.args[0]["prompt"], "a short glass shatter")
        saved_files = list(
            main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR.glob("stable_audio_3_medium_*.wav")
        )
        self.assertEqual(len(saved_files), saved_before + 1)
        self.assertTrue(all(path.parent == main.SOUNDEFFECT_STORAGE_DIR for path in saved_files))

    def test_localai_provider_returns_wav_and_preserves_storage(self):
        localai_config = replace(
            main.GATEWAY_CONFIG,
            provider="localai",
            localai_enabled=True,
            model=main.LOCALAI_MODEL_ID,
        )
        saved_before = len(
            list(main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR.glob("stable_audio_3_medium_*.wav"))
        )
        with (
            patch.object(main, "GATEWAY_CONFIG", localai_config),
            patch.object(main, "generate_localai", return_value=(FAKE_WAV, 0.1)) as generate,
            patch.object(main, "validate_localai_wav", return_value=(44100, 2)),
        ):
            response = self.client.post(
                "/v1/stableAudio/soundEffect",
                json={"prompt": "rain on a metal roof", "seconds": 12},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, FAKE_WAV)
        generate.assert_called_once()
        self.assertEqual(
            len(list(main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR.glob("stable_audio_3_medium_*.wav"))),
            saved_before + 1,
        )

    def test_localai_rejects_unmappable_sampling_controls(self):
        localai_config = replace(
            main.GATEWAY_CONFIG,
            provider="localai",
            localai_enabled=True,
            model=main.LOCALAI_MODEL_ID,
        )
        with (
            patch.object(main, "GATEWAY_CONFIG", localai_config),
            patch.object(main, "generate_localai") as generate,
        ):
            steps_response = self.client.post(
                "/v1/stableAudio/soundEffect", json={"prompt": "rain", "steps": 9}
            )
            scale_response = self.client.post(
                "/v1/stableAudio/soundEffect", json={"prompt": "rain", "cfg_scale": 2}
            )
            seed_response = self.client.post(
                "/v1/stableAudio/soundEffect", json={"prompt": "rain", "seed": 123}
            )
            dtype_response = self.client.post(
                "/v1/stableAudio/soundEffect", json={"prompt": "rain", "dtype": "float16"}
            )

        self.assertEqual(steps_response.status_code, 422)
        self.assertEqual(scale_response.status_code, 422)
        self.assertEqual(seed_response.status_code, 422)
        self.assertEqual(dtype_response.status_code, 422)
        generate.assert_not_called()

    def test_localai_wav_validation_streams_all_frames(self):
        class FakeSoundFile:
            samplerate = 44100
            channels = 2
            frames = 4
            block_sizes = [2, 2]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def blocks(self, **_kwargs):
                for block_size in self.block_sizes:
                    yield [0] * block_size

        soundfile_module = ModuleType("soundfile")

        def open_soundfile(*_args, **_kwargs):
            return FakeSoundFile()

        soundfile_module.SoundFile = open_soundfile
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            self.assertEqual(main.validate_localai_wav(b"fake wav"), (44100, 2))
            FakeSoundFile.block_sizes = [2]
            with self.assertRaisesRegex(main.LocalAIRequestError, "不完整"):
                main.validate_localai_wav(b"truncated wav")


class WorkerRuntimeTests(unittest.TestCase):
    def test_uv_worker_uses_current_python_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            script = temp_dir / "worker.py"
            script.write_text("# fake worker\n", encoding="utf-8")
            config = runtime.WorkerConfig(
                worker_script=script,
                temp_dir=temp_dir,
                timeout=5,
                label="test",
                file_prefix="test",
            )

            class FakeProcess:
                pid = 12345
                returncode = 0

                def communicate(self, timeout=None):
                    Path(fake_command[-1]).write_bytes(FAKE_WAV)
                    return "worker ok", ""

                def poll(self):
                    return self.returncode

            fake_command: list[str] = []

            def fake_popen(command, **kwargs):
                fake_command.extend(command)
                return FakeProcess()

            with patch.object(runtime.subprocess, "Popen", side_effect=fake_popen):
                audio = runtime.run_local_worker({"prompt": "test"}, config)

            self.assertEqual(audio, FAKE_WAV)
            self.assertEqual(fake_command[0], sys.executable)
            self.assertNotIn("conda", fake_command)
            self.assertEqual(list(temp_dir.glob("test_*")), [])

    def test_flash_attention_is_optional_by_default_but_strict_when_requested(self):
        fake_torch = object()
        with patch.dict(sys.modules, {"flash_attn": None}):
            self.assertFalse(worker.check_flash_attention(fake_torch, required=False))
            with self.assertRaises(RuntimeError):
                worker.check_flash_attention(fake_torch, required=True)

    def test_worker_payload_reader_requires_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps({"prompt": "test"}), encoding="utf-8")
            self.assertEqual(worker.read_payload(str(path))["prompt"], "test")


if __name__ == "__main__":
    unittest.main()
