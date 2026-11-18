"""验证 LocalAI audio.cpp 多模型候选由同一清单安全注册。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.localai.manager import LocalAIManager, LocalAIManagerError

ROOT = Path(__file__).resolve().parents[1]


class AudioCppCandidateRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirror = self.root / "hf-mirror"
        manifest = json.loads((ROOT / "localai/manifests/audio-cpp-models.json").read_text())
        self.models = manifest["models"]
        for candidate in self.models:
            model_path = (
                self.mirror
                / "audio.cpp-gguf"
                / candidate["source_directory"]
                / candidate["required_file"]
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"GGUF-test-model")
        self.models_dir = self.root / "localai/models"
        self.manager = LocalAIManager(
            environ={
                "HF_MIRROR_DIR": str(self.mirror),
                "LLM_LOCALAI_DATA_DIR": str(self.root / "localai"),
                "LLM_LOCALAI_MODELS_DIR": str(self.models_dir),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_registers_all_fifteen_local_candidates_and_verifies_each(self) -> None:
        self.manager.register_audio_cpp_candidates()
        self.manager.verify_audio_cpp_candidates()

        self.assertEqual(len(list(self.models_dir.glob("*.yaml"))), 15)
        for candidate in self.models:
            model_id = candidate["model_id"]
            reference = self.models_dir / f"{model_id}.gguf"
            self.assertTrue(reference.is_symlink(), model_id)
            self.assertEqual(
                reference.resolve(),
                (
                    self.mirror
                    / "audio.cpp-gguf"
                    / candidate["source_directory"]
                    / candidate["required_file"]
                ).resolve(),
            )
            config = (self.models_dir / f"{model_id}.yaml").read_text()
            self.assertIn(f"family:{candidate['family']}", config)
            self.assertIn(f"task:{candidate['task']}", config)

    def test_new_audio_tools_use_the_existing_mirror_files_and_service_routes(self) -> None:
        expected = {
            "bs_roformer": (
                "BS-RoFormer-ep368-GGUF",
                "bs-roformer-ep368-q8_0.gguf",
                "bs_roformer",
                "sep",
                8344,
            ),
            "qwen3_forced_aligner": (
                "Qwen3-ForcedAligner-0.6B-GGUF",
                "qwen3-forced-aligner-0.6b-f16.gguf",
                "qwen3_forced_aligner",
                "align",
                8345,
            ),
            "mel_band_roformer": (
                "Mel-Band-RoFormer-GGUF",
                "mel-band-roformer-f16.gguf",
                "mel_band_roformer",
                "sep",
                8346,
            ),
            "htdemucs": (
                "HTDemucs-GGUF",
                "htdemucs-f16.gguf",
                "htdemucs",
                "sep",
                8347,
            ),
        }
        candidates = {candidate["key"]: candidate for candidate in self.models}

        for key, values in expected.items():
            with self.subTest(key=key):
                source_directory, filename, family, task, port = values
                candidate = candidates[key]
                service = candidate["service"]
                self.assertEqual(candidate["source_directory"], source_directory)
                self.assertEqual(candidate["required_file"], filename)
                self.assertEqual(candidate["family"], family)
                self.assertEqual(candidate["task"], task)
                self.assertEqual(service["runtime"], "audio_cpp_runtime")
                self.assertEqual(service["directory"], f"model-audio/{source_directory}")
                self.assertEqual(service["port"], port)
                self.assertEqual(service["route"], f"/v1/audio-cpp/{source_directory}")

    def test_missing_candidate_prevents_partial_registration(self) -> None:
        (
            self.mirror
            / "audio.cpp-gguf"
            / self.models[-1]["source_directory"]
            / self.models[-1]["required_file"]
        ).unlink()

        with self.assertRaisesRegex(LocalAIManagerError, "缺失或为空"):
            self.manager.register_audio_cpp_candidates()

        self.assertFalse(self.models_dir.exists())

    def test_registration_refuses_to_overwrite_unknown_existing_config(self) -> None:
        candidate = self.models[1]
        self.models_dir.mkdir(parents=True)
        (self.models_dir / f"{candidate['model_id']}.yaml").write_text("unreviewed config")

        with self.assertRaisesRegex(LocalAIManagerError, "拒绝覆盖"):
            self.manager.register_audio_cpp_candidates()


if __name__ == "__main__":
    unittest.main()
