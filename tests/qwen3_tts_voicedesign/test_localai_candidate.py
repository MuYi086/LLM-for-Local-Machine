"""覆盖九个固定候选注册中的 VoiceDesign 文件与迁移合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.localai.manager import (
    QWEN3_VOICEDESIGN_MODEL_NAME,
    LocalAIManager,
    LocalAIManagerError,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAME = "qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf"


class VoiceDesignCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirror = self.root / "hf-mirror"
        manifest = json.loads((ROOT / "localai/manifests/audio-cpp-models.json").read_text())
        self.candidates = manifest["models"]
        for candidate in self.candidates:
            model_path = (
                self.mirror
                / "audio.cpp-gguf"
                / candidate["source_directory"]
                / candidate["required_file"]
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"GGUF-test-model")
        self.model_file = (
            self.mirror / "audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF" / MODEL_FILENAME
        )
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

    def test_candidate_registers_and_verifies_controlled_gguf_reference(self) -> None:
        self.manager.register_audio_cpp_candidates()
        self.manager.verify_audio_cpp_candidates()

        model_reference = self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.gguf"
        self.assertTrue(model_reference.is_symlink())
        self.assertEqual(model_reference.resolve(), self.model_file.resolve())
        config = (self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.yaml").read_text()
        self.assertIn(f'model: "{model_reference.name}"', config)
        self.assertIn("family:qwen3_tts", config)
        self.assertIn("task:vdes", config)

    def test_voice_design_candidate_rejects_wrong_filename(self) -> None:
        wrong_file = self.root / "other-model.gguf"
        wrong_file.write_bytes(b"GGUF-test-model")
        self.manager.environ["QWEN_VOICEDESIGN_MODEL_FILE"] = str(wrong_file)

        with self.assertRaisesRegex(LocalAIManagerError, "文件名"):
            self.manager.register_audio_cpp_candidates()

    def test_voice_design_candidate_rejects_empty_gguf(self) -> None:
        self.model_file.write_bytes(b"")

        with self.assertRaisesRegex(LocalAIManagerError, "缺失或为空"):
            self.manager.register_audio_cpp_candidates()

    def test_candidate_default_path_uses_hf_mirror_audio_cpp_gguf(self) -> None:
        self.manager.register_audio_cpp_candidates()

        reference = self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.gguf"
        self.assertEqual(reference.resolve(), self.model_file.resolve())

    def test_register_migrates_only_known_legacy_candidate_and_keeps_backup(self) -> None:
        self.models_dir.mkdir(parents=True)
        legacy_dir = self.root / "legacy-safetensors"
        legacy_dir.mkdir()
        (self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.hf").symlink_to(
            legacy_dir, target_is_directory=True
        )
        old_config = (
            f"name: {QWEN3_VOICEDESIGN_MODEL_NAME}\n"
            "backend: audio-cpp\n"
            "parameters:\n"
            f'  model: "{QWEN3_VOICEDESIGN_MODEL_NAME}.hf"\n'
            "options:\n"
            "  - family:qwen3_tts\n"
            "  - task:vdes\n"
            "  - backend:cuda\n"
            "  - device:0\n"
        )
        config_path = self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.yaml"
        config_path.write_text(old_config)

        self.manager.register_audio_cpp_candidates()

        self.assertIn(".gguf", config_path.read_text())
        backups = list((self.root / "localai/config-backups").rglob("*.yaml"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), old_config)

    def test_register_migrates_known_commented_gguf_config_and_keeps_backup(self) -> None:
        self.models_dir.mkdir(parents=True)
        old_config = (
            f"name: {QWEN3_VOICEDESIGN_MODEL_NAME}\n"
            "backend: audio-cpp\n"
            "parameters:\n"
            "  # manager.py 注册此处所需的受控相对 GGUF 文件引用。\n"
            f'  model: "{QWEN3_VOICEDESIGN_MODEL_NAME}.gguf"\n'
            "options:\n"
            "  - family:qwen3_tts\n"
            "  - task:vdes\n"
            "  - backend:cuda\n"
            "  - device:0\n"
        )
        config_path = self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.yaml"
        config_path.write_text(old_config)

        self.manager.register_audio_cpp_candidates()

        self.assertNotIn("# manager.py", config_path.read_text())
        backups = list((self.root / "localai/config-backups").rglob("*.yaml"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), old_config)


if __name__ == "__main__":
    unittest.main()
