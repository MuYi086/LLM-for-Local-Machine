"""确保统一启动脚本暴露 VoiceDesign GGUF 的固定运行合同。"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class VoiceDesignStartConfigTests(unittest.TestCase):
    def test_start_script_uses_fixed_port_route_and_local_gguf(self) -> None:
        source = (ROOT / "start.sh").read_text(encoding="utf-8")

        self.assertIn('QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"', source)
        self.assertIn(
            "audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/"
            "qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf",
            source,
        )
        self.assertIn(
            'LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER="${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER:-localai}"',
            source,
        )
        self.assertIn(
            'if [[ "${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER,,}" != "localai" || "${LLM_LOCAL_AUDIO_CPP_PROVIDER,,}" != "localai" ]]',
            source,
        )
        self.assertIn("/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign", source)

    def test_candidate_registration_precedes_managed_start_and_preflight(self) -> None:
        source = (ROOT / "start.sh").read_text(encoding="utf-8")
        register_at = source.index("register-audio-cpp-candidates")
        managed_start_at = source.index('manager.py" start', register_at)
        preflight_at = source.index("scripts/localai/preflight.py", managed_start_at)
        service_launch_at = source.index('python "$VOICEDESIGN_PROJECT_DIR/main.py"')

        self.assertLess(register_at, managed_start_at)
        self.assertLess(managed_start_at, preflight_at)
        self.assertLess(preflight_at, service_launch_at)


if __name__ == "__main__":
    unittest.main()
