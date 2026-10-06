"""独立 MiMo VoiceDesign 服务的无网络回归测试。"""

from __future__ import annotations

# MiMo 测试 mock 网络边界，重点验证切分、重试、WAV 拼接和本地存储契约。
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "mimo_tts"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="mimo-tts-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
TIMBRE_DIR = TEST_ROOT / "timbre"
TIMBRE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIMBRE_STORAGE_DIR": str(TIMBRE_DIR),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "MIMO_TTS_HOST": "127.0.0.1",
        "MIMO_TTS_PORT": "8303",
        "MIMO_MAX_RETRIES": "0",
    }
)
sys.path.insert(0, str(SERVICE_DIR))
sys.modules.pop("audio_output", None)
spec = importlib.util.spec_from_file_location(
    "mimo_tts_service_main_for_test", SERVICE_DIR / "main.py"
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class MimoTtsMigrationTests(unittest.TestCase):
    def test_routes_health_and_standalone_startup_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/mimo/timbre"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        response = TestClient(main.app).get("/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["paths"]["timbre_storage_dir"], str(TIMBRE_DIR))
        self.assertEqual(payload["runtime"]["provider"], "mimo")

    def test_design_route_returns_wav_and_persists_timbre_copy(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main, "run_mimo_voice_design", return_value=wav):
            response = TestClient(main.app).post(
                "/v1/mimo/timbre",
                json={
                    "voice_description": "成年女性，温柔、清晰。",
                    "text": "你好。",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        saved_files = list(TIMBRE_DIR.glob("mimo_voicedesign_*.wav"))
        self.assertTrue(saved_files)
        self.assertEqual(saved_files[-1].read_bytes(), wav)

    def test_mimo_logic_is_not_kept_in_control_plane(self) -> None:
        control_plane_source = (REPOSITORY_DIR / "main/main.py").read_text(encoding="utf-8")
        self.assertIn("/v1/mimo/timbre", control_plane_source)
        self.assertNotIn("run_mimo_voice_design", control_plane_source)
        self.assertIn("forward_mimo_design_request", control_plane_source)
        self.assertFalse((REPOSITORY_DIR / "api").exists())

    def test_start_script_launches_mimo_and_direct_step_services(self) -> None:
        start_script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'MIMO_TTS_PROJECT_DIR="${MIMO_TTS_PROJECT_DIR:-$PROJECT_DIR/mimo_tts}"', start_script
        )
        self.assertIn('uv run --no-sync --project "$MIMO_TTS_PROJECT_DIR"', start_script)
        self.assertIn('python "$MIMO_TTS_PROJECT_DIR/main.py"', start_script)
        self.assertIn("MIMO_TTS_PORT:-8303", start_script)
        self.assertIn(
            'MIMO_TTS_PROXY_URL="${MIMO_TTS_PROXY_URL:-http://127.0.0.1:$MIMO_TTS_PORT/v1/mimo/timbre}"',
            start_script,
        )
        self.assertIn("$STEP_AUDIO_EDITX_PORT/v1/stepAudioEditx/edit", start_script)
        for port_default in (
            "PORT:-8300",
            "QWEN_VOICEDESIGN_PORT:-8301",
            "MOSS_VOICEGENERATOR_PORT:-8302",
            "MIMO_TTS_PORT:-8303",
            "STABLE_AUDIO_3_MEDIUM_PORT:-8311",
            "SOUNDEFFECT_PORT:-8312",
            "QWEN3_TTS_PORT:-8321",
            "VOXCPM2_PORT:-8322",
            "LONGCAT_AUDIODIT_PORT:-8323",
            "DOTS_TTS_SOAR_PORT:-8324",
            "STEP_AUDIO_EDITX_PORT:-8331",
        ):
            self.assertIn(port_default, start_script)
        for route in (
            "/v1/qwen/timbre",
            "/v1/moss/timbre",
            "/v1/mimo/timbre",
            "/v1/stableAudio/soundEffect",
            "/v1/moss/soundEffect",
            "/v1/qwen/clone",
            "/v1/voxcpm2/clone",
            "/v1/longCat/clone",
            "/v1/dotsTTS/clone",
            "/v1/stepAudioEditx/edit",
        ):
            self.assertIn(route, start_script)
        for output_variable, storage_variable in (
            ("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", "SOUNDEFFECT_STORAGE_DIR"),
            ("QWEN3_TTS_OUTPUT_DIR", "CLONE_STORAGE_DIR"),
            ("VOXCPM2_OUTPUT_DIR", "CLONE_STORAGE_DIR"),
            ("LONGCAT_AUDIODIT_OUTPUT_DIR", "CLONE_STORAGE_DIR"),
            ("DOTS_TTS_SOAR_OUTPUT_DIR", "CLONE_STORAGE_DIR"),
            ("STEP_AUDIO_EDITX_OUTPUT_DIR", "CLONE_STORAGE_DIR"),
        ):
            self.assertIn(
                f'export {output_variable}="${{{output_variable}:-${storage_variable}}}"',
                start_script,
            )
        self.assertNotIn("$PORT/v1/mimo/timbre", start_script)
        self.assertNotIn("$PORT/v1/stepAudioEditx/edit", start_script)


if __name__ == "__main__":
    unittest.main()
