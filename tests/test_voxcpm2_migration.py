"""独立 VoxCPM2 uv 服务的无模型回归测试。"""

from __future__ import annotations

# VoxCPM2 测试覆盖克隆、sidecar 和 worker 错误清理。
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "clone_voxcpm-2"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="voxcpm2-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
PROMPTS_DIR = TEST_ROOT / "prompts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "VOXCPM2_MODEL_DIR": str(MODEL_DIR),
        "PROMPTS_DIR": str(PROMPTS_DIR),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "timbre"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "VOXCPM2_OUTPUT_DIR": str(TEST_ROOT / "output"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "VOXCPM2_REQUEST_TIMEOUT": "5",
        "LLM_LOCALAI_ENABLED": "false",
    }
)

# 确保扁平目录下的本地导入解析到本服务，而不是其他独立项目中的同名辅助模块。
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
for module_name in (
    "audio_output",
    "audio_trim",
    "gpu_runtime",
    "local_worker",
    "synthesis_request",
):
    sys.modules.pop(module_name, None)
spec = importlib.util.spec_from_file_location(
    "voxcpm2_service_main_for_test",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class VoxCpm2MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "migration-reference.wav"
        content = b"RIFF" + b"\0" * 40
        (PROMPTS_DIR / main.hash_filename(self.filename)).write_bytes(content)
        main.reference_store.prompt_sidecar_path(self.filename).write_text(
            "准确的参考转写。", encoding="utf-8"
        )

    def test_routes_and_health_report_uv_runtime(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v1/voxcpm2/clone"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        with patch.object(main, "cuda_status", return_value={"available": False, "source": "test"}):
            response = TestClient(main.app).get("/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload), {"code", "paths", "available", "cuda", "runtime", "last_errors"}
        )
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["worker_python"], sys.executable)
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertTrue(payload["available"]["worker_script"])
        self.assertIn("not required", payload["runtime"]["flash_attention_policy"])
        self.assertEqual(payload["available"]["selected_provider"], "native")

    def test_upload_and_check_audio_keep_webui_contract(self) -> None:
        from fastapi.testclient import TestClient

        filename = "uploaded-reference.wav"
        content = b"RIFF" + b"\x01\x02" * 32
        client = TestClient(main.app)
        response = client.post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(content), "audio/wav")},
            data={"full_path": filename, "prompt_text": "参考转写"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(response.json()["size_bytes"], len(content))

        checked = client.get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(checked.status_code, 200)
        self.assertEqual(checked.json()["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(checked.json()["size_bytes"], len(content))
        self.assertTrue(checked.json()["has_prompt_text"])

    def test_timbre_reference_upload_does_not_copy_audio_to_clone(self) -> None:
        from fastapi.testclient import TestClient

        filename = "designed-voice.wav"
        content = b"RIFF" + b"\x03\x04" * 32
        timbre_path = Path(main.TIMBRE_STORAGE_DIR) / "voxcpm2-designed.wav"
        timbre_path.write_bytes(content)

        response = TestClient(main.app).post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(content), "audio/wav")},
            data={"full_path": filename, "prompt_text": "设计音色参考文本。"},
        )

        self.assertEqual(response.status_code, 200)
        clone_path = PROMPTS_DIR / main.hash_filename(filename)
        self.assertFalse(clone_path.exists())
        self.assertEqual(main.prompt_audio_path(filename), str(timbre_path))
        self.assertTrue(Path(main.timbre_reference_map_path(filename)).exists())
        self.assertEqual(main.load_prompt_text_sidecar(filename), "设计音色参考文本。")
        checked = TestClient(main.app).get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["exists"])

    def test_clone_payloads_keep_contract(self) -> None:
        clone_request = main.VoxCpm2SynthesizeRequest(
            text="# 要合成的台词",
            audio_path=self.filename,
            prompt_text="准确的参考转写。",
            clone_mode="ultimate",
            cfg_value=1.5,
        )
        clone_payload = main.manager.build_worker_payload(clone_request)
        self.assertEqual(clone_payload["text"], "要合成的台词")
        self.assertEqual(clone_payload["prompt_text"], "准确的参考转写。")
        self.assertEqual(clone_payload["cfg_value"], 1.5)

        controllable = main.VoxCpm2SynthesizeRequest(
            text="台词",
            audio_path=self.filename,
            clone_mode="controllable",
            control_instruction="轻快地说",
            nonverbal_tags=["laughing"],
        )
        controllable_payload = main.manager.build_worker_payload(controllable)
        self.assertIsNone(controllable_payload["prompt_text"])
        self.assertEqual(controllable_payload["nonverbal_tags"], ["laughing"])

        with self.assertRaises(ValueError):
            main.VoxCpm2SynthesizeRequest(
                text="台词",
                audio_path=self.filename,
                clone_mode="controllable",
                control_instruction="轻快地说",
                prompt_text="不应同时出现",
            )

    def test_clone_route_returns_wav_without_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        client = TestClient(main.app)
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = client.post(
                "/v1/voxcpm2/clone",
                json={
                    "text": "你好。",
                    "audio_path": self.filename,
                    "prompt_text": "准确的参考转写。",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()

    def test_worker_uses_current_uv_interpreter_and_cleans_temp_files(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                command = captured["command"]
                output_path = Path(command[command.index("--output-wav") + 1])
                output_path.write_bytes(b"RIFF" + b"\0" * 40)
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = main.manager.run_worker({"operation": "clone"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.VOXCPM2_WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.VOXCPM2_WORKER_TMP_DIR).iterdir()), [])

    def test_flash_attention_is_not_a_project_dependency(self) -> None:
        project_text = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        source_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SERVICE_DIR / "main.py", SERVICE_DIR / "worker.py")
        )
        self.assertNotIn("flash-attn", project_text)
        self.assertNotIn("flash_attn", project_text)
        self.assertNotIn("\nimport flash_attn", source_text)
        self.assertNotIn("\nfrom flash_attn", source_text)

    def test_start_script_uses_uv_without_legacy_voxcpm2_fallback(self) -> None:
        script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertNotIn("LLM_LOCAL_VOXCPM2_PROVIDER", script)
        self.assertNotIn("LLM_LOCAL_VOXCPM2_MODEL", script)
        self.assertIn('"$PROJECT_DIR/scripts/localai/preflight.py"', script)
        self.assertIn('uv run --no-sync --project "$VOXCPM2_PROJECT_DIR"', script)
        self.assertIn('python "$VOXCPM2_PROJECT_DIR/main.py"', script)
        self.assertNotIn("VOXCPM2_RUNTIME", script)
        self.assertNotIn("VOXCPM2_CONDA_ENV", script)
        self.assertNotIn("main/voxcpm2_api.py", script)
        self.assertNotIn("VOXCPM2_UV_BASE_URL", script)
        self.assertIn(
            "http://127.0.0.1:$VOXCPM2_PORT/v1/voxcpm2/clone",
            script,
        )
        control_plane = (REPOSITORY_DIR / "main/main.py").read_text(encoding="utf-8")
        self.assertNotIn("voxcpm2_voice_design", control_plane)
        self.assertNotIn("VOXCPM2_RUNTIME", control_plane)
        self.assertFalse((SERVICE_DIR / "voice_design_worker.py").exists())
        for filename in (
            "voxcpm2_api.py",
            "voxcpm2_helpers.py",
            "voxcpm2_voice_design.py",
            "voxcpm2_voice_design_worker.py",
            "voxcpm2_worker.py",
        ):
            self.assertFalse((REPOSITORY_DIR / "main" / filename).exists())


if __name__ == "__main__":
    unittest.main()
