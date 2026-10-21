"""独立 Qwen3 VoiceDesign 服务的无模型回归测试。"""

from __future__ import annotations

# VoiceDesign 测试验证音色输出只进入 timbre 目录且 worker 使用正确 uv 环境。
import importlib.util
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "qwen3_voiceDesign"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="qwen3-voicedesign-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "QWEN_VOICEDESIGN_MODEL_DIR": str(TEST_ROOT / "model"),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage" / "timbre"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "QWEN_VOICEDESIGN_REQUEST_TIMEOUT": "5",
    }
)
(TEST_ROOT / "model").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))

spec = importlib.util.spec_from_file_location(
    "qwen3_voicedesign_service_main",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class Qwen3VoiceDesignMigrationTests(unittest.TestCase):
    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/qwen/timbre"),
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
            set(payload),
            {"code", "paths", "available", "cuda", "runtime", "last_errors"},
        )
        self.assertTrue(payload["available"]["worker_script"])
        self.assertTrue(payload["available"]["qwen_voicedesign_model_dir"])
        self.assertEqual(payload["available"]["selected_provider"], "native")
        self.assertFalse(payload["available"]["localai_model_registered"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertEqual(payload["runtime"]["flash_attention_policy"].split(";")[0], "optional")

    def test_request_defaults_and_worker_payload(self) -> None:
        request = main.QwenDesignRequest(voice_description="成年女性，清晰自然。")
        payload = main.manager.build_worker_payload(request)

        self.assertEqual(request.text, "这是生成的参考音频预览。")
        self.assertEqual(request.max_chars_per_chunk, 0)
        self.assertEqual(payload["model_path"], str(TEST_ROOT / "model"))
        self.assertEqual(payload["language"], "Chinese")
        self.assertEqual(payload["max_new_tokens"], 2048)
        self.assertEqual(payload["local_files_only"], True)

    def test_design_route_returns_wav_without_loading_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v1/qwen/timbre",
                json={
                    "voice_description": "成年女性，温柔、清晰。",
                    "text": "你好。",
                    "temperature": 0.7,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["temperature"], 0.7)
        saved_files = list(main.TIMBRE_STORAGE_DIR.glob("qwen_voicedesign_*.wav"))
        self.assertTrue(saved_files)
        self.assertTrue(all(path.parent == main.TIMBRE_STORAGE_DIR for path in saved_files))
        self.assertFalse((TEST_ROOT / "legacy-clone").exists())

    def test_localai_voice_design_preserves_chunking_pause_and_timbre_storage(self) -> None:
        from fastapi.testclient import TestClient

        generated_wav = b"RIFF\x04\x00\x00\x00WAVE"
        generated_requests: list[dict[str, object]] = []

        class FakeClient:
            def design_voice(self, **kwargs: object):
                generated_requests.append(kwargs)
                return generated_wav, 0.03

        def execute_operation(*, config, operation):
            self.assertEqual(config.model, "unitale.qwen3-tts.1p7b-voicedesign")
            return operation(FakeClient())

        localai_config = replace(
            main.GATEWAY_CONFIG,
            provider="localai",
            localai_enabled=True,
            model="unitale.qwen3-tts.1p7b-voicedesign",
        )
        with (
            patch.object(main, "GATEWAY_CONFIG", localai_config),
            patch.object(main, "execute_localai_operation", side_effect=execute_operation),
            patch.object(main, "join_localai_wavs", return_value=generated_wav) as join_wavs,
            patch.object(main.manager, "run_worker") as run_worker,
        ):
            response = TestClient(main.app).post(
                "/v1/qwen/timbre",
                json={
                    "voice_description": "成年女性，温柔清晰。",
                    "text": "今天很好。明天也好。",
                    "language": "Chinese",
                    "max_chars_per_chunk": 5,
                    "pause_ms": 50,
                    "max_new_tokens": 512,
                    "top_p": 0.85,
                    "temperature": 0.75,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(
            [item["text"] for item in generated_requests], ["今天很好。", "明天也好。"]
        )
        self.assertTrue(
            all(item["instruction"] == "成年女性，温柔清晰。" for item in generated_requests)
        )
        self.assertTrue(all(item["max_new_tokens"] == 512 for item in generated_requests))
        join_wavs.assert_called_once_with([generated_wav, generated_wav], 50)
        run_worker.assert_not_called()
        self.assertEqual(response.content, generated_wav)
        saved_files = list(main.TIMBRE_STORAGE_DIR.glob("qwen_voicedesign_*.wav"))
        self.assertTrue(saved_files)
        self.assertTrue(all(path.parent == main.TIMBRE_STORAGE_DIR for path in saved_files))

    def test_localai_voice_design_rejects_nondefault_hardware_overrides(self) -> None:
        from fastapi.testclient import TestClient

        localai_config = replace(
            main.GATEWAY_CONFIG,
            provider="localai",
            localai_enabled=True,
            model="unitale.qwen3-tts.1p7b-voicedesign",
        )
        with (
            patch.object(main, "GATEWAY_CONFIG", localai_config),
            patch.object(main, "execute_localai_operation") as execute_operation,
        ):
            response = TestClient(main.app).post(
                "/v1/qwen/timbre",
                json={"voice_description": "成年女性。", "dtype": "float16"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("PyTorch 参数", response.json()["detail"])
        execute_operation.assert_not_called()

    def test_worker_uses_uv_interpreter_and_cleans_temporary_files(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                output_path = Path(captured["command"][-1])
                output_path.write_bytes(b"RIFF" + b"\0" * 40)
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = main.manager.run_worker({"voice_description": "mock"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
