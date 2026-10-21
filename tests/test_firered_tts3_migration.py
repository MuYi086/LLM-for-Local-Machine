"""FireRedTTS3 独立 uv 服务的无模型回归测试。"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "firered_tts3"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="firered-tts3-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
CODE_DIR = TEST_ROOT / "FireRedTTS3"
for relative_path in (
    "fireredtts3_base/config.json",
    "fireredtts3_base/model.safetensors",
    "fireredtts3_instruct/config.json",
    "fireredtts3_instruct/model.safetensors",
    "redae/config.json",
    "redae/model.safetensors",
    "campp/campplus_voxceleb.bin",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
):
    target = MODEL_DIR / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"test")
CODE_DIR.mkdir(parents=True, exist_ok=True)


def load_service(mode: str, port: int):
    """按模式加载一次 HTTP 模块，测试两个端口的路由隔离。"""
    storage_dir = TEST_ROOT / mode / "storage"
    os.environ.pop("UNITALE_FIRERED_TTS3_CLONE_MODEL", None)
    os.environ.pop("UNITALE_FIRERED_TTS3_TIMBRE_MODEL", None)
    os.environ.update(
        {
            "FIRERED_TTS3_MODE": mode,
            "FIRERED_TTS3_PORT": str(port),
            "FIRERED_TTS3_MODEL_DIR": str(MODEL_DIR),
            "FIRERED_TTS3_CODE_PATH": str(CODE_DIR),
            "STORAGE_DIR": str(storage_dir),
            "TIMBRE_STORAGE_DIR": str(storage_dir / "timbre"),
            "CLONE_STORAGE_DIR": str(storage_dir / "clone"),
            "PROMPTS_DIR": str(storage_dir / "clone"),
            "RUNTIME_CACHE_DIR": str(storage_dir / "cache"),
            "GPU_LOCK_FILE": str(storage_dir / "cache" / "gpu.lock"),
            "FIRERED_TTS3_WORKER_TMP_DIR": str(storage_dir / "worker-tmp"),
            "LOCAL_FILES_ONLY": "1",
            "CUDA_RELEASE_DELAY": "0",
            "FIRERED_TTS3_REQUEST_TIMEOUT": "5",
            "UNITALE_LOCALAI_ENABLED": "false",
            "UNITALE_INFERENCE_PROVIDER": "native",
            "UNITALE_FIRERED_TTS3_CLONE_PROVIDER": "native",
            "UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER": "native",
        }
    )
    sys.path.insert(0, str(SERVICE_DIR))
    sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
    module_name = f"firered_tts3_service_{mode}_{port}"
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_DIR / "main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


clone_main = load_service("clone", 8325)
timbre_main = load_service("timbre", 8304)


class FireRedTTS3MigrationTests(unittest.TestCase):
    """验证端口、存储、worker 生命周期和 WebUI 请求契约。"""

    def setUp(self) -> None:
        self.filename = "reference.wav"
        clone_path = clone_main.reference_store.clone_path(self.filename)
        clone_path.write_bytes(b"RIFF" + b"\0" * 40)
        clone_main.reference_store.prompt_sidecar_path(self.filename).write_text(
            "这是一条准确的参考文本。",
            encoding="utf-8",
        )

    def test_mode_routes_and_health_are_separated(self) -> None:
        from fastapi.testclient import TestClient

        clone_routes = {
            (method, route.path)
            for route in clone_main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertIn(("POST", "/v1/FireRedTTS3/clone"), clone_routes)
        self.assertIn(("POST", "/v1/upload_audio"), clone_routes)
        self.assertNotIn(("POST", "/v1/FireRedTTS3/timbre"), clone_routes)

        timbre_routes = {
            (method, route.path)
            for route in timbre_main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertIn(("POST", "/v1/FireRedTTS3/timbre"), timbre_routes)
        self.assertNotIn(("POST", "/v1/FireRedTTS3/clone"), timbre_routes)

        clone_health = TestClient(clone_main.app).get("/v1/health")
        timbre_health = TestClient(timbre_main.app).get("/v1/health")
        self.assertEqual(clone_health.status_code, 200)
        self.assertEqual(timbre_health.status_code, 200)
        self.assertEqual(clone_health.json()["runtime"]["service_mode"], "clone")
        self.assertEqual(timbre_health.json()["runtime"]["service_mode"], "timbre")
        self.assertTrue(clone_health.json()["available"]["model_required_files"])

    def test_default_localai_model_ids_match_registered_q8_candidates(self) -> None:
        self.assertEqual(clone_main.GATEWAY_CONFIG.model, "unitale.fireredtts3.base.q8-0")
        self.assertEqual(timbre_main.GATEWAY_CONFIG.model, "unitale.fireredtts3.instruct.q8-0")

    def test_clone_upload_and_request_payload(self) -> None:
        from fastapi.testclient import TestClient

        content = b"RIFF" + b"\x01\x02" * 32
        response = TestClient(clone_main.app).post(
            "/v1/upload_audio",
            files={"audio": ("uploaded.wav", io.BytesIO(content), "audio/wav")},
            data={"full_path": "uploaded.wav", "prompt_text": "准确参考文本。"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sha256"], hashlib.sha256(content).hexdigest())

        request = clone_main.CloneRequest(
            text="# 目标台词",
            audio_path=self.filename,
            prompt_text="准确参考文本。",
            language="Chinese",
        )
        payload = clone_main.manager.build_payload(request)
        self.assertEqual(payload["operation"], "clone")
        self.assertEqual(payload["text"], "目标台词")
        self.assertEqual(payload["prompt_text"], "准确参考文本。")
        self.assertEqual(payload["model_path"], str(MODEL_DIR))

    def test_localai_mapping_keeps_clone_reference_and_design_instruction_separate(self) -> None:
        from inference_gateway import UnsupportedCapabilityError

        clone_payload = clone_main.manager.build_payload(
            clone_main.CloneRequest(
                text="目标台词",
                audio_path=self.filename,
                prompt_text="准确参考文本。",
            )
        )
        clone_request = clone_main.build_localai_request(clone_payload)
        self.assertEqual(clone_request["text"], "目标台词")
        self.assertEqual(clone_request["language"], "Chinese")
        self.assertEqual(clone_request["reference_text"], "准确参考文本。")
        self.assertEqual(
            clone_request["voice_path"], Path(clone_main.prompt_audio_path(self.filename))
        )
        self.assertIsNone(clone_request["template_name"])

        design_payload = timbre_main.manager.build_payload(
            timbre_main.TimbreRequest(voice_description="年轻女性，温柔清晰。", text="你好。")
        )
        design_request = timbre_main.build_localai_request(design_payload)
        self.assertEqual(design_request["instruction"], "年轻女性，温柔清晰。")
        self.assertEqual(design_request["template_name"], "voice_design")
        self.assertIsNone(design_request["voice_path"])

        with patch.object(clone_main, "DO_TN", False):
            with self.assertRaises(UnsupportedCapabilityError):
                clone_main.build_localai_request(clone_payload)

    def test_localai_routes_preserve_mode_storage_and_skip_native_worker(self) -> None:
        from fastapi.testclient import TestClient

        from inference_gateway import load_gateway_config

        wav = b"RIFF" + b"\0" * 40
        clone_config = load_gateway_config(
            service_provider_env=clone_main.LOCALAI_PROVIDER_ENV,
            default_model=clone_main.LOCALAI_MODEL_ID,
            request_timeout=5,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_FIRERED_TTS3_CLONE_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        with (
            patch.object(clone_main, "GATEWAY_CONFIG", clone_config),
            patch.object(clone_main, "run_localai", return_value=wav) as run_localai,
            patch.object(clone_main.manager, "run_worker") as native_worker,
        ):
            clone_response = TestClient(clone_main.app).post(
                "/v1/FireRedTTS3/clone",
                json={
                    "text": "目标台词",
                    "audio_path": self.filename,
                    "prompt_text": "准确参考文本。",
                },
            )
        self.assertEqual(clone_response.status_code, 200)
        self.assertEqual(clone_response.content, wav)
        run_localai.assert_called_once()
        native_worker.assert_not_called()
        self.assertTrue(list(Path(clone_main.OUTPUT_DIR).glob("fireredtts3_clone_*.wav")))

        timbre_config = load_gateway_config(
            service_provider_env=timbre_main.LOCALAI_PROVIDER_ENV,
            default_model=timbre_main.LOCALAI_MODEL_ID,
            request_timeout=5,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        with (
            patch.object(timbre_main, "GATEWAY_CONFIG", timbre_config),
            patch.object(timbre_main, "run_localai", return_value=wav),
            patch.object(timbre_main.manager, "run_worker") as native_worker,
            patch.object(timbre_main.reference_store, "register_timbre_file") as register_timbre,
        ):
            timbre_response = TestClient(timbre_main.app).post(
                "/v1/FireRedTTS3/timbre",
                json={"voice_description": "年轻女性，温柔清晰。", "text": "你好。"},
            )
        self.assertEqual(timbre_response.status_code, 200)
        self.assertEqual(timbre_response.content, wav)
        native_worker.assert_not_called()
        register_timbre.assert_called_once()

    def test_clone_route_returns_wav_and_saves_to_clone(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(clone_main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(clone_main.app).post(
                "/v1/FireRedTTS3/clone",
                json={
                    "text": "目标台词",
                    "audio_path": self.filename,
                    "prompt_text": "准确参考文本。",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        saved = list(Path(clone_main.OUTPUT_DIR).glob("fireredtts3_clone_*.wav"))
        self.assertTrue(saved)
        self.assertFalse(list(Path(clone_main.TIMBRE_STORAGE_DIR).glob("fireredtts3_clone_*.wav")))

    def test_timbre_route_returns_wav_and_only_saves_to_timbre(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(timbre_main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(timbre_main.app).post(
                "/v1/FireRedTTS3/timbre",
                json={"voice_description": "年轻女性，温柔清晰。", "text": "你好。"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        saved = list(Path(timbre_main.OUTPUT_DIR).glob("fireredtts3_timbre_*.wav"))
        self.assertTrue(saved)
        self.assertEqual({path.parent for path in saved}, {Path(timbre_main.TIMBRE_STORAGE_DIR)})
        self.assertFalse(any(Path(timbre_main.CLONE_STORAGE_DIR).glob("fireredtts3_timbre_*.wav")))

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

        payload = clone_main.manager.build_payload(
            clone_main.CloneRequest(
                text="目标台词",
                audio_path=self.filename,
                prompt_text="准确参考文本。",
            )
        )
        with patch.object(clone_main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = clone_main.manager.run_worker(payload)

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], clone_main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(clone_main.WORKER_TMP_DIR).iterdir()), [])

    def test_worker_falls_back_to_sdpa_when_flash_attention_is_unavailable(self) -> None:
        worker_spec = importlib.util.spec_from_file_location(
            "firered_tts3_worker_for_test", SERVICE_DIR / "worker.py"
        )
        assert worker_spec and worker_spec.loader
        firered_worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(firered_worker)

        class FakeQwen3Config:
            def __init__(self, **kwargs):
                self.attn_implementation = kwargs.get("attn_implementation")

        fake_transformers = SimpleNamespace(Qwen3Config=FakeQwen3Config)
        with patch.object(firered_worker, "flash_attention_available", return_value=False):
            implementation = firered_worker.resolve_attention_implementation("auto")
            firered_worker.install_attention_compatibility(fake_transformers, implementation)

        config = fake_transformers.Qwen3Config(attn_implementation="flash_attention_2")
        self.assertEqual(implementation, "sdpa")
        self.assertEqual(config.attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
