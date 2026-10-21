"""独立 MOSS VoiceGenerator 服务的无模型回归测试。"""

from __future__ import annotations

# 音色设计测试保护 tokenizer 兼容层、worker 生命周期和 timbre 存储目录。
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "moss_voiceGenerator"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="moss-voicegenerator-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "MOSS_VOICEGENERATOR_MODEL_DIR": str(TEST_ROOT / "model"),
        "MOSS_AUDIO_TOKENIZER_PATH": str(TEST_ROOT / "codec"),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage" / "timbre"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "MOSS_VOICEGENERATOR_REQUEST_TIMEOUT": "5",
        "MOSS_VOICEGENERATOR_PORT": "8302",
        "UNITALE_MOSS_VOICEGENERATOR_PROVIDER": "native",
        "UNITALE_LOCALAI_ENABLED": "false",
    }
)
(TEST_ROOT / "model").mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "codec").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))

spec = importlib.util.spec_from_file_location(
    "moss_voicegenerator_service_main",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class MossVoiceGeneratorMigrationTests(unittest.TestCase):
    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/moss/timbre"),
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
        self.assertTrue(payload["available"]["moss_voicegenerator_model_dir"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertEqual(
            payload["runtime"]["flash_attention_policy"].split(";")[0],
            "optional",
        )

    def test_request_defaults_and_worker_payload(self) -> None:
        request = main.MossDesignRequest(voice_description="成年女性，清晰自然。")
        payload = main.manager.build_worker_payload(request)

        self.assertEqual(request.text, "这是生成的参考音频预览。")
        self.assertEqual(request.max_chars_per_chunk, 0)
        self.assertEqual(payload["model_path"], str(TEST_ROOT / "model"))
        self.assertEqual(payload["codec_path"], str(TEST_ROOT / "codec"))
        self.assertEqual(payload["max_new_tokens"], 4096)
        self.assertEqual(payload["local_files_only"], True)

    def test_localai_mapping_preserves_voice_design_and_rejects_python_only_options(self) -> None:
        from inference_gateway import UnsupportedCapabilityError

        request = main.MossDesignRequest(
            voice_description="成年女性，清晰自然。",
            text="你好。",
        )
        payload = main.manager.build_worker_payload(request)
        mapped = main.build_localai_request(request, payload)

        self.assertEqual(mapped["text"], "你好。")
        self.assertEqual(mapped["instruction"], "成年女性，清晰自然。")
        self.assertEqual(mapped["audio_temperature"], 1.5)
        self.assertEqual(mapped["audio_top_p"], 0.6)
        self.assertEqual(mapped["audio_top_k"], 50)
        self.assertEqual(mapped["audio_repetition_penalty"], 1.1)
        self.assertGreaterEqual(mapped["seed"], 0)
        self.assertLess(mapped["seed"], 2**32)
        self.assertEqual(mapped["text_chunk_size"], len("你好。") + 1)

        native_fallbacks = main.MossDesignRequest(
            voice_description="成年女性，清晰自然。",
            text="你好。",
            audio_temperature=0,
            dtype="",
            attn_implementation="",
        )
        fallback_mapping = main.build_localai_request(
            native_fallbacks,
            main.manager.build_worker_payload(native_fallbacks),
        )
        self.assertEqual(fallback_mapping["audio_temperature"], 1.5)

        unsupported = (
            {"text": "很长的文本" * 30, "max_chars_per_chunk": 10},
            {"max_new_tokens": 512},
            {"dtype": "float16"},
            {"attn_implementation": "flash_attention_2"},
        )
        for overrides in unsupported:
            with self.subTest(overrides=overrides):
                invalid = main.MossDesignRequest(
                    voice_description="成年女性，清晰自然。", **overrides
                )
                with self.assertRaises(UnsupportedCapabilityError):
                    main.build_localai_request(
                        invalid,
                        main.manager.build_worker_payload(invalid),
                    )

    def test_localai_route_uses_gateway_and_keeps_timbre_registration(self) -> None:
        from fastapi.testclient import TestClient

        from inference_gateway import load_gateway_config

        localai_config = load_gateway_config(
            service_provider_env="UNITALE_MOSS_VOICEGENERATOR_PROVIDER",
            default_model="unitale.moss-voicegenerator.bf16",
            request_timeout=5,
            environ={
                "UNITALE_LOCALAI_ENABLED": "true",
                "UNITALE_MOSS_VOICEGENERATOR_PROVIDER": "localai",
                "UNITALE_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
                "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "localai.lock"),
                "CUDA_RELEASE_DELAY": "0",
            },
        )
        wav = b"RIFF\x04\x00\x00\x00WAVE"

        def fake_execute(*, config, operation):
            from unittest.mock import Mock

            client = Mock()
            client.design_moss_voice.return_value = (wav, 0.01)
            result = operation(client)
            self.assertEqual(config.model, "unitale.moss-voicegenerator.bf16")
            return result

        with (
            patch.object(main, "GATEWAY_CONFIG", localai_config),
            patch.object(main, "execute_localai_operation", side_effect=fake_execute),
            patch.object(main.manager, "run_worker") as native_worker,
            patch.object(main.reference_store, "register_timbre_file") as register_timbre,
        ):
            response = TestClient(main.app).post(
                "/v1/moss/timbre",
                json={"voice_description": "成年女性，清晰自然。", "text": "你好。"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, wav)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        native_worker.assert_not_called()
        register_timbre.assert_called_once()

    def test_design_route_returns_wav_without_loading_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v1/moss/timbre",
                json={
                    "voice_description": "成年女性，温柔、清晰。",
                    "text": "你好。",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["text"], "你好。")
        saved_files = list(main.TIMBRE_STORAGE_DIR.glob("moss_voicegenerator_*.wav"))
        self.assertTrue(saved_files)
        self.assertTrue(all(path.parent == main.TIMBRE_STORAGE_DIR for path in saved_files))
        self.assertFalse((TEST_ROOT / "legacy-clone").exists())

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

        with (
            patch.object(main, "is_moss_codec_path_ready", return_value=True),
            patch.object(main.subprocess, "Popen", side_effect=fake_popen),
        ):
            audio_bytes = main.manager.run_worker({"voice_description": "mock"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_moss_processor_loader_does_not_forward_local_files_only(self) -> None:
        worker_spec = importlib.util.spec_from_file_location(
            "moss_voicegenerator_worker_for_test",
            SERVICE_DIR / "worker.py",
        )
        assert worker_spec and worker_spec.loader
        moss_worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(moss_worker)

        captured: dict[str, object] = {}

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_path, **kwargs):
                captured["model_path"] = model_path
                captured["kwargs"] = kwargs
                return object()

        moss_worker.load_moss_processor(
            FakeAutoProcessor,
            TEST_ROOT / "model",
            TEST_ROOT / "codec",
        )

        self.assertEqual(captured["model_path"], str(TEST_ROOT / "model"))
        self.assertNotIn("local_files_only", captured["kwargs"])
        self.assertTrue(captured["kwargs"]["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
