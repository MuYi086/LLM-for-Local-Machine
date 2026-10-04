"""Confucius4-TTS 独立服务的无模型回归测试。"""

from __future__ import annotations

# 测试只替换 worker 进程，不下载权重、不导入官方 Torch 推理代码，也不执行 CUDA。
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "Confucius4_TTS"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="confucius4-tts-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
CODE_DIR = TEST_ROOT / "Confucius4-TTS"
W2V_DIR = TEST_ROOT / "w2v-bert-2.0"
VOCODER_DIR = TEST_ROOT / "bigvgan"
STYLE_CHECKPOINT = TEST_ROOT / "campplus_cn_common.bin"

os.environ.update(
    {
        "CONFUCIUS4_TTS_MODEL_DIR": str(MODEL_DIR),
        "CONFUCIUS4_TTS_CODE_PATH": str(CODE_DIR),
        "CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR": str(W2V_DIR),
        "CONFUCIUS4_TTS_VOCODER_MODEL_DIR": str(VOCODER_DIR),
        "CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT": str(STYLE_CHECKPOINT),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "PROMPTS_DIR": str(TEST_ROOT / "storage" / "clone"),
        "CLONE_STORAGE_DIR": str(TEST_ROOT / "storage" / "clone"),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage" / "timbre"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "CONFUCIUS4_TTS_REQUEST_TIMEOUT": "5",
        "CONFUCIUS4_TTS_PORT": "8361",
    }
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for filename in (
    "t2s_model.safetensors",
    "s2a_model.pt",
    "wav2vec2bert_stats.pt",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
):
    (MODEL_DIR / filename).touch()
for path in (
    CODE_DIR / "confuciustts" / "cli" / "inference.py",
    CODE_DIR / "confuciustts" / "flow" / "flow.py",
    CODE_DIR / "external" / "bigvgan" / "bigvgan.py",
    CODE_DIR / "external" / "campplus" / "__init__.py",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
W2V_DIR.mkdir(parents=True, exist_ok=True)
VOCODER_DIR.mkdir(parents=True, exist_ok=True)
STYLE_CHECKPOINT.touch()

sys.path.insert(0, str(SERVICE_DIR))
spec = importlib.util.spec_from_file_location(
    "confucius4_tts_service_main_for_tests",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class Confucius4TtsMigrationTests(unittest.TestCase):
    """验证 HTTP 契约、参考音频存储和一次性 worker 生命周期。"""

    def test_optional_vocalization_duration_reaches_worker_payload(self) -> None:
        """原生感叹音时长只作为可选字段，默认请求仍兼容。"""
        request = main.Confucius4TtsRequest(
            text="아", lang="ko", audio_path="native-test", vocalization_duration_seconds=0.441
        )
        with patch.object(main, "prompt_audio_path", return_value=str(STYLE_CHECKPOINT)):
            payload = main.manager.build_worker_payload(request)
        self.assertEqual(payload["vocalization_duration_seconds"], 0.441)
        self.assertIsNone(
            main.Confucius4TtsRequest(text="hello", audio_path="test").vocalization_duration_seconds
        )

    def test_vocalization_duration_rejects_invalid_values(self) -> None:
        """非有限、零、负数和过长控制值必须在取得 GPU 锁前拒绝。"""
        for duration in (float("nan"), float("inf"), 0, -1, 3):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                main.Confucius4TtsRequest(
                    text="아", audio_path="test", vocalization_duration_seconds=duration
                )

    def test_start_script_exposes_service_on_8361(self) -> None:
        source = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")

        self.assertIn(
            'CONFUCIUS4_TTS_PORT="${CONFUCIUS4_TTS_PORT:-8361}"',
            source,
        )
        self.assertIn(
            'CONFUCIUS4_TTS_MODEL_DIR="${CONFUCIUS4_TTS_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS}"',
            source,
        )
        self.assertIn(
            'CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR="${CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0}"',
            source,
        )
        self.assertIn(
            'CONFUCIUS4_TTS_VOCODER_MODEL_DIR="${CONFUCIUS4_TTS_VOCODER_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x}"',
            source,
        )
        self.assertIn(
            'CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT="${CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT:-$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin}"',
            source,
        )
        self.assertIn("'/v1/confucius4TTS/generate'", source)
        self.assertIn("confucius4_tts_pid", source)

    def test_project_declares_torchcodec_for_current_torchaudio(self) -> None:
        source = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"torchcodec', source)

    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v1/confucius4TTS/generate"),
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
        self.assertTrue(payload["available"]["confucius4_tts_model"])
        self.assertTrue(payload["available"]["confucius4_tts_code"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["worker_python"], sys.executable)

    def test_upload_check_and_generate_route(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        upload = client.post(
            "/v1/upload_audio",
            files={"audio": ("reference.wav", io.BytesIO(b"RIFF" + b"\0" * 40), "audio/wav")},
            data={"full_path": "reference.wav", "prompt_text": "参考文本。"},
        )
        self.assertEqual(upload.status_code, 200)
        self.assertTrue(client.get("/v1/check/audio?file_name=reference.wav").json()["exists"])

        with patch.object(
            main.manager,
            "run_worker",
            return_value=b"RIFF" + b"\0" * 40,
        ) as run_worker:
            response = client.post(
                "/v1/confucius4TTS/generate",
                json={"text": "你好。", "lang": "zh", "audio_path": "reference.wav"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["lang"], "zh")
        self.assertTrue(list((TEST_ROOT / "storage" / "clone").glob("confucius4_tts_*.wav")))

    def test_request_defaults_and_worker_payload(self) -> None:
        audio_path = TEST_ROOT / "reference-default.wav"
        audio_path.write_bytes(b"RIFF" + b"\0" * 40)
        stored_path = main.reference_store.clone_path("reference-default.wav")
        stored_path.write_bytes(audio_path.read_bytes())

        request = main.Confucius4TtsRequest(
            text="你好。",
            audio_path="reference-default.wav",
        )
        payload = main.manager.build_worker_payload(request)

        self.assertEqual(request.lang, "zh")
        self.assertEqual(request.temperature, 0.8)
        self.assertEqual(request.n_timesteps, 25)
        self.assertEqual(payload["model_dir"], str(MODEL_DIR))
        self.assertEqual(payload["code_path"], str(CODE_DIR))
        self.assertEqual(payload["reference_audio_path"], str(stored_path))
        self.assertTrue(payload["local_files_only"])

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
            result = main.manager.run_worker({"text": "mock request"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(result.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_request_contract_rejects_unknown_style_prompt(self) -> None:
        with self.assertRaises(ValueError):
            main.Confucius4TtsRequest.model_validate(
                {
                    "text": "你好。",
                    "lang": "zh",
                    "audio_path": "reference.wav",
                    "style_prompt": "不适用",
                }
            )


if __name__ == "__main__":
    unittest.main()
