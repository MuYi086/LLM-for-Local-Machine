"""IndexTTS-2.5 HTTP 服务的无模型回归测试。"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "indexTTS-2.5"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="index-tts-25-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "IndexTTS-2.5"
CODE_DIR = TEST_ROOT / "index-tts-source"
W2V_DIR = TEST_ROOT / "w2v-bert-2.0"
CAMPPLUS_PATH = TEST_ROOT / "campplus_cn_common.bin"
BIGVGAN_DIR = TEST_ROOT / "bigvgan"
STORAGE_DIR = TEST_ROOT / "storage"
WORKER_TMP_DIR = TEST_ROOT / "cache" / "worker"


def create_complete_local_resources() -> None:
    """创建小型本地资源夹具，避免访问模型权重或网络。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.yaml",
        "gpt.pth",
        "codec.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    ):
        path = MODEL_DIR / filename
        if filename == "config.yaml":
            path.write_text("version: 2.5\n", encoding="utf-8")
        else:
            path.write_bytes(filename.encode("utf-8"))
    (MODEL_DIR / "qwen0.6bemo4-merge").mkdir(exist_ok=True)

    W2V_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "conformer_shaw.pt",
    ):
        (W2V_DIR / filename).write_bytes(filename.encode("utf-8"))
    CAMPPLUS_PATH.write_bytes(b"campplus")
    BIGVGAN_DIR.mkdir(parents=True, exist_ok=True)
    (BIGVGAN_DIR / "config.json").write_text("{}", encoding="utf-8")
    (BIGVGAN_DIR / "bigvgan_generator.pt").write_bytes(b"bigvgan")

    inference_source = CODE_DIR / "indextts" / "infer_v2_5.py"
    inference_source.parent.mkdir(parents=True, exist_ok=True)
    inference_source.write_text("", encoding="utf-8")


create_complete_local_resources()
os.environ.update(
    {
        "INDEXTTS25_MODEL_DIR": str(MODEL_DIR),
        "INDEXTTS25_CODE_PATH": str(CODE_DIR),
        "INDEXTTS25_W2V_BERT_MODEL_DIR": str(W2V_DIR),
        "INDEXTTS25_CAMPPLUS_PATH": str(CAMPPLUS_PATH),
        "INDEXTTS25_BIGVGAN_MODEL_DIR": str(BIGVGAN_DIR),
        "INDEXTTS25_OUTPUT_DIR": str(STORAGE_DIR / "clone"),
        "INDEXTTS25_WORKER_TMP_DIR": str(WORKER_TMP_DIR),
        "INDEXTTS25_REQUEST_TIMEOUT": "5",
        "INDEXTTS25_DEVICE": "cuda:0",
        "INDEXTTS25_USE_BF16": "1",
        "STORAGE_DIR": str(STORAGE_DIR),
        "PROMPTS_DIR": str(STORAGE_DIR / "clone"),
        "CLONE_STORAGE_DIR": str(STORAGE_DIR / "clone"),
        "TIMBRE_STORAGE_DIR": str(STORAGE_DIR / "timbre"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "CUDA_RELEASE_DELAY": "0",
        "LOCAL_FILES_ONLY": "1",
    }
)

sys.path.insert(0, str(SERVICE_DIR))
spec = importlib.util.spec_from_file_location(
    "index_tts_25_service_main_for_tests",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class IndexTTS25ServiceTests(unittest.TestCase):
    """验证 API 输入、资源预检、存储和一次性 worker 生命周期。"""

    def setUp(self) -> None:
        self.output_dir = Path(main.OUTPUT_DIR)
        for path in self.output_dir.glob("indexTTS25_*.wav"):
            path.unlink()

    @staticmethod
    def save_reference(filename: str = "reference.wav") -> Path:
        """通过真实共享存储目录准备一个逻辑参考音频。"""
        reference_path = main.reference_store.clone_path(filename)
        reference_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        return reference_path

    @staticmethod
    def request_payload(**overrides):
        payload = {
            "text": "你好，欢迎使用 IndexTTS。",
            "audio_path": "reference.wav",
            "lang": "ZH",
        }
        payload.update(overrides)
        return payload

    def test_health_reports_routes_and_complete_local_resources_without_loading_models(self):
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v1/index-team/indexTTS-2.5"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        with (
            patch.object(main, "cuda_status", return_value={"available": True, "source": "test"}),
            patch.object(main, "module_available", return_value=True),
        ):
            response = TestClient(main.app).get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"]["model_resources"])
        self.assertTrue(payload["available"]["index_tts_code"])
        self.assertTrue(payload["available"]["selected_provider_ready"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["worker_python"], sys.executable)
        self.assertNotIn("torch", sys.modules)

    def test_missing_resources_return_503_before_gpu_queue(self):
        from fastapi.testclient import TestClient

        self.save_reference()
        missing_model_dir = TEST_ROOT / "missing-model"
        with (
            patch.object(main, "MODEL_DIR", missing_model_dir),
            patch.object(main, "gpu_runtime_lock", side_effect=AssertionError("GPU lock acquired")),
        ):
            response = TestClient(main.app).post(
                "/v1/index-team/indexTTS-2.5",
                json=self.request_payload(),
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("模型资源不完整", response.json()["detail"])

    def test_missing_logical_reference_returns_404_before_gpu_queue(self):
        from fastapi.testclient import TestClient

        with patch.object(
            main,
            "gpu_runtime_lock",
            side_effect=AssertionError("GPU lock acquired for unknown reference"),
        ):
            response = TestClient(main.app).post(
                "/v1/index-team/indexTTS-2.5",
                json=self.request_payload(audio_path="missing.wav"),
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn("参考音频不存在", response.json()["detail"])

    def test_invalid_language_emotion_vector_and_duration_are_rejected(self):
        from fastapi.testclient import TestClient

        client = TestClient(main.app)
        invalid_payloads = (
            self.request_payload(lang="FR"),
            self.request_payload(emo_vector=[0.0] * 7),
            self.request_payload(duration_factor=2.1),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = client.post("/v1/index-team/indexTTS-2.5", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_success_returns_wav_persists_it_and_cleans_model_view(self):
        from fastapi.testclient import TestClient

        self.save_reference()
        worker_payloads: list[dict[str, object]] = []

        @contextlib.contextmanager
        def no_wait_gpu_lock(label: str):
            self.assertEqual(label, "index_tts25/generate")
            yield None

        def fake_run_worker(payload):
            worker_payloads.append(payload)
            view = Path(payload["model_dir"])
            self.assertTrue((view / "config.yaml").is_symlink())
            self.assertTrue((view / "hf_cache" / "w2v-bert-2.0").is_symlink())
            return b"RIFF\x00\x00\x00\x00WAVE"

        with (
            patch.object(main, "gpu_runtime_lock", side_effect=no_wait_gpu_lock),
            patch.object(main.manager, "run_worker", side_effect=fake_run_worker) as run_worker,
        ):
            response = TestClient(main.app).post(
                "/v1/index-team/indexTTS-2.5",
                json=self.request_payload(
                    duration_factor=1.2,
                    emo_vector=[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                    emo_alpha=0.7,
                ),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        run_worker.assert_called_once()
        payload = worker_payloads[0]
        self.assertEqual(payload["lang"], "ZH")
        self.assertEqual(payload["duration_factor"], 1.2)
        self.assertEqual(payload["emo_vector"][2], 0.8)
        self.assertFalse(Path(payload["worker_tmp_dir"]).exists())
        self.assertEqual(len(list(self.output_dir.glob("indexTTS25_*.wav"))), 1)

    def test_gpu_lock_timeout_skips_worker_and_cleans_model_view(self):
        from fastapi.testclient import TestClient

        self.save_reference()

        @contextlib.contextmanager
        def timed_out_gpu_lock(label: str):
            del label
            raise main.GpuLockTimeoutError("index_tts25/generate", 0.01)
            yield None

        with (
            patch.object(main, "gpu_runtime_lock", side_effect=timed_out_gpu_lock),
            patch.object(main.manager, "run_worker") as run_worker,
        ):
            response = TestClient(main.app).post(
                "/v1/index-team/indexTTS-2.5",
                json=self.request_payload(),
            )

        self.assertEqual(response.status_code, 503)
        run_worker.assert_not_called()
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_worker_timeout_terminates_process_and_removes_request_files(self):
        captured: dict[str, object] = {}

        class FakeProcess:
            pid = None
            returncode = None

            def __init__(self, command):
                self.command = command
                self.communicate_calls = 0

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    raise subprocess.TimeoutExpired(self.command, timeout)
                return "", "worker terminated"

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["process"] = FakeProcess(command)
            return captured["process"]

        manager = main.IndexTTS25WorkerManager()
        worker_tmp_dir = Path(tempfile.mkdtemp(prefix="timeout_", dir=WORKER_TMP_DIR))
        with (
            patch.object(main.subprocess, "Popen", side_effect=fake_popen),
            patch.object(main, "terminate_process_group") as terminate,
            self.assertRaisesRegex(RuntimeError, "worker 超时"),
        ):
            manager.run_worker({"worker_tmp_dir": str(worker_tmp_dir), "model_dir": str(MODEL_DIR)})

        terminate.assert_called()
        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(list(worker_tmp_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
