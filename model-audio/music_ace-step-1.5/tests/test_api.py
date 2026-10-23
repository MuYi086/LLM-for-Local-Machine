"""ACE-Step 1.5 服务的无模型 API 契约测试。"""

from __future__ import annotations

# API 合约测试只检查 Pydantic、健康接口和 mock 生成结果，不加载 ACE-Step。
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = SERVICE_DIR.parent.parent
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="ace-step-1-5-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "ACESTEP_MODEL_DIR": str(TEST_ROOT / "model"),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "BGM_STORAGE_DIR": str(TEST_ROOT / "storage" / "bgm"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "ACESTEP_WORKER_TMP_DIR": str(TEST_ROOT / "worker-tmp"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "ACESTEP_REQUEST_TIMEOUT": "5",
    }
)
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
sys.path.insert(0, str(REPOSITORY_DIR / "llm_local_runtime/src"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from runtime import WorkerResult  # noqa: E402

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class AceStepApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_health_does_not_load_model(self) -> None:
        with patch.object(main, "cuda_status", return_value={"available": False}):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], 200)
        self.assertFalse(body["available"]["model_complete"])
        self.assertEqual(body["runtime"]["port"], 8313)
        self.assertEqual(body["runtime"]["sample_rate"], 48000)

    def test_model_status_accepts_sharded_transformer_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
            for component in main.REQUIRED_MODEL_PATHS[1:]:
                (model_dir / component).mkdir()
            (
                model_dir / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
            ).write_bytes(b"shard")

            with patch.object(main, "ACESTEP_MODEL_DIR", model_dir):
                status = main.model_status()

        self.assertTrue(status["complete"])
        self.assertEqual(status["transformer_weight_files"], 1)

    def test_request_validation(self) -> None:
        self.assertEqual(
            self.client.post("/v1/aceStep/bgm", json={"prompt": "x", "seconds": 601}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/v1/aceStep/bgm", json={"prompt": "   "}).status_code, 422
        )

    def test_build_worker_payload_preserves_runtime_defaults(self) -> None:
        request = main.AceStepBgmRequest(
            prompt="  dark underscore  ",
            seconds=30,
            steps=8,
            bpm=58,
            keyscale="D minor",
            timesignature="4",
            seed=42,
        )
        payload = main.manager.build_worker_payload(request)
        self.assertEqual(request.prompt, "dark underscore")
        self.assertEqual(payload["model_path"], str(main.ACESTEP_MODEL_DIR))
        self.assertEqual(payload["dtype"], "bfloat16")
        self.assertEqual(payload["offload"], "model")
        self.assertTrue(payload["vae_tiling"])
        self.assertEqual(payload["seed"], 42)

    def test_missing_model_returns_locatable_error(self) -> None:
        with patch.object(main, "wait_after_cuda_release"):
            response = self.client.post(
                "/v1/aceStep/bgm",
                json={"prompt": "dark underscore", "seconds": 10},
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("ACE-Step", response.json()["detail"])
        self.assertIn(str(main.ACESTEP_MODEL_DIR), response.json()["detail"])

    def test_mocked_generation_returns_wav_and_seed_headers(self) -> None:
        result = WorkerResult(audio=FAKE_WAV, metadata={"seed": 123, "sample_rate": 48000})
        saved_before = len(list(main.ACESTEP_OUTPUT_DIR.glob("ace_step_1_5_*.wav")))
        with patch.object(main.manager, "run_worker", return_value=result):
            response = self.client.post(
                "/v1/aceStep/bgm",
                json={"prompt": "dark underscore", "seconds": 10, "seed": -1},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, FAKE_WAV)
        self.assertEqual(response.headers["x-ace-step-seed"], "123")
        self.assertEqual(response.headers["x-ace-step-sample-rate"], "48000")
        saved = list(main.ACESTEP_OUTPUT_DIR.glob("ace_step_1_5_*.wav"))
        self.assertEqual(len(saved), saved_before + 1)


if __name__ == "__main__":
    unittest.main()
