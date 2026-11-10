"""验证 MOSS-Image-SFT 图片 API，不加载模型权重。"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from inference_gateway.sdcpp import SdcppConfig, load_models
from inference_gateway.sdcpp.execution import GenerationResult

from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError

SERVICE_FILE = Path(__file__).resolve().parents[1] / "model-image/MOSS-Image-SFT-Q8_0/service.py"
SPEC = importlib.util.spec_from_file_location("moss_image_sft_service", SERVICE_FILE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载 MOSS API 模块：{SERVICE_FILE}")
image_service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = image_service
SPEC.loader.exec_module(image_service)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture-image"
MODEL_ID = "MOSS-Image-SFT-Q8_0"
MODEL_MANIFEST = Path(__file__).resolve().parents[1] / "sdcpp/manifests/models.json"


class MossImageSftApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model_root = self.root / "stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF"
        model_root.mkdir(parents=True)
        for name in (
            "MOSS-Image-SFT-Q8_0.gguf",
            "Qwen3.5-9B-Q6_K.gguf",
            "flux2-vae.safetensors",
            "tokenizer.json",
        ):
            (model_root / name).write_bytes(b"fixture")
        with patch.dict(os.environ, {"HF_MIRROR_DIR": str(self.root)}):
            self.model = load_models(MODEL_MANIFEST)[MODEL_ID]

        binary = self.root / "sd-cli"
        binary.write_text("fixture", encoding="utf-8")
        binary.chmod(0o700)
        self.lock_path = self.root / "gpu-runtime.lock"
        self.config = SdcppConfig(
            binary=binary,
            binary_sha256="0" * 64,
            lock_path=self.lock_path,
            output_root=self.root / "outputs",
            queue_timeout=10,
            run_timeout=20,
            cleanup_timeout=5,
            release_delay=0,
        )
        self.generated: list[tuple[SdcppConfig, object, object]] = []
        self.app = image_service.create_app(self.config, self.model, generator=self._generate)
        self.client = TestClient(self.app)

    def _generate(self, config, model, request) -> GenerationResult:
        output = self.root / "result.png"
        output.write_bytes(PNG_BYTES)
        self.generated.append((config, model, request))
        return GenerationResult("fixture-request", output, {"worker_seconds": 1.0})

    def test_health_checks_components_without_generating(self) -> None:
        response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], MODEL_ID)
        self.assertEqual(self.generated, [])

    def test_route_returns_png_and_maps_moss_defaults(self) -> None:
        response = self.client.post(
            "/v1/openmoss/MOSS-Image-SFT-Q8_0",
            json={"prompt": "a tiny orange cat", "seed": 42},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(base64.b64decode(result["data"][0]["b64_json"]), PNG_BYTES)
        self.assertEqual(result["created"] > 0, True)
        request = self.generated[0][2]
        self.assertEqual(request.task, "text-to-image")
        self.assertEqual(request.width, 1024)
        self.assertEqual(request.height, 1024)
        self.assertEqual(request.steps, 20)
        self.assertEqual(request.cfg_scale, 3.0)
        self.assertEqual(request.seed, 42)

    def test_invalid_size_and_batch_are_rejected_before_generation(self) -> None:
        for payload in (
            {"prompt": "cat", "size": "1000x1024"},
            {"prompt": "cat", "n": 2},
        ):
            with self.subTest(payload=payload):
                response = self.client.post("/v1/openmoss/MOSS-Image-SFT-Q8_0", json=payload)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.generated, [])

    def test_queue_timeout_returns_service_unavailable(self) -> None:
        def timeout(*_args):
            raise GpuLockTimeoutError("MOSS-Image-SFT-Q8_0", 10)

        app = image_service.create_app(self.config, self.model, generator=timeout)
        response = TestClient(app).post("/v1/openmoss/MOSS-Image-SFT-Q8_0", json={"prompt": "cat"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("等待超过", response.json()["detail"])

    def test_quarantine_health_and_generation_return_recovery_guidance(self) -> None:
        self.lock_path.with_name(f"{self.lock_path.name}.quarantine").write_text(
            "quarantined", encoding="utf-8"
        )
        health = self.client.get("/v1/health")
        self.assertEqual(health.status_code, 503)

        def quarantined(*_args):
            raise GpuQuarantinedError("private marker path")

        app = image_service.create_app(self.config, self.model, generator=quarantined)
        response = TestClient(app).post("/v1/openmoss/MOSS-Image-SFT-Q8_0", json={"prompt": "cat"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("恢复流程", response.json()["detail"])
        self.assertNotIn("private marker path", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
