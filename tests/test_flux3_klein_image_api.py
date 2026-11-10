"""验证 FLUX.3 Klein 图片 API，不加载模型权重。"""

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

ROOT = Path(__file__).resolve().parents[1]
SERVICE_FILE = ROOT / "model-image/FLUX-3-Image-Klein-GGUF-Q8/service.py"
MODEL_ID = "FLUX-3-Image-Klein-GGUF-Q8"
ROUTE = "/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture-png"
PNG_UPLOAD = b"\x89PNG\r\n\x1a\npayload"

spec = importlib.util.spec_from_file_location("flux3_klein_image_service", SERVICE_FILE)
assert spec is not None and spec.loader is not None
image_service = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = image_service
spec.loader.exec_module(image_service)


class Flux3KleinImageApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "hf-mirror/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF"
        self.model_root.mkdir(parents=True)
        for filename in (
            "flux3-image-klein-Q8_0.gguf",
            "Qwen3VL-4B-Instruct-Q8_0.gguf",
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
            "ae.safetensors",
        ):
            (self.model_root / filename).write_bytes(b"fixture-model-component")
        with patch.dict(os.environ, {"HF_MIRROR_DIR": str(self.root / "hf-mirror")}):
            self.model = load_models(ROOT / "sdcpp/manifests/models.json")[MODEL_ID]
        binary = self.root / "sd-cli"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        lock_path = self.root / "gpu.lock"
        self.config = SdcppConfig(
            binary=binary,
            binary_sha256="a" * 64,
            lock_path=lock_path,
            output_root=self.root / "sdcpp",
        )
        self.upload_dir = self.root / "uploads"
        self.generated: list[tuple[object, object, object]] = []
        self.staged_upload_bytes: list[bytes] = []
        self.client = TestClient(
            image_service.create_app(
                self.config,
                self.model,
                generator=self._generate,
                upload_dir=self.upload_dir,
                max_upload_bytes=128,
            )
        )

    def _generate(self, config, model, request) -> GenerationResult:
        output = self.root / f"result-{len(self.generated)}.png"
        output.write_bytes(PNG_BYTES)
        if request.init_image is not None:
            self.staged_upload_bytes.append(request.init_image.read_bytes())
        self.generated.append((config, model, request))
        return GenerationResult("fixture-request", output, {"worker_seconds": 1.0})

    def test_health_checks_all_components_without_generating(self) -> None:
        response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], MODEL_ID)
        self.assertEqual(self.generated, [])

    def test_generation_route_uses_flux_defaults(self) -> None:
        response = self.client.post(ROUTE, json={"prompt": "a small orange cat", "seed": 42})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(base64.b64decode(response.json()["data"][0]["b64_json"]), PNG_BYTES)
        request = self.generated[0][2]
        self.assertEqual(request.task, "text-to-image")
        self.assertEqual((request.width, request.height), (1024, 1024))
        self.assertEqual(request.steps, 16)
        self.assertEqual(request.cfg_scale, 1.0)
        self.assertEqual(request.seed, 42)

    def test_edit_route_uses_uploaded_image_as_reference_and_cleans_it(self) -> None:
        response = self.client.post(
            f"{ROUTE}/edit",
            data={"prompt": "turn the cat into a fox", "seed": "21"},
            files={"image": ("reference.png", PNG_UPLOAD, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        request = self.generated[0][2]
        self.assertEqual(request.task, "image-to-image")
        self.assertEqual(request.prompt, "turn the cat into a fox")
        self.assertEqual(self.staged_upload_bytes, [PNG_UPLOAD])
        self.assertIsNotNone(request.init_image)
        self.assertFalse(request.init_image.exists())
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_invalid_image_content_is_rejected_before_generation(self) -> None:
        response = self.client.post(
            f"{ROUTE}/edit",
            data={"prompt": "edit"},
            files={"image": ("fake.png", b"not an image", "image/png")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(self.generated, [])
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_oversized_upload_is_rejected_and_removed(self) -> None:
        response = self.client.post(
            f"{ROUTE}/edit",
            data={"prompt": "edit"},
            files={"image": ("large.png", b"\x89PNG\r\n\x1a\n" + b"x" * 129, "image/png")},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.generated, [])
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_edit_upload_is_removed_when_generation_fails(self) -> None:
        def fail(*_args):
            raise RuntimeError("backend failed")

        app = image_service.create_app(
            self.config,
            self.model,
            generator=fail,
            upload_dir=self.upload_dir,
            max_upload_bytes=128,
        )
        with self.assertLogs(image_service.LOGGER, level="ERROR"):
            response = TestClient(app).post(
                f"{ROUTE}/edit",
                data={"prompt": "edit"},
                files={"image": ("reference.png", PNG_UPLOAD, "image/png")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_invalid_size_and_batch_are_rejected_before_generation(self) -> None:
        for payload in (
            {"prompt": "cat", "size": "1000x1024"},
            {"prompt": "cat", "n": 2},
        ):
            with self.subTest(payload=payload):
                response = self.client.post(ROUTE, json=payload)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.generated, [])


if __name__ == "__main__":
    unittest.main()
