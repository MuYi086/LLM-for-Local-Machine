"""Kandinsky 6 Lite 蒸馏版服务的无模型 API 与 worker 合同回归。"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from llm_local_runtime import GpuRuntimeMetrics

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-video/Kandinsky-6.0-Lite-Distill-BF16"
ROUTE = "/v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"kandinsky-video-test"


def load_module(name: str, path: Path):
    """按独立模块名加载代码，避免多个服务的 main/worker 名称冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Kandinsky6LiteDistillServiceTests(unittest.TestCase):
    """隔离模型文件、GPU 锁和推理依赖，不加载真实权重。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="kandinsky6-lite-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            "KANDINSKY6_MODEL_DIR": str(self.root / "model"),
            "KANDINSKY6_IMAGE_DIR": str(self.root / "storage/images"),
            "KANDINSKY6_OUTPUT_DIR": str(self.root / "storage/video"),
            "KANDINSKY6_WORKER_TMP_DIR": str(self.root / "worker"),
            "KANDINSKY6_REQUEST_TIMEOUT": "5",
            "KANDINSKY6_PORT": "8369",
            "GPU_LOCK_WAIT_TIMEOUT": "900",
            "CUDA_RELEASE_DELAY": "0",
            "LOCAL_FILES_ONLY": "1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("kandinsky6_lite_main_tests", SERVICE_DIR / "main.py")
            self.worker = load_module("kandinsky6_lite_worker_tests", SERVICE_DIR / "worker.py")
        self.client = TestClient(self.main.app)
        module_check = patch.object(self.main, "module_available", return_value=True)
        module_check.start()
        self.addCleanup(module_check.stop)
        for path in self.main.REQUIRED_MODEL_FILES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @contextmanager
    def fake_gpu_lock(self, *args, **kwargs):
        """替换共享 GPU 锁并提供请求指标。"""
        yield GpuRuntimeMetrics(label="test")

    def upload_test_image(self, full_path: str = "portrait.png") -> Path:
        """创建受控上传图片路径，避免测试读取任意本地文件。"""
        target = self.main.image_path(full_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test-image")
        return target

    def test_health_reports_local_assets_and_fixed_distilled_settings(self) -> None:
        with patch.object(self.main, "cuda_status", return_value={"available": True}):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.main.API_PORT, 8369)
        self.assertTrue(response.json()["available"]["selected_provider_ready"])
        self.assertEqual(response.json()["runtime"]["inference_steps"], 10)
        self.assertEqual(response.json()["runtime"]["guidance_scale"], 1.0)
        self.assertEqual(response.json()["runtime"]["resolution"], "864x480")
        self.assertIn(ROUTE, {route.path for route in self.main.app.routes})

    def test_image_upload_uses_a_logical_path_and_is_checkable(self) -> None:
        response = self.client.post(
            "/v1/upload_image",
            data={"full_path": "C:/source/portrait.png"},
            files={"image": ("portrait.png", io.BytesIO(b"png-bytes"), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.main.image_path("C:/source/portrait.png").is_file())
        self.assertTrue(
            self.client.get(
                "/v1/check/image", params={"file_name": "C:/source/portrait.png"}
            ).json()["exists"]
        )
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").glob("*.part")), [])

    def test_invalid_requests_and_missing_image_do_not_enter_gpu_queue(self) -> None:
        for extra in ({"unknown": True}, {"prompt": "  "}, {"seed": -1}):
            with self.subTest(extra=extra), patch.object(self.main, "gpu_runtime_lock") as lock:
                response = self.client.post(
                    ROUTE,
                    json={"image_path": "portrait.png", "prompt": "a bird flies", **extra},
                )
                self.assertEqual(response.status_code, 422)
                lock.assert_not_called()

        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                ROUTE, json={"image_path": "portrait.png", "prompt": "a bird flies"}
            )
        self.assertEqual(response.status_code, 404)
        lock.assert_not_called()

    def test_missing_model_asset_fails_before_gpu_queue(self) -> None:
        self.upload_test_image()
        self.main.REQUIRED_MODEL_FILES[0].unlink()
        with (
            patch.object(self.main, "image_info", return_value=(1024, 768)),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post(
                ROUTE, json={"image_path": "portrait.png", "prompt": "a bird flies"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("本地模型文件不完整", response.json()["detail"])
        lock.assert_not_called()

    def test_generation_returns_mp4_with_official_distilled_parameters(self) -> None:
        self.upload_test_image()
        output_path = self.root / "generated.mp4"
        output_path.write_bytes(MP4_BYTES)
        with (
            patch.object(self.main, "image_info", return_value=(1024, 768)),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker", return_value=output_path) as run_worker,
            patch.object(self.main, "wait_after_cuda_release") as wait_after_release,
        ):
            response = self.client.post(
                ROUTE,
                json={"image_path": "portrait.png", "prompt": "a bird flies", "seed": 17},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, MP4_BYTES)
        payload = run_worker.call_args.args[0]
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(payload["num_inference_steps"], 10)
        self.assertEqual(payload["guidance_scale"], 1.0)
        self.assertEqual(payload["width"], 864)
        self.assertEqual(payload["height"], 480)
        self.assertFalse(payload["sample_audio"])
        wait_after_release.assert_called_once()

    def test_worker_loads_local_bf16_pipeline_and_encodes_video_only(self) -> None:
        class FakeGenerator:
            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        class FakePipeline:
            def __init__(self) -> None:
                self.vae = types.SimpleNamespace(enable_tiling=MagicMock())
                self.enable_sequential_cpu_offload = MagicMock()
                self.call = None

            def __call__(self, **kwargs):
                self.call = kwargs
                return types.SimpleNamespace(frames=[["frame"]])

        pipeline = FakePipeline()
        from_pretrained = MagicMock(return_value=pipeline)
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.Kandinsky6TI2VAPipeline = types.SimpleNamespace(
            from_pretrained=from_pretrained
        )
        encode_video = MagicMock()
        fake_utils = types.ModuleType("diffusers.utils")
        fake_utils.load_image = MagicMock(return_value="decoded-image")
        fake_utils.encode_video = encode_video
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bf16"
        fake_torch.Generator = lambda device: FakeGenerator()

        payload = {
            "model_dir": "/models/kandinsky",
            "reference_image_path": "/uploads/image.png",
            "prompt": "a bird flies",
            "seed": 17,
            "num_inference_steps": 10,
            "guidance_scale": 1.0,
            "width": 864,
            "height": 480,
            "num_frames": 121,
            "frame_rate": 24,
            "local_files_only": True,
            "sample_audio": False,
        }
        output_path = self.root / "worker-output.mp4"
        with patch.dict(
            sys.modules,
            {"diffusers": fake_diffusers, "diffusers.utils": fake_utils, "torch": fake_torch},
        ):
            self.worker.generate_video(payload, output_path)

        from_pretrained.assert_called_once_with(
            "/models/kandinsky", torch_dtype="bf16", local_files_only=True
        )
        pipeline.enable_sequential_cpu_offload.assert_called_once()
        pipeline.vae.enable_tiling.assert_called_once()
        self.assertEqual(pipeline.call["num_inference_steps"], 10)
        self.assertEqual(pipeline.call["guidance_scale"], 1.0)
        self.assertFalse(pipeline.call["sample_audio"])
        encode_video.assert_called_once_with(["frame"], fps=24, output_path=str(output_path))


if __name__ == "__main__":
    unittest.main()
