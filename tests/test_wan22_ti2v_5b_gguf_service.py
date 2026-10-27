"""Wan2.2 TI2V 5B GGUF 的无模型 API 与 workflow 回归。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_local_runtime import GpuRuntimeMetrics

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-video/Wan2.2-TI2V-5B-GGUF"
ROUTE = "/v1/QuantStack/Wan2.2-TI2V-5B-GGUF"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"wan22-video-test"


def load_module(name: str, path: Path):
    """按独立模块名载入服务文件，避免多个项目 main/worker 名称冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Wan22Ti2vServiceTests(unittest.TestCase):
    """隔离文件、GPU 锁和 ComfyUI 进程，不加载真实模型。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="wan22-ti2v-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            "WAN22_TI2V_MODEL_DIR": str(self.root / "model"),
            "WAN22_TI2V_COMFYUI_CODE_PATH": str(self.root / "ComfyUI"),
            "WAN22_TI2V_GGUF_NODE_PATH": str(self.root / "ComfyUI-GGUF"),
            "WAN22_TI2V_IMAGE_DIR": str(self.root / "storage/images"),
            "WAN22_TI2V_OUTPUT_DIR": str(self.root / "storage/video"),
            "WAN22_TI2V_WORKER_TMP_DIR": str(self.root / "worker"),
            "WAN22_TI2V_REQUEST_TIMEOUT": "5",
            "WAN22_TI2V_PORT": "8392",
            "GPU_LOCK_WAIT_TIMEOUT": "900",
            "CUDA_RELEASE_DELAY": "0",
            "LOCAL_FILES_ONLY": "1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("wan22_ti2v_main_tests", SERVICE_DIR / "main.py")
            self.worker = load_module("wan22_ti2v_worker_tests", SERVICE_DIR / "worker.py")
        self.client = TestClient(self.main.app)
        module_check = patch.object(self.main, "module_available", return_value=True)
        module_check.start()
        self.addCleanup(module_check.stop)
        for path in self.main.MODEL_FILES.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for path in (*self.main.REQUIRED_COMFYUI_FILES, *self.main.REQUIRED_GGUF_NODE_FILES):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @contextmanager
    def fake_gpu_lock(self, *args, **kwargs):
        """替换共享 GPU 锁并提供可写的测试指标。"""
        yield GpuRuntimeMetrics(label="test")

    def test_health_and_registered_route(self) -> None:
        with (
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main.shutil, "which", return_value="/usr/bin/ffprobe"),
        ):
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.main.API_PORT, 8392)
        self.assertTrue(response.json()["available"]["selected_provider_ready"])
        self.assertEqual(response.json()["runtime"]["text_encoder_device"], "cpu")
        self.assertIn(ROUTE, {route.path for route in self.main.app.routes})
        start_script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn('WAN22_TI2V_PORT="${WAN22_TI2V_PORT:-8392}"', start_script)
        self.assertIn(ROUTE, start_script)
        self.assertIn('"$wan22_ti2v_pid"', start_script)

    def test_workflow_uses_gguf_and_supports_text_or_image_conditioning(self) -> None:
        text_workflow = self.worker.build_workflow(
            {"prompt": "A cloud drifts over a mountain", "width": 832, "height": 480}, None, "test"
        )
        image_workflow = self.worker.build_workflow(
            {"prompt": "A cloud drifts over a mountain", "width": 832, "height": 480},
            "reference.png",
            "test",
        )
        self.assertEqual(text_workflow["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(text_workflow["1"]["inputs"]["unet_name"], "Wan2.2-TI2V-5B-Q8_0.gguf")
        self.assertEqual(text_workflow["3"]["inputs"]["device"], "cpu")
        self.assertEqual(text_workflow["8"]["inputs"]["length"], 49)
        self.assertNotIn("5", text_workflow)
        self.assertEqual(image_workflow["8"]["inputs"]["start_image"], ["5", 0])
        self.assertEqual(image_workflow["12"]["inputs"]["format"], "auto")

    def test_invalid_dimensions_are_rejected_before_worker(self) -> None:
        response = self.client.post(
            ROUTE,
            json={"prompt": "test", "width": 831, "height": 480},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_weights_fail_before_gpu_lock(self) -> None:
        self.main.MODEL_FILES["vae"].unlink()
        lock = patch.object(
            self.main, "gpu_runtime_lock", side_effect=AssertionError("must not lock")
        )
        lock.start()
        self.addCleanup(lock.stop)

        response = self.client.post(ROUTE, json={"prompt": "test"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("缺少文件", response.json()["detail"])

    def test_generation_returns_saved_video_under_shared_lock(self) -> None:
        output_path = self.root / "generated.mp4"
        output_path.write_bytes(MP4_BYTES)
        with (
            patch.object(self.main.manager, "build_worker_payload", return_value={}),
            patch.object(self.main.manager, "run_worker", return_value=output_path),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
        ):
            response = self.client.post(ROUTE, json={"prompt": "A bird flies above the sea"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, MP4_BYTES)

    def test_image_logical_identifier_never_becomes_a_server_path(self) -> None:
        escaped = self.main.image_path("../../etc/passwd.png")
        self.assertEqual(escaped.parent, self.main.IMAGE_DIR)
        self.assertNotEqual(escaped, Path("/etc/passwd.png"))
        expected = self.main.IMAGE_DIR / self.main.image_storage_name("folder/image.png")
        self.assertEqual(self.main.image_path("folder/image.png"), expected)

    def test_comfy_model_path_config_maps_flat_local_model_directory(self) -> None:
        config_path = self.worker.model_path_config(self.root / "model", self.root / "work")
        config = config_path.read_text(encoding="utf-8")
        self.assertIn("diffusion_models: .", config)
        self.assertIn("text_encoders: .", config)
        self.assertIn("vae: .", config)


if __name__ == "__main__":
    unittest.main()
