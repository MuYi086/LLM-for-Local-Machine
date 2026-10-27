"""HunyuanVideo-1.5 Comfy 的无模型 HTTP 和 workflow 回归。"""

from __future__ import annotations

import importlib.util
import io
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
SERVICE_DIR = REPOSITORY_DIR / "model-video/HunyuanVideo-1.5-Comfy"
ROUTE = "/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"hunyuan-video-test"


def load_module(name: str, path: Path):
    """按独立模块名加载服务代码，避免不同项目 main/worker 名称冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HunyuanVideoComfyTests(unittest.TestCase):
    """隔离文件、GPU 锁与 ComfyUI 进程，不加载真实模型。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hunyuan-video-comfy-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            "HUNYUAN_VIDEO_COMFY_MODEL_DIR": str(self.root / "model"),
            "HUNYUAN_VIDEO_COMFY_CODE_PATH": str(self.root / "ComfyUI"),
            "HUNYUAN_VIDEO_COMFY_IMAGE_DIR": str(self.root / "storage/images"),
            "HUNYUAN_VIDEO_COMFY_OUTPUT_DIR": str(self.root / "storage/video"),
            "HUNYUAN_VIDEO_COMFY_WORKER_TMP_DIR": str(self.root / "worker"),
            "HUNYUAN_VIDEO_COMFY_REQUEST_TIMEOUT": "5",
            "HUNYUAN_VIDEO_COMFY_PORT": "8393",
            "GPU_LOCK_WAIT_TIMEOUT": "900",
            "CUDA_RELEASE_DELAY": "0",
            "LOCAL_FILES_ONLY": "1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("hunyuan_video_comfy_main_tests", SERVICE_DIR / "main.py")
            self.worker = load_module("hunyuan_video_comfy_worker_tests", SERVICE_DIR / "worker.py")
        self.client = TestClient(self.main.app)
        module_check = patch.object(self.main, "module_available", return_value=True)
        module_check.start()
        self.addCleanup(module_check.stop)
        for path in self.main.REQUIRED_MODEL_FILES.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for path in self.main.REQUIRED_COMFYUI_FILES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @contextmanager
    def fake_gpu_lock(self, *args, **kwargs):
        """替换共享 GPU 锁并提供可写的请求指标。"""
        yield GpuRuntimeMetrics(label="test")

    def upload_test_image(self, full_path: str = "portrait.png") -> Path:
        """写入测试图片内容并返回其受控上传路径。"""
        target = self.main.image_path(full_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test-image")
        return target

    def test_health_and_service_route(self) -> None:
        with (
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
        ):
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.main.API_PORT, 8393)
        self.assertTrue(response.json()["available"]["selected_provider_ready"])
        self.assertEqual(response.json()["runtime"]["resolution"], "848x480")
        self.assertIn(ROUTE, {route.path for route in self.main.app.routes})
        start_script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn('HUNYUAN_VIDEO_COMFY_PORT="${HUNYUAN_VIDEO_COMFY_PORT:-8393}"', start_script)
        self.assertIn(ROUTE, start_script)
        self.assertIn('"$hunyuan_video_comfy_pid"', start_script)

    def test_image_upload_is_streamed_to_a_logical_path(self) -> None:
        response = self.client.post(
            "/v1/upload_image",
            data={"full_path": "C:/source/portrait.png"},
            files={"image": ("portrait.png", io.BytesIO(b"png-bytes"), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["size_bytes"], len(b"png-bytes"))
        self.assertEqual(len(response.json()["sha256"]), 64)
        self.assertTrue(self.main.image_path("C:/source/portrait.png").is_file())
        self.assertTrue(
            self.client.get(
                "/v1/check/image", params={"file_name": "C:/source/portrait.png"}
            ).json()["exists"]
        )
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").glob("*.part")), [])

    def test_invalid_request_and_missing_inputs_do_not_enter_gpu_queue(self) -> None:
        for extra in (
            {"steps": 4},
            {"steps": 10},
            {"num_frames": 50},
            {"unknown": True},
        ):
            with self.subTest(extra=extra), patch.object(self.main, "gpu_runtime_lock") as lock:
                response = self.client.post(
                    ROUTE, json={"image_path": "portrait.png", "prompt": "walk", **extra}
                )
                self.assertEqual(response.status_code, 422)
                lock.assert_not_called()
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                ROUTE, json={"image_path": "portrait.png", "prompt": "walk"}
            )
        self.assertEqual(response.status_code, 404)
        lock.assert_not_called()

    def test_missing_model_assets_fail_before_gpu_queue(self) -> None:
        self.upload_test_image()
        next(iter(self.main.REQUIRED_MODEL_FILES.values())).unlink()
        with (
            patch.object(self.main, "image_info", return_value=(1024, 768)),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post(
                ROUTE, json={"image_path": "portrait.png", "prompt": "a quiet garden"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("本地运行环境不完整", response.json()["detail"])
        lock.assert_not_called()

    def test_missing_image_dependency_fails_before_decoding_or_gpu_queue(self) -> None:
        self.upload_test_image()

        def available_except_pillow(name: str) -> bool:
            return name != "PIL"

        with (
            patch.object(
                self.main,
                "module_available",
                side_effect=available_except_pillow,
            ),
            patch.object(self.main, "image_info") as decode_image,
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post(
                ROUTE, json={"image_path": "portrait.png", "prompt": "a quiet garden"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("PIL", response.json()["detail"])
        decode_image.assert_not_called()
        lock.assert_not_called()

    def test_generation_returns_worker_mp4_under_shared_gpu_lock(self) -> None:
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
                json={"image_path": "portrait.png", "prompt": "a quiet garden", "steps": 12},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, MP4_BYTES)
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["steps"], 12)
        wait_after_release.assert_called_once()

    def test_workflow_uses_local_step_distilled_i2v_components(self) -> None:
        workflow = self.worker.build_workflow(
            {"prompt": "a red balloon floats over a city", "steps": 8, "num_frames": 73},
            "input.png",
            "hunyuan_video_1_5/test",
        )
        self.assertEqual(
            workflow["1"]["inputs"]["unet_name"],
            "hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
        )
        self.assertEqual(workflow["3"]["inputs"]["type"], "hunyuan_video_15")
        self.assertEqual(workflow["10"]["inputs"]["width"], 848)
        self.assertEqual(workflow["10"]["inputs"]["height"], 480)
        self.assertEqual(workflow["10"]["inputs"]["length"], 73)
        self.assertEqual(workflow["11"]["inputs"]["steps"], 8)
        self.assertEqual(workflow["11"]["inputs"]["cfg"], 1)
        self.assertEqual(workflow["14"]["class_type"], "SaveVideo")


if __name__ == "__main__":
    unittest.main()
