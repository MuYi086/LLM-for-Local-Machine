"""MiniMax H3 本地 ComfyUI 服务的无模型合同回归。"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_local_runtime import GpuRuntimeMetrics

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-video/MiniMax-H3-16G-HiFi"
ROUTE = "/v1/minimax/H3-16G-HiFi"


def load_module(name: str, path: Path):
    """按服务专用名称加载 Python 文件，避免 main/worker 模块冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MiniMaxH3ServiceTests(unittest.TestCase):
    """用临时权重路径、假 GPU 状态和 worker 验证 HTTP 编排合同。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="minimax-h3-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.comfy_dir = self.root / "ComfyUI"
        self.comfy_python = self.root / "comfy-python"
        self.comfy_python.touch()
        self.comfy_python.chmod(0o755)
        (self.comfy_dir / "comfy_extras").mkdir(parents=True)
        (self.comfy_dir / "custom_nodes/ComfyUI-sol-attn-main").mkdir(parents=True)
        (self.comfy_dir / "main.py").touch()
        (self.comfy_dir / "comfy_extras/nodes_minimax_h3.py").touch()
        (self.comfy_dir / "custom_nodes/ComfyUI-sol-attn-main/__init__.py").touch()
        (self.model_dir / "diffusion_models").mkdir(parents=True)
        (self.model_dir / "text_encoders").mkdir()
        (self.model_dir / "vae").mkdir()
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            "MINIMAX_H3_MODEL_DIR": str(self.model_dir),
            "MINIMAX_H3_COMFYUI_CODE_PATH": str(self.comfy_dir),
            "MINIMAX_H3_COMFYUI_PYTHON": str(self.comfy_python),
            "MINIMAX_H3_IMAGE_DIR": str(self.root / "storage/images"),
            "MINIMAX_H3_OUTPUT_DIR": str(self.root / "storage/video"),
            "MINIMAX_H3_WORKER_TMP_DIR": str(self.root / "worker"),
            "MINIMAX_H3_PORT": "8392",
            "MINIMAX_H3_REQUEST_TIMEOUT": "5",
            "GPU_LOCK_WAIT_TIMEOUT": "900",
            "CUDA_RELEASE_DELAY": "0",
            "LOCAL_FILES_ONLY": "1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("minimax_h3_main_tests", SERVICE_DIR / "main.py")
            self.worker = load_module("minimax_h3_worker_tests", SERVICE_DIR / "worker.py")
            self.runtime = sys.modules["minimax_h3_runtime"]
        self.client = TestClient(self.main.app)
        module_check = patch.object(self.main, "module_available", return_value=True)
        module_check.start()
        self.addCleanup(module_check.stop)
        comfy_module_check = patch.object(
            self.main,
            "comfyui_modules_available",
            return_value={
                "torch": True,
                "triton": True,
                "pydantic": True,
                "pydantic_settings": True,
            },
        )
        comfy_module_check.start()
        self.addCleanup(comfy_module_check.stop)
        for path in self.main.REQUIRED_MODEL_FILES.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    @contextmanager
    def fake_gpu_lock(self, *args, **kwargs):
        """替换共享 GPU 锁并提供请求指标。"""
        yield GpuRuntimeMetrics(label="test")

    def upload_image(self, full_path: str) -> Path:
        """创建上传逻辑标识对应的测试图片文件。"""
        target = self.main.image_path(full_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test-image")
        return target

    def test_health_reports_models_comfyui_and_conservative_defaults(self) -> None:
        self.assertEqual(
            self.main.COMFYUI_REQUIRED_MODULES,
            ("torch", "triton", "pydantic", "pydantic_settings"),
        )
        with patch.object(self.main, "cuda_status", return_value={"available": True}):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.main.API_PORT, 8392)
        body = response.json()
        self.assertTrue(body["available"]["selected_provider_ready"])
        self.assertEqual(body["runtime"]["resolution"], "960x544")
        self.assertEqual(body["runtime"]["duration_seconds"], 5)
        self.assertEqual(body["runtime"]["reserved_vram_gib"], 2.5)
        self.assertEqual(body["runtime"]["min_free_vram_mib"], 13_000)
        self.assertEqual(
            body["available"]["comfyui_dependencies"],
            {
                "torch": True,
                "triton": True,
                "pydantic": True,
                "pydantic_settings": True,
            },
        )
        self.assertTrue(body["available"]["native_audio_ready"])
        self.assertIn(ROUTE, {route.path for route in self.main.app.routes})

    def test_health_keeps_silent_generation_ready_without_audio_vae(self) -> None:
        self.main.REQUIRED_MODEL_FILES["audio_vae"].unlink()
        with (
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "query_gpu_memory", return_value={"free_mib": 14_000}),
        ):
            response = self.client.get("/v1/health")

        self.assertTrue(response.json()["available"]["selected_provider_ready"])
        self.assertFalse(response.json()["available"]["native_audio_ready"])

    def test_gpu_memory_gate_matches_selected_physical_gpu_uuid(self) -> None:
        result = SimpleNamespace(
            stdout=(
                "0, GPU-aaa, RTX A, 1000, 16384, 15384\n1, GPU-bbb, RTX B, 12000, 16384, 4384\n"
            )
        )
        with patch.object(self.runtime.subprocess, "run", return_value=result):
            memory = self.runtime.query_gpu_memory("GPU-bbb")

        self.assertEqual(memory, {"free_mib": 4384, "total_mib": 16384})

    def test_image_upload_and_check_use_logical_paths(self) -> None:
        response = self.client.post(
            "/v1/upload_image",
            data={"full_path": "C:/inputs/first.png"},
            files={"image": ("first.png", io.BytesIO(b"png-data"), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.main.image_path("C:/inputs/first.png").is_file())
        self.assertTrue(
            self.client.get("/v1/check/image", params={"file_name": "C:/inputs/first.png"}).json()[
                "exists"
            ]
        )

    def test_request_validation_happens_before_gpu_queue(self) -> None:
        for payload in (
            {"prompt": "  "},
            {"prompt": "a bird", "width": 1024},
            {"prompt": "a bird", "duration_seconds": 6},
            {"prompt": "a bird", "mode": "ref2va"},
            {"prompt": "a bird", "mode": "fl2va", "reference_image_paths": ["ref.png"]},
        ):
            with self.subTest(payload=payload), patch.object(self.main, "gpu_runtime_lock") as lock:
                response = self.client.post(ROUTE, json=payload)
                self.assertEqual(response.status_code, 422)
                lock.assert_not_called()

    def test_missing_reference_and_low_vram_do_not_enter_gpu_queue(self) -> None:
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                ROUTE,
                json={"prompt": "a bird", "first_frame_path": "first.png"},
            )
        self.assertEqual(response.status_code, 404)
        lock.assert_not_called()

        self.upload_image("first.png")
        with (
            patch.object(self.main, "image_info", return_value=(1280, 720)),
            patch.object(self.main, "query_gpu_memory", return_value={"free_mib": 12_999}),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post(
                ROUTE,
                json={"prompt": "a bird", "first_frame_path": "first.png"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("空闲显存不足", response.json()["detail"])
        lock.assert_not_called()

    def test_vram_is_rechecked_after_shared_gpu_lock(self) -> None:
        self.upload_image("first.png")
        with (
            patch.object(self.main, "image_info", return_value=(1280, 720)),
            patch.object(
                self.main,
                "query_gpu_memory",
                side_effect=[{"free_mib": 14_000}, {"free_mib": 12_999}],
            ),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker") as run_worker,
        ):
            response = self.client.post(
                ROUTE,
                json={"prompt": "A bird flies over a forest", "first_frame_path": "first.png"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("空闲显存不足", response.json()["detail"])
        run_worker.assert_not_called()

    def test_gpu_memory_probe_failure_after_lock_fails_closed(self) -> None:
        self.upload_image("first.png")
        with (
            patch.object(self.main, "image_info", return_value=(1280, 720)),
            patch.object(
                self.main,
                "query_gpu_memory",
                side_effect=[{"free_mib": 14_000}, RuntimeError("nvidia-smi unavailable")],
            ),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker") as run_worker,
        ):
            response = self.client.post(
                ROUTE,
                json={"prompt": "A bird flies over a forest", "first_frame_path": "first.png"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "nvidia-smi unavailable")
        run_worker.assert_not_called()

    def test_generation_builds_bounded_fl2va_payload_and_returns_mp4(self) -> None:
        self.upload_image("first.png")
        output_path = self.root / "h3.mp4"
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42test")
        with (
            patch.object(self.main, "image_info", return_value=(1280, 720)),
            patch.object(self.main, "query_gpu_memory", return_value={"free_mib": 14_000}),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker", return_value=output_path) as run_worker,
            patch.object(self.main, "wait_after_cuda_release") as wait_after_release,
        ):
            response = self.client.post(
                ROUTE,
                json={
                    "prompt": "A bird flies over a forest",
                    "first_frame_path": "first.png",
                    "seed": 17,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        payload = run_worker.call_args.args[0]
        self.assertEqual(payload["mode"], "fl2va")
        self.assertEqual(payload["frame_count"], 124)
        self.assertEqual((payload["width"], payload["height"]), (960, 544))
        self.assertEqual(payload["seed"], 17)
        self.assertFalse(payload["include_audio"])
        self.assertEqual(payload["dit_filename"], "minimax_h3_fl2va_dit_16g.safetensors")
        wait_after_release.assert_called_once()

    def test_generation_can_opt_in_native_audio(self) -> None:
        self.upload_image("first.png")
        output_path = self.root / "h3-with-audio.mp4"
        output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42test")
        with (
            patch.object(self.main, "image_info", return_value=(1280, 720)),
            patch.object(self.main, "query_gpu_memory", return_value={"free_mib": 14_000}),
            patch.object(self.main, "shutil_which", return_value="/usr/bin/ffprobe"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker", return_value=output_path) as run_worker,
            patch.object(self.main, "wait_after_cuda_release"),
        ):
            response = self.client.post(
                ROUTE,
                json={
                    "prompt": "A bird flies over a forest",
                    "first_frame_path": "first.png",
                    "include_audio": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(run_worker.call_args.args[0]["include_audio"])

    def test_ref2va_requires_reference_image_and_selects_matching_dit(self) -> None:
        self.upload_image("ref.png")
        with (
            patch.object(self.main, "image_info", return_value=(1024, 1024)),
            patch.object(self.main, "query_gpu_memory", return_value={"free_mib": 14_000}),
        ):
            payload = self.main.manager.build_worker_payload(
                self.main.MiniMaxH3Request(
                    mode="ref2va",
                    prompt="Use <Picture 1> as the identity reference",
                    reference_image_paths=["ref.png"],
                )
            )

        self.assertEqual(payload["dit_filename"], "minimax_h3_ref2va_dit_16g.safetensors")
        self.assertEqual(payload["image_paths"]["ref_0"], str(self.main.image_path("ref.png")))

    def test_worker_prompt_contains_h3_nodes_and_matching_assets(self) -> None:
        payload = {
            "mode": "fl2va",
            "prompt": "a bird flies",
            "seed": 123,
            "width": 960,
            "height": 544,
            "frame_count": 124,
            "frame_rate": 24,
            "steps": 20,
            "dit_filename": "minimax_h3_fl2va_dit_16g.safetensors",
            "first_frame_path": "/tmp/first.png",
            "last_frame_path": None,
        }
        graph = self.worker.build_api_prompt(payload, {"first": "first.png"}, "h3_test")

        self.assertEqual(graph["1"]["inputs"]["unet_name"], payload["dit_filename"])
        self.assertEqual(graph["2"]["inputs"]["clip_name"], "qwen3vl_text_encoder.safetensors")
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["5"]["inputs"]["length"], 124)
        self.assertEqual(graph["5"]["inputs"]["first_frame"], ["20", 0])
        self.assertEqual(graph["20"]["inputs"]["image"], "first.png")
        self.assertEqual(graph["17"]["class_type"], "SaveVideo")
        self.assertNotIn("4", graph)
        self.assertNotIn("15", graph)
        self.assertNotIn("audio", graph["16"]["inputs"])

        audio_graph = self.worker.build_api_prompt(
            {**payload, "include_audio": True}, {"first": "first.png"}, "h3_audio"
        )
        self.assertEqual(audio_graph["15"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(audio_graph["16"]["inputs"]["audio"], ["15", 0])
        self.assertEqual(audio_graph["15"]["inputs"]["vae"], ["4", 0])

        ref_payload = {
            **payload,
            "mode": "ref2va",
            "dit_filename": "minimax_h3_ref2va_dit_16g.safetensors",
            "reference_image_paths": ["/tmp/ref.png"],
        }
        ref_graph = self.worker.build_api_prompt(ref_payload, {"ref_0": "ref.png"}, "h3_ref")
        self.assertEqual(ref_graph["5"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(ref_graph["5"]["inputs"]["ref_images.ref_image_0"], ["20", 0])
        self.assertNotIn("audio_vae", ref_graph["5"]["inputs"])

        ref_audio_graph = self.worker.build_api_prompt(
            {**ref_payload, "include_audio": True}, {"ref_0": "ref.png"}, "h3_ref_audio"
        )
        self.assertEqual(ref_audio_graph["5"]["inputs"]["audio_vae"], ["4", 0])

    def test_video_validation_requires_exact_audio_track_state(self) -> None:
        video_path = self.root / "metadata-only.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

        def probe_result(with_audio: bool):
            streams = [{"codec_type": "video"}]
            if with_audio:
                streams.append({"codec_type": "audio"})
            return SimpleNamespace(stdout=json.dumps({"streams": streams}))

        with patch.object(self.runtime.subprocess, "run", return_value=probe_result(False)):
            self.runtime.validate_video(video_path, "/usr/bin/ffprobe", expect_audio=False)
            with self.assertRaisesRegex(RuntimeError, "请求启用原生音频"):
                self.runtime.validate_video(video_path, "/usr/bin/ffprobe", expect_audio=True)

        with patch.object(self.runtime.subprocess, "run", return_value=probe_result(True)):
            self.runtime.validate_video(video_path, "/usr/bin/ffprobe", expect_audio=True)
            with self.assertRaisesRegex(RuntimeError, "请求关闭原生音频"):
                self.runtime.validate_video(video_path, "/usr/bin/ffprobe", expect_audio=False)

    def test_worker_manager_cleans_request_files_after_success(self) -> None:
        captured_paths: list[Path] = []
        output_video = self.root / "persisted.mp4"

        class FakeProcess:
            pid = 999_999
            returncode = 0

            @staticmethod
            def communicate(timeout: float):
                return "worker done", ""

            @staticmethod
            def poll():
                return 0

        def start_worker(command, **kwargs):
            request_path = Path(command[command.index("--input-json") + 1])
            output_path = Path(command[command.index("--output-video") + 1])
            captured_paths.extend((request_path, output_path))
            output_path.write_bytes(b"temporary-video")
            return FakeProcess()

        with (
            patch.object(self.main.subprocess, "Popen", side_effect=start_worker),
            patch.object(self.main, "validate_video") as validate_output,
            patch.object(self.main, "persist_video", return_value=output_video),
            patch.object(self.main, "terminate_process_group"),
        ):
            result = self.main.manager.run_worker(
                {"ffprobe_bin": "/usr/bin/ffprobe", "include_audio": False}
            )

        self.assertEqual(result, output_video)
        validate_output.assert_called_once_with(
            captured_paths[1], "/usr/bin/ffprobe", expect_audio=False
        )
        self.assertTrue(captured_paths)
        self.assertTrue(all(not path.exists() for path in captured_paths))


if __name__ == "__main__":
    unittest.main()
