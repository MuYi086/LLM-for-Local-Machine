"""LongCat Avatar WanGP 接口的无模型合同回归。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-video/LongCat-Video-Avatar-1.5-Quanto-Int8"
ROUTE = "/v1/meituan/LongCat-Video-Avatar-1.5-Quanto-Int8"


def load_module(name: str, path: Path):
    """按指定名称从文件加载模块，供服务合同测试使用。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LongCatAvatarWanGPContractTests(unittest.TestCase):
    """验证请求必须携带 WanGP LongCat Avatar 所需参考图。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="longcat-wangp-contract-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        environment = {
            "STORAGE_DIR": str(root / "storage"),
            "RUNTIME_CACHE_DIR": str(root / "cache"),
            "GPU_LOCK_FILE": str(root / "cache/gpu.lock"),
            "LONGCAT_AVATAR_MODEL_DIR": str(root / "model"),
            "LONGCAT_AVATAR_BASE_MODEL_DIR": str(root / "base_model"),
            "LONGCAT_AVATAR_WANGP_DIR": str(root / "WanGP"),
            "LONGCAT_AVATAR_WANGP_PYTHON": str(root / "WanGP/.venv/bin/python"),
            "LONGCAT_AVATAR_AUDIO_DIR": str(root / "storage/audio"),
            "LONGCAT_AVATAR_IMAGE_DIR": str(root / "storage/images"),
            "LONGCAT_AVATAR_OUTPUT_DIR": str(root / "storage/output"),
            "LONGCAT_AVATAR_WORKER_TMP_DIR": str(root / "cache/worker"),
            "LONGCAT_AVATAR_CUDA_VISIBLE_DEVICES": "0",
            "LONGCAT_AVATAR_MEMORY_PROFILE": "4",
            "CUDA_RELEASE_DELAY": "0",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            cls.runtime = load_module("longcat_runtime", SERVICE_DIR / "longcat_runtime.py")
            cls.wangp_runtime = load_module("wangp_runtime", SERVICE_DIR / "wangp_runtime.py")
            cls.main = load_module("longcat_avatar_wangp_main_tests", SERVICE_DIR / "main.py")
            cls.worker = load_module("longcat_avatar_wangp_worker_tests", SERVICE_DIR / "worker.py")
        cls.client = TestClient(cls.main.app)

    def test_service_accepts_profile_zero_when_no_profile_override_is_set(self) -> None:
        temporary_wangp_dir = Path(self.temporary.name) / "profile-zero-WanGP"
        with patch.dict(
            os.environ,
            {"LONGCAT_AVATAR_WANGP_DIR": str(temporary_wangp_dir)},
        ):
            os.environ.pop("LONGCAT_AVATAR_MEMORY_PROFILE", None)
            try:
                main = load_module(
                    "longcat_avatar_profile_zero_main_tests", SERVICE_DIR / "main.py"
                )
            except ValueError as exc:
                self.fail(f"8390 should start with the default Profile 0: {exc}")

        self.assertEqual(main.MEMORY_PROFILE, 0)
        self.assertEqual(main.WANGP_CONFIG_FILE, temporary_wangp_dir / "wgp_config.json")
        main.WANGP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        main.WANGP_CONFIG_FILE.write_text(
            json.dumps({"profile": 0, "video_profile": 0}), encoding="utf-8"
        )
        self.assertTrue(main.wangp_config_status()["profile_matches_runtime"])

    def test_wangp_config_defaults_to_profile_zero_and_keeps_profile_four_fallback(self) -> None:
        config = self.main.WANGP_CONFIG_FILE
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("{}", encoding="utf-8")

        self.wangp_runtime.update_wangp_config(self.main.WANGP_DIR)
        self.assertEqual(json.loads(config.read_text(encoding="utf-8"))["profile"], 0)

        self.wangp_runtime.update_wangp_config(self.main.WANGP_DIR, memory_profile=4)
        self.assertEqual(json.loads(config.read_text(encoding="utf-8"))["profile"], 4)

    def test_wangp_config_rejects_unsupported_memory_profile(self) -> None:
        config = self.main.WANGP_CONFIG_FILE
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Profile 0 或 Profile 4"):
            self.wangp_runtime.update_wangp_config(self.main.WANGP_DIR, memory_profile=3)

    def test_missing_reference_image_is_rejected_before_gpu_queue(self) -> None:
        with (
            patch.object(self.main.manager, "build_worker_payload") as preflight,
            patch.object(self.main, "gpu_runtime_lock") as gpu_lock,
        ):
            response = self.client.post(
                ROUTE,
                json={"prompt": "A person speaking", "audio_path": "speech.wav"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("image_path", str(response.json()))
        preflight.assert_not_called()
        gpu_lock.assert_not_called()

    def test_worker_passes_uploaded_reference_image_to_wangp(self) -> None:
        class FakeSession:
            def get_default_settings(self, model_type: str) -> dict[str, object]:
                self.model_type = model_type
                return {}

        image_file = Path(self.temporary.name) / "portrait.png"
        settings = self.worker.build_settings(
            FakeSession(),
            {
                "prompt": "A person speaking",
                "audio_file": str(Path(self.temporary.name) / "speech.wav"),
                "image_file": str(image_file),
                "resolution": "832x480",
                "duration_seconds": 3.5,
            },
        )

        self.assertEqual(settings["image_refs"], [str(image_file.resolve())])

    def test_worker_passes_selected_profile_zero_to_wangp(self) -> None:
        generated = Path(self.temporary.name) / "generated.mp4"
        generated.write_bytes(b"fake mp4")
        output = Path(self.temporary.name) / "worker-output.mp4"

        class FakeJob:
            def result(self, timeout: float) -> SimpleNamespace:
                self.timeout = timeout
                return SimpleNamespace(success=True, errors=[], generated_files=[str(generated)])

        class FakeSession:
            def get_default_settings(self, model_type: str) -> dict[str, object]:
                return {}

            def submit_task(self, settings: dict[str, object]) -> FakeJob:
                return FakeJob()

        fake_api = types.ModuleType("shared.api")
        fake_api.init = unittest.mock.Mock(return_value=FakeSession())
        fake_shared = types.ModuleType("shared")
        fake_shared.__path__ = []
        payload = {
            "wangp_dir": str(Path(self.temporary.name) / "WanGP"),
            "workspace_dir": str(Path(self.temporary.name)),
            "output_video": str(output),
            "audio_file": str(Path(self.temporary.name) / "speech.wav"),
            "image_file": str(Path(self.temporary.name) / "portrait.png"),
            "prompt": "A person speaking",
            "resolution": "832x480",
            "duration_seconds": 3.5,
            "request_timeout": 30,
            "memory_profile": 0,
        }

        with patch.dict(sys.modules, {"shared": fake_shared, "shared.api": fake_api}):
            self.worker.run(payload)

        self.assertEqual(
            fake_api.init.call_args.kwargs["cli_args"],
            ["--offline", "--profile", "0", "--attention", "sdpa", "--gpu", "cuda:0"],
        )
        self.assertEqual(output.read_bytes(), b"fake mp4")

    def test_worker_records_wangp_phase_timing_when_diagnostics_are_enabled(self) -> None:
        generated = Path(self.temporary.name) / "generated-with-trace.mp4"
        generated.write_bytes(b"fake mp4")
        output = Path(self.temporary.name) / "worker-traced-output.mp4"
        trace_dir = Path(self.temporary.name) / "trace"

        class FakeJob:
            def result(self, timeout: float) -> SimpleNamespace:
                return SimpleNamespace(success=True, errors=[], generated_files=[str(generated)])

        class FakeSession:
            callbacks: object

            def get_default_settings(self, model_type: str) -> dict[str, object]:
                return {}

            def submit_task(self, settings: dict[str, object]) -> FakeJob:
                progress = SimpleNamespace(
                    phase="inference",
                    status="Denoising step 1/8",
                    progress=12,
                    current_step=1,
                    total_steps=8,
                    raw_phase="Denoising",
                    window_index=1,
                    window_count=1,
                )
                self.callbacks.on_progress(progress)
                return FakeJob()

        session = FakeSession()

        def initialize_session(**kwargs: object) -> FakeSession:
            session.callbacks = kwargs["callbacks"]
            return session

        fake_api = types.ModuleType("shared.api")
        fake_api.init = unittest.mock.Mock(side_effect=initialize_session)
        fake_shared = types.ModuleType("shared")
        fake_shared.__path__ = []
        payload = {
            "wangp_dir": str(Path(self.temporary.name) / "WanGP"),
            "workspace_dir": str(Path(self.temporary.name)),
            "output_video": str(output),
            "audio_file": str(Path(self.temporary.name) / "speech.wav"),
            "image_file": str(Path(self.temporary.name) / "portrait.png"),
            "prompt": "A person speaking",
            "resolution": "832x480",
            "duration_seconds": 3.5,
            "request_timeout": 30,
            "memory_profile": 0,
        }

        with (
            patch.dict(sys.modules, {"shared": fake_shared, "shared.api": fake_api}),
            patch.dict(os.environ, {"LONGCAT_AVATAR_TRACE_DIR": str(trace_dir)}),
        ):
            self.worker.run(payload)

        trace_files = list(trace_dir.glob("*.jsonl"))
        self.assertEqual(len(trace_files), 1)
        events = [json.loads(line) for line in trace_files[0].read_text().splitlines()]
        stages = {
            event["stage"]
            for event in events
            if event["event"] == "stage_end"
        }
        self.assertTrue({"wan_session_init", "generation_wait", "output_copy"} <= stages)
        progress = [event for event in events if event["event"] == "progress"]
        self.assertEqual(progress[0]["phase"], "inference")
        self.assertEqual(progress[0]["current_step"], 1)


if __name__ == "__main__":
    unittest.main()
