"""LongCat Avatar WanGP 接口的无模型合同回归。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
