"""验证控制面只暴露仍维护的音频处理路由。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
_TEST_STORAGE = tempfile.TemporaryDirectory()
os.environ["STORAGE_DIR"] = _TEST_STORAGE.name
os.environ["RUNTIME_CACHE_DIR"] = str(Path(_TEST_STORAGE.name) / ".cache/runtime")
sys.path[:0] = [str(ROOT / "main"), str(ROOT / "llm_local_runtime/src")]

_SPEC = importlib.util.spec_from_file_location("control_plane_entry", ROOT / "main/main.py")
assert _SPEC and _SPEC.loader
_CONTROL_PLANE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTROL_PLANE
_SPEC.loader.exec_module(_CONTROL_PLANE)


class ControlPlaneContractTests(unittest.TestCase):
    def test_clone_reference_routes_are_not_available(self) -> None:
        client = TestClient(_CONTROL_PLANE.app)
        self.addCleanup(client.close)

        self.assertEqual(client.post("/v1/upload_audio").status_code, 404)
        self.assertEqual(client.get("/v1/check/audio?file_name=voice.wav").status_code, 404)

    def test_active_export_and_spatial_routes_remain_registered(self) -> None:
        paths = {route.path for route in _CONTROL_PLANE.app.routes}

        self.assertIn("/v1/audio/export", paths)
        self.assertIn("/v1/audio/spatial/render", paths)
        self.assertIn("/v1/audio/spatial/render/progress/{job_id}", paths)
        self.assertNotIn("/v1/upload_audio", paths)
        self.assertNotIn("/v1/check/audio", paths)


if __name__ == "__main__":
    unittest.main()
