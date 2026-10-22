"""8300 控制面服务的无模型回归测试。"""

from __future__ import annotations

# 控制面测试确保 8300 只做存储、健康检查和 CPU 导出，不加载模型。
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "main"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="main-control-plane-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
CONTROL_ENV = {
    "STORAGE_DIR": str(TEST_ROOT / "storage"),
    "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage/timbre"),
    "CLONE_STORAGE_DIR": str(TEST_ROOT / "storage/clone"),
    "PROMPTS_DIR": str(TEST_ROOT / "storage/clone"),
    "RUNTIME_CACHE_DIR": str(TEST_ROOT / "storage/.cache/runtime"),
    "STEAM_AUDIO_RENDER_CACHE_DIR": str(TEST_ROOT / "storage/.cache/runtime/steam_audio"),
    "STEAM_AUDIO_RENDERER_BIN": str(TEST_ROOT / "bin/steam-audio-render"),
    "GPU_LOCK_FILE": str(TEST_ROOT / "storage/.cache/runtime/gpu-runtime.lock"),
    "HOST": "127.0.0.1",
    "PORT": "8300",
}
ORIGINAL_ENV = {key: os.environ.get(key) for key in CONTROL_ENV}
os.environ.update(CONTROL_ENV)
sys.path.insert(0, str(SERVICE_DIR))
sys.modules.pop("gpu_runtime", None)
spec = importlib.util.spec_from_file_location(
    "control_plane_main_for_test", SERVICE_DIR / "main.py"
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)
for key, value in ORIGINAL_ENV.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class MainControlPlaneMigrationTests(unittest.TestCase):
    def test_control_plane_routes_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("GET", "/v1/control"),
            ("POST", "/v1/audio/export"),
            ("POST", "/v1/audio/spatial/render"),
            ("GET", "/v1/audio/spatial/render/progress/{job_id}"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))
        self.assertNotIn(("POST", "/v1/mimo/timbre"), actual_routes)

        response = TestClient(main.app).get("/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runtime"]["service_role"], "control_plane")
        self.assertEqual(
            payload["runtime"]["model_inference"],
            "delegated to standalone services",
        )
        self.assertEqual(payload["audio_export"]["sample_rate"], 48000)
        self.assertEqual(payload["audio_export"]["profiles"], ["standard", "balanced", "immersive"])
        self.assertEqual(payload["spatial_renderer"]["manifest_version"], "1.0")
        self.assertFalse(payload["spatial_renderer"]["gpu_lock"])
        self.assertEqual(payload["spatial_renderer"]["max_manifest_bytes"], 8 * 1024 * 1024)

    def test_upload_and_check_preserve_timbre_reference_without_clone_copy(self) -> None:
        from fastapi.testclient import TestClient

        timbre_path = Path(main.TIMBRE_STORAGE_DIR) / "designed.wav"
        timbre_path.write_bytes(b"RIFF-designed-voice")
        client = TestClient(main.app)

        response = client.post(
            "/v1/upload_audio",
            files={"audio": ("designed.wav", timbre_path.read_bytes(), "audio/wav")},
            data={"full_path": "designed.wav"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse((Path(main.PROMPTS_DIR) / main.hash_filename("designed.wav")).exists())
        self.assertTrue(
            (
                Path(main.TIMBRE_REFERENCE_DIR) / f"{main.hash_filename('designed.wav')}.json"
            ).exists()
        )
        checked = client.get("/v1/check/audio", params={"file_name": "designed.wav"})
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["exists"])

    def test_audio_export_streams_processed_result_and_cleans_response_file(self) -> None:
        from fastapi.testclient import TestClient

        output_dir = TEST_ROOT / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.mp3"
        output_path.write_bytes(b"processed-mp3")
        cleanup_called: list[bool] = []

        def fake_export(staged, *, profile: str, output_format: str):
            self.assertEqual(profile, "balanced")
            self.assertEqual(output_format, "mp3")
            self.assertTrue(staged.path.is_file())
            staged.path.unlink()
            return SimpleNamespace(
                path=output_path,
                media_type="audio/mpeg",
                download_name="llm_local_balanced.mp3",
                cleanup=lambda: cleanup_called.append(True),
            )

        with patch.object(
            main.spatial_audio_processor, "export", side_effect=fake_export
        ) as export:
            response = TestClient(main.app).post(
                "/v1/audio/export",
                files={"audio": ("mix.wav", b"RIFF-mixed-audio", "audio/wav")},
                data={"profile": "balanced", "output_format": "mp3"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"processed-mp3")
        self.assertEqual(response.headers["x-audio-sample-rate"], "48000")
        self.assertEqual(response.headers["x-audio-profile"], "balanced")
        self.assertIn("llm_local_balanced.mp3", response.headers["content-disposition"])
        self.assertIn(
            "Content-Disposition",
            response.headers["access-control-expose-headers"],
        )
        self.assertTrue(cleanup_called)
        export.assert_called_once()

    def test_audio_export_rejects_unknown_profile_before_processing(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(main.spatial_audio_processor, "export") as export:
            response = TestClient(main.app).post(
                "/v1/audio/export",
                files={"audio": ("mix.wav", b"RIFF-mixed-audio", "audio/wav")},
                data={"profile": "unknown", "output_format": "wav"},
            )

        self.assertEqual(response.status_code, 422)
        export.assert_not_called()

    def test_steam_audio_render_accepts_object_assets_and_exposes_engine_headers(self) -> None:
        from fastapi.testclient import TestClient

        output_dir = TEST_ROOT / "steam-processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.wav"
        output_path.write_bytes(b"steam-wav")
        cleanup_called: list[bool] = []
        manifest = {
            "version": "1.0",
            "sample_rate": 48000,
            "timeline_duration_ms": 2000,
            "scene": {"acoustic_quality": "balanced"},
            "sources": [
                {
                    "id": "line_1",
                    "kind": "dialogue",
                    "asset_id": "line_audio_1",
                    "asset_filename": "line.wav",
                    "start_ms": 0,
                    "duration_ms": 1500,
                    "gain_db": -2,
                    "spatial": {
                        "mode": "point",
                        "location": "front_left",
                        "distance": "conversational",
                        "movement": "static",
                        "occlusion": "none",
                        "spatial_blend": "medium",
                    },
                }
            ],
        }

        def fake_render(
            render_manifest,
            staged_assets,
            *,
            profile: str,
            output_format: str,
            progress_callback,
        ):
            self.assertEqual(render_manifest.version, "1.0")
            self.assertEqual(set(staged_assets), {"line.wav"})
            self.assertEqual(profile, "balanced")
            self.assertEqual(output_format, "wav")
            progress_callback("rendering", 75, "测试渲染中")
            for staged in staged_assets.values():
                self.assertTrue(staged.path.is_file())
                staged.path.unlink()
            return SimpleNamespace(
                path=output_path,
                media_type="audio/wav",
                download_name="llm_local_steam_balanced.wav",
                cleanup=lambda: cleanup_called.append(True),
            )

        with patch.object(main.steam_audio_processor, "render", side_effect=fake_render) as render:
            response = TestClient(main.app).post(
                "/v1/audio/spatial/render",
                files=[("assets", ("line.wav", b"RIFF-line", "audio/wav"))],
                data={
                    "manifest": json.dumps(manifest),
                    "profile": "balanced",
                    "output_format": "wav",
                    "job_id": "test-job-12345678",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"steam-wav")
        self.assertEqual(response.headers["x-spatial-engine"], "steam-audio")
        self.assertEqual(response.headers["x-spatial-manifest-version"], "1.0")
        self.assertEqual(response.headers["x-spatial-job-id"], "test-job-12345678")
        progress = TestClient(main.app).get("/v1/audio/spatial/render/progress/test-job-12345678")
        self.assertEqual(progress.status_code, 200)
        self.assertEqual(progress.json()["state"], "succeeded")
        self.assertEqual(progress.json()["progress"], 100)
        self.assertTrue(cleanup_called)
        render.assert_called_once()

    def test_steam_audio_render_rejects_missing_and_unreferenced_assets_before_staging(
        self,
    ) -> None:
        from fastapi.testclient import TestClient

        manifest = {
            "version": "1.0",
            "sample_rate": 48000,
            "timeline_duration_ms": 1000,
            "sources": [
                {
                    "id": "sfx_1",
                    "kind": "sfx",
                    "asset_id": "asset_1",
                    "asset_filename": "expected.wav",
                    "start_ms": 0,
                    "duration_ms": 500,
                    "spatial": {},
                }
            ],
        }
        with patch.object(main.steam_audio_processor, "render") as render:
            response = TestClient(main.app).post(
                "/v1/audio/spatial/render",
                files=[("assets", ("unused.wav", b"RIFF-unused", "audio/wav"))],
                data={"manifest": json.dumps(manifest)},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["missing"], ["expected.wav"])
        self.assertEqual(response.json()["unused"], ["unused.wav"])
        render.assert_not_called()

    def test_api_directory_is_removed_after_migration(self) -> None:
        self.assertFalse((REPOSITORY_DIR / "api").exists())
        self.assertTrue((SERVICE_DIR / "main.py").exists())
        self.assertTrue((SERVICE_DIR / "gpu_runtime.py").exists())


if __name__ == "__main__":
    unittest.main()
