"""SheetSage2 服务的无模型路由、worker 与存储回归测试。"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "score_sheetsage-2"
TEST_ROOT = Path()


def load_service(name: str, path: Path):
    """用临时运行目录隔离加载不含模型依赖的 API 模块。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    runtime_spec = importlib.util.spec_from_file_location(
        f"{name}_runtime", path.with_name("runtime.py")
    )
    assert runtime_spec and runtime_spec.loader
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    service_environment = {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "SHEETSAGE2_MODEL_DIR": str(TEST_ROOT / "models" / "SheetSage2"),
        "SHEETSAGE2_BASE_MODEL_DIR": str(TEST_ROOT / "models" / "MERT-v2-FullSong"),
        "SHEETSAGE2_OUTPUT_DIR": str(TEST_ROOT / "storage" / "sheetsage2"),
        "SHEETSAGE2_WORKER_TMP_DIR": str(TEST_ROOT / "worker"),
        "SHEETSAGE2_UPLOAD_STAGING_DIR": str(TEST_ROOT / "uploads"),
        "LOCAL_FILES_ONLY": "1",
        "GPU_LOCK_WAIT_TIMEOUT": "1",
        "CUDA_RELEASE_DELAY": "0",
        "SHEETSAGE2_REQUEST_TIMEOUT": "5",
    }
    source_path = [str(path.parent), str(REPOSITORY_DIR / "llm_local_runtime/src"), *sys.path]
    with (
        patch.dict(os.environ, service_environment, clear=True),
        patch.object(sys, "path", source_path),
        patch.dict(sys.modules, {"runtime": runtime_module}),
    ):
        runtime_spec.loader.exec_module(runtime_module)
        spec.loader.exec_module(module)
    return module


def load_worker(name: str):
    """加载只在任务执行时导入 Torch 的 worker 模块。"""
    path = SERVICE_DIR / "worker.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SheetSage2ServiceTests(unittest.TestCase):
    """验证原生请求映射、隔离 worker 生命周期和结果归档。"""

    def setUp(self) -> None:
        global TEST_ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix="sheetsage2-tests-")
        self.addCleanup(self.temporary.cleanup)
        TEST_ROOT = Path(self.temporary.name)
        self.main = load_service("sheetsage2_service_test_main", SERVICE_DIR / "main.py")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)

    def complete_models(self) -> dict[str, dict[str, object]]:
        """构造通过文件预检的本地模型状态。"""
        return {
            "sheetsage2": {"complete": True, "missing": []},
            "mert_v2_fullsong": {"complete": True, "missing": []},
        }

    def test_request_checks_window_constraints_before_gpu_queue(self) -> None:
        request = self.main.SheetSage2Request(
            preset="paper",
            overlap_seconds=100,
            lookahead_seconds=0,
            prompts=["timestamp", "key"],
        )
        self.assertEqual(request.preset, "paper")
        self.assertEqual(request.prompts, ["timestamp", "key"])

        with patch.object(self.main, "gpu_runtime_lock") as gpu_lock:
            response = self.client.post(
                "/v1/m-a-p/SheetSage2",
                files={"audio": ("song.mp3", b"audio", "audio/mpeg")},
                data={
                    "preset": "default",
                    "overlap_seconds": "80",
                    "lookahead_seconds": "90",
                },
            )
        self.assertEqual(response.status_code, 422, response.text)
        gpu_lock.assert_not_called()

        non_finite_response = self.client.post(
            "/v1/m-a-p/SheetSage2",
            files={"audio": ("song.mp3", b"audio", "audio/mpeg")},
            data={"max_seconds": "NaN"},
        )
        self.assertEqual(non_finite_response.status_code, 422, non_finite_response.text)

    def test_payload_forwards_native_options_and_both_local_model_paths(self) -> None:
        audio_path = TEST_ROOT / "song.mp3"
        audio_path.write_bytes(b"audio")
        request = self.main.SheetSage2Request(
            dtype="fp32",
            preset="default",
            max_seconds=120,
            overlap_seconds=40,
            lookahead_seconds=10,
            prompts=["timestamp", "melody_vocal"],
            melody_only=True,
            export_embeddings=True,
        )
        with (
            patch.object(self.main, "model_status", return_value=self.complete_models()),
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main.shutil, "which", return_value="/usr/bin/ffmpeg"),
        ):
            payload = self.main.manager.build_worker_payload(request, audio_path)

        self.assertEqual(payload["audio_path"], str(audio_path))
        self.assertEqual(payload["model_dir"], str(TEST_ROOT / "models/SheetSage2"))
        self.assertEqual(payload["base_model_dir"], str(TEST_ROOT / "models/MERT-v2-FullSong"))
        self.assertEqual(payload["dtype"], "fp32")
        self.assertEqual(payload["overlap_seconds"], 40)
        self.assertEqual(payload["lookahead_seconds"], 10)
        self.assertEqual(payload["prompts"], ["timestamp", "melody_vocal"])
        self.assertTrue(payload["melody_only"])
        self.assertTrue(payload["export_embeddings"])
        self.assertTrue(payload["local_files_only"])

    def test_successful_route_returns_and_persists_native_result_zip(self) -> None:
        archive = TEST_ROOT / "worker" / "result.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as result_zip:
            result_zip.writestr("score.abc", "X:1\nK:C\nC D E")
            result_zip.writestr("transcription.mid", b"MThd")

        with (
            patch.object(self.main, "model_status", return_value=self.complete_models()),
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "gpu_runtime_lock", return_value=nullcontext()) as gpu_lock,
            patch.object(self.main.manager, "run_worker", return_value=archive),
            patch.object(self.main, "wait_after_cuda_release") as release,
        ):
            response = self.client.post(
                "/v1/m-a-p/SheetSage2",
                files={"audio": ("song.mp3", b"input audio", "audio/mpeg")},
                data={"dtype": "bf16", "preset": "default"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertEqual(response.headers["x-sheetsage2-model"], "m-a-p/SheetSage2")
        self.assertEqual(response.headers["x-sheetsage2-preset"], "default")
        self.assertEqual(len(response.headers["x-sheetsage2-audio-sha256"]), 64)
        with zipfile.ZipFile(io.BytesIO(response.content)) as result_zip:
            self.assertEqual(result_zip.read("score.abc"), b"X:1\nK:C\nC D E")
            self.assertEqual(result_zip.read("transcription.mid"), b"MThd")

        saved_archives = list(self.main.OUTPUT_DIR.glob("sheetsage2_*.zip"))
        self.assertEqual(len(saved_archives), 1)
        self.assertFalse(archive.exists())
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])
        gpu_lock.assert_called_once_with("sheetsage2/transcribe")
        release.assert_called_once()

    def test_missing_models_return_503_without_acquiring_gpu(self) -> None:
        incomplete = {
            "sheetsage2": {"complete": False, "missing": ["model.safetensors"]},
            "mert_v2_fullsong": {"complete": True, "missing": []},
        }
        with (
            patch.object(self.main, "model_status", return_value=incomplete),
            patch.object(self.main.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(self.main, "gpu_runtime_lock") as gpu_lock,
        ):
            response = self.client.post(
                "/v1/m-a-p/SheetSage2",
                files={"audio": ("song.mp3", b"audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("sheetsage2", response.json()["detail"])
        gpu_lock.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_missing_pkg_resources_is_reported_before_starting_worker(self) -> None:
        """pretty_midi 可被找到但传递依赖缺失时，应在健康检查与 CPU 预检中阻止推理。"""
        with (
            patch.object(self.main, "model_status", return_value=self.complete_models()),
            patch.object(
                self.main, "module_available", side_effect=lambda name: name != "pkg_resources"
            ),
            patch.object(self.main.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "gpu_runtime_lock") as gpu_lock,
            patch.object(self.main.manager, "run_worker") as run_worker,
        ):
            health = self.client.get("/v1/health")
            response = self.client.post(
                "/v1/m-a-p/SheetSage2",
                files={"audio": ("song.mp3", b"audio", "audio/mpeg")},
            )

        self.assertFalse(health.json()["ready"])
        self.assertTrue(health.json()["available"]["dependencies"]["pretty_midi"])
        self.assertFalse(health.json()["available"]["dependencies"]["pkg_resources"])
        self.assertEqual(response.status_code, 503)
        self.assertIn("pkg_resources", response.json()["detail"])
        gpu_lock.assert_not_called()
        run_worker.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_worker_cleans_request_and_archive_files_on_success_and_timeout(self) -> None:
        process = Mock(returncode=0)
        popen = Mock(return_value=process)

        def write_worker_archive(timeout: float) -> tuple[str, str]:
            command = popen.call_args.args[0]
            request_path = Path(command[command.index("--input-json") + 1])
            archive_path = Path(command[command.index("--output-zip") + 1])
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["audio_path"], "/tmp/song.mp3")
            archive_path.write_bytes(b"PK-native")
            return "", ""

        process.communicate.side_effect = write_worker_archive
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            archive_path = self.main.manager.run_worker(
                {"audio_path": "/tmp/song.mp3", "local_files_only": True}
            )

        self.assertEqual(archive_path.read_bytes(), b"PK-native")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        terminate.assert_called_once_with(process, "SheetSage2")
        archive_path.unlink()
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

        timeout_process = Mock(returncode=None)
        timeout_process.communicate.side_effect = subprocess.TimeoutExpired("worker", 5)
        with (
            patch.object(self.main.subprocess, "Popen", return_value=timeout_process),
            patch.object(self.main, "terminate_process_group") as terminate_timeout,
            self.assertRaises(TimeoutError),
        ):
            self.main.manager.run_worker({"audio_path": "/tmp/song.mp3"})
        terminate_timeout.assert_called_once_with(timeout_process, "SheetSage2")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_worker_calls_official_transcribe_and_archives_generated_files(self) -> None:
        worker = load_worker("sheetsage2_test_worker")
        fake_torch = types.SimpleNamespace(cuda=Mock())
        fake_torch.cuda.is_available.return_value = False
        model = Mock()

        def create_score(audio_path: str, output_dir: Path, **options: object) -> None:
            self.assertEqual(audio_path, "/tmp/audio.mp3")
            self.assertEqual(options["dtype"], "fp32")
            (output_dir / "score.abc").write_text("K:C\nC D", encoding="utf-8")

        model.transcribe.side_effect = create_score
        archive_path = TEST_ROOT / "worker" / "score.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "audio_path": "/tmp/audio.mp3",
            "base_model_dir": "/models/MERT-v2-FullSong",
            "device": "cuda:0",
            "dtype": "fp32",
            "local_files_only": True,
            "model_dir": "/models/SheetSage2",
            "preset": "default",
            "worker_tmp_dir": str(TEST_ROOT / "worker"),
        }
        with patch.object(worker, "load_model", return_value=(fake_torch, model)):
            worker.transcribe(payload, archive_path)

        model.transcribe.assert_called_once()
        with zipfile.ZipFile(archive_path) as result_zip:
            self.assertEqual(result_zip.read("score.abc"), b"K:C\nC D")
        self.assertEqual(list((TEST_ROOT / "worker").glob("sheetsage2_result_*")), [])

    def test_model_status_requires_weight_code_and_pinned_parent_files(self) -> None:
        model_dir = Path(self.main.MODEL_DIR)
        base_dir = Path(self.main.BASE_MODEL_DIR)
        model_dir.mkdir(parents=True)
        base_dir.mkdir(parents=True)
        for filename in self.main.REQUIRED_MODEL_FILES:
            (model_dir / filename).touch()
        for filename in self.main.REQUIRED_BASE_MODEL_FILES:
            (base_dir / filename).touch()

        status = self.main.model_status()

        self.assertTrue(status["sheetsage2"]["complete"])
        self.assertTrue(status["mert_v2_fullsong"]["complete"])

    def test_startup_quality_gate_and_docs_register_both_m_a_p_services(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        readme = (REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")
        agent_guide = (REPOSITORY_DIR / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn('SHEETSAGE2_PORT="${SHEETSAGE2_PORT:-8315}"', startup)
        self.assertIn('"$SHEETSAGE2_PROJECT_DIR"', startup)
        self.assertIn("'/v1/m-a-p/SheetSage2'", startup)
        self.assertIn("'/v1/m-a-p/YuE2'", startup)
        self.assertIn("model-audio/music_yue2-3b", quality_gate)
        self.assertIn("model-audio/score_sheetsage-2", quality_gate)
        self.assertIn("/v1/m-a-p/SheetSage2", readme)
        self.assertIn("/v1/m-a-p/YuE2", readme)
        self.assertIn("model-audio/score_sheetsage-2/` | 8315", agent_guide)


if __name__ == "__main__":
    unittest.main()
