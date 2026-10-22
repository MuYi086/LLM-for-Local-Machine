"""TIGER-DnR 独立服务的无模型回归测试。"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "TIGER-DnR"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="tiger-dnr-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "hf-mirror" / "JusperLee" / "TIGER-DnR"
SOURCE_DIR = TEST_ROOT / "tiger-source"

for directory in (
    MODEL_DIR,
    SOURCE_DIR / "look2hear" / "models",
    SOURCE_DIR / "look2hear" / "layers",
):
    directory.mkdir(parents=True, exist_ok=True)
(SOURCE_DIR / "look2hear" / "models" / "base_model.py").write_text("# mocked upstream\n")
(SOURCE_DIR / "look2hear" / "models" / "tiger_dnr.py").write_text("# mocked upstream\n")
(SOURCE_DIR / "look2hear" / "layers" / "activations.py").write_text("# mocked upstream\n")
(SOURCE_DIR / "look2hear" / "layers" / "normalizations.py").write_text("# mocked upstream\n")
(MODEL_DIR / "config.json").write_text("{}", encoding="utf-8")
(MODEL_DIR / "model.safetensors").write_bytes(b"test-weights")

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "TIGER_DNR_MODEL_DIR": str(MODEL_DIR),
        "TIGER_DNR_SOURCE_DIR": str(SOURCE_DIR),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIGER_DNR_OUTPUT_DIR": str(TEST_ROOT / "storage" / "separation"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "TIGER_DNR_REQUEST_TIMEOUT": "5",
    }
)
sys.path.insert(0, str(SERVICE_DIR))

spec = importlib.util.spec_from_file_location(
    "tiger_dnr_service_main",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class TigerDnrMigrationTests(unittest.TestCase):
    """验证 HTTP 编排不加载模型，并保留三 Stem 的上游语义。"""

    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/tigerDnr/separate"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        with patch.object(main, "cuda_status", return_value={"available": False, "source": "test"}):
            response = TestClient(main.app).get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {"code", "paths", "available", "cuda", "runtime", "last_errors"},
        )
        self.assertTrue(payload["available"]["worker_script"])
        self.assertTrue(payload["available"]["tiger_dnr_model_dir"])
        self.assertTrue(payload["available"]["tiger_dnr_source_dir"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["model_sample_rate"], 44_100)
        self.assertEqual(payload["runtime"]["stems"], ["dialog", "effects", "music"])

    def test_separate_route_returns_three_stem_archive_without_loading_model(self) -> None:
        from fastapi.testclient import TestClient

        worker_output_dir = TEST_ROOT / "fake-worker-output"
        worker_output_dir.mkdir(exist_ok=True)
        outputs: dict[str, Path] = {}
        for stem in main.STEM_NAMES:
            output_path = worker_output_dir / f"{stem}.wav"
            output_path.write_bytes(b"RIFF" + stem.encode("ascii") + b"\0" * 40)
            outputs[stem] = output_path

        with patch.object(main.manager, "run_worker", return_value=outputs) as run_worker:
            response = TestClient(main.app).post(
                "/v1/tigerDnr/separate",
                files={"audio": ("mixture.wav", b"RIFF" + b"\0" * 40, "audio/wav")},
                data={"device": "cpu"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn("attachment", response.headers["content-disposition"])
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["device"], "cpu")
        self.assertIsNotNone(main.manager.last_archive_path)
        archive_path = main.manager.last_archive_path
        assert archive_path is not None
        self.assertTrue(archive_path.is_file())
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(set(archive.namelist()), {"dialog.wav", "effects.wav", "music.wav"})
        saved_batches = list(main.TIGER_DNR_OUTPUT_DIR.glob("tiger_dnr_*"))
        self.assertEqual(len(saved_batches), 1)
        self.assertTrue(
            all((saved_batches[0] / f"{stem}.wav").is_file() for stem in main.STEM_NAMES)
        )

    def test_separate_route_rejects_non_audio_upload(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(main.manager, "run_worker") as run_worker:
            response = TestClient(main.app).post(
                "/v1/tigerDnr/separate",
                files={"audio": ("notes.txt", b"not an audio file", "text/plain")},
            )

        self.assertEqual(response.status_code, 422)
        run_worker.assert_not_called()

    def test_worker_uses_uv_interpreter_and_cleans_temporary_files(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                command = captured["command"]
                output_dir = Path(command[command.index("--output-dir") + 1])
                for stem in main.STEM_NAMES:
                    (output_dir / f"{stem}.wav").write_bytes(b"RIFF" + b"\0" * 40)
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            outputs = main.manager.run_worker({"device": "cpu"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertEqual(set(outputs), set(main.STEM_NAMES))
        self.assertTrue(all(path.read_bytes().startswith(b"RIFF") for path in outputs.values()))
        main.manager.cleanup_worker_outputs(outputs)
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_worker_failure_and_timeout_clean_temporary_files(self) -> None:
        class FailedProcess:
            returncode = 1
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return "", "mock worker failure"

        with patch.object(main.subprocess, "Popen", return_value=FailedProcess()):
            with self.assertRaisesRegex(RuntimeError, "mock worker failure"):
                main.manager.run_worker({"device": "cpu"})
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

        class TimeoutProcess:
            returncode = None
            pid = None

            def poll(self):
                return None if self.returncode is None else self.returncode

            def communicate(self, timeout=None):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("worker", timeout)
                return "", ""

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                self.returncode = -15

        timeout_process = TimeoutProcess()
        with patch.object(main.subprocess, "Popen", return_value=timeout_process):
            with self.assertRaisesRegex(RuntimeError, "超时"):
                main.manager.run_worker({"device": "cpu"})
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
