"""MSST 服务合同、输出头选择、上传与进程生命周期的无模型回归。"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from llm_local_runtime import GpuLockTimeoutError

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "model-audio/separate_msst-mega53"


def load_module(name, filename):
    """按唯一模块名加载服务文件，防止污染其他服务的同名模块。"""
    spec = importlib.util.spec_from_file_location(name, SERVICE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MsstServiceTests(unittest.TestCase):
    """以临时目录和模拟 worker 验证对外行为，不导入 Torch。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = load_module("msst_contract_test", "contract.py")
        runtime = load_module("msst_runtime_test", "msst_runtime.py")
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "MSST_MODEL_DIR": str(self.root / "model"),
            "MSST_SOURCE_DIR": str(self.root / "source"),
            "CUDA_RELEASE_DELAY": "0",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
            "MSST_REQUEST_TIMEOUT": "1",
            "UPLOAD_MAX_BYTES": "1024",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, {"contract": self.contract, "msst_runtime": runtime}),
        ):
            self.main = load_module("msst_main_test", "main.py")
            self.worker = load_module("msst_worker_test", "worker.py")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)
        self.ready = {
            "ready": True,
            "supported_stems": ["guitar", "strings", "synth", "violin", "piano", "wind"],
        }

    def request(self, data=None, content=b"audio"):
        return self.client.post(
            "/v1/msst/separate",
            data=data or {"stems": "guitar,strings"},
            files={"audio": ("song.wav", content, "audio/wav")},
        )

    def test_caller_must_select_stems_and_can_change_count_and_instruments(self):
        with patch.object(self.main, "runtime_status", return_value=self.ready):
            missing = self.client.post(
                "/v1/msst/separate", files={"audio": ("song.wav", b"x", "audio/wav")}
            )
            self.assertEqual(missing.status_code, 422)
            self.assertEqual(self.request({"stems": "invalid"}).status_code, 422)
            self.assertEqual(self.request({"stems": "guitar,guitar"}).status_code, 422)
            self.assertEqual(self.request({"stems": "guitar,"}).status_code, 422)
            self.assertEqual(
                self.request({"stems": "guitar", "chunk_seconds": "3"}).status_code, 422
            )
        self.assertEqual(
            self.contract.parse_stems("violin", self.ready["supported_stems"]), ["violin"]
        )
        # 不把之前建议的四轨或特定乐器写死在服务层。
        self.assertEqual(
            self.contract.parse_stems(
                ",".join(self.ready["supported_stems"]), self.ready["supported_stems"]
            ),
            self.ready["supported_stems"],
        )

    def test_success_archives_expected_files_and_cleans_temporary_inputs(self):
        def fake_worker(payload, job_dir):
            self.assertEqual(payload["stems"], ["guitar", "strings"])
            for name in ("guitar.wav", "strings.wav", "residual.wav"):
                (job_dir / "result" / name).write_bytes(b"float-wave")
            (job_dir / "result/manifest.json").write_text('{"sample_rate":44100}')
            (job_dir / "result/request.json").write_text("private")

        with (
            patch.object(self.main, "runtime_status", return_value=self.ready),
            patch.object(
                self.main,
                "normalize_audio",
                side_effect=lambda src, dst: dst.write_bytes(b"normalized"),
            ),
            patch.object(self.main, "execute_worker", side_effect=fake_worker),
        ):
            response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertEqual(len(response.headers["x-msst-audio-sha256"]), 64)
        saved = list(self.main.OUTPUT_DIR.glob("*.zip"))
        self.assertEqual(len(saved), 1)
        with zipfile.ZipFile(saved[0]) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"guitar.wav", "strings.wav", "residual.wav", "manifest.json"},
            )
        self.assertEqual(list(self.main.UPLOAD_DIR.iterdir()), [])
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_preflight_and_upload_limits_prevent_worker_execution(self):
        with (
            patch.object(self.main, "runtime_status", return_value=self.ready),
            patch.object(self.main, "normalize_audio", side_effect=ValueError("时长超限")),
            patch.object(self.main, "execute_worker") as worker,
        ):
            self.assertEqual(self.request().status_code, 422)
            self.assertEqual(self.request(content=b"x" * 1025).status_code, 413)
            worker.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_DIR.iterdir()), [])
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_not_ready_timeout_and_queue_failure_return_correct_status(self):
        with patch.object(self.main, "runtime_status", return_value={"ready": False}):
            self.assertEqual(self.request().status_code, 503)
        for exception, status in [
            (TimeoutError("timeout"), 504),
            (GpuLockTimeoutError("busy", 1), 503),
            (RuntimeError("worker failure"), 500),
        ]:
            with (
                self.subTest(status=status),
                patch.object(self.main, "runtime_status", return_value=self.ready),
                patch.object(
                    self.main, "normalize_audio", side_effect=lambda src, dst: dst.write_bytes(b"n")
                ),
                patch.object(self.main, "execute_worker", side_effect=exception),
            ):
                self.assertEqual(self.request().status_code, status)
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])
        self.assertEqual(list(self.main.UPLOAD_DIR.iterdir()), [])

    def test_select_heads_preserves_requested_order_and_shared_backbone(self):
        heads = [object() for _ in range(6)]
        backbone = object()
        model = types.SimpleNamespace(mask_estimators=heads, layers=backbone)
        config = types.SimpleNamespace(
            training=types.SimpleNamespace(instruments=self.ready["supported_stems"])
        )
        self.worker.select_heads(model, config, ["wind", "guitar"], list)
        self.assertEqual(model.mask_estimators, [heads[5], heads[0]])
        self.assertIs(model.layers, backbone)
        self.assertEqual(model.num_stems, 2)
        self.assertEqual(config.training.instruments, ["wind", "guitar"])
        self.assertIsNone(config.training.target_instrument)

    def test_worker_cleanup_covers_success_failure_and_timeout_inside_gpu_lock(self):
        events = []

        @contextlib.contextmanager
        def lock(*args, **kwargs):
            events.append("lock")
            yield
            events.append("unlock")

        for code in (0, 1, "timeout"):
            events.clear()
            job = self.root / str(code)
            job.mkdir()
            process = Mock(pid=123)
            process.returncode = 0 if code == 0 else 1
            if code == "timeout":
                process.wait.side_effect = subprocess.TimeoutExpired("worker", 1)
            with (
                patch.object(self.main, "gpu_runtime_lock", side_effect=lock),
                patch.object(self.main.subprocess, "Popen", return_value=process) as popen,
                patch.object(self.main, "stop_worker", side_effect=lambda p: events.append("stop")),
                patch.object(self.main, "CUDA_RELEASE_DELAY", 1),
                patch.object(self.main.time, "sleep", side_effect=lambda t: events.append("delay")),
            ):
                if code == 0:
                    self.main.execute_worker({}, job)
                else:
                    with self.assertRaises(TimeoutError if code == "timeout" else RuntimeError):
                        self.main.execute_worker({}, job)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(popen.call_args.args[0][0], sys.executable)
            self.assertEqual(events[:3], ["lock", "stop", "delay"])
            if code == 0:
                self.assertEqual(events[-1], "unlock")

    def test_cleanup_signals_group_even_when_parent_has_exited(self):
        process = Mock(pid=123)
        process.poll.return_value = 0
        with patch.object(self.main.os, "killpg") as kill:
            self.main.stop_worker(process)
        self.assertEqual(kill.call_count, 2)

    def test_archive_rejects_symlinks_and_missing_tracks_without_publishing(self):
        result = self.root / "results"
        result.mkdir()
        (result / "guitar.wav").symlink_to(SERVICE / "worker.py")
        with self.assertRaises(RuntimeError):
            self.main.persist_archive(result, ["guitar"], "abc")
        self.assertEqual(list(self.main.OUTPUT_DIR.iterdir()), [])

    def test_health_reports_unavailable_configuration_without_model_import(self):
        model = self.main.MODEL_DIR
        model.mkdir()
        (model / self.contract.WEIGHT_FILENAME).write_bytes(b"weights")
        cfg = model / self.contract.CONFIG_FILENAME
        cfg.write_text(
            "model: {num_stems: 2}\ntraining: {instruments: [guitar, violin]}\naudio: {bands: !!python/tuple [2, 3]}\n"
        )
        with (
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main.shutil, "which", return_value="/tool"),
            patch.object(self.main, "REQUIRED_SOURCE_FILES", ()),
        ):
            response = self.client.get("/v1/health")
            self.assertTrue(response.json()["ready"])
            self.assertEqual(response.json()["supported_stems"], ["guitar", "violin"])
            cfg.write_text("model: {num_stems: 3}\ntraining: {instruments: [guitar]}\n")
            self.assertFalse(self.client.get("/v1/health").json()["ready"])

    def test_yaml_python_object_tags_are_rejected(self):
        import yaml

        config = self.root / "unsafe.yaml"
        config.write_text("!!python/object/apply:os.system ['touch unsafe-file']")
        with self.assertRaises(yaml.constructor.ConstructorError):
            self.contract.supported_stems(config)


if __name__ == "__main__":
    unittest.main()
