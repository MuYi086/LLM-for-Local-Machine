"""使用伪 CLI 验证 sd.cpp 的真实进程隔离、回收和原子发布。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from inference_gateway.sdcpp import GenerationRequest, SdcppConfig, load_models
from inference_gateway.sdcpp.execution import GenerationCancelledError, generate
from inference_gateway.sdcpp.processes import group_active, recover_sdcpp
from test_sdcpp_config import model_document

from inference_gateway import BackendQuarantinedError
from llm_local_runtime import GpuLockTimeoutError, GpuQuarantinedError, gpu_runtime_lock


class SdcppExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "sd-cli"
        self.manifest = self.root / "models.json"
        self.manifest.write_text(json.dumps(model_document()))
        for name in ("weights.gguf", "vae.safetensors"):
            (self.root / name).write_bytes(b"fixture")
        self.model = load_models(self.manifest)["fixture"]
        self.request = GenerationRequest(task="text-to-image", prompt="cat")
        patcher = patch("llm_local_runtime.gpu._read_gpu_memory_mib", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def configure(self, body: str) -> SdcppConfig:
        """生成不依赖模型的可执行替身。"""
        self.binary.write_text(
            f"#!{sys.executable}\nimport sys, pathlib, time, subprocess, os\n{body}\n"
        )
        self.binary.chmod(0o700)
        return SdcppConfig(
            self.binary,
            hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            self.root / "gpu.lock",
            self.root / "results",
            queue_timeout=0.1,
            run_timeout=3,
            cleanup_timeout=2,
            release_delay=0,
        )

    def test_success_publishes_unique_output_and_clears_marker(self) -> None:
        config = self.configure(
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')"
        )
        first = generate(config, self.model, self.request)
        second = generate(config, self.model, self.request)
        self.assertNotEqual(first.output, second.output)
        self.assertTrue(first.output.is_file())
        self.assertTrue((first.output.parent / "generation.json").is_file())
        self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())
        self.assertFalse(any(config.output_root.glob(".partial-*")))
        self.assertIn("queue_wait_seconds", first.metrics)

    def test_failure_and_missing_output_do_not_publish(self) -> None:
        for body in [
            "sys.exit(7)",
            "pass",
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'garbage')",
        ]:
            config = self.configure(body)
            with self.subTest(body=body), self.assertRaises(RuntimeError):
                generate(config, self.model, self.request)
            self.assertEqual(list(config.output_root.iterdir()), [])
            self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())

    def test_timeout_stops_worker_before_next_model_enters(self) -> None:
        config = replace(self.configure("time.sleep(30)"), run_timeout=0.1)
        with self.assertRaises(TimeoutError):
            generate(config, self.model, self.request)
        with gpu_runtime_lock(config.lock_path, "next", timeout_seconds=0.1):
            self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())

    def test_cancellation_stops_worker(self) -> None:
        config = self.configure("time.sleep(30)")
        cancel = threading.Event()
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        try:
            with self.assertRaises(GenerationCancelledError):
                generate(config, self.model, self.request, cancel_event=cancel)
        finally:
            timer.join()
        self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())

    def test_descendants_are_stopped_even_if_main_process_exits(self) -> None:
        config = self.configure(
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\npathlib.Path('group.txt').write_text(str(os.getpgrp()))\nsys.exit(4)"
        )
        with self.assertRaises(RuntimeError):
            generate(config, self.model, self.request)
        self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())

    def test_quarantine_and_queue_timeout_never_launch_cli(self) -> None:
        started = self.root / "started"
        config = self.configure(f"pathlib.Path({str(started)!r}).touch()")
        marker = Path(f"{config.lock_path}.quarantine")
        marker.write_text("{}")
        with self.assertRaises(GpuQuarantinedError):
            generate(config, self.model, self.request)
        marker.unlink()
        with gpu_runtime_lock(config.lock_path, "other-model", timeout_seconds=0.1):
            with self.assertRaises(GpuLockTimeoutError):
                generate(config, self.model, self.request)
        self.assertFalse(started.exists())

    def test_file_preflight_fails_before_lock(self) -> None:
        config = self.configure("pass")
        (self.root / "weights.gguf").unlink()
        with patch("inference_gateway.sdcpp.execution.gpu_runtime_lock") as lock:
            with self.assertRaises(ValueError):
                generate(config, self.model, self.request)
        lock.assert_not_called()

    def test_unconfirmed_cleanup_preserves_marker_and_request_directory(self) -> None:
        config = self.configure("sys.exit(3)")
        with patch(
            "inference_gateway.sdcpp.execution.stop_process_group",
            side_effect=RuntimeError("unconfirmed"),
        ):
            with self.assertRaisesRegex(BackendQuarantinedError, "退出码为 3"):
                generate(config, self.model, self.request)
        marker = Path(f"{config.lock_path}.quarantine")
        self.assertTrue(marker.is_file())
        record = json.loads(marker.read_text())
        self.assertEqual(record["provider"], "stable-diffusion.cpp")
        self.assertTrue(Path(record["request_dir"]).is_dir())
        recover_sdcpp(marker, record, 0.1)
        self.assertFalse(marker.exists())

    def test_changed_marker_is_never_removed(self) -> None:
        marker = self.root / "gpu.lock.quarantine"
        config = self.configure(
            f'pathlib.Path({str(marker)!r}).write_text(\'{{"provider":"other"}}\')\nsys.exit(2)'
        )
        with self.assertRaises(BackendQuarantinedError):
            generate(config, self.model, self.request)
        self.assertEqual(json.loads(marker.read_text()), {"provider": "other"})

    def test_deleted_marker_is_restored_as_quarantine(self) -> None:
        marker = self.root / "gpu.lock.quarantine"
        config = self.configure(f"pathlib.Path({str(marker)!r}).unlink()\nsys.exit(2)")
        with self.assertRaises(BackendQuarantinedError):
            generate(config, self.model, self.request)
        self.assertTrue(marker.exists())
        with self.assertRaises(GpuQuarantinedError):
            with gpu_runtime_lock(config.lock_path, "next", timeout_seconds=0.1):
                self.fail("缺失隔离标记时放行了后续任务")

    def test_registration_failure_never_permits_model_start(self) -> None:
        started = self.root / "started"
        config = self.configure(f"pathlib.Path({str(started)!r}).touch()")
        with patch(
            "inference_gateway.sdcpp.execution.process_identity",
            side_effect=RuntimeError("registration failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                generate(config, self.model, self.request)
        self.assertFalse(started.exists())
        self.assertFalse(Path(f"{config.lock_path}.quarantine").exists())

    def test_video_uses_avi_output(self) -> None:
        config = self.configure(
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'RIFF\\x10\\x00\\x00\\x00AVI fixture')"
        )
        image = self.root / "input.png"
        image.write_bytes(b"fixture")
        result = generate(
            config,
            self.model,
            GenerationRequest(task="image-to-video", prompt="cat", init_image=image),
        )
        self.assertEqual(result.output.suffix, ".avi")

    def test_cancel_while_waiting_never_starts_model_after_lock(self) -> None:
        started = self.root / "started"
        config = self.configure(f"pathlib.Path({str(started)!r}).touch()")
        cancel = threading.Event()
        original_lock = gpu_runtime_lock

        def acquire_and_cancel(*args, **kwargs):
            cancel.set()
            return original_lock(*args, **kwargs)

        with patch(
            "inference_gateway.sdcpp.execution.gpu_runtime_lock", side_effect=acquire_and_cancel
        ):
            with self.assertRaises(GenerationCancelledError):
                generate(config, self.model, self.request, cancel_event=cancel)
        self.assertFalse(started.exists())

    def test_worker_without_permission_does_not_execute(self) -> None:
        started = self.root / "started"
        self.configure(f"pathlib.Path({str(started)!r}).touch()")
        worker = (
            Path(__file__).resolve().parents[1]
            / "inference_gateway/src/inference_gateway/sdcpp/worker.py"
        )
        completed = subprocess.run(
            [sys.executable, "-I", str(worker), str(self.binary)],
            input=b"",
            capture_output=True,
            timeout=3,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(started.exists())

    def test_parent_crash_leaves_recoverable_backend_identity(self) -> None:
        started = self.root / "started"
        config = self.configure(f"pathlib.Path({str(started)!r}).touch()\ntime.sleep(30)")
        root = Path(__file__).resolve().parents[1]
        program = (
            "from pathlib import Path\n"
            "import llm_local_runtime.gpu as gpu\n"
            "gpu._read_gpu_memory_mib = lambda: None\n"
            "from inference_gateway.sdcpp import SdcppConfig, load_models, GenerationRequest\n"
            "from inference_gateway.sdcpp.execution import generate\n"
            f"generate(SdcppConfig.from_env(), load_models(Path({str(self.manifest)!r}))['fixture'], GenerationRequest(task='text-to-image', prompt='cat'))\n"
        )
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(root / "inference_gateway/src"), str(root / "llm_local_runtime/src")]
            ),
            "SDCPP_BIN": str(config.binary),
            "SDCPP_BIN_SHA256": config.binary_sha256,
            "GPU_LOCK_FILE": str(config.lock_path),
            "SDCPP_OUTPUT_DIR": str(config.output_root),
            "CUDA_RELEASE_DELAY": "0",
        }
        parent = subprocess.Popen(
            [sys.executable, "-c", program],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        marker = Path(f"{config.lock_path}.quarantine")
        try:
            deadline = time.monotonic() + 5
            while not started.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(started.exists(), "伪 backend 未启动")
            record = json.loads(marker.read_text())
            parent.kill()
            parent.wait(timeout=3)
            self.assertTrue(group_active(record["pgid"]))
            with patch.dict(os.environ, {"CUDA_RELEASE_DELAY": "0"}):
                # 恢复入口在生产中负责取得同一路径锁；测试直接持有原始 flock。
                import fcntl

                with config.lock_path.open("a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    recover_sdcpp(marker, record, 2)
            self.assertFalse(group_active(record["pgid"]))
            self.assertFalse(marker.exists())
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=3)
            if marker.exists():
                record = json.loads(marker.read_text())
                if record.get("pgid"):
                    from inference_gateway.sdcpp.processes import stop_process_group

                    stop_process_group(record["pgid"], 2)

    def test_recovery_refuses_reused_pid_and_live_unknown_group(self) -> None:
        marker = self.root / "gpu.lock.quarantine"
        record = {
            "schema_version": 1,
            "provider": "stable-diffusion.cpp",
            "phase": "registered",
            "pid": 999991,
            "pgid": 999991,
            "pid_start_time": "10",
            "executable": str(self.binary),
        }
        marker.write_text(json.dumps(record))
        with (
            patch(
                "inference_gateway.sdcpp.processes.process_identity",
                return_value=("20", 999991, "S"),
            ),
            patch("inference_gateway.sdcpp.processes.group_active", return_value=True),
            patch("inference_gateway.sdcpp.processes.os.killpg") as kill,
        ):
            with self.assertRaises(BackendQuarantinedError):
                recover_sdcpp(marker, record, 0.1)
        kill.assert_not_called()
        self.assertTrue(marker.exists())

    def test_recovery_confirms_entire_group_before_clearing(self) -> None:
        marker = self.root / "gpu.lock.quarantine"
        record = {
            "schema_version": 1,
            "provider": "stable-diffusion.cpp",
            "phase": "registered",
            "pid": 999991,
            "pgid": 999991,
            "pid_start_time": "10",
            "executable": str(self.binary),
        }
        marker.write_text(json.dumps(record))
        with (
            patch("inference_gateway.sdcpp.processes.process_identity", return_value=None),
            patch("inference_gateway.sdcpp.processes.group_active", return_value=True),
        ):
            with self.assertRaises(BackendQuarantinedError):
                recover_sdcpp(marker, record, 0.1)
        self.assertTrue(marker.exists())

    def test_worker_has_separate_process_group(self) -> None:
        config = self.configure(
            "assert os.getpgrp() == os.getpid()\npathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')"
        )
        result = generate(config, self.model, self.request)
        record = json.loads((result.output.parent / "generation.json").read_text())
        self.assertFalse(group_active(record["pid"]))
        self.assertNotEqual(record["pid"], os.getpgrp())

    def test_leader_is_not_reaped_before_group_cleanup(self) -> None:
        config = self.configure(
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')"
        )
        from inference_gateway.sdcpp.processes import process_identity, stop_process_group

        observed = []

        def verify_identity_before_cleanup(pid, timeout):
            identity = process_identity(pid)
            self.assertIsNotNone(identity, "组回收前根进程已经被 reap，PID 可以被复用")
            self.assertEqual(identity[2], "Z")
            observed.append(pid)
            stop_process_group(pid, timeout)

        with patch(
            "inference_gateway.sdcpp.execution.stop_process_group",
            side_effect=verify_identity_before_cleanup,
        ):
            result = generate(config, self.model, self.request)
        self.assertTrue(result.output.is_file())
        self.assertEqual(len(observed), 1)
