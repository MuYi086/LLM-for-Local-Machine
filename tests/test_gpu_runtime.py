"""验证 GPU 文件锁跨进程串行、超时、隔离和异常释放合同。"""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llm_local_runtime.gpu as gpu_module

from llm_local_runtime import (
    GpuLockTimeoutError,
    GpuQuarantinedError,
    gpu_quarantine_path,
    gpu_runtime_lock,
)


def _hold_gpu_lock(lock_path: str, acquired: object, release: object) -> None:
    """子进程代表另一模型持锁，等待父进程发出释放信号。"""
    with gpu_runtime_lock(lock_path, "model-a", timeout_seconds=3):
        acquired.set()
        release.wait(3)


class GpuRuntimeLockTests(unittest.TestCase):
    def test_different_models_cannot_hold_the_shared_lock_concurrently(self) -> None:
        context = multiprocessing.get_context("fork")
        acquired = context.Event()
        release = context.Event()
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu-runtime.lock")
            process = context.Process(target=_hold_gpu_lock, args=(lock_path, acquired, release))
            with patch.object(gpu_module, "_read_gpu_memory_mib", return_value=None):
                process.start()
                try:
                    self.assertTrue(acquired.wait(3), "模型 A 未在时限内取得 GPU 锁")
                    with self.assertRaises(GpuLockTimeoutError):
                        with gpu_runtime_lock(lock_path, "model-b", timeout_seconds=0.05):
                            self.fail("模型 B 在模型 A 持锁时进入了临界区")
                finally:
                    release.set()
                    process.join(5)

                self.assertFalse(process.is_alive(), "模型 A 子进程未退出")
                self.assertEqual(process.exitcode, 0)
                with gpu_runtime_lock(lock_path, "model-b", timeout_seconds=1):
                    pass

    def test_quarantine_blocks_models_until_operator_removes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            quarantine = gpu_quarantine_path(lock_path)
            quarantine.touch()

            with patch.object(gpu_module, "_read_gpu_memory_mib", return_value=None):
                with self.assertRaises(GpuQuarantinedError):
                    with gpu_runtime_lock(lock_path, "model-c", timeout_seconds=1):
                        self.fail("quarantine 标记存在时模型进入了临界区")

                quarantine.unlink()
                with gpu_runtime_lock(lock_path, "model-c", timeout_seconds=1):
                    pass

    def test_maintenance_can_acquire_lock_to_stop_processes_while_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            gpu_quarantine_path(lock_path).touch()

            with patch.object(gpu_module, "_read_gpu_memory_mib", return_value=None):
                with gpu_runtime_lock(
                    lock_path,
                    "llama.cpp/shutdown-many",
                    timeout_seconds=1,
                    allow_quarantine=True,
                ):
                    pass

    def test_exception_releases_lock_for_the_next_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            with patch.object(gpu_module, "_read_gpu_memory_mib", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    with gpu_runtime_lock(lock_path, "model-a", timeout_seconds=1):
                        raise RuntimeError("worker failed")
                with gpu_runtime_lock(lock_path, "model-b", timeout_seconds=1):
                    pass


if __name__ == "__main__":
    unittest.main()
