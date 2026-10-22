"""用真实无模型子进程验证 bridge 的整个进程组收尾。"""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

from localai_worker_bridge.runner import WorkerExecutionError, WorkerRunner
from localai_worker_bridge.server import LocalAIWorkerBackend


def process_active(pid: int) -> bool:
    """将等待宿主机收养回收的 zombie 视为已停止。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat[stat.rfind(")") + 2 :].split()[0] not in {"Z", "X"}


class WorkerLifecycleTests(unittest.TestCase):
    def test_unconfirmed_exit_preserves_marker_and_free_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "gpu.lock"
            marker = root / "gpu.lock.quarantine"
            runner = WorkerRunner(root, {})
            context = Mock()
            context.abort.side_effect = WorkerExecutionError("Free refused")
            try:
                with (
                    patch.dict(os.environ, {"GPU_LOCK_FILE": str(lock_path)}),
                    patch(
                        "localai_worker_bridge.runner._process_group_alive",
                        side_effect=WorkerExecutionError("cannot confirm"),
                    ),
                ):
                    with self.assertRaisesRegex(WorkerExecutionError, "GPU 已隔离"):
                        runner._run_process(
                            [sys.executable, "-c", "pass"],
                            cwd=root,
                            timeout_seconds=5,
                            cancelled=None,
                            request_id="cleanup-test",
                            environment=os.environ.copy(),
                        )
                    self.assertTrue(marker.is_file())
                    with self.assertRaisesRegex(WorkerExecutionError, "Free refused"):
                        LocalAIWorkerBackend(runner, set()).Free(None, context)
                    context.abort.assert_called_once()
                    self.assertTrue(marker.is_file())
            finally:
                runner.ensure_stopped()

    def test_descendant_is_stopped_for_every_worker_outcome(self) -> None:
        for outcome in ("success", "error", "cancel", "timeout"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                child_pid_file = root / "child.pid"
                child_code = (
                    "import os,signal,sys,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
                )
                parent_code = (
                    "import subprocess,sys,time; from pathlib import Path\n"
                    f"subprocess.Popen([sys.executable,'-c',{child_code!r},sys.argv[1]])\n"
                    "while not Path(sys.argv[1]).exists(): time.sleep(.01)\n"
                    + ("time.sleep(30)\n" if outcome in {"cancel", "timeout"} else "")
                    + ("sys.exit(7)\n" if outcome == "error" else "")
                )
                runner = WorkerRunner(root, {})

                run = partial(
                    runner._run_process,
                    [sys.executable, "-c", parent_code, str(child_pid_file)],
                    cwd=root,
                    timeout_seconds=0.5 if outcome == "timeout" else 5,
                    cancelled=(lambda path=child_pid_file: not path.exists())
                    if outcome == "cancel"
                    else None,
                    request_id="lifecycle-test",
                    environment=os.environ.copy(),
                )

                try:
                    if outcome == "success":
                        run()
                    else:
                        with self.assertRaises(WorkerExecutionError):
                            run()
                    self.assertTrue(child_pid_file.is_file())
                    child_pid = int(child_pid_file.read_text())
                    self.assertFalse(process_active(child_pid), outcome)
                finally:
                    if child_pid_file.exists():
                        child_pid = int(child_pid_file.read_text())
                        if process_active(child_pid):
                            os.killpg(os.getpgid(child_pid), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
