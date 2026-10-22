"""ACE-Step 1.5 服务的无模型运行时生命周期测试。"""

from __future__ import annotations

# runtime 测试用伪进程验证超时回收、错误摘要和临时文件清理。
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class RuntimeTests(unittest.TestCase):
    def test_termination_cleanup_does_not_mask_final_wait_timeout(self) -> None:
        class StubbornProcess:
            pid = None
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                return None

            def kill(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("worker", timeout)

        runtime.terminate_process_group(StubbornProcess(), "test", 0, 0)

    def test_worker_uses_ace_step_python_and_cleans_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            script = temp_dir / "worker.py"
            script.write_text("# fake worker\n", encoding="utf-8")
            config = runtime.WorkerConfig(
                worker_script=script,
                temp_dir=temp_dir,
                timeout=5,
                label="test",
                file_prefix="ace_step_1_5",
            )

            class FakeProcess:
                pid = 12345
                returncode = 0

                def communicate(self, timeout=None):
                    Path(fake_command[fake_command.index("--output-wav") + 1]).write_bytes(FAKE_WAV)
                    Path(fake_command[fake_command.index("--metadata-json") + 1]).write_text(
                        '{"seed": 42}', encoding="utf-8"
                    )
                    return "worker ok", ""

                def poll(self):
                    return self.returncode

            fake_command: list[str] = []

            def fake_popen(command, **kwargs):
                fake_command.extend(command)
                return FakeProcess()

            with (
                patch.dict(runtime.os.environ, {"ACESTEP_PYTHON": "/tmp/ace-step-python"}),
                patch.object(runtime.subprocess, "Popen", side_effect=fake_popen),
            ):
                result = runtime.run_local_worker({"prompt": "test"}, config)

            self.assertEqual(result.audio, FAKE_WAV)
            self.assertEqual(result.metadata["seed"], 42)
            self.assertEqual(fake_command[0], "/tmp/ace-step-python")
            self.assertNotIn("conda", fake_command)
            self.assertEqual(list(temp_dir.glob("ace_step_1_5_*")), [])

    def test_worker_error_excerpt_keeps_tail(self) -> None:
        self.assertIn("last", runtime.worker_error_excerpt("first\nlast", "ACE-Step"))


if __name__ == "__main__":
    unittest.main()
