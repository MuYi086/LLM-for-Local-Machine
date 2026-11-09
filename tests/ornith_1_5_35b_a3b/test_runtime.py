"""验证共享 llama.cpp runtime 对不同模型的隔离标记支持。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model-text"), str(ROOT / "llm_local_runtime/src")]

from ornith_1_5_35b_a3b.runtime import create_quarantine_marker  # noqa: E402


class LlamaCppRuntimeTests(unittest.TestCase):
    def test_quarantine_marker_accepts_qwen_model_alias_and_loopback_ip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"

            create_quarantine_marker(
                lock_path,
                host="127.0.0.2",
                port=8400,
                pid=None,
                pgid=None,
                executable="/opt/llama-server",
                model_alias="Qwen3.6-35B-A3B-MTP",
            )

            marker = json.loads(Path(f"{lock_path}.quarantine").read_text(encoding="utf-8"))

        self.assertEqual(marker["provider"], "llama.cpp")
        self.assertEqual(marker["model"], "Qwen3.6-35B-A3B-MTP")
        self.assertEqual(marker["host"], "127.0.0.2")
        self.assertEqual(marker["port"], 8400)


if __name__ == "__main__":
    unittest.main()
