"""验证 sd.cpp 运维入口不会在状态检查时启动模型。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_sdcpp_execution as execution_fixtures

from scripts.localai import recover_quarantine
from scripts.sdcpp import manager


class SdcppCliTests(unittest.TestCase):
    def test_status_lists_image_models_without_executing_binary(self) -> None:
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"HF_MIRROR_DIR": "/tmp/hf-mirror"}, clear=True),
            patch("subprocess.Popen") as popen,
            contextlib.redirect_stdout(output),
        ):
            code = manager.main(["status"])
        self.assertEqual(code, 0)
        status = json.loads(output.getvalue())
        self.assertEqual(
            status["models"],
            [
                {"id": "MOSS-Image-SFT-Q8_0", "tasks": ["text-to-image"]},
                {
                    "id": "FLUX-3-Image-Klein-GGUF-Q8",
                    "tasks": ["text-to-image", "image-to-image"],
                },
            ],
        )
        self.assertFalse(status["configured"])
        self.assertEqual(status["model_validation"], "MOSS-Image-SFT-Q8_0-inference-verified")
        popen.assert_not_called()

    def test_generate_without_model_configuration_returns_error(self) -> None:
        error = io.StringIO()
        with (
            patch.dict(os.environ, {"HF_MIRROR_DIR": "/tmp/hf-mirror"}),
            contextlib.redirect_stderr(error),
        ):
            code = manager.main(
                ["generate", "--model", "missing", "--task", "text-to-image", "--prompt", "cat"]
            )
        self.assertEqual(code, 2)
        self.assertIn("missing", error.getvalue())

    def test_operator_cli_generates_with_fake_backend(self) -> None:
        fixture = execution_fixtures.SdcppExecutionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = fixture.configure(
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')"
        )
        environment = {
            "SDCPP_BIN": str(config.binary),
            "SDCPP_BIN_SHA256": config.binary_sha256,
            "GPU_LOCK_FILE": str(config.lock_path),
            "SDCPP_OUTPUT_DIR": str(config.output_root),
            "CUDA_RELEASE_DELAY": "0",
        }
        output = io.StringIO()
        with (
            patch.dict(os.environ, environment),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = manager.main(
                [
                    "--manifest",
                    str(fixture.manifest),
                    "generate",
                    "--model",
                    "fixture",
                    "--task",
                    "text-to-image",
                    "--prompt",
                    "cat",
                ]
            )
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertTrue(Path(result["output"]).is_file())
        self.assertIn("worker_seconds", result["metrics"])

    def test_recovery_entry_dispatches_sdcpp_without_localai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "gpu.lock"
            marker = Path(f"{lock}.quarantine")
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "stable-diffusion.cpp",
                        "phase": "prepared",
                        "pid": None,
                        "pgid": None,
                    }
                )
            )
            with (
                patch.dict(os.environ, {"GPU_LOCK_FILE": str(lock)}),
                patch.object(recover_quarantine, "LocalAIClient") as client,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(recover_quarantine.main(), 0)
            self.assertFalse(marker.exists())
            client.assert_not_called()
