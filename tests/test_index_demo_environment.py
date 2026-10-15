"""Index 原生端口 Demo 的只检查模式和原始响应保存回归。"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from index_demo_environment import EnvironmentCheck, check_service_environment
from index_translate_demo_common import run_echo_demo, run_text_model_demo


class IndexDemoEnvironmentTests(unittest.TestCase):
    """验证 Demo 不再要求外部脚本，也不会误接旧音频编排接口。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "source.md"
        self.input.write_text("源文\n第二行", encoding="utf-8")
        self.output = self.root / "dist/output.txt"

    def service_checks(self, status, **kwargs):
        response = io.BytesIO(json.dumps(status).encode())
        with patch("index_demo_environment.urllib.request.urlopen", return_value=response) as call:
            checks = check_service_environment(
                input_path=self.input,
                base_url="http://127.0.0.1:8383",
                expected_backend="Index-Homura GGUF native",
                output_type="text/plain",
                **kwargs,
            )
        self.assertEqual(call.call_args.args[0], "http://127.0.0.1:8383/v1/health")
        return checks

    def test_correct_native_health_passes(self):
        checks = self.service_checks(
            {
                "ready": True,
                "available": {"model": True},
                "runtime": {
                    "backend": "Index-Homura GGUF native translation",
                    "output": "text/plain",
                },
            }
        )
        self.assertTrue(all(check.passed for check in checks))

    def test_old_or_wrong_model_endpoint_is_rejected(self):
        checks = self.service_checks(
            {
                "ready": True,
                "runtime": {"backend": "GGUF + Qwen3-ASR + Confucius4-TTS", "output": "audio/wav"},
            }
        )
        self.assertFalse(checks[1].passed)
        checks = self.service_checks(
            {
                "ready": True,
                "runtime": {
                    "backend": "Index-Translate GGUF native translation",
                    "output": "text/plain",
                },
            }
        )
        self.assertFalse(checks[1].passed)

    def test_image_requires_vision_readiness(self):
        checks = self.service_checks(
            {
                "ready": True,
                "image_ready": False,
                "runtime": {"backend": "Index-Homura GGUF native", "output": "text/plain"},
            },
            image_input=True,
        )
        self.assertFalse(checks[-1].passed)

    def test_empty_input_fails(self):
        self.input.write_text(" \n")
        checks = self.service_checks(
            {
                "ready": True,
                "runtime": {"backend": "Index-Homura GGUF native", "output": "text/plain"},
            }
        )
        self.assertFalse(checks[0].passed)

    def run_text(self):
        run_text_model_demo(
            demo_name="Index-Homura",
            env_prefix="INDEX_HOMURA",
            port=8383,
            route="homura",
            output_name="output.txt",
            demo_dir=self.root,
        )

    def test_check_only_and_missing_conditions_never_request_generation(self):
        for check_only, ready in ((True, True), (False, False)):
            argv = ["demo.py", "--input", str(self.input)] + (
                ["--check-only"] if check_only else []
            )
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "index_translate_demo_common.check_service_environment",
                    return_value=[EnvironmentCheck("环境", ready, "mock")],
                ),
                patch("index_translate_demo_common.request_output") as request,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                if ready:
                    self.run_text()
                else:
                    with self.assertRaisesRegex(SystemExit, "跳过推理"):
                        self.run_text()
            request.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_text_demo_uses_native_port_and_preserves_response_bytes(self):
        content = "Translation\n\n第二段\n".encode()
        with (
            patch.object(
                sys, "argv", ["demo.py", "--input", str(self.input), "--output", str(self.output)]
            ),
            patch(
                "index_translate_demo_common.check_service_environment",
                return_value=[EnvironmentCheck("环境", True, "mock")],
            ),
            patch("index_translate_demo_common.request_output", return_value=content) as request,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.run_text()
        self.assertEqual(self.output.read_bytes(), content)
        call = request.call_args.args[0]
        self.assertEqual(call.full_url, "http://127.0.0.1:8383/v1/indexTranslate/homura")
        self.assertEqual(json.loads(call.data)["text"], self.input.read_text())
        self.assertEqual(json.loads(call.data)["target_syllables"], 36)
        self.assertFalse(list(self.output.parent.glob(".*")))

    def test_echo_check_only_does_not_upload_audio(self):
        with (
            patch.object(sys, "argv", ["demo.py", "--check-only"]),
            patch(
                "index_translate_demo_common.check_service_environment",
                return_value=[EnvironmentCheck("环境", True, "mock")],
            ),
            patch("index_translate_demo_common.request_output") as request,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            run_echo_demo(self.root)
        request.assert_not_called()

    def test_echo_upload_is_streamed_to_native_endpoint(self):
        audio = self.root / "speech.wav"
        audio.write_bytes(b"RIFF" + b"a" * (2 * 1024 * 1024))
        output = self.root / "dist/result.wav"
        with (
            patch.object(sys, "argv", ["demo.py", "--audio", str(audio), "--output", str(output)]),
            patch(
                "index_translate_demo_common.check_service_environment",
                return_value=[EnvironmentCheck("环境", True, "mock")],
            ),
            patch(
                "index_translate_demo_common.request_output", return_value=b"RIFF-native"
            ) as request,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            run_echo_demo(self.root)
        call = request.call_args.args[0]
        self.assertEqual(call.full_url, "http://127.0.0.1:8382/v1/indexTranslate/echoS2ST")
        chunks = list(call.data)
        self.assertTrue(all(len(chunk) <= 1024 * 1024 for chunk in chunks))
        self.assertEqual(sum(map(len, chunks)), int(call.headers["Content-length"]))
        self.assertIn(b'name="target_language"', chunks[0])
        self.assertEqual(output.read_bytes(), b"RIFF-native")
