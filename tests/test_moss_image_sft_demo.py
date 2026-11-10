"""验证 MOSS-Image-SFT HTTP 示例脚本的请求与安全写入。"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "tests/MOSS-Image-SFT-Q8_0/generate_example.py"
SPEC = importlib.util.spec_from_file_location("moss_image_sft_example", EXAMPLE_FILE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载 MOSS 示例脚本：{EXAMPLE_FILE}")
example = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = example
SPEC.loader.exec_module(example)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture-image"


class MossImageSftDemoTests(unittest.TestCase):
    def test_posts_to_fixed_route_and_saves_png_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.png"
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
            ).encode("utf-8")
            response = io.BytesIO(body)

            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                generated = example.generate_example("http://127.0.0.1:8501", output, 60)

            self.assertEqual(generated, output)
            self.assertEqual(output.read_bytes(), PNG_BYTES)
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8501/v1/openmoss/MOSS-Image-SFT-Q8_0",
            )
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "MOSS-Image-SFT-Q8_0")
            self.assertEqual(payload["size"], "1024x1024")
            self.assertEqual(payload["cfg_scale"], 3.0)

    def test_rejects_non_png_response_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.png"
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(b"not png").decode("ascii")}]}
            ).encode("utf-8")
            with (
                patch("urllib.request.urlopen", return_value=io.BytesIO(body)),
                self.assertRaisesRegex(ValueError, "不是 PNG"),
            ):
                example.generate_example("http://127.0.0.1:8501", output, 60)
            self.assertFalse(output.exists())

    def test_never_overwrites_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.png"
            output.write_bytes(b"keep existing output")
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
            ).encode("utf-8")
            with (
                patch("urllib.request.urlopen", return_value=io.BytesIO(body)),
                self.assertRaises(FileExistsError),
            ):
                example.generate_example("http://127.0.0.1:8501", output, 60)
            self.assertEqual(output.read_bytes(), b"keep existing output")


if __name__ == "__main__":
    unittest.main()
