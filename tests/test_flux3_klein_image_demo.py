"""验证 FLUX.3 Klein HTTP 示例脚本的生成、编辑和安全写入。"""

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

EXAMPLE_FILE = (
    Path(__file__).resolve().parents[1] / "tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py"
)
SPEC = importlib.util.spec_from_file_location("flux3_klein_image_example", EXAMPLE_FILE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载 FLUX.3 Klein 示例脚本：{EXAMPLE_FILE}")
example = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = example
SPEC.loader.exec_module(example)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture-image"


class Flux3KleinImageDemoTests(unittest.TestCase):
    def test_generates_then_edits_image_and_saves_both_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated_output = Path(temporary) / "generated.png"
            edited_output = Path(temporary) / "edited.png"
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
            ).encode("utf-8")

            with patch(
                "urllib.request.urlopen",
                side_effect=[io.BytesIO(body), io.BytesIO(body)],
            ) as urlopen:
                results = example.generate_and_edit_example(
                    "http://127.0.0.1:8502", generated_output, edited_output, 60
                )

            self.assertEqual(results, (generated_output, edited_output))
            self.assertEqual(generated_output.read_bytes(), PNG_BYTES)
            self.assertEqual(edited_output.read_bytes(), PNG_BYTES)
            generation_request = urlopen.call_args_list[0].args[0]
            self.assertEqual(
                generation_request.full_url,
                "http://127.0.0.1:8502/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8",
            )
            generation_payload = json.loads(generation_request.data)
            self.assertEqual(generation_payload["model"], "FLUX-3-Image-Klein-GGUF-Q8")
            self.assertEqual(generation_payload["steps"], 16)
            edit_request = urlopen.call_args_list[1].args[0]
            self.assertTrue(edit_request.full_url.endswith("/edit"))
            self.assertTrue(
                edit_request.get_header("Content-type").startswith("multipart/form-data")
            )
            self.assertIn(b'name="prompt"', edit_request.data)
            self.assertIn(b'name="image"; filename="generated.png"', edit_request.data)
            self.assertIn(PNG_BYTES, edit_request.data)
            self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], 60)

    def test_rejects_non_png_response_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated_output = Path(temporary) / "generated.png"
            edited_output = Path(temporary) / "edited.png"
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(b"not png").decode("ascii")}]}
            ).encode("utf-8")
            with (
                patch("urllib.request.urlopen", return_value=io.BytesIO(body)),
                self.assertRaisesRegex(ValueError, "不是 PNG"),
            ):
                example.generate_and_edit_example(
                    "http://127.0.0.1:8502", generated_output, edited_output, 60
                )
            self.assertFalse(generated_output.exists())
            self.assertFalse(edited_output.exists())

    def test_never_overwrites_an_existing_generated_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated_output = Path(temporary) / "generated.png"
            edited_output = Path(temporary) / "edited.png"
            generated_output.write_bytes(b"keep existing output")
            body = json.dumps(
                {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii")}]}
            ).encode("utf-8")
            with (
                patch("urllib.request.urlopen", return_value=io.BytesIO(body)),
                self.assertRaises(FileExistsError),
            ):
                example.generate_and_edit_example(
                    "http://127.0.0.1:8502", generated_output, edited_output, 60
                )
            self.assertEqual(generated_output.read_bytes(), b"keep existing output")
            self.assertFalse(edited_output.exists())


if __name__ == "__main__":
    unittest.main()
