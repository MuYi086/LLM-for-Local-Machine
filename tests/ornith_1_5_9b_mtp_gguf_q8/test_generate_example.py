"""验证 Ornith MTP 9B Q8 文本与图像调用 demo。"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

try:
    from tests.ornith_1_5_9b_mtp_gguf_q8 import generate_example as example
except ImportError:
    example = None


class OrnithMTP9BQ8ExampleTests(unittest.TestCase):
    def require_example(self):
        self.assertIsNotNone(example, "MTP demo module must be importable")
        return example

    def test_text_request_targets_mtp_route_and_chat_completion(self) -> None:
        module = self.require_example()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": "文字回答"}}]},
                    ensure_ascii=False,
                ).encode("utf-8")

        captured = []
        with patch(
            "tests.ornith_1_5_9b_mtp_gguf_q8.generate_example.urllib.request.urlopen",
            side_effect=lambda request, timeout: (
                captured.append((request, timeout)) or FakeResponse()
            ),
        ):
            text = module.request_completion(
                base_url="http://127.0.0.1:8504/",
                prompt="请用中文介绍你自己。",
                timeout_seconds=12,
            )

        request, timeout = captured[0]
        self.assertEqual(text, "文字回答")
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8504/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8",
        )
        self.assertEqual(request.method, "POST")
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"][0]["content"], "请用中文介绍你自己。")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(timeout, 12)

    def test_image_request_sends_png_and_jpeg_data_urls(self) -> None:
        module = self.require_example()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": "红色方块"}}]},
                    ensure_ascii=False,
                ).encode("utf-8")

        image_bytes = b"image fixture bytes"
        with tempfile.TemporaryDirectory() as temporary:
            for suffix, media_type in ((".png", "image/png"), (".jpeg", "image/jpeg")):
                with self.subTest(suffix=suffix):
                    image_path = Path(temporary) / f"sample{suffix}"
                    image_path.write_bytes(image_bytes)
                    captured = []
                    with patch(
                        "tests.ornith_1_5_9b_mtp_gguf_q8.generate_example.urllib.request.urlopen",
                        side_effect=lambda request, timeout, captured=captured: (
                            captured.append(request) or FakeResponse()
                        ),
                    ):
                        text = module.request_completion(
                            base_url="http://127.0.0.1:8504",
                            prompt="请描述图片。",
                            image_path=image_path,
                        )

                    payload = json.loads(captured[0].data)
                    content = payload["messages"][0]["content"]
                    self.assertEqual(text, "红色方块")
                    self.assertEqual(content[0], {"type": "text", "text": "请描述图片。"})
                    self.assertEqual(
                        content[1],
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            },
                        },
                    )

    def test_empty_assistant_response_is_rejected(self) -> None:
        module = self.require_example()
        for payload in (
            {"choices": []},
            {"choices": [{"message": {"content": " "}}]},
            {"choices": [{"message": {"content": None}}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module.extract_assistant_text(payload)

    def test_http_error_contains_status_and_backend_detail(self) -> None:
        module = self.require_example()
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8504/api",
            503,
            "unavailable",
            hdrs=None,
            fp=None,
        )
        with patch(
            "tests.ornith_1_5_9b_mtp_gguf_q8.generate_example.urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                module.request_completion(
                    base_url="http://127.0.0.1:8504",
                    prompt="你好",
                )

    def test_output_uses_exclusive_unique_file_creation(self) -> None:
        module = self.require_example()
        with tempfile.TemporaryDirectory() as temporary:
            timestamp = datetime(2026, 11, 13, tzinfo=UTC)
            output = module.write_sample("返回样例。", temporary, timestamp=timestamp, pid=42)

            self.assertEqual(output.parent, Path(temporary))
            self.assertEqual(output.read_text(encoding="utf-8"), "返回样例。\n")
            self.assertTrue(
                output.name.startswith("ornith-mtp-9b-q8-example-20261113T000000000000Z-42")
            )
            with self.assertRaises(FileExistsError):
                module.write_sample("不覆盖。", temporary, timestamp=timestamp, pid=42)


if __name__ == "__main__":
    unittest.main()
