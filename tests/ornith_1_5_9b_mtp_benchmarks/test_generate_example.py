"""验证 Ornith-1.5-9B-MTP-GGUF-Q8 样例脚本的 API 合同和输出保护。"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from tests.ornith_1_5_9b_mtp_benchmarks.generate_example import (
    API_ROUTE,
    extract_assistant_text,
    request_completion,
    write_sample,
)


class Ornith9BExampleTests(unittest.TestCase):
    def test_extracts_nonempty_assistant_text(self) -> None:
        text = extract_assistant_text(
            {"choices": [{"message": {"role": "assistant", "content": "你好。"}}]}
        )

        self.assertEqual(text, "你好。")

    def test_missing_or_empty_assistant_text_is_rejected(self) -> None:
        for payload in (
            {"choices": []},
            {"choices": [{"message": {"content": " "}}]},
            {"choices": [{"message": {"content": None}}]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                extract_assistant_text(payload)

    def test_request_uses_9b_route_and_chat_completion_payload(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": "回答"}}]}, ensure_ascii=False
                ).encode("utf-8")

        captured = []
        with patch(
            "tests.ornith_1_5_9b_mtp_benchmarks.generate_example.urllib.request.urlopen",
            side_effect=lambda request, timeout: (
                captured.append((request, timeout)) or FakeResponse()
            ),
        ):
            text = request_completion(
                base_url="http://127.0.0.1:8504/",
                prompt="请回答",
                timeout_seconds=12,
            )

        request, timeout = captured[0]
        self.assertEqual(text, "回答")
        self.assertEqual(request.full_url, f"http://127.0.0.1:8504{API_ROUTE}")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data)["messages"][0]["content"], "请回答")
        self.assertEqual(timeout, 12)

    def test_http_error_reports_status_and_backend_detail(self) -> None:
        error = urllib.error.HTTPError(
            f"http://127.0.0.1:8504{API_ROUTE}",
            503,
            "unavailable",
            hdrs=None,
            fp=None,
        )
        with patch(
            "tests.ornith_1_5_9b_mtp_benchmarks.generate_example.urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                request_completion(base_url="http://127.0.0.1:8504", prompt="你好")

    def test_writes_unique_text_sample_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            timestamp = datetime(2026, 11, 13, tzinfo=UTC)
            output = write_sample("生成文本。", temporary, timestamp=timestamp, pid=99)

            self.assertEqual(output.parent, Path(temporary))
            self.assertEqual(output.read_text(encoding="utf-8"), "生成文本。\n")
            self.assertTrue(output.name.startswith("ornith-9b-example-20261113T000000000000Z-99"))
            with self.assertRaises(FileExistsError):
                write_sample("第二条。", temporary, timestamp=timestamp, pid=99)


if __name__ == "__main__":
    unittest.main()
