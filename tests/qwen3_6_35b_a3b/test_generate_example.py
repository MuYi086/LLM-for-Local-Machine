"""验证 Qwen3.6 样例生成脚本的请求和输出合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from tests.qwen3_6_35b_a3b.generate_example import (
    API_ROUTE,
    extract_assistant_text,
    request_completion,
    write_sample,
)


class QwenExampleTests(unittest.TestCase):
    def test_rejects_missing_or_empty_assistant_text(self) -> None:
        for payload in ({"choices": []}, {"choices": [{"message": {"content": " "}}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                extract_assistant_text(payload)

    def test_request_targets_public_route_and_disables_thinking(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "答案是 2。"}}]},
                    ensure_ascii=False,
                ).encode("utf-8")

        captured = []
        with patch(
            "tests.qwen3_6_35b_a3b.generate_example.urllib.request.urlopen",
            side_effect=lambda request, timeout: (
                captured.append((request, timeout)) or FakeResponse()
            ),
        ):
            result = request_completion(
                base_url="http://127.0.0.1:8402/",
                prompt="1+1 等于几？",
                timeout_seconds=30,
            )

        self.assertEqual(result, "答案是 2。")
        request, timeout = captured[0]
        self.assertEqual(request.full_url, f"http://127.0.0.1:8402{API_ROUTE}")
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"][0]["content"], "1+1 等于几？")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(timeout, 30)

    def test_writes_nonempty_sample_to_requested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = write_sample(
                "模型给出答案。",
                temporary,
                timestamp=datetime(2026, 11, 9, tzinfo=UTC),
            )

            self.assertEqual(output.parent, Path(temporary))
            self.assertEqual(output.read_text(encoding="utf-8"), "模型给出答案。\n")
            with self.assertRaises(ValueError):
                write_sample(" ", temporary)

    def test_http_error_reports_status(self) -> None:
        error = urllib.error.HTTPError(
            f"http://127.0.0.1:8402{API_ROUTE}", 503, "unavailable", hdrs=None, fp=None
        )
        with (
            patch(
                "tests.qwen3_6_35b_a3b.generate_example.urllib.request.urlopen",
                side_effect=error,
            ),
            self.assertRaisesRegex(RuntimeError, "HTTP 503"),
        ):
            request_completion(base_url="http://127.0.0.1:8402", prompt="测试")


if __name__ == "__main__":
    unittest.main()
