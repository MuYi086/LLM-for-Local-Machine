"""验证 Ornith 真实样例脚本的响应解析与运行数据输出。"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from tests.ornith_1_5_35b_a3b.generate_example import (
    API_ROUTE,
    extract_assistant_text,
    request_completion,
    write_sample,
)


class OrnithExampleTests(unittest.TestCase):
    def test_extracts_nonempty_assistant_text(self) -> None:
        result = extract_assistant_text(
            {"choices": [{"message": {"role": "assistant", "content": "生成结果。"}}]}
        )

        self.assertEqual(result, "生成结果。")

    def test_missing_or_empty_completion_is_rejected(self) -> None:
        for payload in ({"choices": []}, {"choices": [{"message": {"content": " "}}]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                extract_assistant_text(payload)

    def test_writes_sample_to_requested_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = write_sample(
                "模型生成文本。",
                temporary,
                timestamp=datetime(2026, 11, 9, tzinfo=UTC),
            )

            self.assertEqual(output.parent, Path(temporary))
            self.assertEqual(output.read_text(encoding="utf-8"), "模型生成文本。\n")
            self.assertTrue(output.name.startswith("ornith-example-20261109T000000-"))

    def test_request_targets_exact_route_and_parses_completion(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "你好。"}}]},
                    ensure_ascii=False,
                ).encode("utf-8")

        captured = []
        with patch(
            "tests.ornith_1_5_35b_a3b.generate_example.urllib.request.urlopen",
            side_effect=lambda request, timeout: (
                captured.append((request, timeout)) or FakeResponse()
            ),
        ):
            text = request_completion(
                base_url="http://127.0.0.1:8401/",
                prompt="你好",
                timeout_seconds=10,
            )

        self.assertEqual(text, "你好。")
        request, timeout = captured[0]
        self.assertEqual(request.full_url, f"http://127.0.0.1:8401{API_ROUTE}")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data)["messages"][0]["content"], "你好")
        self.assertEqual(timeout, 10)

    def test_http_errors_include_status_and_backend_detail(self) -> None:
        error = urllib.error.HTTPError(
            f"http://127.0.0.1:8401{API_ROUTE}",
            503,
            "unavailable",
            hdrs=None,
            fp=None,
        )
        with patch(
            "tests.ornith_1_5_35b_a3b.generate_example.urllib.request.urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                request_completion(base_url="http://127.0.0.1:8401", prompt="你好")


if __name__ == "__main__":
    unittest.main()
