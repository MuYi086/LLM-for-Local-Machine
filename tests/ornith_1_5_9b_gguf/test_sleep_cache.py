"""不访问真实后端，验证缓存能力缺失、关闭或响应异常时拒绝就绪。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-text"))
from llama_cpp_runtime.sleep_cache import require_sleep_cache  # noqa: E402


class SleepCacheCapabilityTests(unittest.TestCase):
    """补丁能力由运行中后端报告，不能只信启动环境变量。"""

    def test_rejects_missing_disabled_invalid_or_non_boolean_capabilities(self) -> None:
        for props in (
            {},
            [],
            None,
            {"sleep_prompt_cache_supported": True},
            {"sleep_prompt_cache_supported": True, "sleep_prompt_cache_enabled": False},
            {"sleep_prompt_cache_supported": "true", "sleep_prompt_cache_enabled": True},
        ):
            with self.subTest(props=props), patch("urllib.request.build_opener") as build:
                response = build.return_value.open.return_value.__enter__.return_value
                response.status = 200
                response.read.return_value = json.dumps(props).encode()
                with self.assertRaisesRegex(RuntimeError, "部署缓存补丁"):
                    require_sleep_cache("http://127.0.0.1:18504")

    def test_accepts_only_explicit_enabled_backend(self) -> None:
        with patch("urllib.request.build_opener") as build:
            response = build.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = (
                b'{"sleep_prompt_cache_supported":true,"sleep_prompt_cache_enabled":true}'
            )
            require_sleep_cache("http://127.0.0.1:18504")
            build.return_value.open.assert_called_once_with(
                "http://127.0.0.1:18504/props", timeout=5
            )

    def test_unreachable_backend_fails_closed(self) -> None:
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = OSError("unreachable")
            with self.assertRaisesRegex(RuntimeError, "不得进入就绪"):
                require_sleep_cache("http://127.0.0.1:18504")


if __name__ == "__main__":
    unittest.main()
