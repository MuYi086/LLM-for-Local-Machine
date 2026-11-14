"""验证 Ornith MTP 9B Q8 API 路由、模型别名和监听端口。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "model-text/Ornith-1.5-9B-MTP-GGUF"
sys.path[:0] = [
    str(SERVICE_DIR),
    str(ROOT / "model-text"),
    str(ROOT / "llm_local_runtime/src"),
]


def _load_service_module(name: str, path: Path):
    """从服务目录加载模块，供测试检查其真实 ASGI 行为。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (FileNotFoundError, ModuleNotFoundError):
        sys.modules.pop(name, None)
        return None
    return module


service_app = _load_service_module("ornith_mtp_9b_q8_app", SERVICE_DIR / "app.py")


class OrnithMTP9BQ8ServiceTests(unittest.TestCase):
    def test_mtp_release_delay_override_keeps_sleep_confirmation(self) -> None:
        """MTP 专属缓冲覆盖全局值，释放前仍两次确认休眠。"""
        paths = []

        def handle(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == "/props":
                return httpx.Response(200, json={"is_sleeping": True})
            return httpx.Response(200, json={"choices": []})

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {
                    "CUDA_RELEASE_DELAY": "2",
                    "ORNITH_MTP_9B_CUDA_RELEASE_DELAY": "0.5",
                },
            ),
            patch("llama_cpp_runtime.app.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            app = service_app.create_app(
                transport=httpx.MockTransport(handle),
                lock_path=Path(temporary) / "gpu.lock",
            )
            with TestClient(app) as client:
                response = client.post(service_app.API_ROUTE, json={"messages": []})
            self.assertEqual(response.status_code, 200)
            sleep.assert_any_await(0.5)
            self.assertEqual(paths[-2:], ["/props", "/props"])
            self.assertFalse((Path(temporary) / "gpu.lock.quarantine").exists())

    def test_default_backend_uses_loopback_port_8506(self) -> None:
        self.assertIsNotNone(service_app, "MTP API adapter must be importable")
        seen_urls = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen_urls.append(request.url)
            return httpx.Response(200, json={"status": "ok"})

        with patch.dict(os.environ, {}, clear=True):
            app = service_app.create_app(transport=httpx.MockTransport(handle))
            with TestClient(app) as client:
                response = client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen_urls[0].host, "127.0.0.1")
        self.assertEqual(seen_urls[0].port, 8506)

    def test_chat_route_forces_mtp_model_alias(self) -> None:
        self.assertIsNotNone(service_app, "MTP API adapter must be importable")
        seen_payloads: list[dict[str, object]] = []
        image_content = [
            {"type": "text", "text": "请描述图片。"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,dGVzdA=="},
            },
        ]

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/props":
                return httpx.Response(200, json={"is_sleeping": True})
            seen_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "MTP 回答。"}}]},
            )

        with tempfile.TemporaryDirectory() as temporary:
            app = service_app.create_app(
                backend_base_url="http://127.0.0.1:18506",
                backend_host="127.0.0.1",
                backend_port=18506,
                transport=httpx.MockTransport(handle),
                lock_path=Path(temporary) / "gpu.lock",
                sleep_timeout=0.5,
                sleep_poll_interval=0.01,
                cuda_release_delay=0,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8",
                    json={
                        "model": "client-choice",
                        "messages": [{"role": "user", "content": image_content}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "MTP 回答。")
        self.assertEqual(
            service_app.API_ROUTE,
            "/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8",
        )
        self.assertEqual(service_app.MODEL_ALIAS, "Ornith-1.5-9B-MTP-GGUF-Q8")
        self.assertEqual(seen_payloads[0]["model"], "Ornith-1.5-9B-MTP-GGUF-Q8")
        self.assertEqual(seen_payloads[0]["messages"][0]["content"], image_content)

    def test_main_defaults_to_public_port_8504(self) -> None:
        self.assertIsNotNone(service_app, "MTP API adapter must be importable")
        previous_app = sys.modules.get("app")
        sys.modules["app"] = service_app
        try:
            main_module = _load_service_module("ornith_mtp_9b_q8_main", SERVICE_DIR / "main.py")
            self.assertIsNotNone(main_module, "MTP API entrypoint must be importable")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(main_module.uvicorn, "run") as run,
            ):
                main_module.main()
        finally:
            if previous_app is None:
                sys.modules.pop("app", None)
            else:
                sys.modules["app"] = previous_app

        run.assert_called_once_with(service_app.app, host="0.0.0.0", port=8504)


if __name__ == "__main__":
    unittest.main()
