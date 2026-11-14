"""验证 Ornith-1.5-9B-GGUF 适配器的路由和模型别名。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "model-text/Ornith-1.5-9B-GGUF"
sys.path[:0] = [
    str(SERVICE_DIR),
    str(ROOT / "model-text"),
    str(ROOT / "llm_local_runtime/src"),
]


def _load_service_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载服务模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


service_app = _load_service_module("ornith_1_5_9b_gguf_app", SERVICE_DIR / "app.py")


class Ornith9BServiceTests(unittest.TestCase):
    def test_default_backend_uses_loopback_port_8505(self) -> None:
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
        self.assertEqual(seen_urls[0].port, 8505)

    def test_app_registers_exact_route_and_forces_9b_model_alias(self) -> None:
        seen_payloads: list[dict[str, object]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/props":
                return httpx.Response(200, json={"is_sleeping": True})
            seen_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "你好。"}}]},
            )

        with tempfile.TemporaryDirectory() as temporary:
            app = service_app.create_app(
                backend_base_url="http://127.0.0.1:18504",
                backend_host="127.0.0.1",
                backend_port=18504,
                transport=httpx.MockTransport(handle),
                lock_path=Path(temporary) / "gpu.lock",
                sleep_timeout=0.5,
                sleep_poll_interval=0.01,
                cuda_release_delay=0,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/v1/ornith-ai/Ornith-1.5-9B-GGUF",
                    json={
                        "model": "client-choice",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好。")
        self.assertEqual(service_app.API_ROUTE, "/v1/ornith-ai/Ornith-1.5-9B-GGUF")
        self.assertEqual(service_app.MODEL_ALIAS, "Ornith-1.5-9B-GGUF")
        self.assertEqual(seen_payloads[0]["model"], service_app.MODEL_ALIAS)

    def test_main_defaults_to_public_port_8503(self) -> None:
        previous_app = sys.modules.get("app")
        sys.modules["app"] = service_app
        try:
            main_module = _load_service_module("ornith_1_5_9b_gguf_main", SERVICE_DIR / "main.py")
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

        run.assert_called_once_with(service_app.app, host="0.0.0.0", port=8503)


if __name__ == "__main__":
    unittest.main()
