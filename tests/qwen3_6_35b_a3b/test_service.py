"""验证 Qwen3.6 llama.cpp API 的路由和 GPU 租约合同。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model-text"), str(ROOT / "llm_local_runtime/src")]

from qwen3_6_35b_a3b.app import API_ROUTE, MODEL_ALIAS, create_qwen_app  # noqa: E402


class QwenServiceTests(unittest.TestCase):
    def _app(self, lock_path: Path, handler):
        return create_qwen_app(
            transport=httpx.MockTransport(handler),
            lock_path=lock_path,
            sleep_poll_interval=0.001,
            sleep_timeout=0.05,
            cuda_release_delay=0,
        )

    def test_exact_route_forwards_fixed_alias_to_qwen_loopback_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            received: list[httpx.Request] = []

            def handle(request: httpx.Request) -> httpx.Response:
                received.append(request)
                if request.url.path == "/props":
                    return httpx.Response(200, json={"is_sleeping": True})
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "答案是 2。"}}]},
                )

            app = self._app(lock_path, handle)
            with TestClient(app) as client:
                response = client.post(
                    API_ROUTE,
                    json={
                        "model": "caller-controlled",
                        "messages": [{"role": "user", "content": "1+1"}],
                    },
                )
                wrong_route = client.post("/v1/qwen/other", json={"messages": []})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "答案是 2。")
            self.assertEqual(wrong_route.status_code, 404)
            chat_request = next(
                request for request in received if request.url.path == "/v1/chat/completions"
            )
            self.assertEqual(str(chat_request.url), "http://127.0.0.2:8400/v1/chat/completions")
            self.assertEqual(json.loads(chat_request.content)["model"], MODEL_ALIAS)
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_streaming_keeps_exact_route_and_clears_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            chunks = b'data: {"choices":[]}' + b"\n\ndata: [DONE]\n\n"

            def handle(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/props":
                    return httpx.Response(200, json={"is_sleeping": True})
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(chunks),
                )

            with TestClient(self._app(lock_path, handle)) as client:
                response = client.post(API_ROUTE, json={"messages": [], "stream": True})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, chunks)
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_health_route_does_not_load_model_or_take_gpu_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            app = self._app(
                lock_path,
                lambda request: httpx.Response(200, json={"status": "ok"}),
            )
            with TestClient(app) as client:
                response = client.get("/v1/health")

            self.assertEqual(response.status_code, 200)
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())


if __name__ == "__main__":
    unittest.main()
