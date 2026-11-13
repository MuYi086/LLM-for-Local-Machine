"""验证摘要预算不足时停止生成，并按既有 GPU 协议收尾。"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from llm_local_runtime import gpu_runtime_lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-text"))
from llama_cpp_runtime.app import create_app  # noqa: E402


class ContextBudgetTests(unittest.TestCase):
    def test_cancel_during_tokenization_releases_marker_and_lease(self):
        async def run(lock):
            started = asyncio.Event()
            calls = []

            async def handle(request):
                calls.append(request.url.path)
                if request.url.path == "/props":
                    return httpx.Response(
                        200,
                        json={"is_sleeping": True, "default_generation_settings": {"n_ctx": 98304}},
                    )
                if request.url.path == "/apply-template":
                    return httpx.Response(200, json={"prompt": "rendered"})
                started.set()
                await asyncio.Event().wait()

            app = create_app(
                backend_base_url="http://127.0.0.1:18504",
                backend_host="127.0.0.1",
                backend_port=18504,
                transport=httpx.MockTransport(handle),
                lock_path=lock,
                sleep_timeout=0.2,
                sleep_poll_interval=0.001,
                cuda_release_delay=0,
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client,
            ):
                request = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        json={"llm_context_guard": True, "max_tokens": 4096, "messages": []},
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=2)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
            self.assertNotIn("/v1/chat/completions", calls)
            self.assertFalse(Path(str(lock) + ".quarantine").exists())
            with gpu_runtime_lock(lock, "after-cancel", timeout_seconds=0.1):
                pass

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MODEL_TELEMETRY_ENABLED": "false"}),
        ):
            asyncio.run(run(Path(tmp) / "gpu.lock"))

    def invoke(self, count, *, guarded=True, broken=False):
        """使用实际 ASGI 生命周期和假 backend，记录模板/分词/生成调用。"""
        calls = []
        payloads = []

        def handle(request):
            calls.append(request.url.path)
            if request.url.path == "/props":
                return httpx.Response(
                    200, json={"is_sleeping": True, "default_generation_settings": {"n_ctx": 98304}}
                )
            body = json.loads(request.content)
            payloads.append(body)
            if request.url.path == "/apply-template":
                return httpx.Response(503 if broken else 200, json={"prompt": "rendered"})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1] * count})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MODEL_TELEMETRY_ENABLED": "false"}),
        ):
            lock = Path(tmp) / "gpu.lock"
            app = create_app(
                backend_base_url="http://127.0.0.1:18504",
                backend_host="127.0.0.1",
                backend_port=18504,
                transport=httpx.MockTransport(handle),
                lock_path=lock,
                sleep_timeout=0.2,
                sleep_poll_interval=0.001,
                cuda_release_delay=0,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "llm_context_guard": guarded,
                        "max_completion_tokens": 4096,
                        "messages": [{"role": "user", "content": "summary"}],
                    },
                )
            self.assertFalse(Path(str(lock) + ".quarantine").exists())
        return response, calls, payloads

    def test_reproduces_97441_input_without_silent_862_token_truncation(self):
        response, calls, _ = self.invoke(97441)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "context_length_exceeded")
        self.assertIn("97441", response.json()["error"]["message"])
        self.assertNotIn("/v1/chat/completions", calls)

    def test_budget_boundary_and_extension_stripped_before_backend(self):
        response, calls, payloads = self.invoke(98304 - 4096 - 256)
        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/chat/completions", calls)
        self.assertTrue(all("llm_context_guard" not in p for p in payloads))
        tokenize = next(p for p in payloads if "add_special" in p)
        self.assertEqual(
            tokenize, {"content": "rendered", "add_special": True, "parse_special": True}
        )
        response, _, _ = self.invoke(98304 - 4096 - 255)
        self.assertEqual(response.status_code, 400)

    def test_primary_request_does_not_add_model_or_tokenizer_calls(self):
        response, calls, _ = self.invoke(97441, guarded=False)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("/apply-template", calls)
        self.assertNotIn("/tokenize", calls)

    def test_guard_unavailable_fails_closed_and_cleans_lease(self):
        response, calls, _ = self.invoke(1, broken=True)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("/v1/chat/completions", calls)
