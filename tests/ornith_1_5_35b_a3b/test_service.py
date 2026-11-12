"""验证 Ornith 聊天 API 的路由、共享 GPU 租约和卸载确认合同。"""

from __future__ import annotations

import asyncio
import fcntl
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model-text"), str(ROOT / "llm_local_runtime/src")]

from ornith_1_5_35b_a3b.app import API_ROUTE, MODEL_ALIAS, create_app  # noqa: E402


class OrnithServiceTests(unittest.TestCase):
    def setUp(self):
        self.telemetry_temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(
            patch.dict(
                "os.environ",
                {"MODEL_TELEMETRY_DIR": self.telemetry_temp, "MODEL_TELEMETRY_ENABLED": "true"},
            )
        )

    def test_telemetry_records_native_timings_and_lease_cleanup(self):
        def handle(request):
            if request.url.path == "/props":
                return httpx.Response(200, json={"is_sleeping": True})
            payload = json.loads(request.content)
            self.assertTrue(payload["stream_options"]["include_usage"])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(
                    b'data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7},"timings":{"prompt_ms":10,"predicted_ms":20,"draft_n":8,"draft_n_accepted":5}}\n\ndata: [DONE]\n\n'
                ),
            )

        with tempfile.TemporaryDirectory() as root:
            app = self._app(Path(root) / "gpu.lock", handle)
            with TestClient(app) as client:
                response = client.post(API_ROUTE, json={"messages": [], "stream": True})
            self.assertEqual(response.status_code, 200)
            row = json.loads(next(Path(self.telemetry_temp).glob("requests-*.jsonl")).read_text())
            self.assertEqual(row["usage"]["prompt_tokens"], 42)
            self.assertEqual(row["timings"]["draft_n_accepted"], 5)
            stages = row["stages_ms"]
            self.assertLessEqual(stages["queue_enter"], stages["lease_acquired"])
            self.assertLessEqual(stages["backend_send"], stages["cleanup_end"])
            self.assertLessEqual(stages["cleanup_end"], stages["lease_released"])
            self.assertEqual(response.headers["x-model-request-id"], row["request_id"])

    def test_disconnect_during_cleanup_preserves_lease_until_sleep_and_allows_next_request(self):
        async def scenario(lock_path: Path) -> None:
            cleanup_started = asyncio.Event()
            chat_count = 0
            marker_path = Path(f"{lock_path}.quarantine")

            async def handle(request: httpx.Request) -> httpx.Response:
                nonlocal chat_count
                if request.url.path == "/props":
                    if chat_count == 1 and not cleanup_started.is_set():
                        self.assertTrue(marker_path.exists())
                        with lock_path.open("a+") as probe:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        cleanup_started.set()
                        await asyncio.sleep(0.02)
                    return httpx.Response(200, json={"is_sleeping": True})
                chat_count += 1
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(b"data: [DONE]\n\n"),
                )

            app = self._app(lock_path, handle, sleep_timeout=0.5)
            scope = {
                "type": "http",
                "method": "POST",
                "path": API_ROUTE,
                "headers": [(b"content-type", b"application/json")],
                "query_string": b"",
                "http_version": "1.1",
                "scheme": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
            }
            body_sent = False

            async def receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": b'{"messages":[],"stream":true}'}
                await cleanup_started.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                pass

            async with app.router.lifespan_context(app):
                await asyncio.wait_for(app(scope, receive, send), timeout=2)
                self.assertTrue(cleanup_started.is_set())
                self.assertFalse(marker_path.exists())
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client:
                    response = await client.post(API_ROUTE, json={"messages": []})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(chat_count, 2)
                self.assertFalse(marker_path.exists())

        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(scenario(Path(temporary) / "gpu-runtime.lock"))

    def test_sleep_confirmation_retries_without_clearing_quarantine_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            states = iter([True, True, False, True, True])
            props_count = 0

            def handle(request: httpx.Request) -> httpx.Response:
                nonlocal props_count
                self.assertTrue(marker_path.exists())
                if request.url.path == "/props":
                    props_count += 1
                    return httpx.Response(200, json={"is_sleeping": next(states)})
                return httpx.Response(200, json={"choices": []})

            with TestClient(self._app(lock_path, handle)) as client:
                response = client.post(API_ROUTE, json={"messages": []})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(props_count, 5)
            self.assertFalse(marker_path.exists())

    def _app(self, lock_path: Path, handler, **kwargs):
        options = {
            "sleep_poll_interval": 0.001,
            "sleep_timeout": 0.05,
            "cuda_release_delay": 0,
            **kwargs,
        }
        return create_app(
            transport=httpx.MockTransport(handler),
            lock_path=lock_path,
            **options,
        )

    def test_chat_route_forwards_payload_and_holds_shared_lease_until_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            received: list[httpx.Request] = []

            def handle(request: httpx.Request) -> httpx.Response:
                received.append(request)
                if request.url.path == "/props":
                    return httpx.Response(200, json={"is_sleeping": True})
                return httpx.Response(
                    200,
                    json={
                        "object": "chat.completion",
                        "choices": [{"message": {"role": "assistant", "content": "你好。"}}],
                    },
                )

            app = self._app(lock_path, handle)
            with TestClient(app) as client:
                response = client.post(
                    API_ROUTE, json={"messages": [{"role": "user", "content": "你好"}]}
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好。")
            chat_request = next(
                request for request in received if request.url.path.endswith("/chat/completions")
            )
            self.assertEqual(str(chat_request.url), "http://127.0.0.1:8403/v1/chat/completions")
            self.assertEqual(json.loads(chat_request.content)["model"], MODEL_ALIAS)
            self.assertTrue(any(request.url.path == "/props" for request in received))
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_chat_route_streams_response_and_clears_marker_after_stream(self) -> None:
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

            app = self._app(lock_path, handle)
            with TestClient(app) as client:
                response = client.post(
                    API_ROUTE,
                    json={"model": "explicit-alias", "messages": [], "stream": True},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/event-stream")
            self.assertEqual(response.content, chunks)
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_chat_waits_for_stable_sleeping_state_before_waking_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            paths: list[str] = []
            props_count = 0

            def handle(request: httpx.Request) -> httpx.Response:
                nonlocal props_count
                paths.append(request.url.path)
                if request.url.path == "/props":
                    props_count += 1
                    return httpx.Response(
                        200,
                        json={"is_sleeping": props_count >= 2},
                    )
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "就绪。"}}]},
                )

            app = self._app(lock_path, handle)
            with TestClient(app) as client:
                response = client.post(API_ROUTE, json={"messages": []})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(paths[0:2], ["/props", "/props"])
            self.assertLess(paths.index("/v1/chat/completions"), len(paths) - 1)

    def test_unconfirmed_sleep_preserves_quarantine_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            app = self._app(
                lock_path,
                lambda request: httpx.Response(
                    200,
                    json={"is_sleeping": False}
                    if request.url.path == "/props"
                    else {"choices": [{"message": {"content": "answer"}}]},
                ),
                sleep_timeout=0.005,
            )

            with TestClient(app) as client:
                response = client.post(API_ROUTE, json={"messages": []})

            self.assertEqual(response.status_code, 503)
            marker_path = Path(f"{lock_path}.quarantine")
            self.assertTrue(marker_path.exists())
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["provider"], "llama.cpp")
            self.assertEqual(marker["model"], MODEL_ALIAS)

    def test_existing_quarantine_blocks_request_before_backend_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text('{"provider":"llama.cpp"}', encoding="utf-8")
            received: list[httpx.Request] = []
            app = self._app(
                lock_path, lambda request: received.append(request) or httpx.Response(200)
            )

            with TestClient(app) as client:
                response = client.post(API_ROUTE, json={"messages": []})

            self.assertEqual(response.status_code, 503)
            self.assertEqual(received, [])
            self.assertTrue(marker_path.exists())

    def test_upstream_connection_error_returns_service_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"

            def handle(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/props":
                    return httpx.Response(200, json={"is_sleeping": True})
                raise httpx.ConnectError("connection refused", request=request)

            app = self._app(lock_path, handle)
            with TestClient(app) as client:
                response = client.post(API_ROUTE, json={"messages": []})

            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"], "llama.cpp 服务暂不可用。")
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_health_route_does_not_require_the_gpu_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            app = self._app(
                lock_path,
                lambda request: httpx.Response(200, json={"status": "ok"}),
            )
            paths = {route.path for route in app.routes}

            self.assertIn(API_ROUTE, paths)
            self.assertIn("/v1/health", paths)

    def test_custom_route_forwards_model_alias_to_loopback_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            route = "/v1/custom/ornith"
            alias = "Ornith-custom-test"
            received: list[httpx.Request] = []

            def handle(request: httpx.Request) -> httpx.Response:
                received.append(request)
                if request.url.path == "/props":
                    return httpx.Response(200, json={"is_sleeping": True})
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "2"}}]},
                )

            app = self._app(
                lock_path,
                handle,
                api_route=route,
                model_alias=alias,
                backend_base_url="http://127.0.0.1:8403",
                backend_host="127.0.0.1",
                backend_port=8403,
            )
            with TestClient(app) as client:
                response = client.post(
                    route, json={"messages": [{"role": "user", "content": "1+1"}]}
                )

            self.assertEqual(response.status_code, 200)
            chat_request = next(
                request for request in received if request.url.path == "/v1/chat/completions"
            )
            self.assertEqual(str(chat_request.url), "http://127.0.0.1:8403/v1/chat/completions")
            self.assertEqual(json.loads(chat_request.content)["model"], alias)

    def test_custom_backend_identity_is_recorded_when_sleep_cannot_be_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            route = "/v1/custom/ornith"
            alias = "Ornith-custom-test"
            app = self._app(
                lock_path,
                lambda request: httpx.Response(
                    200,
                    json={"is_sleeping": False}
                    if request.url.path == "/props"
                    else {"choices": [{"message": {"content": "2"}}]},
                ),
                api_route=route,
                model_alias=alias,
                backend_base_url="http://127.0.0.1:8403",
                backend_host="127.0.0.1",
                backend_port=8403,
                backend_pid=1234,
                backend_pgid=1234,
                backend_executable="/opt/llama-server",
                sleep_timeout=0.005,
            )

            with TestClient(app) as client:
                response = client.post(route, json={"messages": []})

            self.assertEqual(response.status_code, 503)
            marker = json.loads(Path(f"{lock_path}.quarantine").read_text(encoding="utf-8"))
            self.assertEqual(marker["model"], alias)
            self.assertEqual(marker["host"], "127.0.0.1")
            self.assertEqual(marker["port"], 8403)
            self.assertEqual(marker["pid"], 1234)
            self.assertEqual(marker["executable"], "/opt/llama-server")


if __name__ == "__main__":
    unittest.main()
