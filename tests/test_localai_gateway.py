"""LocalAI 配置、HTTP 传输和后端生命周期的无模型测试。"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
sys.path.insert(0, str(REPOSITORY_DIR / "llm_local_runtime/src"))

from inference_gateway.execution import (  # noqa: E402
    execute_localai_operation,
    execute_localai_worker_file,
    execute_localai_worker_file_reference,
)
from inference_gateway.worker_protocol import (  # noqa: E402
    decode_worker_request,
    encode_worker_response,
)

from inference_gateway import (  # noqa: E402
    BackendQuarantinedError,
    GatewayConfigurationError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    load_gateway_config,
)
from llm_local_runtime import (  # noqa: E402
    GpuLockTimeoutError,
    GpuQuarantinedError,
    gpu_runtime_lock,
)


def config(**overrides: str):
    """生成隔离测试使用的 LocalAI 配置。"""
    environment = {
        "LLM_LOCALAI_ENABLED": "true",
        "LLM_LOCAL_QWEN3_TTS_PROVIDER": "localai",
        "LLM_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
        "GPU_LOCK_FILE": "/tmp/llm-local-test.lock",
        "CUDA_RELEASE_DELAY": "0",
        **overrides,
    }
    return load_gateway_config(
        service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
        default_model="llm-local.qwen3-tts.1p7b-base",
        request_timeout=10,
        environ=environment,
    )


class LocalAIGatewayTests(unittest.TestCase):
    """验证网关拒绝不安全配置并且总在推理后尝试停止 backend。"""

    def test_worker_client_uses_versioned_completion_envelope(self) -> None:
        client = LocalAIClient(config(LLM_LOCAL_QWEN3_TTS_MODEL="llm-local.bridge.smoke"))
        captured: dict[str, object] = {}

        def capture_request(method: str, path: str, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            captured["method"] = method
            captured["path"] = path
            request = json.loads(kwargs["body"])
            decoded = decode_worker_request(
                request["prompt"], model_identity="llm-local.bridge.smoke"
            )
            captured["decoded"] = decoded
            result = encode_worker_response(
                model=decoded.model,
                operation=decoded.operation,
                request_id=decoded.request_id,
                result={"kind": "json", "value": {"text": "ok"}},
            ).decode("utf-8")
            return {"choices": [{"text": result}]}

        with patch.object(client, "_request", side_effect=capture_request):
            result, elapsed = client.run_worker(
                operation="run",
                payload={"worker_input": {"text": "hello"}},
            )

        request = json.loads(captured["body"])
        decoded = captured["decoded"]
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/v1/completions")
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(decoded.payload["worker_input"]["text"], "hello")
        self.assertEqual(result["kind"], "json")
        self.assertEqual(result["value"]["text"], "ok")
        self.assertGreaterEqual(elapsed, 0)

    def test_shared_file_reference_is_copied_from_scoped_id_and_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "shared"
            artifact_root.mkdir()
            artifact_id = "a" * 32
            shared_file = artifact_root / f"{artifact_id}.zip"
            shared_file.write_bytes(b"large archive payload")
            with patch(
                "inference_gateway.execution.execute_localai_worker",
                return_value=(
                    {
                        "kind": "file_ref",
                        "artifact_id": artifact_id,
                        "suffix": ".zip",
                        "media_type": "application/zip",
                        "size_bytes": len(b"large archive payload"),
                    },
                    1.0,
                ),
            ):
                copied, elapsed = execute_localai_worker_file_reference(
                    config=config(),
                    operation="run",
                    payload={"worker_input": {}},
                    output_dir=root / "service-output",
                    artifact_dir=artifact_root,
                    suffix=".zip",
                    media_type="application/zip",
                )

            self.assertEqual(copied.read_bytes(), b"large archive payload")
            self.assertGreaterEqual(elapsed, 0)
            self.assertFalse(shared_file.exists())
            copied.unlink()

    def test_shared_file_reference_rejects_invalid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "inference_gateway.execution.execute_localai_worker",
                return_value=(
                    {
                        "kind": "file_ref",
                        "artifact_id": "../outside",
                        "suffix": ".zip",
                        "media_type": "application/zip",
                        "size_bytes": 1,
                    },
                    0.1,
                ),
            ):
                with self.assertRaisesRegex(LocalAIRequestError, "引用格式"):
                    execute_localai_worker_file_reference(
                        config=config(),
                        operation="run",
                        payload={"worker_input": {}},
                        output_dir=Path(temporary) / "out",
                        artifact_dir=Path(temporary) / "shared",
                        suffix=".zip",
                        media_type="application/zip",
                    )

    def test_worker_file_helper_validates_and_stages_artifact(self) -> None:
        class FakeLocalAIClient:
            def run_worker(self, **kwargs: object) -> tuple[dict[str, object], float]:
                self.payload = kwargs["payload"]
                content = b"test artifact"
                return (
                    {
                        "kind": "file",
                        "media_type": "application/zip",
                        "data_base64": base64.b64encode(content).decode("ascii"),
                        "size_bytes": len(content),
                    },
                    0.25,
                )

            def shutdown_model(self) -> None:
                self.shutdown = True

        client = FakeLocalAIClient()
        with tempfile.TemporaryDirectory() as temporary:
            output_path, elapsed = execute_localai_worker_file(
                config=config(GPU_LOCK_FILE=f"{temporary}/gpu.lock"),
                operation="infer",
                payload={"worker_input": {"text": "hello"}},
                output_dir=Path(temporary),
                suffix=".zip",
                media_type="application/zip",
                client=client,
            )
            self.assertEqual(output_path.read_bytes(), b"test artifact")
            self.assertEqual(elapsed, 0.25)
            self.assertEqual(client.payload["media_type"], "application/zip")
            self.assertTrue(client.shutdown)

    def test_native_provider_is_default_and_external_url_is_rejected(self) -> None:
        native = load_gateway_config(
            service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
            default_model="llm-local.qwen3-tts.1p7b-base",
            request_timeout=10,
            environ={},
        )
        self.assertEqual(native.provider, "native")
        with self.assertRaises(GatewayConfigurationError):
            config(LLM_LOCALAI_BASE_URL="http://192.168.1.10:8400")
        with self.assertRaises(GatewayConfigurationError):
            load_gateway_config(
                service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
                default_model="llm-local.qwen3-tts.1p7b-base",
                request_timeout=10,
                environ={"LLM_LOCAL_QWEN3_TTS_PROVIDER": "localai"},
            )

    def test_nonpositive_gpu_wait_is_rejected(self) -> None:
        with self.assertRaises(GatewayConfigurationError):
            config(GPU_LOCK_WAIT_TIMEOUT="0")

    def test_cuda_release_delay_must_be_finite_and_nonnegative(self) -> None:
        for value in ("-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(GatewayConfigurationError):
                config(CUDA_RELEASE_DELAY=value)

    def test_tts_client_maps_qwen3_clone_fields_and_validates_wav_response(self) -> None:
        captured: list[dict[str, object]] = []
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                captured.append({"path": self.path, "body": body})
                if self.path == "/v1/audio/speech":
                    encoded = response_wav
                    content_type = "audio/wav"
                else:
                    encoded = b"stopped"
                    content_type = "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
                default_model="llm-local.qwen3-tts.1p7b-base",
                request_timeout=3,
                environ={
                    "LLM_LOCALAI_ENABLED": "true",
                    "LLM_LOCAL_QWEN3_TTS_PROVIDER": "localai",
                    "LLM_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    "GPU_LOCK_FILE": "/tmp/qwen3-tts-test.lock",
                },
            )
            client = LocalAIClient(client_config)
            with tempfile.TemporaryDirectory() as temporary:
                reference = Path(temporary) / "reference.wav"
                reference.write_bytes(b"ref")
                audio_bytes, elapsed = client.synthesize_speech(
                    text="你好。",
                    voice_path=reference,
                    language="Chinese",
                    reference_text="参考文本。",
                    x_vector_only=False,
                    max_new_tokens=512,
                    top_p=0.8,
                    temperature=0.7,
                )
                client.shutdown_model()
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        body = json.loads(captured[0]["body"])
        self.assertEqual(captured[0]["path"], "/v1/audio/speech")
        self.assertEqual(body["model"], "llm-local.qwen3-tts.1p7b-base")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["voice"], str(reference))
        self.assertEqual(body["language"], "Chinese")
        self.assertEqual(
            body["params"],
            {
                "max_tokens": "512",
                "x_vector_only_mode": "false",
                "reference_text": "参考文本。",
                "top_p": "0.8",
                "temperature": "0.7",
            },
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(captured[1]["path"], "/backend/shutdown")

    def test_voice_design_client_maps_instruction_to_localai_speech_request(self) -> None:
        client_config = load_gateway_config(
            service_provider_env="LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER",
            default_model="llm-local.qwen3-tts.1p7b-voicedesign",
            request_timeout=10,
            environ={
                "LLM_LOCALAI_ENABLED": "true",
                "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER": "localai",
                "LLM_LOCALAI_BASE_URL": "http://127.0.0.1:8400",
            },
        )
        client = LocalAIClient(client_config)
        response_wav = b"RIFF\x04\x00\x00\x00WAVE"

        with patch.object(client, "_request", return_value=(response_wav, "audio/wav")) as request:
            audio_bytes, elapsed = client.design_voice(
                text="你好。",
                instruction="成年女性，温柔清晰。",
                language="Chinese",
                max_new_tokens=768,
                top_p=0.85,
                temperature=0.75,
            )

        body = json.loads(request.call_args.kwargs["body"])
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/audio/speech"))
        self.assertEqual(body["model"], "llm-local.qwen3-tts.1p7b-voicedesign")
        self.assertEqual(body["input"], "你好。")
        self.assertEqual(body["instructions"], "成年女性，温柔清晰。")
        self.assertEqual(body["language"], "Chinese")
        self.assertEqual(
            body["params"],
            {"max_tokens": "768", "top_p": "0.85", "temperature": "0.75"},
        )
        self.assertEqual(audio_bytes, response_wav)
        self.assertGreaterEqual(elapsed, 0)

    def test_model_ready_checks_exact_registered_model_without_loading(self) -> None:
        registered_models = [{"id": "llm-local.qwen3-tts.1p7b-base"}]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                encoded = json.dumps({"object": "list", "data": registered_models}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
                default_model="llm-local.qwen3-tts.1p7b-base",
                request_timeout=3,
                environ={
                    "LLM_LOCALAI_ENABLED": "true",
                    "LLM_LOCAL_QWEN3_TTS_PROVIDER": "localai",
                    "LLM_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                },
            )
            client = LocalAIClient(client_config)
            self.assertTrue(client.model_ready())
            registered_models[:] = [{"id": "different-model"}]
            self.assertFalse(client.model_ready())
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_response_trickle_cannot_extend_total_request_timeout(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = b"slow-response"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    try:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                    time.sleep(0.04)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client_config = load_gateway_config(
                service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
                default_model="llm-local.qwen3-tts.1p7b-base",
                request_timeout=2,
                environ={
                    "LLM_LOCALAI_ENABLED": "true",
                    "LLM_LOCAL_QWEN3_TTS_PROVIDER": "localai",
                    "LLM_LOCALAI_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                },
            )
            client = LocalAIClient(client_config)
            started = time.monotonic()
            with self.assertRaises(LocalAIUnavailableError):
                client._request("GET", "/slow", timeout=0.15)
            self.assertLess(time.monotonic() - started, 0.4)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_success_clears_crash_marker_after_backend_shutdown(self) -> None:
        class Client:
            def shutdown_model(self) -> None:
                return

        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            client = Client()

            def run_operation(_: LocalAIClient) -> str:
                self.assertTrue(Path(f"{lock_path}.quarantine").is_file())
                return "complete"

            with patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock):
                result = execute_localai_operation(
                    config=replace(config(), gpu_lock_file=lock_path),
                    operation=run_operation,
                    client=client,
                )

            self.assertEqual(result, "complete")
            self.assertFalse(Path(f"{lock_path}.quarantine").exists())

    def test_cuda_release_delay_runs_after_shutdown_while_marker_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")
            marker = Path(f"{lock_path}.quarantine")
            events: list[str] = []

            class Client:
                def shutdown_model(self) -> None:
                    assert marker.is_file()
                    events.append("shutdown")

            def fake_sleep(delay: float) -> None:
                self.assertEqual(delay, 0.25)
                self.assertTrue(marker.is_file())
                events.append("release-delay")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            with (
                patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock),
                patch("inference_gateway.execution.time.sleep", side_effect=fake_sleep),
            ):
                result = execute_localai_operation(
                    config=replace(config(), gpu_lock_file=lock_path, cuda_release_delay=0.25),
                    operation=lambda _: "complete",
                    client=Client(),
                )

            self.assertEqual(result, "complete")
            self.assertEqual(events, ["shutdown", "release-delay"])
            self.assertFalse(marker.exists())

    def test_http_success_cannot_clear_worker_cleanup_failure_marker(self) -> None:
        class Client:
            def shutdown_model(self) -> None:
                return

        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")
            marker = Path(f"{lock_path}.quarantine")

            def operation(_: LocalAIClient) -> str:
                marker.write_text('{"reason":"worker_process_exit_unconfirmed"}')
                return "result"

            with self.assertRaises(BackendQuarantinedError):
                execute_localai_operation(
                    config=replace(config(), gpu_lock_file=lock_path),
                    operation=operation,
                    client=Client(),
                )
            self.assertEqual(
                json.loads(marker.read_text())["reason"], "worker_process_exit_unconfirmed"
            )

    def test_failed_backend_shutdown_preserves_quarantine_marker(self) -> None:
        class Client:
            def shutdown_model(self) -> None:
                raise RuntimeError("backend still running")

        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")

            @contextmanager
            def fake_gpu_lock(*args: object, **kwargs: object):
                yield None

            with patch("inference_gateway.execution.gpu_runtime_lock", fake_gpu_lock):
                with self.assertRaises(BackendQuarantinedError):
                    execute_localai_operation(
                        config=replace(config(), gpu_lock_file=lock_path),
                        operation=lambda _: "possibly-running",
                        client=Client(),
                    )

            marker_path = Path(f"{lock_path}.quarantine")
            self.assertTrue(marker_path.is_file())
            with self.assertRaises(GpuQuarantinedError):
                with gpu_runtime_lock(lock_path, "native-after-failure", timeout_seconds=0.1):
                    self.fail("隔离状态下不应获取 GPU 租约")

    def test_native_request_cannot_enter_during_localai_gpu_lease(self) -> None:
        entered_inference = threading.Event()
        finish_inference = threading.Event()
        finished_request = threading.Event()
        errors: list[BaseException] = []

        class Client:
            def shutdown_model(self) -> None:
                return

        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")

            def run_request() -> None:
                try:
                    execute_localai_operation(
                        config=replace(config(), gpu_lock_file=lock_path),
                        operation=lambda _: (
                            entered_inference.set(),
                            finish_inference.wait(timeout=2),
                            "complete",
                        )[-1],
                        client=Client(),
                    )
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    finished_request.set()

            thread = threading.Thread(target=run_request, daemon=True)
            thread.start()
            self.assertTrue(entered_inference.wait(timeout=2))
            try:
                with self.assertRaises(GpuLockTimeoutError):
                    with gpu_runtime_lock(lock_path, "native-request", timeout_seconds=0.05):
                        self.fail("LocalAI 持有 GPU 租约时，native 请求不应进入。")
            finally:
                finish_inference.set()
            self.assertTrue(finished_request.wait(timeout=2))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
