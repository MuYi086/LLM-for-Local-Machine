"""Index-Echo S2ST HTTP 契约与启动接线的无模型回归。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "Index-Echo-S2ST-9B-GGUF"


def load_module(name: str, path: Path):
    """以唯一模块名加载服务，避免多个 main.py 互相覆盖。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IndexEchoS2stMigrationTests(unittest.TestCase):
    """隔离文件和模型边界，验证翻译请求编排及启动登记。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="index-echo-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "PROMPTS_DIR": str(self.root / "storage/clone"),
            "TIMBRE_STORAGE_DIR": str(self.root / "storage/timbre"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "cache/gpu.lock"),
            "INDEX_ECHO_S2ST_MODEL_DIR": str(self.root / "model"),
            "INDEX_ECHO_S2ST_MODEL_PATH": str(self.root / "model/index.gguf"),
            "INDEX_ECHO_S2ST_WORKER_TMP_DIR": str(self.root / "cache/worker"),
            "INDEX_ECHO_S2ST_LLAMA_SERVER_BIN": "llama-server-not-installed",
            "INDEX_ECHO_S2ST_REQUEST_TIMEOUT": "5",
            "INDEX_ECHO_S2ST_STARTUP_TIMEOUT": "5",
            "INDEX_ECHO_S2ST_PORT": "8382",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
            "CUDA_RELEASE_DELAY": "0",
        }
        preserved = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("INDEX_ECHO_S2ST_")
            or key in {"STORAGE_DIR", "PROMPTS_DIR", "CLONE_STORAGE_DIR", "TIMBRE_STORAGE_DIR"}
        }
        with (
            patch.dict(os.environ, {**preserved, **environment}, clear=True),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("index_echo_s2st_main_for_tests", SERVICE_DIR / "main.py")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)
        self.audio_path = "source.wav"
        self.main.reference_store.clone_path(self.audio_path).write_bytes(b"source audio")

    @contextmanager
    def fake_gpu_lock(self, label: str):
        """GPU 队列替身不调用 nvidia-smi 或文件锁。"""
        yield SimpleNamespace(as_dict=lambda: {"label": label, "worker_seconds": 0.1})

    def test_route_transcribes_translates_and_preserves_reference_voice(self) -> None:
        self.main.manager.check_runtime_ready = Mock()
        self.main.manager.build_worker_payload = Mock(
            return_value={"text": "你好", "source_language": "zh", "target_language": "en"}
        )
        self.main.manager.run_worker = Mock(return_value="Hello there.")
        transcribe = Mock(return_value={"text": "你好", "language": "Chinese"})
        synthesize = Mock(return_value=b"RIFF\x00\x00\x00\x00WAVE")
        self.main.transcribe_audio = transcribe
        self.main.synthesize_audio = synthesize

        with (
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main, "wait_after_cuda_release"),
        ):
            response = self.client.post(
                "/v1/indexTranslate/echoS2ST",
                json={
                    "audio_path": self.audio_path,
                    "target_language": "en",
                    "max_new_tokens": 128,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-source-language"], "zh")
        self.assertEqual(response.headers["x-target-language"], "en")
        self.assertEqual(response.content, b"RIFF\x00\x00\x00\x00WAVE")
        transcribe.assert_called_once_with(
            self.main.reference_store.prompt_audio_path(self.audio_path), self.audio_path, None
        )
        self.main.manager.build_worker_payload.assert_called_once_with("你好", "zh", "en", 128)
        self.main.manager.run_worker.assert_called_once()
        synthesize.assert_called_once_with("Hello there.", "en", self.audio_path)

    def test_rejects_missing_or_unsupported_audio_before_asr(self) -> None:
        self.main.transcribe_audio = Mock()
        missing = self.client.post(
            "/v1/indexTranslate/echoS2ST",
            json={"audio_path": "missing.wav", "target_language": "en"},
        )
        self.assertEqual(missing.status_code, 404)

        unsupported_path = "source.txt"
        self.main.reference_store.clone_path(unsupported_path).write_text("not audio")
        unsupported = self.client.post(
            "/v1/indexTranslate/echoS2ST",
            json={"audio_path": unsupported_path, "target_language": "en"},
        )
        self.assertEqual(unsupported.status_code, 422)
        self.main.transcribe_audio.assert_not_called()

    def test_rejects_same_language_before_gpu_and_tts(self) -> None:
        self.main.manager.check_runtime_ready = Mock()
        self.main.transcribe_audio = Mock(return_value={"text": "你好", "language": "Chinese"})
        self.main.manager.build_worker_payload = Mock()
        self.main.synthesize_audio = Mock()
        gpu_lock = Mock(
            side_effect=AssertionError("same-language request must not acquire GPU lock")
        )

        with patch.object(self.main, "gpu_runtime_lock", gpu_lock):
            response = self.client.post(
                "/v1/indexTranslate/echoS2ST",
                json={"audio_path": self.audio_path, "target_language": "zh"},
            )

        self.assertEqual(response.status_code, 422)
        gpu_lock.assert_not_called()
        self.main.manager.build_worker_payload.assert_not_called()
        self.main.synthesize_audio.assert_not_called()

    def test_startup_and_quality_gate_register_service(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        self.assertIn('INDEX_ECHO_S2ST_PORT="${INDEX_ECHO_S2ST_PORT:-8382}"', startup)
        self.assertIn(
            'INDEX_ECHO_S2ST_PROJECT_DIR="${INDEX_ECHO_S2ST_PROJECT_DIR:-$PROJECT_DIR/Index-Echo-S2ST-9B-GGUF}"',
            startup,
        )
        self.assertIn("/v1/indexTranslate/echoS2ST", startup)
        self.assertIn('setsid uv run --no-sync --project "$INDEX_ECHO_S2ST_PROJECT_DIR"', startup)
        cleanup = startup.split("cleanup() {", maxsplit=1)[1].split("trap cleanup", maxsplit=1)[0]
        wait_line = next(line for line in startup.splitlines() if line.startswith("wait -n "))
        self.assertIn('"$index_echo_s2st_pid"', cleanup)
        self.assertIn('"$index_echo_s2st_pid"', wait_line)
        self.assertIn("Index-Echo-S2ST-9B-GGUF", quality_gate)


if __name__ == "__main__":
    unittest.main()
