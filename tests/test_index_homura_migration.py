"""Index-Homura 语音翻译 HTTP 契约与启动接线的无模型回归。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "Index-Homura-9B-GGUF"


def load_module(name: str, path: Path):
    """以唯一模块名加载服务，避免多个 main.py 互相覆盖。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IndexHomuraMigrationTests(unittest.TestCase):
    """隔离文件和模型边界，验证音节约束翻译与启动登记。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="index-homura-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "PROMPTS_DIR": str(self.root / "storage/clone"),
            "TIMBRE_STORAGE_DIR": str(self.root / "storage/timbre"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "cache/gpu.lock"),
            "INDEX_HOMURA_MODEL_DIR": str(self.root / "model"),
            "INDEX_HOMURA_MODEL_PATH": str(self.root / "model/index.gguf"),
            "INDEX_HOMURA_WORKER_TMP_DIR": str(self.root / "cache/worker"),
            "INDEX_HOMURA_LLAMA_SERVER_BIN": "llama-server-not-installed",
            "INDEX_HOMURA_REQUEST_TIMEOUT": "5",
            "INDEX_HOMURA_STARTUP_TIMEOUT": "5",
            "INDEX_HOMURA_PORT": "8383",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
            "CUDA_RELEASE_DELAY": "0",
        }
        preserved = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("INDEX_HOMURA_")
            or key in {"STORAGE_DIR", "PROMPTS_DIR", "CLONE_STORAGE_DIR", "TIMBRE_STORAGE_DIR"}
        }
        with (
            patch.dict(os.environ, {**preserved, **environment}, clear=True),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("index_homura_main_for_tests", SERVICE_DIR / "main.py")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)
        self.audio_path = "source.wav"
        self.main.reference_store.clone_path(self.audio_path).write_bytes(b"source audio")
        self.main.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.main.MODEL_PATH.touch()

    @contextmanager
    def fake_gpu_lock(self, label: str):
        """GPU 队列替身不调用 nvidia-smi 或文件锁。"""
        yield SimpleNamespace(as_dict=lambda: {"label": label, "worker_seconds": 0.1})

    def test_route_passes_syllable_and_glossary_constraints_to_homura(self) -> None:
        self.main.manager.check_runtime_ready = Mock()
        self.main.transcribe_audio = Mock(
            return_value={"text": "我们今天去看电影吧", "language": "Chinese"}
        )
        self.main.manager.run_worker = Mock(return_value="Let's go to the cinema today.")
        self.main.synthesize_audio = Mock(return_value=b"RIFF\x00\x00\x00\x00WAVE")

        with (
            patch.object(self.main, "resolve_executable", return_value="/usr/bin/true"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main, "wait_after_cuda_release"),
        ):
            response = self.client.post(
                "/v1/indexTranslate/homura",
                json={
                    "audio_path": self.audio_path,
                    "target_language": "en",
                    "target_syllables": 8,
                    "glossary": {"电影": "cinema"},
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-target-language"], "en")
        self.assertEqual(response.headers["x-target-syllables"], "8")
        self.assertEqual(response.content, b"RIFF\x00\x00\x00\x00WAVE")
        self.main.transcribe_audio.assert_called_once_with(
            self.main.reference_store.prompt_audio_path(self.audio_path), self.audio_path, None
        )
        payload = self.main.manager.run_worker.call_args.args[0]
        self.assertEqual(payload["target_language_name"], "英语")
        self.assertEqual(payload["target_syllables"], 8)
        self.assertEqual(payload["glossary"], {"电影": "cinema"})
        self.assertEqual(payload["temperature"], 0.3)
        self.assertGreaterEqual(payload["max_new_tokens"], 512)
        self.main.synthesize_audio.assert_called_once_with(
            "Let's go to the cinema today.", "en", self.audio_path
        )

    def test_rejects_missing_audio_unsupported_language_and_missing_syllables(self) -> None:
        self.main.transcribe_audio = Mock()
        missing_audio = self.client.post(
            "/v1/indexTranslate/homura",
            json={"audio_path": "missing.wav", "target_syllables": 8},
        )
        self.assertEqual(missing_audio.status_code, 404)

        missing_syllables = self.client.post(
            "/v1/indexTranslate/homura", json={"audio_path": self.audio_path}
        )
        self.assertEqual(missing_syllables.status_code, 422)

        unsupported_language = self.client.post(
            "/v1/indexTranslate/homura",
            json={
                "audio_path": self.audio_path,
                "target_language": "ar",
                "target_syllables": 8,
            },
        )
        self.assertEqual(unsupported_language.status_code, 422)

        invalid_glossary = self.client.post(
            "/v1/indexTranslate/homura",
            json={
                "audio_path": self.audio_path,
                "target_language": "en",
                "target_syllables": 8,
                "glossary": {" ": "cinema"},
            },
        )
        self.assertEqual(invalid_glossary.status_code, 422)
        self.main.transcribe_audio.assert_not_called()

    def test_asr_language_codes_are_expanded_to_qwen_language_names(self) -> None:
        response = Mock(is_success=True)
        response.json.return_value = {"text": "hello"}
        client = MagicMock()
        client.__enter__.return_value.post.return_value = response

        with patch.object(self.main.httpx, "Client", return_value=client):
            result = self.main.transcribe_audio(
                self.main.reference_store.prompt_audio_path(self.audio_path),
                self.audio_path,
                " zh ",
            )

        self.assertEqual(result["text"], "hello")
        request = client.__enter__.return_value.post.call_args
        self.assertEqual(request.kwargs["data"], {"language": "Chinese"})

    def test_homura_prompt_matches_syllable_control_and_glossary_contract(self) -> None:
        worker = load_module("index_homura_worker_for_tests", SERVICE_DIR / "worker.py")
        prompt = worker.build_translation_prompt(
            "我们今天去看电影吧", "英语", 8, {"电影": "cinema"}
        )
        self.assertIn("翻译为英语", prompt)
        self.assertIn("严格控制在 8 个音节", prompt)
        self.assertIn("术语映射表", prompt)
        self.assertIn("电影 -> cinema", prompt)
        self.assertTrue(prompt.endswith("我们今天去看电影吧"))

    def test_worker_starts_server_with_thinking_disabled_and_homura_sampling(self) -> None:
        worker = load_module("index_homura_worker_request_for_tests", SERVICE_DIR / "worker.py")
        model_path = self.root / "model/worker.gguf"
        model_path.touch()
        llama_server = self.root / "llama-server"
        llama_server.write_text("mock executable", encoding="utf-8")
        process = MagicMock()
        process.poll.return_value = None
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Let’s go today."}}]}
        ).encode()
        request = {
            "model_path": str(model_path),
            "llama_server_bin": str(llama_server),
            "text": "我们今天去看电影吧",
            "target_language": "en",
            "target_language_name": "英语",
            "target_syllables": 8,
            "glossary": {"电影": "cinema"},
            "max_new_tokens": 512,
            "temperature": 0.3,
        }

        with (
            patch.object(worker, "find_free_port", return_value=19321),
            patch.object(worker, "wait_for_server"),
            patch.object(worker.subprocess, "Popen", return_value=process) as start_server,
            patch.object(
                worker.urllib.request, "urlopen", return_value=response
            ) as post_completion,
        ):
            translated = worker.translate(request)

        self.assertEqual(translated, "Let’s go today.")
        command = start_server.call_args.args[0]
        self.assertIn("--jinja", command)
        self.assertIn("--chat-template-kwargs", command)
        self.assertEqual(
            command[command.index("--chat-template-kwargs") + 1], '{"enable_thinking":false}'
        )
        self.assertEqual(command[command.index("--alias") + 1], "index-homura-9b")
        completion_request = post_completion.call_args.args[0]
        completion = json.loads(completion_request.data)
        self.assertEqual(completion["temperature"], 0.3)
        self.assertEqual(len(completion["messages"]), 1)
        self.assertIn("严格控制在 8 个音节", completion["messages"][0]["content"])
        self.assertIn("电影 -> cinema", completion["messages"][0]["content"])
        process.terminate.assert_called_once()

    def test_startup_and_quality_gate_register_homura(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        self.assertIn('INDEX_HOMURA_PORT="${INDEX_HOMURA_PORT:-8383}"', startup)
        self.assertIn(
            'INDEX_HOMURA_PROJECT_DIR="${INDEX_HOMURA_PROJECT_DIR:-$PROJECT_DIR/Index-Homura-9B-GGUF}"',
            startup,
        )
        self.assertIn("/v1/indexTranslate/homura", startup)
        self.assertIn('setsid uv run --no-sync --project "$INDEX_HOMURA_PROJECT_DIR"', startup)
        cleanup = startup.split("cleanup() {", maxsplit=1)[1].split("trap cleanup", maxsplit=1)[0]
        wait_line = next(line for line in startup.splitlines() if line.startswith("wait -n "))
        self.assertIn('"$index_homura_pid"', cleanup)
        self.assertIn('"$index_homura_pid"', wait_line)
        self.assertIn("Index-Homura-9B-GGUF", quality_gate)


if __name__ == "__main__":
    unittest.main()
