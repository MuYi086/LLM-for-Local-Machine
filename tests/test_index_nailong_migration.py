"""Index-Nailong 语音翻译 HTTP 契约和启动接线的无模型回归。"""

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
SERVICE_DIR = REPOSITORY_DIR / "Index-Nailong-9B-GGUF"


def load_module(name: str, path: Path):
    """以唯一模块名加载服务，避免多个 main.py 互相覆盖。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IndexNailongMigrationTests(unittest.TestCase):
    """隔离模型、网络和文件系统边界，验证服务契约。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="index-nailong-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "PROMPTS_DIR": str(self.root / "storage/clone"),
            "TIMBRE_STORAGE_DIR": str(self.root / "storage/timbre"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "cache/gpu.lock"),
            "INDEX_NAILONG_MODEL_DIR": str(self.root / "model"),
            "INDEX_NAILONG_MODEL_PATH": str(self.root / "model/nailong.gguf"),
            "INDEX_NAILONG_WORKER_TMP_DIR": str(self.root / "cache/worker"),
            "INDEX_NAILONG_LLAMA_SERVER_BIN": "llama-server-not-installed",
            "INDEX_NAILONG_REQUEST_TIMEOUT": "5",
            "INDEX_NAILONG_STARTUP_TIMEOUT": "5",
            "INDEX_NAILONG_PORT": "8384",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
            "CUDA_RELEASE_DELAY": "0",
        }
        preserved = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("INDEX_NAILONG_")
            or key in {"STORAGE_DIR", "PROMPTS_DIR", "CLONE_STORAGE_DIR", "TIMBRE_STORAGE_DIR"}
        }
        with (
            patch.dict(os.environ, {**preserved, **environment}, clear=True),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("index_nailong_main_for_tests", SERVICE_DIR / "main.py")
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

    def test_route_transcribes_translates_and_preserves_reference_voice(self) -> None:
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
                "/v1/indexTranslate/naiLong",
                json={
                    "audio_path": self.audio_path,
                    "glossary": {"电影": "cinema"},
                    "hard_constraints": ["保留人名"],
                    "soft_constraints": ["译文使用自然口语"],
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-source-language"], "Chinese")
        self.assertEqual(response.headers["x-target-language"], "en")
        self.assertEqual(response.content, b"RIFF\x00\x00\x00\x00WAVE")
        self.main.transcribe_audio.assert_called_once_with(
            self.main.reference_store.prompt_audio_path(self.audio_path), self.audio_path, None
        )
        payload = self.main.manager.run_worker.call_args.args[0]
        self.assertEqual(payload["target_language_name"], "英语")
        self.assertEqual(payload["source_language_name"], "中文")
        self.assertEqual(payload["glossary"], {"电影": "cinema"})
        self.assertEqual(payload["hard_constraints"], ["保留人名"])
        self.assertEqual(payload["soft_constraints"], ["译文使用自然口语"])
        self.assertEqual(payload["temperature"], 0)
        self.assertGreaterEqual(payload["max_new_tokens"], 512)
        self.main.synthesize_audio.assert_called_once_with(
            "Let's go to the cinema today.", "en", self.audio_path
        )

    def test_rejects_invalid_audio_language_and_constraint_before_asr(self) -> None:
        self.main.transcribe_audio = Mock()
        missing_audio = self.client.post(
            "/v1/indexTranslate/naiLong",
            json={"audio_path": "missing.wav", "target_language": "en"},
        )
        self.assertEqual(missing_audio.status_code, 404)

        unsupported_language = self.client.post(
            "/v1/indexTranslate/naiLong",
            json={"audio_path": self.audio_path, "target_language": "ar"},
        )
        self.assertEqual(unsupported_language.status_code, 422)

        invalid_constraint = self.client.post(
            "/v1/indexTranslate/naiLong",
            json={"audio_path": self.audio_path, "hard_constraints": [" "]},
        )
        self.assertEqual(invalid_constraint.status_code, 422)
        self.main.transcribe_audio.assert_not_called()

    def test_source_language_codes_become_qwen_language_hints(self) -> None:
        response = Mock(is_success=True)
        response.json.return_value = {"text": "hello", "language": "English"}
        client = MagicMock()
        client.__enter__.return_value.post.return_value = response
        with patch.object(self.main.httpx, "Client", return_value=client):
            result = self.main.transcribe_audio(
                self.main.reference_store.prompt_audio_path(self.audio_path),
                self.audio_path,
                "zh",
            )

        self.assertEqual(result["text"], "hello")
        request = client.__enter__.return_value.post.call_args
        self.assertEqual(request.kwargs["data"], {"language": "Chinese"})

    def test_health_reports_model_server_and_audio_dependencies(self) -> None:
        self.main.remote_service_available = Mock(return_value=True)
        with patch.object(self.main, "resolve_executable", return_value="/usr/bin/llama-server"):
            result = self.main.health()

        self.assertTrue(result["available"]["model"])
        self.assertTrue(result["available"]["llama_server"])
        self.assertTrue(result["available"]["qwen3_asr"])
        self.assertTrue(result["available"]["confucius4_tts"])
        self.assertIn("Nailong", result["runtime"]["backend"])
        self.assertEqual(
            result["runtime"]["target_languages"], self.main.SUPPORTED_TARGET_LANGUAGES
        )

    def test_worker_prompts_match_nailong_plain_and_insttrans_formats(self) -> None:
        worker = load_module("index_nailong_worker_prompt_for_tests", SERVICE_DIR / "worker.py")
        plain_prompt = worker.build_translation_prompt(
            "你好，世界。", "英语", "", {}, [], [], "口语"
        )
        self.assertEqual(
            plain_prompt,
            "请将以下文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n你好，世界。",
        )

        constrained_prompt = worker.build_translation_prompt(
            "我们今天去看电影吧",
            "英语",
            "中文",
            {"电影": "cinema"},
            ["保留人名"],
            ["译文使用自然口语"],
            "口语",
        )
        self.assertIn("中文口语翻译成英语", constrained_prompt)
        self.assertIn("〖源文〗\n我们今天去看电影吧", constrained_prompt)
        self.assertIn("〖硬性要求〗专名/术语对照: 电影→cinema", constrained_prompt)
        self.assertIn("〖硬性要求〗保留人名", constrained_prompt)
        self.assertIn("〖注意〗译文使用自然口语", constrained_prompt)
        self.assertTrue(constrained_prompt.endswith("只输出译文，不要有任何额外说明。"))

    def test_worker_uses_greedy_nailong_settings_and_cleans_server(self) -> None:
        worker = load_module("index_nailong_worker_request_for_tests", SERVICE_DIR / "worker.py")
        model_path = self.root / "model/worker.gguf"
        model_path.touch()
        llama_server = self.root / "llama-server"
        llama_server.write_text("mock executable", encoding="utf-8")
        process = MagicMock()
        process.poll.return_value = None
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Let's go today."}}]}
        ).encode()
        request = {
            "model_path": str(model_path),
            "llama_server_bin": str(llama_server),
            "text": "我们今天去看电影吧",
            "target_language": "en",
            "target_language_name": "英语",
            "source_language_name": "中文",
            "glossary": {"电影": "cinema"},
            "hard_constraints": [],
            "soft_constraints": [],
            "genre": "口语",
            "max_new_tokens": 512,
            "temperature": 0,
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

        self.assertEqual(translated, "Let's go today.")
        command = start_server.call_args.args[0]
        self.assertIn("--jinja", command)
        self.assertIn("--chat-template-kwargs", command)
        self.assertEqual(command[command.index("--alias") + 1], "index-nailong-9b")
        completion = json.loads(post_completion.call_args.args[0].data)
        self.assertEqual(completion["temperature"], 0)
        self.assertEqual(completion["top_p"], 1)
        self.assertEqual(completion["seed"], 42)
        self.assertIn("电影→cinema", completion["messages"][0]["content"])
        process.terminate.assert_called_once()

    def test_worker_manager_cleans_json_files_and_process_group(self) -> None:
        process = MagicMock()
        process.returncode = 0
        popen = Mock(return_value=process)

        def finish_worker(*, timeout: float):
            command = popen.call_args.args[0]
            output_path = Path(command[command.index("--output-json") + 1])
            output_path.write_text(json.dumps({"translated_text": "Hello."}), encoding="utf-8")
            return "", ""

        process.communicate.side_effect = finish_worker
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate_group,
        ):
            result = self.main.manager.run_worker({"text": "你好"})

        self.assertEqual(result, "Hello.")
        self.assertEqual(process.communicate.call_args.kwargs["timeout"], self.main.WORKER_TIMEOUT)
        terminate_group.assert_called_once_with(process, "Index-Nailong")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_startup_and_quality_gate_register_nailong(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        self.assertIn('INDEX_NAILONG_PORT="${INDEX_NAILONG_PORT:-8384}"', startup)
        self.assertIn(
            'INDEX_NAILONG_PROJECT_DIR="${INDEX_NAILONG_PROJECT_DIR:-$PROJECT_DIR/Index-Nailong-9B-GGUF}"',
            startup,
        )
        self.assertIn("/v1/indexTranslate/naiLong", startup)
        self.assertIn('setsid uv run --no-sync --project "$INDEX_NAILONG_PROJECT_DIR"', startup)
        cleanup = startup.split("cleanup() {", maxsplit=1)[1].split("trap cleanup", maxsplit=1)[0]
        wait_line = next(line for line in startup.splitlines() if line.startswith("wait -n "))
        self.assertIn('"$index_nailong_pid"', cleanup)
        self.assertIn('"$index_nailong_pid"', wait_line)
        self.assertIn("Index-Nailong-9B-GGUF", quality_gate)


if __name__ == "__main__":
    unittest.main()
