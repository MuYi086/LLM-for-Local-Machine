"""MOSS-Music HTTP 合同与一次性 GGUF worker 的无模型回归。"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from llm_local_runtime import GpuLockTimeoutError

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "model-audio/analyse_moss-music-8b-Instruct"
ROUTE = "/v1/openMoss/moss-music"


def load_module(name: str, filename: str):
    """加载独立服务入口，不解析其模型环境。"""
    spec = importlib.util.spec_from_file_location(name, SERVICE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, seconds: float = 1) -> None:
    """为 CPU 音频预检生成小型 PCM16 测试输入。"""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * int(seconds * 16000))


class MossMusicServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="moss-music-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch.dict(
            os.environ,
            {
                "STORAGE_DIR": str(self.root),
                "RUNTIME_CACHE_DIR": str(self.root / "cache"),
                "MOSS_MUSIC_WORKER_TMP_DIR": str(self.root / "worker"),
                "MOSS_MUSIC_PORT": "8310",
            },
        ):
            self.main = load_module("moss_main_test", "main.py")
        self.client = TestClient(self.main.app)
        self.ready = patch.object(self.main, "ensure_runtime_ready")
        self.ready.start()
        self.addCleanup(self.ready.stop)

    def post(self, **data):
        return self.client.post(
            ROUTE, files={"audio": ("song.wav", b"wave-data", "audio/wav")}, data=data
        )

    def assert_clean(self) -> None:
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_health_does_not_start_worker_or_load_model(self) -> None:
        with (
            patch.object(self.main.subprocess, "Popen", side_effect=AssertionError("不能运行模型")),
            patch.object(self.main, "backend_available", return_value=True),
        ):
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["runtime"]["sampling_rate"], 16000)
        self.assertEqual(self.main.API_PORT, 8310)

    def test_analysis_passes_validated_prompt_and_returns_json_then_cleans_uploads(self) -> None:
        def execute(payload, directory):
            self.assertTrue(Path(payload["audio_path"]).is_relative_to(directory))
            self.assertEqual(payload["model_dir"], str(self.main.MODEL_DIR))
            self.assertEqual(payload["prompt"], "有哪些乐器？")
            self.assertEqual(payload["max_new_tokens"], 512)
            self.assertEqual(payload["temperature"], 0.5)
            return "以钢琴和弦乐为主。"

        with (
            patch.object(self.main, "normalize_audio", return_value=12.5),
            patch.object(self.main, "execute_analysis", side_effect=execute),
        ):
            response = self.post(prompt="有哪些乐器？", max_new_tokens="512", temperature="0.5")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["text"], "以钢琴和弦乐为主。")
        self.assertEqual(response.json()["duration_seconds"], 12.5)
        self.assertEqual(len(response.json()["audio_sha256"]), 64)
        self.assert_clean()

    def test_invalid_generation_fields_are_rejected_before_gpu(self) -> None:
        with patch.object(
            self.main, "execute_analysis", side_effect=AssertionError("不能进入 GPU")
        ):
            for data in (
                {"prompt": " "},
                {"max_new_tokens": "0"},
                {"max_new_tokens": "4097"},
                {"temperature": "nan"},
                {"temperature": "3"},
            ):
                with self.subTest(data=data):
                    self.assertEqual(self.post(**data).status_code, 422)
        self.assert_clean()

    def test_upload_rejects_invalid_type_empty_content_and_size_limit(self) -> None:
        with patch.object(self.main, "MAX_UPLOAD_BYTES", 3):
            self.assertEqual(self.post().status_code, 413)
        for filename, content, media_type in (
            ("song.exe", b"data", "audio/wav"),
            ("song.wav", b"", "audio/wav"),
            ("song.wav", b"data", "text/html"),
        ):
            with self.subTest(filename=filename, content=content):
                response = self.client.post(ROUTE, files={"audio": (filename, content, media_type)})
                self.assertEqual(response.status_code, 422)
        self.assert_clean()

    def test_request_failures_preserve_status_and_remove_temporary_files(self) -> None:
        for error, code in (
            (FileNotFoundError("missing"), 503),
            (ValueError("bad audio"), 422),
            (TimeoutError("worker timed out"), 504),
            (GpuLockTimeoutError("moss-music", 1), 503),
            (RuntimeError("worker failed"), 500),
        ):
            with (
                self.subTest(code=code),
                patch.object(self.main, "normalize_audio", return_value=1),
                patch.object(self.main, "execute_analysis", side_effect=error),
            ):
                self.assertEqual(self.post().status_code, code)
                self.assert_clean()

    def test_missing_runtime_fails_before_upload_and_gpu(self) -> None:
        with (
            patch.object(
                self.main, "ensure_runtime_ready", side_effect=FileNotFoundError("missing model")
            ),
            patch.object(self.main, "stage_audio_upload", side_effect=AssertionError("不能暂存")),
        ):
            self.assertEqual(self.post().status_code, 503)
        self.assert_clean()

    def test_normalize_rejects_long_audio_before_decode_and_gpu(self) -> None:
        with patch.object(self.main.subprocess, "run", return_value=MagicMock(stdout="421")) as run:
            with self.assertRaises(ValueError):
                self.main.normalize_audio(self.root / "in.wav", self.root / "out.wav")
        self.assertEqual(run.call_count, 1)

    def test_normalize_uses_16khz_mono_and_verifies_decoded_duration(self) -> None:
        destination = self.root / "out.wav"
        write_wav(destination)
        with patch.object(self.main.subprocess, "run", return_value=MagicMock(stdout="1.0")) as run:
            self.assertEqual(self.main.normalize_audio(self.root / "in.mp3", destination), 1)
        command = run.call_args_list[1].args[0]
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        with (
            patch.object(self.main, "MAX_AUDIO_SECONDS", 0.5),
            patch.object(self.main.subprocess, "run", return_value=MagicMock(stdout="0.1")),
        ):
            with self.assertRaisesRegex(ValueError, "解码后"):
                self.main.normalize_audio(self.root / "in.wav", destination)

    def test_model_status_requires_complete_both_weights_and_gguf_header(self) -> None:
        model = self.root / "model"
        model.mkdir()
        files = {"moss-lm-q8_0.gguf": 8, "moss-aud-f16.gguf": 8}
        with (
            patch.object(self.main, "MODEL_DIR", model),
            patch.object(self.main, "MODEL_FILES", files),
        ):
            self.assertFalse(self.main.model_status()["complete"])
            for name in files:
                (model / name).write_bytes(b"GGUF\x03\x00\x00\x00")
            self.assertTrue(self.main.model_status()["complete"])
            (model / "moss-aud-f16.gguf").write_bytes(b"broken")
            self.assertFalse(self.main.model_status()["complete"])

    def test_lock_covers_failure_cleanup_and_cuda_delay(self) -> None:
        events = []

        @contextmanager
        def lock(*args, **kwargs):
            self.assertGreater(kwargs["timeout_seconds"], 0)
            events.append("locked")
            try:
                yield
            finally:
                events.append("released")

        def fail(*args):
            events.append("worker")
            raise RuntimeError("worker failed")

        with (
            patch.object(self.main, "gpu_runtime_lock", side_effect=lock),
            patch.object(self.main, "CUDA_RELEASE_DELAY", 0.01),
            patch.object(self.main, "run_worker", side_effect=fail),
            patch.object(self.main.time, "sleep", side_effect=lambda _: events.append("delay")),
        ):
            with self.assertRaises(RuntimeError):
                self.main.execute_analysis({}, self.root)
        self.assertEqual(events, ["locked", "worker", "delay", "released"])

    def test_worker_manager_terminates_group_on_success_error_timeout_and_empty_output(
        self,
    ) -> None:
        for outcome in ("success", "error", "timeout", "empty"):
            with (
                self.subTest(outcome=outcome),
                tempfile.TemporaryDirectory(dir=self.root) as temporary,
            ):
                directory = Path(temporary)
                process = MagicMock(pid=12345, returncode=0 if outcome != "error" else 1)
                if outcome == "timeout":
                    process.communicate.side_effect = subprocess.TimeoutExpired("worker", 1)
                else:
                    process.communicate.return_value = ("", "failure")
                if outcome == "success":
                    (directory / "answer.txt").write_text("音乐分析", encoding="utf-8")
                with (
                    patch.object(self.main.subprocess, "Popen", return_value=process) as popen,
                    patch.object(self.main, "terminate_process_group") as terminate,
                ):
                    if outcome == "success":
                        self.assertEqual(self.main.run_worker({}, directory), "音乐分析")
                    else:
                        with self.assertRaises((RuntimeError, TimeoutError)):
                            self.main.run_worker({}, directory)
                    terminate.assert_called_once_with(process)
                    self.assertTrue(popen.call_args.kwargs["start_new_session"])
                    self.assertEqual(popen.call_args.args[0][0], sys.executable)

    def test_cleanup_kills_descendants_even_when_group_leader_has_exited(self) -> None:
        process = MagicMock(pid=12345)
        process.poll.return_value = 0
        with patch.object(self.main.os, "killpg") as killpg:
            self.main.terminate_process_group(process)
        self.assertEqual(killpg.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (12345, signal.SIGKILL))

    def test_worker_execs_fixed_cli_without_shell_and_writes_prompt_file(self) -> None:
        worker = load_module("moss_worker_test", "worker.py")
        request = self.root / "request.json"
        payload = {
            "caption_bin": "/local/ace-caption",
            "model_dir": "/local/model",
            "audio_path": "/tmp/audio.wav",
            "backend": "CUDA0",
            "prompt": "请分析 '引号' 和 $()",
            "max_new_tokens": 123,
            "max_audio_seconds": 420,
            "temperature": 0.2,
        }
        request.write_text(json.dumps(payload), encoding="utf-8")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "worker.py",
                    "--input-json",
                    str(request),
                    "--output-file",
                    str(self.root / "answer.txt"),
                ],
            ),
            patch.object(worker.os, "execv") as execute,
            patch.dict(os.environ),
        ):
            worker.main()
            self.assertEqual(os.environ["GGML_BACKEND"], "CUDA0")
        command = execute.call_args.args[1]
        self.assertEqual(command[0], "/local/ace-caption")
        self.assertNotIn("--src-list", command)
        self.assertEqual(command[command.index("--max-tokens") + 1], "123")
        self.assertEqual(request.with_suffix(".prompt.txt").read_text(), payload["prompt"])

    def test_startup_and_gate_include_service_and_cleanup(self) -> None:
        startup = (ROOT / "start.sh").read_text()
        self.assertIn('MOSS_MUSIC_PORT="${MOSS_MUSIC_PORT:-8310}"', startup)
        self.assertIn('\'MOSS-Music\' "$MOSS_MUSIC_HOST" "$MOSS_MUSIC_PORT"', startup)
        self.assertIn('setsid uv run --no-sync --project "$MOSS_MUSIC_PROJECT_DIR"', startup)
        self.assertIn('"$moss_music_pid"', startup[startup.rindex("wait -n") :])
        self.assertIn('    "$moss_music_pid"', startup)
        self.assertIn(
            "model-audio/analyse_moss-music-8b-Instruct",
            (ROOT / "scripts/quality_gate.sh").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
