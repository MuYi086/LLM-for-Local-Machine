"""MOSS-SoundEffect 服务与 worker 生命周期的无模型回归。"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm_local_runtime import GpuLockTimeoutError

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "soundEffect-v2.0"
ROUTE = "/v1/moss/soundEffect-v2.0"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="moss-soundeffect-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
STORAGE_DIR = TEST_ROOT / "storage"
WORKER_TMP_DIR = TEST_ROOT / "worker"

os.environ.update(
    {
        "MOSS_SOUNDEFFECT_V2_MODEL_DIR": str(MODEL_DIR),
        "MOSS_SOUNDEFFECT_V2_OUTPUT_DIR": str(STORAGE_DIR / "sound_effect_v2"),
        "MOSS_SOUNDEFFECT_V2_WORKER_TMP_DIR": str(WORKER_TMP_DIR),
        "MOSS_SOUNDEFFECT_V2_PORT": "8362",
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "GPU_LOCK_WAIT_TIMEOUT": "5",
        "CUDA_RELEASE_DELAY": "0",
    }
)
sys.path.insert(0, str(REPOSITORY_DIR / "llm_local_runtime" / "src"))
sys.path.insert(0, str(SERVICE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "moss_soundeffect_service_main_for_tests", SERVICE_DIR / "main.py"
)
assert SPEC and SPEC.loader
main = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = main
SPEC.loader.exec_module(main)


def make_wav() -> bytes:
    """生成 48 kHz 单声道 PCM16 测试 WAV。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0\0" * 480)
    return buffer.getvalue()


class MossSoundEffectServiceTests(unittest.TestCase):
    """验证请求契约、模型预检及一次性 worker 接口。"""

    def setUp(self) -> None:
        WORKER_TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        for path in WORKER_TMP_DIR.iterdir():
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            else:
                path.unlink()

    def make_model_ready(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with patch.object(main, "MODEL_FILE_SIZES", {name: 1 for name in main.MODEL_FILE_SIZES}):
            for relative_path in main.MODEL_FILE_SIZES:
                path = MODEL_DIR / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
        for relative_path in main.MODEL_CONFIG_FILES:
            path = MODEL_DIR / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}")

    def test_request_accepts_native_defaults_and_rejects_invalid_values(self) -> None:
        request = main.MossSoundEffectRequest(prompt="Rain falling on a roof")
        self.assertEqual(request.seconds, 10.0)
        self.assertEqual(request.num_inference_steps, 100)
        self.assertEqual(request.cfg_scale, 4.0)
        self.assertEqual(request.sigma_shift, 5.0)
        self.assertEqual(request.seed, 0)

        for payload in (
            {"prompt": "  "},
            {"prompt": "rain", "seconds": 0.05},
            {"prompt": "rain", "seconds": 30.1},
            {"prompt": "rain", "seconds": float("nan")},
            {"prompt": "rain", "num_inference_steps": 0},
            {"prompt": "rain", "cfg_scale": float("inf")},
            {"prompt": "rain", "unexpected": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                main.MossSoundEffectRequest(**payload)

    def test_model_status_requires_all_full_local_artifacts(self) -> None:
        self.assertFalse(main.model_status()["complete"])
        self.make_model_ready()
        with patch.object(main, "MODEL_FILE_SIZES", {name: 1 for name in main.MODEL_FILE_SIZES}):
            self.assertTrue(main.model_status()["complete"])
        (MODEL_DIR / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
        self.assertFalse(main.model_status()["complete"])

    def test_worker_output_rejects_wav_with_truncated_frame_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "帧数据不完整"):
            main.validate_generated_wav(make_wav()[:-20])

    def test_health_reports_readiness_without_starting_a_worker(self) -> None:
        self.make_model_ready()
        with (
            patch.object(main, "module_available", return_value=True),
            patch.object(main, "cuda_status", return_value={"available": True}),
            patch.object(
                main.subprocess, "Popen", side_effect=AssertionError("健康检查不能启动 worker")
            ),
        ):
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["runtime"]["ready"])
        self.assertFalse(response.json()["available"]["source"]["complete"])

    def test_generate_returns_wav_and_atomically_persists_output(self) -> None:
        self.make_model_ready()
        expected_wav = make_wav()
        with (
            patch.object(main, "ensure_runtime_ready"),
            patch.object(main, "gpu_runtime_lock", return_value=nullcontext()) as lock,
            patch.object(main.manager, "run_worker", return_value=expected_wav) as run_worker,
        ):
            response = self.client.post(
                ROUTE,
                json={
                    "prompt": "Rain falling on a roof",
                    "seconds": 4.5,
                    "seed": 42,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, expected_wav)
        saved_outputs = list((STORAGE_DIR / "sound_effect_v2").glob("*.wav"))
        self.assertEqual(len(saved_outputs), 1)
        self.assertEqual(saved_outputs[0].read_bytes(), expected_wav)
        self.assertEqual(run_worker.call_args.args[0]["seed"], 42)
        lock.assert_called_once_with("moss_soundeffect_v2/generate")

    def test_gpu_lock_uses_configured_positive_wait_timeout(self) -> None:
        with patch.object(main, "shared_gpu_runtime_lock", return_value=nullcontext()) as lock:
            with main.gpu_runtime_lock("test/generate"):
                pass
        lock.assert_called_once_with(
            main.GPU_LOCK_FILE,
            "test/generate",
            timeout_seconds=5,
        )

    def test_generate_maps_queue_and_worker_errors_and_cleans_temporary_files(self) -> None:
        error_cases = (
            (GpuLockTimeoutError("moss_soundeffect_v2/generate", 1), 503),
            (FileNotFoundError("model missing"), 503),
            (TimeoutError("worker timeout"), 504),
            (RuntimeError("worker failed"), 500),
        )
        for error, expected_status in error_cases:
            with (
                self.subTest(status=expected_status),
                patch.object(main, "ensure_runtime_ready"),
                patch.object(main, "gpu_runtime_lock", side_effect=error),
            ):
                response = self.client.post(ROUTE, json={"prompt": "Rain falling"})
                self.assertEqual(response.status_code, expected_status)
        self.assertEqual(list(WORKER_TMP_DIR.iterdir()), [])

    def test_worker_uses_service_python_and_removes_request_files(self) -> None:
        expected_wav = make_wav()
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def communicate(self, timeout=None):
                command = captured["command"]
                request_path = Path(command[command.index("--input-json") + 1])
                output_path = Path(command[command.index("--output-wav") + 1])
                captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
                output_path.write_bytes(expected_wav)
                return "worker complete", ""

            def poll(self):
                return self.returncode

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            output = main.manager.run_worker(
                {
                    "prompt": "Rain falling",
                    "seconds": 3,
                    "model_dir": str(MODEL_DIR),
                    "code_path": str(main.CODE_PATH),
                    "device": "cuda",
                }
            )

        self.assertEqual(output, expected_wav)
        self.assertEqual(captured["command"][0], sys.executable)
        self.assertTrue(captured["kwargs"]["start_new_session"])
        self.assertEqual(captured["payload"]["model_dir"], str(MODEL_DIR))
        self.assertEqual(captured["payload"]["code_path"], str(main.CODE_PATH))
        self.assertEqual(list(WORKER_TMP_DIR.iterdir()), [])

    def test_worker_timeout_terminates_and_cleans_temporary_files(self) -> None:
        class TimeoutProcess:
            returncode = None
            pid = None

            def communicate(self, timeout=None):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("worker", timeout)
                return "", ""

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                self.returncode = -15

        with (
            patch.object(main.subprocess, "Popen", return_value=TimeoutProcess()),
            patch.object(main, "REQUEST_TIMEOUT", 0.01),
        ):
            with self.assertRaisesRegex(TimeoutError, "超时"):
                main.manager.run_worker(
                    {
                        "prompt": "Rain falling",
                        "seconds": 3,
                        "model_dir": str(MODEL_DIR),
                        "code_path": str(main.CODE_PATH),
                        "device": "cuda",
                    }
                )
        self.assertEqual(list(WORKER_TMP_DIR.iterdir()), [])

    def test_worker_timeout_does_not_wait_forever_for_inherited_pipes(self) -> None:
        class OpenPipe:
            closed = False

            def close(self):
                self.closed = True

        class DescendantKeepsPipesOpen:
            pid = None
            returncode = None

            def __init__(self):
                self.stdout = OpenPipe()
                self.stderr = OpenPipe()
                self.communicate_calls = 0

            def communicate(self, timeout=None):
                self.communicate_calls += 1
                raise subprocess.TimeoutExpired("worker", timeout)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        process = DescendantKeepsPipesOpen()
        with (
            patch.object(main.subprocess, "Popen", return_value=process),
            patch.object(main, "REQUEST_TIMEOUT", 0.01),
        ):
            with self.assertRaisesRegex(TimeoutError, "超时"):
                main.manager.run_worker(
                    {
                        "prompt": "Rain falling",
                        "seconds": 3,
                        "model_dir": str(MODEL_DIR),
                        "code_path": str(main.CODE_PATH),
                        "device": "cuda",
                    }
                )

        self.assertEqual(process.communicate_calls, 2)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertEqual(list(WORKER_TMP_DIR.iterdir()), [])

    def test_process_group_cleanup_signals_descendants_after_worker_exit(self) -> None:
        marker = TEST_ROOT / "orphan_terminated.txt"
        ready = TEST_ROOT / "orphan_ready.txt"
        child_code = (
            "import pathlib, signal, sys, time; "
            f"marker = pathlib.Path({str(marker)!r}); "
            f"ready = pathlib.Path({str(ready)!r}); "
            "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('term'), sys.exit(0))); "
            "ready.write_text('ready'); "
            "time.sleep(30)"
        )
        parent_code = "\n".join(
            (
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', "
                f"{child_code!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL)",
                f"ready = pathlib.Path({str(ready)!r})",
                "deadline = time.monotonic() + 5",
                "while not ready.is_file() and time.monotonic() < deadline:",
                "    time.sleep(0.01)",
                "print(child.pid, flush=True)",
            )
        )
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=5)
            child_pid = int(stdout.strip())
            self.assertIsNotNone(process.poll())

            main.terminate_process_group(
                process,
                "MOSS-SoundEffect test",
                terminate_timeout=0.5,
                kill_timeout=0.5,
            )

            self.assertTrue(marker.is_file(), f"worker 子进程 {child_pid} 没有收到 SIGTERM")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_process_group_cleanup_waits_after_sigkill_for_descendants(self) -> None:
        class ExitedWorker:
            pid = 12_345

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        runtime_module = sys.modules["soundeffect_runtime"]
        process = ExitedWorker()
        with (
            patch.object(
                runtime_module, "_process_group_exists", side_effect=[True, True, True, False]
            ) as group_exists,
            patch.object(runtime_module.os, "killpg") as killpg,
        ):
            main.terminate_process_group(
                process,
                "MOSS-SoundEffect test",
                terminate_timeout=0,
                kill_timeout=0.5,
            )

        self.assertEqual(group_exists.call_count, 4)
        killpg.assert_any_call(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
