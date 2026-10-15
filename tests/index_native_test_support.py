"""Index 原生接口无模型回归的隔离工具。"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    """隔离加载服务及其本地辅助模块，不导入模型。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(sys, "path", [str(path.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


def wav_bytes(sample_rate: int = 24_000) -> bytes:
    """生成可解析的短 PCM WAV，仅用于文件边界验证。"""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


class NativeServiceCase(unittest.TestCase):
    """准备临时配置和 API 进程，真实 worker 和 GPU 始终被替换。"""

    project = ""
    prefix = ""
    route = ""

    def setUp(self) -> None:
        """用独立目录加载服务，避免修改真实模型与运行环境。"""
        temporary = tempfile.TemporaryDirectory(prefix="index-native-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.service_dir = REPOSITORY_DIR / self.project
        self.model_path = self.root / "model/model.gguf"
        self.model_path.parent.mkdir()
        self.model_path.write_bytes(b"GGUF")
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "cache/gpu.lock"),
            f"{self.prefix}_MODEL_DIR": str(self.model_path.parent),
            f"{self.prefix}_MODEL_PATH": str(self.model_path),
            f"{self.prefix}_WORKER_TMP_DIR": str(self.root / "worker"),
            f"{self.prefix}_LLAMA_SERVER_BIN": "missing-llama-server",
            f"{self.prefix}_MMPROJ_PATH": str(self.root / "model/mmproj.gguf"),
            f"{self.prefix}_REQUEST_TIMEOUT": "5",
            f"{self.prefix}_STARTUP_TIMEOUT": "3",
            "CUDA_RELEASE_DELAY": "0",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.main = load_module(
                f"{self.prefix.lower()}_native_test_main", self.service_dir / "main.py"
            )
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)

    def post_text(self, payload: dict, output: str = "第一段译文\n\nSecond paragraph."):
        """只替换 worker 返回边界，保留实际校验和 payload 构造。"""
        process = Mock(return_value=output)
        with (
            patch.object(self.main, "resolve_executable", return_value="/usr/bin/true"),
            patch.object(self.main, "gpu_runtime_lock", return_value=nullcontext()) as lock,
            patch.object(self.main, "wait_after_cuda_release") as release,
            patch.object(self.main.manager, "run_worker", process),
        ):
            response = self.client.post(f"/v1/indexTranslate/{self.route}", json=payload)
        if response.status_code == 200:
            lock.assert_called_once()
            release.assert_called_once()
        return response, process

    def check_worker_request(self, payload: dict, response_text: str = "Translated text."):
        """用本地聊天响应替身检查 worker 的原生消息和采样参数。"""
        worker = load_module(f"{self.prefix.lower()}_native_worker", self.service_dir / "worker.py")
        executable = self.root / "llama-server"
        executable.touch()
        request = {
            "model_path": str(self.model_path),
            "llama_server_bin": str(executable),
            **payload,
        }
        process = MagicMock()
        process.poll.return_value = None
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": response_text}, "finish_reason": "stop"}]}
        ).encode()
        extra = (
            patch.object(worker, "load_stop_tokens", return_value=["<|endoftext|>", "<|im_end|>"])
            if hasattr(worker, "load_stop_tokens")
            else nullcontext()
        )
        with (
            extra,
            patch.object(worker, "find_free_port", return_value=19001),
            patch.object(worker, "wait_for_server"),
            patch.object(worker.subprocess, "Popen", return_value=process) as start,
            patch.object(worker.urllib.request, "urlopen", return_value=response) as call,
        ):
            result = worker.translate(request)
        self.assertEqual(
            result, response_text if self.route == "naiLong" else response_text.strip()
        )
        process.terminate.assert_called_once()
        self.assertIn("--parallel", start.call_args.args[0])
        self.assertIn("--no-context-shift", start.call_args.args[0])
        self.assertEqual(
            call.call_args.args[0].full_url, "http://127.0.0.1:19001/v1/chat/completions"
        )
        return json.loads(call.call_args.args[0].data), start.call_args.args[0]

    def check_manager_cleanup(self, failure: str | None = None) -> None:
        """成功、失败和超时均回收进程组与全部临时 JSON/WAV。"""
        process = MagicMock()
        process.returncode = 0 if failure != "exit" else 1
        popen = Mock(return_value=process)
        calls = 0

        def finish(*, timeout=None):
            nonlocal calls
            calls += 1
            if failure == "timeout" and calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            command = popen.call_args.args[0]
            request_path = Path(command[command.index("--input-json") + 1])
            output_path = Path(command[command.index("--output-json") + 1])
            if self.prefix == "INDEX_ECHO_S2ST":
                payload = json.loads(request_path.read_text())
                Path(payload["output_wav"]).write_bytes(wav_bytes())
                result = {"sample_rate": 24_000}
            else:
                result = {"translated_text": "Hello."}
            output_path.write_text(json.dumps(result), encoding="utf-8")
            return "", "failure" if failure else ""

        process.communicate.side_effect = finish
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            if failure:
                with self.assertRaises(TimeoutError if failure == "timeout" else RuntimeError):
                    self.main.manager.run_worker({"text": "你好"})
            else:
                self.main.manager.run_worker({"text": "你好"})
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertGreaterEqual(terminate.call_count, 1)
        self.assertFalse(list(self.main.WORKER_TMP_DIR.glob("*.json")))
        self.assertFalse(list(self.main.WORKER_TMP_DIR.glob("*.wav")))
