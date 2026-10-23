"""Qwen3-ASR-1.7B 独立服务的无模型回归测试。"""

from __future__ import annotations

# 测试替换 worker 子进程，不安装 qwen-asr、不加载模型权重，也不执行 CUDA 推理。
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "transcribe_qwen3-asr-1.7b"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="qwen3-asr-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "Qwen3-ASR-1.7B"

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "QWEN3_ASR_MODEL_DIR": str(MODEL_DIR),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "QWEN3_ASR_REQUEST_TIMEOUT": "5",
        "QWEN3_ASR_PORT": "8371",
        "LLM_LOCAL_INFERENCE_PROVIDER": "native",
    }
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for filename in (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
):
    (MODEL_DIR / filename).touch()
(MODEL_DIR / "model.safetensors.index.json").write_text(
    json.dumps(
        {
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        }
    ),
    encoding="utf-8",
)

sys.path.insert(0, str(REPOSITORY_DIR / "inference_gateway/src"))
sys.path.insert(0, str(REPOSITORY_DIR / "llm_local_runtime/src"))
sys.path.insert(0, str(SERVICE_DIR))
spec = importlib.util.spec_from_file_location(
    "qwen3_asr_service_main_for_tests",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class Qwen3AsrMigrationTests(unittest.TestCase):
    """验证 HTTP 契约、上传清理、模型预检和 worker 生命周期。"""

    def test_start_script_exposes_service_on_8371(self) -> None:
        """验证启动脚本配置模型路径、端口、路由和进程清理。"""
        source = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")

        self.assertIn(
            'QWEN3_ASR_MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B}"',
            source,
        )
        self.assertIn('QWEN3_ASR_PORT="${QWEN3_ASR_PORT:-8371}"', source)
        self.assertIn("inference_gateway/src", source)
        self.assertIn("'/v1/qwen3/asr'", source)
        self.assertIn("qwen3_asr_pid", source)
        self.assertIn('uv run --no-sync --project "$QWEN3_ASR_PROJECT_DIR"', source)

    def test_project_locks_official_qwen_asr_runtime(self) -> None:
        """验证项目通过单独锁文件固定官方 ASR Python 包。"""
        source = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        lock = (SERVICE_DIR / "uv.lock").read_text(encoding="utf-8")

        self.assertIn('"qwen-asr==0.0.6"', source)
        self.assertIn('name = "qwen-asr"', lock)
        self.assertIn('version = "0.0.6"', lock)
        self.assertIn('name = "llm-local-inference-gateway"', lock)

    def test_route_and_health_contract(self) -> None:
        """验证服务只注册最终识别路由，并可在不加载模型时报告状态。"""
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/qwen3/asr"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))
        self.assertNotIn(("POST", "/v1/qwen3/asr/transcribe"), actual_routes)

        with patch.object(main, "cuda_status", return_value={"available": False, "source": "test"}):
            response = TestClient(main.app).get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {"code", "paths", "available", "cuda", "runtime", "last_errors"},
        )
        self.assertTrue(payload["available"]["worker_script"])
        self.assertTrue(payload["available"]["qwen3_asr_model"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["worker_python"], sys.executable)

    def test_model_preflight_checks_indexed_weight_shards(self) -> None:
        """检查权重索引列出的文件缺失时健康和请求预检都不会误报就绪。"""
        shard = MODEL_DIR / "model-00002-of-00002.safetensors"
        shard.unlink()
        try:
            self.assertFalse(main.required_model_files_ready())
        finally:
            shard.touch()

    def test_request_defaults_and_worker_payload(self) -> None:
        """验证自动语言识别默认值和本地 worker 参数。"""
        audio_path = TEST_ROOT / "uploaded.wav"
        audio_path.write_bytes(b"RIFF" + b"\0" * 40)
        request = main.Qwen3AsrRequest()
        with patch.object(main, "module_available", return_value=True):
            payload = main.manager.build_worker_payload(request, audio_path)

        self.assertIsNone(request.language)
        self.assertEqual(request.context, "")
        self.assertEqual(request.max_new_tokens, main.DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(payload["model_dir"], str(MODEL_DIR))
        self.assertEqual(payload["audio_path"], str(audio_path))
        self.assertEqual(payload["device"], main.DEVICE)
        self.assertEqual(payload["dtype"], main.DTYPE)
        self.assertTrue(payload["local_files_only"])

    def test_transcription_route_returns_json_and_removes_upload(self) -> None:
        """验证音频后缀传入 worker，识别后临时上传文件会被清理。"""
        from fastapi.testclient import TestClient

        captured: dict[str, object] = {}

        def fake_execute(payload: dict[str, object]) -> dict[str, object]:
            captured.update(payload)
            return {
                "text": "  你好，世界。  ",
                "language": "Chinese",
                "elapsed_seconds": 1.25,
            }

        with (
            patch.object(main, "module_available", return_value=True),
            patch.object(main, "execute_transcription_payload", side_effect=fake_execute),
        ):
            response = TestClient(main.app).post(
                "/v1/qwen3/asr",
                files={"audio": ("speech.wav", b"RIFF" + b"\0" * 40, "audio/wav")},
                data={"language": "Chinese", "context": "人名：张三", "max_new_tokens": "512"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], 200)
        self.assertEqual(body["text"], "你好，世界。")
        self.assertEqual(body["language"], "Chinese")
        self.assertEqual(body["elapsed_seconds"], 1.25)
        self.assertEqual(body["audio"]["suffix"], ".wav")
        self.assertEqual(captured["language"], "Chinese")
        self.assertEqual(captured["context"], "人名：张三")
        self.assertEqual(captured["max_new_tokens"], 512)
        self.assertFalse(Path(str(captured["audio_path"])).exists())
        self.assertEqual(list(Path(main.UPLOAD_STAGING_DIR).iterdir()), [])

    def test_worker_uses_uv_interpreter_and_cleans_temporary_files(self) -> None:
        """验证 worker 使用服务解释器，正常退出后清理 JSON 临时文件。"""
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                output_path = Path(str(captured["command"][-1]))
                output_path.write_text(
                    json.dumps({"text": "recognized", "language": "English"}),
                    encoding="utf-8",
                )
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            result = main.manager.run_worker({"language": "English"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertEqual(result["text"], "recognized")
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
