"""YuE2-3B 歌曲生成服务的无模型回归测试。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "music_yue2-3b"
TEST_ROOT = Path()
COMPLETE_MODELS = {
    "inference_package": {"complete": True, "missing": []},
    "mot": {"complete": True, "missing": []},
    "vae": {"complete": True, "missing": []},
}


def load_service(name: str, path: Path):
    """用临时运行目录隔离加载轻量 API 模块。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    runtime_spec = importlib.util.spec_from_file_location(
        f"{name}_runtime", path.with_name("runtime.py")
    )
    assert runtime_spec and runtime_spec.loader
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    service_environment = {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "BGM_STORAGE_DIR": str(TEST_ROOT / "storage" / "bgm"),
        "YUE2_3B_OUTPUT_DIR": str(TEST_ROOT / "storage" / "bgm"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "YUE2_3B_MODEL_DIR": str(TEST_ROOT / "models" / "YuE2-3B"),
        "YUE2_VAE_MODEL_DIR": str(TEST_ROOT / "models" / "YuE2-Vae"),
        "YUE2_3B_WORKER_TMP_DIR": str(TEST_ROOT / "worker"),
        "LOCAL_FILES_ONLY": "1",
        "GPU_LOCK_WAIT_TIMEOUT": "1",
        "CUDA_RELEASE_DELAY": "0",
        "YUE2_3B_REQUEST_TIMEOUT": "5",
    }
    source_path = [str(path.parent), str(REPOSITORY_DIR / "llm_local_runtime/src"), *sys.path]
    with (
        patch.dict(os.environ, service_environment, clear=True),
        patch.object(sys, "path", source_path),
        patch.dict(sys.modules, {"runtime": runtime_module}),
    ):
        runtime_spec.loader.exec_module(runtime_module)
        spec.loader.exec_module(module)
    return module


class YuE2ThreeBMigrationTests(unittest.TestCase):
    """验证 API 原生参数、worker 生命周期和启动配置，不运行模型。"""

    def setUp(self) -> None:
        global TEST_ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix="yue2-3b-tests-")
        self.addCleanup(self.temporary.cleanup)
        TEST_ROOT = Path(self.temporary.name)
        self.main = load_service("yue2_3b_migration_test_main", SERVICE_DIR / "main.py")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)

    def test_native_request_validation_happens_before_gpu_queue(self) -> None:
        """拒绝空歌词、冲突别名、无效 ABC 组合和非原生字段。"""
        invalid_payloads = (
            {"lyrics": "verse"},
            {"style": "jazz", "tags": "rock", "lyrics": "verse"},
            {"style": "jazz", "lyrics": " ", "cot": "full"},
            {"style": "jazz", "lyrics": "verse", "abc": "C D E", "cot": "off"},
            {"style": "jazz", "lyrics": "verse", "audio_path": "other-service.wav"},
            {
                "style": "jazz",
                "lyrics": "verse",
                "semantic_sampling": {"min_tokens": 10, "max_tokens": 2},
            },
        )
        with patch.object(self.main, "gpu_runtime_lock") as gpu_lock:
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    response = self.client.post("/v1/m-a-p/YuE2", json=payload)
                    self.assertEqual(response.status_code, 422)
            gpu_lock.assert_not_called()

    def test_native_request_and_local_model_configuration_reach_worker_payload(self) -> None:
        request = self.main.YuE2MusicRequest(
            style="cinematic folk",
            lyrics="[Verse]\nA quiet morning",
            cot="melody",
            seed=123,
            abc="X:1\nK:C\nC D E",
            cfg_scale=1.2,
            id="quiet-morning",
            semantic_sampling={"temperature": 0.8, "top_k": 32},
        )

        payload = self.main.manager.build_worker_payload(request)

        self.assertEqual(payload["lyrics"], "[Verse]\nA quiet morning")
        self.assertEqual(payload["cot"], "melody")
        self.assertEqual(payload["abc"], "X:1\nK:C\nC D E")
        self.assertEqual(payload["semantic_sampling"], {"temperature": 0.8, "top_k": 32})
        self.assertEqual(payload["model_dir"], str(TEST_ROOT / "models/YuE2-3B"))
        self.assertEqual(payload["vae_model_dir"], str(TEST_ROOT / "models/YuE2-Vae"))
        self.assertEqual(
            payload["inference_wheel"],
            str(TEST_ROOT / "models/YuE2-3B/yue2_infer-0.1.5-py3-none-any.whl"),
        )
        self.assertTrue(payload["local_files_only"])

    def test_successful_route_returns_and_persists_native_wav(self) -> None:
        wav = b"RIFF\x00\x00\x00\x00WAVEtest"
        with (
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main, "model_status", return_value=COMPLETE_MODELS),
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "gpu_runtime_lock", return_value=nullcontext()) as gpu_lock,
            patch.object(self.main.manager, "run_worker", return_value=wav) as run_worker,
            patch.object(
                self.main, "persist_audio_bytes", return_value=TEST_ROOT / "song.wav"
            ) as persist,
            patch.object(self.main, "wait_after_cuda_release") as release,
        ):
            response = self.client.post(
                "/v1/m-a-p/YuE2",
                json={"style": "folk", "lyrics": "[Verse]\nhello", "seed": 777},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-yue2-3b-seed"], "777")
        self.assertEqual(response.headers["x-yue2-3b-sample-rate"], "48000")
        self.assertEqual(response.headers["x-yue2-3b-model"], "m-a-p/YuE2-3B")
        self.assertEqual(response.content, wav)
        self.assertEqual(run_worker.call_args.args[0]["lyrics"], "[Verse]\nhello")
        gpu_lock.assert_called_once_with(self.main.GPU_LOCK_FILE, "yue2_3b/music")
        persist.assert_called_once_with(wav, self.main.OUTPUT_DIR)
        release.assert_called_once()

    def test_persisted_wav_uses_atomic_yue2_prefix_in_bgm_storage(self) -> None:
        wav = b"RIFF\x00\x00WAVE"

        saved_path = self.main.persist_audio_bytes(wav, self.main.OUTPUT_DIR)

        self.assertEqual(saved_path.parent, self.main.OUTPUT_DIR)
        self.assertTrue(saved_path.name.startswith("yue2_3b_"))
        self.assertEqual(saved_path.read_bytes(), wav)
        self.assertEqual(list(self.main.OUTPUT_DIR.glob(".yue2_3b_*.tmp")), [])

    def test_preflight_failures_do_not_acquire_gpu(self) -> None:
        with (
            patch.object(self.main, "module_available", return_value=False),
            patch.object(self.main, "gpu_runtime_lock") as gpu_lock,
        ):
            dependency_response = self.client.post(
                "/v1/m-a-p/YuE2", json={"style": "folk", "lyrics": "hello"}
            )
        self.assertEqual(dependency_response.status_code, 503)
        gpu_lock.assert_not_called()

        incomplete_models = {
            **COMPLETE_MODELS,
            "vae": {"complete": False, "missing": ["config.json"]},
        }
        with (
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main, "model_status", return_value=incomplete_models),
            patch.object(self.main, "gpu_runtime_lock") as gpu_lock,
        ):
            model_response = self.client.post(
                "/v1/m-a-p/YuE2", json={"style": "folk", "lyrics": "hello"}
            )
        self.assertEqual(model_response.status_code, 503)
        self.assertIn("vae", model_response.json()["detail"]["models"])
        gpu_lock.assert_not_called()

    def test_worker_manager_cleans_request_and_output_files_on_success_and_timeout(self) -> None:
        process = Mock(returncode=0)
        popen = Mock(return_value=process)

        def write_worker_output(timeout: float) -> tuple[str, str]:
            command = popen.call_args.args[0]
            request_path = Path(command[command.index("--input-json") + 1])
            output_path = Path(command[command.index("--output-wav") + 1])
            self.assertEqual(
                json.loads(request_path.read_text(encoding="utf-8"))["lyrics"], "hello"
            )
            output_path.write_bytes(b"RIFF-native")
            return "", ""

        process.communicate.side_effect = write_worker_output
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            result = self.main.manager.run_worker({"lyrics": "hello", "local_files_only": True})

        self.assertEqual(result, b"RIFF-native")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        terminate.assert_called_once_with(process, "YuE2-3B")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

        timeout_process = Mock(returncode=None)
        timeout_process.communicate.side_effect = subprocess.TimeoutExpired("worker", 5)
        with (
            patch.object(self.main.subprocess, "Popen", return_value=timeout_process),
            patch.object(self.main, "terminate_process_group") as terminate_timeout,
        ):
            with self.assertRaises(TimeoutError):
                self.main.manager.run_worker({"lyrics": "hello"})

        terminate_timeout.assert_called_once_with(timeout_process, "YuE2-3B")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_official_pipeline_call_is_isolated_in_worker(self) -> None:
        worker_path = SERVICE_DIR / "worker.py"
        spec = importlib.util.spec_from_file_location("yue2_3b_worker_test", worker_path)
        assert spec and spec.loader
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)

        wheel_path = TEST_ROOT / "yue2_infer-0.1.5-py3-none-any.whl"
        wheel_path.touch()
        output_path = TEST_ROOT / "result.wav"
        song = Mock()
        song.save.side_effect = lambda path: Path(path).write_bytes(b"RIFF-song")
        pipeline = Mock(return_value=song)
        fake_yue2 = types.ModuleType("yue2")
        fake_yue2.YuE2Pipeline = Mock(from_pretrained=Mock(return_value=pipeline))
        request = {
            "inference_wheel": str(wheel_path),
            "model_dir": str(TEST_ROOT / "mot"),
            "vae_model_dir": str(TEST_ROOT / "vae"),
            "local_files_only": True,
            "device": "cuda",
            "memory_budget_gib": 24,
            "backend": "torch",
            "quantization": "none",
            "offload_ar": False,
            "style": "cinematic folk",
            "lyrics": "[Verse]\nhello",
            "cot": "full",
            "seed": 10,
            "abc": "X:1\nK:C\nC D E",
            "cfg_scale": 1.2,
            "id": "test-song",
            "abc_sampling": {"temperature": 0.7},
            "semantic_sampling": {"top_k": 32},
        }

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"yue2": fake_yue2}),
            patch.object(sys, "path", list(sys.path)),
        ):
            worker.generate_song(request, output_path)

        fake_yue2.YuE2Pipeline.from_pretrained.assert_called_once_with(
            model=str(TEST_ROOT / "mot"),
            vae=str(TEST_ROOT / "vae"),
            local_files_only=True,
            device="cuda",
            memory_budget_gib=24.0,
            backend="torch",
            quantization="none",
            offload_ar=False,
            progress=False,
        )
        pipeline.assert_called_once_with(
            style="cinematic folk",
            tags=None,
            lyrics="[Verse]\nhello",
            cot="full",
            seed=10,
            abc="X:1\nK:C\nC D E",
            cfg_scale=1.2,
            id="test-song",
            abc_sampling={"temperature": 0.7},
            semantic_sampling={"top_k": 32},
        )
        song.save.assert_called_once_with(output_path)
        pipeline.close.assert_called_once()
        self.assertEqual(output_path.read_bytes(), b"RIFF-song")

    def test_health_reports_model_readiness_without_loading_weights(self) -> None:
        with (
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main, "model_status", return_value=COMPLETE_MODELS),
            patch.object(self.main, "cuda_status", return_value={"available": False}),
        ):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["models"], COMPLETE_MODELS)

    def test_startup_quality_and_retention_configuration_include_yue2(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        maintenance = (REPOSITORY_DIR / "main/storage_maintenance.py").read_text(encoding="utf-8")
        readme = (REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")

        self.assertIn('YUE2_3B_PORT="${YUE2_3B_PORT:-8314}"', startup)
        self.assertIn("'/v1/m-a-p/YuE2'", startup)
        self.assertIn('setsid uv run --no-sync --project "$YUE2_3B_PROJECT_DIR"', startup)
        self.assertEqual(startup.count("setsid uv run --no-sync --project"), 12)
        self.assertIn("model-audio/music_yue2-3b", quality_gate)
        self.assertIn('"yue2_3b_"', maintenance)
        self.assertIn("YuE2-3B 歌曲生成", readme)

        configuration = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        lock = (SERVICE_DIR / "uv.lock").read_text(encoding="utf-8")
        self.assertIn('requires-python = "==3.12.13"', configuration)
        self.assertIn('name = "yue2-3b-service"', lock)


if __name__ == "__main__":
    unittest.main()
