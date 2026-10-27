"""MERT-v2-FullSong 服务、模型加载映射与启动接入的无模型回归。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "embedding_mert-v2-fullsong"
TEST_ROOT = Path()


def load_service(name: str):
    """在隔离环境中加载轻量 API 模块和运行时辅助函数。"""
    main_path = SERVICE_DIR / "main.py"
    main_spec = importlib.util.spec_from_file_location(name, main_path)
    assert main_spec and main_spec.loader
    main_module = importlib.util.module_from_spec(main_spec)
    runtime_spec = importlib.util.spec_from_file_location(
        f"{name}_runtime", SERVICE_DIR / "runtime.py"
    )
    assert runtime_spec and runtime_spec.loader
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    environment = {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "MERT_V2_FULLSONG_MODEL_DIR": str(TEST_ROOT / "models" / "MERT-v2-FullSong"),
        "MERT_V2_FULLSONG_OUTPUT_DIR": str(TEST_ROOT / "storage" / "mert_v2_fullsong"),
        "MERT_V2_FULLSONG_UPLOAD_STAGING_DIR": str(TEST_ROOT / "uploads"),
        "MERT_V2_FULLSONG_WORKER_TMP_DIR": str(TEST_ROOT / "worker"),
        "LOCAL_FILES_ONLY": "1",
        "GPU_LOCK_WAIT_TIMEOUT": "1",
        "CUDA_RELEASE_DELAY": "0",
        "MERT_V2_FULLSONG_REQUEST_TIMEOUT": "5",
        "MERT_V2_FULLSONG_MIN_AUDIO_SECONDS": "30",
        "MERT_V2_FULLSONG_MAX_AUDIO_SECONDS": "360",
        "UPLOAD_MAX_BYTES": "1048576",
    }
    source_path = [str(SERVICE_DIR), str(REPOSITORY_DIR / "llm_local_runtime/src"), *sys.path]
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(sys, "path", source_path),
        patch.dict(sys.modules, {"runtime": runtime_module, name: main_module}),
    ):
        runtime_spec.loader.exec_module(runtime_module)
        main_spec.loader.exec_module(main_module)
    return main_module


def load_worker(name: str):
    """加载只在任务执行时导入模型和 Torch 的 worker 模块。"""
    worker_path = SERVICE_DIR / "worker.py"
    spec = importlib.util.spec_from_file_location(name, worker_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MertV2FullSongServiceTests(unittest.TestCase):
    """验证请求契约、模型加载参数、存储和进程生命周期。"""

    def setUp(self) -> None:
        global TEST_ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix="mert-v2-fullsong-tests-")
        self.addCleanup(self.temporary.cleanup)
        TEST_ROOT = Path(self.temporary.name)
        self.main = load_service("mert_v2_fullsong_service_test_main")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)

    def test_model_status_validates_custom_code_and_manifest_size(self) -> None:
        model_dir = Path(self.main.MODEL_DIR)
        model_dir.mkdir(parents=True)
        for name in self.main.REQUIRED_MODEL_FILES:
            if name != "weights_manifest.json":
                (model_dir / name).touch()
        weight_path = model_dir / "model.safetensors"
        weight_path.write_bytes(b"weights")
        (model_dir / "weights_manifest.json").write_text(
            json.dumps({"filename": "model.safetensors", "bytes": 7}),
            encoding="utf-8",
        )

        self.assertTrue(self.main.model_status()["complete"])
        weight_path.write_bytes(b"short")
        self.assertFalse(self.main.model_status()["complete"])

    def test_health_reports_readiness_without_loading_model(self) -> None:
        ready_model = {"path": "/models/MERT-v2-FullSong", "complete": True, "missing": []}
        with (
            patch.object(self.main, "model_status", return_value=ready_model),
            patch.object(self.main, "module_available", return_value=True),
            patch.object(self.main, "cuda_status", return_value={"available": True}),
            patch.object(self.main, "shutil") as shutil_module,
        ):
            shutil_module.which.return_value = "/usr/bin/tool"
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])
        self.assertTrue(response.json()["available"]["model"]["complete"])
        self.assertEqual(response.json()["runtime"]["sampling_rate"], 24_000)
        self.assertEqual(response.json()["runtime"]["frame_rate"], 25)

    def test_embedding_route_returns_json_and_persists_result(self) -> None:
        normalized = TEST_ROOT / "worker" / "normalized.wav"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(b"normalized-wave")
        output_path = TEST_ROOT / "worker" / "embedding.json"
        output_path.write_text('{"embedding":[0.25]}', encoding="utf-8")

        with (
            patch.object(self.main, "ensure_runtime_ready"),
            patch.object(self.main, "inspect_audio_duration", return_value=45.0),
            patch.object(self.main, "normalize_audio", return_value=normalized),
            patch.object(
                self.main.manager,
                "build_payload",
                return_value={"output": "embedding", "audio_path": str(normalized)},
            ) as build_payload,
            patch.object(self.main, "execute_worker", return_value=output_path),
        ):
            response = self.client.post(
                "/v1/m-a-p/MERT-v2-FullSong",
                data={"output": "embedding"},
                files={"audio": ("song.wav", b"source-audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["x-mert-v2-fullsong-output"], "embedding")
        self.assertEqual(len(response.headers["x-mert-v2-fullsong-audio-sha256"]), 64)
        self.assertEqual(response.json(), {"embedding": [0.25]})
        build_payload.assert_called_once_with(normalized, "embedding", 45.0)
        self.assertFalse(normalized.exists())
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])
        self.assertEqual(
            len(list(self.main.OUTPUT_DIR.glob("mert_v2_fullsong_embedding_*.json"))), 1
        )
        self.assertFalse(output_path.exists())

    def test_frames_route_returns_npz_and_rejects_short_audio_before_worker(self) -> None:
        normalized = TEST_ROOT / "worker" / "normalized.wav"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(b"normalized-wave")
        output_path = TEST_ROOT / "worker" / "frames.npz"
        output_path.write_bytes(b"NPZ-features")

        with (
            patch.object(self.main, "ensure_runtime_ready"),
            patch.object(self.main, "inspect_audio_duration", return_value=30.0),
            patch.object(self.main, "normalize_audio", return_value=normalized),
            patch.object(
                self.main.manager,
                "build_payload",
                return_value={"output": "frames", "audio_path": str(normalized)},
            ),
            patch.object(self.main, "execute_worker", return_value=output_path),
        ):
            response = self.client.post(
                "/v1/m-a-p/MERT-v2-FullSong",
                data={"output": "frames"},
                files={"audio": ("song.mp3", b"source-audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertEqual(response.content, b"NPZ-features")
        self.assertEqual(
            list(self.main.OUTPUT_DIR.glob("mert_v2_fullsong_frames_*.npz")),
            [
                self.main.OUTPUT_DIR
                / response.headers["content-disposition"].split('filename="')[1][:-1]
            ],
        )

        with (
            patch.object(self.main, "ensure_runtime_ready"),
            patch.object(self.main, "inspect_audio_duration", side_effect=ValueError("短于 30 秒")),
            patch.object(self.main, "normalize_audio") as normalize_audio,
            patch.object(self.main, "execute_worker") as execute_worker,
        ):
            rejected = self.client.post(
                "/v1/m-a-p/MERT-v2-FullSong",
                files={"audio": ("short.wav", b"source-audio", "audio/wav")},
            )

        self.assertEqual(rejected.status_code, 422)
        normalize_audio.assert_not_called()
        execute_worker.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_invalid_output_and_missing_audio_do_not_start_worker(self) -> None:
        with patch.object(self.main, "execute_worker") as execute_worker:
            missing_audio = self.client.post("/v1/m-a-p/MERT-v2-FullSong", data={})
            invalid_output = self.client.post(
                "/v1/m-a-p/MERT-v2-FullSong",
                data={"output": "hidden_states"},
                files={"audio": ("song.wav", b"source-audio", "audio/wav")},
            )

        self.assertEqual(missing_audio.status_code, 422)
        self.assertEqual(invalid_output.status_code, 422)
        execute_worker.assert_not_called()

    def test_worker_uses_official_transformers_auto_interfaces(self) -> None:
        worker = load_worker("mert_v2_fullsong_worker_loader")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True)),
            get_num_threads=Mock(return_value=8),
            set_num_threads=Mock(),
        )
        processor = types.SimpleNamespace(sampling_rate=24_000)
        model = Mock()
        model.config = types.SimpleNamespace(sampling_rate=24_000)
        model.eval.return_value = model
        model.to.return_value = model
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoFeatureExtractor = Mock()
        transformers_module.AutoModel = Mock()
        transformers_module.AutoFeatureExtractor.from_pretrained.return_value = processor
        transformers_module.AutoModel.from_pretrained.return_value = model

        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": transformers_module}):
            loaded_torch, loaded_processor, loaded_model = worker.load_model(
                {
                    "device": "cuda:0",
                    "local_files_only": True,
                    "model_dir": "/models/MERT-v2-FullSong",
                }
            )

        self.assertIs(loaded_torch, fake_torch)
        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        transformers_module.AutoFeatureExtractor.from_pretrained.assert_called_once_with(
            "/models/MERT-v2-FullSong",
            trust_remote_code=True,
            local_files_only=True,
        )
        transformers_module.AutoModel.from_pretrained.assert_called_once_with(
            "/models/MERT-v2-FullSong",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )

    def test_worker_run_cleans_cuda_state_after_inference(self) -> None:
        worker = load_worker("mert_v2_fullsong_worker_dispatch")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=Mock(return_value=True),
                synchronize=Mock(),
                empty_cache=Mock(),
                ipc_collect=Mock(),
            ),
        )
        processor = Mock()
        model = Mock()
        output_path = TEST_ROOT / "worker" / "result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(worker, "load_model", return_value=(fake_torch, processor, model)),
            patch.object(worker, "extract_features") as extract,
        ):
            worker.run({"output": "embedding"}, output_path)

        extract.assert_called_once_with(
            {"output": "embedding"}, output_path, fake_torch, processor, model
        )
        fake_torch.cuda.synchronize.assert_called_once()
        fake_torch.cuda.empty_cache.assert_called_once()
        fake_torch.cuda.ipc_collect.assert_called_once()

    def test_worker_manager_cleans_files_after_success_and_timeout(self) -> None:
        process = Mock(returncode=0)
        popen = Mock(return_value=process)

        def write_output(timeout: float) -> tuple[str, str]:
            command = popen.call_args.args[0]
            request_path = Path(command[command.index("--input-json") + 1])
            output_path = Path(command[command.index("--output-file") + 1])
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["output"], "embedding")
            output_path.write_text('{"embedding":[]}', encoding="utf-8")
            return "", ""

        process.communicate.side_effect = write_output
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            output_path = self.main.manager.run_worker({"output": "embedding"})

        self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"embedding": []})
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        terminate.assert_called_once_with(process, "MERT-v2-FullSong")
        output_path.unlink()
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

        timeout_process = Mock(returncode=None)
        timeout_process.communicate.side_effect = subprocess.TimeoutExpired("worker", 5)
        with (
            patch.object(self.main.subprocess, "Popen", return_value=timeout_process),
            patch.object(self.main, "terminate_process_group") as terminate_timeout,
            self.assertRaises(TimeoutError),
        ):
            self.main.manager.run_worker({"output": "frames"})
        terminate_timeout.assert_called_once_with(timeout_process, "MERT-v2-FullSong")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_startup_quality_docs_and_project_lock_register_service(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        readme = (REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")
        agent_guide = (REPOSITORY_DIR / "AGENTS.md").read_text(encoding="utf-8")
        project = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        lock = (SERVICE_DIR / "uv.lock").read_text(encoding="utf-8")

        self.assertIn('MERT_V2_FULLSONG_PORT="${MERT_V2_FULLSONG_PORT:-8317}"', startup)
        self.assertIn('setsid uv run --no-sync --project "$MERT_V2_FULLSONG_PROJECT_DIR"', startup)
        self.assertIn("'/v1/m-a-p/MERT-v2-FullSong'", startup)
        self.assertIn("model-audio/embedding_mert-v2-fullsong", quality_gate)
        self.assertIn("/v1/m-a-p/MERT-v2-FullSong", readme)
        self.assertIn("model-audio/embedding_mert-v2-fullsong/` | 8317", agent_guide)
        self.assertEqual(startup.count("setsid uv run --no-sync --project"), 17)
        self.assertIn('requires-python = "==3.12.13"', project)
        self.assertIn('name = "mert-v2-fullsong-service"', project)
        self.assertIn('name = "mert-v2-fullsong-service"', lock)


if __name__ == "__main__":
    unittest.main()
