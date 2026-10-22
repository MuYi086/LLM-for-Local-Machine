"""YuE2-Vae API、worker 与启动接入的无模型回归测试。"""

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
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "YuE2-Vae"
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
        "YUE2_VAE_MODEL_DIR": str(TEST_ROOT / "models" / "YuE2-Vae"),
        "YUE2_VAE_OUTPUT_DIR": str(TEST_ROOT / "storage" / "yue2_vae"),
        "YUE2_VAE_UPLOAD_STAGING_DIR": str(TEST_ROOT / "uploads"),
        "YUE2_VAE_WORKER_TMP_DIR": str(TEST_ROOT / "worker"),
        "LOCAL_FILES_ONLY": "1",
        "GPU_LOCK_WAIT_TIMEOUT": "1",
        "CUDA_RELEASE_DELAY": "0",
        "YUE2_VAE_REQUEST_TIMEOUT": "5",
        "YUE2_VAE_MAX_AUDIO_SECONDS": "10",
        "YUE2_VAE_MAX_LATENT_FRAMES": "250",
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


class YuE2VaeServiceTests(unittest.TestCase):
    """验证服务契约、原生模型调用映射和临时文件清理。"""

    def setUp(self) -> None:
        global TEST_ROOT
        self.temporary = tempfile.TemporaryDirectory(prefix="yue2-vae-tests-")
        self.addCleanup(self.temporary.cleanup)
        TEST_ROOT = Path(self.temporary.name)
        self.main = load_service("yue2_vae_service_test_main")
        self.client = TestClient(self.main.app)
        self.addCleanup(self.client.close)

    def test_route_returns_and_persists_decoded_wav(self) -> None:
        output_path = TEST_ROOT / "worker" / "decoded.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF-native-wave")

        with (
            patch.object(self.main, "validate_latent", return_value=3) as validate,
            patch.object(
                self.main.manager,
                "build_payload",
                return_value={"operation": "decode", "latent_path": "/tmp/latent.npy"},
            ) as build_payload,
            patch.object(self.main, "execute_worker", return_value=output_path),
        ):
            response = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "decode"},
                files={"latent": ("latent.npy", b"NPY-bytes", "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-yue2-vae-operation"], "decode")
        self.assertEqual(response.headers["x-yue2-vae-sample-rate"], "48000")
        self.assertEqual(response.headers["x-yue2-vae-latent-frames"], "3")
        self.assertEqual(response.content, b"RIFF-native-wave")
        validate.assert_called_once()
        build_payload.assert_called_once()
        self.assertEqual(len(response.headers["x-yue2-vae-input-sha256"]), 64)
        saved_outputs = list(self.main.OUTPUT_DIR.glob("yue2_vae_decode_*.wav"))
        self.assertEqual(len(saved_outputs), 1)
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])
        self.assertFalse(output_path.exists())

    def test_route_encodes_audio_and_returns_native_npy(self) -> None:
        normalized_path = TEST_ROOT / "worker" / "normalized.wav"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_bytes(b"normalized")
        output_path = TEST_ROOT / "worker" / "encoded.npy"
        output_path.write_bytes(b"NPY-native-latents")
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.load = Mock(return_value=types.SimpleNamespace(shape=(4, 64)))

        with (
            patch.object(self.main, "inspect_audio_duration", return_value=1.0),
            patch.object(self.main, "normalize_audio", return_value=normalized_path),
            patch.object(
                self.main.manager,
                "build_payload",
                return_value={"operation": "encode", "audio_path": str(normalized_path)},
            ) as build_payload,
            patch.object(self.main, "execute_worker", return_value=output_path),
            patch.dict(sys.modules, {"numpy": numpy_stub}),
        ):
            response = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "encode", "sample": "false", "seed": "42"},
                files={"audio": ("song.mp3", b"source-audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/octet-stream")
        self.assertEqual(response.headers["x-yue2-vae-operation"], "encode")
        self.assertEqual(response.headers["x-yue2-vae-latent-frames"], "4")
        self.assertEqual(response.content, b"NPY-native-latents")
        build_payload.assert_called_once_with(
            "encode",
            audio_path=normalized_path,
            sample=False,
            seed=42,
        )
        self.assertFalse(normalized_path.exists())
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_route_rejects_wrong_inputs_before_worker(self) -> None:
        with patch.object(self.main, "execute_worker") as execute_worker:
            missing_decode_input = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "decode"},
            )
            wrong_decode_suffix = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "decode"},
                files={"latent": ("latent.txt", b"not-npy", "application/octet-stream")},
            )
            missing_encode_input = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "encode"},
            )

        self.assertEqual(missing_decode_input.status_code, 422)
        self.assertEqual(wrong_decode_suffix.status_code, 422)
        self.assertEqual(missing_encode_input.status_code, 422)
        execute_worker.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_model_status_checks_manifest_weight_size_and_custom_code(self) -> None:
        model_dir = Path(self.main.MODEL_DIR)
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        (model_dir / "modeling_vae.py").touch()
        weight_path = model_dir / "model.safetensors"
        weight_path.write_bytes(b"weights")
        (model_dir / "weights_manifest.json").write_text(
            json.dumps({"files": {"model.safetensors": {"bytes": 7}}}),
            encoding="utf-8",
        )

        self.assertTrue(self.main.model_status()["complete"])
        weight_path.write_bytes(b"short")
        self.assertFalse(self.main.model_status()["complete"])

    def test_payload_preflight_rejects_missing_model_before_gpu_queue(self) -> None:
        with (
            patch.object(self.main, "validate_latent", return_value=2),
            patch.object(
                self.main,
                "model_status",
                return_value={"complete": False, "missing": ["model.safetensors"]},
            ),
            patch.object(self.main, "execute_worker") as execute_worker,
        ):
            response = self.client.post(
                "/v1/m-a-p/YuE2-Vae",
                data={"operation": "decode"},
                files={"latent": ("latent.npy", b"NPY-bytes", "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 503, response.text)
        execute_worker.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_worker_loads_official_model_in_decoder_only_mode_for_decode(self) -> None:
        worker = load_worker("yue2_vae_test_worker_load")
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(),
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
            ),
            cuda=types.SimpleNamespace(is_available=Mock(return_value=True)),
            set_float32_matmul_precision=Mock(),
            get_num_threads=Mock(return_value=8),
            set_num_threads=Mock(),
        )
        model = Mock()
        transformers_module = types.ModuleType("transformers")
        auto_model = Mock()
        auto_model.from_pretrained.return_value = model
        transformers_module.AutoModel = auto_model

        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": transformers_module}):
            loaded_torch, loaded_model = worker.load_model(
                {
                    "device": "cuda",
                    "local_files_only": True,
                    "model_dir": "/models/YuE2-Vae",
                    "operation": "decode",
                }
            )

        self.assertIs(loaded_torch, fake_torch)
        self.assertIs(loaded_model, model)
        auto_model.from_pretrained.assert_called_once_with(
            "/models/YuE2-Vae",
            trust_remote_code=True,
            local_files_only=True,
            decoder_only=True,
            device="cuda",
        )

    def test_worker_dispatches_encode_and_decode_and_releases_cuda(self) -> None:
        worker = load_worker("yue2_vae_test_worker_dispatch")
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        model = Mock()
        output_path = TEST_ROOT / "worker" / "output.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(worker, "load_model", return_value=(fake_torch, model)),
            patch.object(worker, "decode_latents") as decode,
            patch.object(worker, "encode_audio") as encode,
        ):
            worker.run({"operation": "decode"}, output_path)
            worker.run({"operation": "encode"}, output_path)

        decode.assert_called_once()
        encode.assert_called_once()

    def test_worker_manager_cleans_request_and_output_after_success_or_timeout(self) -> None:
        output_bytes = b"RIFF-or-NPY"
        process = Mock(returncode=0)
        popen = Mock(return_value=process)

        def write_output(timeout: float) -> tuple[str, str]:
            command = popen.call_args.args[0]
            request_path = Path(command[command.index("--input-json") + 1])
            output_path = Path(command[command.index("--output-file") + 1])
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["operation"], "decode")
            output_path.write_bytes(output_bytes)
            return "", ""

        process.communicate.side_effect = write_output
        with (
            patch.object(self.main.subprocess, "Popen", popen),
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            output_path = self.main.manager.run_worker({"operation": "decode"})

        self.assertEqual(output_path.read_bytes(), output_bytes)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        terminate.assert_called_once_with(process, "YuE2-Vae")
        output_path.unlink()
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

        timeout_process = Mock(returncode=None)
        timeout_process.communicate.side_effect = subprocess.TimeoutExpired("worker", 5)
        with (
            patch.object(self.main.subprocess, "Popen", return_value=timeout_process),
            patch.object(self.main, "terminate_process_group") as terminate_timeout,
            self.assertRaises(TimeoutError),
        ):
            self.main.manager.run_worker({"operation": "encode"})
        terminate_timeout.assert_called_once_with(timeout_process, "YuE2-Vae")
        self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])

    def test_startup_quality_and_docs_register_the_service(self) -> None:
        startup = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        quality_gate = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text(encoding="utf-8")
        readme = (REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")
        agent_guide = (REPOSITORY_DIR / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn('YUE2_VAE_PORT="' + chr(36) + "{YUE2_VAE_PORT:-8316}", startup)
        self.assertIn('"$YUE2_VAE_PROJECT_DIR"', startup)
        self.assertIn("'/v1/m-a-p/YuE2-Vae'", startup)
        self.assertIn("YuE2-Vae", quality_gate)
        self.assertIn("/v1/m-a-p/YuE2-Vae", readme)
        self.assertIn("model-audio/YuE2-Vae/` | 8316", agent_guide)


if __name__ == "__main__":
    unittest.main()
