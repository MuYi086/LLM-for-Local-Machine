"""Seed-VC HTTP 契约、官方推理适配和 worker 生命周期的无模型回归。"""

from __future__ import annotations

import ast
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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from unitale_runtime import GpuLockTimeoutError

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "seed-vc"


def load_module(name: str, path: Path):
    """以唯一模块名加载服务，避免多个 main.py 互相覆盖。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav_bytes() -> bytes:
    """生成短 PCM WAV，仅供无模型文件和响应测试。"""
    destination = io.BytesIO()
    with wave.open(destination, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(b"\0\0" * 2205)
    return destination.getvalue()


class SeedVcMigrationTests(unittest.TestCase):
    """隔离环境与临时文件，验证新增服务的所有模型边界。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="seed-vc-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "CLONE_STORAGE_DIR": str(self.root / "storage/clone"),
            "PROMPTS_DIR": str(self.root / "storage/clone"),
            "TIMBRE_STORAGE_DIR": str(self.root / "storage/timbre"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "cache/gpu.lock"),
            "SEED_VC_MODEL_DIR": str(self.root / "models"),
            "SEED_VC_CODE_PATH": str(self.root / "code"),
            "SEED_VC_WHISPER_MODEL_DIR": str(self.root / "whisper"),
            "SEED_VC_VOCODER_MODEL_DIR": str(self.root / "vocoder"),
            "SEED_VC_F0_VOCODER_MODEL_DIR": str(self.root / "f0-vocoder"),
            "SEED_VC_STYLE_ENCODER_CHECKPOINT": str(self.root / "campplus.bin"),
            "SEED_VC_RMVPE_CHECKPOINT": str(self.root / "rmvpe.pt"),
            "SEED_VC_OUTPUT_DIR": str(self.root / "storage/clone"),
            "SEED_VC_WORKER_TMP_DIR": str(self.root / "cache/worker"),
            "SEED_VC_REQUEST_TIMEOUT": "5",
            "SEED_VC_PORT": "8381",
            "GPU_LOCK_WAIT_TIMEOUT": "1",
            "LOCAL_FILES_ONLY": "1",
            "CUDA_RELEASE_DELAY": "0",
        }
        with (
            patch.dict(
                os.environ,
                {
                    **{
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith("SEED_VC_")
                    },
                    **environment,
                },
                clear=True,
            ),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("seed_vc_main_for_tests", SERVICE_DIR / "main.py")
        self.worker = load_module("seed_vc_worker_for_tests", SERVICE_DIR / "worker.py")
        self.runtime = load_module("seed_vc_runtime_for_tests", SERVICE_DIR / "seed_vc_runtime.py")
        self.request_data = {
            "source_audio_path": "source.wav",
            "reference_audio_path": "target.wav",
        }
        for f0 in (False, True):
            request = self.main.SeedVcRequest(**self.request_data, f0_condition=f0)
            for path in self.main.required_files(request).values():
                if path == Path(self.main.WORKER_SCRIPT):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        (self.main.WHISPER_MODEL_DIR / "model.safetensors").touch()
        self.write_config()
        self.write_config(True)
        self.client = TestClient(self.main.app)
        self.audio_bytes = wav_bytes()
        for logical_path in self.request_data.values():
            self.main.reference_store.clone_path(logical_path).write_bytes(self.audio_bytes)

    @contextmanager
    def fake_gpu_lock(self, label: str):
        """GPU 队列替身不调用 nvidia-smi。"""
        yield SimpleNamespace(as_dict=lambda: {"label": label, "worker_seconds": 0.1})

    def test_only_final_routes_and_lightweight_health(self) -> None:
        routes = {
            (method, route.path)
            for route in self.main.app.routes
            if hasattr(route, "methods") and route.path.startswith("/v1/")
            for method in route.methods
        }
        self.assertEqual(
            routes,
            {
                ("GET", "/v1/health"),
                ("POST", "/v1/upload_audio"),
                ("GET", "/v1/check/audio"),
                ("POST", "/v1/seedVc/voiceConversion"),
            },
        )
        with patch.object(self.main.manager, "run_worker") as run_worker:
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        health = response.json()
        self.assertTrue(health["available"]["voice_conversion"]["ready"])
        self.assertTrue(health["available"]["f0_conversion"]["ready"])
        self.assertEqual(health["runtime"]["worker_python"], sys.executable)
        self.assertIn("storage", health)
        run_worker.assert_not_called()
        (self.main.WHISPER_MODEL_DIR / "model.safetensors").unlink()
        self.assertFalse(
            self.client.get("/v1/health").json()["available"]["voice_conversion"]["ready"]
        )
        for relative in ("main.py", "seed_vc_runtime.py"):
            tree = ast.parse((SERVICE_DIR / relative).read_text())
            imports = [
                alias.name if isinstance(node, ast.Import) else node.module
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (node.names if isinstance(node, ast.Import) else [None])
            ]
            self.assertFalse(set(imports) & {"torch", "torchaudio", "transformers", "inference"})

    def test_upload_check_and_timbre_reference_avoid_duplicate_audio(self) -> None:
        timbre_path = self.main.TIMBRE_STORAGE_DIR / "designed.wav"
        timbre_path.write_bytes(self.audio_bytes)
        response = self.client.post(
            "/v1/upload_audio",
            files={"audio": ("designed.wav", self.audio_bytes, "audio/wav")},
            data={"full_path": "webui/designed.wav", "prompt_text": "参考台词。"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.main.reference_store.clone_path("webui/designed.wav").exists())
        self.assertEqual(
            self.main.reference_store.prompt_audio_path("webui/designed.wav"), timbre_path
        )
        checked = self.client.get("/v1/check/audio", params={"file_name": "webui/designed.wav"})
        self.assertTrue(checked.json()["exists"])
        self.assertTrue(checked.json()["has_prompt_text"])
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").iterdir()), [])

    def test_ordinary_upload_is_committed_with_digest(self) -> None:
        audio = self.audio_bytes + b"ordinary"
        response = self.client.post(
            "/v1/upload_audio",
            files={"audio": ("ordinary.wav", audio, "audio/wav")},
            data={"full_path": "webui/ordinary.wav"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.main.reference_store.prompt_audio_path("webui/ordinary.wav").read_bytes(), audio
        )
        self.assertEqual(len(response.json()["sha256"]), 64)

    def test_upload_validation_and_limit_cleanup(self) -> None:
        for filename, content_type in (("bad.txt", "audio/wav"), ("bad.wav", "text/plain")):
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/v1/upload_audio",
                    files={"audio": (filename, self.audio_bytes, content_type)},
                    data={"full_path": "bad.wav"},
                )
                self.assertEqual(response.status_code, 422)
        with patch.dict(os.environ, {"UPLOAD_MAX_BYTES": "32"}):
            response = self.client.post(
                "/v1/upload_audio",
                files={"audio": ("too-large.wav", self.audio_bytes, "audio/wav")},
                data={"full_path": "too-large.wav"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").iterdir()), [])

    def test_defaults_payload_and_f0_model_selection(self) -> None:
        request = self.main.SeedVcRequest(**self.request_data)
        payload = self.main.manager.build_worker_payload(request)
        self.assertEqual(
            (request.diffusion_steps, request.length_adjust, request.inference_cfg_rate),
            (30, 1.0, 0.7),
        )
        self.assertEqual(
            payload["source_audio_path"], str(self.main.reference_store.clone_path("source.wav"))
        )
        self.assertEqual(
            payload["reference_audio_path"], str(self.main.reference_store.clone_path("target.wav"))
        )
        self.assertEqual(payload["checkpoint_path"], str(self.main.CHECKPOINT_PATH))
        request = self.main.SeedVcRequest(
            **self.request_data, f0_condition=True, semi_tone_shift=3, auto_f0_adjust=True
        )
        payload = self.main.manager.build_worker_payload(request)
        self.assertEqual(payload["checkpoint_path"], str(self.main.F0_CHECKPOINT_PATH))
        self.assertEqual(payload["vocoder_model_dir"], str(self.main.F0_VOCODER_MODEL_DIR))
        self.assertEqual(payload["semi_tone_shift"], 3)

    def test_conversion_response_persistence_metrics_and_release(self) -> None:
        with (
            patch.object(self.main, "gpu_runtime_lock", side_effect=self.fake_gpu_lock),
            patch.object(
                self.main.manager, "run_worker", return_value=self.audio_bytes
            ) as run_worker,
            patch.object(self.main, "wait_after_cuda_release") as release,
        ):
            response = self.client.post("/v1/seedVc/voiceConversion", json=self.request_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, self.audio_bytes)
        run_worker.assert_called_once()
        release.assert_called_once()
        saved = list(self.main.OUTPUT_DIR.glob("seed_vc_*.wav"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), self.audio_bytes)
        self.assertEqual(self.main.manager.last_metrics["worker_seconds"], 0.1)

    def test_invalid_parameters_and_missing_inputs_fail_before_gpu_queue(self) -> None:
        invalid = (
            {"diffusion_steps": 0},
            {"length_adjust": float("nan")},
            {"length_adjust": float("inf")},
            {"length_adjust": 0},
            {"inference_cfg_rate": 2},
            {"semi_tone_shift": 1},
            {"auto_f0_adjust": True},
            {"text": "不接受 TTS 字段"},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.main.SeedVcRequest(**self.request_data, **parameters)
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                "/v1/seedVc/voiceConversion",
                json={**self.request_data, "source_audio_path": "missing.wav"},
            )
            self.assertEqual(response.status_code, 404)
            self.main.CHECKPOINT_PATH.unlink()
            response = self.client.post("/v1/seedVc/voiceConversion", json=self.request_data)
            self.assertEqual(response.status_code, 503)
            lock.assert_not_called()

    def test_f0_auxiliary_is_required_only_for_f0_mode(self) -> None:
        self.main.RMVPE_CHECKPOINT.unlink()
        self.main.manager.build_worker_payload(self.main.SeedVcRequest(**self.request_data))
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                "/v1/seedVc/voiceConversion", json={**self.request_data, "f0_condition": True}
            )
        self.assertEqual(response.status_code, 503)
        lock.assert_not_called()

    def test_invalid_config_is_rejected_before_gpu_queue(self) -> None:
        for invalid in (
            "{}",
            "model_params: [",
            "null",
            '{"model_params": {"speech_tokenizer": null}}',
        ):
            with self.subTest(config=invalid), patch.object(self.main, "gpu_runtime_lock") as lock:
                self.main.CONFIG_PATH.write_text(invalid)
                response = self.client.post("/v1/seedVc/voiceConversion", json=self.request_data)
                self.assertEqual(response.status_code, 503)
                lock.assert_not_called()

    def test_queue_timeout_is_503_without_worker(self) -> None:
        with (
            patch.object(self.main, "gpu_runtime_lock", side_effect=GpuLockTimeoutError("seed", 1)),
            patch.object(self.main.manager, "run_worker") as worker,
        ):
            response = self.client.post("/v1/seedVc/voiceConversion", json=self.request_data)
        self.assertEqual(response.status_code, 503)
        worker.assert_not_called()

    def test_failed_worker_releases_queue_and_does_not_persist_result(self) -> None:
        with (
            patch.object(self.main, "gpu_runtime_lock", side_effect=self.fake_gpu_lock),
            patch.object(self.main.manager, "run_worker", side_effect=RuntimeError("model error")),
            patch.object(self.main, "wait_after_cuda_release") as release,
        ):
            response = self.client.post("/v1/seedVc/voiceConversion", json=self.request_data)
        self.assertEqual(response.status_code, 500)
        release.assert_called_once()
        self.assertEqual(list(self.main.OUTPUT_DIR.glob("seed_vc_*.wav")), [])

    def test_worker_interpreter_offline_process_group_and_cleanup(self) -> None:
        for behavior in ("success", "failure", "timeout", "invalid-wav", "spawn-failure"):
            with self.subTest(behavior=behavior):
                captured = {}
                process = MagicMock()
                process.pid = None
                process.returncode = 1 if behavior == "failure" else 0
                process.poll.return_value = None if behavior == "timeout" else process.returncode

                def communicate(timeout, behavior=behavior, captured=captured):
                    if behavior == "timeout":
                        raise subprocess.TimeoutExpired("worker", timeout)
                    Path(captured["command"][-1]).write_bytes(
                        b"not wav" if behavior == "invalid-wav" else self.audio_bytes
                    )
                    return "ok", "model failed" if behavior == "failure" else ""

                process.communicate.side_effect = communicate

                def spawn(command, behavior=behavior, captured=captured, process=process, **kwargs):
                    captured.update(command=command, kwargs=kwargs)
                    if behavior == "spawn-failure":
                        raise OSError("cannot spawn")
                    return process

                with patch.object(self.main.subprocess, "Popen", side_effect=spawn):
                    if behavior == "success":
                        self.assertEqual(
                            self.main.manager.run_worker({"test": True}), self.audio_bytes
                        )
                    else:
                        with self.assertRaises((OSError, RuntimeError)):
                            self.main.manager.run_worker({"test": True})
                self.assertEqual(captured["command"][0], sys.executable)
                self.assertEqual(captured["command"][1], self.main.WORKER_SCRIPT)
                self.assertTrue(captured["kwargs"]["start_new_session"])
                self.assertEqual(captured["kwargs"]["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(list(self.main.WORKER_TMP_DIR.iterdir()), [])
                if behavior == "timeout":
                    process.terminate.assert_called_once()
                if behavior != "spawn-failure":
                    process.wait.assert_called()

    def test_completed_parent_still_cleans_descendant_group(self) -> None:
        process = MagicMock(pid=12345, returncode=0)
        process.poll.return_value = 0
        with patch.object(self.runtime.os, "killpg") as killpg:
            self.runtime.terminate_process_group(process)
        self.assertEqual(killpg.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[-1].args, (12345, signal.SIGKILL))

    def test_atomic_result_failure_leaves_no_partial_output(self) -> None:
        with (
            patch.object(self.runtime.os, "replace", side_effect=OSError("write failed")),
            self.assertRaises(OSError),
        ):
            self.runtime.persist_audio_bytes(self.audio_bytes, self.root / "outputs")
        self.assertEqual(list((self.root / "outputs").iterdir()), [])

    def write_config(self, f0: bool = False) -> Path:
        """写入可被 YAML 解析的最小官方模式配置。"""
        path = self.main.F0_CONFIG_PATH if f0 else self.main.CONFIG_PATH
        path.write_text(
            json.dumps(
                {
                    "model_params": {
                        "speech_tokenizer": {"type": "whisper", "name": "openai/whisper-small"},
                        "vocoder": {"type": "bigvgan", "name": "nvidia/bigvgan"},
                        "length_regulator": {"f0_condition": f0},
                        "DiT": {"f0_condition": f0},
                    },
                    "preprocess_params": {
                        "sr": 44100 if f0 else 22050,
                        "spect_params": {"hop_length": 512 if f0 else 256},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_worker_calls_official_entry_with_local_config_and_cleans_artifacts(self) -> None:
        import yaml

        self.write_config()
        payload = self.main.manager.build_worker_payload(
            self.main.SeedVcRequest(**self.request_data, diffusion_steps=42, length_adjust=1.2)
        )
        captured = {}

        def official_main(arguments):
            captured["arguments"] = arguments
            config = yaml.safe_load(Path(arguments.config).read_text())
            self.assertEqual(
                config["model_params"]["speech_tokenizer"]["name"], str(self.main.WHISPER_MODEL_DIR)
            )
            self.assertEqual(
                config["model_params"]["vocoder"]["name"], str(self.main.VOCODER_MODEL_DIR)
            )
            result_dir = Path(arguments.output)
            result_dir.mkdir()
            (result_dir / "official.wav").write_bytes(self.audio_bytes)

        torch = SimpleNamespace(
            inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False)
        )
        inference = SimpleNamespace(main=official_main, device=SimpleNamespace(type="cpu"))
        output_path = self.root / "converted.wav"
        with (
            patch.dict(os.environ),
            patch.object(self.worker, "load_runtime", return_value=(torch, inference)),
        ):
            self.worker.convert_audio(payload, output_path)
        self.assertEqual(output_path.read_bytes(), self.audio_bytes)
        arguments = captured["arguments"]
        self.assertEqual((arguments.diffusion_steps, arguments.length_adjust), (42, 1.2))
        self.assertEqual(arguments.source, payload["source_audio_path"])
        self.assertEqual(arguments.target, payload["reference_audio_path"])
        self.assertFalse(arguments.fp16)
        self.assertFalse(Path(arguments.output).exists())
        self.assertFalse(Path(arguments.config).exists())
        with (
            patch.dict(os.environ),
            patch.object(
                self.worker,
                "load_runtime",
                return_value=(
                    torch,
                    SimpleNamespace(
                        main=MagicMock(side_effect=RuntimeError("inference error")),
                        device=inference.device,
                    ),
                ),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.worker.convert_audio(payload, self.root / "failed.wav")
        self.assertEqual(list(self.root.glob("inference_*")), [])
        self.assertFalse((self.root / "failed.wav").exists())

    def test_config_mode_mismatch_and_local_auxiliary_resolution(self) -> None:
        import yaml

        self.write_config()
        payload = self.main.manager.build_worker_payload(
            self.main.SeedVcRequest(**self.request_data)
        )
        with self.assertRaises(ValueError):
            self.worker.build_runtime_config(
                {**payload, "f0_condition": True}, self.root / "bad.yaml"
            )
        self.write_config(True)
        f0_payload = self.main.manager.build_worker_payload(
            self.main.SeedVcRequest(**self.request_data, f0_condition=True)
        )
        config_path = self.worker.build_runtime_config(f0_payload, self.root / "f0.yaml")
        self.assertEqual(
            yaml.safe_load(config_path.read_text())["model_params"]["vocoder"]["name"],
            str(self.main.F0_VOCODER_MODEL_DIR),
        )
        inference = SimpleNamespace()
        self.worker.install_local_resolver(inference, payload)
        self.assertEqual(
            inference.load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin"),
            str(self.main.STYLE_ENCODER_CHECKPOINT),
        )
        self.assertEqual(
            inference.load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt"),
            str(self.main.RMVPE_CHECKPOINT),
        )
        with self.assertRaises(FileNotFoundError):
            inference.load_custom_model_from_hf("unknown", "missing.bin")

    def test_startup_exposes_8381_and_includes_cleanup_and_wait(self) -> None:
        source = (REPOSITORY_DIR / "start.sh").read_text()
        self.assertIn('SEED_VC_PORT="${SEED_VC_PORT:-8381}"', source)
        self.assertIn(
            'SEED_VC_MODEL_DIR="${SEED_VC_MODEL_DIR:-$HF_MIRROR_DIR/Plachta/Seed-VC}"', source
        )
        self.assertIn('setsid uv run --no-sync --project "$SEED_VC_PROJECT_DIR"', source)
        self.assertIn('python "$SEED_VC_PROJECT_DIR/main.py"', source)
        self.assertIn("'/v1/seedVc/voiceConversion'", source)
        self.assertIn(
            '"$seed_vc_pid"', source[source.index("cleanup() {") : source.index("trap cleanup")]
        )
        self.assertIn('"$seed_vc_pid"', source[source.rindex("wait -n") :])
        for name in (
            "DIFFUSION_STEPS",
            "LENGTH_ADJUST",
            "INFERENCE_CFG_RATE",
            "F0_CONDITION",
            "AUTO_F0_ADJUST",
            "SEMI_TONE_SHIFT",
            "FP16",
        ):
            self.assertNotIn(f"export SEED_VC_{name}", source)


if __name__ == "__main__":
    unittest.main()
