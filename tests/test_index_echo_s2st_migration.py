"""Echo 原生完整语音包接入的无模型回归。"""

import io
import json
import os
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from unittest.mock import Mock, patch

from index_native_test_support import NativeServiceCase, load_module, wav_bytes


class IndexEchoS2stMigrationTests(NativeServiceCase):
    """验证原生语音调用、输入暂存、WAV/info 以及缺失条件拒绝。"""

    project = "Index-Echo-S2ST-9B-GGUF"
    prefix = "INDEX_ECHO_S2ST"
    route = "echoS2ST"

    def post_audio(self, data=None, failure=None):
        """替换原生 worker 边界，保留 multipart 解析和输入清理。"""
        info = {"zh": "源文", "tgt_raw": "Translation", "src_lang": "zh"}
        worker = Mock(
            side_effect=failure, return_value=(wav_bytes(), {"sample_rate": 24_000, "info": info})
        )
        with (
            patch.object(self.main.manager, "build_worker_payload", return_value={"native": True}),
            patch.object(self.main, "gpu_runtime_lock", return_value=nullcontext()) as lock,
            patch.object(self.main, "wait_after_cuda_release") as release,
            patch.object(self.main.manager, "run_worker", worker),
        ):
            response = self.client.post(
                "/v1/indexTranslate/echoS2ST",
                files={"audio": ("source.wav", wav_bytes(), "audio/wav")},
                data=data or {"target_language": "en"},
            )
        if response.status_code == 200 or failure:
            lock.assert_called_once()
            release.assert_called_once()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])
        return response

    def test_audio_in_native_waveform_out(self):
        response = self.post_audio()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["x-audio-sample-rate"], "24000")
        self.assertEqual(response.content, wav_bytes())
        self.assertFalse(hasattr(self.main, "transcribe_audio"))
        self.assertFalse(hasattr(self.main, "synthesize_audio"))

    def test_native_info_is_retained_with_waveform(self):
        response = self.post_audio({"target_language": "en", "return_info": "true"})
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(archive.read("audio.wav"), wav_bytes())
            self.assertEqual(json.loads(archive.read("info.json"))["tgt_raw"], "Translation")

    def test_officially_unsupported_chunk_mode_and_language_are_rejected(self):
        for data in ({"target_language": "fr"}, {"chunk": "true"}, {"mode": "other"}):
            with self.subTest(data=data):
                self.assertEqual(self.post_audio(data).status_code, 422)
        response = self.client.post(
            "/v1/indexTranslate/echoS2ST", json={"audio_path": "old.wav", "target_language": "en"}
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_limits_and_media_types(self):
        with patch.dict(os.environ, {"UPLOAD_MAX_BYTES": "4"}):
            response = self.client.post(
                "/v1/indexTranslate/echoS2ST",
                files={"audio": ("source.wav", wav_bytes(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 413)
        response = self.client.post(
            "/v1/indexTranslate/echoS2ST", files={"audio": ("source.txt", b"text", "text/plain")}
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_missing_native_package_has_no_gpu_or_other_service_calls(self):
        status = {
            "ready": False,
            "available": {"native_model": False, "gpu_memory": False},
            "gpu": "16 GB",
        }
        with (
            patch.object(self.main, "native_runtime_status", return_value=status),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post(
                "/v1/indexTranslate/echoS2ST",
                files={"audio": ("source.wav", wav_bytes(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("文本 GGUF 不能代替", response.text)
        lock.assert_not_called()
        self.assertEqual(list(self.main.UPLOAD_STAGING_DIR.iterdir()), [])

    def test_failed_worker_releases_lock_and_input(self):
        self.assertEqual(self.post_audio(failure=TimeoutError("timeout")).status_code, 504)
        self.assertEqual(self.post_audio(failure=RuntimeError("native error")).status_code, 500)

    def test_worker_lifecycle(self):
        for failure in (None, "exit", "timeout"):
            with self.subTest(failure=failure):
                self.check_manager_cleanup(failure)

    def test_worker_calls_only_official_native_api(self):
        worker = load_module("echo_native_worker_test", self.service_dir / "worker.py")
        model_dir = self.root / "native"
        model_dir.mkdir()
        (model_dir / "modeling_dubbing.py").touch()
        audio = self.root / "source.wav"
        audio.write_bytes(wav_bytes())
        output = self.root / "output.wav"
        model = Mock()

        def finish(*args, **kwargs):
            output.write_bytes(wav_bytes())
            return Mock(), 24_000, {"tgt_raw": "Translation"}

        model.dub.side_effect = finish
        api = Mock()
        api.DubbingBridgeModel.from_pretrained.return_value = model
        with patch.dict(sys.modules, {"modeling_dubbing": api}), patch.object(worker.os, "chdir"):
            result = worker.dub(
                {
                    "model_dir": str(model_dir),
                    "audio_path": str(audio),
                    "output_wav": str(output),
                    "target_language": "en",
                    "return_info": True,
                }
            )
        api.DubbingBridgeModel.from_pretrained.assert_called_once_with(
            str(model_dir), device="cuda"
        )
        model.dub.assert_called_once_with(
            str(audio), lang="en", out_wav=str(output), mode="h2cv", chunk=False, return_info=True
        )
        self.assertEqual(result["info"]["tgt_raw"], "Translation")

    def test_health_and_preflight_use_selected_gpu_without_torch_import(self):
        runtime = load_module(
            "echo_native_runtime_test", self.service_dir / "index_echo_runtime.py"
        )
        for name in (
            "modeling_dubbing.py",
            "config.json",
            "stlm_llm",
            "stlm_ckpt",
            "stlm_omni",
            "cosyvoice3",
            "bridge",
            "code",
        ):
            (self.model_path.parent / name).touch()
        gpu = subprocess.CompletedProcess([], 0, stdout="16376\n24576\n", stderr="")
        with (
            patch.object(
                runtime.importlib.metadata,
                "version",
                side_effect=lambda name: runtime.NATIVE_PACKAGES[name],
            ),
            patch.object(runtime.subprocess, "run", return_value=gpu),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertFalse(
                runtime.native_runtime_status(self.model_path.parent, "cuda:0")["ready"]
            )
            self.assertTrue(
                runtime.native_runtime_status(self.model_path.parent, "cuda:1")["ready"]
            )
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                self.assertTrue(
                    runtime.native_runtime_status(self.model_path.parent, "cuda")["ready"]
                )
