"""Ditto 服务的无模型 HTTP 和 worker 生命周期回归。"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from fastapi.testclient import TestClient

from unitale_runtime import GpuLockTimeoutError, GpuRuntimeMetrics

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "ditto"
ROUTE = "/v1/ditto/talkingHead"
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"video-and-audio"


def load_module(name: str, path: Path):
    """加载服务模块，避免不同服务的 main 名称互相覆盖。"""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DittoServiceTests(unittest.TestCase):
    """隔离模型、存储和环境，替换所有 GPU 与外部进程边界。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ditto-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "STORAGE_DIR": str(self.root / "storage"),
            "PROMPTS_DIR": str(self.root / "storage/clone"),
            "TIMBRE_STORAGE_DIR": str(self.root / "storage/timbre"),
            "RUNTIME_CACHE_DIR": str(self.root / "cache"),
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            "DITTO_MODEL_DIR": str(self.root / "model"),
            "DITTO_CODE_PATH": str(self.root / "code"),
            "DITTO_OUTPUT_DIR": str(self.root / "output"),
            "DITTO_IMAGE_DIR": str(self.root / "images"),
            "DITTO_WORKER_TMP_DIR": str(self.root / "worker"),
            "DITTO_REQUEST_TIMEOUT": "5",
            "DITTO_PORT": "8392",
            "CUDA_RELEASE_DELAY": "0",
            "LOCAL_FILES_ONLY": "1",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]),
        ):
            self.main = load_module("ditto_main_tests", SERVICE_DIR / "main.py")
            self.worker = load_module("ditto_worker_tests", SERVICE_DIR / "worker.py")
        self.client = TestClient(self.main.app)
        self.runtime = sys.modules["ditto_runtime"]
        self.data = {"image_path": "portrait.png", "audio_path": "speech.wav"}
        self.request = self.main.TalkingHeadRequest(**self.data)
        for path in self.main.required_files().values():
            if path == Path(self.main.WORKER_SCRIPT):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.main.image_path("portrait.png").write_bytes(b"png")
        self.main.reference_store.clone_path("speech.wav").write_bytes(b"wav")

    @contextmanager
    def fake_gpu_lock(self, *args, **kwargs):
        """代替共享 GPU 锁，并暴露请求指标。"""
        yield GpuRuntimeMetrics(label="test")

    def test_routes_health_and_startup(self) -> None:
        with patch.object(self.main, "cuda_status", return_value={"available": False}):
            response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.main.API_PORT, 8392)
        self.assertTrue(response.json()["available"]["model"])
        self.assertEqual(response.json()["runtime"]["backend"], "onnxruntime-cuda")
        self.assertIn("storage", response.json())
        source = (REPOSITORY_DIR / "start.sh").read_text()
        self.assertIn('DITTO_PORT="${DITTO_PORT:-8392}"', source)
        self.assertIn("'/v1/ditto/talkingHead'", source)
        self.assertIn('"$ditto_pid"', source[source.index("cleanup() {") :])
        self.assertIn('"$ditto_pid"', source[source.rindex("wait -n") :])
        routes = {route.path for route in self.main.app.routes}
        self.assertIn(ROUTE, routes)
        self.assertNotIn("/v1/confucius4TTS/generate", routes)

    def test_image_and_audio_uploads_use_logical_paths(self) -> None:
        for route, field, logical, media, content in (
            ("/v1/upload_image", "image", "C:/portrait.png", "image/png", b"png-data"),
            ("/v1/upload_audio", "audio", "C:/speech.wav", "audio/wav", b"wav-data"),
        ):
            with self.subTest(route=route):
                response = self.client.post(
                    route,
                    data={"full_path": logical},
                    files={field: (Path(logical).name, io.BytesIO(content), media)},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["size_bytes"], len(content))
                self.assertEqual(len(response.json()["sha256"]), 64)
        self.assertEqual(self.main.image_path("C:/portrait.png").read_bytes(), b"png-data")
        self.assertEqual(
            self.main.reference_store.prompt_audio_path("C:/speech.wav").read_bytes(), b"wav-data"
        )
        self.assertTrue(
            self.client.get("/v1/check/image", params={"file_name": "C:/portrait.png"}).json()[
                "exists"
            ]
        )
        self.assertTrue(
            self.client.get("/v1/check/audio", params={"file_name": "C:/speech.wav"}).json()[
                "exists"
            ]
        )
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").glob("*.part")), [])

    def test_upload_rejects_empty_oversize_and_wrong_media(self) -> None:
        for name, content, media, status in (
            ("bad.exe", b"bad", "image/png", 422),
            ("bad.png", b"bad", "text/plain", 422),
            ("empty.png", b"", "image/png", 422),
            ("big.png", b"12345", "image/png", 413),
        ):
            with self.subTest(name=name), patch.dict(os.environ, {"UPLOAD_MAX_BYTES": "4"}):
                response = self.client.post(
                    "/v1/upload_image",
                    data={"full_path": name},
                    files={"image": (name, content, media)},
                )
                self.assertEqual(response.status_code, status)
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").glob("*.part")), [])

    def test_invalid_fields_do_not_enter_gpu_queue(self) -> None:
        for extra in (
            {"model_type": "unknown"},
            {"seed": -1},
            {"image_path": ""},
            {"steps": 8},
            {"sampling_timesteps": 0},
            {"sampling_timesteps": 101},
            {"max_size": 4097},
        ):
            with self.subTest(extra=extra), patch.object(self.main, "gpu_runtime_lock") as lock:
                response = self.client.post(ROUTE, json={**self.data, **extra})
                self.assertEqual(response.status_code, 422)
                lock.assert_not_called()

    def test_missing_inputs_and_dependencies_fail_before_gpu_lock(self) -> None:
        self.main.image_path("portrait.png").unlink()
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            self.assertEqual(self.client.post(ROUTE, json=self.data).status_code, 404)
            lock.assert_not_called()
        self.main.image_path("portrait.png").touch()
        self.main.required_files()["hubert_cfg"].unlink()
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            self.assertEqual(self.client.post(ROUTE, json=self.data).status_code, 503)
            lock.assert_not_called()

    def test_bad_media_fails_before_gpu_lock(self) -> None:
        with (
            patch.object(self.main, "validate_inputs", side_effect=ValueError("无法解码图片")),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            self.assertEqual(self.client.post(ROUTE, json=self.data).status_code, 422)
            lock.assert_not_called()

    def test_success_returns_persistent_mp4_and_cleans_job_files(self) -> None:
        captured = {}

        def start_worker(command, **kwargs):
            captured.update(
                json.loads(Path(command[command.index("--input-json") + 1]).read_text())
            )
            Path(command[command.index("--output-video") + 1]).write_bytes(MP4_BYTES)
            process = MagicMock()
            process.communicate.return_value = ("done", "")
            process.returncode = 0
            process.pid = None
            return process

        with (
            patch.object(self.main, "validate_inputs"),
            patch.object(self.main, "validate_video"),
            patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
            patch.object(self.main.subprocess, "Popen", side_effect=start_worker) as popen,
            patch.object(self.main, "terminate_process_group") as terminate,
        ):
            response = self.client.post(ROUTE, json={**self.data, "seed": 123})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, MP4_BYTES)
        self.assertEqual(captured["seed"], 123)
        self.assertEqual(captured["sampling_timesteps"], 50)
        self.assertEqual(popen.call_args.args[0][0], sys.executable)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["WORLD_SIZE"], "1")
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        terminate.assert_called_once()
        self.assertEqual(list(Path(self.main.WORKER_TMP_DIR).iterdir()), [])
        saved = list(self.main.OUTPUT_DIR.glob("*.mp4"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), MP4_BYTES)

    def test_worker_timeout_failure_and_invalid_output_clean_everything(self) -> None:
        for failure in ("timeout", "failure", "missing", "invalid"):
            process = MagicMock()
            process.pid = None
            process.returncode = 1 if failure == "failure" else 0
            process.communicate.side_effect = (
                subprocess.TimeoutExpired("worker", 5) if failure == "timeout" else None
            )
            process.communicate.return_value = ("", "failure message")

            def start_worker(command, failure=failure, process=process, **kwargs):
                if failure == "invalid":
                    Path(command[command.index("--output-video") + 1]).write_bytes(
                        b"invalid MP4 data"
                    )
                return process

            with (
                self.subTest(failure=failure),
                patch.object(self.main, "validate_inputs"),
                patch.object(self.main, "gpu_runtime_lock", self.fake_gpu_lock),
                patch.object(self.main.subprocess, "Popen", side_effect=start_worker),
                patch.object(self.main, "terminate_process_group") as terminate,
                patch.object(self.main, "wait_after_cuda_release") as release,
            ):
                response = self.client.post(ROUTE, json=self.data)
                self.assertEqual(response.status_code, 504 if failure == "timeout" else 500)
                terminate.assert_called()
                release.assert_called_once()
                self.assertEqual(list(Path(self.main.WORKER_TMP_DIR).iterdir()), [])
                self.assertEqual(list(self.main.OUTPUT_DIR.iterdir()), [])

    def test_http_and_shared_runtime_do_not_import_model_packages(self) -> None:
        for filename in ("main.py", "ditto_runtime.py"):
            tree = ast.parse((SERVICE_DIR / filename).read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(
                {"torch", "onnx", "onnxruntime", "pickle", "stream_pipeline_offline"}.isdisjoint(
                    imported
                )
            )

    def test_image_commit_failure_removes_staged_upload(self) -> None:
        with patch.object(
            self.main, "commit_plain_upload", side_effect=ValueError("commit failure")
        ):
            response = self.client.post(
                "/v1/upload_image",
                data={"full_path": "new.png"},
                files={"image": ("new.png", b"png", "image/png")},
            )
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.main.image_path("new.png").exists())
        self.assertEqual(list((self.main.RUNTIME_CACHE_DIR / "uploads").iterdir()), [])

    def test_gpu_queue_timeout_returns_503_without_worker(self) -> None:
        with (
            patch.object(self.main, "validate_inputs"),
            patch.object(
                self.main, "gpu_runtime_lock", side_effect=GpuLockTimeoutError("ditto", 1)
            ),
            patch.object(self.main.manager, "run_worker") as worker,
        ):
            self.assertEqual(self.client.post(ROUTE, json=self.data).status_code, 503)
            worker.assert_not_called()

    def test_cpu_media_preflight_checks_streams_and_duration(self) -> None:
        image_info = {"streams": [{"codec_name": "png", "width": 512, "height": 512}]}
        for duration in ("0", "-1", "nan", "inf", "301", "unknown", "1"):
            audio_info = {"streams": [{"codec_type": "audio"}], "format": {"duration": duration}}
            with (
                self.subTest(duration=duration),
                patch.object(self.runtime, "probe_media", side_effect=[image_info, audio_info]),
            ):
                if duration == "1":
                    self.runtime.validate_inputs(Path("image"), Path("audio"), "ffprobe", 300)
                else:
                    with self.assertRaises(ValueError):
                        self.runtime.validate_inputs(Path("image"), Path("audio"), "ffprobe", 300)
        for image in (
            {"streams": []},
            {"streams": [{"codec_name": "h264", "width": 512, "height": 512}]},
        ):
            with (
                patch.object(self.runtime, "probe_media", return_value=image),
                self.assertRaises(ValueError),
            ):
                self.runtime.validate_inputs(Path("image"), Path("audio"), "ffprobe", 300)
        with (
            patch.object(self.runtime, "probe_media", side_effect=[image_info, {"streams": []}]),
            self.assertRaises(ValueError),
        ):
            self.runtime.validate_inputs(Path("image"), Path("audio"), "ffprobe", 300)

    def test_mp4_validation_requires_audio_and_video(self) -> None:
        output = self.root / "output.mp4"
        output.write_bytes(MP4_BYTES)
        for streams in ([{"codec_type": "video"}], [{"codec_type": "audio"}], []):
            with (
                patch.object(self.runtime, "probe_media", return_value={"streams": streams}),
                self.assertRaises(RuntimeError),
            ):
                self.runtime.validate_video(output, "ffprobe")
        with patch.object(
            self.runtime,
            "probe_media",
            return_value={"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
        ):
            self.runtime.validate_video(output, "ffprobe")
        output.write_bytes(b"this is not an mp4 container")
        with self.assertRaises(RuntimeError):
            self.runtime.validate_video(output, "ffprobe")

    def test_atomic_video_commit_rolls_back_on_replace_failure(self) -> None:
        output = self.root / "generated.mp4"
        output.write_bytes(MP4_BYTES)
        with (
            patch.object(self.runtime.os, "replace", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            self.runtime.persist_video(output, self.main.OUTPUT_DIR)
        self.assertEqual(list(self.main.OUTPUT_DIR.iterdir()), [])

    def test_process_group_cleanup_kills_children_after_parent_exits(self) -> None:
        import signal

        process = MagicMock(pid=1234, returncode=0)
        with patch.object(self.runtime.os, "killpg") as killpg:
            self.runtime.terminate_process_group(process, "test")
        self.assertEqual(
            killpg.call_args_list, [call(1234, signal.SIGTERM), call(1234, signal.SIGKILL)]
        )

    def test_process_group_cleanup_escalates_after_timeout(self) -> None:
        import signal

        process = MagicMock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), None]
        with patch.object(self.runtime.os, "killpg") as killpg:
            self.runtime.terminate_process_group(process, "test")
        self.assertIn(call(1234, signal.SIGKILL), killpg.call_args_list)
        self.assertEqual(process.wait.call_count, 2)

    def test_worker_config_rewrites_all_local_paths_without_modifying_source(self) -> None:
        config = {
            "base_cfg": {name: {"model_path": "old.engine"} for name in self.main.ONNX_COMPONENTS},
            "audio2motion_cfg": {"w2f_type": "hubert", "model_path": "old.engine"},
            "default_kwargs": {"online_mode": True, "crop_scale": 2.3},
        }
        config["base_cfg"]["wavlm_cfg"] = {"model_path": "missing.engine"}
        source = Path(self.main.CONFIG_PATH)
        original = pickle.dumps(config)
        source.write_bytes(original)
        with patch.object(self.main, "validate_inputs"):
            payload = self.main.manager.build_worker_payload(self.request)
        job = self.root / "config-job"
        job.mkdir()
        with patch.object(self.worker, "prepare_warp_model") as warp:
            generated = self.worker.build_runtime_config(payload, job)
        updated = pickle.loads(generated.read_bytes())
        self.assertEqual(source.read_bytes(), original)
        self.assertFalse(updated["default_kwargs"]["online_mode"])
        self.assertEqual(updated["default_kwargs"]["crop_scale"], 2.3)
        self.assertNotIn("wavlm_cfg", updated["base_cfg"])
        for name, filename in self.main.ONNX_COMPONENTS.items():
            expected = (
                job / "warp.onnx"
                if name == "warp_network_cfg"
                else Path(self.main.DATA_ROOT) / filename
            )
            self.assertEqual(updated["base_cfg"][name]["model_path"], str(expected))
            self.assertEqual(set(updated["base_cfg"][name]), {"model_path", "device"})
        warp.assert_called_once()
        self.assertEqual(
            updated["audio2motion_cfg"]["model_path"],
            str(Path(self.main.DATA_ROOT) / "lmdm_v0.4_hubert.onnx"),
        )
        config["audio2motion_cfg"]["w2f_type"] = "wavlm"
        source.write_bytes(pickle.dumps(config))
        with self.assertRaises(ValueError):
            self.worker.build_runtime_config(payload, job)

    def test_warp_conversion_uses_official_converter_and_rejects_plugin_graph(self) -> None:
        model = SimpleNamespace(graph=SimpleNamespace(node=[SimpleNamespace(op_type="GridSample")]))
        onnx = MagicMock()
        onnx.load.return_value = model
        with patch.dict(sys.modules, {"onnx": onnx}):
            self.worker.prepare_warp_model(Path("source.onnx"), Path("temporary.onnx"))
        onnx.version_converter.convert_version.assert_called_once_with(model, 20)
        converted = onnx.version_converter.convert_version.return_value
        onnx.checker.check_model.assert_called_once_with(converted)
        onnx.save.assert_called_once_with(converted, "temporary.onnx")
        model.graph.node = [SimpleNamespace(op_type="GridSample3D")]
        with patch.dict(sys.modules, {"onnx": onnx}), self.assertRaises(ValueError):
            self.worker.prepare_warp_model(Path("source.onnx"), Path("temporary.onnx"))

    def test_sdk_uses_offline_audio_and_preserves_short_tail(self) -> None:
        sdk = MagicMock()
        payload = {"sampling_timesteps": 12, "max_size": 1024}
        audio = [0.0] * 1281
        self.worker.run_sdk(sdk, audio, "portrait.png", self.root / "result.mp4", payload)
        sdk.setup.assert_called_once_with(
            "portrait.png",
            str(self.root / "result.mp4"),
            online_mode=False,
            sampling_timesteps=12,
            max_size=1024,
        )
        sdk.setup_Nd.assert_called_once_with(N_d=3)
        sdk.wav2feat.wav2feat.assert_called_once_with(audio)
        sdk.audio2motion_queue.put.assert_called_once_with(sdk.wav2feat.wav2feat.return_value)
        sdk.close.assert_called_once()

    def test_sdk_failure_stops_threads_and_preserves_original_error(self) -> None:
        sdk = MagicMock()
        sdk.thread_list = [MagicMock()]
        sdk.wav2feat.wav2feat.side_effect = RuntimeError("feature failure")
        sdk.writer.close.side_effect = RuntimeError("cleanup failure")
        with self.assertRaisesRegex(RuntimeError, "feature failure"):
            self.worker.run_sdk(
                sdk,
                [0.0],
                "portrait.png",
                self.root / "result.mp4",
                {"sampling_timesteps": 50, "max_size": 1920},
            )
        sdk.stop_event.set.assert_called_once()
        sdk.thread_list[0].join.assert_called_once_with(timeout=2)
        sdk.close.assert_not_called()

    def test_cuda_loader_rejects_silent_cpu_fallback(self) -> None:
        for providers, accepted in (
            (["CUDAExecutionProvider", "CPUExecutionProvider"], True),
            (["CPUExecutionProvider"], False),
        ):
            loader = MagicMock()
            model = MagicMock()
            model.get_providers.return_value = providers
            loader.load_model.return_value = (model, "onnx")
            with patch.object(self.worker.importlib, "import_module", return_value=loader):
                self.worker.configure_model_loader()
            if accepted:
                self.assertEqual(loader.load_model("test.onnx"), (model, "onnx"))
            else:
                with self.assertRaisesRegex(RuntimeError, "未启用 CUDA"):
                    loader.load_model("test.onnx")

    def test_worker_checks_ffmpeg_failures_and_removes_intermediates(self) -> None:
        librosa = MagicMock()
        librosa.load.return_value = ([0.0] * 1281, 16000)
        ort = MagicMock()
        ort.get_available_providers.return_value = ["CUDAExecutionProvider"]
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        modules = {
            "librosa": librosa,
            "numpy": MagicMock(),
            "onnxruntime": ort,
            "torch": torch,
            "pyximport": MagicMock(),
        }
        payload = {
            "seed": 123,
            "code_path": "code",
            "data_root": "models",
            "ffmpeg_bin": "/usr/bin/ffmpeg",
            "reference_audio_path": "speech.wav",
            "reference_image_path": "portrait.png",
            "sampling_timesteps": 50,
            "max_size": 1920,
        }
        output = self.root / "result.mp4"
        for failure in (False, True):
            with (
                self.subTest(failure=failure),
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, {}),
                patch.object(sys, "path", list(sys.path)),
                patch.object(self.worker, "configure_model_loader"),
                patch.object(
                    self.worker, "build_runtime_config", return_value=self.root / "config.pkl"
                ),
                patch.object(self.worker.importlib, "import_module") as pipeline,
                patch.object(self.worker.subprocess, "run") as run,
            ):
                for name in ("driver.wav", "result.mp4.tmp.mp4", "warp.onnx", "config.pkl"):
                    (self.root / name).touch()
                if failure:
                    run.side_effect = [None, subprocess.CalledProcessError(1, "ffmpeg")]
                    with self.assertRaises(subprocess.CalledProcessError):
                        self.worker.generate_video(payload, output)
                else:
                    self.worker.generate_video(payload, output)
                self.assertEqual(run.call_count, 2)
                self.assertTrue(all(item.kwargs["check"] for item in run.call_args_list))
                self.assertTrue(all(not item.kwargs.get("shell") for item in run.call_args_list))
                self.assertIn("speech.wav", run.call_args_list[-1].args[0])
                pipeline.return_value.StreamSDK.return_value.setup_Nd.assert_called_once_with(N_d=3)
                self.assertTrue(
                    all(
                        not (self.root / name).exists()
                        for name in ("driver.wav", "result.mp4.tmp.mp4", "warp.onnx", "config.pkl")
                    )
                )


if __name__ == "__main__":
    unittest.main()
