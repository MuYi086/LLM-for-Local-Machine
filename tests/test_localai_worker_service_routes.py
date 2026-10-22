"""验证五个保留服务的本机 LocalAI worker bridge 路由。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def load_service(project: str) -> Iterator[object]:
    """在临时存储配置下加载单个服务模块，不加载模型环境。"""
    project_dir = ROOT / "model-audio" / project
    module_name = f"localai_test_{project.lower().replace('-', '_')}"
    previous_runtime = sys.modules.pop("runtime", None)
    sys.path.insert(0, str(project_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, project_dir / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载服务入口：{project}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(project_dir))
        sys.modules.pop("runtime", None)
        if previous_runtime is not None:
            sys.modules["runtime"] = previous_runtime


class LocalAIWorkerServiceRouteTests(unittest.TestCase):
    """确认 LocalAI bridge 复用各保留服务的 worker payload 与输出合同。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage_dir = Path(self.temporary.name) / "storage"
        self.environment = patch.dict(
            "os.environ",
            {
                "STORAGE_DIR": str(self.storage_dir),
                "GPU_LOCK_FILE": str(self.storage_dir / ".cache/runtime/gpu.lock"),
                "LLM_LOCALAI_ENABLED": "false",
                "LOCAL_FILES_ONLY": "1",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def select_localai(module: object, model: str) -> None:
        module.GATEWAY_CONFIG = replace(
            module.GATEWAY_CONFIG,
            provider="localai",
            localai_enabled=True,
            model=model,
        )

    def test_sheetsage2_routes_zip_through_localai_and_persists_it(self) -> None:
        with load_service("SheetSage2") as module:
            self.select_localai(module, "llm-local.sheetsage2.worker.v1")
            payload = {"audio_path": "/tmp/checked.wav", "dtype": "bf16"}
            artifact = Path(self.temporary.name) / "worker.zip"
            saved = Path(self.temporary.name) / "saved.zip"
            with (
                patch.object(
                    module,
                    "execute_localai_worker_file_reference",
                    return_value=(artifact, 1.25),
                ) as execute,
                patch.object(module, "persist_archive_file", return_value=saved) as persist,
                patch.object(
                    module,
                    "gpu_runtime_lock",
                    side_effect=AssertionError("LocalAI 分支不能二次获取 GPU 锁"),
                ),
            ):
                self.assertEqual(module.execute_transcription_payload(payload), saved)

            execute.assert_called_once_with(
                config=module.GATEWAY_CONFIG,
                operation="run",
                payload={"worker_input": payload},
                output_dir=module.WORKER_TMP_DIR,
                artifact_dir=module.LOCALAI_ARTIFACT_DIR,
                suffix=".zip",
                media_type="application/zip",
            )
            persist.assert_called_once_with(artifact, module.OUTPUT_DIR)

    def test_mert_selects_npz_output_for_frame_features(self) -> None:
        with load_service("MERT-v2-FullSong") as module:
            self.select_localai(module, "llm-local.mert-v2-fullsong.worker.v1")
            payload = {"audio_path": "/tmp/checked.wav", "output": "frames"}
            artifact = Path(self.temporary.name) / "worker.npz"
            with (
                patch.object(
                    module,
                    "execute_localai_worker_file_reference",
                    return_value=(artifact, 2.0),
                ) as execute,
                patch.object(
                    module,
                    "shared_gpu_runtime_lock",
                    side_effect=AssertionError("LocalAI 分支不能二次获取 GPU 锁"),
                ),
            ):
                self.assertEqual(module.execute_worker(payload), artifact)

            self.assertEqual(execute.call_args.kwargs["suffix"], ".npz")
            self.assertEqual(execute.call_args.kwargs["media_type"], "application/x-npz")
            self.assertEqual(execute.call_args.kwargs["payload"], {"worker_input": payload})

    def test_yue2_vae_selects_wav_output_for_decode(self) -> None:
        with load_service("YuE2-Vae") as module:
            self.select_localai(module, "llm-local.yue2-vae.worker.v1")
            payload = {"operation": "decode", "latent_path": "/tmp/checked.npy"}
            artifact = Path(self.temporary.name) / "worker.wav"
            with (
                patch.object(
                    module,
                    "execute_localai_worker_file_reference",
                    return_value=(artifact, 1.5),
                ) as execute,
                patch.object(
                    module,
                    "shared_gpu_runtime_lock",
                    side_effect=AssertionError("LocalAI 分支不能二次获取 GPU 锁"),
                ),
            ):
                self.assertEqual(module.execute_worker(payload), artifact)

            self.assertEqual(execute.call_args.kwargs["suffix"], ".wav")
            self.assertEqual(execute.call_args.kwargs["media_type"], "audio/wav")
            self.assertEqual(execute.call_args.kwargs["payload"], {"worker_input": payload})

    def test_yue2_3b_routes_complete_request_through_worker_bridge(self) -> None:
        with load_service("YuE2-3B") as module:
            self.select_localai(module, "llm-local.yue2-3b.worker.v1")
            request = module.YuE2MusicRequest(lyrics="你好", style="民谣")
            expected_payload = {"lyrics": "你好", "style": "民谣", "seed": request.seed}
            audio = b"valid mocked wav bytes"
            saved = Path(self.temporary.name) / "saved.wav"
            with (
                patch.object(module, "module_available", return_value=True),
                patch.object(
                    module,
                    "model_status",
                    return_value={"main": {"complete": True, "missing": []}},
                ),
                patch.object(module, "cuda_status", return_value={"available": True}),
                patch.object(module.manager, "build_worker_payload", return_value=expected_payload),
                patch.object(module.manager, "run_localai_worker", return_value=audio) as execute,
                patch.object(module, "persist_audio_bytes", return_value=saved) as persist,
                patch.object(
                    module,
                    "gpu_runtime_lock",
                    side_effect=AssertionError("LocalAI 分支不能二次获取 GPU 锁"),
                ),
            ):
                response = module.generate_music(request)

            execute.assert_called_once_with(expected_payload)
            persist.assert_called_once_with(audio, module.OUTPUT_DIR)
            self.assertEqual(response.body, audio)
            self.assertEqual(response.media_type, "audio/wav")

    def test_tiger_dnr_unpacks_only_three_named_stems_from_file_reference(self) -> None:
        with load_service("TIGER-DnR") as module:
            self.select_localai(module, "llm-local.tiger-dnr.worker.v1")
            archive = Path(self.temporary.name) / "tiger.zip"
            with zipfile.ZipFile(archive, "w") as worker_archive:
                for stem in ("dialog", "effects", "music"):
                    worker_archive.writestr(f"{stem}.wav", f"RIFF-{stem}".encode())
            with patch.object(
                module,
                "execute_localai_worker_file_reference",
                return_value=(archive, 1.0),
            ) as execute:
                outputs = module.manager.run_localai_worker({"audio_path": "/tmp/input.wav"})

            self.assertEqual(set(outputs), {"dialog", "effects", "music"})
            self.assertEqual(outputs["effects"].read_bytes(), b"RIFF-effects")
            self.assertFalse(archive.exists())
            execute.assert_called_once()
            module.manager.cleanup_worker_outputs(outputs)

    def test_tiger_dnr_rejects_unexpected_zip_entries(self) -> None:
        with load_service("TIGER-DnR") as module:
            self.select_localai(module, "llm-local.tiger-dnr.worker.v1")
            archive = Path(self.temporary.name) / "bad-tiger.zip"
            with zipfile.ZipFile(archive, "w") as worker_archive:
                worker_archive.writestr("dialog.wav", b"RIFF-dialog")
                worker_archive.writestr("effects.wav", b"RIFF-effects")
                worker_archive.writestr("music.wav", b"RIFF-music")
                worker_archive.writestr("../other.wav", b"bad")
            with patch.object(
                module,
                "execute_localai_worker_file_reference",
                return_value=(archive, 1.0),
            ):
                with self.assertRaisesRegex(module.LocalAIRequestError, "只能包含三个"):
                    module.manager.run_localai_worker({"audio_path": "/tmp/input.wav"})
            self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
