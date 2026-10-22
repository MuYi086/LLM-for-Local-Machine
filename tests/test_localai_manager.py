"""验证 LocalAI 裸机版本校验、loopback 启动和进程组回收。"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.localai.manager import LocalAIManager, LocalAIManagerError

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


class LocalAIManagerTests(unittest.TestCase):
    """验证宿主机 LocalAI 运行管理，不依赖容器、权重或 CUDA。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name) / "localai"
        self.binary_path = self.data_dir / "bin/local-ai"
        self.backend_run = self.data_dir / "backends/audio-cpp/run.sh"
        self.binary_path.parent.mkdir(parents=True)
        self.backend_run.parent.mkdir(parents=True)
        self.binary_path.write_text(
            "#!/bin/sh\nprintf '%s\\n' 'local-ai version v4.11.0'\n", encoding="utf-8"
        )
        self.binary_path.chmod(0o755)
        self.backend_run.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.backend_run.chmod(0o755)

    def manager(self, **overrides: str) -> LocalAIManager:
        environment = {
            "LLM_LOCALAI_DATA_DIR": str(self.data_dir),
            "LLM_LOCALAI_BINARY": str(self.binary_path),
            "LLM_LOCALAI_BACKENDS_DIR": str(self.data_dir / "backends"),
            "LLM_LOCALAI_ENABLED": "true",
            "LLM_LOCALAI_BASE_URL": "http://127.0.0.1:18400",
            **overrides,
        }
        return LocalAIManager(environ=environment, readiness_timeout=0.01)

    def install_stamp(self) -> None:
        self.manager().verify_install()

    def qwen3_tts_model(self) -> Path:
        model_dir = Path(self.temporary.name) / "Qwen3-TTS-12Hz-1.7B-Base"
        for relative_path in (
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ):
            path = model_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"prepared model fixture")
        return model_dir

    def qwen3_voicedesign_model(self) -> Path:
        model_dir = Path(self.temporary.name) / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        for relative_path in (
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ):
            path = model_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"prepared model fixture")
        return model_dir

    def test_verify_install_checks_binary_and_native_backend_files(self) -> None:
        manager = self.manager()
        manager.verify_install()
        stamp = json.loads(manager.install_stamp_path.read_text(encoding="utf-8"))

        self.assertEqual(stamp["localai_version"], "v4.11.0")
        self.assertEqual(stamp["expected_localai_source_commit"], manager.localai_commit)
        self.assertEqual(stamp["expected_audio_cpp_source_commit"], manager.audio_cpp_commit)
        self.assertEqual(stamp["localai_binary_sha256"], manager._sha256_file(self.binary_path))
        self.assertEqual(
            stamp["audio_cpp_backend_tree_sha256"],
            manager._sha256_tree(self.backend_run.parent),
        )
        manager._ensure_install_stamp()

    def test_verify_install_rejects_wrong_version_and_missing_backend(self) -> None:
        self.binary_path.write_text("#!/bin/sh\necho local-ai version v4.10.0\n", encoding="utf-8")
        self.binary_path.chmod(0o755)
        with self.assertRaisesRegex(LocalAIManagerError, "版本"):
            self.manager().verify_install()

        self.binary_path.write_text("#!/bin/sh\necho local-ai version v4.11.0\n", encoding="utf-8")
        self.binary_path.chmod(0o755)
        self.backend_run.unlink()
        with self.assertRaisesRegex(LocalAIManagerError, "audio-cpp"):
            self.manager().verify_install()

    def test_install_stamp_detects_binary_or_backend_change(self) -> None:
        manager = self.manager()
        manager.verify_install()
        self.backend_run.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        with self.assertRaisesRegex(LocalAIManagerError, "不一致"):
            manager._ensure_install_stamp()

    def test_backend_run_script_cannot_escape_backend_directory(self) -> None:
        outside = Path(self.temporary.name) / "outside.sh"
        outside.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        outside.chmod(0o755)
        self.backend_run.unlink()
        self.backend_run.symlink_to(outside)

        with self.assertRaisesRegex(LocalAIManagerError, "逃逸"):
            self.manager()._backend_run_script()

    def test_backend_tree_fingerprint_rejects_external_symlink(self) -> None:
        outside = Path(self.temporary.name) / "external.bin"
        outside.write_bytes(b"outside")
        (self.backend_run.parent / "external.bin").symlink_to(outside)

        with self.assertRaisesRegex(LocalAIManagerError, "软链接越界"):
            self.manager()._sha256_tree(self.backend_run.parent)

    def test_worker_bridge_candidate_registration_needs_no_model_weights(self) -> None:
        manager = self.manager()
        manager.register_worker_bridge_candidates()
        manager.verify_worker_bridge_candidates()

        for model_name in (
            "llm-local.sheetsage2.worker.v1",
            "llm-local.mert-v2-fullsong.worker.v1",
            "llm-local.yue2-vae.worker.v1",
            "llm-local.yue2-3b.worker.v1",
            "llm-local.tiger-dnr.worker.v1",
        ):
            installed = self.data_dir / "models" / f"{model_name}.yaml"
            self.assertTrue(installed.is_file())
            self.assertIn("backend: llm-local-worker", installed.read_text(encoding="utf-8"))

    def test_worker_bridge_candidate_verification_rejects_modified_yaml(self) -> None:
        manager = self.manager()
        manager.register_worker_bridge_candidates()
        candidate = self.data_dir / "models/llm-local.sheetsage2.worker.v1.yaml"
        candidate.write_text(candidate.read_text(encoding="utf-8") + "# changed\n")

        with self.assertRaisesRegex(LocalAIManagerError, "YAML 不匹配"):
            manager.verify_worker_bridge_candidates()

    def test_qwen3_tts_registration_requires_manual_model_and_renders_model_path(self) -> None:
        missing_model_dir = Path(self.temporary.name) / "not-downloaded"
        with self.assertRaisesRegex(LocalAIManagerError, "模型文件缺失"):
            self.manager(QWEN3_TTS_MODEL_DIR=str(missing_model_dir)).register_qwen3_tts_candidate()

        model_dir = self.qwen3_tts_model()
        manager = self.manager(QWEN3_TTS_MODEL_DIR=str(model_dir))
        manager.register_qwen3_tts_candidate()
        manager.verify_qwen3_tts_candidate()

        installed_config = self.data_dir / "models/llm-local.qwen3-tts.1p7b-base.yaml"
        rendered = installed_config.read_text(encoding="utf-8")
        self.assertIn(f"  model: {json.dumps(str(model_dir.resolve()))}\n", rendered)
        self.assertIn("family:qwen3_tts", rendered)
        self.assertIn("task:tts", rendered)

    def test_qwen3_tts_registration_rejects_modified_config(self) -> None:
        model_dir = self.qwen3_tts_model()
        manager = self.manager(QWEN3_TTS_MODEL_DIR=str(model_dir))
        manager.register_qwen3_tts_candidate()
        config_path = self.data_dir / "models/llm-local.qwen3-tts.1p7b-base.yaml"
        config_path.write_text(config_path.read_text(encoding="utf-8") + "# changed\n")

        with self.assertRaisesRegex(LocalAIManagerError, "不一致"):
            manager.verify_qwen3_tts_candidate()

    def test_qwen3_voicedesign_registration_requires_manual_model_and_pins_vdes(self) -> None:
        missing_model_dir = Path(self.temporary.name) / "voice-design-not-downloaded"
        with self.assertRaisesRegex(LocalAIManagerError, "模型文件缺失"):
            self.manager(
                QWEN_VOICEDESIGN_MODEL_DIR=str(missing_model_dir)
            ).register_qwen3_voicedesign_candidate()

        model_dir = self.qwen3_voicedesign_model()
        manager = self.manager(QWEN_VOICEDESIGN_MODEL_DIR=str(model_dir))
        manager.register_qwen3_voicedesign_candidate()
        manager.verify_qwen3_voicedesign_candidate()

        installed_config = self.data_dir / "models/llm-local.qwen3-tts.1p7b-voicedesign.yaml"
        rendered = installed_config.read_text(encoding="utf-8")
        self.assertIn(f"  model: {json.dumps(str(model_dir.resolve()))}\n", rendered)
        self.assertIn("family:qwen3_tts", rendered)
        self.assertIn("task:vdes", rendered)
        self.assertNotIn("task:tts", rendered)

    def test_qwen3_voicedesign_registration_rejects_modified_config(self) -> None:
        model_dir = self.qwen3_voicedesign_model()
        manager = self.manager(QWEN_VOICEDESIGN_MODEL_DIR=str(model_dir))
        manager.register_qwen3_voicedesign_candidate()
        config_path = self.data_dir / "models/llm-local.qwen3-tts.1p7b-voicedesign.yaml"
        config_path.write_text(config_path.read_text(encoding="utf-8") + "# changed\n")

        with self.assertRaisesRegex(LocalAIManagerError, "不一致"):
            manager.verify_qwen3_voicedesign_candidate()

    def test_start_is_loopback_only_single_backend_and_waits_for_ready(self) -> None:
        manager = self.manager(LLM_LOCALAI_API_KEY="test-key")
        manager.verify_install()
        with (
            patch("scripts.localai.manager.os.posix_spawn", return_value=4242) as spawn,
            patch.object(manager, "_ensure_detached_process_group"),
            patch.object(
                manager, "_localai_version_output", return_value="local-ai version v4.11.0"
            ),
            patch.object(manager, "_process_start_time", return_value="17"),
            patch.object(manager, "_ensure_port_available"),
            patch.object(manager, "_wait_until_ready") as wait_ready,
        ):
            manager.start()

        setsid_path = spawn.call_args.args[0]
        command = spawn.call_args.args[1]
        environment = spawn.call_args.args[2]
        self.assertEqual(command, [setsid_path, "--", str(self.binary_path), "run"])
        self.assertEqual(spawn.call_args.kwargs["file_actions"][0][1:4], (0, "/dev/null", 0))
        self.assertEqual(environment["LOCALAI_ADDRESS"], "127.0.0.1:18400")
        self.assertEqual(environment["LOCALAI_MAX_ACTIVE_BACKENDS"], "1")
        self.assertEqual(environment["LOCALAI_AUTOLOAD_GALLERIES"], "false")
        self.assertEqual(environment["LOCALAI_AUTOLOAD_BACKEND_GALLERIES"], "false")
        self.assertEqual(environment["LOCALAI_DISABLE_GALLERY_ENDPOINT"], "true")
        self.assertEqual(environment["LOCALAI_DISABLE_WEBUI"], "true")
        self.assertEqual(environment["LOCALAI_API_KEY"], "test-key")
        self.assertEqual(wait_ready.call_args.args[0], "http://127.0.0.1:18400")

    def test_start_registers_managed_worker_bridge_in_external_backend_map(self) -> None:
        manager = self.manager(
            LLM_LOCALAI_WORKER_BRIDGE_ENABLED="true",
            LLM_LOCALAI_BRIDGE_ADDRESS="127.0.0.1:18401",
            LOCALAI_EXTERNAL_GRPC_BACKENDS="other:127.0.0.1:50060,llm-local-worker:127.0.0.1:9",
        )
        manager.verify_install()
        with (
            patch("scripts.localai.manager.os.posix_spawn", return_value=4242) as spawn,
            patch.object(manager, "_ensure_detached_process_group"),
            patch.object(
                manager, "_localai_version_output", return_value="local-ai version v4.11.0"
            ),
            patch.object(manager, "_process_start_time", return_value="17"),
            patch.object(manager, "_ensure_port_available") as ensure_port,
            patch.object(manager, "_wait_until_ready"),
        ):
            manager.start()

        command = spawn.call_args.args[1]
        environment = spawn.call_args.args[2]
        self.assertEqual(
            command,
            [
                spawn.call_args.args[0],
                "--",
                sys.executable,
                str(REPOSITORY_DIR / "scripts/localai/host.py"),
                str(self.binary_path),
            ],
        )
        self.assertEqual(
            environment["LOCALAI_EXTERNAL_GRPC_BACKENDS"],
            "other:127.0.0.1:50060,llm-local-worker:127.0.0.1:18401",
        )
        self.assertEqual(environment["LLM_LOCALAI_WORKER_BRIDGE_ENABLED"], "true")
        self.assertEqual(environment["LLM_LOCALAI_BRIDGE_ADDRESS"], "127.0.0.1:18401")
        self.assertEqual([call.args[0] for call in ensure_port.call_args_list], [18400, 18401])

    def test_worker_bridge_rejects_non_loopback_address(self) -> None:
        manager = self.manager(
            LLM_LOCALAI_WORKER_BRIDGE_ENABLED="true",
            LLM_LOCALAI_BRIDGE_ADDRESS="0.0.0.0:50052",
        )
        manager.verify_install()
        with self.assertRaisesRegex(LocalAIManagerError, "127.0.0.1"):
            manager.start()

    def test_manager_reaps_orphaned_worker_bridge_process_group(self) -> None:
        worker_script = REPOSITORY_DIR / "model-audio/SheetSage2/worker.py"
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                str(worker_script),
                "--input-json",
                "unused.json",
            ],
            env={**os.environ, "LLM_LOCALAI_REQUEST_ID": "test-request"},
            start_new_session=True,
        )
        try:
            self.manager()._reap_worker_bridge_processes()
            self.assertIsNotNone(worker.poll())
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=2)

    def test_fake_host_process_runs_ready_and_is_reaped_from_its_process_group(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.binary_path.write_text(
            "#!/usr/bin/env python3\n"
            "import http.server\n"
            "import os\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('local-ai version v4.11.0')\n"
            "    raise SystemExit(0)\n"
            "host, port = os.environ['LOCALAI_ADDRESS'].rsplit(':', 1)\n"
            "class Handler(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200 if self.path == '/readyz' else 404)\n"
            "        self.end_headers()\n"
            "    def log_message(self, *args):\n"
            "        pass\n"
            "http.server.HTTPServer((host, int(port)), Handler).serve_forever()\n",
            encoding="utf-8",
        )
        self.binary_path.chmod(0o755)
        manager = self.manager(LLM_LOCALAI_BASE_URL=f"http://127.0.0.1:{port}")
        manager.readiness_timeout = 3
        manager.verify_install()

        manager.start()
        self.assertTrue(manager.state_path.is_file())
        try:
            manager.stop()
            manager.verify_managed_localai_stopped()
        finally:
            if manager.state_path.exists():
                manager.stop()

    def test_owner_watcher_reaps_fake_host_process_after_owner_is_killed(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.binary_path.write_text(
            "#!/usr/bin/env python3\n"
            "import http.server\n"
            "import os\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('local-ai version v4.11.0')\n"
            "    raise SystemExit(0)\n"
            "host, port = os.environ['LOCALAI_ADDRESS'].rsplit(':', 1)\n"
            "class Handler(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200 if self.path == '/readyz' else 404)\n"
            "        self.end_headers()\n"
            "    def log_message(self, *args):\n"
            "        pass\n"
            "http.server.HTTPServer((host, int(port)), Handler).serve_forever()\n",
            encoding="utf-8",
        )
        self.binary_path.chmod(0o755)
        owner = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        manager = self.manager(
            LLM_LOCALAI_BASE_URL=f"http://127.0.0.1:{port}",
            LLM_LOCALAI_OWNER_PID=str(owner.pid),
        )
        manager.readiness_timeout = 3
        manager.verify_install()

        try:
            manager.start()
            os.kill(owner.pid, signal.SIGKILL)
            owner.wait(timeout=3)

            deadline = time.monotonic() + 5
            while manager.state_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)

            self.assertFalse(manager.state_path.exists())
            manager.verify_managed_localai_stopped()
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=3)
            if manager.state_path.exists():
                manager.stop()

    def test_unhealthy_start_terminates_group_and_removes_state_only_after_reap(self) -> None:
        manager = self.manager()
        manager.verify_install()
        with (
            patch("scripts.localai.manager.os.posix_spawn", return_value=4242),
            patch.object(manager, "_ensure_detached_process_group"),
            patch.object(
                manager, "_localai_version_output", return_value="local-ai version v4.11.0"
            ),
            patch.object(manager, "_process_start_time", return_value="17"),
            patch.object(manager, "_ensure_port_available"),
            patch.object(manager, "_terminate_process_group") as terminate,
            patch.object(
                manager, "_wait_until_ready", side_effect=LocalAIManagerError("ready timeout")
            ),
        ):
            with self.assertRaisesRegex(LocalAIManagerError, "ready timeout"):
                manager.start()

        terminate.assert_called_once()
        self.assertFalse(manager.state_path.exists())

    def test_start_requires_explicit_enable_and_verified_native_install(self) -> None:
        disabled = self.manager(LLM_LOCALAI_ENABLED="false")
        with self.assertRaises(LocalAIManagerError):
            disabled.start()
        with self.assertRaisesRegex(LocalAIManagerError, "verify-install"):
            self.manager().start()

    def test_start_rejects_non_loopback_or_path_prefixed_base_url(self) -> None:
        manager = self.manager()
        manager.verify_install()
        for base_url in ("http://192.168.1.20:18400", "http://127.0.0.1:18400/api"):
            with self.subTest(base_url=base_url):
                with self.assertRaises(LocalAIManagerError):
                    self.manager(LLM_LOCALAI_BASE_URL=base_url)._base_url_and_port()

    def test_owner_watcher_is_started_before_readiness_without_copying_api_key(self) -> None:
        manager = self.manager(
            LLM_LOCALAI_API_KEY="private-key",
            LOCALAI_API_KEY="legacy-private-key",
            LLM_LOCALAI_OWNER_PID="42",
        )
        state = {"pid": 4242, "process_start_time": "18"}
        events: list[str] = []

        def spawn(*args: object, **kwargs: object) -> int:
            events.append("watcher")
            return 5151

        with (
            patch("scripts.localai.manager.os.posix_spawn", side_effect=spawn) as start_watcher,
            patch.object(manager, "_ensure_detached_process_group"),
        ):
            manager._start_owner_watcher(42, "17", state)

        call = start_watcher.call_args
        self.assertEqual(call.args[1][4:9], ["watch", "42", "17", "4242", "18"])
        self.assertNotIn("LLM_LOCALAI_API_KEY", call.args[2])
        self.assertNotIn("LOCALAI_API_KEY", call.args[2])

    def test_owner_watcher_stops_group_after_owner_disappears(self) -> None:
        manager = self.manager()
        manager._ensure_directories()
        state = {
            "schema_version": 1,
            "boot_id": manager._boot_id(),
            "pid": 4242,
            "pgid": 4242,
            "process_start_time": "18",
            "owner_pid": 42,
            "owner_start_time": "17",
            "run_id": "a" * 32,
            "base_url": "http://127.0.0.1:18400",
        }
        manager._atomic_write_json(manager.state_path, state)
        with (
            patch.object(manager, "_process_start_time", side_effect=["18", None]),
            patch.object(manager, "_terminate_process_group") as terminate,
        ):
            manager.watch_owner(42, "17", 4242, "18")

        terminate.assert_called_once_with(state)
        self.assertFalse(manager.state_path.exists())

    def test_stop_confirms_process_group_and_port_before_removing_state(self) -> None:
        manager = self.manager()
        manager._ensure_directories()
        state = {
            "schema_version": 1,
            "boot_id": manager._boot_id(),
            "pid": 4242,
            "pgid": 4242,
            "process_start_time": "18",
            "run_id": "a" * 32,
            "base_url": "http://127.0.0.1:18400",
        }
        manager._atomic_write_json(manager.state_path, state)
        with (
            patch.object(manager, "_process_start_time", return_value="18"),
            patch.object(manager, "_process_group_alive", side_effect=[True, False, False]),
            patch.object(manager, "_wait_for_process_group"),
            patch.object(manager, "_ensure_port_closed"),
            patch("scripts.localai.manager.os.killpg") as killpg,
        ):
            manager.stop()

        killpg.assert_called_once()
        self.assertFalse(manager.state_path.exists())

    def test_stop_clears_stale_process_state_after_host_reboot(self) -> None:
        manager = self.manager()
        manager._ensure_directories()
        state = {
            "schema_version": 1,
            "boot_id": "f" * 36,
            "pid": 4242,
            "pgid": 4242,
            "process_start_time": "18",
            "run_id": "a" * 32,
            "base_url": "http://127.0.0.1:18400",
        }
        manager._atomic_write_json(manager.state_path, state)
        with (
            patch.object(manager, "_boot_id", return_value="e" * 36),
            patch.object(manager, "_managed_process_ids", return_value=[]),
            patch.object(manager, "_ensure_port_closed"),
            patch("scripts.localai.manager.os.killpg") as killpg,
        ):
            manager.stop()

        killpg.assert_not_called()
        self.assertFalse(manager.state_path.exists())

    def test_stop_refuses_pid_reuse(self) -> None:
        manager = self.manager()
        state = {
            "pid": 4242,
            "pgid": 4242,
            "process_start_time": "18",
            "run_id": "a" * 32,
        }
        with (
            patch.object(manager, "_process_start_time", return_value="99"),
            patch("scripts.localai.manager.os.killpg") as killpg,
        ):
            with self.assertRaisesRegex(LocalAIManagerError, "复用"):
                manager._terminate_process_group(state)
        killpg.assert_not_called()

    def test_stop_escalates_to_sigkill_only_after_term_timeout(self) -> None:
        manager = self.manager()
        state = {
            "pid": 4242,
            "pgid": 4242,
            "process_start_time": "18",
            "run_id": "a" * 32,
        }
        with (
            patch.object(manager, "_process_start_time", return_value="18"),
            patch.object(manager, "_process_group_alive", side_effect=[True, True, False]),
            patch.object(manager, "_wait_for_process_group") as wait_group,
            patch.object(manager, "_ensure_port_closed"),
            patch("scripts.localai.manager.os.killpg") as killpg,
        ):
            manager._terminate_process_group(state)

        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )
        self.assertEqual(wait_group.call_count, 2)

    def test_managed_process_recovery_requires_state_absence_and_closed_port(self) -> None:
        manager = self.manager()
        with patch(
            "scripts.localai.manager.socket.create_connection",
            side_effect=ConnectionRefusedError,
        ):
            manager.verify_managed_localai_stopped()

        with patch("scripts.localai.manager.socket.create_connection") as connect:
            connect.return_value.close = Mock()
            with self.assertRaisesRegex(LocalAIManagerError, "仍可连接"):
                manager.verify_managed_localai_stopped()

    def test_stopped_verification_detects_orphan_backend_processes(self) -> None:
        manager = self.manager()
        with patch.object(manager, "_managed_process_ids", return_value=[4242]):
            with self.assertRaisesRegex(LocalAIManagerError, "仍发现受管"):
                manager.verify_managed_localai_stopped()


if __name__ == "__main__":
    unittest.main()
