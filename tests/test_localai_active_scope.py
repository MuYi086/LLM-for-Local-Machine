"""验证 LocalAI 管理面和能力清单覆盖当前 audio.cpp 服务范围。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "inference_gateway/src"), str(ROOT / "llm_local_runtime/src")]

from inference_gateway.config import GatewayConfig  # noqa: E402
from llama_cpp_runtime import runtime as llama_runtime  # noqa: E402

from scripts.localai import recover_quarantine  # noqa: E402
from scripts.localai.validate_capabilities import validate_capabilities_manifest  # noqa: E402


class LocalAIActiveScopeTests(unittest.TestCase):
    def test_capabilities_manifest_lists_only_active_http_projects(self) -> None:
        manifest = json.loads((ROOT / "localai/manifests/capabilities.json").read_text())
        audio_cpp_projects = {
            "audio_cpp_confucius4_tts",
            "audio_cpp_index_tts25",
            "audio_cpp_midashenglm_gen",
            "audio_cpp_qwen3_asr",
            "audio_cpp_qwen3_tts_base",
            "audio_cpp_seed_vc",
            "audio_cpp_stable_audio_sfx",
            "audio_cpp_voxcpm2",
        }
        self.assertEqual(
            {service["project"] for service in manifest["services"]},
            {"main", "qwen3_voiceDesign", *audio_cpp_projects},
        )
        self.assertEqual(manifest["service_count_with_uv_projects"], 3)
        self.assertEqual(
            {entry["project"] for entry in manifest["audio_cpp_family_screen"]["services"]},
            {"main", "qwen3_voiceDesign", *audio_cpp_projects},
        )
        self.assertFalse((ROOT / "localai/manifests/qwen3-tts-base-candidate.yaml").exists())

        version_manifest = json.loads((ROOT / "localai/manifests/versions.json").read_text())
        self.assertEqual(
            version_manifest["active_model"]["name"],
            "llm-local.qwen3-tts.1p7b-voicedesign",
        )
        self.assertEqual(version_manifest["audio_cpp_model_candidate_count"], 9)
        self.assertNotIn("qwen3_tts_candidates", version_manifest)

    def test_capability_validator_resolves_shared_audio_cpp_router(self) -> None:
        summary = validate_capabilities_manifest()

        self.assertEqual(summary.service_count, 10)
        self.assertEqual(summary.uv_project_count, 3)
        self.assertEqual(summary.route_count, 23)
        self.assertEqual(summary.family_candidate_count, 9)

    def test_qa_dependencies_and_quality_gate_exclude_retired_integrations(self) -> None:
        qa_project = (ROOT / "qa/pyproject.toml").read_text()
        quality_gate = (ROOT / "scripts/quality_gate.sh").read_text()
        for retired_dependency in ("localai-worker-bridge", "grpcio", "protobuf"):
            self.assertNotIn(retired_dependency, qa_project)
        for retired_path in (
            "separate_tiger-dnr",
            "analyse_moss-music",
            "soundEffect-v2.0",
            "localai_worker_bridge",
        ):
            self.assertNotIn(retired_path, quality_gate)

    def test_manager_cli_only_exposes_the_voicedesign_candidate(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/localai/manager.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("register-audio-cpp-candidates", completed.stdout)
        self.assertIn("verify-audio-cpp-candidates", completed.stdout)
        for retired_command in (
            "register-qwen3-tts-candidate",
            "verify-qwen3-tts-candidate",
            "register-worker-bridge-candidates",
            "verify-worker-bridge-candidates",
        ):
            self.assertNotIn(retired_command, completed.stdout)

    def test_quarantine_recovery_targets_the_voicedesign_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps({"schema_version": 1, "model": "llm-local.qwen3-tts.1p7b-voicedesign"}),
                encoding="utf-8",
            )
            config = GatewayConfig(
                provider="localai",
                localai_enabled=True,
                base_url="http://127.0.0.1:8400",
                model="llm-local.qwen3-tts.1p7b-voicedesign",
                api_key=None,
                connect_timeout=5,
                request_timeout=30,
                cleanup_timeout=30,
                gpu_lock_file=str(lock_path),
                gpu_lock_wait_timeout=900,
                cuda_release_delay=0,
            )
            with (
                patch.object(
                    recover_quarantine, "load_gateway_config", return_value=config
                ) as load,
                patch.object(recover_quarantine, "LocalAIClient") as client_factory,
                patch.dict(
                    "os.environ",
                    {
                        "GPU_LOCK_FILE": str(lock_path),
                        "LLM_LOCALAI_MANAGED": "false",
                    },
                ),
            ):
                client_factory.return_value.shutdown_model.return_value = None

                result = recover_quarantine.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            load.call_args.kwargs["service_provider_env"],
            "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER",
        )
        self.assertEqual(
            load.call_args.kwargs["default_model"],
            "llm-local.qwen3-tts.1p7b-voicedesign",
        )
        self.assertEqual(client_factory.call_args.args[0].model, config.model)
        self.assertFalse(marker_path.exists())

    def test_quarantine_recovery_stops_llama_group_on_loopback(self) -> None:
        group_states = iter([True, False])
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "llama.cpp",
                        "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                        "host": "127.0.0.1",
                        "port": 18504,
                        "pid": 999991,
                        "pgid": 999991,
                        "pid_start_time": "12345",
                        "executable": "/opt/llama.cpp/llama-server",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {
                        "GPU_LOCK_FILE": str(lock_path),
                        "LLM_LOCALAI_CLEANUP_TIMEOUT": "1",
                    },
                ),
                patch.object(recover_quarantine, "process_start_time", return_value="12345"),
                patch.object(
                    recover_quarantine,
                    "_process_command_line",
                    return_value=["/opt/llama.cpp/llama-server", "--port", "18504"],
                ),
                patch.object(
                    recover_quarantine,
                    "_process_group_exists",
                    side_effect=lambda _pgid: next(group_states, False),
                ),
                patch.object(
                    recover_quarantine, "_loopback_port_open", return_value=False
                ) as port_open,
                patch.object(recover_quarantine.os, "killpg") as killpg,
                patch.object(recover_quarantine.time, "sleep"),
            ):
                result = recover_quarantine.main()

            self.assertEqual(result, 0)
            self.assertEqual(killpg.call_args.args, (999991, recover_quarantine.signal.SIGTERM))
            self.assertTrue(port_open.called)
            self.assertEqual(port_open.call_args.args, ("127.0.0.1", 18504))
            self.assertFalse(marker_path.exists())

    def test_quarantine_recovery_stops_every_llama_backend_target(self) -> None:
        targets = [
            {
                "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                "host": "127.0.0.1",
                "port": 18505,
                "pid": 999991,
                "pgid": 999991,
                "pid_start_time": "12345",
                "executable": "/opt/llama.cpp/llama-server",
            },
            {
                "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                "host": "127.0.0.1",
                "port": 18506,
                "pid": 999992,
                "pgid": 999992,
                "pid_start_time": "67890",
                "executable": "/opt/llama.cpp/llama-server",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "provider": "llama.cpp",
                        **targets[0],
                        "backends": targets,
                    }
                ),
                encoding="utf-8",
            )
            group_states = {target["pgid"]: True for target in targets}

            def group_exists(pgid: int) -> bool:
                was_running = group_states[pgid]
                group_states[pgid] = False
                return was_running

            def command_line(pid: int) -> list[str]:
                target = next(target for target in targets if target["pid"] == pid)
                return [
                    "/opt/llama.cpp/llama-server",
                    "--port",
                    str(target["port"]),
                ]

            with (
                patch.dict(
                    "os.environ",
                    {
                        "GPU_LOCK_FILE": str(lock_path),
                        "LLM_LOCALAI_CLEANUP_TIMEOUT": "1",
                    },
                ),
                patch.object(
                    recover_quarantine,
                    "process_start_time",
                    side_effect={
                        999991: "12345",
                        999992: "67890",
                    }.get,
                ),
                patch.object(recover_quarantine, "_process_command_line", side_effect=command_line),
                patch.object(recover_quarantine, "_process_group_exists", side_effect=group_exists),
                patch.object(
                    recover_quarantine, "_loopback_port_open", return_value=False
                ) as port_open,
                patch.object(recover_quarantine.os, "killpg") as killpg,
                patch.object(recover_quarantine.time, "sleep"),
            ):
                result = recover_quarantine.main()

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args for call in killpg.call_args_list],
                [
                    (999991, recover_quarantine.signal.SIGTERM),
                    (999992, recover_quarantine.signal.SIGTERM),
                ],
            )
            self.assertEqual(
                {call.args for call in port_open.call_args_list},
                {("127.0.0.1", 18505), ("127.0.0.1", 18506)},
            )
            self.assertFalse(marker_path.exists())

    def test_multi_backend_shutdown_quarantines_all_targets_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            targets = [
                (
                    "Ornith-1.5-9B-MTP-GGUF-Q8",
                    "127.0.0.1",
                    18505,
                    999991,
                    999991,
                    "/opt/llama.cpp/llama-server",
                ),
                (
                    "Ornith-1.5-9B-MTP-GGUF-Q8",
                    "127.0.0.1",
                    18506,
                    999992,
                    999992,
                    "/opt/llama.cpp/llama-server",
                ),
            ]
            with (
                patch.object(llama_runtime, "gpu_runtime_lock", return_value=nullcontext()),
                patch.object(llama_runtime, "process_start_time", return_value=None),
                patch.object(llama_runtime, "_process_group_exists", return_value=False),
                patch.object(
                    llama_runtime,
                    "loopback_port_open",
                    side_effect=lambda _host, port: port == 18506,
                ),
                patch.object(llama_runtime.time, "sleep"),
                self.assertRaises(TimeoutError),
            ):
                llama_runtime.stop_backends(
                    lock_path=lock_path,
                    backends=targets,
                    lock_timeout=1,
                    cleanup_timeout=0.01,
                    cuda_release_delay=0,
                )

            marker = json.loads(Path(f"{lock_path}.quarantine").read_text(encoding="utf-8"))
            self.assertEqual(marker["schema_version"], 2)
            self.assertEqual(
                [backend["port"] for backend in marker["backends"]],
                [18505, 18506],
            )

    def test_multi_backend_shutdown_merges_existing_llama_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "llama.cpp",
                        "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                        "host": "127.0.0.1",
                        "port": 18505,
                        "pid": None,
                        "pgid": None,
                        "pid_start_time": None,
                        "executable": "/opt/llama.cpp/llama-server",
                    }
                ),
                encoding="utf-8",
            )
            checked_ports: list[int] = []

            def port_closed(_host: str, port: int) -> bool:
                checked_ports.append(port)
                return False

            with (
                patch.object(llama_runtime, "gpu_runtime_lock", return_value=nullcontext()),
                patch.object(llama_runtime, "process_start_time", return_value=None),
                patch.object(llama_runtime, "_process_group_exists", return_value=False),
                patch.object(llama_runtime, "loopback_port_open", side_effect=port_closed),
                patch.object(llama_runtime.time, "sleep"),
            ):
                llama_runtime.stop_backends(
                    lock_path=lock_path,
                    backends=[
                        (
                            "Ornith-1.5-9B-MTP-GGUF-Q8",
                            "127.0.0.1",
                            18506,
                            999992,
                            999992,
                            "/opt/llama.cpp/llama-server",
                        )
                    ],
                    lock_timeout=1,
                    cleanup_timeout=1,
                    cuda_release_delay=0,
                )

            self.assertFalse(marker_path.exists())
            self.assertEqual(set(checked_ports), {18505, 18506})

    def test_multi_backend_shutdown_does_not_signal_a_reused_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            start_times = iter(("original-start", "original-start", "reused-start"))
            with (
                patch.object(llama_runtime, "gpu_runtime_lock", return_value=nullcontext()),
                patch.object(
                    llama_runtime,
                    "process_start_time",
                    side_effect=lambda _pid: next(start_times),
                ),
                patch.object(llama_runtime, "_command_matches_backend", return_value=True),
                patch.object(llama_runtime, "_process_group_exists", return_value=True),
                patch.object(llama_runtime, "loopback_port_open", return_value=True),
                patch.object(llama_runtime.os, "killpg") as killpg,
                patch.object(llama_runtime.time, "sleep"),
                self.assertRaises(TimeoutError),
            ):
                llama_runtime.stop_backends(
                    lock_path=lock_path,
                    backends=[
                        (
                            "Ornith-1.5-9B-MTP-GGUF-Q8",
                            "127.0.0.1",
                            18505,
                            999991,
                            999991,
                            "/opt/llama.cpp/llama-server",
                        )
                    ],
                    lock_timeout=1,
                    cleanup_timeout=0.01,
                    cuda_release_delay=0,
                )

            killpg.assert_not_called()
            self.assertTrue(Path(f"{lock_path}.quarantine").exists())

    def test_recovery_stops_extra_llama_backends_before_other_provider(self) -> None:
        targets = [
            {
                "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                "host": "127.0.0.1",
                "port": 18505,
                "pid": 999991,
                "pgid": 999991,
                "pid_start_time": "12345",
                "executable": "/opt/llama.cpp/llama-server",
            },
            {
                "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                "host": "127.0.0.1",
                "port": 18506,
                "pid": 999992,
                "pgid": 999992,
                "pid_start_time": "67890",
                "executable": "/opt/llama.cpp/llama-server",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "stable-diffusion.cpp",
                        "model": "fixture",
                        "llama_cpp_backends": targets,
                    }
                ),
                encoding="utf-8",
            )
            group_states = {target["pgid"]: True for target in targets}

            def group_exists(pgid: int) -> bool:
                was_running = group_states[pgid]
                group_states[pgid] = False
                return was_running

            def command_line(pid: int) -> list[str]:
                target = next(target for target in targets if target["pid"] == pid)
                return [
                    "/opt/llama.cpp/llama-server",
                    "--port",
                    str(target["port"]),
                ]

            def recover_sdcpp(marker: Path, _payload: dict[str, object], _timeout: float) -> None:
                marker.unlink()

            with (
                patch.dict(
                    "os.environ",
                    {
                        "GPU_LOCK_FILE": str(lock_path),
                        "LLM_LOCALAI_CLEANUP_TIMEOUT": "1",
                    },
                ),
                patch.object(
                    recover_quarantine,
                    "process_start_time",
                    side_effect={
                        999991: "12345",
                        999992: "67890",
                    }.get,
                ),
                patch.object(recover_quarantine, "_process_command_line", side_effect=command_line),
                patch.object(recover_quarantine, "_process_group_exists", side_effect=group_exists),
                patch.object(
                    recover_quarantine, "_loopback_port_open", return_value=False
                ) as probe,
                patch.object(recover_quarantine.os, "killpg") as killpg,
                patch.object(
                    recover_quarantine, "recover_sdcpp", side_effect=recover_sdcpp
                ) as sdcpp,
                patch.object(recover_quarantine.time, "sleep"),
            ):
                result = recover_quarantine.main()

            self.assertEqual(result, 0)
            self.assertEqual(killpg.call_count, 2)
            self.assertEqual(
                {call.args for call in probe.call_args_list},
                {("127.0.0.1", 18505), ("127.0.0.1", 18506)},
            )
            sdcpp.assert_called_once()
            self.assertFalse(marker_path.exists())

    def test_ornith_recovery_keeps_marker_while_loopback_api_is_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu-runtime.lock"
            marker_path = Path(f"{lock_path}.quarantine")
            marker_path.write_text(
                json.dumps(
                    {
                        "provider": "llama.cpp",
                        "model": "Ornith-1.5-9B-MTP-GGUF-Q8",
                        "host": "127.0.0.1",
                        "port": 18504,
                        "pid": None,
                        "pgid": None,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {
                        "GPU_LOCK_FILE": str(lock_path),
                        "LLM_LOCALAI_CLEANUP_TIMEOUT": "0.1",
                    },
                ),
                patch.object(recover_quarantine, "_loopback_port_open", return_value=True),
                patch.object(recover_quarantine.time, "sleep"),
            ):
                result = recover_quarantine.main()

            self.assertEqual(result, 1)
            self.assertTrue(marker_path.exists())

    def test_loopback_probe_detects_bound_secondary_loopback_port_without_connecting(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.2", 0))
        listener.listen()
        port = listener.getsockname()[1]
        try:
            self.assertTrue(recover_quarantine._loopback_port_open("127.0.0.2", port))
        finally:
            listener.close()

        self.assertFalse(recover_quarantine._loopback_port_open("127.0.0.2", port))

    def test_worker_bridge_sources_are_not_part_of_the_repository_runtime(self) -> None:
        bridge_sources = list((ROOT / "localai_worker_bridge/src").rglob("*.py"))
        self.assertEqual(bridge_sources, [])
        self.assertFalse((ROOT / "localai_worker_bridge/pyproject.toml").exists())
        self.assertFalse((ROOT / "localai/manifests/worker-bridge.json").exists())
        self.assertFalse((ROOT / "localai/manifests/tiger-dnr-worker-candidate.yaml").exists())


if __name__ == "__main__":
    unittest.main()
