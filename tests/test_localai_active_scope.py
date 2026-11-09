"""验证 LocalAI 管理面和能力清单覆盖当前 audio.cpp 服务范围。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "inference_gateway/src"), str(ROOT / "llm_local_runtime/src")]

from inference_gateway.config import GatewayConfig  # noqa: E402

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
                patch.dict("os.environ", {"LLM_LOCALAI_MANAGED": "false"}),
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

    def test_worker_bridge_sources_are_not_part_of_the_repository_runtime(self) -> None:
        bridge_sources = list((ROOT / "localai_worker_bridge/src").rglob("*.py"))
        self.assertEqual(bridge_sources, [])
        self.assertFalse((ROOT / "localai_worker_bridge/pyproject.toml").exists())
        self.assertFalse((ROOT / "localai/manifests/worker-bridge.json").exists())
        self.assertFalse((ROOT / "localai/manifests/tiger-dnr-worker-candidate.yaml").exists())


if __name__ == "__main__":
    unittest.main()
