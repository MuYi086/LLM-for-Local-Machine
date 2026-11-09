"""验证启动脚本实际只调用仍维护的 HTTP 服务。"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StartServicesTests(unittest.TestCase):
    def test_start_invokes_control_plane_voicedesign_and_eight_audio_cpp_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            trace_path = root / "commands.log"
            for command in ("uv", "python3"):
                executable = bin_dir / command
                executable.write_text(
                    f"#!/usr/bin/env bash\nprintf '%s %s\\n' '{command}' \"$*\" >> \"$TRACE\"\n"
                    'if [[ "$*" == *main.py* ]]; then sleep 1; exit 0; fi\n',
                    encoding="utf-8",
                )
                executable.chmod(0o755)

            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "TRACE": str(trace_path),
                "STORAGE_DIR": str(root / "storage"),
                "RUNTIME_CACHE_DIR": str(root / "storage/.cache/runtime"),
                "HOST": "127.0.0.1",
                "PORT": "18300",
                "QWEN_VOICEDESIGN_HOST": "127.0.0.1",
                "QWEN_VOICEDESIGN_PORT": "18301",
                "LLM_LOCALAI_ENABLED": "true",
                "LLM_LOCALAI_MANAGED": "true",
                "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER": "localai",
                "LLM_LOCAL_AUDIO_CPP_PROVIDER": "localai",
            }
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            uv_calls = [call for call in calls if call.startswith("uv ")]
            self.assertEqual(len(uv_calls), 12)
            self.assertTrue(any(f"--project {ROOT / 'main'}" in call for call in uv_calls))
            self.assertTrue(
                any(
                    f"--project {ROOT / 'model-audio/design_qwen3-tts-1.7b-voicedesign'}" in call
                    for call in uv_calls
                )
            )
            self.assertTrue(any("check_ports.py" in call for call in uv_calls))
            self.assertEqual(sum("main/main.py" in call for call in uv_calls), 1)
            self.assertEqual(
                sum("design_qwen3-tts-1.7b-voicedesign/main.py" in call for call in uv_calls),
                1,
            )
            audio_cpp_calls = [
                call
                for call in uv_calls
                if f"--project {ROOT / 'model-audio/audio_cpp_runtime'}" in call
            ]
            self.assertEqual(len(audio_cpp_calls), 8)
            for service_directory in (
                "tts_confucius4-tts",
                "tts_indextts2.5",
                "gen_midashenglm",
                "asr_qwen3-1.7b",
                "tts_qwen3-1.7b-base",
                "vc_seed-vc",
                "gen_stable-audio-3-small-sfx",
                "tts_voxcpm2",
            ):
                self.assertEqual(
                    sum(f"{service_directory}/main.py" in call for call in uv_calls),
                    1,
                )
            port_check = next(call for call in uv_calls if "check_ports.py" in call)
            for port in (
                "18300",
                "8301",
                "8331",
                "8312",
                "8341",
                "8321",
                "8311",
                "8342",
                "8343",
                "8313",
            ):
                self.assertIn(port, port_check)
            source = (ROOT / "start.sh").read_text(encoding="utf-8")
            self.assertIn('AUDIO_CPP_RUNTIME_SRC="$AUDIO_CPP_RUNTIME_PROJECT_DIR/src"', source)
            self.assertIn('export PYTHONPATH="$AUDIO_CPP_RUNTIME_SRC:', source)
            self.assertIn(
                'export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"',
                source,
            )
            self.assertIn('export GPU_LOCK_WAIT_TIMEOUT="${GPU_LOCK_WAIT_TIMEOUT:-900}"', source)
            self.assertFalse(
                any(
                    retired in call
                    for call in calls
                    for retired in (
                        "analyse_moss",
                        "soundEffect-v2.0",
                        "clone_qwen3-tts",
                        "indexTTS-2.5",
                        "separate_tiger-dnr",
                        "clone_confucius",
                        "transcribe_qwen3-asr",
                        "separate_msst-mega53",
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
