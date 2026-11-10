"""验证启动脚本实际只调用仍维护的 HTTP 服务。"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(executable: Path) -> None:
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'python3 %s\\n\' "$*" >> "$TRACE"\n'
        'if [[ "$*" == *"-m ornith_1_5_35b_a3b.runtime"* ]]; then exec /usr/bin/python3 "$@"; fi\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)


def _write_fake_llama_server(executable: Path) -> None:
    executable.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "if '--list-devices' in sys.argv:\n"
        "    print('Available devices: CUDA0')\n"
        "    raise SystemExit(0)\n"
        "if '--help' in sys.argv:\n"
        "    print('--n-cpu-moe --sleep-idle-seconds --mmproj-offload --spec-type --spec-draft-model')\n"
        "    raise SystemExit(0)\n"
        "if 'Qwen3.6-35B-A3B-MTP' in sys.argv and os.getenv('FAKE_QWEN_FAIL') == 'true':\n"
        "    raise SystemExit(9)\n"
        "with open(os.environ['TRACE'], 'a', encoding='utf-8') as trace:\n"
        "    trace.write('llama-server ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "host = sys.argv[sys.argv.index('--host') + 1]\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        body = json.dumps({'is_sleeping': True, 'status': 'ok'}).encode()\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "HTTPServer((host, port), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def _preflight_environment(root: Path) -> tuple[dict[str, str], Path]:
    """建立只记录命令的启动环境，避免访问模型与 LocalAI。"""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    trace_path = root / "commands.log"
    for command in ("uv",):
        executable = bin_dir / command
        executable.write_text(
            f"#!/usr/bin/env bash\nprintf '%s %s\\n' '{command}' \"$*\" >> \"$TRACE\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    _write_fake_python(bin_dir / "python3")
    llama_server = bin_dir / "llama-server"
    _write_fake_llama_server(llama_server)
    model_file = root / "Ornith.gguf"
    mmproj_file = root / "mmproj.gguf"
    mtp_file = root / "mtp.gguf"
    qwen_model_file = root / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    moss_image_root = root / "hf-mirror/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF"
    moss_image_root.mkdir(parents=True)
    for name in (
        "MOSS-Image-SFT-Q8_0.gguf",
        "Qwen3.5-9B-Q6_K.gguf",
        "flux2-vae.safetensors",
        "tokenizer.json",
    ):
        (moss_image_root / name).write_bytes(b"MOSS-test-component")
    flux3_image_root = root / "hf-mirror/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF"
    flux3_image_root.mkdir(parents=True)
    for name in (
        "flux3-image-klein-Q8_0.gguf",
        "Qwen3VL-4B-Instruct-Q8_0.gguf",
        "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        "ae.safetensors",
    ):
        (flux3_image_root / name).write_bytes(b"FLUX3-test-component")
    sd_cpp_binary = bin_dir / "sd-cli"
    sd_cpp_binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sd_cpp_binary.chmod(0o755)
    model_file.write_bytes(b"GGUF-test-model")
    mmproj_file.write_bytes(b"GGUF-test-mmproj")
    mtp_file.write_bytes(b"GGUF-test-mtp")
    qwen_model_file.write_bytes(b"GGUF-test-qwen-model")
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRACE": str(trace_path),
        "STORAGE_DIR": str(root / "storage"),
        "RUNTIME_CACHE_DIR": str(root / "storage/.cache/runtime"),
        "HF_MIRROR_DIR": str(root / "hf-mirror"),
        "SDCPP_BIN": str(sd_cpp_binary),
        "SDCPP_BIN_SHA256": hashlib.sha256(sd_cpp_binary.read_bytes()).hexdigest(),
        "MOSS_IMAGE_HOST": "127.0.0.1",
        "MOSS_IMAGE_PORT": "18501",
        "FLUX3_IMAGE_HOST": "127.0.0.1",
        "FLUX3_IMAGE_PORT": "18502",
        "FLUX3_IMAGE_UPLOAD_DIR": str(root / "storage/.cache/runtime/sdcpp_uploads"),
        "FLUX3_IMAGE_MAX_UPLOAD_BYTES": "67108864",
        "HOST": "127.0.0.1",
        "PORT": "18300",
        "QWEN_VOICEDESIGN_HOST": "127.0.0.1",
        "QWEN_VOICEDESIGN_PORT": "18301",
        "ORNITH_HOST": "127.0.0.1",
        "ORNITH_PORT": "18401",
        "LLAMA_CPP_PORT": "18403",
        "QWEN36_PORT": "18404",
        "QWEN36_BACKEND_HOST": "127.0.0.2",
        "QWEN36_LLAMA_CPP_PORT": "18405",
        "QWEN36_MODEL_FILE": str(qwen_model_file),
        "QWEN36_STARTUP_TIMEOUT": "5",
        "QWEN36_SLEEP_TIMEOUT": "1",
        "QWEN36_SLEEP_POLL_INTERVAL": "0.01",
        "LLAMA_SERVER_BIN": str(llama_server),
        "ORNITH_MODEL_FILE": str(model_file),
        "ORNITH_MTP_FILE": str(mtp_file),
        "ORNITH_MMPROJ_FILE": str(mmproj_file),
        "ORNITH_STARTUP_TIMEOUT": "5",
        "ORNITH_SLEEP_TIMEOUT": "1",
        "ORNITH_SLEEP_POLL_INTERVAL": "0.01",
        "LLM_LOCALAI_ENABLED": "true",
        "LLM_LOCALAI_MANAGED": "true",
        "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER": "localai",
        "LLM_LOCAL_AUDIO_CPP_PROVIDER": "localai",
    }
    return environment, trace_path


class StartServicesTests(unittest.TestCase):
    def test_positive_timeouts_reject_zero_and_invalid_decimals_before_startup(self) -> None:
        for variable in (
            "ORNITH_SLEEP_TIMEOUT",
            "ORNITH_SLEEP_POLL_INTERVAL",
            "QWEN36_SLEEP_TIMEOUT",
            "QWEN36_SLEEP_POLL_INTERVAL",
            "GPU_LOCK_WAIT_TIMEOUT",
        ):
            for value in ("0", "00.000", "-1", "nan", "1.2.3"):
                with (
                    self.subTest(variable=variable, value=value),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    environment, trace_path = _preflight_environment(Path(temporary))
                    environment[variable] = value
                    completed = subprocess.run(
                        ["bash", str(ROOT / "start.sh")],
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=15,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(f"{variable} 必须是正数", completed.stderr)
                    self.assertNotIn("manager.py", trace_path.read_text())

    def test_default_text_settings_match_measured_profile(self) -> None:
        expected = {
            "ORNITH": {"threads": "6", "threads-batch": "8"},
            "QWEN36": {"threads": "8", "threads-batch": "8"},
        }
        common = {
            "n-cpu-moe": "23",
            "ctx-size": "32768",
            "batch-size": "512",
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "load-mode": "none",
            "spec-type": "draft-mtp",
            "spec-draft-n-max": "4",
            "spec-draft-type-k": "f16",
            "spec-draft-type-v": "f16",
            "parallel": "1",
            "sleep-idle-seconds": "1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            environment, trace_path = _preflight_environment(Path(temporary))
            for prefix in expected:
                for setting in (
                    "THREADS",
                    "THREADS_BATCH",
                    "N_CPU_MOE",
                    "CONTEXT_SIZE",
                    "BATCH_SIZE",
                    "CACHE_TYPE_K",
                    "CACHE_TYPE_V",
                    "LOAD_MODE",
                    "SPEC_TYPE",
                    "SPEC_DRAFT_N_MAX",
                    "SPEC_DRAFT_CACHE_TYPE_K",
                    "SPEC_DRAFT_CACHE_TYPE_V",
                    "SLEEP_IDLE_SECONDS",
                ):
                    environment.pop(f"{prefix}_{setting}", None)
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            starts = [
                line
                for line in trace_path.read_text().splitlines()
                if line.startswith("llama-server ")
            ]
            self.assertEqual(len(starts), 2)
            for command, (prefix, settings) in zip(starts, expected.items(), strict=True):
                arguments = command.split()
                for flag, value in (common | settings).items():
                    with self.subTest(model=prefix, flag=flag):
                        self.assertIn(f"--{flag}", arguments)
                        self.assertEqual(arguments[arguments.index(f"--{flag}") + 1], value)

    def test_start_invokes_managed_services_and_both_llama_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            trace_path = root / "commands.log"
            for command in ("uv",):
                executable = bin_dir / command
                executable.write_text(
                    f"#!/usr/bin/env bash\nprintf '%s %s\\n' '{command}' \"$*\" >> \"$TRACE\"\n"
                    'if [[ "$*" == *main.py* ]]; then sleep 1; exit 0; fi\n',
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            _write_fake_python(bin_dir / "python3")
            llama_server = bin_dir / "llama server"
            _write_fake_llama_server(llama_server)
            model_file = root / "Ornith.gguf"
            mmproj_file = root / "mmproj.gguf"
            mtp_file = root / "mtp.gguf"
            qwen_model_file = root / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
            moss_image_root = root / "hf-mirror/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF"
            moss_image_root.mkdir(parents=True)
            for name in (
                "MOSS-Image-SFT-Q8_0.gguf",
                "Qwen3.5-9B-Q6_K.gguf",
                "flux2-vae.safetensors",
                "tokenizer.json",
            ):
                (moss_image_root / name).write_bytes(b"MOSS-test-component")
            flux3_image_root = root / "hf-mirror/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF"
            flux3_image_root.mkdir(parents=True)
            for name in (
                "flux3-image-klein-Q8_0.gguf",
                "Qwen3VL-4B-Instruct-Q8_0.gguf",
                "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
                "ae.safetensors",
            ):
                (flux3_image_root / name).write_bytes(b"FLUX3-test-component")
            sd_cpp_binary = bin_dir / "sd-cli"
            sd_cpp_binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            sd_cpp_binary.chmod(0o755)
            model_file.write_bytes(b"GGUF-test-model")
            mmproj_file.write_bytes(b"GGUF-test-mmproj")
            mtp_file.write_bytes(b"GGUF-test-mtp")
            qwen_model_file.write_bytes(b"GGUF-test-qwen-model")

            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "TRACE": str(trace_path),
                "STORAGE_DIR": str(root / "storage"),
                "RUNTIME_CACHE_DIR": str(root / "storage/.cache/runtime"),
                "HF_MIRROR_DIR": str(root / "hf-mirror"),
                "SDCPP_BIN": str(sd_cpp_binary),
                "SDCPP_BIN_SHA256": hashlib.sha256(sd_cpp_binary.read_bytes()).hexdigest(),
                "MOSS_IMAGE_HOST": "127.0.0.1",
                "MOSS_IMAGE_PORT": "18501",
                "FLUX3_IMAGE_HOST": "127.0.0.1",
                "FLUX3_IMAGE_PORT": "18502",
                "FLUX3_IMAGE_UPLOAD_DIR": str(root / "storage/.cache/runtime/sdcpp_uploads"),
                "FLUX3_IMAGE_MAX_UPLOAD_BYTES": "67108864",
                "HOST": "127.0.0.1",
                "PORT": "18300",
                "QWEN_VOICEDESIGN_HOST": "127.0.0.1",
                "QWEN_VOICEDESIGN_PORT": "18301",
                "ORNITH_HOST": "127.0.0.1",
                "ORNITH_PORT": "18401",
                "LLAMA_CPP_PORT": "18403",
                "QWEN36_PORT": "18404",
                "QWEN36_BACKEND_HOST": "127.0.0.2",
                "QWEN36_LLAMA_CPP_PORT": "18405",
                "LLAMA_SERVER_BIN": str(llama_server),
                "ORNITH_MODEL_FILE": str(model_file),
                "ORNITH_MMPROJ_FILE": str(mmproj_file),
                "ORNITH_MTP_FILE": str(mtp_file),
                "QWEN36_MODEL_FILE": str(qwen_model_file),
                "QWEN36_THREADS_BATCH": "12",
                "ORNITH_N_CPU_MOE": "24",
                "QWEN36_N_CPU_MOE": "25",
                "QWEN36_SPEC_DRAFT_N_MAX": "4",
                "ORNITH_SPEC_DRAFT_N_MAX": "5",
                "ORNITH_SPEC_DRAFT_CACHE_TYPE_K": "q8_0",
                "ORNITH_SPEC_DRAFT_CACHE_TYPE_V": "q8_0",
                "QWEN36_SPEC_DRAFT_CACHE_TYPE_K": "q8_0",
                "QWEN36_SPEC_DRAFT_CACHE_TYPE_V": "q8_0",
                "ORNITH_STARTUP_TIMEOUT": "5",
                "ORNITH_SLEEP_TIMEOUT": "1",
                "ORNITH_SLEEP_POLL_INTERVAL": "0.01",
                "QWEN36_STARTUP_TIMEOUT": "5",
                "QWEN36_SLEEP_TIMEOUT": "1",
                "QWEN36_SLEEP_POLL_INTERVAL": "0.01",
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
            self.assertEqual(len(uv_calls), 18)
            self.assertTrue(any(f"--project {ROOT / 'main'}" in call for call in uv_calls))
            self.assertTrue(
                any(
                    f"--project {ROOT / 'model-audio/design_qwen3-tts-1.7b-voicedesign'}" in call
                    for call in uv_calls
                )
            )
            self.assertTrue(any("check_ports.py" in call for call in uv_calls))
            self.assertTrue(any("scripts/sdcpp/manager.py" in call for call in uv_calls))
            self.assertEqual(sum("main/main.py" in call for call in uv_calls), 1)
            self.assertEqual(
                sum("design_qwen3-tts-1.7b-voicedesign/main.py" in call for call in uv_calls),
                1,
            )
            self.assertEqual(
                sum("ornith_1_5_35b_a3b/main.py" in call for call in uv_calls),
                1,
            )
            self.assertEqual(
                sum("qwen3_6_35b_a3b/main.py" in call for call in uv_calls),
                1,
            )
            self.assertEqual(
                sum("model-image/MOSS-Image-SFT-Q8_0/main.py" in call for call in uv_calls),
                1,
            )
            self.assertEqual(
                sum("model-image/FLUX-3-Image-Klein-GGUF-Q8/main.py" in call for call in uv_calls),
                1,
            )
            flux3_validation = [call for call in uv_calls if "scripts/sdcpp/manager.py" in call]
            self.assertEqual(len(flux3_validation), 2)
            self.assertTrue(
                any("--model FLUX-3-Image-Klein-GGUF-Q8" in call for call in flux3_validation)
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
                "18401",
                "18403",
                "18404",
                "18405",
                "18501",
                "18502",
            ):
                self.assertIn(port, port_check)
            llama_starts = [call for call in calls if call.startswith("llama-server ")]
            self.assertEqual(len(llama_starts), 2)
            ornith_start, qwen_start = llama_starts
            self.assertIn(f"--model {model_file}", ornith_start)
            self.assertIn(f"--mmproj {mmproj_file}", ornith_start)
            self.assertIn(f"--spec-draft-model {mtp_file}", ornith_start)
            self.assertIn("--spec-type draft-mtp", ornith_start)
            self.assertIn("--spec-draft-n-max 5", ornith_start)
            self.assertIn("--spec-draft-type-k q8_0", ornith_start)
            self.assertIn("--spec-draft-type-v q8_0", ornith_start)
            self.assertIn("--host 127.0.0.1", ornith_start)
            self.assertIn("--port 18403", ornith_start)
            self.assertIn("--alias Ornith-1.5-35B-A3B-AD-Q5", ornith_start)
            self.assertIn("--device CUDA0", ornith_start)
            self.assertIn("--gpu-layers 99", ornith_start)
            self.assertIn("--n-cpu-moe 24", ornith_start)
            self.assertIn("--mmproj-offload", ornith_start)
            self.assertIn("--flash-attn on", ornith_start)
            self.assertIn("--jinja", ornith_start)
            self.assertIn("--ctx-size 32768", ornith_start)
            self.assertIn("--sleep-idle-seconds 1", ornith_start)
            self.assertIn(f"--model {qwen_model_file}", qwen_start)
            self.assertIn("--host 127.0.0.2", qwen_start)
            self.assertIn("--port 18405", qwen_start)
            self.assertIn("--alias Qwen3.6-35B-A3B-MTP", qwen_start)
            self.assertIn("--device CUDA0", qwen_start)
            self.assertIn("--gpu-layers 99", qwen_start)
            self.assertIn("--n-cpu-moe 25", qwen_start)
            self.assertIn("--spec-type draft-mtp", qwen_start)
            self.assertIn("--threads-batch 12", qwen_start)
            self.assertIn("--spec-draft-n-max 4", qwen_start)
            self.assertIn("--spec-draft-type-k q8_0", qwen_start)
            self.assertIn("--spec-draft-type-v q8_0", qwen_start)
            self.assertIn("--flash-attn on", qwen_start)
            self.assertIn("--jinja", qwen_start)
            self.assertIn("--ctx-size 32768", qwen_start)
            self.assertIn("--sleep-idle-seconds 1", qwen_start)
            self.assertNotIn("--mmproj", qwen_start)
            commands_before_main = calls.index(qwen_start)
            first_service_start = next(
                index
                for index, call in enumerate(calls)
                if call.startswith("uv ") and "main/main.py" in call
            )
            self.assertLess(commands_before_main, first_service_start)
            self.assertLess(calls.index(ornith_start), calls.index(qwen_start))
            source = (ROOT / "start.sh").read_text(encoding="utf-8")
            self.assertIn('AUDIO_CPP_RUNTIME_SRC="$AUDIO_CPP_RUNTIME_PROJECT_DIR/src"', source)
            self.assertIn('export PYTHONPATH="$MODEL_TEXT_DIR:$AUDIO_CPP_RUNTIME_SRC:', source)
            self.assertIn(
                "$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf",
                source,
            )
            self.assertIn(
                "$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/mmproj-Ornith-1.5-35B-A3B-F16.gguf",
                source,
            )
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

    def test_moss_assets_and_sd_cpp_digest_fail_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cases = (
                ("SDCPP_BIN", str(Path(temporary) / "missing-sd-cli"), "sd-cli"),
                ("SDCPP_BIN_SHA256", "invalid", "SDCPP_BIN_SHA256"),
            )
            for variable, value, error_text in cases:
                with self.subTest(variable=variable):
                    environment, trace_path = _preflight_environment(Path(temporary) / variable)
                    environment[variable] = value
                    completed = subprocess.run(
                        ["bash", str(ROOT / "start.sh")],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=15,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(error_text, completed.stderr)
                    self.assertNotIn("scripts/localai/manager.py", trace_path.read_text())

            environment, trace_path = _preflight_environment(Path(temporary) / "missing-tokenizer")
            tokenizer = (
                Path(environment["HF_MIRROR_DIR"])
                / "stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF/tokenizer.json"
            )
            tokenizer.unlink()
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("tokenizer.json", completed.stderr)
            self.assertNotIn("scripts/localai/manager.py", trace_path.read_text())

    def test_flux3_missing_component_fails_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment, trace_path = _preflight_environment(Path(temporary))
            component = (
                Path(environment["HF_MIRROR_DIR"])
                / "stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
            )
            component.unlink()

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", completed.stderr)
            self.assertNotIn("scripts/localai/manager.py", trace_path.read_text())

    def test_flux3_upload_limit_must_be_positive_integer(self) -> None:
        for value in ("0", "-1", "nan", "1.5"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                environment, trace_path = _preflight_environment(Path(temporary))
                environment["FLUX3_IMAGE_MAX_UPLOAD_BYTES"] = value
                completed = subprocess.run(
                    ["bash", str(ROOT / "start.sh")],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=15,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("FLUX3_IMAGE_MAX_UPLOAD_BYTES 必须是正整数", completed.stderr)
                self.assertNotIn("scripts/localai/manager.py", trace_path.read_text())

    def test_invalid_draft_settings_fail_before_localai_starts(self) -> None:
        for variable, value in (
            ("ORNITH_SPEC_TYPE", "draft-simple"),
            ("ORNITH_SPEC_DRAFT_N_MAX", "-1"),
            ("ORNITH_SPEC_DRAFT_CACHE_TYPE_K", "invalid"),
            ("ORNITH_SPEC_DRAFT_CACHE_TYPE_V", "invalid"),
            ("QWEN36_SPEC_DRAFT_N_MAX", "-1"),
            ("QWEN36_SPEC_DRAFT_CACHE_TYPE_K", "invalid"),
            ("QWEN36_SPEC_DRAFT_CACHE_TYPE_V", "invalid"),
        ):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as temporary:
                environment, trace_path = _preflight_environment(Path(temporary))
                environment[variable] = value
                completed = subprocess.run(
                    ["bash", str(ROOT / "start.sh")],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=15,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(variable, completed.stderr)
                self.assertNotIn("manager.py", trace_path.read_text())

    def test_missing_llama_server_fails_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["LLAMA_SERVER_BIN"] = str(root / "missing-llama-server")

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("llama-server", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_discovers_user_local_cuda_llama_server_without_path_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment.pop("LLAMA_SERVER_BIN", None)
            fake_bin = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
            (fake_bin / "llama-server").unlink()
            home = root / "home"
            fallback = home / ".local/src/llama.cpp/build-cuda/bin/llama-server"
            fallback.parent.mkdir(parents=True)
            _write_fake_llama_server(fallback)
            environment["HOME"] = str(home)
            environment["ORNITH_MODEL_FILE"] = str(root / "missing-Ornith.gguf")

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Ornith GGUF 缺失或为空", completed.stderr)
            self.assertNotIn("找不到可执行的 llama-server", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_ornith_without_mtp_preserves_legacy_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["ORNITH_SPEC_TYPE"] = "none"
            environment["ORNITH_MTP_FILE"] = str(root / "missing-mtp.gguf")
            environment["ORNITH_SPEC_DRAFT_N_MAX"] = "unused"
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            ornith_start = next(call for call in calls if call.startswith("llama-server "))
            self.assertIn("--spec-type none", ornith_start)
            self.assertNotIn("--spec-draft-", ornith_start)

    def test_missing_ornith_mtp_fails_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["ORNITH_MTP_FILE"] = str(root / "missing-mtp.gguf")
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Ornith MTP 缺失或为空", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_missing_ornith_weights_fail_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["ORNITH_MODEL_FILE"] = str(root / "missing-Ornith.gguf")

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("GGUF 缺失或为空", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_missing_ornith_mmproj_fails_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["ORNITH_MMPROJ_FILE"] = str(root / "missing-mmproj.gguf")

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("mmproj 缺失或为空", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_missing_qwen36_weights_fail_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["QWEN36_MODEL_FILE"] = str(root / "missing-Qwen3.6.gguf")

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Qwen3.6 GGUF 缺失或为空", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))

    def test_qwen36_backend_url_and_host_are_validated_before_localai_starts(self) -> None:
        for variable, value, message in (
            ("QWEN36_LLAMA_BASE_URL", "http://127.0.0.1:18405", "必须与 Qwen3.6 backend"),
            ("QWEN36_BACKEND_HOST", "0.0.0.0", "必须是 loopback 地址"),
        ):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                environment, trace_path = _preflight_environment(root)
                environment[variable] = value

                completed = subprocess.run(
                    ["bash", str(ROOT / "start.sh")],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=15,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                calls = trace_path.read_text(encoding="utf-8").splitlines()
                self.assertFalse(any("manager.py" in call for call in calls))

    def test_qwen_backend_start_failure_stops_earlier_ornith_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["FAKE_QWEN_FAIL"] = "true"

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=20,
            )

            self.assertNotEqual(completed.returncode, 0)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(
                any("shutdown" in call and "Ornith-1.5-35B-A3B-AD-Q5" in call for call in calls),
                calls,
            )
            runtime_cache = Path(environment["RUNTIME_CACHE_DIR"])
            self.assertFalse((runtime_cache / "ornith-llama-server.pid").exists())
            self.assertFalse((runtime_cache / "qwen3-6-llama-server.pid").exists())
            self.assertFalse((runtime_cache / "gpu-runtime.lock.quarantine").exists())

    def test_cpu_only_llama_server_fails_before_localai_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            cpu_server = root / "cpu-llama-server"
            cpu_server.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "--list-devices" ]]; then echo "Available devices: (none)"; exit 0; fi\n'
                'echo "--n-cpu-moe --sleep-idle-seconds --mmproj-offload"\n',
                encoding="utf-8",
            )
            cpu_server.chmod(0o755)
            environment["LLAMA_SERVER_BIN"] = str(cpu_server)

            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("未提供 CUDA0", completed.stderr)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any("manager.py" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
