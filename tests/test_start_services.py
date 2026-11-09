"""验证启动脚本实际只调用仍维护的 HTTP 服务。"""

from __future__ import annotations

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
        "    print('--n-cpu-moe --sleep-idle-seconds --mmproj-offload --spec-type')\n"
        "    raise SystemExit(0)\n"
        "if '--spec-type' in sys.argv and os.getenv('FAKE_QWEN_FAIL') == 'true':\n"
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
    bin_dir.mkdir()
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
    qwen_model_file = root / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    model_file.write_bytes(b"GGUF-test-model")
    mmproj_file.write_bytes(b"GGUF-test-mmproj")
    qwen_model_file.write_bytes(b"GGUF-test-qwen-model")
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
            llama_server = bin_dir / "llama-server"
            _write_fake_llama_server(llama_server)
            model_file = root / "Ornith.gguf"
            mmproj_file = root / "mmproj.gguf"
            qwen_model_file = root / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
            model_file.write_bytes(b"GGUF-test-model")
            mmproj_file.write_bytes(b"GGUF-test-mmproj")
            qwen_model_file.write_bytes(b"GGUF-test-qwen-model")

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
                "ORNITH_HOST": "127.0.0.1",
                "ORNITH_PORT": "18401",
                "LLAMA_CPP_PORT": "18403",
                "QWEN36_PORT": "18404",
                "QWEN36_BACKEND_HOST": "127.0.0.2",
                "QWEN36_LLAMA_CPP_PORT": "18405",
                "LLAMA_SERVER_BIN": str(llama_server),
                "ORNITH_MODEL_FILE": str(model_file),
                "ORNITH_MMPROJ_FILE": str(mmproj_file),
                "QWEN36_MODEL_FILE": str(qwen_model_file),
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
            self.assertEqual(len(uv_calls), 14)
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
            self.assertEqual(
                sum("ornith_1_5_35b_a3b/main.py" in call for call in uv_calls),
                1,
            )
            self.assertEqual(
                sum("qwen3_6_35b_a3b/main.py" in call for call in uv_calls),
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
                "18401",
                "18403",
                "18404",
                "18405",
            ):
                self.assertIn(port, port_check)
            llama_starts = [call for call in calls if call.startswith("llama-server ")]
            self.assertEqual(len(llama_starts), 2)
            ornith_start, qwen_start = llama_starts
            self.assertIn(f"--model {model_file}", ornith_start)
            self.assertIn(f"--mmproj {mmproj_file}", ornith_start)
            self.assertIn("--host 127.0.0.1", ornith_start)
            self.assertIn("--port 18403", ornith_start)
            self.assertIn("--alias Ornith-1.5-35B-A3B-AD-Q5", ornith_start)
            self.assertIn("--device CUDA0", ornith_start)
            self.assertIn("--gpu-layers 99", ornith_start)
            self.assertIn("--n-cpu-moe 20", ornith_start)
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
