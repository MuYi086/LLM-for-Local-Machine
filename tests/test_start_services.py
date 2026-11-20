"""验证启动脚本实际只调用仍维护的 HTTP 服务。"""

from __future__ import annotations

import hashlib
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
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(executable: Path) -> None:
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'python3 %s\\n\' "$*" >> "$TRACE"\n'
        'if [[ "$*" == *scripts/model_telemetry.py* ]]; then exec /usr/bin/python3 "$@"; fi\n',
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
            f"#!/usr/bin/env bash\nprintf '%s %s\\n' '{command}' \"$*\" >> \"$TRACE\"\n"
            + 'if [[ "$*" == *check_ports.py* ]]; then\n'
            + "  parsed=()\n"
            + "  capture=0\n"
            + '  for argument in "$@"; do\n'
            + '    if [[ "$argument" == */check_ports.py ]]; then capture=1; continue; fi\n'
            + '    if [[ "$capture" == 1 ]]; then parsed+=("$argument"); fi\n'
            + "  done\n"
            + f"  exec /usr/bin/python3 '{ROOT / 'main/check_ports.py'}' \"${{parsed[@]}}\"\n"
            + "fi\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    _write_fake_python(bin_dir / "python3")
    # 性能采样器必须能在没有 NVIDIA 工具的机器上运行。
    gpu_query = bin_dir / "nvidia-smi"
    gpu_query.write_text("#!/usr/bin/env bash\nexit 1\n")
    gpu_query.chmod(0o755)
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
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TRACE": str(trace_path),
        "EASEL_BRIDGE_ENV_FILE": str(root / "bridge.env"),
        "EASEL_BRIDGE_API_KEY": "test-bridge-secret-" + "x" * 32,
        "EASEL_BRIDGE_PORT": "18500",
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
        "LLM_LOCALAI_ENABLED": "true",
        "LLM_LOCALAI_MANAGED": "true",
        "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER": "localai",
        "LLM_LOCAL_AUDIO_CPP_PROVIDER": "localai",
    }
    return environment, trace_path


class StartServicesTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "CONFUCIUS4_TTS_PORT": "28331",
                    "INDEX_TTS25_PORT": "28312",
                    "MIDASHENGLM_GEN_PORT": "28341",
                    "QWEN3_ASR_PORT": "28321",
                    "QWEN3_TTS_BASE_PORT": "28311",
                    "SEED_VC_PORT": "28342",
                    "STABLE_AUDIO_SFX_PORT": "28343",
                    "BS_ROFORMER_PORT": "28344",
                    "QWEN3_FORCED_ALIGNER_PORT": "28345",
                    "MEL_BAND_ROFORMER_PORT": "28346",
                    "HTDEMUCS_PORT": "28347",
                    "VOXCPM2_PORT": "28313",
                },
            )
        )

    def test_disabled_telemetry_does_not_start_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, trace_path = _preflight_environment(root)
            environment["MODEL_TELEMETRY_ENABLED"] = "false"
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                capture_output=True,
                text=True,
                env=environment,
                timeout=25,
            )
            self.assertIn("[性能记录] 已关闭", completed.stdout)
            self.assertNotIn("scripts/model_telemetry.py", trace_path.read_text())
            self.assertFalse((root / "storage/.cache/runtime/telemetry").exists())

    def test_bridge_rejects_missing_or_short_key_before_model_startup(self) -> None:
        for key in ("", "short", "replace-with-" + "x" * 40):
            with self.subTest(key_length=len(key)), tempfile.TemporaryDirectory() as temporary:
                environment, trace_path = _preflight_environment(Path(temporary))
                environment["EASEL_BRIDGE_API_KEY"] = key
                completed = subprocess.run(
                    ["bash", str(ROOT / "start.sh")],
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=15,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("EASEL_BRIDGE_API_KEY", completed.stdout + completed.stderr)
                if key:
                    self.assertNotIn(key, completed.stdout + completed.stderr)
                calls = trace_path.read_text() if trace_path.exists() else ""
                self.assertNotIn("manager.py start", calls)

    def test_bridge_loads_env_and_stops_with_supervisor_or_on_its_own_exit(self) -> None:
        for trigger in (
            "terminal-interrupt",
            "terminal-hangup",
            "repeated-interrupt",
            "logger-exit",
            "interrupt",
            "bridge-exit",
        ):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                environment, trace_path = _preflight_environment(root)
                key = environment.pop("EASEL_BRIDGE_API_KEY")
                Path(environment["EASEL_BRIDGE_ENV_FILE"]).write_text(
                    f"EASEL_BRIDGE_API_KEY={key}\n", encoding="utf-8"
                )
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                environment["EASEL_BRIDGE_PORT"] = str(port)
                fake_uv = root / "bin/uv"
                fake_uv.write_text(
                    "#!/usr/bin/env bash\n"
                    'printf \'uv %s\\n\' "$*" >> "$TRACE"\n'
                    'printf \'service-pid %s\\n\' "$$" >> "$TRACE"\n'
                    'if [[ "$*" == *"uvicorn easel_bridge.app:app"* ]]; then\n'
                    '  printf \'bridge-pid %s\\n\' "$$" >> "$TRACE"\n'
                    '  while [[ "$1" != python ]]; do shift; done\n'
                    "  shift\n"
                    f'  exec "{sys.executable}" "$@"\n'
                    "fi\n"
                    'if [[ "$*" == *main.py* ]]; then exec sleep 120; fi\n',
                    encoding="utf-8",
                )
                with (root / "console.log").open("w+") as console:
                    process = subprocess.Popen(
                        ["bash", str(ROOT / "start.sh")],
                        stdout=console,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    try:
                        request = Request(
                            f"http://127.0.0.1:{port}/v1/models",
                            headers={"Authorization": f"Bearer {key}"},
                        )
                        opener = build_opener(ProxyHandler({}))
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline and process.poll() is None:
                            try:
                                with opener.open(request, timeout=0.5) as response:
                                    models = json.load(response)
                                break
                            except (URLError, TimeoutError):
                                time.sleep(0.05)
                        else:
                            console.seek(0)
                            self.fail(f"网关未就绪：{console.read()}")
                        self.assertEqual(
                            {model["id"] for model in models["data"]},
                            {"MOSS-Image-SFT-Q8_0", "OpenBMB/VoxCPM2-GGUF"},
                        )
                        if trigger == "interrupt":
                            process.send_signal(signal.SIGINT)
                        elif trigger.startswith("terminal-") or trigger == "repeated-interrupt":
                            terminal_signal = (
                                signal.SIGHUP if trigger == "terminal-hangup" else signal.SIGINT
                            )
                            os.killpg(process.pid, terminal_signal)
                            if trigger == "repeated-interrupt":
                                time.sleep(0.1)
                                os.killpg(process.pid, signal.SIGINT)
                        elif trigger == "logger-exit":
                            for child in Path("/proc").iterdir():
                                if not child.name.isdigit():
                                    continue
                                try:
                                    arguments = (child / "cmdline").read_bytes().split(b"\0")
                                    parent = int(
                                        (child / "stat").read_text().rsplit(")", 1)[1].split()[1]
                                    )
                                except (FileNotFoundError, ProcessLookupError):
                                    continue
                                if parent == process.pid and arguments[0] == b"tee":
                                    os.kill(int(child.name), signal.SIGKILL)
                                    break
                            else:
                                self.fail("未找到启动脚本的日志进程")
                            time.sleep(0.1)
                            process.send_signal(signal.SIGINT)
                        else:
                            bridge_pid = next(
                                int(line.split()[1])
                                for line in trace_path.read_text().splitlines()
                                if line.startswith("bridge-pid ")
                            )
                            os.kill(bridge_pid, signal.SIGTERM)
                        process.wait(timeout=20)
                        with socket.socket() as probe:
                            self.assertNotEqual(
                                probe.connect_ex(("127.0.0.1", port)), 0, "退出后网关仍在监听"
                            )
                        with socket.socket() as listener:
                            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            listener.bind(("127.0.0.1", port))
                        runtime = root / "storage/.cache/runtime"
                        manifests = list((runtime / "telemetry").glob("*/manifest.jsonl"))
                        self.assertEqual(len(manifests), 1)
                        manifest = json.loads(manifests[0].read_text())
                        self.assertEqual(manifest["owner_pid"], process.pid)
                        samples = (manifests[0].parent / "resources.jsonl").read_text().splitlines()
                        self.assertGreaterEqual(len(samples), 2)
                        self.assertIsNone(json.loads(samples[-1])["gpu"]["gpus"])
                        self.assertFalse((runtime / "gpu-runtime.lock.quarantine").exists())
                        console.seek(0)
                        self.assertNotIn(key, console.read())
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=20)
                        # 即使回归失败，也只清理该测试启动的独立进程组，避免污染后续端口。
                        groups = {
                            int(line.split()[1])
                            for line in trace_path.read_text().splitlines()
                            if line.startswith("service-pid ")
                        }
                        for pgid in groups:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass

    def test_positive_timeouts_reject_zero_and_invalid_decimals_before_startup(self) -> None:
        for variable in ("GPU_LOCK_WAIT_TIMEOUT",):
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
                    self.assertIn(f"{variable} 必须是正数", completed.stdout + completed.stderr)
                    calls = trace_path.read_text() if trace_path.exists() else ""
                    self.assertNotIn("manager.py", calls)

    def test_start_invokes_managed_services_without_ornith(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            trace_path = root / "commands.log"
            for command in ("uv",):
                executable = bin_dir / command
                executable.write_text(
                    f"#!/usr/bin/env bash\nprintf '%s %s\\n' '{command}' \"$*\" >> \"$TRACE\"\n"
                    'if [[ "$*" == *check_ports.py* ]]; then\n'
                    "  parsed=()\n"
                    "  capture=0\n"
                    '  for argument in "$@"; do\n'
                    '    if [[ "$argument" == */check_ports.py ]]; then capture=1; continue; fi\n'
                    '    if [[ "$capture" == 1 ]]; then parsed+=("$argument"); fi\n'
                    "  done\n"
                    f"  exec /usr/bin/python3 '{ROOT / 'main/check_ports.py'}' \"${{parsed[@]}}\"\n"
                    "fi\n"
                    'if [[ "$*" == *"uvicorn easel_bridge.app:app"* ]]; then\n'
                    "  printf 'bridge-upstreams %s %s\\n' "
                    '"$EASEL_BRIDGE_MOSS_URL" '
                    '"$EASEL_BRIDGE_VOXCPM_URL" >> "$TRACE"\n'
                    "fi\n"
                    'if [[ "$*" == *main.py* || "$*" == *"uvicorn easel_bridge.app:app"* ]]; then sleep 1; exit 0; fi\n',
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            _write_fake_python(bin_dir / "python3")
            gpu_query = bin_dir / "nvidia-smi"
            gpu_query.write_text("#!/usr/bin/env bash\nexit 1\n")
            gpu_query.chmod(0o755)
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

            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "TRACE": str(trace_path),
                "EASEL_BRIDGE_ENV_FILE": str(root / "bridge.env"),
                "EASEL_BRIDGE_API_KEY": "test-bridge-secret-" + "x" * 32,
                "EASEL_BRIDGE_PORT": "18500",
                "STORAGE_DIR": str(root / "storage"),
                "RUNTIME_CACHE_DIR": str(root / "storage/.cache/runtime"),
                "HF_MIRROR_DIR": str(root / "hf-mirror"),
                "SDCPP_BIN": str(sd_cpp_binary),
                "SDCPP_BIN_SHA256": hashlib.sha256(sd_cpp_binary.read_bytes()).hexdigest(),
                "MOSS_IMAGE_PORT": "18501",
                "FLUX3_IMAGE_HOST": "127.0.0.1",
                "FLUX3_IMAGE_PORT": "18502",
                "FLUX3_IMAGE_UPLOAD_DIR": str(root / "storage/.cache/runtime/sdcpp_uploads"),
                "FLUX3_IMAGE_MAX_UPLOAD_BYTES": "67108864",
                "HOST": "0.0.0.0",
                "PORT": "18300",
                "QWEN_VOICEDESIGN_HOST": "127.0.0.1",
                "QWEN_VOICEDESIGN_PORT": "18301",
                "LLM_LOCALAI_ENABLED": "true",
                "LLM_LOCALAI_MANAGED": "true",
                "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER": "localai",
                "LLM_LOCAL_AUDIO_CPP_PROVIDER": "localai",
            }
            environment.pop("MOSS_IMAGE_HOST", None)
            environment.pop("VOXCPM2_HOST", None)
            for variable in ("EASEL_BRIDGE_MOSS_URL", "EASEL_BRIDGE_VOXCPM_URL"):
                environment.pop(variable, None)
            completed = subprocess.run(
                ["bash", str(ROOT / "start.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("http://127.0.0.1:18501/v1/openmoss/", completed.stdout)
            self.assertIn("http://127.0.0.1:28313/v1/OpenBMB/VoxCPM2-GGUF", completed.stdout)
            self.assertNotIn("Ornith", completed.stdout + completed.stderr)
            self.assertNotIn("8504", completed.stdout)
            self.assertNotIn("8506", completed.stdout)
            calls = trace_path.read_text(encoding="utf-8").splitlines()
            uv_calls = [call for call in calls if call.startswith("uv ")]
            self.assertEqual(len(uv_calls), 23)
            bridge_call = next(call for call in uv_calls if "uvicorn easel_bridge.app:app" in call)
            self.assertIn("--host 127.0.0.1 --port 18500", bridge_call)
            upstreams = next(call for call in calls if call.startswith("bridge-upstreams "))
            self.assertIn("http://127.0.0.1:18501/v1/openmoss/MOSS-Image-SFT-Q8_0", upstreams)
            self.assertIn("http://127.0.0.1:28313/v1/OpenBMB/VoxCPM2-GGUF", upstreams)
            self.assertNotIn("chat", upstreams)
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
            self.assertFalse(any("model-text/" in call for call in uv_calls))
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
            self.assertEqual(len(audio_cpp_calls), 14)
            for service_directory in (
                "tts_confucius4-tts",
                "tts_indextts2.5",
                "gen_midashenglm",
                "asr_qwen3-1.7b",
                "tts_qwen3-1.7b-base",
                "vc_seed-vc",
                "gen_stable-audio-3-small-sfx",
                "Stable-Audio-3-Medium-GGUF",
                "MOSS-SoundEffect-v2-GGUF",
                "tts_voxcpm2",
                "BS-RoFormer-ep368-GGUF",
                "Qwen3-ForcedAligner-0.6B-GGUF",
                "Mel-Band-RoFormer-GGUF",
                "HTDemucs-GGUF",
            ):
                self.assertEqual(
                    sum(f"{service_directory}/main.py" in call for call in uv_calls),
                    1,
                )
            port_check = next(call for call in uv_calls if "check_ports.py" in call)
            for port in (
                "18300",
                "8301",
                "28331",
                "28312",
                "28341",
                "28321",
                "28311",
                "28342",
                "28343",
                "28344",
                "28345",
                "28346",
                "28347",
                "8314",
                "8315",
                "28313",
                "18501",
                "18502",
                "18500",
            ):
                self.assertIn(port, port_check)
            for port in ("18504", "18506"):
                self.assertNotIn(port, port_check)
            self.assertFalse(any(call.startswith("llama-server ") for call in calls))
            self.assertFalse(any("llama_cpp_runtime" in call for call in calls))
            source = (ROOT / "start.sh").read_text(encoding="utf-8")
            self.assertIn('AUDIO_CPP_RUNTIME_SRC="$AUDIO_CPP_RUNTIME_PROJECT_DIR/src"', source)
            self.assertIn('export PYTHONPATH="$AUDIO_CPP_RUNTIME_SRC:', source)
            self.assertNotIn("MODEL_TEXT_DIR", source)
            self.assertNotIn("ornith-ai", source)
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
                    self.assertIn(error_text, completed.stdout + completed.stderr)
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
            self.assertIn("tokenizer.json", completed.stdout + completed.stderr)
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
            self.assertIn(
                "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", completed.stdout + completed.stderr
            )
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
                self.assertIn(
                    "FLUX3_IMAGE_MAX_UPLOAD_BYTES 必须是正整数", completed.stdout + completed.stderr
                )
                self.assertNotIn("scripts/localai/manager.py", trace_path.read_text())


if __name__ == "__main__":
    unittest.main()
