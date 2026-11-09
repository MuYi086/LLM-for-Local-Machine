"""管理固定版本的宿主机 LocalAI 进程及其 audio-cpp 后端目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "localai/manifests/versions.json"
MODEL_CANDIDATES_PATH = ROOT / "localai/manifests/audio-cpp-models.json"
QWEN3_VOICEDESIGN_MODEL_NAME = "llm-local.qwen3-tts.1p7b-voicedesign"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_COMMAND_TIMEOUT_SECONDS = 15
_STOP_TIMEOUT_SECONDS = 15


class LocalAIManagerError(RuntimeError):
    """LocalAI 宿主机进程或固定后端无法安全管理。"""


class LocalAIManager:
    """校验本机安装并管理 LocalAI 宿主机进程组。"""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        readiness_timeout: float = 60.0,
        stop_timeout: float = _STOP_TIMEOUT_SECONDS,
    ) -> None:
        self.environ = dict(os.environ if environ is None else environ)
        self.readiness_timeout = readiness_timeout
        self.stop_timeout = stop_timeout
        self.manifest = self._load_manifest()
        self.data_dir = Path(
            self.environ.get(
                "LLM_LOCALAI_DATA_DIR",
                str(Path(self.environ.get("STORAGE_DIR", ROOT / "storage")) / ".cache/localai"),
            )
        ).expanduser()
        self.models_dir = Path(
            self.environ.get("LLM_LOCALAI_MODELS_DIR", str(self.data_dir / "models"))
        ).expanduser()
        self.backends_dir = Path(
            self.environ.get("LLM_LOCALAI_BACKENDS_DIR", str(self.data_dir / "backends"))
        ).expanduser()
        self.binary_path = Path(
            self.environ.get("LLM_LOCALAI_BINARY", str(self.data_dir / "bin/local-ai"))
        ).expanduser()
        self.logs_dir = self.data_dir / "logs"
        self.state_dir = self.data_dir / "run"
        self.state_path = self.state_dir / "localai.json"
        self.install_stamp_path = self.data_dir / ".llm-local-backend-install.json"

    @staticmethod
    def _load_manifest() -> dict[str, object]:
        """读取并校验 P0 中固定的 LocalAI/audio.cpp 源码版本。"""
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            localai = manifest["localai"]
            backend = manifest["audio_cpp_backend"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError("LocalAI 版本清单缺失或格式无效。") from exc

        if not isinstance(localai, dict) or not isinstance(backend, dict):
            raise LocalAIManagerError("LocalAI/audio.cpp 版本清单格式无效。")
        if localai.get("version") != "v4.11.0":
            raise LocalAIManagerError("当前 manager 只支持清单锁定的 LocalAI v4.11.0。")
        for label, value in (
            ("LocalAI", localai.get("source_commit")),
            ("audio.cpp", backend.get("source_commit")),
        ):
            if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
                raise LocalAIManagerError(f"{label} 缺少完整的固定源码 commit。")
        if backend.get("localai_source_commit") != localai["source_commit"]:
            raise LocalAIManagerError("audio.cpp backend 与 LocalAI 源码版本不匹配。")
        if backend.get("compute_variant") != "CUDA 12":
            raise LocalAIManagerError("当前 manager 只支持 P0 锁定的 CUDA 12 后端构建。")
        return manifest

    @property
    def localai_version(self) -> str:
        """返回锁定的 LocalAI 版本。"""
        return str(self.manifest["localai"]["version"])

    @property
    def localai_commit(self) -> str:
        """返回锁定的 LocalAI 源码 commit。"""
        return str(self.manifest["localai"]["source_commit"])

    @property
    def audio_cpp_commit(self) -> str:
        """返回 LocalAI v4.11.0 内嵌的 audio.cpp commit。"""
        return str(self.manifest["audio_cpp_backend"]["source_commit"])

    def verify_install(self) -> None:
        """校验预先准备的宿主机二进制和后端，并写入内容指纹。"""
        self._ensure_linux_amd64()
        self._ensure_directories()
        self._write_install_stamp()
        print("已校验宿主机 LocalAI/audio.cpp 文件；该检查不代表模型推理验收。")

    def _audio_cpp_candidates(self) -> list[dict[str, object]]:
        """读取所有被固定 audio.cpp revision 筛选并登记的模型候选。"""
        try:
            manifest = json.loads(MODEL_CANDIDATES_PATH.read_text(encoding="utf-8"))
            candidates = manifest["models"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError("audio.cpp 模型候选清单缺失或格式无效。") from exc
        if (
            manifest.get("schema_version") != 1
            or not isinstance(candidates, list)
            or not candidates
        ):
            raise LocalAIManagerError("audio.cpp 模型候选清单版本或 models 格式无效。")
        return candidates

    def _verify_candidate_model_file(self, candidate: Mapping[str, object]) -> Path:
        """检查清单中指定的本地 GGUF 文件，不下载模型。"""
        filename = candidate.get("required_file")
        source_directory = candidate.get("source_directory")
        model_file_env = candidate.get("model_file_env")
        if not all(
            isinstance(value, str) and value
            for value in (filename, source_directory, model_file_env)
        ):
            raise LocalAIManagerError("audio.cpp 候选缺少模型文件名、目录或环境变量。")
        model_file_value = self.environ.get(str(model_file_env))
        if model_file_value:
            model_file = Path(os.path.expandvars(model_file_value)).expanduser().resolve()
        else:
            hf_mirror = Path(self.environ.get("HF_MIRROR_DIR", "~/hf-mirror")).expanduser()
            model_file = (
                hf_mirror / "audio.cpp-gguf" / str(source_directory) / str(filename)
            ).resolve()
        if model_file.name != filename:
            raise LocalAIManagerError(f"{model_file_env} 文件名必须为 {filename}。")
        if not model_file.is_file() or model_file.stat().st_size == 0:
            raise LocalAIManagerError(f"audio.cpp 模型 GGUF 缺失或为空：{model_file}。")
        return model_file

    @staticmethod
    def _render_candidate_config(candidate: Mapping[str, object], model_reference: str) -> str:
        """生成固定 family/task、CUDA 设备与模型引用的 LocalAI YAML。"""
        model_id = candidate.get("model_id")
        family = candidate.get("family")
        task = candidate.get("task")
        if not all(isinstance(value, str) and value for value in (model_id, family, task)):
            raise LocalAIManagerError("audio.cpp 候选缺少 model_id、family 或 task。")
        return (
            f"name: {model_id}\n"
            "backend: audio-cpp\n"
            "parameters:\n"
            f'  model: "{model_reference}"\n'
            "options:\n"
            f"  - family:{family}\n"
            f"  - task:{task}\n"
            "  - backend:cuda\n"
            "  - device:0\n"
        )

    def register_audio_cpp_candidates(self) -> None:
        """先预检全部本地 GGUF 与配置冲突，再注册所有 audio.cpp 候选。"""
        prepared: list[tuple[dict[str, object], Path, str, set[str]]] = []
        candidates = self._audio_cpp_candidates()
        for candidate_value in candidates:
            if not isinstance(candidate_value, dict):
                raise LocalAIManagerError("audio.cpp 候选条目格式无效。")
            candidate = candidate_value
            model_id = candidate.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise LocalAIManagerError("audio.cpp 候选缺少 model_id。")
            model_file = self._verify_candidate_model_file(candidate)
            reference = f"{model_id}.gguf"
            link = self.models_dir / reference
            if link.exists() or link.is_symlink():
                self._verify_model_file(model_id, model_file)
            config_path = self.models_dir / f"{model_id}.yaml"
            rendered = self._render_candidate_config(candidate, reference)
            allowed_previous_contents: set[str] = set()
            if model_id == QWEN3_VOICEDESIGN_MODEL_NAME:
                allowed_previous_contents.add(
                    f"name: {model_id}\n"
                    "backend: audio-cpp\n"
                    "parameters:\n"
                    f'  model: "{model_id}.hf"\n'
                    "options:\n"
                    "  - family:qwen3_tts\n"
                    "  - task:vdes\n"
                    "  - backend:cuda\n"
                    "  - device:0\n"
                )
                allowed_previous_contents.add(
                    f"name: {model_id}\n"
                    "backend: audio-cpp\n"
                    "parameters:\n"
                    "  # manager.py 注册此处所需的受控相对 GGUF 文件引用。\n"
                    f'  model: "{model_id}.gguf"\n'
                    "options:\n"
                    "  - family:qwen3_tts\n"
                    "  - task:vdes\n"
                    "  - backend:cuda\n"
                    "  - device:0\n"
                )
            if config_path.exists():
                existing = config_path.read_text(encoding="utf-8")
                if existing != rendered and existing not in allowed_previous_contents:
                    raise LocalAIManagerError(f"拒绝覆盖不同的 LocalAI 模型配置：{config_path}")
            prepared.append((candidate, model_file, rendered, allowed_previous_contents))

        for candidate, model_file, rendered, allowed_previous_contents in prepared:
            model_id = str(candidate["model_id"])
            self._register_model_file(model_id, model_file)
            self._register_model_config(
                model_id,
                rendered,
                allowed_previous_contents=allowed_previous_contents,
            )

    def verify_audio_cpp_candidates(self) -> None:
        """检查所有固定 audio.cpp GGUF 引用和候选 YAML。"""
        for candidate_value in self._audio_cpp_candidates():
            if not isinstance(candidate_value, dict):
                raise LocalAIManagerError("audio.cpp 候选条目格式无效。")
            candidate = candidate_value
            model_id = candidate.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise LocalAIManagerError("audio.cpp 候选缺少 model_id。")
            model_file = self._verify_candidate_model_file(candidate)
            try:
                model_reference = self._verify_model_file(model_id, model_file)
                installed_yaml = (self.models_dir / f"{model_id}.yaml").read_text(encoding="utf-8")
            except OSError as exc:
                raise LocalAIManagerError(
                    "audio.cpp 候选尚未注册；先准备模型并运行 "
                    "manager.py register-audio-cpp-candidates。"
                ) from exc
            expected_yaml = self._render_candidate_config(candidate, model_reference)
            if installed_yaml != expected_yaml:
                raise LocalAIManagerError(f"LocalAI 模型 YAML 与固定候选不一致：{model_id}")

    def _register_model_file(self, model_name: str, model_file: Path) -> str:
        """在 LocalAI models 内建立受控相对 GGUF 文件引用。"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        reference = f"{model_name}.gguf"
        destination = self.models_dir / reference
        if destination.exists() or destination.is_symlink():
            return self._verify_model_file(model_name, model_file)
        temporary = self.models_dir / f".{reference}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.symlink_to(model_file)
            os.replace(temporary, destination)
            self._fsync_directory(self.models_dir)
        finally:
            temporary.unlink(missing_ok=True)
        return reference

    def _verify_model_file(self, model_name: str, model_file: Path) -> str:
        """确认 LocalAI 文件引用指向经过校验的 GGUF。"""
        reference = f"{model_name}.gguf"
        link = self.models_dir / reference
        if not link.is_symlink() or link.resolve() != model_file.resolve():
            raise LocalAIManagerError(f"LocalAI GGUF 文件引用缺失或目标不一致：{link}")
        return reference

    def _register_model_config(
        self,
        model_name: str,
        rendered: str,
        *,
        allowed_previous_contents: set[str] | None = None,
    ) -> None:
        """原子注册模型 YAML，拒绝覆盖内容不同的既有配置。"""
        destination = self.models_dir / f"{model_name}.yaml"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_text(encoding="utf-8")
            if existing == rendered:
                print(f"模型配置已匹配：{destination}")
                return
            if existing not in (allowed_previous_contents or set()):
                raise LocalAIManagerError(
                    f"拒绝覆盖不同的 LocalAI 模型配置：{destination}；请先人工检查并移除旧配置。"
                )
            backup_dir = self.data_dir / "config-backups" / time.strftime("%Y-%m-%d")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{destination.stem}.{uuid.uuid4().hex}.yaml"
            shutil.copy2(destination, backup_path)
            with backup_path.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{model_name}-", suffix=".yaml.tmp", dir=self.models_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(rendered)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, destination)
            self._fsync_directory(self.models_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"已注册 E0 候选配置（不是推理验收）：{destination}")

    def start(self) -> None:
        """以独立进程组启动只监听 loopback 的 LocalAI，并等待 readiness。"""
        if not self._enabled():
            raise LocalAIManagerError("启动前必须显式设置 LLM_LOCALAI_ENABLED=true。")
        self._ensure_linux_amd64()
        self._ensure_install_stamp()
        self._ensure_directories()
        base_url, port = self._base_url_and_port()
        self._ensure_port_available(port)
        self._ensure_no_managed_processes()
        if self._read_state(required=False) is not None:
            raise LocalAIManagerError("受管 LocalAI 已有进程状态；先检查并显式执行 stop。")

        owner_pid: int | None = None
        owner_start_time: str | None = None
        raw_owner_pid = self.environ.get("LLM_LOCALAI_OWNER_PID")
        if raw_owner_pid:
            try:
                owner_pid = int(raw_owner_pid)
            except ValueError as exc:
                raise LocalAIManagerError("LLM_LOCALAI_OWNER_PID 必须是进程号。") from exc
            owner_start_time = self._process_start_time(owner_pid)
            if owner_start_time is None:
                raise LocalAIManagerError("LocalAI 启动所有者已退出。")

        run_id = uuid.uuid4().hex
        log_path = self.logs_dir / "localai.log"
        localai_environment = {
            **os.environ,
            **self.environ,
            "LOCALAI_ADDRESS": f"127.0.0.1:{port}",
            "LOCALAI_MODELS_PATH": str(self.models_dir.resolve()),
            "LOCALAI_BACKENDS_PATH": str(self.backends_dir.resolve()),
            "LOCALAI_DATA_PATH": str((self.data_dir / "data").resolve()),
            "LOCALAI_MAX_ACTIVE_BACKENDS": "1",
            "LOCALAI_AUTOLOAD_GALLERIES": "false",
            "LOCALAI_AUTOLOAD_BACKEND_GALLERIES": "false",
            "LOCALAI_DISABLE_GALLERY_ENDPOINT": "true",
            "LOCALAI_DISABLE_WEBUI": "true",
            "LLM_LOCALAI_RUN_ID": run_id,
        }
        runtime_cache = Path(
            localai_environment.get(
                "RUNTIME_CACHE_DIR",
                str(
                    Path(localai_environment.get("STORAGE_DIR", ROOT / "storage"))
                    / ".cache/runtime"
                ),
            )
        ).expanduser()
        localai_environment.setdefault(
            "LLM_LOCALAI_ARTIFACT_DIR", str(runtime_cache / "localai_artifacts")
        )
        command = [str(self.binary_path), "run"]
        if api_key := self.environ.get("LLM_LOCALAI_API_KEY"):
            localai_environment["LOCALAI_API_KEY"] = api_key
        try:
            process_id = self._spawn_detached(command, localai_environment, stdout_path=log_path)
        except OSError as exc:
            raise LocalAIManagerError(f"无法启动宿主机 LocalAI：{exc}") from exc

        process_start_time = self._process_start_time(process_id)
        if process_start_time is None:
            self._reap_spawned_child(process_id)
            raise LocalAIManagerError("LocalAI 子进程启动后立即退出。")
        state: dict[str, object] = {
            "schema_version": 1,
            "boot_id": self._boot_id(),
            "pid": process_id,
            "pgid": process_id,
            "process_start_time": process_start_time,
            "owner_pid": owner_pid,
            "owner_start_time": owner_start_time,
            "run_id": run_id,
            "base_url": base_url,
            "command": command,
            "started_at": time.time(),
        }
        try:
            self._atomic_write_json(self.state_path, state)
            if owner_pid is not None and owner_start_time is not None:
                self._start_owner_watcher(owner_pid, owner_start_time, state)
            self._wait_until_ready(base_url)
        except BaseException:
            self._terminate_process_group(state)
            self.state_path.unlink(missing_ok=True)
            raise
        print(f"LocalAI 宿主机进程已就绪：{base_url}（PID {process_id}）")

    def stop(self) -> None:
        """终止受管进程组并确认所有组成员退出且 API 端口关闭。"""
        state = self._read_state(required=False)
        if state is None:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            print("受管 LocalAI 未运行。")
            return
        self._terminate_process_group(state)
        self.state_path.unlink(missing_ok=True)
        self._fsync_directory(self.state_dir)
        print("LocalAI 进程组已停止并确认回收。")

    def status(self) -> None:
        """报告受管进程及监听端口状态，不触发模型加载。"""
        state = self._read_state(required=False)
        if state is None:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            print("受管 LocalAI 未运行。")
            return
        pid = int(state["pid"])
        if self._process_start_time(pid) != str(state["process_start_time"]):
            raise LocalAIManagerError("LocalAI 状态文件中的 PID 已失效或发生复用；保留状态供排查。")
        print(f"LocalAI 正在运行：PID {pid}，地址 {state['base_url']}")

    def verify_managed_localai_stopped(self) -> None:
        """确认受管 LocalAI 进程组不存在且 loopback API 端口关闭。"""
        state = self._read_state(required=False)
        if state is not None:
            raise LocalAIManagerError("受管 LocalAI 状态仍存在；先运行 manager.py stop。")
        self._ensure_no_managed_processes()
        self._ensure_port_closed()

    @staticmethod
    def _ensure_port_available(port: int) -> None:
        """在启动前检测目标 loopback 端口是否已占用。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # 与 Uvicorn 的端口合同一致，允许已关闭连接的 TIME_WAIT 状态下立即重启。
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise LocalAIManagerError(f"LocalAI loopback 端口 {port} 已被占用。") from exc

    def _spawn_detached(
        self,
        command: list[str],
        environment: Mapping[str, str],
        *,
        stdout_path: Path | None = None,
    ) -> int:
        """使用 posix_spawn 建立独立会话，避免长驻子进程的 Popen 析构告警。"""
        setsid_path = shutil.which("setsid")
        if setsid_path is None:
            raise FileNotFoundError("找不到 setsid；请按仓库部署要求安装 util-linux。")
        output_path = str(stdout_path) if stdout_path is not None else os.devnull
        output_flags = os.O_WRONLY
        if stdout_path is not None:
            output_flags |= os.O_CREAT | os.O_APPEND
        file_actions = [
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
            (os.POSIX_SPAWN_OPEN, 1, output_path, output_flags, 0o600),
            (os.POSIX_SPAWN_DUP2, 1, 2),
        ]
        process_id = os.posix_spawn(
            setsid_path,
            [setsid_path, "--", *command],
            dict(environment),
            file_actions=file_actions,
        )
        try:
            self._ensure_detached_process_group(process_id)
        except BaseException:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._reap_spawned_child(process_id)
            raise
        return process_id

    def _ensure_detached_process_group(self, process_id: int) -> None:
        """等 setsid exec 完成，确认 PID、PGID、SID 一致后再记录进程状态。"""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if os.getpgid(process_id) == process_id and os.getsid(process_id) == process_id:
                    return
            except ProcessLookupError:
                break
            time.sleep(0.01)
        raise LocalAIManagerError("setsid 子进程未能建立独立会话和进程组。")

    def _reap_spawned_child(self, process_id: int) -> None:
        """有界回收本进程直接启动的子进程；独立 stop 命令会收到 ChildProcessError。"""
        deadline = time.monotonic() + self.stop_timeout
        while True:
            try:
                waited_pid, _ = os.waitpid(process_id, os.WNOHANG)
            except ChildProcessError:
                return
            if waited_pid == process_id:
                return
            if time.monotonic() >= deadline:
                raise LocalAIManagerError("LocalAI 子进程退出后未能在时限内回收。")
            time.sleep(0.05)

    def _localai_version_output(self) -> str:
        """读取本机二进制版本，并在启动前确认文件可执行。"""
        if not self.binary_path.is_file() or not os.access(self.binary_path, os.X_OK):
            raise LocalAIManagerError(
                f"LocalAI 二进制不存在或不可执行：{self.binary_path}；"
                "请按固定源码版本准备宿主机二进制后重试。"
            )
        try:
            result = subprocess.run(
                [str(self.binary_path), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalAIManagerError("LocalAI --version 检查超时。") from exc
        except OSError as exc:
            raise LocalAIManagerError(f"无法读取 LocalAI 二进制版本：{exc}") from exc
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0 or self.localai_version not in output:
            raise LocalAIManagerError(
                f"LocalAI 二进制版本检查失败；期望 {self.localai_version}，输出：{output}"
            )
        return output

    def _backend_run_script(self) -> Path:
        """查找并校验 LocalAI audio-cpp backend 的启动入口。"""
        backend_dir = self.backends_dir / "audio-cpp"
        run_script = backend_dir / "run.sh"
        try:
            resolved_dir = backend_dir.resolve(strict=True)
            resolved_script = run_script.resolve(strict=True)
        except OSError as exc:
            raise LocalAIManagerError(
                "缺少 audio-cpp 原生 backend 包；需准备 run.sh 到 LocalAI backends/audio-cpp。"
            ) from exc
        if not resolved_dir.is_relative_to(self.backends_dir.resolve()):
            raise LocalAIManagerError("audio-cpp backend 目录不能指向 backends 路径之外。")
        if not resolved_script.is_relative_to(resolved_dir):
            raise LocalAIManagerError("audio-cpp backend run.sh 不能逃逸其固定目录。")
        if not resolved_script.is_file() or not os.access(resolved_script, os.X_OK):
            raise LocalAIManagerError("audio-cpp backend run.sh 不存在或不可执行。")
        return resolved_script

    def _write_install_stamp(self) -> None:
        """校验 LocalAI 与后端文件后写入其内容指纹。"""
        output = self._localai_version_output()
        run_script = self._backend_run_script()
        stamp = {
            "localai_version": self.localai_version,
            "expected_localai_source_commit": self.localai_commit,
            "localai_binary": str(self.binary_path.resolve()),
            "localai_binary_sha256": self._sha256_file(self.binary_path),
            "localai_version_output": output,
            "expected_audio_cpp_source_commit": self.audio_cpp_commit,
            "audio_cpp_backend_run_script": str(run_script),
            "audio_cpp_backend_run_script_sha256": self._sha256_file(run_script),
            "audio_cpp_backend_tree_sha256": self._sha256_tree(self.backends_dir / "audio-cpp"),
            "compute_variant": "CUDA 12",
        }
        self._atomic_write_json(self.install_stamp_path, stamp)

    def _ensure_install_stamp(self) -> None:
        """拒绝运行版本未核验或文件内容发生变化的本机安装。"""
        try:
            stamp = json.loads(self.install_stamp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "缺少 LocalAI/audio-cpp 校验记录；先运行 manager.py verify-install。"
            ) from exc
        current_version = self._localai_version_output()
        run_script = self._backend_run_script()
        expected = {
            "localai_version": self.localai_version,
            "expected_localai_source_commit": self.localai_commit,
            "localai_binary": str(self.binary_path.resolve()),
            "localai_binary_sha256": self._sha256_file(self.binary_path),
            "localai_version_output": current_version,
            "expected_audio_cpp_source_commit": self.audio_cpp_commit,
            "audio_cpp_backend_run_script": str(run_script),
            "audio_cpp_backend_run_script_sha256": self._sha256_file(run_script),
            "audio_cpp_backend_tree_sha256": self._sha256_tree(self.backends_dir / "audio-cpp"),
            "compute_variant": "CUDA 12",
        }
        if stamp != expected:
            raise LocalAIManagerError(
                "本机 LocalAI/audio-cpp 文件与校验记录不一致；重新运行 verify-install。"
            )

    def _ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.models_dir,
            self.backends_dir,
            self.logs_dir,
            self.state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_linux_amd64() -> None:
        if sys.platform != "linux" or os.uname().machine not in {"x86_64", "amd64"}:
            raise LocalAIManagerError("固定宿主机二进制方案仅支持 linux/amd64。")

    def _base_url_and_port(self) -> tuple[str, int]:
        raw_url = self.environ.get("LLM_LOCALAI_BASE_URL", "http://127.0.0.1:8400")
        parsed = urlsplit(raw_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise LocalAIManagerError("受管 LocalAI 仅允许无路径的 HTTP loopback 地址。")
        try:
            port = parsed.port or 80
        except ValueError as exc:
            raise LocalAIManagerError("LLM_LOCALAI_BASE_URL 端口无效。") from exc
        if not 1 <= port <= 65535:
            raise LocalAIManagerError("LLM_LOCALAI_BASE_URL 端口超出有效范围。")
        return f"http://{parsed.hostname}:{port}", port

    def _ensure_port_closed(self, *, timeout: float = 0) -> None:
        base_url, port = self._base_url_and_port()
        deadline = time.monotonic() + timeout
        while True:
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            except OSError:
                return
            connection.close()
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        raise LocalAIManagerError(f"LocalAI 状态文件不存在，但 loopback 端口仍可连接：{base_url}")

    def _managed_process_ids(self) -> list[int]:
        """查找命令行或可执行路径仍指向本地安装目录的 LocalAI/backend 进程。"""
        markers = (
            str(self.binary_path.resolve()).encode(),
            str(self.backends_dir.resolve()).encode(),
        )
        process_ids: list[int] = []
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = self._read_process_stat(process_id)
            if stat is None or stat[0] in {"Z", "X"}:
                continue
            try:
                command_line = (process_path / "cmdline").read_bytes()
                executable = os.readlink(process_path / "exe").encode()
            except OSError:
                continue
            if any(marker in command_line or marker in executable for marker in markers):
                process_ids.append(process_id)
        return process_ids

    def _ensure_no_managed_processes(self) -> None:
        process_ids = self._managed_process_ids()
        if process_ids:
            raise LocalAIManagerError(
                f"状态文件缺失但仍发现受管 LocalAI/backend 进程：{process_ids}。"
            )

    def _read_state(self, *, required: bool) -> dict[str, object] | None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if required:
                raise LocalAIManagerError("LocalAI 进程状态文件不存在。") from None
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError("LocalAI 状态文件损坏；保留隔离并人工检查。") from exc
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise LocalAIManagerError("LocalAI 状态文件格式无效；保留隔离并人工检查。")
        try:
            pid = int(state["pid"])
            pgid = int(state["pgid"])
            boot_id = state["boot_id"]
            start_time = state["process_start_time"]
            base_url = state["base_url"]
            run_id = state["run_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LocalAIManagerError("LocalAI 状态文件缺少进程身份字段。") from exc
        if (
            pid <= 1
            or pgid != pid
            or not isinstance(boot_id, str)
            or not re.fullmatch(r"[0-9a-f-]{36}", boot_id)
            or not isinstance(start_time, str)
        ):
            raise LocalAIManagerError("LocalAI 状态文件中的进程身份无效。")
        if not isinstance(base_url, str) or not isinstance(run_id, str):
            raise LocalAIManagerError("LocalAI 状态文件中的地址或运行标识无效。")
        expected_url, _ = self._base_url_and_port()
        if base_url != expected_url or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise LocalAIManagerError("LocalAI 状态文件与当前实例配置不匹配。")
        return state

    def _terminate_process_group(self, state: Mapping[str, object]) -> None:
        """按 PID 启动时间和运行标识验证进程组，再执行 TERM/KILL 回收。"""
        if state.get("boot_id") not in {None, self._boot_id()}:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            return
        pid = int(state["pid"])
        pgid = int(state["pgid"])
        start_time = str(state["process_start_time"])
        run_id = str(state["run_id"])
        current_start = self._process_start_time(pid)
        if current_start is not None and current_start != start_time:
            raise LocalAIManagerError("LocalAI PID 已被复用；拒绝向该进程组发送信号。")
        if self._process_group_alive(pgid):
            if current_start != start_time and not self._group_has_run_id(pgid, run_id):
                raise LocalAIManagerError("无法确认遗留进程组归属；保留状态供人工恢复。")
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._wait_for_process_group(pgid, self.stop_timeout)
            if self._process_group_alive(pgid):
                if not self._group_has_run_id(pgid, run_id) and current_start != start_time:
                    raise LocalAIManagerError("进程组身份无法复核；不发送 SIGKILL。")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._wait_for_process_group(pgid, self.stop_timeout)
        if self._process_group_alive(pgid):
            raise LocalAIManagerError("LocalAI 进程组在 SIGKILL 后仍有活动进程。")
        # 进程组消失与内核关闭监听之间存在短暂间隔，限时复核后再报告失败。
        self._ensure_port_closed(timeout=min(self.stop_timeout, 2.0))
        self._reap_spawned_child(pid)

    def _wait_for_process_group(self, pgid: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _read_process_stat(pid: int) -> tuple[str, int, str] | None:
        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        closing_parenthesis = raw_stat.rfind(")")
        if closing_parenthesis < 0:
            return None
        fields = raw_stat[closing_parenthesis + 1 :].split()
        if len(fields) <= 19:
            return None
        return fields[0], int(fields[2]), fields[19]

    @staticmethod
    def _boot_id() -> str:
        """返回 Linux boot ID，避免重启后 PID/启动 tick 复用旧运行状态。"""
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        except OSError as exc:
            raise LocalAIManagerError(
                "无法读取 Linux boot ID；拒绝管理无法识别的进程状态。"
            ) from exc

    @classmethod
    def _process_start_time(cls, process_id: int) -> str | None:
        """返回 Linux 进程启动 tick；已退出或 zombie 时返回 None。"""
        stat = cls._read_process_stat(process_id)
        if stat is None or stat[0] in {"Z", "X"}:
            return None
        return stat[2]

    @classmethod
    def _process_group_alive(cls, pgid: int) -> bool:
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = cls._read_process_stat(process_id)
            if stat is not None and stat[0] not in {"Z", "X"} and stat[1] == pgid:
                return True
        return False

    @classmethod
    def _group_has_run_id(cls, pgid: int, run_id: str) -> bool:
        expected = f"LLM_LOCALAI_RUN_ID={run_id}".encode()
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = cls._read_process_stat(process_id)
            if stat is None or stat[0] in {"Z", "X"} or stat[1] != pgid:
                continue
            try:
                if expected in (process_path / "environ").read_bytes().split(b"\0"):
                    return True
            except OSError:
                continue
        return False

    def _start_owner_watcher(
        self, owner_pid: int, owner_start_time: str, state: Mapping[str, object]
    ) -> None:
        """启动独立 watcher，在 start.sh 异常退出时回收宿主机进程组。"""
        child_environment = {**os.environ, **self.environ}
        child_environment.pop("LLM_LOCALAI_API_KEY", None)
        child_environment.pop("LOCALAI_API_KEY", None)
        child_environment.pop("LLM_LOCALAI_OWNER_PID", None)
        try:
            self._spawn_detached(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "watch",
                    str(owner_pid),
                    owner_start_time,
                    str(state["pid"]),
                    str(state["process_start_time"]),
                ],
                child_environment,
            )
        except OSError as exc:
            self._terminate_process_group(state)
            self.state_path.unlink(missing_ok=True)
            raise LocalAIManagerError(f"无法启动 LocalAI 所有者 watcher：{exc}") from exc

    def watch_owner(
        self, owner_pid: int, owner_start_time: str, process_pid: int, process_start_time: str
    ) -> None:
        """监控启动脚本所有者；所有者退出后回收所记录的 LocalAI 进程组。"""
        while True:
            state = self._read_state(required=False)
            if state is None:
                return
            if (
                int(state["pid"]) != process_pid
                or str(state["process_start_time"]) != process_start_time
            ):
                return
            if self._process_start_time(process_pid) is None and not self._process_group_alive(
                process_pid
            ):
                return
            if self._process_start_time(owner_pid) != owner_start_time:
                self._terminate_process_group(state)
                self.state_path.unlink(missing_ok=True)
                return
            time.sleep(1)

    def _wait_until_ready(self, base_url: str) -> None:
        """以有界轮询确认 LocalAI HTTP API 已开始服务。"""
        deadline = time.monotonic() + self.readiness_timeout
        last_error: Exception | None = None
        health_url = f"{base_url}/readyz"
        headers: dict[str, str] = {}
        if api_key := self.environ.get("LLM_LOCALAI_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"
        while True:
            state = self._read_state(required=False)
            if state is None or self._process_start_time(int(state["pid"])) is None:
                raise LocalAIManagerError("LocalAI 在 readiness 成功前退出。") from last_error
            try:
                with urlopen(Request(health_url, headers=headers), timeout=1.0) as response:
                    if 200 <= response.status < 300:
                        return
                    last_error = LocalAIManagerError(
                        f"LocalAI readiness 返回 HTTP {response.status}。"
                    )
            except (OSError, URLError, TimeoutError) as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise LocalAIManagerError(
                    f"等待 LocalAI 就绪超时（{self.readiness_timeout:g}s）。"
                ) from last_error
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """按固定块大小校验大文件，避免把权重或二进制整体读入内存。"""
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            while chunk := artifact.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _sha256_tree(cls, root: Path) -> str:
        """对 backend 目录内容做稳定 hash，并拒绝越界软链接和特殊文件。"""
        resolved_root = root.resolve(strict=True)
        entries: list[dict[str, object]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                try:
                    resolved_path = path.resolve(strict=True)
                except OSError as exc:
                    raise LocalAIManagerError(
                        f"audio-cpp backend 存在失效软链接：{relative_path}"
                    ) from exc
                if not resolved_path.is_relative_to(resolved_root):
                    raise LocalAIManagerError(f"audio-cpp backend 软链接越界：{relative_path}")
                if resolved_path.is_file():
                    entries.append(
                        {
                            "path": relative_path,
                            "kind": "symlink-file",
                            "target": os.readlink(path),
                            "size": resolved_path.stat().st_size,
                            "sha256": cls._sha256_file(resolved_path),
                        }
                    )
                elif resolved_path.is_dir():
                    entries.append(
                        {"path": relative_path, "kind": "symlink-dir", "target": os.readlink(path)}
                    )
                else:
                    raise LocalAIManagerError(
                        f"audio-cpp backend 软链接指向特殊文件：{relative_path}"
                    )
            elif path.is_dir():
                entries.append({"path": relative_path, "kind": "dir"})
            elif path.is_file():
                entries.append(
                    {
                        "path": relative_path,
                        "kind": "file",
                        "mode": path.stat().st_mode & 0o777,
                        "size": path.stat().st_size,
                        "sha256": cls._sha256_file(path),
                    }
                )
            else:
                raise LocalAIManagerError(f"audio-cpp backend 包含特殊文件：{relative_path}")
        serialized = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_temporary_file(directory: Path, prefix: str, suffix: str) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
        os.close(descriptor)
        return Path(temporary_name)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _atomic_write_json(cls, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
            )
            cls._fsync_file(temporary_path)
            os.replace(temporary_path, path)
            cls._fsync_directory(path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _enabled(self) -> bool:
        value = self.environ.get("LLM_LOCALAI_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}


def main() -> int:
    """解析固定版本校验、启停与状态子命令。"""
    parser = argparse.ArgumentParser(description="管理 LLM Local Backend 裸机 LocalAI 实例")
    parser.add_argument(
        "command",
        choices=(
            "verify-install",
            "register-audio-cpp-candidates",
            "verify-audio-cpp-candidates",
            "start",
            "stop",
            "status",
            "watch",
        ),
    )
    parser.add_argument("watch_args", nargs="*")
    arguments = parser.parse_args()
    try:
        manager = LocalAIManager()
        if arguments.command == "watch":
            if len(arguments.watch_args) != 4:
                raise LocalAIManagerError("watch 参数格式无效。")
            manager.watch_owner(
                int(arguments.watch_args[0]),
                arguments.watch_args[1],
                int(arguments.watch_args[2]),
                arguments.watch_args[3],
            )
        elif arguments.command == "register-audio-cpp-candidates":
            manager.register_audio_cpp_candidates()
        elif arguments.command == "verify-audio-cpp-candidates":
            manager.verify_audio_cpp_candidates()
        elif arguments.command == "verify-install":
            manager.verify_install()
        else:
            getattr(manager, arguments.command)()
    except LocalAIManagerError as exc:
        print(f"LocalAI 管理失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
