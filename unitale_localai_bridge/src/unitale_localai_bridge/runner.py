"""受 manifest 白名单约束的一次性 worker 启动与结果封装。"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_RESULT_BYTES = 128 * 1024 * 1024
MAX_SHARED_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 8 * 1024


class WorkerExecutionError(RuntimeError):
    """一次性模型 worker 失败或未能确认退出。"""


@dataclass(frozen=True)
class WorkerSpec:
    """固定 worker 入口、CLI 参数和产物类型。"""

    project: Path
    script: Path
    output_kind: str
    timeout_seconds: float
    output_argument: str | None = None
    output_suffix: str | None = None
    output_suffix_field: str | None = None
    output_suffixes: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)


class WorkerRunner:
    """通过每请求子进程调用现有服务 worker，不在 adapter 导入模型库。"""

    def __init__(
        self,
        repository_root: Path,
        specs: dict[str, WorkerSpec],
        *,
        uv_executable: str = "uv",
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.specs = specs
        self.uv_executable = uv_executable

    def run(
        self,
        model: str,
        operation: str,
        payload: dict[str, Any],
        *,
        request_id: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """验证请求并运行一个模型项目的固定一次性 worker。"""
        spec = self.specs.get(model)
        if spec is None:
            raise WorkerExecutionError("该 LocalAI bridge model 未登记 worker。")
        if operation not in {"run", "infer"}:
            raise WorkerExecutionError("该 LocalAI bridge operation 不受支持。")
        worker_input = payload.get("worker_input")
        if not isinstance(worker_input, dict):
            raise WorkerExecutionError("worker_input 必须是 JSON 对象。")
        if spec.output_kind not in {"json", "file", "file_ref"}:
            raise WorkerExecutionError("worker manifest 的 output_kind 无效。")

        output_suffix = spec.output_suffix
        if spec.output_suffix_field is not None:
            selected_suffix = worker_input.get(spec.output_suffix_field)
            output_suffix = spec.output_suffixes.get(str(selected_suffix))
            if output_suffix is None:
                raise WorkerExecutionError("worker 请求没有匹配到 manifest 输出扩展名。")
        if output_suffix is None:
            output_suffix = ".bin"

        with tempfile.TemporaryDirectory(prefix="unitale-localai-worker-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.json"
            output_path = temp_path / f"output{output_suffix}"
            input_path.write_text(json.dumps(worker_input, ensure_ascii=False), encoding="utf-8")
            command = [
                self.uv_executable,
                "run",
                "--project",
                str(spec.project),
                "--no-sync",
                "python",
                str(spec.script),
                "--input-json",
                str(input_path),
            ]
            if spec.output_argument:
                command.extend([spec.output_argument, str(output_path)])
            worker_environment = os.environ.copy()
            for key in (
                "RANK",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "LOCAL_WORLD_SIZE",
            ):
                worker_environment.pop(key, None)
            worker_environment["WORLD_SIZE"] = "1"
            worker_environment["UNITALE_LOCALAI_REQUEST_ID"] = request_id
            storage_dir = Path(os.getenv("STORAGE_DIR", self.repository_root / "storage"))
            worker_environment.setdefault(
                "RUNTIME_CACHE_DIR",
                os.getenv("RUNTIME_CACHE_DIR", str(storage_dir / ".cache/runtime")),
            )
            if worker_input.get("local_files_only", True):
                worker_environment["HF_HUB_OFFLINE"] = "1"
                worker_environment["TRANSFORMERS_OFFLINE"] = "1"
            if hf_mirror_dir := worker_input.get("hf_mirror_dir"):
                worker_environment.setdefault("HF_HOME", str(hf_mirror_dir))
            for name, value in spec.environment.items():
                worker_environment[name] = os.path.expandvars(value)
            result = self._run_process(
                command,
                cwd=spec.project,
                timeout_seconds=spec.timeout_seconds,
                cancelled=cancelled,
                request_id=request_id,
                environment=worker_environment,
            )
            if spec.output_kind == "json":
                if not output_path.is_file():
                    raise WorkerExecutionError("worker 未生成要求的 JSON 结果文件。")
                if output_path.stat().st_size > MAX_RESULT_BYTES:
                    raise WorkerExecutionError("worker JSON 结果超过大小限制。")
                try:
                    decoded = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise WorkerExecutionError("worker JSON 结果无效。") from exc
                if not isinstance(decoded, dict):
                    raise WorkerExecutionError("worker JSON 结果必须是对象。")
                return {"kind": "json", "value": decoded, "elapsed_seconds": result}
            if not output_path.is_file():
                raise WorkerExecutionError("worker 未生成要求的二进制产物。")
            output_size = output_path.stat().st_size
            if spec.output_kind == "file_ref":
                return self._publish_file_reference(
                    output_path,
                    request_id=request_id,
                    suffix=output_suffix,
                    media_type=str(payload.get("media_type") or "application/octet-stream"),
                    size_bytes=output_size,
                )
            if output_size <= 0 or output_size > MAX_RESULT_BYTES:
                raise WorkerExecutionError("worker 二进制结果为空或超过大小限制。")
            content = base64.b64encode(output_path.read_bytes()).decode("ascii")
            return {
                "kind": "file",
                "media_type": str(payload.get("media_type") or "application/octet-stream"),
                "data_base64": content,
                "size_bytes": output_size,
                "elapsed_seconds": result,
            }

    @staticmethod
    def _publish_file_reference(
        source: Path,
        *,
        request_id: str,
        suffix: str,
        media_type: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        """将大 artifact 原子发布到受控共享目录，RPC 仅返回不透明 ID。"""
        if not re.fullmatch(r"[a-f0-9]{32}", request_id):
            raise WorkerExecutionError("file_ref 请求 ID 必须是 32 位随机十六进制值。")
        if size_bytes <= 0 or size_bytes > MAX_SHARED_ARTIFACT_BYTES:
            raise WorkerExecutionError("共享 artifact 为空或超过 4 GiB 上限。")
        cache_dir = Path(
            os.getenv(
                "RUNTIME_CACHE_DIR",
                str(Path(os.getenv("STORAGE_DIR", Path.cwd() / "storage")) / ".cache/runtime"),
            )
        ).expanduser()
        configured_root = Path(
            os.getenv("UNITALE_LOCALAI_ARTIFACT_DIR", str(cache_dir / "localai_artifacts"))
        ).expanduser()
        if configured_root.exists() and configured_root.is_symlink():
            raise WorkerExecutionError("共享 artifact 根目录不能是符号链接。")
        root = configured_root.resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        now = time.time()
        for stale in root.iterdir():
            if (
                stale.is_file()
                and re.fullmatch(r"[a-f0-9]{32}\.(zip|json|npz|npy|wav|mp4)", stale.name)
                and now - stale.stat().st_mtime > 24 * 60 * 60
            ):
                stale.unlink(missing_ok=True)
        destination = root / f"{request_id}{suffix}"
        if destination.exists():
            raise WorkerExecutionError("共享 artifact 请求 ID 已存在。")
        temporary = root / f".{request_id}.{os.getpid()}.partial"
        try:
            with source.open("rb") as input_file, temporary.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "kind": "file_ref",
            "artifact_id": request_id,
            "suffix": suffix,
            "media_type": media_type,
            "size_bytes": size_bytes,
        }

    @staticmethod
    def _run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        cancelled: Callable[[], bool] | None,
        request_id: str,
        environment: dict[str, str],
    ) -> float:
        started = time.monotonic()
        with tempfile.TemporaryFile() as diagnostic:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=diagnostic,
                    start_new_session=True,
                    close_fds=True,
                    env=environment,
                )
            except OSError as exc:
                raise WorkerExecutionError("无法启动固定的一次性 worker。") from exc

            deadline = started + timeout_seconds
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if cancelled is not None and not cancelled():
                    _stop_process_group(process, request_id)
                    raise WorkerExecutionError("LocalAI 请求已取消，worker 进程组已停止。")
                if time.monotonic() >= deadline:
                    _stop_process_group(process, request_id)
                    raise WorkerExecutionError("一次性 worker 超时，进程组已停止。")
                time.sleep(0.05)
            elapsed = time.monotonic() - started
            if return_code != 0:
                raise WorkerExecutionError(
                    f"一次性 worker 退出码 {return_code}；请求 {request_id}。"
                )
            return elapsed


def _stop_process_group(process: subprocess.Popen[bytes], request_id: str) -> None:
    """发送 TERM 后限时 KILL，并在无法确认回收时写入 GPU 隔离标记。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as exc:
        _write_quarantine(request_id)
        raise WorkerExecutionError("worker 在 TERM/KILL 后仍无法确认退出；GPU 已隔离。") from exc


def _write_quarantine(request_id: str) -> None:
    """写入与共享 GPU lock 兼容的隔离标记，要求人工确认后恢复。"""
    from unitale_runtime import gpu_quarantine_path

    lock_path = Path(os.getenv("GPU_LOCK_FILE", "storage/.cache/runtime/gpu-runtime.lock"))
    marker = gpu_quarantine_path(lock_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{request_id}.tmp")
    payload = {
        "schema_version": 1,
        "model": "unitale.localai.worker-bridge",
        "request_id": request_id,
        "reason": "worker_process_exit_unconfirmed",
    }
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, marker)
