#!/usr/bin/env python3
"""确认 GPU backend 停止后解除共享 GPU 隔离。"""

from __future__ import annotations

import fcntl
import ipaddress
import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference_gateway/src"))
sys.path.insert(0, str(ROOT / "llm_local_runtime/src"))
sys.path.insert(0, str(ROOT / "model-text"))

from inference_gateway.sdcpp.processes import recover_sdcpp  # noqa: E402
from llama_cpp_runtime import runtime as llama_runtime  # noqa: E402

from inference_gateway import LocalAIClient, load_gateway_config  # noqa: E402
from llm_local_runtime import gpu_quarantine_path  # noqa: E402
from scripts.localai.manager import LocalAIManager, LocalAIManagerError  # noqa: E402

_loopback_port_open = llama_runtime.loopback_port_open
process_start_time = llama_runtime.process_start_time


def _environment_lock_path() -> Path:
    """按启动脚本环境定位共享 GPU 锁文件。"""
    configured = os.getenv("GPU_LOCK_FILE")
    if configured:
        return Path(configured)
    runtime_cache = os.getenv("RUNTIME_CACHE_DIR")
    if runtime_cache:
        return Path(runtime_cache) / "gpu-runtime.lock"
    return Path("storage/.cache/runtime/gpu-runtime.lock")


def _sync_marker_directory(marker_path: Path) -> None:
    """同步隔离标记所在目录。"""
    descriptor = os.open(marker_path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_group_exists(pgid: int) -> bool:
    """检查 backend 进程组是否仍存在。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_command_line(pid: int) -> list[str] | None:
    """读取进程参数；进程已经退出时返回空值。"""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [argument.decode(errors="replace") for argument in raw.split(b"\0") if argument]


def _validate_llama_cpp_marker(
    marker: dict[str, object],
) -> tuple[str, int, int | None, int | None]:
    """检查 llama.cpp 标记中的 backend 地址和进程身份。"""
    host = marker.get("host")
    port = marker.get("port")
    pid = marker.get("pid")
    pgid = marker.get("pgid")
    if not isinstance(host, str):
        raise ValueError("llama.cpp 隔离标记缺少 host。")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("llama.cpp 隔离标记 host 不是 loopback。")
        except ValueError as exc:
            raise ValueError("llama.cpp 隔离标记 host 不是 loopback。") from exc
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("llama.cpp 隔离标记缺少有效 loopback 端口。")
    if pid is not None and (not isinstance(pid, int) or pid <= 0):
        raise ValueError("llama.cpp 隔离标记 pid 无效。")
    if pgid is not None and (not isinstance(pgid, int) or pgid <= 0):
        raise ValueError("llama.cpp 隔离标记 pgid 无效。")
    if (pid is None) != (pgid is None):
        raise ValueError("llama.cpp 隔离标记必须同时包含 pid 和 pgid，或二者都为空。")
    if pid is not None and pgid != pid:
        raise ValueError("llama.cpp backend 进程组与 pid 不一致，保留隔离状态。")
    return host, port, pid, pgid


def _recover_llama_cpp_target(marker: dict[str, object]) -> int:
    """停止一个 llama.cpp backend，并确认进程组与 loopback API 均已退出。"""
    try:
        host, port, pid, pgid = _validate_llama_cpp_marker(marker)
        timeout = float(os.getenv("LLM_LOCALAI_CLEANUP_TIMEOUT", "30"))
        if timeout <= 0:
            raise ValueError("LLM_LOCALAI_CLEANUP_TIMEOUT 必须为正数。")
        original_start_time = marker.get("pid_start_time")
        if pid is not None:
            current_start_time = process_start_time(pid)
            if current_start_time is not None and original_start_time != current_start_time:
                pid = None
                pgid = None
            elif current_start_time is not None:
                command_line = _process_command_line(pid)
                executable = marker.get("executable")
                port_argument = ["--port", str(port)]
                if (
                    not isinstance(executable, str)
                    or command_line is None
                    or not command_line
                    or not any(
                        argument.startswith("/")
                        and Path(argument).resolve() == Path(executable).resolve()
                        for argument in command_line[:2]
                    )
                    or not any(
                        command_line[index : index + 2] == port_argument
                        for index in range(len(command_line) - 1)
                    )
                ):
                    print(
                        "隔离标记中的 PID 与 llama-server 命令不匹配，保留隔离状态。",
                        file=sys.stderr,
                    )
                    return 1
            elif original_start_time is not None and _process_group_exists(pgid or 0):
                print("无法核对 llama.cpp backend 的进程启动时间，保留隔离状态。", file=sys.stderr)
                return 1

        if pgid is not None and _process_group_exists(pgid):
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + min(timeout, 5.0)
            while time.monotonic() < deadline and _process_group_exists(pgid):
                time.sleep(0.1)
            if _process_group_exists(pgid):
                os.killpg(pgid, signal.SIGKILL)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            group_stopped = pgid is None or not _process_group_exists(pgid)
            api_stopped = not _loopback_port_open(host, port)
            if group_stopped and api_stopped:
                print(f"llama.cpp 进程组和 loopback API 已退出：{host}:{port}")
                return 0
            time.sleep(0.2)
        print(
            "llama.cpp backend 进程组或 loopback API 仍活动，保留 GPU 隔离标记。",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"llama.cpp backend 恢复失败，保留隔离标记：{exc}", file=sys.stderr)
        return 1


def _recover_llama_cpp_targets(targets: object) -> int:
    """按 marker 中的完整列表停止并确认每个 llama.cpp backend。"""
    if not isinstance(targets, list) or not targets:
        print("llama.cpp 隔离标记缺少有效 backend 列表。", file=sys.stderr)
        return 1
    for target in targets:
        if not isinstance(target, dict) or _recover_llama_cpp_target(target) != 0:
            return 1
    return 0


def _recover_llama_cpp(marker_path: Path, marker: dict[str, object]) -> int:
    """停止 marker 中所有 llama.cpp backend 后再解除共享 GPU 隔离。"""
    targets = marker.get("backends")
    if targets is None:
        targets = [marker]
    if _recover_llama_cpp_targets(targets) != 0:
        return 1
    marker_path.unlink()
    _sync_marker_directory(marker_path)
    print("所有 llama.cpp 进程组和 loopback API 已退出，GPU 隔离已解除。")
    return 0


def _recover_localai(marker_path: Path, marker: dict[str, object]) -> int:
    """保留原有 LocalAI backend 恢复流程。"""
    config = load_gateway_config(
        service_provider_env="LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER",
        default_model="llm-local.qwen3-tts.1p7b-voicedesign",
        request_timeout=float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900")),
    )
    if not config.localai_enabled:
        print("恢复 LocalAI 前需设置 LLM_LOCALAI_ENABLED=true。", file=sys.stderr)
        return 2
    model = marker.get("model")
    if not isinstance(model, str) or not model:
        print("LocalAI 隔离标记缺少有效 model，保留隔离状态。", file=sys.stderr)
        return 1
    client = LocalAIClient(replace(config, model=model))
    shutdown_error: Exception | None = None
    try:
        client.shutdown_model()
    except Exception as exc:
        shutdown_error = exc
    if shutdown_error is not None:
        managed = os.getenv("LLM_LOCALAI_MANAGED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not managed:
            print(f"LocalAI 未确认停止 {model}：{shutdown_error}", file=sys.stderr)
            return 1
        try:
            manager = LocalAIManager()
            manager.stop()
            manager.verify_managed_localai_stopped()
        except LocalAIManagerError as verification_error:
            print(
                f"LocalAI backend 未确认停止 {model}：{shutdown_error}；进程组退出复核失败："
                f"{verification_error}",
                file=sys.stderr,
            )
            return 1
        print("已停止受管 LocalAI 进程组并确认 loopback API 端口关闭。")
    marker_path.unlink()
    _sync_marker_directory(marker_path)
    print(f"LocalAI backend 已确认停止，GPU 隔离已解除：{model}")
    return 0


def main() -> int:
    """持有共享 GPU 锁，按 marker provider 恢复 backend 并清除标记。"""
    lock_path = _environment_lock_path()
    marker_path = gpu_quarantine_path(lock_path)
    if not marker_path.is_file():
        print("没有待恢复的 GPU 隔离标记。")
        return 0
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU 仍有活动请求，暂不解除隔离。", file=sys.stderr)
            return 1
        try:
            # 必须在持锁后重读；等待锁期间旧请求可能已更新或删除标记。
            if not marker_path.exists():
                print("没有待恢复的 GPU 隔离标记。")
                return 0
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"隔离标记无法读取：{exc}", file=sys.stderr)
                return 1
            if not isinstance(marker, dict):
                print("隔离标记格式无效，保留隔离状态。", file=sys.stderr)
                return 1
            provider = marker.get("provider", "localai")
            if provider != "llama.cpp" and "llama_cpp_backends" in marker:
                if _recover_llama_cpp_targets(marker["llama_cpp_backends"]) != 0:
                    return 1
            if provider == "stable-diffusion.cpp":
                try:
                    recover_sdcpp(
                        marker_path, marker, float(os.getenv("SDCPP_CLEANUP_TIMEOUT", "30"))
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"sd.cpp 恢复失败，保留隔离状态：{exc}", file=sys.stderr)
                    return 1
                print("sd.cpp 进程组已确认停止，GPU 隔离已解除。")
                return 0
            if provider == "llama.cpp":
                return _recover_llama_cpp(marker_path, marker)
            if provider != "localai":
                print(f"隔离标记 provider 不受支持：{provider}。", file=sys.stderr)
                return 1
            return _recover_localai(marker_path, marker)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
