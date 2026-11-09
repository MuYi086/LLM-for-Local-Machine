"""管理 llama.cpp 后端的隔离标记和休眠 readiness。"""

from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from llm_local_runtime import gpu_runtime_lock

PROVIDER = "llama.cpp"
MODEL_ALIAS = "Ornith-1.5-35B-A3B-AD-Q5"


def _sync_directory(path: Path) -> None:
    """同步目录项更新，确保原子标记写入可恢复。"""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_start_time(pid: int | None) -> str | None:
    """读取 Linux `/proc` 进程启动 tick，防止恢复时误杀 PID 复用进程。"""
    if pid is None:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def create_quarantine_marker(
    lock_path: str | Path,
    *,
    host: str,
    port: int,
    pid: int | None,
    pgid: int | None,
    executable: str | None,
    model_alias: str = MODEL_ALIAS,
) -> bytes:
    """原子创建一次 llama.cpp GPU 使用隔离标记。"""
    marker_path = Path(f"{lock_path}.quarantine")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "provider": PROVIDER,
        "request_id": uuid.uuid4().hex,
        "model": model_alias,
        "started_at": time.time(),
        "host": host,
        "port": port,
        "pid": pid,
        "pgid": pgid,
        "pid_start_time": process_start_time(pid),
        "executable": executable,
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".ornith-quarantine-", dir=marker_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as marker_file:
            marker_file.write(content)
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.chmod(temporary_path, 0o600)
        if marker_path.exists():
            raise FileExistsError(marker_path)
        os.link(temporary_path, marker_path)
        _sync_directory(marker_path.parent)
        return content
    finally:
        temporary_path.unlink(missing_ok=True)


def update_backend_identity(
    lock_path: str | Path,
    *,
    pid: int,
    pgid: int,
    executable: str,
) -> None:
    """将已经启动的 backend 进程身份原子写入现有标记。"""
    marker_path = Path(f"{lock_path}.quarantine")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or marker.get("provider") != PROVIDER:
        raise ValueError("llama.cpp 隔离标记不存在或 provider 不匹配。")
    marker.update(
        {
            "pid": pid,
            "pgid": pgid,
            "pid_start_time": process_start_time(pid),
            "executable": executable,
        }
    )
    _replace_marker(marker_path, marker)


def _replace_marker(marker_path: Path, payload: dict[str, object]) -> None:
    """原子替换隔离标记并同步目录。"""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".llama-cpp-quarantine-", dir=marker_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as marker_file:
            json.dump(payload, marker_file, ensure_ascii=False, sort_keys=True)
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, marker_path)
        _sync_directory(marker_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def remove_quarantine_marker(
    lock_path: str | Path,
    *,
    expected_content: bytes | None = None,
) -> bool:
    """移除确认仍属于当前请求的 llama.cpp 隔离标记。"""
    marker_path = Path(f"{lock_path}.quarantine")
    try:
        current_content = marker_path.read_bytes()
    except FileNotFoundError:
        return False
    if expected_content is not None and current_content != expected_content:
        return False
    marker = json.loads(current_content)
    if not isinstance(marker, dict) or marker.get("provider") != PROVIDER:
        return False
    marker_path.unlink()
    _sync_directory(marker_path.parent)
    return True


def wait_for_sleeping(
    base_url: str,
    *,
    timeout_seconds: float,
    poll_interval: float,
    pid: int | None = None,
) -> None:
    """等待 llama-server 的 `/props` 确认模型已卸载。"""
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("llama.cpp backend 地址必须是 loopback HTTP URL。")
    if parsed.hostname != "localhost":
        try:
            if not ipaddress.ip_address(parsed.hostname).is_loopback:
                raise ValueError("llama.cpp backend 地址必须是 loopback HTTP URL。")
        except ValueError as exc:
            raise ValueError("llama.cpp backend 地址必须是 loopback HTTP URL。") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("llama.cpp backend URL 不得包含凭据、query 或 fragment。")

    deadline = time.monotonic() + timeout_seconds
    props_url = f"{base_url.rstrip('/')}/props"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = "llama-server 尚未报告 sleeping 状态。"
    while time.monotonic() < deadline:
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError("llama-server 在确认模型休眠前退出。") from exc
            except PermissionError:
                pass
        try:
            with opener.open(props_url, timeout=5) as response:
                if response.status == 200:
                    payload = json.loads(response.read())
                    if isinstance(payload, dict) and payload.get("is_sleeping") is True:
                        return
                    last_error = "llama-server 模型仍处于加载或运行状态。"
                else:
                    last_error = f"llama-server /props 返回 HTTP {response.status}。"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"无法读取 llama-server /props：{exc}"
        time.sleep(poll_interval)
    raise TimeoutError(f"等待 llama.cpp 模型进入 sleep-idle 超时：{last_error}")


def _pid_exists(pid: int) -> bool:
    """检查当前启动协调进程是否仍存在。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def loopback_port_open(host: str, port: int) -> bool:
    """检测 llama-server 的 loopback 地址和端口是否仍被占用。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return True
        return True
    else:
        return False
    finally:
        probe.close()


def _process_group_exists(pgid: int) -> bool:
    """检查 llama-server 进程组是否还存在。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _command_matches_backend(pid: int, executable: str, port: int) -> bool:
    """确认 PID 当前仍是本服务启动的 llama-server。"""
    try:
        command_line = [
            value.decode(errors="replace")
            for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        return False
    return (
        bool(command_line)
        and any(
            argument.startswith("/") and Path(argument).resolve() == Path(executable).resolve()
            for argument in command_line[:2]
        )
        and any(
            command_line[index : index + 2] == ["--port", str(port)]
            for index in range(len(command_line) - 1)
        )
    )


def _write_pid_file(path: Path, pid: int) -> None:
    """原子写入已经确认进入 sleep-idle 的 backend PID。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".llama-cpp-pid-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as pid_file:
            pid_file.write(f"{pid}\n")
            pid_file.flush()
            os.fsync(pid_file.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _stop_failed_start(
    process: subprocess.Popen[bytes],
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> bool:
    """启动 readiness 失败时停止进程组并确认 API 端口关闭。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=min(timeout_seconds, 5.0))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=min(timeout_seconds, 5.0))
        except subprocess.TimeoutExpired:
            return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
            group_exists = True
        except ProcessLookupError:
            group_exists = False
        except PermissionError:
            group_exists = True
        if not group_exists and not loopback_port_open(host, port):
            return True
        time.sleep(0.1)
    return False


def stop_backend(
    *,
    lock_path: str | Path,
    host: str,
    port: int,
    pid: int,
    pgid: int,
    executable: str,
    lock_timeout: float,
    cleanup_timeout: float,
    cuda_release_delay: float,
    model_alias: str = MODEL_ALIAS,
) -> None:
    """在共享 GPU 租约内停止 backend 并核实进程组和端口退出。"""
    executable = str(Path(executable).resolve())
    with gpu_runtime_lock(
        lock_path,
        f"llama.cpp/shutdown/{model_alias}",
        timeout_seconds=lock_timeout,
    ):
        marker_content = create_quarantine_marker(
            lock_path,
            host=host,
            port=port,
            pid=pid,
            pgid=pgid,
            executable=executable,
            model_alias=model_alias,
        )
        start_time = process_start_time(pid)
        if start_time is not None and not _command_matches_backend(pid, executable, port):
            raise RuntimeError("llama.cpp backend PID 与可执行文件或端口不匹配，保留 quarantine。")
        if start_time is not None and _process_group_exists(pgid):
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + min(cleanup_timeout, 5.0)
            while time.monotonic() < deadline and _process_group_exists(pgid):
                time.sleep(0.1)
            if _process_group_exists(pgid):
                os.killpg(pgid, signal.SIGKILL)

        deadline = time.monotonic() + cleanup_timeout
        while time.monotonic() < deadline:
            if not _process_group_exists(pgid) and not loopback_port_open(host, port):
                if cuda_release_delay:
                    time.sleep(cuda_release_delay)
                if not _process_group_exists(pgid) and not loopback_port_open(host, port):
                    if remove_quarantine_marker(lock_path, expected_content=marker_content):
                        return
                    raise RuntimeError("llama.cpp backend 已停止，但隔离标记发生变化。")
            time.sleep(0.1)
        raise TimeoutError("llama.cpp 进程组或 loopback API 未确认退出，保留 quarantine。")


def start_backend(
    *,
    lock_path: str | Path,
    pid_file: str | Path,
    host: str,
    port: int,
    executable: str,
    backend_args: list[str],
    base_url: str,
    lock_timeout: float,
    startup_timeout: float,
    poll_interval: float,
    cuda_release_delay: float,
    model_alias: str = MODEL_ALIAS,
) -> int:
    """在共享 GPU 租约中启动 backend 并等到初次模型卸载。"""
    executable = str(Path(executable).resolve())
    with gpu_runtime_lock(
        lock_path,
        f"llama.cpp/startup/{model_alias}",
        timeout_seconds=lock_timeout,
    ):
        marker_content = create_quarantine_marker(
            lock_path,
            host=host,
            port=port,
            pid=None,
            pgid=None,
            executable=executable,
            model_alias=model_alias,
        )
        gate_descriptor, gate_name = tempfile.mkstemp(
            prefix="llama-cpp-start-gate-", dir=Path(lock_path).parent
        )
        os.close(gate_descriptor)
        gate_path = Path(gate_name)
        gate_path.unlink()
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ornith_1_5_35b_a3b.runtime",
                    "exec-backend",
                    "--gate",
                    str(gate_path),
                    "--parent-pid",
                    str(os.getpid()),
                    "--executable",
                    executable,
                    "--",
                    *backend_args,
                ],
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            update_backend_identity(
                lock_path,
                pid=process.pid,
                pgid=process.pid,
                executable=executable,
            )
            marker_content = Path(f"{lock_path}.quarantine").read_bytes()
            gate_path.touch(mode=0o600)
            wait_for_sleeping(
                base_url,
                timeout_seconds=startup_timeout,
                poll_interval=poll_interval,
                pid=process.pid,
            )
            if cuda_release_delay:
                time.sleep(cuda_release_delay)
            wait_for_sleeping(
                base_url,
                timeout_seconds=max(10.0, poll_interval * 2),
                poll_interval=poll_interval,
                pid=process.pid,
            )
            _write_pid_file(Path(pid_file), process.pid)
            if not remove_quarantine_marker(lock_path, expected_content=marker_content):
                raise RuntimeError("llama.cpp 启动期间隔离标记发生变化。")
            return process.pid
        except Exception:
            if process is None:
                remove_quarantine_marker(lock_path, expected_content=marker_content)
            else:
                stopped = _stop_failed_start(
                    process,
                    host=host,
                    port=port,
                    timeout_seconds=10.0,
                )
                if stopped:
                    remove_quarantine_marker(lock_path)
            raise
        finally:
            gate_path.unlink(missing_ok=True)


def _exec_backend(gate_path: Path, parent_pid: int, executable: str, args: list[str]) -> int:
    """等待父进程写入身份标记后以 llama-server 替换启动包装进程。"""
    deadline = time.monotonic() + 60.0
    while not gate_path.exists():
        if not _pid_exists(parent_pid):
            return 1
        if time.monotonic() >= deadline:
            return 1
        time.sleep(0.05)
    try:
        gate_path.unlink()
    except FileNotFoundError:
        return 1
    os.execv(executable, [executable, *args])
    return 1


def _positive_float(value: str) -> float:
    """解析正秒数参数。"""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("值必须为正数。")
    return parsed


def _positive_int(value: str) -> int:
    """解析正整数参数。"""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("值必须为正整数。")
    return parsed


def main() -> int:
    """处理启动脚本使用的标记和 backend readiness 命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("--lock-file", required=True)
    mark_parser.add_argument("--host", default="127.0.0.1")
    mark_parser.add_argument("--port", type=_positive_int, required=True)
    mark_parser.add_argument("--model", default=MODEL_ALIAS)
    mark_parser.add_argument("--executable", required=True)
    identify_parser = subparsers.add_parser("identify")
    identify_parser.add_argument("--lock-file", required=True)
    identify_parser.add_argument("--pid", type=_positive_int, required=True)
    identify_parser.add_argument("--pgid", type=_positive_int, required=True)
    identify_parser.add_argument("--executable", required=True)
    wait_parser = subparsers.add_parser("wait-sleeping")
    wait_parser.add_argument("--base-url", required=True)
    wait_parser.add_argument("--timeout", type=_positive_float, required=True)
    wait_parser.add_argument("--interval", type=_positive_float, default=0.5)
    wait_parser.add_argument("--pid", type=_positive_int)
    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument("--lock-file", required=True)
    startup_parser = subparsers.add_parser("startup")
    startup_parser.add_argument("--lock-file", required=True)
    startup_parser.add_argument("--pid-file", required=True)
    startup_parser.add_argument("--host", default="127.0.0.1")
    startup_parser.add_argument("--port", type=_positive_int, required=True)
    startup_parser.add_argument("--model", default=MODEL_ALIAS)
    startup_parser.add_argument("--base-url", required=True)
    startup_parser.add_argument("--executable", required=True)
    startup_parser.add_argument("--lock-timeout", type=_positive_float, required=True)
    startup_parser.add_argument("--timeout", type=_positive_float, required=True)
    startup_parser.add_argument("--interval", type=_positive_float, default=0.5)
    startup_parser.add_argument("--cuda-release-delay", type=float, default=2.0)
    startup_parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    exec_parser = subparsers.add_parser("exec-backend")
    exec_parser.add_argument("--gate", required=True)
    exec_parser.add_argument("--parent-pid", type=_positive_int, required=True)
    exec_parser.add_argument("--executable", required=True)
    exec_parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    shutdown_parser = subparsers.add_parser("shutdown")
    shutdown_parser.add_argument("--lock-file", required=True)
    shutdown_parser.add_argument("--host", default="127.0.0.1")
    shutdown_parser.add_argument("--port", type=_positive_int, required=True)
    shutdown_parser.add_argument("--model", default=MODEL_ALIAS)
    shutdown_parser.add_argument("--pid", type=_positive_int, required=True)
    shutdown_parser.add_argument("--pgid", type=_positive_int, required=True)
    shutdown_parser.add_argument("--executable", required=True)
    shutdown_parser.add_argument("--lock-timeout", type=_positive_float, required=True)
    shutdown_parser.add_argument("--cleanup-timeout", type=_positive_float, default=30.0)
    shutdown_parser.add_argument("--cuda-release-delay", type=float, default=2.0)
    args = parser.parse_args()

    try:
        if args.command == "mark":
            if not args.model.strip():
                raise ValueError("marker model alias 不能为空。")
            create_quarantine_marker(
                args.lock_file,
                host=args.host,
                port=args.port,
                pid=None,
                pgid=None,
                executable=args.executable,
                model_alias=args.model,
            )
        elif args.command == "identify":
            if args.pid != args.pgid:
                raise ValueError("llama-server 必须独占 setsid 进程组。")
            update_backend_identity(
                args.lock_file,
                pid=args.pid,
                pgid=args.pgid,
                executable=args.executable,
            )
        elif args.command == "wait-sleeping":
            wait_for_sleeping(
                args.base_url,
                timeout_seconds=args.timeout,
                poll_interval=args.interval,
                pid=args.pid,
            )
        elif args.command == "clear":
            if not remove_quarantine_marker(args.lock_file):
                raise ValueError("没有可清除的 llama.cpp 隔离标记，保留当前隔离状态。")
        elif args.command == "startup":
            backend_args = (
                args.backend_args[1:] if args.backend_args[:1] == ["--"] else args.backend_args
            )
            if not backend_args or args.cuda_release_delay < 0:
                raise ValueError("startup 必须提供 backend 参数和非负 CUDA_RELEASE_DELAY。")
            start_backend(
                lock_path=args.lock_file,
                pid_file=args.pid_file,
                host=args.host,
                port=args.port,
                executable=args.executable,
                backend_args=backend_args,
                base_url=args.base_url,
                lock_timeout=args.lock_timeout,
                startup_timeout=args.timeout,
                poll_interval=args.interval,
                cuda_release_delay=args.cuda_release_delay,
                model_alias=args.model,
            )
        elif args.command == "exec-backend":
            backend_args = (
                args.backend_args[1:] if args.backend_args[:1] == ["--"] else args.backend_args
            )
            return _exec_backend(
                Path(args.gate),
                args.parent_pid,
                args.executable,
                backend_args,
            )
        elif args.command == "shutdown":
            if args.pid != args.pgid or args.cuda_release_delay < 0:
                raise ValueError("shutdown 需要匹配的进程组和非负 CUDA_RELEASE_DELAY。")
            stop_backend(
                lock_path=args.lock_file,
                host=args.host,
                port=args.port,
                pid=args.pid,
                pgid=args.pgid,
                executable=args.executable,
                lock_timeout=args.lock_timeout,
                cleanup_timeout=args.cleanup_timeout,
                cuda_release_delay=args.cuda_release_delay,
                model_alias=args.model,
            )
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"llama.cpp backend 管理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
