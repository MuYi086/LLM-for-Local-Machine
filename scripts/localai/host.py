#!/usr/bin/env python3
"""受管 LocalAI 进程组的宿主机 supervisor，可选启动外部 worker bridge。"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PROJECT = ROOT / "localai_worker_bridge"
BRIDGE_MODULE = "localai_worker_bridge.server"
_CHILDREN: list[subprocess.Popen[bytes]] = []
_STOP_REQUESTED = False


def _handle_signal(signum: int, frame: object) -> None:
    """收到进程组停止信号后进入有界子进程回收。"""
    del frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _bridge_enabled() -> bool:
    return os.getenv("LLM_LOCALAI_WORKER_BRIDGE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_address(address: str) -> tuple[str, int]:
    """限制 bridge 地址到回环并解析有效端口。"""
    host, separator, raw_port = address.rpartition(":")
    if not separator or host not in {"127.0.0.1", "localhost", "[::1]"}:
        raise ValueError("LLM_LOCALAI_BRIDGE_ADDRESS 只允许本机回环地址。")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("LocalAI bridge gRPC 端口无效。") from exc
    if not 1 <= port <= 65535:
        raise ValueError("LocalAI bridge gRPC 端口超出有效范围。")
    return host, port


def _wait_for_bridge(process: subprocess.Popen[bytes], address: str) -> None:
    """确认 gRPC server 已监听，再启动 LocalAI HTTP host。"""
    host, port = _validate_address(address)
    if host == "[::1]":
        host = "::1"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("LocalAI worker bridge 启动后立即退出。")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("LocalAI worker bridge 未在 20 秒内监听 gRPC 端口。")


def _start_bridge() -> None:
    address = os.getenv("LLM_LOCALAI_BRIDGE_ADDRESS", "127.0.0.1:50052")
    _validate_address(address)
    manifest = Path(
        os.getenv(
            "LLM_LOCALAI_BRIDGE_WORKERS",
            str(ROOT / "localai/manifests/worker-bridge.json"),
        )
    ).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"LocalAI bridge worker manifest 不存在：{manifest}")
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("找不到 uv，无法启动 LocalAI worker bridge。")
    command = [
        uv,
        "run",
        "--project",
        str(BRIDGE_PROJECT),
        "--no-sync",
        "python",
        "-m",
        BRIDGE_MODULE,
        "--manifest",
        str(manifest.resolve()),
        "--address",
        address,
    ]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True)
    _CHILDREN.append(process)
    _wait_for_bridge(process, address)


def _stop_children() -> None:
    """对可见子进程发送 TERM，等待收尾并在超时后 KILL。"""
    for process in _CHILDREN:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for process in _CHILDREN:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    """先启 bridge，再启 LocalAI；任一子进程退出时停止另一方。"""
    if len(sys.argv) != 2:
        print("用法：host.py /absolute/path/to/local-ai", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    localai_binary = Path(sys.argv[1]).resolve()
    if not localai_binary.is_file() or not os.access(localai_binary, os.X_OK):
        print("LocalAI binary 不存在或不可执行。", file=sys.stderr)
        return 2

    try:
        if _bridge_enabled():
            _start_bridge()
        localai = subprocess.Popen(
            [str(localai_binary), "run"], stdin=subprocess.DEVNULL, close_fds=True
        )
        _CHILDREN.append(localai)
        while not _STOP_REQUESTED:
            for process in _CHILDREN:
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"LocalAI host 子进程 {process.pid} 已退出：{return_code}", file=sys.stderr
                    )
                    _stop_children()
                    return return_code or 1
            time.sleep(0.1)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"LocalAI host 启动失败：{exc}", file=sys.stderr)
        _stop_children()
        return 1
    _stop_children()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
