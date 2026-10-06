"""用标准库检查整组 HTTP 服务的监听地址，避免重复启动后才逐个报错。"""

from __future__ import annotations

import socket
import sys
from collections.abc import Sequence
from contextlib import ExitStack


def check_ports(services: Sequence[tuple[str, str, int]]) -> list[str]:
    """检查已有监听和配置内冲突，返回全部错误并释放所有探测 socket。"""
    errors: list[str] = []
    # 保持探测 socket 到整组检查结束，才能发现通配地址与具体地址之间的配置冲突。
    # 检查结束后释放端口；实际启动仍由 Uvicorn 处理其它应用随后抢占端口的情况。
    with ExitStack() as stack:
        for name, host, port in services:
            label = f"{name} ({host}:{port})"
            if not 1 <= port <= 65535:
                errors.append(f"{label}: 端口必须在 1 到 65535 之间")
                continue
            try:
                addresses = socket.getaddrinfo(
                    host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
                )
                for family, sock_type, protocol, _, address in dict.fromkeys(addresses):
                    listener = stack.enter_context(socket.socket(family, sock_type, protocol))
                    # 与 Uvicorn/asyncio 一致：允许 TIME_WAIT 重启，并分开 IPv4/IPv6 监听。
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if family == socket.AF_INET6:
                        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    listener.bind(address)
                    listener.listen(1)
            except OSError as exc:
                errors.append(f"{label}: {exc}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """接收重复的服务名、host、port 三元组，冲突时使启动脚本立即退出。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or len(arguments) % 3:
        print("端口检查参数必须按服务名、host、port 成组提供。", file=sys.stderr)
        return 2
    try:
        services = [
            (arguments[index], arguments[index + 1], int(arguments[index + 2]))
            for index in range(0, len(arguments), 3)
        ]
    except ValueError:
        print("服务端口必须是整数。", file=sys.stderr)
        return 2
    errors = check_ports(services)
    if errors:
        print("启动前端口检查失败，未启动任何服务：", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print("请先在原启动终端按 Ctrl+C，或停止占用端口的服务后重试。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
