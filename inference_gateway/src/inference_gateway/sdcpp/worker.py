"""等待父进程持久化身份后再 exec sd-cli，避免未登记的 GPU 子进程。"""

from __future__ import annotations

import os
import sys


def main() -> int:
    """仅在收到启动许可时替换当前进程，PID 和 starttime 保持不变。"""
    if len(sys.argv) < 2 or sys.stdin.buffer.readline(4) != b"run\n":
        return 125
    with open(os.devnull, "rb") as null:
        os.dup2(null.fileno(), 0)
    os.execv(sys.argv[1], sys.argv[1:])
    return 126


if __name__ == "__main__":
    raise SystemExit(main())
