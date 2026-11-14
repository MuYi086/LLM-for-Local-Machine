"""在可由父进程限时终止的客户端进程中流式接收模型响应。"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

LIMIT = 64 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """禁止模型服务将任务内容重定向到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """将重定向视为请求失败。"""
        return None


def receive(url: str, content_type: str, body: bytes, destination: Path) -> None:
    """流式接收有界响应；父进程负责总时限，不执行重试。"""
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=7200) as response, destination.open("xb") as stream:
        total = 0
        while chunk := response.read(64 * 1024):
            total += len(chunk)
            if total > LIMIT:
                raise ValueError("模型响应超过 64 MiB。")
            stream.write(chunk)


if __name__ == "__main__":
    try:
        receive(
            sys.argv[1], sys.argv[2], sys.stdin.buffer.read(1024 * 1024 + 4096), Path(sys.argv[3])
        )
    except Exception as exc:
        # 不输出服务响应或 traceback，避免把请求内容写入普通日志。
        print(type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None
