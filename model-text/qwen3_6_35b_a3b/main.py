#!/usr/bin/env python3
"""运行 Qwen3.6-35B-A3B-MTP 的 HTTP 适配器。"""

from __future__ import annotations

import os

import uvicorn

from qwen3_6_35b_a3b.app import app


def main() -> None:
    """按统一启动脚本提供 Qwen API。"""
    host = os.getenv("QWEN36_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("QWEN36_PORT", "8402"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
