#!/usr/bin/env python3
"""运行 Ornith-1.5-35B-A3B 的 HTTP 适配器。"""

from __future__ import annotations

import os

import uvicorn

from ornith_1_5_35b_a3b.app import app


def main() -> None:
    """按统一启动脚本提供的地址运行 API。"""
    host = os.getenv("ORNITH_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("ORNITH_PORT", "8401"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
