#!/usr/bin/env python3
"""运行 Ornith-1.5-9B-GGUF HTTP API。"""

from __future__ import annotations

import os

import uvicorn
from app import app


def main() -> None:
    """按统一启动脚本提供的地址运行 API。"""
    host = os.getenv("ORNITH_9B_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("ORNITH_9B_PORT", "8503"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
