"""通过真实 HTTP API 冒烟测试 BS-RoFormer 音乐分离。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.audio_cpp.generate_examples import run_single_model  # noqa: E402

MODEL_KEY = "bs_roformer"


def main() -> int:
    """请求 BS-RoFormer 分离 API 并检查 ZIP stems。"""
    return run_single_model(MODEL_KEY)


if __name__ == "__main__":
    raise SystemExit(main())
