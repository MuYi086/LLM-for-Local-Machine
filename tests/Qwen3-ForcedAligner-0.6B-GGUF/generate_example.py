"""通过真实 HTTP API 冒烟测试 Qwen3 ForcedAligner。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.audio_cpp.generate_examples import run_single_model  # noqa: E402

MODEL_KEY = "qwen3_forced_aligner"


def main() -> int:
    """请求 ForcedAligner API 并检查词级时间戳 JSON。"""
    return run_single_model(MODEL_KEY)


if __name__ == "__main__":
    raise SystemExit(main())
