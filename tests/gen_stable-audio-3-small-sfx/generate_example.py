"""通过真实 HTTP API 冒烟测试 Stable Audio 3 Small SFX。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.audio_cpp.generate_examples import run_single_model  # noqa: E402

MODEL_KEY = "stable_audio_sfx"


def main() -> int:
    """固定请求 Stable Audio SFX 服务并验证 WAV 响应。"""
    return run_single_model(MODEL_KEY)


if __name__ == "__main__":
    raise SystemExit(main())
