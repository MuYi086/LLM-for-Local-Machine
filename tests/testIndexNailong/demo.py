"""调用 8384 的 Index-Nailong 原生接口，保存 TXT 译文。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_text_model_demo  # noqa: E402


def main() -> None:
    """保持模型原生文本输入和输出，先检查运行环境。"""
    run_text_model_demo(
        demo_name="Index-Nailong",
        env_prefix="INDEX_NAILONG",
        port=8384,
        route="naiLong",
        output_name="index_nailong_zh_to_en.txt",
        demo_dir=Path(__file__).resolve().parent,
    )


if __name__ == "__main__":
    main()
