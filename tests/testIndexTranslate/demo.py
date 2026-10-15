"""调用 8385 的 Index-Translate 原生接口，保存 TXT 译文。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_text_model_demo  # noqa: E402


def main() -> None:
    """保持模型原生文本输入和输出，先检查运行环境。"""
    run_text_model_demo(
        demo_name="Index-Translate",
        env_prefix="INDEX_TRANSLATE",
        port=8385,
        route="translate",
        output_name="index_translate_zh_to_en.txt",
        demo_dir=Path(__file__).resolve().parent,
    )


if __name__ == "__main__":
    main()
