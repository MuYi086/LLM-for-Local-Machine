"""使用 Index-Translate 官方多语种文本翻译 API 运行 Demo。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_text_model_demo  # noqa: E402


def main() -> None:
    """调用官方 translate.py，原样保存模型文本输出。"""
    run_text_model_demo(
        demo_name="Index-Translate",
        env_prefix="INDEX_TRANSLATE",
        script_relative_path="inference/llm/translate.py",
        model_name="IndexTeam/Index-Translate-9B",
        output_name="index_translate_zh_to_en.txt",
        demo_dir=Path(__file__).resolve().parent,
        model_arguments=["--source", "zh", "--target", "en"],
    )


if __name__ == "__main__":
    main()
