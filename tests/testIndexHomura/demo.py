"""使用 Index-Homura 官方音节约束翻译 API 运行 Demo。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_text_model_demo  # noqa: E402


def main() -> None:
    """调用官方 syllable_translate.py，原样保存模型文本输出。"""
    run_text_model_demo(
        demo_name="Index-Homura",
        env_prefix="INDEX_HOMURA",
        script_relative_path="inference/llm/syllable_translate.py",
        model_name="IndexTeam/Index-Homura-9B",
        output_name="index_homura_zh_to_en.txt",
        demo_dir=Path(__file__).resolve().parent,
        model_arguments=["--target", "en"],
        default_target_syllables=36,
    )


if __name__ == "__main__":
    main()
