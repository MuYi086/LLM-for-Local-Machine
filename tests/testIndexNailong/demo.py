"""使用 Index-NativeLong（Index-Nailong）官方长文翻译 API 运行 Demo。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_text_model_demo  # noqa: E402


def main() -> None:
    """调用官方 doc_translate.py，原样保存模型文本输出。"""
    run_text_model_demo(
        demo_name="Index-NativeLong (Index-Nailong)",
        env_prefix="INDEX_NAILONG",
        script_relative_path="inference/llm/doc_translate.py",
        model_name="IndexTeam/Index-Nailong-9B",
        output_name="index_nailong_zh_to_en.txt",
        demo_dir=Path(__file__).resolve().parent,
        model_arguments=["--direction", "zh-en"],
        stdin_input=True,
    )


if __name__ == "__main__":
    main()
