"""调用 8382 的 Echo 完整原生语音接口，保存 WAV。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from index_translate_demo_common import run_echo_demo  # noqa: E402


def main() -> None:
    """直接上传源语音，不调用外部 ASR 或 TTS。"""
    run_echo_demo(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
