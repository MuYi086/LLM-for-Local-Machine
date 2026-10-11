"""使用 Index-Echo-S2ST 官方语音 API 运行中译英 Demo。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
SOURCE_AUDIO = REPOSITORY_DIR / "tests/testIndexTranslate/speech.wav"
DEFAULT_MODEL_DIR = Path(os.getenv("HF_MIRROR_DIR", str(REPOSITORY_DIR.parent))) / (
    "IndexTeam/Index-Echo-S2ST-9B"
)


def main() -> None:
    """直接调用 Index-Echo-S2ST 的官方 DubbingBridgeModel API。"""
    parser = argparse.ArgumentParser(description="运行 Index-Echo-S2ST 官方语音翻译 Demo")
    parser.add_argument("--audio", type=Path, default=SOURCE_AUDIO, help="输入语音 WAV")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("INDEX_ECHO_S2ST_NATIVE_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
        help="官方完整 Index-Echo-S2ST 模型目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dist" / "index_echo_s2st_zh_to_en.wav",
        help="官方 API 生成的 WAV 输出路径",
    )
    parser.add_argument("--target-language", choices=("en", "es", "ja"), default="en")
    parser.add_argument(
        "--device",
        default=os.getenv("INDEX_ECHO_S2ST_DEVICE", "cuda"),
        help="官方模型运行设备",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"找不到输入音频：{args.audio}")
    if not (args.model_dir / "modeling_dubbing.py").is_file():
        raise SystemExit(
            f"{args.model_dir} 不是完整的 Index-Echo-S2ST 模型包；请先按官方说明下载模型，"
            "或通过 --model-dir / INDEX_ECHO_S2ST_NATIVE_MODEL_DIR 指定目录。"
        )

    sys.path.insert(0, str(args.model_dir.resolve()))
    try:
        from modeling_dubbing import DubbingBridgeModel
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 环境缺少 Index-Echo-S2ST 官方模型依赖；请使用其官方推理环境运行。"
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = DubbingBridgeModel.from_pretrained(str(args.model_dir), device=args.device)
    model.dub(
        str(args.audio.resolve()),
        lang=args.target_language,
        out_wav=str(args.output.resolve()),
    )
    if not args.output.is_file():
        raise SystemExit("Index-Echo-S2ST 官方 API 未生成 WAV 文件。")

    print("模型：Index-Echo-S2ST 官方模型包")
    print(f"参考音频：{args.audio}")
    print(f"输出音频：{args.output}")
    print(f"文件大小：{args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
