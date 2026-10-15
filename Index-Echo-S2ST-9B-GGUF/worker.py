#!/usr/bin/env python3
"""一次性加载官方 Echo 完整原生包，直接调用 DubbingBridgeModel。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def dub(request: dict) -> dict:
    """保持官方 from_pretrained/dub 的输入和输出，不加入其他模型。

    来源：https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B
    模型的 CosyVoice3 是原生包自身组件，整个推理仅在此子进程驻留。
    """
    model_dir = Path(request["model_dir"]).expanduser().resolve()
    audio_path = Path(request["audio_path"]).resolve()
    output_wav = Path(request["output_wav"]).resolve()
    if not (model_dir / "modeling_dubbing.py").is_file() or not audio_path.is_file():
        raise FileNotFoundError("Echo 完整原生包或输入音频缺失。")
    if os.getenv("LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes", "on"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(model_dir))
    # 官方包使用相对资源路径；worker 独立进程允许安全地切换工作目录。
    os.chdir(model_dir)
    from modeling_dubbing import DubbingBridgeModel

    model = DubbingBridgeModel.from_pretrained(str(model_dir), device=request.get("device", "cuda"))
    result = model.dub(
        str(audio_path),
        lang=request["target_language"],
        out_wav=str(output_wav),
        mode=request.get("mode", "h2cv"),
        chunk=False,
        return_info=request.get("return_info", False),
    )
    metadata = {"sample_rate": int(result[1])}
    if request.get("return_info"):
        metadata["info"] = result[2]
    if not output_wav.is_file():
        raise RuntimeError("Echo 官方 API 未生成 WAV。")
    return metadata


def main() -> int:
    """读取一次性任务，保留原生 WAV 和可选诊断信息。"""
    parser = argparse.ArgumentParser(description="Index-Echo-S2ST 原生 worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    output_path = Path(args.output_json).resolve()
    try:
        request = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("worker 输入必须是对象。")
        result = dub(request)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
