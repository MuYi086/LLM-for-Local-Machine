#!/usr/bin/env python3
"""MOSS-SoundEffect v2.0 单次音效生成 worker。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def generate(payload: dict[str, Any], output_wav: Path) -> None:
    """加载本地官方 pipeline 并生成 48 kHz 单声道 WAV。"""
    code_path = Path(str(payload["code_path"])).resolve()
    if not (code_path / "pipeline_moss_soundeffect.py").is_file():
        raise FileNotFoundError(f"MOSS-SoundEffect 官方源码不完整: {code_path}")
    sys.path.insert(0, str(code_path.parent))

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-SoundEffect v2.0 需要可用的 CUDA GPU。")

    model_dir = Path(str(payload["model_dir"]))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MOSS-SoundEffect 模型目录不存在: {model_dir}")
    pipeline = MossSoundEffectPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device=str(payload["device"]),
    )
    with torch.inference_mode():
        audio = pipeline(
            prompt=str(payload["prompt"]),
            seconds=float(payload["seconds"]),
            num_inference_steps=int(payload["num_inference_steps"]),
            cfg_scale=float(payload["cfg_scale"]),
            sigma_shift=float(payload["sigma_shift"]),
            seed=int(payload["seed"]),
        )
    pipeline.save_audio(audio, output_wav, sample_rate=48_000)


def main() -> None:
    """读取 API 请求 JSON 并执行一次模型生成。"""
    parser = argparse.ArgumentParser(description="MOSS-SoundEffect v2.0 one-shot worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    arguments = parser.parse_args()
    with Path(arguments.input_json).open(encoding="utf-8") as request_file:
        payload = json.load(request_file)
    generate(payload, Path(arguments.output_wav))


if __name__ == "__main__":
    main()
