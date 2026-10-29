#!/usr/bin/env python3
"""一次性 Kandinsky 6 worker；仅此进程导入 Torch、Diffusers 并加载权重。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析服务请求 JSON 与临时 MP4 路径。"""
    parser = argparse.ArgumentParser(description="Kandinsky 6 Lite distilled one-shot worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-video", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并验证父进程写入的请求对象。"""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("worker 输入 JSON 顶层必须是对象。")
    return payload


def generate_video(payload: dict[str, Any], output_path: Path) -> None:
    """使用本地 BF16 模型和官方 10 步配置生成无音轨视频。"""
    import torch
    from diffusers import Kandinsky6TI2VAPipeline
    from diffusers.utils import encode_video, load_image

    pipeline = None
    try:
        pipeline = Kandinsky6TI2VAPipeline.from_pretrained(
            payload["model_dir"],
            torch_dtype=torch.bfloat16,
            local_files_only=bool(payload["local_files_only"]),
        )
        # 逐层卸载能把消费级显卡上的峰值显存压低，代价是模块搬运增加推理时长。
        pipeline.enable_sequential_cpu_offload()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        generator = torch.Generator(device="cpu").manual_seed(int(payload["seed"]))
        image = load_image(payload["reference_image_path"])
        result = pipeline(
            prompt=payload["prompt"],
            image=image,
            height=int(payload["height"]),
            width=int(payload["width"]),
            num_frames=int(payload["num_frames"]),
            frame_rate=int(payload["frame_rate"]),
            num_inference_steps=int(payload["num_inference_steps"]),
            guidance_scale=float(payload["guidance_scale"]),
            sample_audio=False,
            expand_prompts=False,
            generator=generator,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(result.frames[0], fps=int(payload["frame_rate"]), output_path=str(output_path))
        del result
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> None:
    """在 worker 进程内设置离线标记并执行一次推理。"""
    arguments = parse_args()
    payload = load_request(arguments.input_json)
    if bool(payload.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    generate_video(payload, Path(arguments.output_video))


if __name__ == "__main__":
    main()
