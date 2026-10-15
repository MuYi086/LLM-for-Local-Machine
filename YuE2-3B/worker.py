#!/usr/bin/env python3
"""YuE2-3B 一次性歌曲生成 worker。"""

from __future__ import annotations

# YuE2、Torch 和 CUDA 只在一次性 worker 中导入，进程退出后释放显存。
import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """读取 worker 请求 JSON 与生成 WAV 的输出路径。"""
    parser = argparse.ArgumentParser(description="One-shot YuE2-3B song generation worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    """读取 worker 请求，并确认 JSON 顶层是对象。"""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def configure_offline_environment(request: dict[str, Any]) -> None:
    """设置本地模型缓存和离线标记，避免推理时隐式联网。"""
    hf_mirror_dir = str(request.get("hf_mirror_dir") or "").strip()
    if hf_mirror_dir:
        os.environ.setdefault("HF_HOME", hf_mirror_dir)
    if bool(request.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def generate_song(request: dict[str, Any], output_path: Path) -> None:
    """调用 YuE2 原生管线并保存 48 kHz WAV。"""
    configure_offline_environment(request)
    inference_wheel = Path(str(request.get("inference_wheel") or "")).expanduser()
    if not inference_wheel.is_file():
        raise FileNotFoundError(f"YuE2 官方推理 wheel 不存在: {inference_wheel}")
    sys.path.insert(0, str(inference_wheel.resolve()))
    try:
        from yue2 import YuE2Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "YuE2 推理依赖导入失败，请先执行 uv sync，并确认模型目录含官方 yue2_infer 0.1.5 wheel。"
        ) from exc

    pipeline = None
    try:
        pipeline = YuE2Pipeline.from_pretrained(
            model=str(request["model_dir"]),
            vae=str(request["vae_model_dir"]),
            local_files_only=bool(request.get("local_files_only", True)),
            device=str(request.get("device", "cuda")),
            memory_budget_gib=float(request.get("memory_budget_gib", 24)),
            backend=str(request.get("backend", "torch")),
            quantization=str(request.get("quantization", "none")),
            offload_ar=bool(request.get("offload_ar", False)),
            progress=False,
        )
        song = pipeline(
            style=request.get("style"),
            tags=request.get("tags"),
            lyrics=request["lyrics"],
            cot=request.get("cot", "full"),
            seed=request.get("seed", 831001),
            abc=request.get("abc"),
            cfg_scale=request.get("cfg_scale"),
            id=request.get("id", "song"),
            abc_sampling=request.get("abc_sampling"),
            semantic_sampling=request.get("semantic_sampling"),
        )
        song.save(output_path)
    finally:
        if pipeline is not None:
            pipeline.close()
        gc.collect()


def main() -> int:
    """执行一次歌曲生成，错误 traceback 写入 stderr 供 API 进程截取。"""
    args = parse_args()
    try:
        request = load_request(Path(args.input_json))
        generate_song(request, Path(args.output_wav))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
