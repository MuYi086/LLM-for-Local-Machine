#!/usr/bin/env python3
"""通过 WanGP 官方 Python API 执行一次 LongCat Avatar 推理。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

MODEL_TYPE = "longcat_avatar_v1_5"
MEMORY_PROFILE = "4"


def load_request(path: str | Path) -> dict[str, Any]:
    """读取一次性 worker 请求并检查顶层结构。"""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("worker 输入 JSON 顶层必须是对象。")
    return payload


def build_settings(session: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """以 WanGP 自带的 LongCat Avatar 默认项为基础构造任务设置。"""
    settings = session.get_default_settings(MODEL_TYPE)
    settings.update(
        {
            "model_type": MODEL_TYPE,
            "prompt": payload["prompt"],
            "audio_guide": str(Path(payload["audio_file"]).resolve()),
            "resolution": payload["resolution"],
            "video_length": f"{float(payload['duration_seconds']):.4f}s",
            "force_fps": 25,
            "sample_solver": "distill",
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "sliding_window_size": 93,
            "sliding_window_overlap": 13,
            "repeat_generation": 1,
        }
    )
    image_file = payload.get("image_file")
    if image_file:
        settings["image_refs"] = [str(Path(image_file).resolve())]
    return settings


def select_video(paths: list[str]) -> Path:
    """从 WanGP 结果中选取最近生成的 MP4。"""
    candidates = [
        Path(path).resolve()
        for path in paths
        if Path(path).suffix.lower() == ".mp4" and Path(path).is_file()
    ]
    if not candidates:
        raise RuntimeError("WanGP API 没有返回可用的 MP4 视频。")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run(payload: dict[str, Any]) -> Path:
    """初始化 WanGP 低显存运行时，等待生成并导出单个 MP4。"""
    wangp_dir = Path(payload["wangp_dir"]).resolve()
    output_dir = Path(payload["workspace_dir"]).resolve() / "wangp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(wangp_dir))

    from shared.api import init

    session = init(
        root=wangp_dir,
        output_dir=output_dir,
        cli_args=[
            "--offline",
            "--profile",
            MEMORY_PROFILE,
            "--attention",
            "sdpa",
            "--gpu",
            "cuda:0",
        ],
        console_output=True,
        console_isatty=False,
    )
    job = session.submit_task(build_settings(session, payload))
    result = job.result(timeout=float(payload["request_timeout"]))
    if not result.success:
        details = "; ".join(error.message for error in result.errors)
        raise RuntimeError(details or "WanGP LongCat Avatar 推理失败。")

    generated = select_video(result.generated_files)
    output_video = Path(payload["output_video"]).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated, output_video)
    print(f"WanGP 已生成 MP4：{output_video}")
    return output_video


def main() -> int:
    """解析单次请求并运行 WanGP 任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    payload = load_request(args.input_json)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
