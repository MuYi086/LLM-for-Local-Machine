#!/usr/bin/env python3
"""一次性 SoulX-FlashHead worker；模型依赖只在这个进程内加载。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def generate_video(payload: dict[str, Any], output_path: Path) -> None:
    """调用官方 pipeline，逐块写入视频并以检查退出码的 FFmpeg 合并音频。"""
    import math
    from collections import deque

    import imageio.v2 as imageio
    import librosa
    import numpy as np
    import torch

    code_path = Path(payload["code_path"]).resolve()
    sys.path.insert(0, str(code_path))
    previous_directory = Path.cwd()
    silent_video = output_path.with_name("silent.mp4")
    normalized_audio = output_path.with_name("driver.wav")
    try:
        # 官方 inference.py 在 import 时用相对路径读取 YAML，cwd 只在隔离 worker 内切换。
        os.chdir(code_path)
        inference = importlib.import_module("flash_head.inference")
        if not torch.cuda.is_available():
            raise RuntimeError("SoulX-FlashHead 推理需要可用的 CUDA 设备。")
        ffmpeg = payload["ffmpeg_bin"]
        # 官方编码器输入固定 16 kHz；保留驱动音频内容，统一容器以支持所有上传格式。
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                payload["reference_audio_path"],
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(normalized_audio),
            ],
            check=True,
        )
        speech, _ = librosa.load(str(normalized_audio), sr=16000, mono=True)
        if len(speech) == 0:
            raise ValueError("驱动音频为空。")
        pipeline = inference.get_pipeline(
            world_size=1,
            ckpt_dir=payload["model_dir"],
            wav2vec_dir=payload["wav2vec_dir"],
            model_type=payload["model_type"],
        )
        inference.get_base_data(
            pipeline,
            cond_image_path_or_dir=payload["reference_image_path"],
            base_seed=payload["seed"],
            use_face_crop=payload["use_face_crop"],
        )
        params = inference.get_infer_params()
        sample_rate = params["sample_rate"]
        if sample_rate != 16000:
            raise RuntimeError("官方 SoulX-FlashHead 配置 sample_rate 必须为 16000。")
        fps = params["tgt_fps"]
        motion_frames = params["motion_frames_num"]
        frame_count = params["frame_num"]
        slice_length = (frame_count - motion_frames) * sample_rate // fps
        cached_length = int(sample_rate * params["cached_audio_duration"])
        audio_end = int(params["cached_audio_duration"] * fps)
        cache = deque([0.0] * cached_length, maxlen=cached_length)
        # 上游每块删去 motion_frames；补齐末块并按原始音频长度裁剪，避免丢失短音频/尾音。
        target_frames = math.ceil(len(speech) * fps / sample_rate)
        frame_written = 0
        # 逐帧编码，不保留整段视频的 CPU tensor 列表，长音频不会线性增长内存。
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
        with (
            imageio.get_writer(
                str(silent_video),
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=["-bf", "0"],
            ) as writer,
            torch.inference_mode(),
        ):
            for offset in range(0, len(speech), slice_length):
                chunk = speech[offset : offset + slice_length]
                if len(chunk) < slice_length:
                    chunk = np.pad(chunk, (0, slice_length - len(chunk)))
                cache.extend(chunk.tolist())
                embedding = inference.get_audio_embedding(
                    pipeline,
                    np.asarray(cache),
                    audio_end - frame_count,
                    audio_end,
                )
                video = inference.run_pipeline(pipeline, embedding)[motion_frames:]
                for frame in video.cpu().numpy().astype(np.uint8):
                    if frame_written >= target_frames:
                        break
                    writer.append_data(frame)
                    frame_written += 1
                del video, embedding
        if frame_written == 0:
            raise RuntimeError("SoulX-FlashHead 未生成视频帧。")
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                payload["reference_audio_path"],
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
    finally:
        os.chdir(previous_directory)
        silent_video.unlink(missing_ok=True)
        normalized_audio.unlink(missing_ok=True)


def main() -> None:
    """读取父进程请求，离线加载本地模型并写入临时 MP4。"""
    parser = argparse.ArgumentParser(description="SoulX-FlashHead one-shot worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-video", required=True)
    args = parser.parse_args()
    with Path(args.input_json).open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    generate_video(payload, Path(args.output_video).resolve())


if __name__ == "__main__":
    main()
