#!/usr/bin/env python3
"""调用 MERT-v2-FullSong 提取音乐的整曲与逐帧表征。"""

from __future__ import annotations

# Transformers、Torch 和 CUDA 只在一次性 worker 中导入，进程退出后释放显存。
import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def configure_offline_environment(payload: dict[str, Any]) -> None:
    """设置本地缓存和离线标记，避免推理时隐式联网。"""
    hf_mirror_dir = str(payload.get("hf_mirror_dir") or "").strip()
    if hf_mirror_dir:
        os.environ.setdefault("HF_HOME", hf_mirror_dir)
    if bool(payload.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_model(payload: dict[str, Any]):
    """使用模型卡推荐的 AutoFeatureExtractor 与 AutoModel 加载本地权重。"""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = str(payload["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，MERT-v2-FullSong 服务默认要求 GPU。")

    model_dir = str(payload["model_dir"])
    local_files_only = bool(payload.get("local_files_only", True))
    processor = AutoFeatureExtractor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
    )
    model.eval().to(device)
    expected_rate = int(getattr(processor, "sampling_rate", 0))
    configured_rate = int(getattr(model.config, "sampling_rate", 0))
    if expected_rate != configured_rate or expected_rate != 24_000:
        raise RuntimeError("MERT-v2-FullSong processor 与模型采样率不一致，预期均为 24000 Hz。")
    return torch, processor, model


def read_waveform(payload: dict[str, Any]):
    """读取 API 预处理的 24 kHz 单声道 float32 WAV。"""
    import numpy as np
    import soundfile as sf

    waveform, sample_rate = sf.read(
        payload["audio_path"],
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != 24_000 or waveform.shape[1] != 1:
        raise ValueError("模型输入必须是 24 kHz 单声道音频。")
    if waveform.shape[0] < 1_025 or not np.isfinite(waveform).all():
        raise ValueError("模型输入长度不足或包含非有限采样值。")
    return waveform[:, 0], sample_rate


def extract_features(
    payload: dict[str, Any],
    output_path: Path,
    torch,
    processor,
    model,
) -> None:
    """执行官方特征提取流程并保存整曲 embedding 或逐帧特征 NPZ。"""
    import numpy as np

    waveform, sample_rate = read_waveform(payload)
    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(str(payload["device"]))

    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=False)

    frames = output.last_hidden_state[0]
    mask = output.feature_attention_mask[0].to(dtype=torch.bool)
    valid_frames = frames[mask]
    if valid_frames.shape[0] == 0:
        raise RuntimeError("MERT-v2-FullSong 未生成有效音频帧。")
    embedding = valid_frames.float().mean(dim=0)
    if not torch.isfinite(embedding).all() or not torch.isfinite(valid_frames).all():
        raise RuntimeError("MERT-v2-FullSong 输出包含非有限值。")

    embedding_array = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    metadata = {
        "model": "m-a-p/MERT-v2-FullSong",
        "sampling_rate": sample_rate,
        "frame_rate": float(model.config.frame_rate),
        "frame_count": int(valid_frames.shape[0]),
        "embedding_dimension": int(embedding_array.shape[0]),
        "duration_seconds": len(waveform) / sample_rate,
        "pooling": "masked_mean",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if payload["output"] == "embedding":
        result = {**metadata, "embedding": embedding_array.tolist()}
        with output_path.open("w", encoding="utf-8") as result_file:
            json.dump(result, result_file, ensure_ascii=False, allow_nan=False)
            result_file.flush()
            os.fsync(result_file.fileno())
        return

    frame_array = valid_frames.detach().cpu().float().numpy().astype(np.float32, copy=False)
    np.savez_compressed(
        output_path,
        last_hidden_state=frame_array,
        embedding=embedding_array,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def run(payload: dict[str, Any], output_path: Path) -> None:
    """加载模型、执行特征提取并在 finally 中释放 CUDA 资源。"""
    configure_offline_environment(payload)
    torch, processor, model = load_model(payload)
    try:
        extract_features(payload, output_path, torch, processor, model)
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[MERT-v2-FullSong worker] CUDA synchronize 跳过: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """执行一次特征提取；错误 traceback 写入 stderr 供 API 进程截取。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    try:
        with Path(args.input_json).open(encoding="utf-8") as request_file:
            payload = json.load(request_file)
        if not isinstance(payload, dict):
            raise ValueError("worker JSON 顶层必须是对象。")
        run(payload, Path(args.output_file).resolve())
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
