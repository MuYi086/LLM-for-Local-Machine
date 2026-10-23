#!/usr/bin/env python3
"""YuE2-Vae 一次性 latent 编码与音频解码 worker。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def configure_offline_environment(payload: dict[str, Any]) -> None:
    """阻止一次性 worker 在本地推理期间隐式下载模型。"""
    hf_mirror_dir = str(payload.get("hf_mirror_dir") or "").strip()
    if hf_mirror_dir:
        os.environ.setdefault("HF_HOME", hf_mirror_dir)
    if bool(payload.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_model(payload: dict[str, Any]):
    """按操作加载完整 VAE 或仅加载解码器。"""
    import torch
    from transformers import AutoModel

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(min(4, torch.get_num_threads()))

    device = str(payload["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，YuE2-Vae 服务默认要求 GPU。")

    model = AutoModel.from_pretrained(
        payload["model_dir"],
        trust_remote_code=True,
        local_files_only=bool(payload.get("local_files_only", True)),
        decoder_only=payload["operation"] == "decode",
        device=device,
    )
    return torch, model


def decode_latents(payload: dict[str, Any], output_path: Path, torch, model) -> None:
    """将原生 [T, 64] latent.npy 分块解码为 48 kHz 立体声 WAV。"""
    import numpy as np
    import soundfile as sf

    latent_array = np.load(payload["latent_path"], allow_pickle=False)
    if latent_array.ndim != 2 or latent_array.shape[1] != 64 or latent_array.shape[0] == 0:
        raise ValueError("latent.npy 必须是非空 [T, 64] 数组。")
    if latent_array.shape[0] > int(payload["max_latent_frames"]):
        raise ValueError(
            f"latent 帧数超过限制 {payload['max_latent_frames']}，请缩短音频或调整配置。"
        )
    if not np.issubdtype(latent_array.dtype, np.number) or not np.isfinite(latent_array).all():
        raise ValueError("latent.npy 必须只包含有限数值。")

    latent = torch.from_numpy(np.asarray(latent_array, dtype=np.float32).T.copy()).unsqueeze(0)
    audio = model.decode_tiled(latent, output_device="cpu")[0].transpose(0, 1).contiguous()
    waveform = audio.numpy().clip(-1.0, 1.0)
    sf.write(output_path, waveform, samplerate=48_000, subtype="PCM_24", format="WAV")


def encode_audio(payload: dict[str, Any], output_path: Path, torch, model) -> None:
    """将标准化的 48 kHz 立体声 WAV 编码为原生 [T, 64] latent.npy。"""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(
        payload["audio_path"],
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != 48_000 or audio.shape[1] != 2:
        raise ValueError("编码输入必须是 48 kHz 立体声 WAV。")
    if audio.shape[0] < 1_920 or not np.isfinite(audio).all():
        raise ValueError("音频必须至少包含一个 latent 帧且只包含有限数值。")

    waveform = (
        torch.from_numpy(audio.T.copy())
        .unsqueeze(0)
        .to(
            device=str(payload["device"]),
            dtype=torch.float32,
        )
    )
    generator = None
    if payload.get("seed") is not None:
        generator = torch.Generator(device=str(payload["device"]))
        generator.manual_seed(int(payload["seed"]))
    latent = model.encode(waveform, sample=bool(payload.get("sample", False)), generator=generator)
    latent_array = latent[0].transpose(0, 1).detach().cpu().float().numpy()
    np.save(output_path, latent_array, allow_pickle=False)


def run(payload: dict[str, Any], output_path: Path) -> None:
    """执行单次原生 VAE 编码或分块解码并保存输出。"""
    configure_offline_environment(payload)
    torch, model = load_model(payload)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if payload["operation"] == "decode":
            decode_latents(payload, output_path, torch, model)
        elif payload["operation"] == "encode":
            encode_audio(payload, output_path, torch, model)
        else:
            raise ValueError(f"不支持的操作: {payload['operation']}")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[YuE2-Vae worker] CUDA synchronize 跳过: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """读取 JSON 请求并执行一次推理。"""
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
