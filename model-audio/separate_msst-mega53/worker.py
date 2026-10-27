#!/usr/bin/env python3
"""在一次性进程中运行 Mega 53 重点声部分离，HTTP 进程不加载模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from contract import CONFIG_FILENAME, SAMPLE_RATE, WEIGHT_FILENAME, parse_stems


def select_heads(model, config, stems: list[str], module_list) -> None:
    """完整加载权重后只保留目标输出头，避免在 GPU 上计算未选择的输出头。"""
    instruments = list(config.training.instruments)
    indices = [instruments.index(stem) for stem in stems]
    model.mask_estimators = module_list([model.mask_estimators[index] for index in indices])
    model.num_stems = len(stems)
    config.training.instruments = stems
    config.training.target_instrument = None


def separate(payload: dict, output_dir: Path) -> None:
    """离线加载模型、分离选定声部并保留数学重建底轨及质量元数据。"""
    source_dir = Path(payload["source_dir"])
    sys.path.insert(0, str(source_dir))
    import numpy as np
    import soundfile as sf
    import torch
    from utils.model_utils import demix
    from utils.settings import get_model_from_config

    model_dir = Path(payload["model_dir"])
    model, config = get_model_from_config("bs_roformer", str(model_dir / CONFIG_FILENAME))
    stems = parse_stems(",".join(payload["stems"]), list(config.training.instruments))
    # CPU mmap 避免同时复制完整 checkpoint；只接受纯权重，不执行 pickle 自定义对象。
    state = torch.load(
        model_dir / WEIGHT_FILENAME, map_location="cpu", weights_only=True, mmap=True
    )
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    select_heads(model, config, stems, torch.nn.ModuleList)
    config.inference.batch_size = 1
    config.inference.chunk_size = int(payload["chunk_seconds"]) * SAMPLE_RATE
    config.inference.num_overlap = 2
    config.training.use_amp = payload["device"].startswith("cuda")
    model.eval().to(payload["device"])
    mixture, rate = sf.read(payload["audio_path"], dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or mixture.shape[1] != 2 or not np.isfinite(mixture).all():
        raise ValueError("worker 输入必须是有限值 44.1 kHz 双声道 WAV。")
    with torch.inference_mode():
        estimates = demix(config, model, mixture.T, torch.device(payload["device"]), "bs_roformer")
    residual = mixture.copy()
    tracks = []
    reconstructed = np.zeros_like(mixture)
    for stem in stems:
        samples = np.asarray(estimates[stem].T, dtype=np.float32)
        if samples.shape != mixture.shape or not np.isfinite(samples).all():
            raise ValueError(f"无效的 {stem} 输出。")
        residual -= samples
        reconstructed += samples
        tracks.append(write_track(output_dir, stem, samples, rate, sf, np))
    # residual 只保证干声相加接近输入，不代表消除了重叠或分离伪影；保持 float 防止削波。
    tracks.append(write_track(output_dir, "residual", residual, rate, sf, np))
    error = float(np.max(np.abs(reconstructed + residual - mixture)))
    try:
        source_revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        source_revision = None
    metadata = {
        "version": "1.0",
        "model": "MVSep Mega 53 v1",
        "model_sha256": file_sha256(model_dir / WEIGHT_FILENAME),
        "config_sha256": file_sha256(model_dir / CONFIG_FILENAME),
        "source_revision": source_revision,
        "input_sha256": payload["input_sha256"],
        "sample_rate": rate,
        "channels": 2,
        "samples": len(mixture),
        "duration_seconds": len(mixture) / rate,
        "selected_stems": stems,
        "chunk_seconds": payload["chunk_seconds"],
        "num_overlap": 2,
        "tracks": tracks,
        "reconstruction_max_abs_error": error,
        "residual_definition": "normalized_input - sum(selected_stems)",
        "warnings": [
            "目标轨可能串音或重叠；residual 为数学差值，不是独立训练的伴奏声部。",
            "原曲不可再次叠加；删除目标轨时须将其干声加回 residual 或重新构建底轨。",
            "空间移动会改变原混音中的相位抵消关系，需试听并检查单声道兼容性。",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    """分块计算实际文件哈希，记录模型与输出的来源而不额外复制文件。"""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_track(output_dir, stem, samples, rate, sf, np) -> dict:
    """写入不截幅的 FLOAT WAV，并记录业务验收所需统计与内容哈希。"""
    path = output_dir / f"{stem}.wav"
    sf.write(path, samples, rate, subtype="FLOAT")
    digest = file_sha256(path)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return {
        "stem": stem,
        "filename": path.name,
        "sha256": digest,
        "samples": len(samples),
        "peak": float(np.max(np.abs(samples))),
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
    }


def main() -> None:
    """接收父进程控制的 JSON 和输出目录，异常交由父进程报告与回收。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    separate(json.loads(Path(args.input_json).read_text(encoding="utf-8")), Path(args.output_dir))


if __name__ == "__main__":
    main()
