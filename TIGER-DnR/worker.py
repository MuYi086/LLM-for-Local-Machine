#!/usr/bin/env python3
"""TIGER-DnR 一次性三 Stem 分离 worker。"""

from __future__ import annotations

# Torch、Torchaudio 与上游 TIGER 只在 worker 中导入；请求结束即退出并释放 CUDA 上下文。
import gc
import importlib
import json
import os
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

STEM_NAMES = ("dialog", "effects", "music")
MODEL_SAMPLE_RATE = 44_100
RUNTIME_PACKAGE = "_unitale_tiger_dnr"


def parse_args():
    """解析一次性 worker 的 JSON 输入和三 Stem 输出方式。"""
    import argparse

    parser = argparse.ArgumentParser(description="One-shot TIGER-DnR worker")
    parser.add_argument("--input-json", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir")
    output.add_argument("--output-zip")
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取 JSON 请求并确认顶层为对象。"""
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def require_path(path_value: Any, label: str, *, directory: bool = False) -> Path:
    """解析并验证所需文件或目录。"""
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.exists() or (directory and not path.is_dir()):
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def install_namespace_package(name: str, path: Path) -> None:
    """建立不执行上游聚合 __init__ 的命名空间包，避免无关训练依赖。"""
    if name in sys.modules:
        return
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package


def tiger_model_class(source_dir: Path):
    """加载上游 TIGERDNR 类，仅引入 DnR 推理所需的模型和层模块。"""
    package_root = source_dir / "look2hear"
    model_dir = package_root / "models"
    layers_dir = package_root / "layers"
    required_paths = (
        model_dir / "base_model.py",
        model_dir / "tiger_dnr.py",
        layers_dir / "activations.py",
        layers_dir / "normalizations.py",
    )
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError(f"TIGER-DnR 上游源码目录不完整：{source_dir}")
    install_namespace_package(RUNTIME_PACKAGE, package_root)
    install_namespace_package(f"{RUNTIME_PACKAGE}.models", model_dir)
    install_namespace_package(f"{RUNTIME_PACKAGE}.layers", layers_dir)
    module = importlib.import_module(f"{RUNTIME_PACKAGE}.models.tiger_dnr")
    return module.TIGERDNR


def resample(torchaudio, waveform, source_rate: int, target_rate: int):
    """仅在采样率变化时重采样，减少不必要的信号变换。"""
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def fit_sample_count(torch, waveform, expected_samples: int):
    """修正重采样边界误差，使每个 Stem 与输入音频严格等长。"""
    actual_samples = waveform.shape[-1]
    if actual_samples > expected_samples:
        return waveform[..., :expected_samples]
    if actual_samples < expected_samples:
        return torch.nn.functional.pad(waveform, (0, expected_samples - actual_samples))
    return waveform


def validate_stem(torch, stem, stem_name: str):
    """规范化上游输出为 Torchaudio 可写入的 [channels, samples] 张量。"""
    waveform = stem.detach().cpu() if isinstance(stem, torch.Tensor) else torch.as_tensor(stem)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise RuntimeError(f"TIGER-DnR 返回了非法 {stem_name} Stem 张量。")
    return waveform


def separate(request: dict[str, Any], output_dir: Path) -> None:
    """从本地权重执行 TIGER-DnR，并保存 dialog、effects、music 三个 WAV。"""
    torch = None
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(f"TIGER-DnR 运行时不可导入：{exc.name or exc}") from exc

    if bool(request.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    audio_path = require_path(request.get("audio_path"), "输入音频")
    model_dir = require_path(request.get("model_dir"), "TIGER-DnR 模型目录", directory=True)
    source_dir = require_path(request.get("source_dir"), "TIGER-DnR 上游源码目录", directory=True)
    model_files = (model_dir / "config.json", model_dir / "model.safetensors")
    if not all(path.is_file() and path.stat().st_size > 0 for path in model_files):
        raise FileNotFoundError(f"TIGER-DnR 模型目录不完整：{model_dir}")

    device_name = str(request.get("device") or "cuda")
    if device_name not in {"cuda", "cpu"}:
        raise ValueError("device 仅支持 cuda 或 cpu。")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TIGER-DnR 请求 CUDA，但当前 PyTorch 未检测到可用 CUDA。")

    model = None
    try:
        model_class = tiger_model_class(source_dir)
        print(f"[TIGER-DnR worker] 模型目录: {model_dir}")
        print(f"[TIGER-DnR worker] source={source_dir}, device={device_name}")
        model = model_class.from_pretrained(str(model_dir), local_files_only=True)
        model = model.to(torch.device(device_name)).eval()

        waveform, input_sample_rate = torchaudio.load(audio_path)
        if input_sample_rate <= 0 or waveform.numel() == 0:
            raise RuntimeError("输入音频为空或采样率非法。")
        model_waveform = resample(torchaudio, waveform, input_sample_rate, MODEL_SAMPLE_RATE)
        with torch.inference_mode():
            dialog, effects, music = model(
                model_waveform.unsqueeze(0).to(torch.device(device_name))
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        for stem_name, stem in zip(STEM_NAMES, (dialog, effects, music), strict=True):
            restored = resample(
                torchaudio,
                validate_stem(torch, stem, stem_name),
                MODEL_SAMPLE_RATE,
                int(input_sample_rate),
            )
            restored = fit_sample_count(torch, restored, waveform.shape[-1])
            destination = output_dir / f"{stem_name}.wav"
            temporary = output_dir / f".{stem_name}.partial.wav"
            torchaudio.save(
                temporary,
                restored,
                int(input_sample_rate),
                encoding="PCM_S",
                bits_per_sample=16,
            )
            os.replace(temporary, destination)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[TIGER-DnR worker] CUDA synchronize 跳过: {exc}")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """运行一次三 Stem 分离并返回适合父进程判定的退出码。"""
    args = parse_args()
    try:
        request = load_request(args.input_json)
        if args.output_dir:
            separate(request, Path(args.output_dir).expanduser().resolve())
        else:
            output_zip = Path(args.output_zip).expanduser().resolve()
            output_zip.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="tiger-dnr-stems-") as temporary:
                output_dir = Path(temporary)
                separate(request, output_dir)
                with zipfile.ZipFile(
                    output_zip, mode="w", compression=zipfile.ZIP_STORED
                ) as archive:
                    for stem_name in STEM_NAMES:
                        stem_path = output_dir / f"{stem_name}.wav"
                        if not stem_path.is_file() or stem_path.stat().st_size == 0:
                            raise RuntimeError(f"TIGER-DnR 缺少有效 {stem_name}.wav。")
                        archive.write(stem_path, arcname=stem_path.name)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
