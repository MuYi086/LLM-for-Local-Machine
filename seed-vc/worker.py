#!/usr/bin/env python3
"""Seed-VC 一次性音色转换 worker，复用官方分块推理流程。"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from seed_vc_runtime import read_inference_config


def parse_args() -> argparse.Namespace:
    """解析父进程提供的 JSON 和 WAV 路径。"""
    parser = argparse.ArgumentParser(description="One-shot Seed-VC voice conversion worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取 worker 请求并校验顶层结构。"""
    with open(path, encoding="utf-8") as source:
        request = json.load(source)
    if not isinstance(request, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return request


def require_path(value: str, label: str, *, directory: bool = False) -> Path:
    """确认本地文件或目录存在，不接受空路径。"""
    if not value.strip():
        raise ValueError(f"{label}路径不能为空。")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"{label}不存在: {path}")
    return path


def configure_environment(request: dict[str, Any], cache_dir: Path) -> None:
    """在导入模型前配置缓存、离线模式和官方旧检查点的加载语义。"""
    if request.get("hf_mirror_dir"):
        os.environ.setdefault("HF_HOME", str(request["hf_mirror_dir"]))
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hf_cache")
    for name, suffix in (
        ("NUMBA_CACHE_DIR", "numba"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        os.environ.setdefault(name, str(cache_dir / suffix))
    if request.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # 官方旧版检查点包含非 Tensor 元数据；PyTorch 2.6+ 的默认 weights_only
    # 无法读取它们。只在隔离 worker 中对运维准备的本地官方权重恢复旧加载语义。
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def build_runtime_config(request: dict[str, Any], output_path: Path) -> Path:
    """把配置中的 Whisper 和 BigVGAN repo ID 改为已准备的本地目录。"""
    import yaml

    template = require_path(str(request.get("config_path") or ""), "Seed-VC 配置")
    config = read_inference_config(template, bool(request.get("f0_condition", False)))
    parameters = config["model_params"]
    parameters["speech_tokenizer"]["name"] = str(
        require_path(str(request["whisper_model_dir"]), "Whisper 模型", directory=True)
    )
    parameters["vocoder"]["name"] = str(
        require_path(str(request["vocoder_model_dir"]), "BigVGAN 模型", directory=True)
    )
    with output_path.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination, allow_unicode=True, sort_keys=False)
    return output_path


def install_local_resolver(inference: Any, request: dict[str, Any]) -> None:
    """替换官方直接导入的下载函数，使 CAMPPlus/RMVPE 使用本地权重。"""
    known_files = {
        ("funasr/campplus", "campplus_cn_common.bin"): request["style_encoder_checkpoint"],
        ("lj1995/VoiceConversionWebUI", "rmvpe.pt"): request.get("rmvpe_checkpoint", ""),
    }

    def resolve_file(
        repo_id: str, model_filename: str = "pytorch_model.bin", config_filename: str | None = None
    ):
        """仅解析已声明的辅助权重，防止未知依赖绕过文件预检。"""
        value = known_files.get((repo_id, model_filename))
        if value is None or config_filename is not None:
            raise FileNotFoundError(f"Seed-VC 未配置本地依赖: {repo_id}/{model_filename}")
        return str(require_path(str(value), f"{repo_id}/{model_filename}"))

    inference.load_custom_model_from_hf = resolve_file


def load_runtime(code_path: Path, request: dict[str, Any]):
    """只在 worker 中导入 Torch 和外部官方 inference.py。"""
    sys.path.insert(0, str(code_path))
    import torch

    device = torch.device(str(request.get("device") or "cuda:0"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，Seed-VC 默认要求 GPU。")
        # 官方代码创建部分 Tensor 时使用当前 CUDA 设备，必须与配置的 device 一致。
        torch.cuda.set_device(device)
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用。")
    elif device.type not in {"cpu", "mps"}:
        raise ValueError(f"Seed-VC 不支持设备: {device}")
    spec = importlib.util.spec_from_file_location(
        "unitale_seed_vc_inference", code_path / "inference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 Seed-VC 官方 inference.py。")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    inference.device = device
    install_local_resolver(inference, request)
    return torch, inference


def convert_audio(request: dict[str, Any], output_path: Path) -> None:
    """调用官方推理入口，保留长音频分块、交叉淡化和可选 F0 转换。"""
    code_path = require_path(str(request.get("code_path") or ""), "Seed-VC 源码", directory=True)
    checkpoint = require_path(str(request.get("checkpoint_path") or ""), "Seed-VC 权重")
    source_path = require_path(str(request.get("source_audio_path") or ""), "原始音频")
    reference_path = require_path(str(request.get("reference_audio_path") or ""), "参考音频")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch = None
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix="inference_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        configure_environment(request, temporary_path)
        config_path = build_runtime_config(request, temporary_path / "config.yaml")
        try:
            torch, inference = load_runtime(code_path, request)
            # 官方 inference.py 在导入时重设 HF_HUB_CACHE；在调用入口前恢复隔离缓存。
            configure_environment(request, temporary_path)
            arguments = argparse.Namespace(
                source=str(source_path),
                target=str(reference_path),
                output=str(temporary_path / "results"),
                checkpoint=str(checkpoint),
                config=str(config_path),
                diffusion_steps=request["diffusion_steps"],
                length_adjust=request["length_adjust"],
                inference_cfg_rate=request["inference_cfg_rate"],
                f0_condition=request["f0_condition"],
                auto_f0_adjust=request["auto_f0_adjust"],
                semi_tone_shift=request["semi_tone_shift"],
                fp16=bool(request["fp16"] and inference.device.type == "cuda"),
            )
            with torch.inference_mode():
                inference.main(arguments)
            generated = list((temporary_path / "results").glob("*.wav"))
            if len(generated) != 1 or generated[0].stat().st_size <= 44:
                raise RuntimeError("Seed-VC 没有生成唯一的非空 WAV。")
            shutil.move(str(generated[0]), str(output_path))
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


def main() -> int:
    """执行一次音色转换并通过退出码向 HTTP 父进程报告结果。"""
    args = parse_args()
    try:
        convert_audio(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
