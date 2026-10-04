#!/usr/bin/env python3
"""Confucius4-TTS 一次性零样本语音克隆 worker。"""

from __future__ import annotations

# 模型、Torch 与官方 Confucius4-TTS 源码只在一次性进程中导入，退出后释放显存。
import argparse
import gc
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

CONFUCIUS_REPO_ID = "netease-youdao/Confucius4-TTS"
CAMPPLUS_REPO_ID = "funasr/campplus"


def parse_args() -> argparse.Namespace:
    """解析 worker JSON 输入和 WAV 输出路径。"""
    parser = argparse.ArgumentParser(description="One-shot Confucius4-TTS synthesis worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并确认 worker 输入 JSON 顶层为对象。"""
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def require_path(path: str, label: str, *, directory: bool = False) -> Path:
    """验证文件或目录存在，并返回规范化绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "目录" if directory else "文件"
        raise FileNotFoundError(f"{label}{kind}不存在: {resolved}")
    return resolved


def require_text(value: Any, label: str) -> str:
    """提取非空文本字段并去除首尾空白。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} 不能为空。")
    return text


def configure_offline_environment(request: dict[str, Any]) -> None:
    """在 worker 中设置缓存路径和离线模式，避免请求期间隐式联网下载。"""
    hf_mirror_dir = str(request.get("hf_mirror_dir") or "").strip()
    if hf_mirror_dir:
        os.environ.setdefault("HF_HOME", hf_mirror_dir)
    if bool(request.get("local_files_only", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_runtime(code_path: Path, model_dir: Path, style_checkpoint: Path, local_files_only: bool):
    """从外部源码加载 ConfuciusTTS，并把 Hugging Face 权重解析到本地文件。"""
    code_string = str(code_path)
    if code_string not in sys.path:
        sys.path.insert(0, code_string)

    try:
        import torch
        import torchaudio
        from confuciustts.cli import inference as inference_module
    except ImportError as exc:
        raise RuntimeError(
            "Confucius4-TTS 运行时导入失败，请先为 Confucius4_TTS 执行 uv sync，"
            f"并确认 CONFUCIUS4_TTS_CODE_PATH 指向官方源码。缺少或无法导入: {exc}"
        ) from exc

    original_hf_hub_download = inference_module.hf_hub_download

    def resolve_hf_file(repo_id: str, filename: str, *args, **kwargs) -> str:
        """优先解析已准备好的权重，只有显式关闭离线模式时才允许联网。"""
        direct_path = Path(filename).expanduser()
        if direct_path.is_file():
            return str(direct_path.resolve())

        if repo_id == CONFUCIUS_REPO_ID:
            local_path = model_dir / filename
        elif repo_id == CAMPPLUS_REPO_ID:
            local_path = style_checkpoint
        else:
            local_path = Path()

        if local_path.is_file():
            return str(local_path.resolve())
        if local_files_only:
            raise FileNotFoundError(f"离线模式下找不到 {repo_id}/{filename}，请准备本地模型文件。")
        return original_hf_hub_download(repo_id, filename, *args, **kwargs)

    # 官方 inference.py 直接导入了 hf_hub_download；在 worker 中替换同一模块引用，
    # 让已有模型目录真正遵守本服务的 LOCAL_FILES_ONLY 约束。
    inference_module.hf_hub_download = resolve_hf_file
    return torch, torchaudio, inference_module.ConfuciusTTS


def build_runtime_config(request: dict[str, Any], template_path: Path) -> Path:
    """把官方配置模板改写为当前机器上的本地模型路径。"""
    import yaml

    with template_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Confucius4-TTS 配置模板顶层必须是对象。")

    paths = config.setdefault("paths", {})
    paths["tokenizer_path"] = str(request["model_dir"])
    paths["w2v_bert_path"] = str(request["w2v_bert_model_dir"])
    paths["w2v_stat"] = str(Path(request["model_dir"]) / "wav2vec2bert_stats.pt")
    paths["vocoder_path"] = str(request["vocoder_model_dir"])
    paths["t2s_checkpoint"] = "t2s_model.safetensors"
    paths["s2a_checkpoint"] = "s2a_model.pt"
    style_encoder = dict(paths.get("style_encoder") or {})
    style_encoder["checkpoint"] = str(request["style_encoder_checkpoint"])
    paths["style_encoder"] = style_encoder

    descriptor, temporary_name = tempfile.mkstemp(prefix="confucius4_tts_config_", suffix=".yaml")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def resolve_device(torch, device: str) -> str:
    """验证请求设备可用性，避免加载大模型后才暴露配置错误。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，Confucius4-TTS 默认要求 GPU。")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求使用 MPS，但当前 PyTorch 未检测到可用的 MPS 设备。")
    return device


def configure_vocalization_duration(model: Any, seconds: float | None) -> None:
    """为单个感叹音设置 S2A 长度；保留完整 T2S 输出，不截断词汇语音。"""
    if seconds is None:
        return
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 2
    ):
        raise ValueError("vocalization_duration_seconds 必须是 (0, 2] 内的有限秒数。")
    frames = round(seconds * model.sample_rate / model.hop_length)
    if frames < 1:
        raise ValueError("vocalization_duration_seconds 小于一个 Mel 帧，无法合成。")
    original_inference = model.s2a_model.inference
    calls = 0

    def timed_inference(*args: Any, **kwargs: Any) -> Any:
        """只改变长度调节器的目标帧数，不裁剪语义 token、latent 或最终 WAV。"""
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("vocalization_duration_seconds 只支持单个合成块，不能用于多段台词。")
        lengths = kwargs["target_feat_len"]
        kwargs["target_feat_len"] = lengths.new_full(lengths.shape, frames)
        return original_inference(*args, **kwargs)

    model.s2a_model.inference = timed_inference


def generate_audio(request: dict[str, Any], output_path: Path) -> None:
    """加载 Confucius4-TTS 两阶段模型并生成 WAV。"""
    configure_offline_environment(request)
    model_dir = require_path(
        str(request.get("model_dir") or ""), "Confucius4-TTS 模型", directory=True
    )
    code_path = require_path(
        str(request.get("code_path") or ""), "Confucius4-TTS 源码", directory=True
    )
    reference_audio_path = require_path(str(request.get("reference_audio_path") or ""), "参考音频")
    template_path = require_path(
        str(request.get("config_template_path") or ""), "Confucius4-TTS 配置模板"
    )
    require_path(str(request.get("w2v_bert_model_dir") or ""), "Wav2Vec2-BERT 模型", directory=True)
    require_path(str(request.get("vocoder_model_dir") or ""), "BigVGAN 模型", directory=True)
    style_checkpoint = require_path(
        str(request.get("style_encoder_checkpoint") or ""), "CAMPPlus 权重"
    )
    device = str(request.get("device") or "cuda:0")
    local_files_only = bool(request.get("local_files_only", True))
    torch, torchaudio, confucius_tts = load_runtime(
        code_path,
        model_dir,
        style_checkpoint,
        local_files_only,
    )
    device = resolve_device(torch, device)
    runtime_config_path = build_runtime_config(request, template_path)
    model = None
    try:
        model = confucius_tts(config_path=str(runtime_config_path), device=device)
        configure_vocalization_duration(model, request.get("vocalization_duration_seconds"))
        audio = model.generate(
            text=require_text(request.get("text"), "text"),
            lang=require_text(request.get("lang"), "lang"),
            prompt_wav=str(reference_audio_path),
            raw=bool(request.get("raw", False)),
            temperature=float(request.get("temperature", 0.8)),
            top_p=float(request.get("top_p", 0.8)),
            top_k=int(request.get("top_k", 30)),
            num_beams=int(request.get("num_beams", 3)),
            repetition_penalty=float(request.get("repetition_penalty", 10.0)),
            max_length=int(request.get("max_length", 1520)),
            n_timesteps=int(request.get("n_timesteps", 25)),
            inference_cfg_rate=float(request.get("inference_cfg_rate", 0.7)),
            max_text_tokens_per_segment=int(request.get("max_text_tokens_per_segment", 80)),
            cross_fade_duration=float(request.get("cross_fade_duration", 0.3)),
            edge_fade_duration=float(request.get("edge_fade_duration", 0.1)),
            edge_pad_duration=float(request.get("edge_pad_duration", 0.1)),
            verbose=bool(request.get("verbose", False)),
        )
        if not torch.is_tensor(audio) or audio.numel() == 0:
            raise RuntimeError("Confucius4-TTS 返回了空音频。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), audio.detach().cpu(), model.sample_rate)
    finally:
        if model is not None:
            del model
        runtime_config_path.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[Confucius4-TTS worker] CUDA synchronize 跳过: {exc}")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """执行一次语音合成，并将结果写入输出 WAV。"""
    args = parse_args()
    try:
        generate_audio(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
