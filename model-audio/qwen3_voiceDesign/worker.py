#!/usr/bin/env python3
"""Qwen3-TTS VoiceDesign 一次性推理 worker。"""

from __future__ import annotations

# 只有此进程导入 Qwen3-TTS 和 torch；请求结束后进程退出以释放 CUDA 上下文。
import gc
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voicedesign_runtime import split_text


def parse_args():
    """解析 VoiceDesign 请求文件和输出 WAV 路径。"""
    import argparse

    parser = argparse.ArgumentParser(description="One-shot Qwen3-TTS VoiceDesign worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取 JSON 请求并确认顶层结构为对象。"""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: Any, label: str) -> str:
    """校验必须有内容的文本字段并去掉首尾空白。"""
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", str(value or "").strip())
    if not text:
        raise ValueError(f"{label} 不能为空。")
    return text


def resolve_dtype(torch, value: str):
    """把配置中的 dtype 名称解析成 torch 类型。"""
    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.bfloat16
    try:
        return getattr(torch, requested)
    except AttributeError as exc:
        raise ValueError(f"不支持的 Qwen dtype: {requested}") from exc


def resolve_attention(torch, requested: str, dtype: Any) -> str:
    """选择 FlashAttention 或 SDPA，避免 API 进程导入 CUDA 扩展。"""
    value = str(requested or "auto")
    if value != "auto":
        return value
    if importlib.util.find_spec("flash_attn") is not None and dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int) -> np.ndarray:
    """在多个生成片段之间插入静音并合并为一个 numpy 波形。"""
    if not waveforms:
        raise RuntimeError("Qwen3-TTS VoiceDesign 未返回音频。")

    segments = []
    for waveform in waveforms:
        if hasattr(waveform, "detach"):
            audio = waveform.detach().float().cpu().numpy()
        else:
            audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        segments.append(audio.reshape(-1).astype(np.float32, copy=False))

    pause = np.zeros(int(sample_rate * max(pause_ms, 0) / 1000), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        joined.append(segment)
        if index + 1 < len(segments) and len(pause):
            joined.append(pause)
    return np.concatenate(joined)


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    """加载 VoiceDesign 模型，按文本片段生成并保存参考音色 WAV。"""
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(f"Qwen3-TTS VoiceDesign 运行时不可导入：{exc.name or exc}") from exc

    local_files_only = bool(request.get("local_files_only", True))
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS VoiceDesign 需要 CUDA GPU。")

    model_path = Path(str(request.get("model_path") or "")).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3-TTS VoiceDesign 模型目录不存在：{model_path}")

    text = normalize_text(request.get("text"), "text")
    instruction = normalize_text(request.get("voice_description"), "voice_description")
    # 0 明确表示不分片，不能被 ``or`` 变成其他默认值。
    chunks = split_text(
        text,
        int(
            request["max_chars_per_chunk"] if request.get("max_chars_per_chunk") is not None else 0
        ),
    )
    dtype = resolve_dtype(torch, request.get("dtype", "auto"))
    attention = resolve_attention(torch, request.get("attn_implementation", "auto"), dtype)
    load_kwargs = {
        "device_map": request.get("device_map") or "cuda:0",
        "dtype": dtype,
        "attn_implementation": attention,
        "local_files_only": local_files_only,
    }
    model = None
    try:
        print(f"[Qwen3-TTS VoiceDesign worker] 模型目录: {model_path}")
        print(
            f"[Qwen3-TTS VoiceDesign worker] device_map={load_kwargs['device_map']}, "
            f"dtype={dtype}, attn_implementation={attention}, chunks={len(chunks)}"
        )
        model = Qwen3TTSModel.from_pretrained(str(model_path), **load_kwargs)
        generation_kwargs = {"max_new_tokens": int(request.get("max_new_tokens") or 2048)}
        for key in ("top_p", "temperature"):
            if request.get(key) is not None:
                generation_kwargs[key] = request[key]

        language = request.get("language") or "Chinese"
        wavs, sample_rate = model.generate_voice_design(
            text=chunks if len(chunks) > 1 else chunks[0],
            instruct=[instruction] * len(chunks) if len(chunks) > 1 else instruction,
            language=[language] * len(chunks) if len(chunks) > 1 else language,
            non_streaming_mode=True,
            **generation_kwargs,
        )
        waveform = join_waveforms(
            wavs,
            int(sample_rate),
            int(request["pause_ms"] if request.get("pause_ms") is not None else 250),
        )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, int(sample_rate))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[Qwen3-TTS VoiceDesign worker] CUDA synchronize 跳过: {exc}")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """执行一次 VoiceDesign worker，并报告可读的错误摘要。"""
    args = parse_args()
    try:
        synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
