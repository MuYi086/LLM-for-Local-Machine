#!/usr/bin/env python3
"""Qwen3-TTS 一次性语音克隆 worker。

该进程从 JSON 请求文件读取参数，按文本切分执行推理，写出一个 WAV，
最后清理临时文件并退出；模型不在 FastAPI 父进程中加载。
"""

from __future__ import annotations

# worker 才导入 torch、transformers 和 Qwen3-TTS，父进程因此可以保持无模型启动。
import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析父进程传入的请求 JSON、输出 WAV 和可选运行参数。"""
    parser = argparse.ArgumentParser(description="One-shot Qwen3-TTS worker")
    parser.add_argument("--input-json", required=True, help="Request JSON file path")
    parser.add_argument("--output-wav", required=True, help="Output wav file path")
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并校验一个 JSON 对象，拒绝数组或空请求。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def normalize_text(text: str) -> str:
    """清理文本空白，保证后续切分不会产生空片段。"""
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise RuntimeError("text 不能为空。")
    return normalized


def split_text(text: str, max_chars: int) -> list[str]:
    """按句号、逗号等边界切分长文本，控制单次模型上下文长度。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentences = re.findall(r".+?[。！？；;!?]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence, max_chars))
            continue

        candidate = current + sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def split_long_sentence(text: str, max_chars: int) -> list[str]:
    """对没有自然标点的超长句做硬切分。"""
    parts = re.findall(r".+?[，,、：:]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                part[index : index + max_chars] for index in range(0, len(part), max_chars)
            )
            continue
        candidate = current + part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def resolve_dtype(torch: Any, dtype_name: Any, device: str) -> Any:
    """把配置字符串转换成 torch dtype，并处理 CPU 不适合的精度。"""
    normalized = str(dtype_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"不支持的 dtype：{dtype_name}")


def resolve_attn_implementation(torch: Any, requested: Any, device: str, dtype: Any) -> str:
    """选择可用的 attention 后端，缺少编译扩展时回退到 SDPA。"""
    normalized = str(requested or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def to_mono_float32(audio: Any, np: Any) -> Any:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 2 else 1)
    return waveform.reshape(-1)


def trim_leading_silence(
    waveform: Any,
    sample_rate: int,
    enabled: bool,
    threshold_db: float,
    min_silence_ms: int,
    analysis_window_ms: int,
    pre_roll_ms: int,
    max_trim_ms: int,
    np: Any,
):
    audio = to_mono_float32(waveform, np)
    if not enabled or audio.size == 0:
        return audio, 0

    analysis_window = max(1, int(sample_rate * max(analysis_window_ms, 1) / 1000))
    threshold = float(10 ** (threshold_db / 20.0))
    power = np.square(audio, dtype=np.float32)
    kernel = np.ones(analysis_window, dtype=np.float32) / analysis_window
    rms = np.sqrt(np.convolve(power, kernel, mode="same")).astype(np.float32, copy=False)
    active_indices = np.flatnonzero(rms >= threshold)
    if active_indices.size == 0:
        return audio, 0

    trim_index = int(active_indices[0])
    min_silence_samples = int(sample_rate * max(min_silence_ms, 0) / 1000)
    if trim_index < min_silence_samples:
        return audio, 0

    max_trim_samples = int(sample_rate * max(max_trim_ms, 0) / 1000)
    if max_trim_samples > 0:
        trim_index = min(trim_index, max_trim_samples)

    pre_roll_samples = int(sample_rate * max(pre_roll_ms, 0) / 1000)
    trim_start = max(0, trim_index - pre_roll_samples)
    if trim_start <= 0:
        return audio, 0

    trimmed_audio = audio[trim_start:]
    if trimmed_audio.size == 0:
        return audio, 0
    return trimmed_audio, trim_start


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any) -> Any:
    """在分段波形之间插入静音并沿帧轴拼接。"""
    if not waveforms:
        raise RuntimeError("Qwen3-TTS 未返回音频片段。")

    segments = [to_mono_float32(waveform, np) for waveform in waveforms]
    pause_samples = int(sample_rate * max(pause_ms, 0) / 1000)
    if pause_samples <= 0 or len(segments) == 1:
        return np.concatenate(segments)

    pause = np.zeros(pause_samples, dtype=np.float32)
    joined = []
    for index, segment in enumerate(segments):
        joined.append(segment)
        if index < len(segments) - 1:
            joined.append(pause)
    return np.concatenate(joined)


def clear_cuda_cache(torch: Any) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def prepare_environment(request: dict[str, Any]) -> None:
    """设置 Hugging Face、Numba 等缓存目录，避免污染仓库源码目录。"""
    runtime_cache_dir = str(
        request.get("runtime_cache_dir") or Path(__file__).resolve().parent / ".cache/runtime"
    )
    hf_mirror_dir = str(request.get("hf_mirror_dir") or Path.home() / "hf-mirror")
    local_files_only = bool(request.get("local_files_only", True))

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("HF_HOME", hf_mirror_dir)
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(runtime_cache_dir, "hf_modules"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(runtime_cache_dir, "numba"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(runtime_cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(runtime_cache_dir, "xdg"))
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def clear_module_tree(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def should_fallback_to_sidecar(exc: Exception) -> bool:
    return isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) == "qwen_tts"


def load_qwen3_model_class(qwen_libs_path: str | None):
    try:
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel
    except Exception as exc:
        direct_exc = exc

    expanded = None
    if qwen_libs_path:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(qwen_libs_path)))

    if not should_fallback_to_sidecar(direct_exc):
        raise direct_exc
    if not expanded or not os.path.isdir(expanded):
        raise direct_exc

    clear_module_tree("qwen_tts")
    if expanded not in sys.path:
        sys.path.insert(0, expanded)

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception:
        clear_module_tree("qwen_tts")
        raise
    return Qwen3TTSModel


def import_runtime(request: dict[str, Any]):
    """延迟导入 Qwen 运行时，并按配置选择仓库内或 sidecar 实现。"""
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-TTS 基础依赖不可导入。请确认 qwen3_tts uv 环境已安装 qwen-tts、"
            f"torch、numpy、soundfile。缺失导入：{exc.name or exc}"
        ) from exc

    qwen_libs_path = normalize_optional_text(request.get("qwen_libs_path"))
    try:
        Qwen3TTSModel = load_qwen3_model_class(qwen_libs_path)
    except Exception as exc:
        raise RuntimeError(
            "Qwen3-TTS 运行时不可导入。请优先检查 qwen3_tts uv 环境里的 qwen-tts 及其依赖；"
            f"当前错误：{exc}"
        ) from exc

    return Qwen3TTSModel, np, sf, torch


def build_generation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """把可选生成参数整理成模型 ``generate_voice`` 可接受的 kwargs。"""
    kwargs: dict[str, Any] = {"max_new_tokens": int(request.get("max_new_tokens") or 2048)}

    top_p = request.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = float(top_p)

    temperature = request.get("temperature")
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    return kwargs


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    """加载一次 Qwen3-TTS，按片段生成语音、拼接并写出 WAV。"""
    prepare_environment(request)
    Qwen3TTSModel, np, sf, torch = import_runtime(request)

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    ref_text = normalize_optional_text(request.get("ref_text"))
    x_vector_only = bool(request.get("x_vector_only", False)) or ref_text is None
    language = normalize_optional_text(request.get("language")) or "Chinese"
    device_map = normalize_optional_text(request.get("device_map")) or "cuda:0"
    local_files_only = bool(request.get("local_files_only", True))

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS 合成需要 CUDA GPU。")

    device = "cuda"
    dtype = resolve_dtype(torch, request.get("dtype") or "auto", device)
    attn_implementation = resolve_attn_implementation(
        torch,
        request.get("attn_implementation") or "auto",
        device,
        dtype,
    )
    # 0 是 API 约定的“不分片/无停顿”，不能用 ``or`` 误替换为默认值。
    max_chars_per_chunk = int(
        request["max_chars_per_chunk"] if request.get("max_chars_per_chunk") is not None else 120
    )
    pause_ms = int(request["pause_ms"] if request.get("pause_ms") is not None else 250)
    trim_leading_silence_enabled = bool(request.get("trim_leading_silence", True))
    trim_leading_silence_threshold_db = (
        float(request["trim_leading_silence_threshold_db"])
        if request.get("trim_leading_silence_threshold_db") is not None
        else -42.0
    )
    trim_leading_silence_min_ms = (
        int(request["trim_leading_silence_min_ms"])
        if request.get("trim_leading_silence_min_ms") is not None
        else 120
    )
    trim_leading_silence_analysis_window_ms = (
        int(request["trim_leading_silence_analysis_window_ms"])
        if request.get("trim_leading_silence_analysis_window_ms") is not None
        else 30
    )
    trim_leading_silence_pre_roll_ms = (
        int(request["trim_leading_silence_pre_roll_ms"])
        if request.get("trim_leading_silence_pre_roll_ms") is not None
        else 40
    )
    trim_leading_silence_max_ms = (
        int(request["trim_leading_silence_max_ms"])
        if request.get("trim_leading_silence_max_ms") is not None
        else 8000
    )
    chunks = split_text(text, max_chars_per_chunk)

    model = None
    voice_clone_prompt = None
    started = time.perf_counter()
    try:
        print(f"[Qwen3-TTS worker] 模型目录: {model_path}")
        print(f"[Qwen3-TTS worker] 参考音频: {ref_audio_path}")
        print(
            f"[Qwen3-TTS worker] 参考文本: {'provided' if ref_text else 'not provided; using x-vector-only'}"
        )
        print(f"[Qwen3-TTS worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(
            f"[Qwen3-TTS worker] device_map={device_map}, dtype={dtype}, attn={attn_implementation}"
        )
        print(
            f"[Qwen3-TTS worker] trim_leading_silence={trim_leading_silence_enabled}, "
            f"threshold_db={trim_leading_silence_threshold_db}, "
            f"min_ms={trim_leading_silence_min_ms}, "
            f"pre_roll_ms={trim_leading_silence_pre_roll_ms}, "
            f"max_ms={trim_leading_silence_max_ms}"
        )

        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

        generation_kwargs = build_generation_kwargs(request)
        wavs, sample_rate = model.generate_voice_clone(
            text=chunks if len(chunks) > 1 else chunks[0],
            language=[language] * len(chunks) if len(chunks) > 1 else language,
            voice_clone_prompt=voice_clone_prompt,
            **generation_kwargs,
        )
        raw_waveforms = list(wavs) if isinstance(wavs, (list, tuple)) else [wavs]
        waveforms = []
        for index, raw_waveform in enumerate(raw_waveforms, start=1):
            trimmed_waveform, trimmed_samples = trim_leading_silence(
                raw_waveform,
                sample_rate=int(sample_rate),
                enabled=trim_leading_silence_enabled,
                threshold_db=trim_leading_silence_threshold_db,
                min_silence_ms=trim_leading_silence_min_ms,
                analysis_window_ms=trim_leading_silence_analysis_window_ms,
                pre_roll_ms=trim_leading_silence_pre_roll_ms,
                max_trim_ms=trim_leading_silence_max_ms,
                np=np,
            )
            if trimmed_samples > 0:
                print(
                    f"[Qwen3-TTS worker] chunk {index}/{len(raw_waveforms)} "
                    f"裁掉前导静音 {trimmed_samples / int(sample_rate):.2f}s"
                )
            waveforms.append(trimmed_waveform)
        waveform = join_waveforms(waveforms, int(sample_rate), pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(
            waveform,
            sample_rate=int(sample_rate),
            enabled=trim_leading_silence_enabled,
            threshold_db=trim_leading_silence_threshold_db,
            min_silence_ms=trim_leading_silence_min_ms,
            analysis_window_ms=trim_leading_silence_analysis_window_ms,
            pre_roll_ms=trim_leading_silence_pre_roll_ms,
            max_trim_ms=trim_leading_silence_max_ms,
            np=np,
        )
        if trimmed_samples > 0:
            print(
                f"[Qwen3-TTS worker] 最终音频裁掉前导静音 {trimmed_samples / int(sample_rate):.2f}s"
            )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, int(sample_rate))
        elapsed = time.perf_counter() - started
        print(
            f"[Qwen3-TTS worker] 完成: sample_rate={int(sample_rate)}, "
            f"elapsed={elapsed:.2f}s, output={output_wav}"
        )
    finally:
        if voice_clone_prompt is not None:
            try:
                del voice_clone_prompt
            except Exception:
                pass
        if model is not None:
            try:
                del model
            except Exception:
                pass
        clear_cuda_cache(torch)


def main() -> int:
    """执行 worker 入口，并把异常转换为非零退出码。"""
    args = parse_args()
    request = load_request(args.input_json)
    try:
        synthesize(request, Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
