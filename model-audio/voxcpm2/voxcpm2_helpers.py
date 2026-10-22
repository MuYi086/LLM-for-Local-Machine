"""一次性 worker 使用的 VoxCPM2 运行时共用辅助函数。

重型运行时依赖采用延迟导入，因此 API 进程和单元测试不需要加载 VoxCPM2
模型运行时。
"""

from __future__ import annotations

# 这些 helper 在 worker 中延迟导入官方 VoxCPM2 依赖，API 和无模型测试不会触发它们。
import inspect
import random
from pathlib import Path
from typing import Any


def import_runtime():
    """导入 numpy、soundfile、torch 和 VoxCPM2 官方包。"""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 runtime 无法导入。请确认 voxcpm2 uv 环境已安装 "
            f"voxcpm、torch、numpy 和 soundfile。缺少依赖：{exc.name or exc}"
        ) from exc
    return VoxCPM, np, sf, torch


def set_seed(seed: int, np: Any, torch: Any) -> None:
    """设置 numpy、Python 和 torch 的随机种子。"""
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def from_pretrained_kwargs(VoxCPM: Any, args: Any) -> dict[str, Any]:
    """根据官方构造函数签名筛选兼容的加载参数。"""
    signature = inspect.signature(VoxCPM.from_pretrained)
    options = {
        "load_denoiser": args.load_denoiser,
        "local_files_only": args.local_files_only,
        "optimize": args.optimize,
    }
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def apply_control_instruction(text: str, control_instruction: str | None) -> str:
    """把 controllable 模式的指令追加到模型文本。"""
    instruction = (control_instruction or "").strip()
    return f"({instruction}){text}" if instruction else text


def apply_nonverbal_tags(text: str, nonverbal_tags: list[str] | None) -> str:
    """按配置补充笑声、停顿等非语言标签。"""
    tags = nonverbal_tags or []
    return "".join(f"[{tag}]" for tag in tags) + text


def build_model_text(
    chunk: str, control_instruction: str | None, nonverbal_tags: list[str] | None
) -> str:
    """按官方可控克隆格式组装最终文本：(instruction)[tag]正文。"""
    return apply_control_instruction(
        apply_nonverbal_tags(chunk, nonverbal_tags),
        control_instruction,
    )


def generate_kwargs(
    model: Any,
    args: Any,
    chunk: str,
    ref_audio: Path,
    prompt_text: str | None,
) -> dict[str, Any]:
    """从请求参数生成官方推理函数可接受的 kwargs。"""
    options = {
        # VoxCPM2 将可控克隆指令写在目标文本前；Ultimate Cloning 则由 prompt_* 参数决定。
        "text": build_model_text(
            chunk,
            getattr(args, "control_instruction", None),
            getattr(args, "nonverbal_tags", None),
        ),
        "reference_wav_path": str(ref_audio),
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "normalize": getattr(args, "normalize", False),
        "denoise": getattr(args, "denoise", False),
        "retry_badcase": getattr(args, "retry_badcase", True),
    }
    if prompt_text is not None:
        options["prompt_text"] = prompt_text
        options["prompt_wav_path"] = str(ref_audio)

    signature = inspect.signature(model.generate)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def to_mono_float32(waveform: Any, np: Any):
    """将各种声道布局统一为单声道 float32 波形。"""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any):
    """用静音连接多个分段波形。"""
    if not waveforms:
        raise RuntimeError("VoxCPM2 未返回任何音频片段。")

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


def resolve_sample_rate(model: Any) -> int:
    """从模型配置中解析采样率，并在缺失时抛出明确错误。"""
    tts_model = getattr(model, "tts_model", None)
    sample_rate = getattr(tts_model, "sample_rate", None)
    if sample_rate is None:
        raise RuntimeError("无法从 model.tts_model.sample_rate 获取 VoxCPM2 采样率。")
    return int(sample_rate)
