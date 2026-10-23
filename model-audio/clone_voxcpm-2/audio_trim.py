"""用于清理生成 WAV 输出的轻量音频工具。"""

from __future__ import annotations

# 仅在 worker 中处理 numpy 波形，API 进程不需要加载音频分析依赖。
from typing import Any

LEADING_SILENCE_THRESHOLD_DB = -42.0
LEADING_SILENCE_MIN_MS = 120
LEADING_SILENCE_ANALYSIS_WINDOW_MS = 30
LEADING_SILENCE_PRE_ROLL_MS = 40
LEADING_SILENCE_MAX_TRIM_MS = 8000


def trim_leading_silence(
    waveform: Any,
    sample_rate: int,
    np: Any,
    *,
    threshold_db: float = LEADING_SILENCE_THRESHOLD_DB,
    min_silence_ms: int = LEADING_SILENCE_MIN_MS,
    analysis_window_ms: int = LEADING_SILENCE_ANALYSIS_WINDOW_MS,
    pre_roll_ms: int = LEADING_SILENCE_PRE_ROLL_MS,
    max_trim_ms: int = LEADING_SILENCE_MAX_TRIM_MS,
) -> tuple[Any, int]:
    """用分窗能量裁剪前导静音，并限制最大裁剪长度避免误删起音。

    输入可以是单声道、``(frames, channels)`` 或 ``(channels, frames)``；
    函数会保留原始布局，调用方可以直接将结果写回 WAV。短于
    ``min_silence_ms`` 的前导部分保持不变，避免切掉自然语音起音。
    """

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0 or audio.ndim not in {1, 2}:
        return audio, 0

    time_axis = 0
    if audio.ndim == 1:
        analysis_audio = audio
    elif audio.shape[0] >= audio.shape[1]:
        analysis_audio = audio.mean(axis=1)
    else:
        time_axis = 1
        analysis_audio = audio.mean(axis=0)

    analysis_window = max(1, int(sample_rate * max(analysis_window_ms, 1) / 1000))
    threshold = float(10 ** (threshold_db / 20.0))
    power = np.square(analysis_audio, dtype=np.float32)
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

    trimmed_audio = audio[trim_start:] if time_axis == 0 else audio[:, trim_start:]
    if trimmed_audio.size == 0:
        return audio, 0
    return trimmed_audio, trim_start
