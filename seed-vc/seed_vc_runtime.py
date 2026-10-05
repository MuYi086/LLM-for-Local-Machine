"""Seed-VC 的无模型进程清理与原子输出工具。"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def read_inference_config(config_path: Path, f0_condition: bool) -> dict[str, Any]:
    """在 CPU 上校验配置与所选模式，避免进入 GPU 队列后才发现不匹配。"""
    import yaml

    try:
        with config_path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        parameters = config["model_params"]
        preprocessing = config["preprocess_params"]
        expected_sr, expected_hop = (44100, 512) if f0_condition else (22050, 256)
        valid = (
            parameters["speech_tokenizer"]["type"] == "whisper"
            and parameters["vocoder"]["type"] == "bigvgan"
            and preprocessing["sr"] == expected_sr
            and preprocessing["spect_params"]["hop_length"] == expected_hop
            and bool(parameters["length_regulator"].get("f0_condition", False)) == f0_condition
            and bool(parameters["DiT"].get("f0_condition", False)) == f0_condition
        )
    except (AttributeError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Seed-VC 配置结构不完整: {config_path}") from exc
    if not valid:
        raise ValueError(
            "Seed-VC 配置必须使用 Whisper + BigVGAN，采样率和 F0 条件须与请求模式匹配。"
        )
    return config


def terminate_process_group(process: Any, timeout: float = 10) -> None:
    """终止整个 worker 会话，包括主进程已退出后仍存活的子进程。"""
    if process is None:
        return
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if pid is None:
            process.kill()
    finally:
        # 主进程退出不代表其子进程释放了 CUDA；一次性任务结束后必须清理整个组。
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def persist_audio_bytes(audio_bytes: bytes, output_dir: Path) -> Path:
    """用临时文件加原子替换保存转换结果，避免暴露不完整的 WAV。"""
    if not audio_bytes:
        raise ValueError("不能保存空音频。")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".seed_vc_{time.strftime('%Y%m%d_%H%M%S')}_", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    output_path = output_dir / f"{temporary_path.name[1:-4]}.wav"
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(audio_bytes)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
        return output_path
    finally:
        temporary_path.unlink(missing_ok=True)
