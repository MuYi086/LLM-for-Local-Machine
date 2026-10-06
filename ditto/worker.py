#!/usr/bin/env python3
"""一次性 Ditto worker；所有模型、Torch 和 ONNX 依赖只在此进程加载。"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def prepare_warp_model(source: Path, target: Path) -> None:
    """用官方 ONNX 转换器把原始三维 GridSample 图升级为 opset 20。"""
    import onnx

    model = onnx.load(str(source))
    if not any(node.op_type == "GridSample" for node in model.graph.node):
        raise ValueError("Ditto 需要包含标准 GridSample 的 warp_network_ori.onnx。")
    # opset 17 的 GridSample 只支持二维；官方转换器升级其他算子并将 bilinear 改为 linear，
    # 让五维输入使用标准三维采样，同时保持 padding_mode 和 align_corners 语义。
    converted = onnx.version_converter.convert_version(model, 20)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(target))


def build_runtime_config(payload: dict[str, Any], job: Path) -> Path:
    """只读取运维配置的本地 pickle，把 TensorRT 文件路径换为当前 ONNX 权重。"""
    with Path(payload["config_path"]).open("rb") as source:
        config = pickle.load(source)
    if config["audio2motion_cfg"]["w2f_type"] != "hubert":
        raise ValueError("本地 Ditto 权重只支持 HuBERT 配置。")
    root = Path(payload["data_root"])
    warp = job / "warp.onnx"
    prepare_warp_model(root / payload["onnx_components"]["warp_network_cfg"], warp)
    base = config["base_cfg"]
    # parse_cfg 会遍历全部 base_cfg；删除未使用且镜像中不存在的 WavLM 路径。
    base.pop("wavlm_cfg", None)
    for name, filename in payload["onnx_components"].items():
        base[name]["model_path"] = str(warp if name == "warp_network_cfg" else root / filename)
        base[name]["device"] = "cuda"
        # ONNX wrapper 的构造函数只接受 model_path/device；不传 PyTorch 后端专用参数。
        base[name].pop("force_ori_type", None)
    base["landmark478_cfg"] = {
        "blaze_face_model_path": str(root / "blaze_face.onnx"),
        "face_mesh_model_path": str(root / "face_mesh.onnx"),
        "device": "cuda",
    }
    config["audio2motion_cfg"].update(
        model_path=str(root / "lmdm_v0.4_hubert.onnx"),
        device="cuda",
    )
    config["default_kwargs"]["online_mode"] = False
    path = job / "config.pkl"
    with path.open("wb") as destination:
        pickle.dump(config, destination)
        destination.flush()
        os.fsync(destination.fileno())
    return path


def run_sdk(
    sdk: Any, audio: Any, image_path: str, output_path: Path, payload: dict[str, Any]
) -> None:
    """按官方离线流程排入音频特征，并在失败时停止线程和编码器。"""
    try:
        sdk.setup(
            image_path,
            str(output_path),
            online_mode=False,
            sampling_timesteps=payload["sampling_timesteps"],
            max_size=payload["max_size"],
        )
        sdk.setup_Nd(N_d=math.ceil(len(audio) / 16000 * 25))
        sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
        sdk.close()
    except BaseException:
        # close 可能因上游队列阻塞而无法返回；失败时不再调用 close，由父进程超时兜底。
        if hasattr(sdk, "stop_event"):
            sdk.stop_event.set()
        for thread in getattr(sdk, "thread_list", []):
            thread.join(timeout=2)
        writer = getattr(sdk, "writer", None)
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"[Ditto worker] 编码器清理失败: {exc}", file=sys.stderr)
        raise


def configure_model_loader() -> None:
    """检查每个模型实际启用的 provider，阻止动态库加载失败后的静默 CPU 回退。"""
    loader = importlib.import_module("core.utils.load_model")
    original = loader.load_model

    def load_cuda_model(*args: Any, **kwargs: Any) -> Any:
        model, model_type = original(*args, **kwargs)
        if model_type == "onnx" and "CUDAExecutionProvider" not in model.get_providers():
            raise RuntimeError(f"Ditto ONNX 模型未启用 CUDA: {args[0]}")
        return model, model_type

    loader.load_model = load_cuda_model


def generate_video(payload: dict[str, Any], output_path: Path) -> None:
    """加载官方 StreamSDK，以可检查退出码的 FFmpeg 标准化输入并合并音频。"""
    import librosa
    import numpy as np
    import onnxruntime
    import pyximport
    import torch

    if (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in onnxruntime.get_available_providers()
    ):
        raise RuntimeError("Ditto 需要可用的 Torch CUDA 与 ONNX Runtime CUDA 后端。")
    # 先加载 Torch 的 CUDA/cuDNN，再让 ORT 复用同一套动态库，避免 provider 静默退回 CPU。
    onnxruntime.preload_dlls()
    seed = payload["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    code_path = Path(payload["code_path"]).resolve()
    sys.path.insert(0, str(code_path))
    ffmpeg = payload["ffmpeg_bin"]
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 上游 blend 在 import 时编译 Cython；缓存跟随请求目录一起清理，不写入用户家目录。
    pyximport.install(
        build_dir=str(output_path.parent / "pyxbld"), setup_args={"include_dirs": np.get_include()}
    )
    normalized_audio = output_path.with_name("driver.wav")
    silent_video = Path(str(output_path) + ".tmp.mp4")
    config_path = output_path.with_name("config.pkl")
    warp_path = output_path.with_name("warp.onnx")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                payload["reference_audio_path"],
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(normalized_audio),
            ],
            check=True,
        )
        audio, _ = librosa.load(str(normalized_audio), sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("驱动音频为空。")
        config_path = build_runtime_config(payload, output_path.parent)
        configure_model_loader()
        pipeline = importlib.import_module("stream_pipeline_offline")
        sdk = pipeline.StreamSDK(str(config_path), payload["data_root"])
        run_sdk(sdk, audio, payload["reference_image_path"], output_path, payload)
        # 不调用官方 inference.run 的 os.system；显式检查编码失败并保留原始驱动音频。
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                payload["reference_audio_path"],
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
    finally:
        for temporary in (normalized_audio, silent_video, config_path, warp_path):
            temporary.unlink(missing_ok=True)


def main() -> None:
    """读取父进程生成的请求，离线加载权重并写入临时 MP4。"""
    parser = argparse.ArgumentParser(description="Ditto one-shot worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-video", required=True)
    args = parser.parse_args()
    with Path(args.input_json).open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("local_files_only", True):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    generate_video(payload, Path(args.output_video).resolve())


if __name__ == "__main__":
    main()
