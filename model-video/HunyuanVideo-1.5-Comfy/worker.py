#!/usr/bin/env python3
"""一次性 ComfyUI worker；ComfyUI 与模型仅在该进程组中启动。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from hunyuan_runtime import DEFAULT_HEIGHT, DEFAULT_WIDTH, validate_output_dimensions

MODEL_FILES = {
    "diffusion_model": "hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
    "qwen_encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "byt5_encoder": "byt5_small_glyphxl_fp16.safetensors",
    "clip_vision": "sigclip_vision_patch14_384.safetensors",
    "vae": "hunyuanvideo15_vae_fp16.safetensors",
}


def parse_args() -> argparse.Namespace:
    """解析父进程请求 JSON 和临时 MP4 目标路径。"""
    parser = argparse.ArgumentParser(description="HunyuanVideo-1.5 Comfy one-shot worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-video", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并验证 worker 请求对象。"""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("worker 输入 JSON 顶层必须是对象。")
    return payload


def tail_text(path: Path, limit: int = 16_000) -> str:
    """读取 ComfyUI 日志末尾，限制错误消息长度。"""
    try:
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - limit))
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def require_file(path: str | Path, label: str) -> Path:
    """确认必要文件存在并返回绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}不存在: {resolved}")
    return resolved


def choose_free_port() -> int:
    """向系统申请临时 loopback 端口，避免 ComfyUI 暴露到外网。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def model_path_config(model_dir: Path, work_dir: Path) -> Path:
    """把仓库外的分块权重目录映射为 ComfyUI 模型类别。"""
    paths = {
        "clip_vision": "split_files/clip_vision",
        "diffusion_models": "split_files/diffusion_models",
        "text_encoders": "split_files/text_encoders",
        "vae": "split_files/vae",
    }
    lines = ["hunyuanvideo15:", f"  base_path: {json.dumps(str(model_dir))}"]
    lines.extend(f"  {key}: {json.dumps(relative)}" for key, relative in paths.items())
    config_path = work_dir / "extra_model_paths.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def build_workflow(payload: dict[str, Any], image_name: str, prefix: str) -> dict[str, Any]:
    """构建 HunyuanVideo-1.5 480p step-distilled 图生视频 ComfyUI API 图。"""
    prompt = str(payload["prompt"]).strip()
    if not prompt:
        raise ValueError("prompt 不能为空。")
    negative_prompt = str(payload.get("negative_prompt") or "")
    steps = int(payload.get("steps", 8))
    width = int(payload.get("width", DEFAULT_WIDTH))
    height = int(payload.get("height", DEFAULT_HEIGHT))
    frames = int(payload.get("num_frames", 121))
    validate_output_dimensions(width, height)
    if steps not in {8, 12} or frames not in {49, 81, 97, 121}:
        raise ValueError("step-distilled 模型只允许 8/12 步和 49/81/97/121 帧。")
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": MODEL_FILES["diffusion_model"], "weight_dtype": "default"},
        },
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 7}},
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": MODEL_FILES["qwen_encoder"],
                "clip_name2": MODEL_FILES["byt5_encoder"],
                "type": "hunyuan_video_15",
                "device": "default",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["vae"]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": MODEL_FILES["clip_vision"]},
        },
        "7": {
            "class_type": "CLIPVisionEncode",
            "inputs": {"clip_vision": ["6", 0], "image": ["5", 0], "crop": "center"},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": prompt},
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": negative_prompt},
        },
        "10": {
            "class_type": "HunyuanVideo15ImageToVideo",
            "inputs": {
                "positive": ["8", 0],
                "negative": ["9", 0],
                "vae": ["4", 0],
                "start_image": ["5", 0],
                "clip_vision_output": ["7", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "seed": int(payload.get("seed", 42)),
                "steps": steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
        },
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": 24}},
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "h264",
            },
        },
    }


def http_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    """通过 ComfyUI loopback API 发送 JSON 请求。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ComfyUI API 返回 HTTP {exc.code}: {detail}") from exc


def wait_for_server(base_url: str, process: subprocess.Popen, log_path: Path) -> None:
    """等待 ComfyUI API 就绪，服务启动失败时附带末尾日志。"""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI 在启动时退出。\n{tail_text(log_path)}")
        try:
            http_json(f"{base_url}/system_stats", timeout=2)
            return
        except (OSError, RuntimeError, ValueError):
            time.sleep(1)
    raise TimeoutError(f"等待 ComfyUI API 超时。\n{tail_text(log_path)}")


def queue_and_wait(
    base_url: str,
    workflow: dict[str, Any],
    process: subprocess.Popen,
    log_path: Path,
    timeout: float,
) -> None:
    """提交 workflow 并轮询历史，识别 ComfyUI 执行失败或服务异常退出。"""
    queued = http_json(f"{base_url}/prompt", body={"prompt": workflow}, timeout=30)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI 没有返回 prompt_id: {queued}")
    deadline = time.monotonic() + timeout
    history_url = f"{base_url}/history/{urllib.parse.quote(str(prompt_id))}"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI 在生成期间退出。\n{tail_text(log_path)}")
        history = http_json(history_url, timeout=5)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            status_text = str(status.get("status_str", "")).lower()
            messages = status.get("messages", [])
            if status_text == "error" or any("execution_error" in str(item) for item in messages):
                raise RuntimeError(f"ComfyUI workflow 执行失败: {messages}\n{tail_text(log_path)}")
            if entry.get("outputs"):
                if "14" not in entry["outputs"]:
                    raise RuntimeError("ComfyUI workflow 完成但没有 SaveVideo 输出。")
                return
        time.sleep(1)
    raise TimeoutError(f"ComfyUI workflow 超时（>{timeout:.0f}s）。\n{tail_text(log_path)}")


def generate_video(payload: dict[str, Any], output_path: Path) -> None:
    """启动隔离的 ComfyUI API 服务，执行 workflow 并收集唯一 MP4。"""
    comfyui = require_file(Path(payload["comfyui_code_path"]) / "main.py", "ComfyUI 入口")
    reference = require_file(payload["reference_image_path"], "参考图片")
    model_dir = Path(payload["model_dir"]).expanduser().resolve()
    for category, filename in MODEL_FILES.items():
        relative_dir = {
            "diffusion_model": "split_files/diffusion_models",
            "qwen_encoder": "split_files/text_encoders",
            "byt5_encoder": "split_files/text_encoders",
            "clip_vision": "split_files/clip_vision",
            "vae": "split_files/vae",
        }[category]
        require_file(model_dir / relative_dir / filename, f"{category} 权重")

    work_root = Path(payload["worker_tmp_dir"]).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hunyuanvideo15_", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        temp_dir = work_dir / "temp"
        user_dir = work_dir / "user"
        for directory in (input_dir, output_dir, temp_dir, user_dir):
            directory.mkdir()
        image_name = f"input_{uuid.uuid4().hex}{reference.suffix.lower()}"
        shutil.copyfile(reference, input_dir / image_name)
        paths_config = model_path_config(model_dir, work_dir)
        port = choose_free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = work_dir / "comfyui.log"
        command = [
            sys.executable,
            str(comfyui),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--input-directory",
            str(input_dir),
            "--output-directory",
            str(output_dir),
            "--temp-directory",
            str(temp_dir),
            "--user-directory",
            str(user_dir),
            "--extra-model-paths-config",
            str(paths_config),
            "--disable-auto-launch",
            "--disable-metadata",
            "--disable-all-custom-nodes",
        ]
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_visible_devices"])
        if payload.get("local_files_only", True):
            worker_env["HF_HUB_OFFLINE"] = "1"
            worker_env["TRANSFORMERS_OFFLINE"] = "1"
        server: subprocess.Popen | None = None
        try:
            with log_path.open("wb") as log_file:
                server = subprocess.Popen(
                    command,
                    cwd=str(Path(payload["comfyui_code_path"]).resolve()),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=worker_env,
                )
                wait_for_server(base_url, server, log_path)
                filename_prefix = f"hunyuan_video_1_5/{uuid.uuid4().hex}"
                workflow = build_workflow(payload, image_name, filename_prefix)
                queue_and_wait(
                    base_url,
                    workflow,
                    server,
                    log_path,
                    float(payload.get("request_timeout", 2400)),
                )
            videos = list(output_dir.rglob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"ComfyUI 应生成一个 MP4，实际找到 {len(videos)} 个。")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[0], output_path)
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def main() -> None:
    """读取请求并启动一次性 ComfyUI 服务完成生成。"""
    args = parse_args()
    payload = load_request(args.input_json)
    if payload.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output_path = Path(args.output_video).expanduser().resolve()
    generate_video(payload, output_path)


if __name__ == "__main__":
    main()
