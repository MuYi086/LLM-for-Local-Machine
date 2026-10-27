#!/usr/bin/env python3
"""一次性 Wan2.2 worker；ComfyUI 与模型仅在该进程组内运行。"""

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

MODEL_FILES = {
    "diffusion_model": "Wan2.2-TI2V-5B-Q8_0.gguf",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
}


def parse_args() -> argparse.Namespace:
    """解析父进程请求 JSON 与临时 MP4 目标路径。"""
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V GGUF one-shot worker")
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
    """读取 ComfyUI 日志末尾，限制错误响应长度。"""
    try:
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - limit))
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def require_file(path: str | Path, label: str) -> Path:
    """确认文件存在并返回绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}不存在: {resolved}")
    return resolved


def require_dir(path: str | Path, label: str) -> Path:
    """确认目录存在并返回绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label}不存在: {resolved}")
    return resolved


def choose_free_port() -> int:
    """申请临时 loopback 端口，ComfyUI API 不对外监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def model_path_config(model_dir: Path, work_dir: Path) -> Path:
    """映射外置 Wan 权重，并把临时节点目录作为 ComfyUI base directory。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "wan22:",
        f"  base_path: {json.dumps(str(model_dir))}",
        "  diffusion_models: .",
        "  text_encoders: .",
        "  vae: .",
    ]
    config_path = work_dir / "extra_model_paths.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def build_workflow(payload: dict[str, Any], image_name: str | None, prefix: str) -> dict[str, Any]:
    """构建官方 Wan2.2 TI2V 5B 采样参数的 ComfyUI API workflow。"""
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt 不能为空。")
    negative_prompt = str(payload.get("negative_prompt") or "")
    width = int(payload.get("width", 832))
    height = int(payload.get("height", 480))
    frames = int(payload.get("num_frames", 49))
    steps = int(payload.get("steps", 20))
    seed = int(payload.get("seed", 42))
    fps = int(payload.get("fps", 24))
    if width % 32 or height % 32 or not (256 <= width <= 1280 and 256 <= height <= 1280):
        raise ValueError("width/height 必须是 256 到 1280 之间的 32 倍数。")
    if width * height > 1280 * 704:
        raise ValueError("宽高面积不能超过 1280×704。")
    if frames not in {49, 81, 121} or not 1 <= steps <= 50 or not 0 <= seed <= 2**64 - 1:
        raise ValueError("Wan2.2 帧数、步数或随机种子超出允许范围。")
    if not 1 <= fps <= 60:
        raise ValueError("fps 必须在 1 到 60 之间。")

    latent_inputs: dict[str, Any] = {
        "vae": ["4", 0],
        "width": width,
        "height": height,
        "length": frames,
        "batch_size": 1,
    }
    if image_name:
        latent_inputs["start_image"] = ["5", 0]

    workflow = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": MODEL_FILES["diffusion_model"]},
        },
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8}},
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODEL_FILES["text_encoder"],
                "type": "wan",
                "device": "cpu",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["vae"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": negative_prompt},
        },
        "8": {"class_type": "Wan22ImageToVideoLatent", "inputs": latent_inputs},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 5,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "10": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["4", 0],
                "tile_size": 256,
                "overlap": 64,
                "temporal_size": 32,
                "temporal_overlap": 8,
            },
        },
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": fps}},
        "12": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["11", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if image_name:
        workflow["5"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    return workflow


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
        detail = exc.read().decode("utf-8", errors="replace")[-2_000:]
        raise RuntimeError(f"ComfyUI API 返回 HTTP {exc.code}: {detail}") from exc


def wait_for_server(base_url: str, process: subprocess.Popen, log_path: Path) -> None:
    """等待 ComfyUI API 就绪，并附带启动失败日志。"""
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
    """提交 workflow 并轮询执行历史，识别节点错误和子进程退出。"""
    queued = http_json(f"{base_url}/prompt", body={"prompt": workflow}, timeout=30)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI 没有返回 prompt_id: {queued}")
    history_url = f"{base_url}/history/{urllib.parse.quote(str(prompt_id))}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI 在生成期间退出。\n{tail_text(log_path)}")
        history = http_json(history_url, timeout=5)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            messages = status.get("messages", [])
            status_text = str(status.get("status_str", "")).lower()
            if status_text == "error" or any("execution_error" in str(item) for item in messages):
                raise RuntimeError(f"ComfyUI workflow 执行失败: {messages}\n{tail_text(log_path)}")
            if entry.get("outputs"):
                if "12" not in entry["outputs"]:
                    raise RuntimeError("ComfyUI workflow 完成但没有 SaveVideo 输出。")
                return
        time.sleep(1)
    raise TimeoutError(f"ComfyUI workflow 超时（>{timeout:.0f}s）。\n{tail_text(log_path)}")


def generate_video(payload: dict[str, Any], output_path: Path) -> None:
    """在临时目录启动单个白名单 GGUF 节点的 ComfyUI 并生成视频。"""
    comfyui = require_file(Path(payload["comfyui_code_path"]) / "main.py", "ComfyUI 入口")
    model_dir = require_dir(payload["model_dir"], "Wan2.2 模型目录")
    node_dir = require_dir(payload["gguf_node_path"], "ComfyUI-GGUF 节点源码")
    for key, filename in MODEL_FILES.items():
        require_file(model_dir / filename, f"{key} 权重")
    image_path: Path | None = None
    if payload.get("reference_image_path"):
        image_path = require_file(payload["reference_image_path"], "参考图片")

    work_root = Path(payload["worker_tmp_dir"]).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wan22_ti2v_", dir=work_root) as temporary:
        work_dir = Path(temporary)
        for directory in ("input", "output", "temp", "user", "custom_nodes"):
            (work_dir / directory).mkdir()
        node_link = work_dir / "custom_nodes" / "ComfyUI-GGUF"
        node_link.symlink_to(node_dir, target_is_directory=True)
        image_name: str | None = None
        if image_path is not None:
            image_name = f"input_{uuid.uuid4().hex}{image_path.suffix.lower()}"
            shutil.copyfile(image_path, work_dir / "input" / image_name)
        paths_config = model_path_config(model_dir, work_dir)
        port = choose_free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = work_dir / "comfyui.log"
        command = [
            sys.executable,
            str(comfyui),
            "--base-directory",
            str(work_dir),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--input-directory",
            str(work_dir / "input"),
            "--output-directory",
            str(work_dir / "output"),
            "--temp-directory",
            str(work_dir / "temp"),
            "--user-directory",
            str(work_dir / "user"),
            "--extra-model-paths-config",
            str(paths_config),
            "--disable-auto-launch",
            "--disable-metadata",
            "--disable-assets",
            "--offline",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "ComfyUI-GGUF",
            "--disable-dynamic-vram",
            "--lowvram",
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
                workflow = build_workflow(payload, image_name, f"wan22_ti2v/{uuid.uuid4().hex}")
                queue_and_wait(
                    base_url,
                    workflow,
                    server,
                    log_path,
                    float(payload.get("request_timeout", 3600)),
                )
            videos = list((work_dir / "output").rglob("*.mp4"))
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


def main() -> int:
    """载入请求并启动一次性 ComfyUI 服务完成视频生成。"""
    args = parse_args()
    payload = load_request(args.input_json)
    if payload.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    generate_video(payload, Path(args.output_video).expanduser().resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
