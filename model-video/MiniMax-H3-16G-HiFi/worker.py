#!/usr/bin/env python3
"""MiniMax H3 一次性推理 worker，通过本地 ComfyUI API 生成 MP4。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 worker 请求 JSON 和结果视频路径。"""
    parser = argparse.ArgumentParser(description="MiniMax H3 one-shot ComfyUI worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-video", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并确认 worker 请求 JSON 顶层为对象。"""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("worker 输入 JSON 顶层必须是对象。")
    return payload


def build_api_prompt(
    payload: dict[str, Any], image_names: dict[str, str], filename_prefix: str
) -> dict[str, dict[str, Any]]:
    """构造 H3 ComfyUI 工作流，并按请求可选解码和封装原生音频。"""
    include_audio = payload.get("include_audio", False)
    if not isinstance(include_audio, bool):
        raise ValueError("include_audio 必须是布尔值。")
    prompt: dict[str, dict[str, Any]] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": payload["dit_filename"], "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_text_encoder.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
        },
        "6": {
            "class_type": "MiniMaxH3MemoryEfficientSolAttentionPatch",
            "inputs": {
                "model": ["1", 0],
                "enabled": True,
                "tau": 1.0,
                "min_tokens": 4096,
                "strict": False,
                "thresh_type": "diag",
                "int8_qk": False,
                "int8_pv": False,
                "sink_conditioning": "exact_kv",
                "dense_blocks": "",
            },
        },
        "7": {
            "class_type": "MiniMaxH3FusedModulation",
            "inputs": {"model": ["6", 0], "enabled": True},
        },
        "8": {
            "class_type": "MiniMaxH3ChunkFeedForward",
            "inputs": {"model": ["7", 0], "enabled": True, "chunks": 4, "min_tokens": 8192},
        },
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "10": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["8", 0],
                "scheduler": "simple",
                "steps": payload["steps"],
                "denoise": 1.0,
            },
        },
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": payload["seed"]}},
        "12": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["5", 0]},
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["12", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["5", 1],
            },
        },
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "16": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["14", 0], "fps": payload["frame_rate"]},
        },
        "17": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["16", 0],
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if include_audio:
        prompt["4"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
        }
        prompt["15"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["13", 0], "vae": ["4", 0]},
        }
        prompt["16"]["inputs"]["audio"] = ["15", 0]

    image_node_id = 20
    if payload["mode"] == "fl2va":
        conditioning_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": payload["prompt"],
            "width": payload["width"],
            "height": payload["height"],
            "length": payload["frame_count"],
        }
        for role, input_name in (("first", "first_frame"), ("last", "last_frame")):
            if role in image_names:
                node_id = str(image_node_id)
                prompt[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": image_names[role]},
                }
                conditioning_inputs[input_name] = [node_id, 0]
                image_node_id += 1
        prompt["5"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": conditioning_inputs,
        }
    elif payload["mode"] == "ref2va":
        conditioning_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": payload["prompt"],
            "width": payload["width"],
            "height": payload["height"],
            "length": payload["frame_count"],
            "ref_image_size": "match",
        }
        if include_audio:
            conditioning_inputs["audio_vae"] = ["4", 0]
        for index in range(len(payload["reference_image_paths"])):
            role = f"ref_{index}"
            if role not in image_names:
                raise ValueError(f"缺少 Ref2VA 第 {index + 1} 张参考图片。")
            node_id = str(image_node_id)
            prompt[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": image_names[role]},
            }
            conditioning_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
            image_node_id += 1
        prompt["5"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": conditioning_inputs,
        }
    else:
        raise ValueError(f"不支持的 MiniMax H3 mode: {payload['mode']}")

    return prompt


def _free_port() -> int:
    """申请一个本机临时端口，供本次私有 ComfyUI 子进程使用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _api_request(
    base_url: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 10,
) -> bytes:
    """向当前 worker 私有的 ComfyUI loopback API 发出请求。"""
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(base_url + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API {path} 返回 HTTP {exc.code}: {detail[:2000]}") from exc


def _multipart_image(name: str, path: Path) -> tuple[bytes, str]:
    """按 ComfyUI upload/image 接口编码单张隔离目录内的图片。"""
    boundary = "----minimax-h3-" + uuid.uuid4().hex
    filename = path.name
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    chunks = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        path.read_bytes(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n',
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _upload_images(base_url: str, payload: dict[str, Any]) -> dict[str, str]:
    """将受控图片副本上传至当前请求专属的 ComfyUI 输入目录。"""
    image_names: dict[str, str] = {}
    for role, source_value in payload["image_paths"].items():
        source = Path(source_value)
        filename = f"h3_{uuid.uuid4().hex}_{role}{source.suffix.lower()}"
        body, content_type = _multipart_image(filename, source)
        response = _api_request(
            base_url,
            "/upload/image",
            body=body,
            content_type=content_type,
            timeout=60,
        )
        result = json.loads(response)
        if not result.get("name"):
            raise RuntimeError("ComfyUI 未返回已上传图片名称。")
        image_names[role] = str(result["name"])
    return image_names


def _tail(path: Path, limit: int = 8_000) -> str:
    """读取本次 ComfyUI 日志尾部，限制返回错误信息长度。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _wait_for_comfyui(
    process: subprocess.Popen, base_url: str, timeout: float, log_path: Path
) -> None:
    """等待 ComfyUI loopback API 就绪，或在退出/超时时返回诊断信息。"""
    deadline = time.monotonic() + timeout
    last_error = "尚未响应"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI 在启动阶段退出（code={process.returncode}）：{_tail(log_path)}"
            )
        try:
            _api_request(base_url, "/system_stats", timeout=3)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(
        f"ComfyUI 启动超时（>{timeout:.0f}s，最后错误：{last_error}）：{_tail(log_path)}"
    )


def _require_nodes(base_url: str, node_names: tuple[str, ...]) -> None:
    """确认核心 H3 节点和配套扩展都已成功注册。"""
    for node_name in node_names:
        response = json.loads(_api_request(base_url, f"/object_info/{node_name}", timeout=10))
        if node_name not in response:
            raise RuntimeError(f"ComfyUI 未注册工作流依赖节点: {node_name}")


def _wait_for_prompt(
    base_url: str,
    prompt_id: str,
    timeout: float,
    process: subprocess.Popen,
    log_path: Path,
) -> None:
    """轮询 ComfyUI 请求历史，直到结束、子进程异常或超过生成时限。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI 推理期间退出（code={process.returncode}）：{_tail(log_path)}"
            )
        try:
            response = _api_request(base_url, f"/history/{prompt_id}", timeout=10)
            history = json.loads(response).get(prompt_id)
        except (OSError, RuntimeError, json.JSONDecodeError, urllib.error.URLError):
            history = None
        if history:
            status = history.get("status", {})
            status_text = str(status.get("status_str", "")).lower()
            if status.get("completed") or status_text in {"success", "completed"}:
                return
            if status_text in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"ComfyUI 工作流执行失败：{_tail(log_path)}")
        time.sleep(2)
    raise RuntimeError(f"MiniMax H3 推理超时（>{timeout:.0f}s）：{_tail(log_path)}")


def run_comfyui(payload: dict[str, Any], output_path: Path) -> None:
    """为一次请求启动私有 ComfyUI，生成带音频 MP4 后回收子进程和临时文件。"""
    comfy_root = Path(payload["comfyui_code_path"])
    model_dir = Path(payload["model_dir"])
    worker_tmp_dir = Path(payload["worker_tmp_dir"])
    worker_tmp_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(
        prefix=f"minimax_h3_{request_id}_", dir=worker_tmp_dir
    ) as temp:
        work_dir = Path(temp)
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        temp_dir = work_dir / "temp"
        user_dir = work_dir / "user"
        for directory in (input_dir, output_dir, temp_dir, user_dir):
            directory.mkdir(parents=True, exist_ok=True)
        model_paths_path = work_dir / "extra_model_paths.yaml"
        model_paths = {
            "minimax_h3_hifi": {
                "base_path": str(model_dir),
                "diffusion_models": "diffusion_models",
                "text_encoders": "text_encoders",
                "vae": "vae",
            }
        }
        model_paths_path.write_text(json.dumps(model_paths), encoding="utf-8")
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = work_dir / "comfyui.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_visible_devices"])
        if payload.get("local_files_only", True):
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            str(payload["comfyui_python"]),
            str(comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-auto-launch",
            "--input-directory",
            str(input_dir),
            "--output-directory",
            str(output_dir),
            "--temp-directory",
            str(temp_dir),
            "--user-directory",
            str(user_dir),
            "--extra-model-paths-config",
            str(model_paths_path),
            "--reserve-vram",
            str(payload["reserve_vram_gib"]),
        ]
        process: subprocess.Popen | None = None
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=comfy_root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                _wait_for_comfyui(
                    process,
                    base_url,
                    float(payload["comfyui_startup_timeout"]),
                    log_path,
                )
                _require_nodes(
                    base_url,
                    (
                        "UNETLoader",
                        "CLIPLoader",
                        "VAELoader",
                        "MiniMaxH3ImageToVideo",
                        "MiniMaxH3ReferenceToVideo",
                        "MiniMaxH3MemoryEfficientSolAttentionPatch",
                        "MiniMaxH3FusedModulation",
                        "MiniMaxH3ChunkFeedForward",
                        "SaveVideo",
                    ),
                )
                image_names = _upload_images(base_url, payload)
                api_prompt = build_api_prompt(payload, image_names, f"minimax_h3/{request_id}")
                prompt_request = json.dumps(
                    {"prompt": api_prompt, "client_id": request_id}, ensure_ascii=False
                ).encode("utf-8")
                result = json.loads(
                    _api_request(
                        base_url,
                        "/prompt",
                        body=prompt_request,
                        content_type="application/json",
                        timeout=30,
                    )
                )
                if result.get("error") or not result.get("prompt_id"):
                    raise RuntimeError(
                        "ComfyUI 拒绝 MiniMax H3 工作流："
                        + json.dumps(result, ensure_ascii=False)[:4000]
                    )
                _wait_for_prompt(
                    base_url,
                    str(result["prompt_id"]),
                    float(payload["request_timeout"]),
                    process,
                    log_path,
                )
                videos = sorted(output_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime)
                if not videos:
                    raise RuntimeError(f"ComfyUI 成功结束但没有找到 MP4：{_tail(log_path)}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(videos[-1], output_path)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def main() -> None:
    """在一次性 worker 内启动 ComfyUI 并生成本地音视频。"""
    arguments = parse_args()
    payload = load_request(arguments.input_json)
    if payload.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run_comfyui(payload, Path(arguments.output_video))


if __name__ == "__main__":
    main()
