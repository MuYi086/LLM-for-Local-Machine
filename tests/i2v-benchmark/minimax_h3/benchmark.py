"""通过本地 MiniMax H3 API 执行与 Kandinsky 相同输入的 I2V 基准。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TEST_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TEST_DIR.parent
CONFIG_PATH = Path(
    os.environ.get("MINIMAX_H3_BENCHMARK_CONFIG", str(TEST_DIR / "config.json"))
).expanduser().resolve()
RUN_FIELDS = [
    "run_id",
    "model",
    "scene",
    "seed",
    "status",
    "wall_seconds",
    "model_load_seconds",
    "gpu_peak_mib",
    "gpu_baseline_mib",
    "gpu_delta_peak_mib",
    "gpu_util_peak_pct",
    "wsl_mem_peak_mib",
    "width",
    "height",
    "fps",
    "frames",
    "video_seconds",
    "audio_streams",
    "render_seconds_per_video_second",
    "output",
    "timestamp",
    "error",
]
TELEMETRY_FIELDS = [
    "elapsed_seconds",
    "gpu_memory_used_mib",
    "gpu_utilization_pct",
    "wsl_mem_used_mib",
]
MODEL_FILES = [
    "diffusion_models/minimax_h3_fl2va_dit_16g.safetensors",
    "diffusion_models/minimax_h3_ref2va_dit_16g.safetensors",
    "text_encoders/qwen3vl_text_encoder.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]


class BenchmarkBlocked(RuntimeError):
    """前置条件不满足，基准应在启动推理前停止。"""


def load_config() -> dict[str, Any]:
    """读取 H3 设置并从 Kandinsky 配置载入完全相同的场景素材。"""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    shared_path = (TEST_DIR / config["shared_conditions"]).resolve()
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    config["shared"] = shared
    for field in ("width", "height", "fps_target", "seeds", "scenes"):
        config[field] = shared[field]
    config["service_url"] = os.getenv(
        "MINIMAX_H3_BENCHMARK_URL", config["service_url"]
    ).rstrip("/")
    return config


def now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


def benchmark_path(relative: str) -> Path:
    """解析 tests/i2v-benchmark 下的相对路径。"""
    return BENCHMARK_DIR / relative


def output_root(config: dict[str, Any]) -> Path:
    """返回被 tests/i2v-benchmark/.gitignore 忽略的 H3 结果目录。"""
    return benchmark_path(config["output_dir"])


def trials(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    """按共享配置的场景顺序和 seed 稳定生成测试计划。"""
    return [
        (config["model"], scene, seed)
        for scene in config["scenes"]
        for seed in config["seeds"]
    ]


def expected_frame_count(config: dict[str, Any]) -> int:
    """按 H3 原生 `17k+5` 帧网格计算 5 秒请求的输出帧数。"""
    frame_count = max(5, round(config["duration_seconds"] * config["fps_target"]))
    return frame_count + (5 - frame_count % 17) % 17


def build_request(
    config: dict[str, Any], scene: str, seed: int, image_path: str
) -> dict[str, Any]:
    """构造使用共享提示词/seed/画幅的 H3 原生 FL2VA 请求。"""
    return {
        "mode": "fl2va",
        "prompt": config["scenes"][scene]["prompt"],
        "first_frame_path": image_path,
        "seed": seed,
        "width": config["width"],
        "height": config["height"],
        "duration_seconds": config["duration_seconds"],
        "include_audio": config["include_audio"],
    }


def _api_url(config: dict[str, Any], path: str) -> str:
    return f"{config['service_url']}{path}"


def _http_json(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    request = Request(url, data=data, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"H3 API HTTP {exc.code}: {detail[:2000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 H3 API {url}: {exc.reason}") from exc


def service_health(config: dict[str, Any]) -> dict[str, Any]:
    """读取不加载模型的 H3 健康状态。"""
    return _http_json(_api_url(config, "/v1/health"))


def _require_ready(config: dict[str, Any], health: dict[str, Any]) -> None:
    available = health.get("available", {})
    if not available.get("selected_provider_ready"):
        raise BenchmarkBlocked("H3 服务报告 selected_provider_ready=false。")
    if config["include_audio"] and not available.get("native_audio_ready"):
        raise BenchmarkBlocked("H3 基准启用了原生音频，但服务报告 native_audio_ready=false。")
    if health.get("runtime", {}).get("steps") != config["native_steps"]:
        raise BenchmarkBlocked(
            "H3 服务步数与 benchmark 声明不一致："
            f"{health.get('runtime', {}).get('steps')} != {config['native_steps']}。"
        )
    if health.get("runtime", {}).get("frame_rate") != config["fps_target"]:
        raise BenchmarkBlocked("H3 服务输出帧率与 Kandinsky 对照配置不一致。")
    free_mib = int(health.get("gpu_memory", {}).get("free_mib", 0))
    if free_mib < config["min_free_vram_mib"]:
        raise BenchmarkBlocked(
            f"当前空闲显存 {free_mib} MiB，低于安全门槛 "
            f"{config['min_free_vram_mib']} MiB；未启动推理。"
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_manifest(config: dict[str, Any], health: dict[str, Any]) -> None:
    model_dir = Path(health["paths"]["model_dir"])
    manifest = [
        {"relative_path": relative, "size_bytes": (model_dir / relative).stat().st_size}
        for relative in MODEL_FILES
        if (model_dir / relative).is_file()
    ]
    _write_json(output_root(config) / "model_manifest.json", manifest)


def preflight(_: argparse.Namespace | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """检查本地素材、Kandinsky 基线、H3 健康状态和显存，不生成视频。"""
    config = load_config()
    if not shutil.which("nvidia-smi"):
        raise BenchmarkBlocked("缺少 nvidia-smi。")
    if not shutil.which("ffprobe"):
        raise BenchmarkBlocked("缺少 ffprobe。")
    if config["width"] % 32 or config["height"] % 32:
        raise BenchmarkBlocked("共享画幅不是 H3 要求的 32 像素倍数。")
    if config["width"] * config["height"] > 960 * 544:
        raise BenchmarkBlocked("共享画幅超过 H3 服务安全像素预算。")

    for scene, scenario in config["scenes"].items():
        image = benchmark_path(scenario["image"])
        if not image.is_file():
            raise BenchmarkBlocked(f"缺少场景图片 {scene}: {image}")

    kandinsky_results = benchmark_path(config["shared"]["output_dir"])
    kandinsky_rows = read_csv(kandinsky_results / "runs.csv")
    successful = {row["run_id"] for row in kandinsky_rows if row.get("status") == "success"}
    for scene in config["scenes"]:
        for seed in config["seeds"]:
            run_id = f"{config['shared']['model']}__{scene}__s{seed}"
            video = kandinsky_results / "videos" / f"{run_id}.mp4"
            if run_id not in successful or not video.is_file():
                raise BenchmarkBlocked(f"Kandinsky 对照样本不完整: {run_id}")

    health = service_health(config)
    _require_ready(config, health)
    output_root(config).mkdir(parents=True, exist_ok=True)
    _write_manifest(config, health)
    print("H3 API:", config["service_url"])
    print("GPU:", health["cuda"].get("selected_device", "unknown"))
    print(
        f"共享条件: {len(config['scenes'])} 场景 × {len(config['seeds'])} seed，"
        f"{config['width']}×{config['height']}，{config['fps_target']} FPS；"
        f"H3 原生 {config['native_steps']} 步、{expected_frame_count(config)} 帧、带音轨。"
    )
    print(
        f"空闲显存: {health['gpu_memory']['free_mib']} MiB；门槛: "
        f"{config['min_free_vram_mib']} MiB。Preflight passed; 尚未推理。"
    )
    return config, health


def _gpu_snapshot() -> tuple[int | None, float | None]:
    """读取 GPU 全卡显存和利用率。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        memory, utilization = (part.strip() for part in result.stdout.splitlines()[0].split(","))
        return int(memory), float(utilization)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None, None


def _wsl_memory_used_mib() -> float | None:
    """按 Linux MemAvailable 计算 WSL 已用主机内存。"""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw, *_ = line.replace(":", "").split()
            values[key] = int(raw)
        return round((values["MemTotal"] - values["MemAvailable"]) / 1024, 1)
    except (OSError, KeyError, ValueError):
        return None


class Telemetry:
    """定期采样全卡 GPU 显存、利用率和 WSL 主机内存。"""

    def __init__(self, interval: float, outfile: Path) -> None:
        self.interval = interval
        self.outfile = outfile
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            memory, utilization = _gpu_snapshot()
            self.samples.append(
                {
                    "elapsed_seconds": round(time.monotonic() - self.started, 2),
                    "gpu_memory_used_mib": memory,
                    "gpu_utilization_pct": utilization,
                    "wsl_mem_used_mib": _wsl_memory_used_mib(),
                }
            )
            self.stop_event.wait(self.interval)

    def __enter__(self) -> Telemetry:
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        write_csv(self.outfile, TELEMETRY_FIELDS, self.samples)

    def metric(self, field: str) -> int | float | None:
        values = [sample[field] for sample in self.samples if sample[field] is not None]
        return max(values) if values else None


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """原子写入 CSV，保留已完成和失败的样本记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV；目标不存在时返回空列表。"""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _multipart_image(image_path: Path, logical_path: str) -> tuple[bytes, str]:
    """编码 H3 图片上传接口所需的 multipart/form-data 请求体。"""
    boundary = "----h3-benchmark-" + secrets.token_hex(16)
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="full_path"\r\n\r\n',
            logical_path.encode("utf-8"),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def upload_image(
    config: dict[str, Any], image_path: Path, logical_path: str
) -> None:
    """上传一张只读基准输入图，供 H3 API 使用逻辑路径引用。"""
    body, content_type = _multipart_image(image_path, logical_path)
    result = _http_json(
        _api_url(config, "/v1/upload_image"),
        data=body,
        headers={"Content-Type": content_type},
        timeout=120,
    )
    if result.get("code") != 200 or result.get("filename") != logical_path:
        raise RuntimeError(f"H3 图片上传失败: {json.dumps(result, ensure_ascii=False)}")


def _uploaded_file_path(image_dir: str, logical_path: str) -> Path:
    suffix = Path(logical_path).suffix.lower()
    digest = hashlib.sha256(logical_path.encode("utf-8")).hexdigest()
    return Path(image_dir) / f"{digest}{suffix}"


def _remove_uploaded_images(paths: list[Path]) -> None:
    """删除本次上传的临时图片和原子提交留下的锁文件。"""
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_name(f".{path.name}.lock").unlink(missing_ok=True)


def _response_filename(headers: Any) -> str:
    disposition = headers.get("Content-Disposition", "")
    message = Message()
    message["content-disposition"] = disposition
    filename = message.get_filename()
    if not filename or Path(filename).name != filename:
        raise RuntimeError(f"H3 API 未返回安全的视频文件名: {disposition!r}")
    return filename


def request_video(
    config: dict[str, Any], payload: dict[str, Any], output: Path
) -> tuple[float, str]:
    """请求 H3 生成并流式保存 MP4，返回接口完整耗时和服务端文件名。"""
    request = Request(
        _api_url(config, "/v1/minimax/H3-16G-HiFi"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    temporary = output.with_name(output.name + ".part")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=config["request_timeout_s"]) as response:
            filename = _response_filename(response.headers)
            output.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(output)
        return time.monotonic() - started, filename
    except HTTPError as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"H3 生成 API HTTP {exc.code}: {detail[:4000]}") from exc
    except (OSError, URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"H3 生成请求失败: {exc}") from exc


def _remove_service_video(service_output_dir: str, filename: str) -> None:
    """删除服务为本次 API 响应留下的副本，测试 MP4 保留在 results 目录。"""
    if Path(filename).name != filename:
        return
    path = Path(service_output_dir) / filename
    path.unlink(missing_ok=True)


def probe_video(path: Path) -> dict[str, Any]:
    """使用 ffprobe 读取视频分辨率、帧率、帧数、时长和音轨数。"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    metadata = json.loads(result.stdout)
    video = next(
        (stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise RuntimeError("H3 输出不包含视频流。")
    numerator, _, denominator = (video.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(numerator) / float(denominator or 1)
    duration = float(video.get("duration") or metadata.get("format", {}).get("duration") or 0)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": round(fps, 2),
        "frames": int(video.get("nb_read_frames") or video.get("nb_frames") or 0),
        "video_seconds": round(duration, 3),
        "audio_streams": sum(
            stream.get("codec_type") == "audio" for stream in metadata.get("streams", [])
        ),
    }


def validate_video(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """确认 H3 MP4 满足共享画幅与 H3 原生帧网格。"""
    if not path.is_file() or path.stat().st_size < 4096:
        raise RuntimeError("H3 输出 MP4 缺失或小于 4 KiB。")
    metadata = probe_video(path)
    actual_size = (metadata["width"], metadata["height"])
    target_size = (config["width"], config["height"])
    if actual_size != target_size:
        raise RuntimeError(f"H3 输出分辨率 {actual_size}，目标为 {target_size}。")
    if metadata["frames"] != expected_frame_count(config):
        raise RuntimeError(
            f"H3 输出 {metadata['frames']} 帧，预期原生帧网格为 "
            f"{expected_frame_count(config)} 帧。"
        )
    if abs(metadata["fps"] - config["fps_target"]) > 0.5:
        raise RuntimeError(f"H3 输出 {metadata['fps']} FPS，与目标不符。")
    if metadata["audio_streams"] != config["expected_audio_streams"]:
        raise RuntimeError(
            f"H3 音轨数为 {metadata['audio_streams']}，"
            f"预期 {config['expected_audio_streams']}。"
        )
    return metadata


def _read_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["run_id"]: row for row in read_csv(path) if row.get("run_id")}


def run(args: argparse.Namespace) -> None:
    """逐条执行 H3 API 生成，采样显存并按样本保存视频与 CSV。"""
    config, health = preflight(args)
    results = output_root(config)
    videos_dir = results / "videos"
    rows_path = results / "runs.csv"
    existing = _read_existing_rows(rows_path)
    plan = trials(config)
    pending = []
    for model, scene, seed in plan:
        run_id = f"{model}__{scene}__s{seed}"
        row = existing.get(run_id, {})
        if row.get("status") == "success" and (BENCHMARK_DIR / row.get("output", "")).is_file():
            continue
        pending.append((model, scene, seed))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit 必须是正整数。")
        pending = pending[: args.limit]
    if not pending:
        print("所有 H3 样本都已成功完成；没有新推理。")
        return

    batch_id = secrets.token_hex(6)
    image_dir = health["paths"]["image_dir"]
    service_output_dir = health["paths"]["output_dir"]
    uploaded_paths: list[Path] = []
    uploaded_logical_paths: dict[str, str] = {}
    scenes_needed = dict.fromkeys(scene for _, scene, _ in pending)
    try:
        for scene in scenes_needed:
            scenario = config["scenes"][scene]
            source_image = benchmark_path(scenario["image"])
            logical_path = f"i2v-benchmark/minimax-h3-{batch_id}/{scene}.png"
            uploaded_paths.append(_uploaded_file_path(image_dir, logical_path))
            upload_image(config, source_image, logical_path)
            uploaded_logical_paths[scene] = logical_path

        submitted = 0
        for model, scene, seed in pending:
            run_id = f"{model}__{scene}__s{seed}"
            output = videos_dir / f"{run_id}.mp4"
            record: dict[str, Any] = {field: "" for field in RUN_FIELDS}
            record.update(
                run_id=run_id,
                model=model,
                scene=scene,
                seed=seed,
                status="failed",
                timestamp=now(),
                width=config["width"],
                height=config["height"],
                fps=config["fps_target"],
            )
            monitor: Telemetry | None = None
            response_filename: str | None = None
            baseline, _ = _gpu_snapshot()
            record["gpu_baseline_mib"] = baseline if baseline is not None else ""
            started = time.monotonic()
            stop_batch = False
            try:
                current_health = service_health(config)
                _require_ready(config, current_health)
                payload = build_request(config, scene, seed, uploaded_logical_paths[scene])
                monitor_path = results / "gpu" / f"{run_id}.csv"
                with Telemetry(config["sample_interval_s"], monitor_path) as monitor:
                    wall_seconds, response_filename = request_video(config, payload, output)
                    metadata = validate_video(output, config)
                record.update(
                    status="success",
                    wall_seconds=round(wall_seconds, 2),
                    width=metadata["width"],
                    height=metadata["height"],
                    fps=metadata["fps"],
                    frames=metadata["frames"],
                    video_seconds=metadata["video_seconds"],
                    audio_streams=metadata["audio_streams"],
                    render_seconds_per_video_second=round(
                        wall_seconds / metadata["video_seconds"], 2
                    ),
                    output=str(output.relative_to(BENCHMARK_DIR)),
                )
            except BenchmarkBlocked as exc:
                record["status"] = "blocked"
                record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                stop_batch = True
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:4000]
                stop_batch = True
                print("FAILED", run_id, record["error"], file=sys.stderr)
            finally:
                if not record["wall_seconds"]:
                    record["wall_seconds"] = round(time.monotonic() - started, 2)
                if monitor is not None:
                    gpu_peak = monitor.metric("gpu_memory_used_mib")
                    record["gpu_peak_mib"] = gpu_peak if gpu_peak is not None else ""
                    record["gpu_delta_peak_mib"] = (
                        max(0, gpu_peak - baseline)
                        if gpu_peak is not None and baseline is not None
                        else ""
                    )
                    record["gpu_util_peak_pct"] = monitor.metric("gpu_utilization_pct") or ""
                    record["wsl_mem_peak_mib"] = monitor.metric("wsl_mem_used_mib") or ""
                if response_filename:
                    _remove_service_video(service_output_dir, response_filename)
                existing[run_id] = record
                write_csv(rows_path, RUN_FIELDS, [existing[key] for key in sorted(existing)])
                print(
                    f"{run_id} => {record['status']} time={record['wall_seconds']}s "
                    f"vram={record['gpu_peak_mib']} MiB"
                )
                submitted += 1
            if stop_batch:
                break
            if args.limit is not None and submitted >= args.limit:
                break
    finally:
        _remove_uploaded_images(uploaded_paths)

    print("Results:", rows_path)


def _successful(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["run_id"]: row for row in rows if row.get("status") == "success"}


def _video_element(source: str) -> str:
    if not source:
        return "<p>此样本尚未生成。</p>"
    safe_source = html.escape(source, quote=True)
    return f'<video controls preload="metadata" src="{safe_source}"></video>'


def _median(values: list[float]) -> str:
    return f"{median(values):.2f}" if values else "待测"


def comparison_page(config: dict[str, Any]) -> Path:
    """生成按场景和 seed 对齐的 Kandinsky/H3 视频对照页。"""
    results = output_root(config)
    kandinsky = benchmark_path(config["shared"]["output_dir"])
    h3_rows = _successful(read_csv(results / "runs.csv"))
    kandinsky_rows = _successful(read_csv(kandinsky / "runs.csv"))
    cards = []
    for scene in config["scenes"]:
        for seed in config["seeds"]:
            kandinsky_id = f"{config['shared']['model']}__{scene}__s{seed}"
            h3_id = f"{config['model']}__{scene}__s{seed}"
            kandinsky_row = kandinsky_rows.get(kandinsky_id)
            h3_row = h3_rows.get(h3_id)
            kandinsky_video = (
                os.path.relpath(benchmark_path(kandinsky_row["output"]), start=results)
                if kandinsky_row
                else ""
            )
            h3_video = (
                f"videos/{Path(h3_row['output']).name}" if h3_row else ""
            )
            prompt = html.escape(config["scenes"][scene]["prompt"])
            cards.append(
                "<section>"
                f"<h2>{html.escape(scene)} · seed {seed}</h2>"
                f"<p>{prompt}</p><div class='pair'>"
                f"<article><h3>Kandinsky 6</h3>{_video_element(kandinsky_video)}</article>"
                f"<article><h3>MiniMax H3</h3>{_video_element(h3_video)}</article>"
                "</div></section>"
            )
    page = """<!doctype html><meta charset="utf-8"><title>Kandinsky 6 vs MiniMax H3</title>
<style>body{font:16px system-ui,sans-serif;background:#17191d;color:#eee;margin:2rem}
section{margin:2rem 0;padding:1rem;border:1px solid #444;border-radius:8px}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
video{width:100%;max-width:480px;background:#000}p{max-width:1000px;line-height:1.5;color:#ccc}</style>
<h1>Kandinsky 6 Lite vs MiniMax H3 16G HiFi</h1>
<p>相同输入图片、提示词、seed 和 480×832 输出画幅。H3 使用原生 20 步，输出 124 帧并带音轨；
Kandinsky 使用 10 步，输出 121 帧且无音轨。</p>""" + "".join(cards)
    path = results / "comparison.html"
    path.write_text(page, encoding="utf-8")
    return path


def report(_: argparse.Namespace | None = None) -> None:
    """汇总 H3 速度/显存并与现有 Kandinsky CSV 对照。"""
    config = load_config()
    results = output_root(config)
    h3_rows_all = read_csv(results / "runs.csv")
    if not h3_rows_all:
        raise RuntimeError("没有 H3 运行记录；先执行 run。")
    kandinsky_root = benchmark_path(config["shared"]["output_dir"])
    kandinsky_rows_all = read_csv(kandinsky_root / "runs.csv")
    h3_successes = [row for row in h3_rows_all if row.get("status") == "success"]
    h3_failures = [row for row in h3_rows_all if row.get("status") != "success"]
    kandinsky_successes = [
        row for row in kandinsky_rows_all if row.get("status") == "success"
    ]
    h3_times = [float(row["wall_seconds"]) for row in h3_successes if row.get("wall_seconds")]
    h3_peaks = [int(row["gpu_peak_mib"]) for row in h3_rows_all if row.get("gpu_peak_mib")]
    h3_deltas = [
        int(row["gpu_delta_peak_mib"])
        for row in h3_rows_all
        if row.get("gpu_delta_peak_mib")
    ]
    kandinsky_cold = [
        float(row["wall_seconds"])
        for row in kandinsky_successes
        if row.get("model_load_seconds") and row.get("wall_seconds")
    ]
    kandinsky_warm = [
        float(row["wall_seconds"])
        for row in kandinsky_successes
        if not row.get("model_load_seconds") and row.get("wall_seconds")
    ]
    h3_success_map = _successful(h3_rows_all)
    kandinsky_success_map = _successful(kandinsky_rows_all)
    planned = len(trials(config))
    lines = [
        "# MiniMax H3 16G HiFi I2V 基准报告",
        "",
        f"生成时间：{now()}",
        "",
        "## 对照条件",
        "",
        f"复用 Kandinsky 的 {len(config['scenes'])} 张输入图、提示词和 "
        f"{len(config['seeds'])} 个 seed；共同输出画幅 {config['width']}×{config['height']}，"
        f"{config['fps_target']} FPS。H3 请求显式覆盖服务默认 960×544 画幅，使用原生 "
        f"{config['native_steps']} 步、"
        f"5 秒请求、{expected_frame_count(config)} 帧（约 5.17 秒）和原生音轨；"
        "Kandinsky 为 10 步、121 帧（约 5.04 秒）、无音轨。",
        "H3 的壁钟时间是完整 API 请求时间，包含每次 ComfyUI 子进程启动、模型加载、推理、视频返回和服务清理。",
        "Kandinsky 在单进程内驻留 pipeline，报告分别列冷启动与热运行；比较时应以 H3 请求时间对照 Kandinsky 热运行，"
        "同时参考 Kandinsky 冷启动。",
        "显存是 nvidia-smi 全卡用量，以 benchmark 采样到的峰值计；delta 为峰值减去请求前全卡基线，"
        "不是框架报告的模型专属 allocated memory。",
        "",
        "## 汇总",
        "",
        "| 模型 | 成功/计划 | 失败/阻塞 | 冷启动中位秒 | 后续/请求中位秒 | GPU 峰值 MiB | GPU 增量峰值中位 MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Kandinsky 6 | {len(kandinsky_successes)}/{planned} | "
        f"{len(kandinsky_rows_all) - len(kandinsky_successes)} | {_median(kandinsky_cold)} | "
        f"{_median(kandinsky_warm)} | "
        f"{max((int(row['gpu_peak_mib']) for row in kandinsky_successes if row.get('gpu_peak_mib')), default=0)} | "
        f"{_median([float(row['gpu_delta_peak_mib']) for row in kandinsky_successes if row.get('gpu_delta_peak_mib')])} |",
        f"| MiniMax H3 | {len(h3_successes)}/{planned} | {len(h3_failures)} | "
        f"每条请求均为一次性 worker | {_median(h3_times)} | "
        f"{max(h3_peaks, default=0)} | {_median([float(value) for value in h3_deltas])} |",
        "",
        "## 同图同 seed 结果",
        "",
        "| 场景 | Seed | Kandinsky 秒 | H3 API 秒 | Kandinsky GPU 峰值 MiB | H3 GPU 峰值 MiB | H3 视频帧/音轨 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for scene in config["scenes"]:
        for seed in config["seeds"]:
            kandinsky_id = f"{config['shared']['model']}__{scene}__s{seed}"
            h3_id = f"{config['model']}__{scene}__s{seed}"
            kandinsky_row = kandinsky_success_map.get(kandinsky_id, {})
            h3_row = h3_success_map.get(h3_id, {})
            h3_shape = (
                f"{h3_row.get('frames', '')} 帧 / {h3_row.get('audio_streams', '')} 音轨"
                if h3_row
                else h3_row.get("status", "待测")
            )
            lines.append(
                f"| {scene} | {seed} | {kandinsky_row.get('wall_seconds', '—')} | "
                f"{h3_row.get('wall_seconds', '—')} | {kandinsky_row.get('gpu_peak_mib', '—')} | "
                f"{h3_row.get('gpu_peak_mib', '—')} | {h3_shape} |"
            )
    lines.extend(
        [
            "",
            "## 人工画面比较",
            "",
            "打开 [`comparison.html`](comparison.html) 可按相同场景和 seed 并排播放两个模型的视频。"
            "本报告不自动判断画质；构图、主体稳定性、运动自然度和提示词符合度需要结合视频人工比较。",
            "",
            "H3 逐样本状态和采样明细见 `runs.csv` 与 `gpu/`；视频在 `videos/`。",
        ]
    )
    report_path = results / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison_path = comparison_page(config)
    print("Saved", report_path)
    print("Saved", comparison_path)


def main() -> None:
    """解析 H3 benchmark 命令。"""
    parser = argparse.ArgumentParser(description="MiniMax H3 I2V benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="检查 H3 API、共享素材、CUDA 和安全显存门槛")
    run_parser = commands.add_parser("run", help="运行相同图片/提示词/seed 的 H3 原生矩阵")
    run_parser.add_argument("--limit", type=int, default=None)
    commands.add_parser("report", help="生成与 Kandinsky 同图同 seed 的性能和画面比较")
    args = parser.parse_args()
    actions = {"preflight": preflight, "run": run, "report": report}
    try:
        actions[args.command](args)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
