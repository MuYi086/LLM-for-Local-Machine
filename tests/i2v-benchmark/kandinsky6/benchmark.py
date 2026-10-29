"""Kandinsky 6 的隔离式 I2V 基准；真实推理需从其独立 uv 环境启动。"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

TEST_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TEST_DIR.parent
REPOSITORY_DIR = BENCHMARK_DIR.parents[1]
CONFIG_PATH = TEST_DIR / "config.json"
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
SCORE_FIELDS = ["blind_id", "visual_quality", "motion_stability", "prompt_adherence", "notes"]


def load_config() -> dict[str, Any]:
    """读取 Kandinsky 基准的固定条件配置。"""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


def repository_path(relative: str) -> Path:
    """解析仓库相对路径。"""
    return REPOSITORY_DIR / relative


def benchmark_path(config: dict[str, Any], relative: str) -> Path:
    """解析 benchmark 目录内的素材和结果路径。"""
    return BENCHMARK_DIR / relative


def output_root(config: dict[str, Any]) -> Path:
    """返回被 tests/i2v-benchmark/.gitignore 忽略的 Kandinsky 结果目录。"""
    return benchmark_path(config, config["output_dir"])


def model_dir(config: dict[str, Any]) -> Path:
    """读取环境变量覆盖项或本地默认模型目录。"""
    configured = os.environ.get("KANDINSKY6_MODEL_DIR", config["model_dir"])
    return Path(os.path.expandvars(os.path.expanduser(configured))).resolve()


def trials(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    """按场景名和 seed 稳定生成 9 条测试计划。"""
    return [
        (config["model"], scene, seed) for scene in config["scenes"] for seed in config["seeds"]
    ]


def build_payload(
    config: dict[str, Any], scene: str, seed: int, image_file: Path
) -> dict[str, Any]:
    """生成固定画幅、帧率、帧数和输入条件的 worker 请求。"""
    scenario = config["scenes"][scene]
    return {
        "model_dir": str(model_dir(config)),
        "reference_image_path": str(image_file.resolve()),
        "prompt": scenario["prompt"],
        "seed": seed,
        "width": config["width"],
        "height": config["height"],
        "num_frames": config["frames"],
        "frame_rate": config["fps_target"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "local_files_only": True,
        "sample_audio": config["sample_audio"],
        "expand_prompts": config["expand_prompts"],
    }


def _configure_service_environment(config: dict[str, Any]) -> None:
    """把服务导入时的缓存和输出重定向到测试结果目录。"""
    root = output_root(config).resolve()
    cache = root / "runtime"
    os.environ["KANDINSKY6_MODEL_DIR"] = str(model_dir(config))
    os.environ["STORAGE_DIR"] = str(cache / "storage")
    os.environ["RUNTIME_CACHE_DIR"] = str(cache / "runtime")
    os.environ["KANDINSKY6_IMAGE_DIR"] = str(cache / "images")
    os.environ["KANDINSKY6_OUTPUT_DIR"] = str(root / "service-output")
    os.environ["KANDINSKY6_WORKER_TMP_DIR"] = str(cache / "worker")
    os.environ["LOCAL_FILES_ONLY"] = "1"


def _load_service_main(config: dict[str, Any]):
    """加载轻量服务模块，以复用它的权重清单，不加载模型。"""
    _configure_service_environment(config)
    project_dir = repository_path(config["model_project"])
    sys.path.insert(0, str(project_dir))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "kandinsky6_benchmark_service_main", project_dir / "main.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 Kandinsky 服务模块: {project_dir / 'main.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(project_dir))


def _gpu_snapshot() -> tuple[int | None, float | None]:
    """读取全卡显存和利用率采样。"""
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
    """按 Linux MemAvailable 计算 WSL 已用主机内存，与 psutil 口径一致。"""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw, *_ = line.replace(":", "").split()
            values[key] = int(raw)
        return round((values["MemTotal"] - values["MemAvailable"]) / 1024, 1)
    except (OSError, KeyError, ValueError):
        return None


class Telemetry:
    """采样 GPU 全卡显存、利用率和 WSL 已用内存。"""

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
        write_csv(
            self.outfile,
            [
                "elapsed_seconds",
                "gpu_memory_used_mib",
                "gpu_utilization_pct",
                "wsl_mem_used_mib",
            ],
            self.samples,
        )

    def metric(self, field: str) -> int | float | None:
        values = [sample[field] for sample in self.samples if sample[field] is not None]
        return max(values) if values else None


def write_csv(file: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """原子写入 CSV，保留已完成或失败的试验记录。"""
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(file)


def read_csv(file: Path) -> list[dict[str, str]]:
    """读取已有 CSV；不存在时返回空列表。"""
    if not file.is_file():
        return []
    with file.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def preflight(_: argparse.Namespace | None = None) -> None:
    """检查素材、模型清单、推理依赖、CUDA 和视频探测工具。"""
    config = load_config()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config["cuda_visible_devices"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from PIL import Image

    if not shutil.which("nvidia-smi"):
        raise RuntimeError("缺少 nvidia-smi；请在 WSL 中检查 NVIDIA 驱动和 CUDA。")
    if not shutil.which("ffprobe"):
        raise RuntimeError("缺少 ffprobe；请先安装 FFmpeg。")
    service = _load_service_main(config)
    missing_files = [str(item) for item in service.REQUIRED_MODEL_FILES if not item.is_file()]
    if missing_files:
        raise RuntimeError("Kandinsky 模型文件缺失:\n" + "\n".join(missing_files))

    import torch
    from diffusers import Kandinsky6TI2VAPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("当前 Kandinsky uv 环境无法使用 CUDA。")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16。")
    if not callable(Kandinsky6TI2VAPipeline.from_pretrained):
        raise RuntimeError("当前 Diffusers 环境缺少 Kandinsky6TI2VAPipeline。")

    print(
        "GPU:",
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader", "--id=0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout.strip(),
    )
    print("CUDA:", torch.version.cuda, "BF16:", torch.cuda.is_bf16_supported())
    print("PyTorch:", importlib.metadata.version("torch"))
    print("Diffusers:", importlib.metadata.version("diffusers"))
    print("模型目录:", model_dir(config))
    print(f"模型文件: {len(service.REQUIRED_MODEL_FILES)} 项齐全")

    manifest = [
        {
            "relative_path": str(item.relative_to(model_dir(config))),
            "size_bytes": item.stat().st_size,
        }
        for item in service.REQUIRED_MODEL_FILES
    ]
    results = output_root(config)
    results.mkdir(parents=True, exist_ok=True)
    (results / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for scene, scenario in config["scenes"].items():
        image_path = benchmark_path(config, scenario["image"])
        if not image_path.is_file():
            raise RuntimeError(f"缺少场景图片 {scene}: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            print(f"场景 {scene}: {image.width}x{image.height} {image.mode}")
    print(
        "条件: 3 场景 × 3 seed = 9 次，"
        f"{config['width']}x{config['height']}，{config['fps_target']} FPS，"
        f"{config['frames']} 帧，{config['num_inference_steps']} 步，"
        f"guidance={config['guidance_scale']}。"
    )
    print("Preflight passed; 尚未加载完整 pipeline 或开始推理。")


def probe_video(file: Path) -> dict[str, Any]:
    """检查 MP4 视频流的分辨率、帧数、帧率、时长和音轨。"""
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
            str(file),
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
        raise RuntimeError("输出文件不包含视频流。")
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


def validate_video(file: Path, config: dict[str, Any]) -> dict[str, Any]:
    """校验生成视频的分辨率、帧数、帧率和音轨。"""
    if not file.is_file() or file.stat().st_size < 4096:
        raise RuntimeError("输出 MP4 缺失或小于 4 KiB。")
    metadata = probe_video(file)
    if (metadata["width"], metadata["height"]) != (config["width"], config["height"]):
        raise RuntimeError(
            f"输出分辨率 {metadata['width']}x{metadata['height']}，"
            f"目标为 {config['width']}x{config['height']}。"
        )
    if metadata["frames"] != config["frames"]:
        raise RuntimeError(f"输出帧数 {metadata['frames']}，目标为 {config['frames']}。")
    if abs(metadata["fps"] - config["fps_target"]) > 0.5:
        raise RuntimeError(f"输出帧率 {metadata['fps']}，目标为 {config['fps_target']}。")
    if not 2 <= metadata["video_seconds"] <= 9:
        raise RuntimeError(f"输出时长 {metadata['video_seconds']} 秒，不在 2–9 秒范围内。")
    if metadata["audio_streams"]:
        raise RuntimeError(f"视频模式不应带音轨，实际音轨数为 {metadata['audio_streams']}。")
    return metadata


def _load_pipeline(config: dict[str, Any]):
    """加载一次 BF16 pipeline，记录同一进程中的热运行耗时。"""
    import torch
    from diffusers import Kandinsky6TI2VAPipeline

    pipeline = Kandinsky6TI2VAPipeline.from_pretrained(
        str(model_dir(config)), torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipeline.enable_sequential_cpu_offload()
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    return pipeline


def _generate(pipeline: Any, config: dict[str, Any], scene: str, seed: int, output: Path) -> None:
    """按 Kandinsky Lite 蒸馏版官方配置生成无音轨视频。"""
    import torch
    from diffusers.utils import encode_video, load_image

    image_file = benchmark_path(config, config["scenes"][scene]["image"])
    image = load_image(str(image_file))
    result = pipeline(
        prompt=config["scenes"][scene]["prompt"],
        image=image,
        height=config["height"],
        width=config["width"],
        num_frames=config["frames"],
        frame_rate=config["fps_target"],
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        sample_audio=False,
        expand_prompts=False,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(result.frames[0], fps=config["fps_target"], output_path=str(output))
    del result


def run(args: argparse.Namespace) -> None:
    """运行 3 场景 × 3 seed 矩阵并保存指标、视频与逐次显存采样。"""
    preflight(args)
    config = load_config()
    results = output_root(config)
    videos_dir = results / "videos"
    rows_path = results / "runs.csv"
    existing_rows = read_csv(rows_path)
    existing = {row["run_id"]: row for row in existing_rows}
    plan = trials(config)
    submitted = 0
    pipeline = None
    try:
        for _, scene, seed in plan:
            run_id = f"{config['model']}__{scene}__s{seed}"
            if run_id in existing:
                continue
            if args.limit is not None and submitted >= args.limit:
                break
            output = videos_dir / f"{run_id}.mp4"
            record: dict[str, Any] = {field: "" for field in RUN_FIELDS}
            record.update(
                run_id=run_id,
                model=config["model"],
                scene=scene,
                seed=seed,
                status="failed",
                timestamp=now(),
            )
            baseline, _ = _gpu_snapshot()
            record["gpu_baseline_mib"] = baseline if baseline is not None else ""
            started = time.monotonic()
            monitor = None
            stop_batch = False
            try:
                monitor_path = results / "gpu" / f"{run_id}.csv"
                with Telemetry(config["sample_interval_s"], monitor_path) as monitor:
                    if pipeline is None:
                        load_started = time.monotonic()
                        pipeline = _load_pipeline(config)
                        record["model_load_seconds"] = round(time.monotonic() - load_started, 2)
                    _generate(pipeline, config, scene, seed, output)
                    metadata = validate_video(output, config)
                wall = time.monotonic() - started
                record.update(
                    status="success",
                    wall_seconds=round(wall, 2),
                    width=metadata["width"],
                    height=metadata["height"],
                    fps=metadata["fps"],
                    frames=metadata["frames"],
                    video_seconds=metadata["video_seconds"],
                    audio_streams=metadata["audio_streams"],
                    render_seconds_per_video_second=round(wall / metadata["video_seconds"], 2),
                    output=str(output.relative_to(BENCHMARK_DIR)),
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                print("FAILED", run_id, record["error"], file=sys.stderr)
                if pipeline is None:
                    print("模型未能加载；停止剩余计划。", file=sys.stderr)
                    stop_batch = True
            finally:
                record["wall_seconds"] = record["wall_seconds"] or round(
                    time.monotonic() - started, 2
                )
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
                existing[run_id] = record
                write_csv(rows_path, RUN_FIELDS, [existing[key] for key in sorted(existing)])
                print(
                    f"{run_id} => {record['status']} time={record['wall_seconds']}s "
                    f"vram={record['gpu_peak_mib']} MiB"
                )
                submitted += 1
            if stop_batch:
                break
    finally:
        if pipeline is not None:
            del pipeline
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    print("Results:", rows_path)


def review_sheet(_: argparse.Namespace | None = None) -> None:
    """生成隐藏模型名的盲看片单、视频别名和空白评分表。"""
    config = load_config()
    results = output_root(config)
    rows = [row for row in read_csv(results / "runs.csv") if row.get("status") == "success"]
    if not rows:
        raise RuntimeError("还没有成功生成的视频，无法创建盲评表。")
    blind_dir = results / "blind_videos"
    blind_dir.mkdir(parents=True, exist_ok=True)
    random.SystemRandom().shuffle(rows)
    key_rows = []
    score_rows = []
    cards = []
    for row in rows:
        blind_id = secrets.token_hex(3).upper()
        source = BENCHMARK_DIR / row["output"]
        alias = blind_dir / f"{blind_id}.mp4"
        shutil.copyfile(source, alias)
        key_rows.append(
            {
                "blind_id": blind_id,
                "run_id": row["run_id"],
                "model": row["model"],
                "scene": row["scene"],
                "seed": row["seed"],
            }
        )
        score_rows.append(
            {
                "blind_id": blind_id,
                "visual_quality": "",
                "motion_stability": "",
                "prompt_adherence": "",
                "notes": "",
            }
        )
        cards.append(
            f'<section><h2>{blind_id}</h2><video controls preload="metadata" '
            f'src="blind_videos/{blind_id}.mp4"></video></section>'
        )
    write_csv(results / "blind_key.csv", list(key_rows[0]), key_rows)
    write_csv(results / "scores.csv", SCORE_FIELDS, score_rows)
    page = "<!doctype html><meta charset='utf-8'><title>Kandinsky I2V blind review</title>"
    page += "<style>body{font:16px sans-serif;background:#181818;color:#eee}section{margin:2rem}video{width:min(480px,90vw)}</style>"
    page += "<h1>Kandinsky I2V 盲评</h1>" + "".join(cards)
    (results / "review.html").write_text(page, encoding="utf-8")
    print("Saved blind review files under", results)


def report(_: argparse.Namespace | None = None) -> None:
    """从真实 CSV 运行记录生成时长、显存和盲评分摘要。"""
    config = load_config()
    results = output_root(config)
    rows = read_csv(results / "runs.csv")
    if not rows:
        raise RuntimeError("没有运行记录；先执行 run。")
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") == "failed"]
    warm = [float(row["wall_seconds"]) for row in successes if not row.get("model_load_seconds")]
    cold = [float(row["wall_seconds"]) for row in successes if row.get("model_load_seconds")]
    scores = read_csv(results / "scores.csv")
    scored = [row for row in scores if row.get("visual_quality")]

    def med(values: list[float]) -> str:
        return f"{median(values):.2f}" if values else "待测"

    lines = [
        "# Kandinsky 6 Lite I2V 基准报告",
        "",
        f"生成时间：{now()}",
        "",
        "口径：3 场景 × 3 seed = 9 条计划；输出为 480×832、24 FPS、121 帧。",
        "Kandinsky 使用蒸馏 checkpoint 原生的 10 步、guidance=1.0，音频采样关闭。",
        "此脚本在一个 benchmark 进程中驻留 pipeline，以便分别记录冷启动和同进程热运行。",
        "正式 HTTP 服务为一次性 worker；两者的生命周期不同，壁钟时间不代表服务接口耗时。",
        "",
        "| 成功/计划 | 失败 | 冷启动秒 | 热运行中位秒 | GPU 峰值 MiB | WSL RAM 峰值 MiB | 已评分/成功 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(successes)}/9 | {len(failures)} | {med(cold)} | {med(warm)} | "
        f"{max((int(row['gpu_peak_mib']) for row in rows if row.get('gpu_peak_mib')), default=0)} | "
        f"{max((float(row['wsl_mem_peak_mib']) for row in rows if row.get('wsl_mem_peak_mib')), default=0):.1f} | "
        f"{len(scored)}/{len(successes)} |",
        "",
        "| Run | 状态 | Seed | 秒 | 分辨率 | FPS | 帧 | GPU 峰值 MiB | 错误 |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['seed']} | {row['wall_seconds']} | "
            f"{row['width']}×{row['height']} | {row['fps']} | {row['frames']} | "
            f"{row['gpu_peak_mib']} | {html.escape(row['error'])} |"
        )
    lines.extend(
        [
            "",
            "性能读数使用 nvidia-smi 全卡显存，不能精确代表模型独占分配；WSL 内存按 MemTotal−MemAvailable。",
            "画质和运动稳定性只汇总人工评分；没有填写的分数保持待测。",
        ]
    )
    report_path = results / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved", report_path)


def main() -> None:
    """解析基准命令。"""
    parser = argparse.ArgumentParser(description="Kandinsky 6 Lite I2V benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="检查模型快照、图片、CUDA 和 FFprobe")
    run_parser = commands.add_parser("run", help="运行相同场景/seed 矩阵")
    run_parser.add_argument("--limit", type=int, default=None)
    commands.add_parser("review-sheet", help="生成盲评页面和空白评分表")
    commands.add_parser("report", help="从真实运行记录生成报告")
    args = parser.parse_args()
    actions = {
        "preflight": preflight,
        "run": run,
        "review-sheet": review_sheet,
        "report": report,
    }
    try:
        actions[args.command](args)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"ERROR: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
