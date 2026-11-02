#!/usr/bin/env python3
"""通过 WanGP 官方 Python API 执行一次 LongCat Avatar 推理。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from wangp_runtime import DEFAULT_MEMORY_PROFILE, validate_memory_profile

MODEL_TYPE = "longcat_avatar_v1_5"
WORKER_PROCESS_STARTED = time.monotonic()


class WorkerTrace:
    """可选地记录 WanGP worker 阶段与官方 API 进度事件。"""

    def __init__(self, directory: Path, profile: int | str) -> None:
        self.started = WORKER_PROCESS_STARTED
        self.lock = threading.Lock()
        self.write_error: str | None = None
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"worker-profile{profile}-{timestamp}-{os.getpid()}.jsonl"
        self.emit("worker_entry", profile=profile)

    @classmethod
    def from_environment(cls, profile: int | str) -> WorkerTrace | None:
        """仅在显式设置诊断目录时创建追踪器。"""
        directory = os.getenv("LONGCAT_AVATAR_TRACE_DIR", "").strip()
        if not directory:
            return None
        try:
            return cls(Path(directory).expanduser(), profile)
        except OSError as exc:
            print(f"LongCat diagnostics disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    def emit(self, event: str, **fields: Any) -> None:
        """将一个时间戳事件追加到逐行 JSON 文件。"""
        if self.write_error is not None:
            return
        record = {
            "event": event,
            "at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started, 6),
            "pid": os.getpid(),
            **fields,
        }
        try:
            with self.lock, self.path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                sink.flush()
        except OSError as exc:
            # 观测写入失败不应中断模型推理。
            self.write_error = f"{type(exc).__name__}: {exc}"

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """记录一个同步阶段的开始、结束及异常。"""
        started = time.monotonic()
        self.emit("stage_start", stage=name)
        try:
            yield
        except BaseException as exc:
            self.emit(
                "stage_error",
                stage=name,
                duration_seconds=round(time.monotonic() - started, 6),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        else:
            self.emit(
                "stage_end",
                stage=name,
                duration_seconds=round(time.monotonic() - started, 6),
            )

    def on_progress(self, update: Any) -> None:
        """记录 WanGP 的模型阶段、进度及滑窗信息。"""
        self.emit(
            "progress",
            **{
                key: getattr(update, key, None)
                for key in (
                    "phase",
                    "status",
                    "progress",
                    "current_step",
                    "total_steps",
                    "raw_phase",
                    "unit",
                    "window_index",
                    "window_count",
                )
            },
        )

    def on_stream(self, message: Any) -> None:
        """记录 WanGP 控制台状态行并保留 stdout/stderr 来源。"""
        text = str(getattr(message, "text", ""))
        self.emit(
            "stream",
            stream=getattr(message, "stream", None),
            text=text[:4000],
            truncated=len(text) > 4000,
        )

    def on_status(self, status: Any) -> None:
        self.emit("status", status=str(status))

    def on_info(self, info: Any) -> None:
        self.emit("info", info=str(info))

    def on_error(self, error: Any) -> None:
        self.emit("wanpg_error", error=str(getattr(error, "message", error)))

    def on_complete(self, result: Any) -> None:
        self.emit(
            "wanpg_complete",
            success=getattr(result, "success", None),
            total_tasks=getattr(result, "total_tasks", None),
            successful_tasks=getattr(result, "successful_tasks", None),
            failed_tasks=getattr(result, "failed_tasks", None),
            errors=[str(error) for error in getattr(result, "errors", [])],
        )

    def close(self) -> None:
        """记录追踪器结束状态。"""
        self.emit("worker_trace_closed", write_error=self.write_error)


def load_request(path: str | Path) -> dict[str, Any]:
    """读取一次性 worker 请求并检查顶层结构。"""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("worker 输入 JSON 顶层必须是对象。")
    return payload


def build_settings(session: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """以 WanGP 自带的 LongCat Avatar 默认项为基础构造任务设置。"""
    settings = session.get_default_settings(MODEL_TYPE)
    settings.update(
        {
            "model_type": MODEL_TYPE,
            "prompt": payload["prompt"],
            "audio_guide": str(Path(payload["audio_file"]).resolve()),
            "resolution": payload["resolution"],
            "video_length": f"{float(payload['duration_seconds']):.4f}s",
            "force_fps": 25,
            "sample_solver": "distill",
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "sliding_window_size": 93,
            "sliding_window_overlap": 13,
            "repeat_generation": 1,
        }
    )
    image_file = payload.get("image_file")
    if image_file:
        settings["image_refs"] = [str(Path(image_file).resolve())]
    return settings


def select_video(paths: list[str]) -> Path:
    """从 WanGP 结果中选取最近生成的 MP4。"""
    candidates = [
        Path(path).resolve()
        for path in paths
        if Path(path).suffix.lower() == ".mp4" and Path(path).is_file()
    ]
    if not candidates:
        raise RuntimeError("WanGP API 没有返回可用的 MP4 视频。")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run(payload: dict[str, Any], diagnostics: WorkerTrace | None = None) -> Path:
    """初始化 WanGP 低显存运行时，等待生成并导出单个 MP4。"""
    memory_profile = validate_memory_profile(payload.get("memory_profile", DEFAULT_MEMORY_PROFILE))
    trace = diagnostics or WorkerTrace.from_environment(memory_profile)
    wangp_dir = Path(payload["wangp_dir"]).resolve()
    output_dir = Path(payload["workspace_dir"]).resolve() / "wangp_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(wangp_dir))

    try:
        if trace is None:
            from shared.api import init
        else:
            with trace.stage("wan_api_import"):
                from shared.api import init

        init_options = {
            "root": wangp_dir,
            "output_dir": output_dir,
            "cli_args": [
                "--offline",
                "--profile",
                str(memory_profile),
                "--attention",
                "sdpa",
                "--gpu",
                "cuda:0",
            ],
            "console_output": True,
            "console_isatty": False,
        }
        if trace is not None:
            init_options["callbacks"] = trace
            with trace.stage("wan_session_init"):
                session = init(**init_options)
            with trace.stage("settings_build"):
                settings = build_settings(session, payload)
            with trace.stage("task_submit"):
                job = session.submit_task(settings)
            with trace.stage("generation_wait"):
                result = job.result(timeout=float(payload["request_timeout"]))
            trace.emit(
                "generation_result",
                success=result.success,
                generated_files=result.generated_files,
                errors=[str(error) for error in result.errors],
            )
        else:
            session = init(**init_options)
            job = session.submit_task(build_settings(session, payload))
            result = job.result(timeout=float(payload["request_timeout"]))
        if not result.success:
            details = "; ".join(error.message for error in result.errors)
            raise RuntimeError(details or "WanGP LongCat Avatar 推理失败。")

        if trace is None:
            generated = select_video(result.generated_files)
        else:
            with trace.stage("output_selection"):
                generated = select_video(result.generated_files)
        output_video = Path(payload["output_video"]).resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        if trace is None:
            shutil.copyfile(generated, output_video)
        else:
            with trace.stage("output_copy"):
                shutil.copyfile(generated, output_video)
        print(f"WanGP 已生成 MP4：{output_video}")
        return output_video
    except BaseException as exc:
        if trace is not None:
            trace.emit("worker_error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if diagnostics is None and trace is not None:
            trace.close()


def main() -> int:
    """解析单次请求并运行 WanGP 任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    payload = load_request(args.input_json)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
