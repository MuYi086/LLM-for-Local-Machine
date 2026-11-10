"""在全局 GPU 锁内执行一次 sd.cpp CLI 并确认进程组回收。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event

from llm_local_runtime import gpu_quarantine_path, gpu_runtime_lock

from inference_gateway.errors import BackendQuarantinedError

from .config import SdcppConfig, positive_number
from .models import GenerationRequest, ModelSpec
from .processes import (
    clear_marker,
    process_identity,
    stop_process_group,
    sync_directory,
    write_marker,
)


class GenerationCancelledError(RuntimeError):
    """生成被调用方取消，进程组已完成收尾。"""


@dataclass(frozen=True)
class GenerationResult:
    """原子发布的生成文件及完整请求生命周期指标。"""

    request_id: str
    output: Path
    metrics: dict


def _validate_output(path: Path, video: bool) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("sd.cpp 未生成预期的常规输出文件。")
    with path.open("rb") as stream:
        header = stream.read(12)
        os.fsync(stream.fileno())
    valid = (
        header[:4] == b"RIFF" and header[8:12] == b"AVI "
        if video
        else header.startswith(b"\x89PNG\r\n\x1a\n")
    )
    if not valid:
        raise RuntimeError("sd.cpp 输出的文件头与预期 PNG/AVI 格式不符。")


def generate(
    config: SdcppConfig,
    model: ModelSpec,
    request: GenerationRequest,
    *,
    cancel_event: Event | None = None,
) -> GenerationResult:
    """执行生成并在确认停止后发布产物；失败只清理本次请求目录。

    Args:
        config: 外部二进制、共享锁和超时配置。
        model: 清单中明确声明的模型合同。
        request: 已结构化的生成请求。
        cancel_event: 可选取消信号；等待 GPU 锁时仍由 queue_timeout 限时。

    Returns:
        含独立产物路径和共享 runtime 指标的结果。

    Raises:
        BackendQuarantinedError: 无法确认资源释放，需运行恢复脚本。
    """
    config.validate_binary()
    for name in ("queue_timeout", "run_timeout", "cleanup_timeout"):
        positive_number(getattr(config, name), name)
    positive_number(config.release_delay, "release_delay", allow_zero=True)
    model.validate_files()
    model.arguments(request, Path("preflight.png"))
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelledError("生成已取消。")
    request_id = uuid.uuid4().hex
    config.output_root.mkdir(parents=True, exist_ok=True)
    staging = config.output_root / f".partial-{request_id}"
    destination = config.output_root / request_id
    video = request.task.endswith("video")
    filename = "result.avi" if video else "result.png"
    marker_path = gpu_quarantine_path(config.lock_path)
    with gpu_runtime_lock(
        config.lock_path, f"stable-diffusion.cpp/{model.id}", timeout_seconds=config.queue_timeout
    ) as metrics:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("生成在等待 GPU 期间被取消。")
        staging.mkdir(mode=0o700)
        record = {
            "schema_version": 1,
            "provider": "stable-diffusion.cpp",
            "phase": "prepared",
            "request_id": request_id,
            "model": model.id,
            "pid": None,
            "pgid": None,
            "executable": str(config.binary),
            "request_dir": str(staging),
            "started_at": time.time(),
        }
        process: subprocess.Popen | None = None
        expected: bytes | None = None
        confirmed = False
        try:
            expected = write_marker(marker_path, record)
            arguments = model.arguments(request, staging / filename)
            inference_error: BaseException | None = None
            with (staging / "worker.log").open("wb") as log:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-I",
                            str(Path(__file__).with_name("worker.py")),
                            str(config.binary),
                            *arguments,
                        ],
                        stdin=subprocess.PIPE,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=staging,
                        start_new_session=True,
                    )
                    identity = process_identity(process.pid)
                    if identity is None or identity[1] != process.pid:
                        raise RuntimeError("sd.cpp worker 身份登记失败。")
                    record.update(
                        phase="registered",
                        pid=process.pid,
                        pgid=process.pid,
                        pid_start_time=identity[0],
                    )
                    if marker_path.read_bytes() != expected:
                        raise BackendQuarantinedError("sd.cpp 启动前隔离标记发生变化。")
                    expected = write_marker(marker_path, record)
                    assert process.stdin is not None
                    process.stdin.write(b"run\n")
                    process.stdin.close()
                    deadline = time.monotonic() + config.run_timeout
                    while True:
                        # WNOWAIT 保留已退出的根进程，防止 PGID 在组回收前被复用。
                        exited = os.waitid(
                            os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                        )
                        if exited is not None:
                            break
                        if cancel_event is not None and cancel_event.is_set():
                            raise GenerationCancelledError("生成已取消。")
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"sd.cpp 执行超过 {config.run_timeout}s。")
                        if log.tell() > 16 * 1024 * 1024:
                            raise RuntimeError("sd.cpp 日志超过单次任务上限。")
                        time.sleep(0.05)
                    returncode = (
                        exited.si_status if exited.si_code == os.CLD_EXITED else -exited.si_status
                    )
                    if returncode != 0:
                        raise RuntimeError(f"sd.cpp 退出码为 {returncode}。")
                except BaseException as exc:
                    inference_error = exc
                    raise
                finally:
                    try:
                        if process is not None:
                            if process.stdin is not None and not process.stdin.closed:
                                process.stdin.close()
                            stop_process_group(process.pid, config.cleanup_timeout)
                            process.wait(timeout=config.cleanup_timeout)
                        confirmed = True
                        if config.release_delay:
                            time.sleep(config.release_delay)
                        if expected is not None:
                            clear_marker(marker_path, expected)
                    except Exception as cleanup_error:
                        confirmed = False
                        if not marker_path.exists():
                            write_marker(marker_path, record)
                        original = (
                            f"；原始执行错误：{inference_error}"
                            if inference_error is not None
                            else ""
                        )
                        raise BackendQuarantinedError(
                            f"sd.cpp 回收未确认，保留 GPU 隔离和任务目录：{staging}{original}"
                        ) from cleanup_error
            _validate_output(staging / filename, video)
            # 共享指标在退出锁后才能得到完整 worker_seconds，此处先发布生成参数。
            metadata = record | {
                "request": {
                    **asdict(request),
                    "init_image": str(request.init_image) if request.init_image else None,
                },
                "binary_sha256": config.binary_sha256,
                "arguments": arguments,
            }
            with (staging / "generation.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(staging)
            staging.rename(destination)
            sync_directory(config.output_root)
        finally:
            if confirmed or expected is None:
                shutil.rmtree(staging, ignore_errors=True)
    return GenerationResult(request_id, destination / filename, metrics.as_dict())
