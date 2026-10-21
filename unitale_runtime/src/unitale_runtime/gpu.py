"""跨进程 GPU 队列和轻量指标采集。

所有模型服务都通过同一个文件锁串行使用 GPU。这里采用非阻塞 ``flock``
轮询，而不是无限期阻塞，才能在排队过久时向客户端返回可恢复的服务繁忙错误。
显存采样只调用 ``nvidia-smi``，不会在 HTTP 进程中导入 torch 或加载模型。
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


class GpuLockTimeoutError(RuntimeError):
    """等待共享 GPU 队列超过配置时限。"""

    def __init__(self, label: str, timeout_seconds: float) -> None:
        super().__init__(f"GPU 队列等待超过 {timeout_seconds:.1f}s：{label}")
        self.label = label
        self.timeout_seconds = timeout_seconds


class GpuQuarantinedError(GpuLockTimeoutError):
    """上一次 GPU 请求没有确认模型进程退出，需要人工恢复。"""

    def __init__(self, marker_path: str | Path) -> None:
        RuntimeError.__init__(
            self,
            f"GPU 已隔离，需先确认遗留模型进程退出并检查标记：{marker_path}",
        )
        self.label = "GPU quarantine"
        self.timeout_seconds = 0
        self.marker_path = Path(marker_path)


def gpu_quarantine_path(lock_path: str | Path) -> Path:
    """返回与共享 GPU 锁配套的崩溃隔离标记路径。"""
    return Path(f"{lock_path}.quarantine")


@dataclass
class GpuRuntimeMetrics:
    """一次 GPU 请求的排队、执行和显存观测指标。"""

    label: str
    queue_wait_seconds: float = 0.0
    worker_seconds: float | None = None
    model_load_seconds: float | None = None
    inference_seconds: float | None = None
    worker_exit_seconds: float | None = None
    peak_vram_mib: int | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        """返回可直接放入 health 响应的 JSON 兼容结构。"""
        return asdict(self)


def _env_positive_float(name: str, default: float) -> float | None:
    """读取正数秒数；非正值显式表示不设置时限。"""
    value = float(os.getenv(name, str(default)))
    return value if value > 0 else None


def _read_gpu_memory_mib() -> int | None:
    """读取所有可见 GPU 的已用显存总和，失败时不影响推理。"""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


class _VramSampler:
    """在持锁期间采样显存峰值，避免要求 worker 修改输出协议。"""

    def __init__(self, metrics: GpuRuntimeMetrics) -> None:
        self._metrics = metrics
        self._stop = threading.Event()
        self._interval = max(float(os.getenv("GPU_METRICS_SAMPLE_INTERVAL", "0.5")), 0.1)
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        """开始后台采样。"""
        self._thread.start()

    def stop(self) -> None:
        """停止后台采样并等待短暂收尾。"""
        self._stop.set()
        self._thread.join(timeout=self._interval + 2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            used_mib = _read_gpu_memory_mib()
            if used_mib is not None:
                current_peak = self._metrics.peak_vram_mib
                self._metrics.peak_vram_mib = max(current_peak or used_mib, used_mib)
            self._stop.wait(self._interval)


@contextmanager
def gpu_runtime_lock(
    lock_path: str | Path,
    label: str,
    *,
    timeout_seconds: float | None = None,
) -> Iterator[GpuRuntimeMetrics]:
    """获取共享 GPU 锁并返回本次请求指标。

    锁文件用于服务进程间的互斥；锁释放必须位于 ``finally``，否则 worker
    崩溃或请求异常会让后续任务永久堵塞。默认等待 900 秒，部署方可以通过
    ``GPU_LOCK_WAIT_TIMEOUT`` 调整，设为非正值时保留无限等待语义。
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = timeout_seconds
    if timeout is None:
        timeout = _env_positive_float("GPU_LOCK_WAIT_TIMEOUT", 900.0)
    started = time.perf_counter()
    metrics = GpuRuntimeMetrics(label=label)

    with path.open("a+", encoding="utf-8") as lock_file:
        print(f"[GPU 队列] 等待进入: {label}")
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                waited = time.perf_counter() - started
                if timeout is not None and waited >= timeout:
                    raise GpuLockTimeoutError(label, timeout) from None
                time.sleep(min(0.1, max((timeout or 0.1) - waited, 0.01)))

        metrics.queue_wait_seconds = time.perf_counter() - started
        quarantine = gpu_quarantine_path(path)
        if quarantine.exists():
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            raise GpuQuarantinedError(quarantine)
        execution_started = time.perf_counter()
        sampler = _VramSampler(metrics)
        sampler.start()
        print(f"[GPU 队列] 已进入: {label}，排队 {metrics.queue_wait_seconds:.2f}s")
        try:
            yield metrics
        finally:
            metrics.worker_seconds = time.perf_counter() - execution_started
            metrics.worker_exit_seconds = metrics.worker_seconds
            sampler.stop()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[GPU 指标] {metrics.as_dict()}")
            print(f"[GPU 队列] 已退出: {label}")
