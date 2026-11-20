"""Linux/WSL 资源采样，不访问模型 API。"""

from __future__ import annotations

import csv
import os
import platform
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .telemetry import append_record


def read_memory(path: Path = Path("/proc/meminfo")) -> dict[str, float]:
    """读取 Linux/WSL 内存口径，交换空间单独记录。"""
    try:
        fields = {
            line.split(":")[0]: int(line.split()[1]) / 1024
            for line in path.read_text().splitlines()
            if ":" in line
        }
        return {
            "total_mib": fields["MemTotal"],
            "used_mib": fields["MemTotal"] - fields["MemAvailable"],
            "available_mib": fields["MemAvailable"],
            "swap_used_mib": fields["SwapTotal"] - fields["SwapFree"],
        }
    except (OSError, ValueError, KeyError, IndexError):
        return {}


def _query_gpu(query: str, fields: list[str]) -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{query}=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return [
            dict(zip(fields, (value.strip() for value in row), strict=True))
            for row in csv.reader(result.stdout.splitlines())
            if len(row) == len(fields)
        ]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
        return result if result >= 0 and result < float("inf") else None
    except ValueError:
        return None


def read_gpu() -> dict[str, Any]:
    """查询整卡及可获得的进程显存；WSL 不支持的指标保留 null。"""
    rows = _query_gpu(
        "gpu",
        [
            "uuid",
            "name",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
        ],
    )
    processes = _query_gpu("compute-apps", ["gpu_uuid", "pid", "used_gpu_memory"])
    return {
        "gpus": None
        if rows is None
        else [
            {
                "uuid": row["uuid"],
                "name": row["name"],
                **{key: _number(row[key]) for key in row if key not in {"uuid", "name"}},
            }
            for row in rows
        ],
        "processes": None
        if processes is None
        else [
            {
                "gpu_uuid": row["gpu_uuid"],
                "pid": _number(row["pid"]),
                "used_mib": _number(row["used_gpu_memory"]),
            }
            for row in processes
        ],
    }


def read_processes(owner_pid: int, *, pss: bool) -> list[dict[str, Any]]:
    """记录启动树和 OpenCode 进程；PID 与 starttime 一同保存以区分复用。"""
    candidates: dict[int, tuple[int, list[str], Path]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            candidates[int(path.name)] = (int(stat[1]), stat, path)
        except (OSError, ValueError, IndexError):
            continue
    opencode = set()
    for pid, (_, _, path) in candidates.items():
        try:
            if "opencode" in (path / "comm").read_text().lower():
                opencode.add(pid)
        except OSError:
            pass
    descendants = {owner_pid, *opencode}
    while True:
        added = {pid for pid, (parent, _, _) in candidates.items() if parent in descendants}
        if added <= descendants:
            break
        descendants.update(added)
    rows = []
    for pid, (_, stat, path) in candidates.items():
        if pid == os.getpid():
            continue
        try:
            comm = (path / "comm").read_text().strip()
            if pid not in descendants:
                continue
            # 仅用固定标签描述角色，不把进程命令行或环境变量持久化。
            role = "opencode" if "opencode" in comm.lower() else "service_tree"
            if role == "service_tree":
                command = (path / "cmdline").read_bytes()
                if b"easel_bridge.app:app" in command:
                    role = "bridge"
            rss_mib = int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 1048576
            row = {
                "pid": pid,
                "ppid": int(stat[1]),
                "start_ticks": int(stat[19]),
                "role": role,
                "rss_mib": rss_mib,
                "cpu_seconds": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
                "pss_mib": None,
            }
            if pss:
                try:
                    for line in (path / "smaps_rollup").read_text().splitlines():
                        if line.startswith("Pss:"):
                            row["pss_mib"] = int(line.split()[1]) / 1024
                            break
                except (OSError, ValueError):
                    pass
            rows.append(row)
        except (OSError, ValueError, IndexError):
            continue
    return rows


def collect(directory: Path, owner_pid: int, interval: float, log_path: Path | None) -> None:
    """持续采样到收到退出信号或启动器退出；最后再记录一次结束快照。"""
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())
    directory.mkdir(parents=True, exist_ok=True)
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration, IndexError):
        cpu_model = None
    manifest = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "owner_pid": owner_pid,
        "interval_seconds": interval,
        "pss_every_samples": 5,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "memory": read_memory(),
        "memory_scope": "Linux/WSL guest; not Windows host",
        "start_log": str(log_path) if log_path else None,
    }
    append_record(directory / "manifest.jsonl", manifest)
    count = 0
    while True:
        started = time.monotonic()
        gpu = read_gpu()
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "monotonic": started,
            "memory": read_memory(),
            "gpu": gpu,
            "processes": read_processes(owner_pid, pss=count % 5 == 0),
            "load_average": list(os.getloadavg()),
        }
        row["sample_duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        append_record(directory / "resources.jsonl", row)
        count += 1
        if stop.is_set() or os.getppid() != owner_pid:
            break
        stop.wait(max(0.01, interval - (time.monotonic() - started)))
