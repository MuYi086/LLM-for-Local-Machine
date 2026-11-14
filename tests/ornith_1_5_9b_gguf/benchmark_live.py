#!/usr/bin/env python3
"""经由 Ornith 9B API 实测生成，并记录时延、吞吐、内存和显存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
API_ROUTE = "/v1/ornith-ai/Ornith-1.5-9B-GGUF"
MODEL_ALIAS = "Ornith-1.5-9B-GGUF"
DEFAULT_PROMPT = (
    "请用中文分点解释模型量化是什么，并说明 Q4 与 Q8 的主要区别。回答控制在 150 到 250 个汉字。"
)


@dataclass(frozen=True)
class StreamResult:
    """保存 SSE 文本、服务端 token 用量和正文事件时间。"""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    first_content_at: float | None
    last_content_at: float | None
    finish_reasons: list[str]


@dataclass(frozen=True)
class ResourceSample:
    """保存一次系统内存、backend RSS 和 GPU 0 采样。"""

    timestamp: float
    gpu_used_mib: int | None
    gpu_total_mib: int | None
    backend_rss_mib: float | None
    backend_hwm_mib: float | None
    host_memory_used_mib: float | None
    host_memory_available_mib: float | None
    host_memory_total_mib: float | None


def parse_sse_stream(
    response: Iterable[bytes | str], *, clock: Callable[[], float] = time.monotonic
) -> StreamResult:
    """解析 OpenAI SSE 响应并记录首末非空正文块的到达时间。"""
    content_chunks: list[str] = []
    prompt_tokens = None
    completion_tokens = None
    first_content_at = None
    last_content_at = None
    finish_reasons: list[str] = []

    for raw_line in response:
        line = (
            raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        )
        line = line.strip()
        if not line.startswith("data:"):
            continue
        event_data = line[5:].strip()
        if event_data == "[DONE]":
            continue
        try:
            event = json.loads(event_data)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        usage = event.get("usage")
        if isinstance(usage, dict):
            prompt_value = usage.get("prompt_tokens")
            completion_value = usage.get("completion_tokens")
            if isinstance(prompt_value, int):
                prompt_tokens = prompt_value
            if isinstance(completion_value, int):
                completion_tokens = completion_value

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons.append(finish_reason)
        delta = choice.get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str) and content:
            now = clock()
            if first_content_at is None:
                first_content_at = now
            last_content_at = now
            content_chunks.append(content)

    return StreamResult(
        text="".join(content_chunks),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        first_content_at=first_content_at,
        last_content_at=last_content_at,
        finish_reasons=finish_reasons,
    )


def summarize_stream(
    result: StreamResult, *, request_started_at: float, request_finished_at: float
) -> dict[str, float | int | None]:
    """计算端到端耗时、首 token 延迟和正文生成阶段平均 token/s。"""
    generation_seconds = (
        result.last_content_at - result.first_content_at
        if result.first_content_at is not None and result.last_content_at is not None
        else None
    )
    tokens_per_second = (
        result.completion_tokens / generation_seconds
        if result.completion_tokens is not None and generation_seconds and generation_seconds > 0
        else None
    )
    return {
        "elapsed_seconds": round(request_finished_at - request_started_at, 3),
        "time_to_first_token_seconds": (
            round(result.first_content_at - request_started_at, 3)
            if result.first_content_at is not None
            else None
        ),
        "generation_interval_seconds": (
            round(generation_seconds, 3) if generation_seconds is not None else None
        ),
        "average_tokens_per_second": (
            round(tokens_per_second, 2) if tokens_per_second is not None else None
        ),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def build_request_payload(prompt: str, *, max_tokens: int) -> dict[str, Any]:
    """构造返回可见正文且带 token 用量的 9B 流式请求。"""
    return {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def read_backend_memory(pid: int) -> tuple[float | None, float | None]:
    """读取 llama-server 当前 RSS 和生命周期峰值 RSS，单位为 MiB。"""
    try:
        status = (Path("/proc") / str(pid) / "status").read_text(encoding="utf-8")
    except OSError:
        return None, None
    values: dict[str, float] = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, amount, *_ = line.split()
            values[name.rstrip(":")] = float(amount) / 1024
    return values.get("VmRSS"), values.get("VmHWM")


def read_host_memory() -> tuple[float | None, float | None, float | None]:
    """读取 Linux/WSL guest 总内存、已用内存和可用内存，单位为 MiB。"""
    try:
        fields = {
            match.group(1): float(match.group(2)) / 1024
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if (match := re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB$", line))
        }
    except OSError:
        return None, None, None
    total = fields.get("MemTotal")
    available = fields.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return total, used, available


def read_gpu_memory() -> tuple[int | None, int | None]:
    """查询 GPU 0 全局显存已用量和总量；其他进程占用也会计入。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or fields[0] != "0":
            continue
        try:
            return int(fields[1]), int(fields[2])
        except ValueError:
            return None, None
    return None, None


def take_resource_sample(pid: int) -> ResourceSample:
    """采样 GPU、llama-server RSS 和 Linux/WSL guest 内存。"""
    rss_mib, hwm_mib = read_backend_memory(pid)
    host_total, host_used, host_available = read_host_memory()
    gpu_used, gpu_total = read_gpu_memory()
    return ResourceSample(
        timestamp=time.monotonic(),
        gpu_used_mib=gpu_used,
        gpu_total_mib=gpu_total,
        backend_rss_mib=rss_mib,
        backend_hwm_mib=hwm_mib,
        host_memory_used_mib=host_used,
        host_memory_available_mib=host_available,
        host_memory_total_mib=host_total,
    )


class ResourceSampler:
    """在一次真实推理期间定时采集内存和显存。"""

    def __init__(self, pid: int, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self) -> None:
        while not self._stop.is_set():
            self.samples.append(take_resource_sample(self.pid))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        """先记录请求前基线，再启动周期采样线程。"""
        self.samples.append(take_resource_sample(self.pid))
        self._thread.start()

    def stop(self) -> None:
        """记录结束快照并等待采样线程退出。"""
        self.samples.append(take_resource_sample(self.pid))
        self._stop.set()
        self._thread.join(timeout=5)


def summarize_resources(
    samples: list[ResourceSample], *, interval_seconds: float
) -> dict[str, float | int | None]:
    """从采样序列汇总基线、峰值和相对基线增量。"""
    if not samples:
        raise ValueError("没有可汇总的资源采样。")

    def values(field: str) -> list[float]:
        return [value for sample in samples if (value := getattr(sample, field)) is not None]

    def rounded(value: float | None) -> float | None:
        return round(value, 2) if value is not None else None

    gpu_used = values("gpu_used_mib")
    backend_rss = values("backend_rss_mib")
    backend_hwm = values("backend_hwm_mib")
    host_used = values("host_memory_used_mib")
    host_available = values("host_memory_available_mib")
    host_total = values("host_memory_total_mib")

    def peak_delta(series: list[float]) -> float | None:
        return max(series) - series[0] if series else None

    return {
        "sample_interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "gpu_used_baseline_mib": gpu_used[0] if gpu_used else None,
        "gpu_used_peak_mib": max(gpu_used) if gpu_used else None,
        "gpu_used_peak_delta_mib": peak_delta(gpu_used),
        "gpu_total_mib": max(values("gpu_total_mib"), default=None),
        "backend_rss_baseline_mib": rounded(backend_rss[0] if backend_rss else None),
        "backend_rss_peak_mib": rounded(max(backend_rss) if backend_rss else None),
        "backend_rss_peak_delta_mib": rounded(peak_delta(backend_rss)),
        "backend_hwm_peak_mib": rounded(max(backend_hwm) if backend_hwm else None),
        "host_memory_total_mib": rounded(max(host_total) if host_total else None),
        "host_memory_used_baseline_mib": rounded(host_used[0] if host_used else None),
        "host_memory_used_peak_mib": rounded(max(host_used) if host_used else None),
        "host_memory_used_peak_delta_mib": rounded(peak_delta(host_used)),
        "host_memory_available_min_mib": rounded(min(host_available) if host_available else None),
    }


def find_backend_pid(port: int) -> int:
    """从进程命令行查找监听指定端口的 llama-server。"""
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            arguments = (process_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not arguments or not arguments[0].endswith(b"llama-server"):
            continue
        try:
            port_index = arguments.index(b"--port")
        except ValueError:
            continue
        if port_index + 1 < len(arguments) and arguments[port_index + 1] == str(port).encode():
            return int(process_dir.name)
    raise RuntimeError(f"没有找到监听端口 {port} 的 llama-server 进程。")


def parse_backend_context_size(arguments: list[bytes]) -> int | None:
    """从 llama-server 启动参数解析实际 context token 上限。"""
    try:
        index = arguments.index(b"--ctx-size")
        return int(arguments[index + 1])
    except (ValueError, IndexError):
        return None


def read_backend_context_size(pid: int) -> int | None:
    """读取运行中 llama-server 的 `--ctx-size` 参数。"""
    try:
        arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return None
    return parse_backend_context_size(arguments)


def verify_service(base_url: str, backend_port: int) -> None:
    """检查 9B 路由和 backend 空闲状态，避免直接调用绕过 API 锁。"""
    runtime_cache = Path(os.getenv("RUNTIME_CACHE_DIR", ROOT / "storage/.cache/runtime"))
    lock_file = Path(os.getenv("GPU_LOCK_FILE", runtime_cache / "gpu-runtime.lock"))
    quarantine = Path(f"{lock_file}.quarantine")
    if quarantine.exists():
        raise RuntimeError(f"存在 GPU 隔离标记 {quarantine}；请先按仓库恢复流程处理。")

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/openapi.json", timeout=5) as response:
            schema = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 Ornith 9B API OpenAPI：{exc}") from exc
    if API_ROUTE not in schema.get("paths", {}):
        raise RuntimeError(f"Ornith 9B API 未公开预期路由 {API_ROUTE}。")

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{backend_port}/props", timeout=5
        ) as response:
            properties = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 Ornith 9B backend 状态：{exc}") from exc
    if properties.get("is_sleeping") is not True:
        raise RuntimeError("Ornith 9B backend 尚未进入 sleep-idle；停止实测以遵守共享 GPU 锁约定。")


def request_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    sampler: ResourceSampler,
) -> tuple[StreamResult, dict[str, float | int | None]]:
    """通过 9B API 发起流式实测并返回响应与性能指标。"""
    payload = build_request_payload(prompt, max_tokens=max_tokens)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{API_ROUTE}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    sampler.start()
    request_started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = parse_sse_stream(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ornith 9B API 返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 Ornith 9B API：{exc}") from exc
    finally:
        sampler.stop()
    request_finished_at = time.monotonic()
    if not result.text.strip():
        raise RuntimeError("Ornith 9B API 返回了空的 assistant 文本。")
    metrics = summarize_stream(
        result,
        request_started_at=request_started_at,
        request_finished_at=request_finished_at,
    )
    return result, metrics


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """以唯一文件名保存 JSON 与 Markdown 实测报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = output_dir / f"ornith-9b-live-{run_id}.json"
    markdown_path = output_dir / f"ornith-9b-live-{run_id}.md"
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    result = report["result"]
    resources = report["resources"]
    lines = [
        "# Ornith-1.5-9B-GGUF 调用实测",
        "",
        f"- 时间：{report['started_at']}",
        f"- 状态：{report['status']}",
        f"- 路由：`{report['api_url']}`",
        f"- Backend context：{report['backend_context_tokens']} tokens",
        f"- 请求参数：max_tokens={report['request']['max_tokens']}，stream=true",
        f"- 输出文件：`{result['text_file']}`",
        "",
        "## 推理指标",
        "",
        "| 指标 | 实测值 |",
        "| --- | ---: |",
        f"| 总耗时（含排队、加载及服务端收尾） | {result['elapsed_seconds']} 秒 |",
        f"| 首 token 延迟（到首个非空正文块） | {result['time_to_first_token_seconds']} 秒 |",
        f"| 平均生成速度（completion tokens / 首末正文块间隔） | {result['average_tokens_per_second']} token/s |",
        f"| prompt tokens | {result['prompt_tokens']} |",
        f"| completion tokens | {result['completion_tokens']} |",
        f"| finish reason | {', '.join(result['finish_reasons']) or '未返回'} |",
        "",
        "## 内存和显存",
        "",
        "| 指标 | 基线 | 峰值/最低值 | 相对基线变化 |",
        "| --- | ---: | ---: | ---: |",
        f"| GPU 0 显存已用 | {resources['gpu_used_baseline_mib']} MiB | {resources['gpu_used_peak_mib']} MiB | +{resources['gpu_used_peak_delta_mib']} MiB |",
        f"| llama-server RSS | {resources['backend_rss_baseline_mib']} MiB | {resources['backend_rss_peak_mib']} MiB | +{resources['backend_rss_peak_delta_mib']} MiB |",
        f"| WSL/Linux guest 已用内存 | {resources['host_memory_used_baseline_mib']} MiB | {resources['host_memory_used_peak_mib']} MiB | +{resources['host_memory_used_peak_delta_mib']} MiB |",
        f"| WSL/Linux guest 可用内存 | — | {resources['host_memory_available_min_mib']} MiB（最低） | — |",
        "",
        f"采样间隔 {resources['sample_interval_seconds']} 秒，共 {resources['sample_count']} 次。GPU 数值是 GPU 0 全局占用；主机内存按 WSL/Linux guest 的 `/proc/meminfo` 统计；RSS 仅指 llama-server 进程。",
        "",
    ]
    with markdown_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    return json_path, markdown_path


def main() -> int:
    """检查现有服务并执行一条真实 9B 请求，保存性能报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("ORNITH_9B_TEST_BASE_URL", "http://127.0.0.1:8503"),
        help="Ornith 9B API 根地址。",
    )
    parser.add_argument(
        "--backend-port", type=int, default=int(os.getenv("ORNITH_9B_LLAMA_CPP_PORT", "8505"))
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给模型的中文提示。")
    parser.add_argument("--max-tokens", type=int, default=256, help="生成 token 上限。")
    parser.add_argument("--timeout", type=float, default=1800, help="请求超时秒数。")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="资源采样间隔秒数。")
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "ORNITH_9B_TEST_OUTPUT_DIR",
            str(Path(os.getenv("STORAGE_DIR", ROOT / "storage")) / "llm_tests/ornith_1_5_9b_gguf"),
        ),
        help="实测报告和生成文本输出目录。",
    )
    args = parser.parse_args()
    if (
        not args.prompt.strip()
        or args.max_tokens <= 0
        or args.timeout <= 0
        or args.sample_interval <= 0
        or not 1 <= args.backend_port <= 65535
    ):
        parser.error("prompt、max-tokens、timeout、sample-interval 和 backend-port 必须有效。")

    try:
        verify_service(args.base_url, args.backend_port)
        backend_pid = find_backend_pid(args.backend_port)
        backend_context_tokens = read_backend_context_size(backend_pid)
        sampler = ResourceSampler(backend_pid, args.sample_interval)
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        result, metrics = request_completion(
            base_url=args.base_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
            sampler=sampler,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        text_path = output_dir / f"ornith-9b-live-{run_id}.txt"
        with text_path.open("x", encoding="utf-8") as output:
            output.write(result.text.strip())
            output.write("\n")
        report = {
            "schema_version": 1,
            "status": "success",
            "started_at": started_at,
            "model": MODEL_ALIAS,
            "api_url": f"{args.base_url.rstrip('/')}{API_ROUTE}",
            "backend_port": args.backend_port,
            "backend_pid": backend_pid,
            "backend_context_tokens": backend_context_tokens,
            "request": {
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
            },
            "result": {
                **metrics,
                "finish_reasons": result.finish_reasons,
                "text_characters": len(result.text),
                "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                "text_file": str(text_path),
            },
            "resources": summarize_resources(
                sampler.samples, interval_seconds=args.sample_interval
            ),
            "measurement_notes": {
                "time_to_first_token": "HTTP 请求开始到首个非空正文块，包含排队、模型加载和 prompt eval。",
                "average_tokens_per_second": "服务端 completion_tokens 除以首末正文块到达间隔，不含首 token 前的等待。",
                "gpu_memory": "GPU 0 全局显存占用，可能包含其他进程。",
                "host_memory": "WSL/Linux guest 的 MemTotal - MemAvailable。",
                "backend_rss": "llama-server 进程 RSS；采样按指定间隔，瞬时峰值可能漏采。",
            },
        }
        json_path, markdown_path = write_report(report, output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ornith 9B 实测失败：{exc}", file=sys.stderr)
        return 1

    print(f"生成文本：{text_path}")
    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{markdown_path}")
    print(f"Backend context：{backend_context_tokens} tokens")
    print(f"首 token：{metrics['time_to_first_token_seconds']} 秒")
    print(f"平均生成速度：{metrics['average_tokens_per_second']} token/s")
    print(
        "GPU 0 显存："
        f"{report['resources']['gpu_used_baseline_mib']} → "
        f"{report['resources']['gpu_used_peak_mib']} MiB"
    )
    print(
        "llama-server RSS 峰值："
        f"{report['resources']['backend_rss_peak_mib']} MiB；"
        "guest 内存峰值："
        f"{report['resources']['host_memory_used_peak_mib']} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
