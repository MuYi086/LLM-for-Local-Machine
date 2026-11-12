#!/usr/bin/env python3
"""运行 Ornith 网页任务并记录生成结果和资源占用。"""

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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
PROMPT_PATH = TASK_DIR / "task_prompt.txt"
MAX_TOKENS = 8192
TEMPERATURE = 0.0
TOP_P = 1.0
MODELS = (
    {
        "key": "ornith_1_5_35b_a3b",
        "name": "Ornith-1.5-35B-A3B-AD-Q5",
        "api_base": "http://127.0.0.1:8401",
        "route": "/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5",
        "backend_host": "127.0.0.1",
        "backend_port": 8403,
        "output_dir": ROOT / "tests/ornith_1_5_35b_a3b/outputs",
    },

)


@dataclass(frozen=True)
class ResourceSample:
    """记录一次系统、进程和 GPU 资源采样。"""

    timestamp: float
    gpu_used_mib: int | None
    process_rss_mib: float | None
    system_available_mib: float | None


def read_proc_memory(pid: int) -> tuple[float | None, float | None]:
    """读取进程当前 RSS 和生命周期峰值 RSS，单位为 MiB。"""
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


def find_llama_server_pid(port: int) -> int:
    """根据命令行中的监听端口找到指定 llama-server 进程。"""
    proc_dir = Path("/proc")
    for process_dir in proc_dir.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            arguments = (process_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not any(argument.endswith(b"llama-server") for argument in arguments):
            continue
        try:
            port_index = arguments.index(b"--port")
        except ValueError:
            continue
        if port_index + 1 < len(arguments) and arguments[port_index + 1] == str(port).encode():
            return int(process_dir.name)
    raise RuntimeError(f"没有找到监听端口 {port} 的 llama-server 进程。")


def read_system_available_mib() -> float | None:
    """读取 Linux MemAvailable 值，单位为 MiB。"""
    try:
        memory_info = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", memory_info, re.MULTILINE)
    return float(match.group(1)) / 1024 if match else None


def read_gpu_used_mib() -> int | None:
    """查询 CUDA 设备 0 的显存使用量。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == "0":
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def take_sample(pid: int) -> ResourceSample:
    """同时采集显存、模型进程 RSS 和系统可用内存。"""
    rss_mib, _ = read_proc_memory(pid)
    return ResourceSample(
        timestamp=time.monotonic(),
        gpu_used_mib=read_gpu_used_mib(),
        process_rss_mib=rss_mib,
        system_available_mib=read_system_available_mib(),
    )


class ResourceSampler:
    """在模型请求期间以固定间隔采集本机资源数据。"""

    def __init__(self, pid: int, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self) -> None:
        while not self._stop.is_set():
            self.samples.append(take_sample(self.pid))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        """记录基线并启动后台采样。"""
        self.samples.append(take_sample(self.pid))
        self._thread.start()

    def stop(self) -> None:
        """记录结束采样并等待后台采样线程退出。"""
        self.samples.append(take_sample(self.pid))
        self._stop.set()
        self._thread.join(timeout=5)


def parse_stream(
    response: Any,
) -> tuple[str, int | None, float | None, float | None, list[str]]:
    """解析 OpenAI SSE 流并读取实际 completion token 用量。"""
    chunks: list[str] = []
    completion_tokens: int | None = None
    first_content_at: float | None = None
    last_content_at: float | None = None
    finish_reasons: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        event_data = line[5:].strip()
        if event_data == "[DONE]":
            continue
        try:
            event = json.loads(event_data)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
            completion_tokens = usage["completion_tokens"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if isinstance(choice.get("finish_reason"), str):
            finish_reasons.append(choice["finish_reason"])
        delta = choice.get("delta")
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str) and content:
            now = time.monotonic()
            first_content_at = first_content_at or now
            last_content_at = now
            chunks.append(content)
    return "".join(chunks), completion_tokens, first_content_at, last_content_at, finish_reasons


def request_model(
    model: dict[str, Any],
    prompt: str,
    *,
    timeout_seconds: float,
    sampler: ResourceSampler,
) -> dict[str, Any]:
    """向单个模型发送统一的流式 HTML 生成请求并计算耗时。"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{model['api_base']}{model['route']}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    sampler.start()
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"{model['name']} API 返回 HTTP {response.status}。")
            text, completion_tokens, first_content_at, last_content_at, finish_reasons = (
                parse_stream(response)
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{model['name']} API 返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 {model['name']} API：{exc}") from exc
    finally:
        sampler.stop()
    finished_at = time.monotonic()
    text = strip_markdown_fence(text)
    if not text.strip():
        raise RuntimeError(f"{model['name']} 返回了空网页。")
    if "<html" not in text.lower() or "</html>" not in text.lower():
        preview = f"开头={text[:240]!r}；结尾={text[-240:]!r}"
        raise RuntimeError(
            f"{model['name']} 输出缺少完整的 <html>...</html> 文档；"
            f"字符数={len(text)}，completion_tokens={completion_tokens}，"
            f"finish_reasons={finish_reasons}，{preview}"
        )
    generation_seconds = (
        last_content_at - first_content_at
        if first_content_at is not None and last_content_at is not None
        else None
    )
    generated_tokens_per_second = (
        completion_tokens / generation_seconds
        if completion_tokens is not None and generation_seconds and generation_seconds > 0
        else None
    )
    return {
        "model": model["name"],
        "key": model["key"],
        "elapsed_seconds": round(finished_at - started_at, 3),
        "time_to_first_token_seconds": (
            round(first_content_at - started_at, 3) if first_content_at is not None else None
        ),
        "generation_interval_seconds": (
            round(generation_seconds, 3) if generation_seconds is not None else None
        ),
        "completion_tokens": completion_tokens,
        "tokens_per_second": (
            round(generated_tokens_per_second, 2)
            if generated_tokens_per_second is not None
            else None
        ),
        "finish_reasons": finish_reasons,
        "html_characters": len(text),
        "html_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "html": text,
        "samples": [asdict(sample) for sample in sampler.samples],
    }


def strip_markdown_fence(text: str) -> str:
    """清除模型意外添加在完整 HTML 外侧的代码围栏。"""
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:html)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


def verify_service(model: dict[str, Any]) -> None:
    """确认 API 路由可用，且 backend 在请求前已卸载模型。"""
    openapi_url = f"{model['api_base']}/openapi.json"
    try:
        with urllib.request.urlopen(openapi_url, timeout=5) as response:
            schema = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 {model['name']} 服务 OpenAPI：{exc}") from exc
    if model["route"] not in schema.get("paths", {}):
        raise RuntimeError(f"{model['name']} 服务未公开预期路由 {model['route']}。")
    backend_url = f"http://{model['backend_host']}:{model['backend_port']}/props"
    try:
        with urllib.request.urlopen(backend_url, timeout=5) as response:
            properties = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 {model['name']} backend 状态：{exc}") from exc
    if properties.get("is_sleeping") is not True:
        raise RuntimeError(f"{model['name']} backend 尚未进入 sleep-idle，停止比较以保持公平。")


def summarize_samples(
    samples: list[dict[str, Any]], pid: int, sample_interval_seconds: float
) -> dict[str, Any]:
    """汇总 GPU 峰值、模型进程 RSS 和系统可用内存。"""
    gpu_values = [
        sample["gpu_used_mib"] for sample in samples if sample["gpu_used_mib"] is not None
    ]
    rss_values = [
        sample["process_rss_mib"] for sample in samples if sample["process_rss_mib"] is not None
    ]
    available_values = [
        sample["system_available_mib"]
        for sample in samples
        if sample["system_available_mib"] is not None
    ]
    baseline = samples[0] if samples else {}
    _, process_high_water_mib = read_proc_memory(pid)
    gpu_baseline = baseline.get("gpu_used_mib")
    rss_baseline = baseline.get("process_rss_mib")
    available_baseline = baseline.get("system_available_mib")
    gpu_peak = max(gpu_values) if gpu_values else None
    rss_peak = max(rss_values) if rss_values else None
    available_min = min(available_values) if available_values else None
    return {
        "sample_count": len(samples),
        "sample_interval_seconds": sample_interval_seconds,
        "gpu_device_index": 0,
        "gpu_memory_baseline_mib": gpu_baseline,
        "gpu_memory_peak_mib": gpu_peak,
        "gpu_memory_peak_delta_mib": (
            gpu_peak - gpu_baseline if gpu_peak is not None and gpu_baseline is not None else None
        ),
        "llama_server_pid": pid,
        "llama_server_rss_baseline_mib": rss_baseline,
        "llama_server_rss_peak_sampled_mib": rss_peak,
        "llama_server_rss_peak_delta_mib": (
            rss_peak - rss_baseline if rss_peak is not None and rss_baseline is not None else None
        ),
        "llama_server_rss_high_water_mib_at_end": process_high_water_mib,
        "system_available_memory_baseline_mib": available_baseline,
        "system_available_memory_min_mib": available_min,
        "system_memory_used_peak_delta_mib": (
            available_baseline - available_min
            if available_baseline is not None and available_min is not None
            else None
        ),
    }


def inspect_page_quality(model_key: str, page: str) -> list[str]:
    """对照任务要求检查页面源码中的几项明显缺漏。"""
    notes: list[str] = []
    malformed_label_fragments = re.findall(r"\+ item\.(?:place|name) \+ 」", page)
    if malformed_label_fragments:
        notes.append(
            f"aria-label 字符串拼接有 {len(malformed_label_fragments)} 处引号错误，"
            "原始页的内联 JavaScript 无法解析；并排预览副本只补齐这些引号，原始页保持不变。"
        )
    return notes


def make_preview_copy(model_key: str, page: str) -> tuple[str, int]:
    """为浏览器预览修复已识别的标签字符串引号，不改写原始模型输出。"""
    if model_key != "ornith_1_5_35b_a3b":
        return page, 0
    replacements = (
        ("+ item.place + 」", '+ item.place + "」'),
        ("+ item.name + 」", '+ item.name + "」'),
    )
    repaired_page = page
    repair_count = 0
    for original, repaired in replacements:
        repair_count += repaired_page.count(original)
        repaired_page = repaired_page.replace(original, repaired)
    return repaired_page, repair_count


def render_report(report: dict[str, Any]) -> str:
    """生成人类可读的 Markdown 指标摘要。"""
    performance_rows = []
    resource_rows = []
    for result in report["results"]:
        metrics = result["metrics"]
        performance_rows.append(
            "| {model} | {elapsed} | {ttft} | {tokens} | {speed} |".format(
                model=result["model"],
                elapsed=format_metric(metrics["elapsed_seconds"], " 秒"),
                ttft=format_metric(metrics["time_to_first_token_seconds"], "秒"),
                tokens=format_metric(metrics["completion_tokens"], " token"),
                speed=format_metric(metrics["tokens_per_second"], " token/s"),
            )
        )
        resource_rows.append(
            "| {model} | {rss} | {ram} | {vram} |".format(
                model=result["model"],
                rss=format_memory_triplet(
                    metrics["llama_server_rss_baseline_mib"],
                    metrics["llama_server_rss_peak_sampled_mib"],
                    metrics["llama_server_rss_peak_delta_mib"],
                ),
                ram=(
                    f"{format_metric(metrics['system_available_memory_baseline_mib'], ' MiB')} → "
                    f"{format_metric(metrics['system_available_memory_min_mib'], ' MiB')} "
                    f"(已用增量峰值 {format_metric(metrics['system_memory_used_peak_delta_mib'], ' MiB')})"
                ),
                vram=format_memory_triplet(
                    metrics["gpu_memory_baseline_mib"],
                    metrics["gpu_memory_peak_mib"],
                    metrics["gpu_memory_peak_delta_mib"],
                ),
            )
        )
    order = " → ".join(result["model"] for result in report["results"])
    quality_notes = [
        f"- **{result['model']}**："
        + ("；".join(result["quality_notes"]) or "源码抽查未发现上述明显缺漏。")
        for result in report["results"]
    ]
    return "\n".join(
        [
            "# 网页任务模型对比报告",
            "",
            f"- 执行时间：{report['started_at']}",
            f"- 任务：周末城市微旅行规划器（相同提示词 SHA-256：`{report['prompt_sha256']}`）",
            f"- 执行顺序：{order}；严格串行。",
            f"- 生成配置：`temperature={TEMPERATURE}`、`top_p={TOP_P}`、`max_tokens={MAX_TOKENS}`；关闭 thinking；流式响应。",
            "",
            "## 性能",
            "",
            "| 模型 | 任务耗时 | 首 token 延迟 | 输出 token | 生成速度 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *performance_rows,
            "",
            "## 内存与显存",
            "",
            "| 模型 | llama.cpp RSS 基线 → 峰值（增量） | 系统可用内存 基线 → 最低值（已用增量峰值） | GPU 0 显存 基线 → 峰值（增量） |",
            "| --- | ---: | ---: | ---: |",
            *resource_rows,
            "",
            "## 源码观察",
            "",
            *quality_notes,
            "",
            "## 页面预览",
            "",
            "- [并排预览](latest_comparison.html)",
            *[
                f"- [{result['model']} 原始 HTML]({os.path.relpath(ROOT / result['html_file'], TASK_DIR)})"
                for result in report["results"]
            ],
            *[
                f"- [{result['model']} 预览副本]({os.path.relpath(ROOT / result['preview_file'], TASK_DIR)})"
                for result in report["results"]
                if result["preview_file"] != result["html_file"]
            ],
            "",
            "任务耗时包含 HTTP 排队、模型加载、流式生成和服务端卸载等待。速度使用服务返回的实际 `completion_tokens` 除以首末文本块到达时间差。内存与显存每 0.25 秒采样一次，短暂峰值可能未被采到；系统其他进程会影响全局内存和显存读数。源码观察为静态检查，不是浏览器交互测试。",
            "",
        ]
    )


def format_metric(value: int | float | None, suffix: str) -> str:
    """格式化指标值，缺少数据时明确标为不可用。"""
    if value is None:
        return "不可用"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def format_memory_triplet(
    baseline: int | float | None,
    peak: int | float | None,
    delta: int | float | None,
) -> str:
    """格式化内存基线、峰值和峰值增量。"""
    return (
        f"{format_metric(baseline, ' MiB')} → {format_metric(peak, ' MiB')} "
        f"({format_metric(delta, ' MiB')})"
    )


def write_comparison_page(report: dict[str, Any]) -> None:
    """生成离线预览页面。"""
    frames = []
    for result in report["results"]:
        source_path = ROOT / result["preview_file"]
        source = escape_html(source_path.read_text(encoding="utf-8"))
        note = "；".join(result["quality_notes"])
        note_markup = f'<p class="note">{escape_html(note)}</p>' if note else ""
        frames.append(
            f"<section><h2>{escape_html(result['model'])}</h2>{note_markup}"
            f'<iframe sandbox="allow-scripts allow-modals" '
            f'title="{escape_html(result["model"])} 生成网页" srcdoc="{source}"></iframe></section>'
        )
    page = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ornith 网页任务结果</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f3f5f8; color: #172033; }
    header { padding: 1rem 1.5rem; background: #fff; border-bottom: 1px solid #dfe4ec; }
    h1 { margin: 0; font-size: 1.2rem; }
    main { display: grid; grid-template-columns: minmax(0, 1fr); gap: 1rem; padding: 1rem; }
    section { min-width: 0; }
    h2 { margin: 0 0 .5rem; font-size: 1rem; }
    .note { margin: 0 0 .5rem; color: #9a3412; font-size: .85rem; }
    iframe { display: block; width: 100%; height: calc(100vh - 6rem); border: 1px solid #d7dce5; border-radius: .5rem; background: white; }
    @media (max-width: 800px) { main { grid-template-columns: 1fr; } iframe { height: 75vh; } }
  </style>
</head>
<body>
  <header><h1>同一网页任务：模型输出并排预览</h1></header>
  <main>__FRAMES__</main>
</body>
</html>
""".replace("__FRAMES__", "\n".join(frames))
    (TASK_DIR / "latest_comparison.html").write_text(page, encoding="utf-8")


def escape_html(value: str) -> str:
    """转义用于比较页面标题的 HTML 特殊字符。"""
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def main() -> int:
    """串行执行两种模型，保存 HTML 输出和对比指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=1800, help="每次请求超时秒数。")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="资源采样间隔秒数。",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.sample_interval <= 0:
        parser.error("timeout 和 sample-interval 必须为正数。")
    try:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        if not prompt:
            raise RuntimeError("网页任务提示词为空。")
        cache_dir = Path(os.getenv("RUNTIME_CACHE_DIR", ROOT / "storage/.cache/runtime"))
        lock_file = Path(os.getenv("GPU_LOCK_FILE", cache_dir / "gpu-runtime.lock"))
        if Path(f"{lock_file}.quarantine").exists():
            raise RuntimeError(
                f"检测到 GPU quarantine 标记 {lock_file}.quarantine；请先按仓库恢复流程处理。"
            )
        for model in MODELS:
            verify_service(model)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        report: dict[str, Any] = {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": run_id,
            "prompt_file": "task_prompt.txt",
            "prompt_sha256": prompt_sha256,
            "generation": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "thinking_enabled": False,
                "stream": True,
            },
            "results": [],
        }
        for model in MODELS:
            print(f"开始执行：{model['name']}", flush=True)
            pid = find_llama_server_pid(model["backend_port"])
            sampler = ResourceSampler(pid, args.sample_interval)
            result = request_model(
                model,
                prompt,
                timeout_seconds=args.timeout,
                sampler=sampler,
            )
            model["output_dir"].mkdir(parents=True, exist_ok=True)
            html_path = model["output_dir"] / f"{run_id}.html"
            page = result.pop("html")
            html_path.write_text(page, encoding="utf-8")
            preview_page, repair_count = make_preview_copy(model["key"], page)
            preview_path = html_path
            if repair_count:
                preview_path = model["output_dir"] / f"{run_id}.preview.html"
                preview_path.write_text(preview_page, encoding="utf-8")
            quality_notes = inspect_page_quality(model["key"], page)
            metrics = summarize_samples(result.pop("samples"), pid, args.sample_interval)
            report["results"].append(
                {
                    "model": model["name"],
                    "key": model["key"],
                    "html_file": str(html_path.relative_to(ROOT)),
                    "preview_file": str(preview_path.relative_to(ROOT)),
                    "quality_notes": quality_notes,
                    "metrics": {
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"model", "key"}
                        },
                        **metrics,
                    },
                }
            )
            print(f"  页面：{html_path.relative_to(ROOT)}", flush=True)
            print(f"  总耗时：{result['elapsed_seconds']} 秒", flush=True)
            print(f"  输出 token：{result['completion_tokens']}", flush=True)
            print(f"  速度：{result['tokens_per_second']} token/s", flush=True)
        json_path = TASK_DIR / "latest_report.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path = TASK_DIR / "latest_report.md"
        markdown_path.write_text(render_report(report), encoding="utf-8")
        write_comparison_page(report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"网页模型对比执行失败：{exc}", file=sys.stderr)
        return 1
    print(f"对比报告：{markdown_path.relative_to(ROOT)}")
    print(f"并排预览：{(TASK_DIR / 'latest_comparison.html').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
