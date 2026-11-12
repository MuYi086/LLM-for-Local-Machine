"""离线标记和汇总性能数据；不调用模型，不覆盖历史报告。"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import stat
import statistics
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .telemetry import enabled, telemetry_directory
from .telemetry_resources import collect


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _test_path(directory: Path, test_id: str, suffix: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", test_id):
        raise ValueError("测试 ID 必须是 begin 返回的 32 位十六进制值。")
    return directory / "tests" / f"{test_id}.{suffix}.json"


def begin_test(directory: Path, label: str) -> str:
    """创建独立测试起点，标签仅用作元数据，不参与文件路径。"""
    test_id = uuid.uuid4().hex
    path = _test_path(directory, test_id, "begin")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(
            {"test_id": test_id, "label": label, "started_at": _now()}, output, ensure_ascii=False
        )
    return test_id


def end_test(directory: Path, test_id: str) -> None:
    """创建独立结束标记；重复结束明确报错而不覆盖。"""
    _test_path(directory, test_id, "begin").read_text()
    with _test_path(directory, test_id, "end").open("x", encoding="utf-8") as output:
        json.dump({"ended_at": _now()}, output)


def _rows(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
                rows.append(row)
            except (ValueError, UnicodeError):
                warnings.append(f"{path.name}:{number} 无效或未完成的记录已跳过。")
    return rows


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含时区。")
    return parsed


def _stats(values: list[float | int | None]) -> dict[str, Any]:
    valid = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
    baseline = values[0] if values else None
    end = values[-1] if values else None
    peak = max(valid) if valid else None
    return {
        "baseline": baseline,
        "sampled_peak": peak,
        "sampled_min": min(valid) if valid else None,
        "end": end,
        "peak_delta": peak - baseline if peak is not None and baseline is not None else None,
        "valid_samples": len(valid),
    }


def build_report(directory: Path, start: str, end: str) -> dict[str, Any]:
    """按带时区的窗口选择调用；不同服务分别汇总以防重复计数。"""
    lower, upper = _time(start), _time(end)
    if upper <= lower:
        raise ValueError("结束时间必须晚于开始时间。")
    warnings = []
    requests = []
    for path in sorted(directory.glob("requests-*.jsonl")):
        for row in _rows(path, warnings):
            if "started_at" not in row:
                continue
            began = _time(row["started_at"])
            finished = _time(row.get("ended_at", row["started_at"]))
            if began <= upper and finished >= lower:
                requests.append(row)
                if began < lower or finished > upper:
                    warnings.append(
                        f"请求 {row.get('request_id')} 跨越窗口边界；保留完整调用数值，不能当窗口内独占耗时。"
                    )
    finished_ids = {row.get("trace_id") for row in requests}
    for path in sorted(directory.glob("starts-*.jsonl")):
        for row in _rows(path, warnings):
            if (
                lower <= _time(row["started_at"]) <= upper
                and row.get("trace_id") not in finished_ids
            ):
                requests.append({**row, "outcome": "unfinished", "duration_ms": None})
                warnings.append(
                    f"请求 {row.get('request_id')} 尚无结束记录：可能仍在收尾、采集失败或进程被终止。"
                )
    requests.sort(key=lambda row: row["started_at"])
    services = {}
    for service in sorted({row.get("service", "unknown") for row in requests}):
        selected = [row for row in requests if row.get("service", "unknown") == service]
        tokens = [row.get("usage", {}).get("completion_tokens") for row in selected]
        delays = [
            row["first_content_ms"]
            for row in selected
            if row.get("first_content_ms") is not None
            and row.get("metadata", {}).get("response_mode") != "nonstream"
        ]
        first_events = [min(values) for row in selected
                        if (values := [row[key] for key in
                            ("first_content_ms", "first_tool_ms", "first_reasoning_ms")
                            if row.get(key) is not None])
                        and row.get("metadata", {}).get("response_mode") != "nonstream"]
        native = [row.get("timings", {}) for row in selected]
        valid_native = [t for t in native if isinstance(t.get("predicted_n"), (int, float))
                        and isinstance(t.get("predicted_ms"), (int, float))
                        and t["predicted_ms"] > 0]
        services[service] = {
            "mean_first_event_ms": statistics.mean(first_events) if first_events else None,
            "weighted_decode_tokens_per_second": (
                sum(t["predicted_n"] for t in valid_native) * 1000
                / sum(t["predicted_ms"] for t in valid_native)
            ) if valid_native else None,
            "native_speed_coverage": f"{len(valid_native)}/{len(selected)}",
            "requests": len(selected),
            "completion_tokens": sum(tokens)
            if all(value is not None for value in tokens)
            else None,
            "usage_coverage": f"{sum(value is not None for value in tokens)}/{len(selected)}",
            "median_first_content_ms": statistics.median(delays) if delays else None,
        }
    all_samples = _rows(directory / "resources.jsonl", warnings)
    before = [row for row in all_samples if _time(row["timestamp"]) <= lower]
    samples = ([before[-1]] if before else []) + [
        row for row in all_samples if lower < _time(row["timestamp"]) <= upper
    ]
    if not before:
        warnings.append("缺少测试前资源基线，首个采样值不能视为任务提交前基线。")
    if not samples or (upper - _time(samples[-1]["timestamp"])).total_seconds() > 3:
        warnings.append("资源采样未覆盖窗口末尾；结束值仅代表最后一次可见采样。")

    def resource_stats(values):
        # 基线不参与任务窗口内的峰值，允许峰值低于基线，不把增量强制截为零。
        stats = _stats(values[1:] if before else values)
        baseline = values[0] if before and values else None
        stats["baseline"] = baseline
        stats["peak_delta"] = (
            stats["sampled_peak"] - baseline
            if stats["sampled_peak"] is not None and baseline is not None
            else None
        )
        return stats

    memory = {
        key: resource_stats([row.get("memory", {}).get(key) for row in samples])
        for key in ("used_mib", "available_mib", "swap_used_mib")
    }
    gpu_ids = {gpu["uuid"] for row in samples for gpu in (row.get("gpu", {}).get("gpus") or [])}
    gpus = {}
    for gpu_id in sorted(gpu_ids):
        values = [
            next(
                (
                    gpu.get("memory.used")
                    for gpu in (row.get("gpu", {}).get("gpus") or [])
                    if gpu.get("uuid") == gpu_id
                ),
                None,
            )
            for row in samples
        ]
        gpus[gpu_id] = resource_stats(values)
    identities = {
        (process["pid"], process["start_ticks"])
        for row in samples
        for process in row.get("processes", [])
    }
    processes = []
    for pid, start_ticks in sorted(identities):
        matched = [
            next(
                (
                    process
                    for process in row.get("processes", [])
                    if (process["pid"], process["start_ticks"]) == (pid, start_ticks)
                ),
                {},
            )
            for row in samples
        ]
        processes.append(
            {
                "pid": pid,
                "start_ticks": start_ticks,
                "role": next(row["role"] for row in matched if row),
                "rss_mib": resource_stats([row.get("rss_mib") for row in matched]),
                "pss_mib": resource_stats([row.get("pss_mib") for row in matched]),
            }
        )
    backend = [
        row
        for row in _rows(directory / "backend.jsonl", warnings)
        if lower <= _time(row["observed_at"]) <= upper
    ]
    return {
        "schema_version": 1,
        "directory": str(directory),
        "manifest": _rows(directory / "manifest.jsonl", warnings),
        "window_start": start,
        "window_end": end,
        "window_seconds": (upper - lower).total_seconds(),
        "services": services,
        "requests": requests,
        "resources": {
            "sample_count": len(samples),
            "window_sample_count": len(samples) - bool(before),
            "first_sample": samples[0]["timestamp"] if samples else None,
            "last_sample": samples[-1]["timestamp"] if samples else None,
            "memory_mib": memory,
            "gpu_used_mib": gpus,
            "processes": processes,
        },
        "backend_unassigned": backend,
        "warnings": warnings,
        "limitations": [
            "网关侧首个非空正文事件延迟不是 OpenCode 客户端 TTFT。非流式正文时间仅为完整响应到达时间。",
            "后端混合日志的 task 无模型/请求关联，不自动归入某个请求；observed_at 是采集时刻。",
            "窗口是手动标记的测试范围，不自动识别 OpenCode 会话、压缩、工具耗时或界面统计。",
            "峰值是采样观测值；整卡/WSL 内存变化不能当作任务独占占用；RSS 不求和冒充物理内存。",
            "未完成或进程被强杀的请求可能没有最终记录；成功响应不能证明没有 OOM。",
        ],
    }


def _display(value: Any) -> str:
    return (
        "未知"
        if value is None
        else str(round(value, 3))
        if isinstance(value, float)
        else str(value)
    )


def render_markdown(report: dict[str, Any]) -> str:
    """生成包含逐请求阶段和资源口径的中文汇总。"""
    lines = [
        "# 本地模型性能记录",
        "",
        f"窗口：{report['window_start']} — {report['window_end']}",
        f"标记窗口耗时：{report['window_seconds']:.3f} 秒（不是 OpenCode 界面总耗时）",
        "",
        "## 测试环境与配置",
        "",
        "以下为启动器传入配置，实际后端运行值仍以日志确认为准。",
        "",
        "```json",
        json.dumps(report.get("manifest", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 逐次调用",
        "",
        "| 服务 | 请求 ID | HTTP/结果 | 用量 输入/输出 | 首正文 ms | 请求总 ms | 排队 ms | 原生 prefill/decode tok/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["requests"]:
        usage = row.get("usage", {})
        stages = row.get("stages_ms", {})
        queue = (
            stages.get("lease_acquired", 0) - stages["queue_enter"]
            if "lease_acquired" in stages and "queue_enter" in stages
            else None
        )
        timings = row.get("timings", {})
        lines.append(
            f"| {row.get('service')} | {row.get('request_id')} | {row.get('status')}/{row.get('outcome', '未知')} | {_display(usage.get('prompt_tokens'))}/{_display(usage.get('completion_tokens'))} | {_display(row.get('first_content_ms'))} | {_display(row.get('duration_ms'))} | {_display(queue)} | {_display(timings.get('prompt_per_second'))}/{_display(timings.get('predicted_per_second'))} |"
        )
    lines += [
        "",
        "## 原生推理与收尾",
        "",
        "| 服务 / 请求 ID | prefill ms | decode ms | cache tokens | MTP 接受/草稿 | 卸载收尾 ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["requests"]:
        native = row.get("timings", {})
        stages = row.get("stages_ms", {})
        cleanup = (
            stages["lease_released"] - stages["cleanup_start"]
            if "lease_released" in stages and "cleanup_start" in stages
            else None
        )
        lines.append(
            f"| {row.get('service')} / {row.get('request_id')} | {_display(native.get('prompt_ms'))} | {_display(native.get('predicted_ms'))} | {_display(native.get('cache_n'))} | {_display(native.get('draft_n_accepted'))}/{_display(native.get('draft_n'))} | {_display(cleanup)} |"
        )
    lines += [
        "",
        "## 分层汇总",
        "",
        "| 服务 | 调用次数 | 输出 tokens 合计 | 用量覆盖 | 首正文事件延迟中位 ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for service, stats in report["services"].items():
        lines.append(
            f"| {service} | {stats['requests']} | {_display(stats['completion_tokens'])} | {stats['usage_coverage']} | {_display(stats['median_first_content_ms'])} |"
        )
    lines += ["", "## 上下文对比指标", "",
              "首事件为服务端收到的首个正文、reasoning 或工具参数事件，包含排队/加载等等待；不是客户端首 token 时间。生成速度按原生 decode 总 tokens / 总秒数加权，缺失请求不补零。", "",
              "| 服务 | 首事件平均 ms | 加权 decode tok/s | 原生速度覆盖 |",
              "| --- | --- | --- | --- |"]
    for service, stats in report["services"].items():
        lines.append(f"| {service} | {_display(stats.get('mean_first_event_ms'))} | {_display(stats.get('weighted_decode_tokens_per_second'))} | {stats.get('native_speed_coverage', '未知')} |")
    lines += [
        "",
        "完整阶段时间、reasoning/工具首事件、finish_reason 和可用 timing 见同名 JSON。服务分别统计，网关与适配器的 token 不能相加。",
        "",
        "## 资源（MiB）",
        "",
        "| 对象 | 基线 | 观测峰值 | 最后可见值 | 峰值减基线 | 有效采样 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    resources = report["resources"]
    objects = {f"WSL/Linux {key}": value for key, value in resources["memory_mib"].items()}
    objects.update({f"整卡 {key}": value for key, value in resources["gpu_used_mib"].items()})
    for process in resources["processes"]:
        for metric in ("rss_mib", "pss_mib"):
            objects[
                f"PID {process['pid']} ({process['role']}, start={process['start_ticks']}) {metric}"
            ] = process[metric]
    for label, stats in objects.items():
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                _display(stats[key])
                for key in ("baseline", "sampled_peak", "end", "peak_delta", "valid_samples")
            )
            + " |"
        )
    lines += [
        "",
        "## 后端原生记录（未关联请求）",
        "",
        "| task | 阶段 | tokens | ms | tok/s | MTP 接受率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["backend_unassigned"]:
        lines.append(
            "| "
            + " | ".join(
                _display(row.get(key))
                for key in (
                    "task_id",
                    "phase",
                    "tokens",
                    "duration_ms",
                    "tokens_per_second",
                    "acceptance",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## 缺失与口径",
        "",
        *["- " + item for item in report["warnings"] + report["limitations"]],
        "",
    ]
    return "\n".join(lines)


def attach_result(target: Path, report: dict[str, Any], source_json: Path) -> None:
    """按测试 ID 原子替换本轮报告的统计块，保留审查正文。

    Args:
        target: 已存在、含本轮唯一统计标记的 Markdown 文件。
        report: 带 test_id 和 test_state 的离线汇总。
        source_json: 对应的详细原始汇总文件路径。
    """
    test_id = report.get("test_id", "")
    _test_path(target.parent, test_id, "begin")
    if target.suffix.lower() != ".md" or target.is_symlink():
        raise ValueError("仅允许更新现有普通 Markdown 报告，不接受符号链接。")
    begin = f"<!-- MODEL_TELEMETRY_BEGIN test_id={test_id} -->"
    end = "<!-- MODEL_TELEMETRY_END -->"
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as original:
        original_stat = os.fstat(original.fileno())
        if not stat.S_ISREG(original_stat.st_mode):
            raise ValueError("统计目标必须是普通文件。")
        fcntl.flock(original.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        content = original.read()
        if (
            content.count("<!-- MODEL_TELEMETRY_BEGIN") != 1
            or content.count(begin) != 1
            or content.count(end) != 1
        ):
            raise ValueError(
                "报告必须包含唯一且匹配本轮测试 ID 的统计块；拒绝覆盖历史或不匹配的报告。"
            )
        left, right = content.index(begin) + len(begin), content.index(end)
        if right < left:
            raise ValueError("统计块结束标记必须位于开始标记之后。")
        state = (
            "结束窗口已固定；未完成请求和缺失覆盖见下表。"
            if report.get("test_state") == "ended"
            else "阶段快照；测试尚未结束，不包含快照后的调用和最终回复。"
        )
        # 降低汇总的标题层级，保留原审查报告的章节结构。
        rendered = "\n".join(
            "##" + line if line.startswith("# ") or line.startswith("## ") else line
            for line in render_markdown(report).splitlines()
        )
        replacement = f"\n\n统计状态：{state}\n\n测试 ID：`{test_id}`\n\n详细数据：`{source_json}`\n\n{rendered}\n\n"
        updated = content[:left] + replacement + content[right:]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=target.parent,
                prefix=".telemetry-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                os.fchmod(output.fileno(), stat.S_IMODE(original_stat.st_mode))
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            current = target.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
                original_stat.st_dev,
                original_stat.st_ino,
                original_stat.st_size,
                original_stat.st_mtime_ns,
            ):
                raise ValueError("报告在汇总期间已被修改，请稍后重新执行 report。")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _directory(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    default = telemetry_directory()
    manifests = list(default.parent.glob("*/manifest.jsonl"))
    if manifests:
        return max(manifests, key=lambda path: path.stat().st_mtime).parent
    if default.exists():
        return default
    raise ValueError("没有性能采集目录，请先运行 bash start.sh 或指定 --directory。")


def main() -> None:
    """执行采样、测试标记或离线汇总子命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", help="start.sh 打印的性能目录；默认自动选择最近启动批次")
    commands = parser.add_subparsers(dest="command", required=True)
    sampler = commands.add_parser("collect")
    sampler.add_argument("--owner-pid", type=int, required=True)
    sampler.add_argument("--interval", type=float, default=1)
    sampler.add_argument("--log", type=Path)
    begin = commands.add_parser("begin")
    begin.add_argument("--label", default="benchmark")
    end = commands.add_parser("end")
    end.add_argument("test_id")
    end.add_argument("--attach-result", type=Path, help="更新含相同测试 ID 标记的本轮结果 Markdown")
    end.add_argument(
        "--settle-seconds",
        type=float,
        default=10,
        help="等待最后一次请求收尾写盘的上限，默认 10 秒",
    )
    report = commands.add_parser("report")
    report.add_argument("--test-id")
    report.add_argument("--attach-result", type=Path, help="将本轮快照写入已有结果的统计块")
    report.add_argument("--start")
    report.add_argument("--end")
    args = parser.parse_args()
    try:
        if args.command == "collect":
            if not enabled():
                return
            if not math.isfinite(args.interval) or args.interval <= 0 or args.owner_pid <= 0:
                raise ValueError("采样间隔和 owner-pid 必须为有限正数。")
            collect(
                Path(args.directory) if args.directory else telemetry_directory(),
                args.owner_pid,
                args.interval,
                args.log,
            )
            return
        if getattr(args, "attach_result", None) and (
            not args.directory or (args.command == "report" and not args.test_id)
        ):
            raise ValueError(
                "--attach-result 必须同时明确 --directory 和本轮测试 ID，不能按最新目录或任意时间段合并。"
            )
        directory = _directory(args.directory)
        if args.command == "begin":
            print(
                f"性能目录：{directory}\n测试 ID：{begin_test(directory, args.label)}", flush=True
            )
            return
        if args.command == "end":
            end_test(directory, args.test_id)
            test_id = args.test_id
        else:
            test_id = args.test_id
        if test_id:
            start = json.loads(_test_path(directory, test_id, "begin").read_text())["started_at"]
            end_path = _test_path(directory, test_id, "end")
            # 同一次观测固定窗口和状态，避免并发 end 把较早快照误标为最终窗口。
            test_state = "ended" if end_path.exists() else "open"
            finish = (
                json.loads(end_path.read_text())["ended_at"] if test_state == "ended" else _now()
            )
        else:
            if not args.start or not args.end:
                raise ValueError("report 需要 --test-id 或同时指定 --start 和 --end。")
            start, finish = args.start, args.end
        result = build_report(directory, start, finish)
        if args.command == "end":
            if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
                raise ValueError("settle-seconds 必须为有限非负数。")
            deadline = time.monotonic() + args.settle_seconds
            while (
                any(row.get("outcome") == "unfinished" for row in result["requests"])
                and time.monotonic() < deadline
            ):
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
                result = build_report(directory, start, finish)
        if test_id:
            result["test_id"] = test_id
            result["test_state"] = test_state
        stem = "report-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        json_path, markdown_path = directory / f"{stem}.json", directory / f"{stem}.md"
        with json_path.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
        with markdown_path.open("x", encoding="utf-8") as output:
            output.write(render_markdown(result))
        print(f"JSON：{json_path}\n报告：{markdown_path}")
        if args.attach_result:
            attach_result(args.attach_result, result, json_path)
            print(f"已更新本轮结果：{args.attach_result}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
