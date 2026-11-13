"""将详细遥测收敛为有严格字节上限的模型可读简报。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_BRIEF_BYTES = 4096


def render_brief(report: dict[str, Any], source_json: Path, target: Path | None = None) -> str:
    """仅输出固定指标，禁止携带请求、逐进程资源或任意额外字段。

    Args:
        report: 离线构造的完整报告。
        source_json: 详细数据的保存位置。
        target: 已成功完成统计块合并的结果文件，未请求合并则为 None。

    Returns:
        不超过 4096 UTF-8 字节、始终可解析的单行 JSON。
    """

    def number(value: Any) -> float | int | None:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            return None
        if isinstance(value, int):
            return value if abs(value) <= 2**63 - 1 else None
        return round(value, 3) if math.isfinite(value) else None

    def peak(value: Any) -> float | int | None:
        return number(value.get("sampled_peak")) if isinstance(value, dict) else None

    resources = report.get("resources", {})
    backend = [
        peak(p.get("rss_mib"))
        for p in resources.get("processes", [])
        if p.get("role") == "ornith_backend"
    ]
    gpu = [peak(v) for v in resources.get("gpu_used_mib", {}).values()]
    services = {}
    selected = sorted(report.get("services", {}).items())[:4]
    for name, stats in selected:
        services[str(name)[:32]] = {
            key: number(stats.get(key))
            for key in (
                "requests",
                "completion_tokens",
                "mean_first_event_ms",
                "median_first_content_ms",
                "weighted_decode_tokens_per_second",
            )
        }
        services[str(name)[:32]]["usage_coverage"] = str(stats.get("usage_coverage", "unknown"))[
            :32
        ]
        services[str(name)[:32]]["native_speed_coverage"] = str(
            stats.get("native_speed_coverage", "unknown")
        )[:32]
    warnings = report.get("warnings", [])
    brief = {
        "test_id": str(report.get("test_id", "unknown"))[:32],
        "test_state": str(report.get("test_state", "unknown"))[:16],
        "window_seconds": number(report.get("window_seconds")),
        "attached": target is not None,
        "json_path": str(source_json)[:256],
        "result_path": str(target)[:256] if target else None,
        "services": services,
        "omitted_services": max(0, len(report.get("services", {})) - len(selected)),
        "resources": {
            "sample_count": number(resources.get("sample_count")),
            "gpu_peak_mib": max((v for v in gpu if v is not None), default=None),
            "guest_used_peak_mib": peak(resources.get("memory_mib", {}).get("used_mib")),
            "backend_rss_peak_mib": max((v for v in backend if v is not None), default=None),
        },
        "warnings_count": len(warnings),
        "warnings": [str(w)[:160] for w in warnings[:3]],
        "warnings_omitted": max(0, len(warnings) - 3),
    }
    encoded = json.dumps(brief, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) >= MAX_BRIEF_BYTES:
        brief.update(
            json_path=str(source_json)[:64],
            result_path=str(target)[:64] if target else None,
            warnings=[],
            warnings_omitted=len(warnings),
        )
        encoded = json.dumps(brief, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) >= MAX_BRIEF_BYTES:
        brief.update(
            services={},
            omitted_services=len(report.get("services", {})),
            json_path=None,
            result_path=None,
        )
        encoded = json.dumps(brief, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) >= MAX_BRIEF_BYTES:
        raise ValueError("简报超出固定字节上限，拒绝输出详细数据。")
    return encoded
