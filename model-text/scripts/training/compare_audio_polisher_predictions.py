#!/usr/bin/env python3
"""比较有声书文本润色 POC 的基线与 LoRA 预测结果。"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any

_FORBIDDEN_PATTERN = re.compile(
    r"```|</?think>|\[(?:sfx|bgm|emotion|音效|配乐|情绪)[^]]*]|"
    r"schemaVersion|LLM Local Backend|^\s*(?:润色结果|输出|正文|答案)\s*[:：]",
    re.IGNORECASE,
)


def _read_predictions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not {"predict", "label"} <= value.keys():
            raise ValueError(f"{path}:{line_number} 缺少 predict 或 label")
        rows.append({"predict": str(value["predict"]), "label": str(value["label"])})
    return rows


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("预测结果为空")
    similarities = [
        SequenceMatcher(None, row["predict"], row["label"], autojunk=False).ratio() for row in rows
    ]
    length_ratios = [len(row["predict"]) / max(1, len(row["label"])) for row in rows]
    return {
        "samples": len(rows),
        "plain_text_pass": sum(not _FORBIDDEN_PATTERN.search(row["predict"]) for row in rows),
        "mean_character_similarity_to_teacher": round(fmean(similarities), 6),
        "mean_length_ratio_to_teacher": round(fmean(length_ratios), 6),
        "per_sample_character_similarity": [round(value, 6) for value in similarities],
    }


def compare_predictions(baseline_path: Path, adapter_path: Path) -> dict[str, Any]:
    """计算两个预测文件相对于同一教师标签的轻量代理指标。"""
    baseline_rows = _read_predictions(baseline_path)
    adapter_rows = _read_predictions(adapter_path)
    if len(baseline_rows) != len(adapter_rows):
        raise ValueError("基线与 LoRA 预测条数不一致")
    if [row["label"] for row in baseline_rows] != [row["label"] for row in adapter_rows]:
        raise ValueError("基线与 LoRA 使用的教师标签不一致")

    baseline = _summarize(baseline_rows)
    adapter = _summarize(adapter_rows)
    return {
        "metric_scope": "3 条隔离测试样本的格式检查和字符级代理指标，不代表生产质量",
        "baseline": baseline,
        "adapter": adapter,
        "character_similarity_delta": round(
            adapter["mean_character_similarity_to_teacher"]
            - baseline["mean_character_similarity_to_teacher"],
            6,
        ),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """比较预测并写入 JSON 报告。"""
    args = parse_args()
    report = compare_predictions(args.baseline, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
