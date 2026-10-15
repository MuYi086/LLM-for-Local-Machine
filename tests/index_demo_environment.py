"""Index 原生 HTTP Demo 的轻量环境检查。"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentCheck:
    """单项前置检查的名称、结果与诊断信息。"""

    name: str
    passed: bool
    detail: str


def print_checks(checks: list[EnvironmentCheck]) -> bool:
    """打印全部检查项，返回是否满足前置条件。"""
    for check in checks:
        print(f"[{'通过' if check.passed else '未满足'}] {check.name}：{check.detail}")
    return all(check.passed for check in checks)


def check_service_environment(
    *,
    input_path: Path,
    base_url: str,
    expected_backend: str,
    output_type: str,
    text_input: bool = True,
    image_input: bool = False,
) -> list[EnvironmentCheck]:
    """检查输入和原生服务健康，不发生成请求，不需要 OpenAI 客户端或外部源码。"""
    checks = []
    try:
        if text_input:
            valid = bool(input_path.read_text(encoding="utf-8").strip())
        else:
            valid = input_path.is_file() and input_path.stat().st_size > 0
        checks.append(EnvironmentCheck("输入素材", valid, str(input_path)))
    except (OSError, UnicodeError) as exc:
        checks.append(EnvironmentCheck("输入素材", False, str(exc)))
    url = f"{base_url.rstrip('/')}/v1/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = json.load(response)
        backend = status.get("runtime", {}).get("backend", "")
        native = (
            expected_backend in backend and status.get("runtime", {}).get("output") == output_type
        )
        checks.append(
            EnvironmentCheck(
                "原生接口契约", native, backend or "缺少原生接口标识；需重启更新后的服务。"
            )
        )
        ready = status.get("image_ready") if image_input else status.get("ready")
        checks.append(
            EnvironmentCheck(
                "模型运行条件",
                ready is True,
                json.dumps(status.get("available", {}), ensure_ascii=False),
            )
        )
        if "gpu" in status:
            checks.append(
                EnvironmentCheck(
                    "原生 GPU 条件", status["available"].get("gpu_memory") is True, status["gpu"]
                )
            )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        checks.append(EnvironmentCheck("原生服务健康", False, f"{url}：{exc}"))
    return checks
