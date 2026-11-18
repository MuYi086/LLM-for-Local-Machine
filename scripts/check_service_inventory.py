"""核对服务清单与 README、AGENTS、启动脚本的服务口径一致性。

服务事实的唯一权威是 ``localai/manifests/audio-cpp-models.json`` 与 ``start.sh``：
新增或下线音频模型后，README 服务表、AGENTS 服务说明和启动端口注册必须同步更新。
本脚本从清单推导期望口径（音频服务数、audio.cpp 服务数、LocalAI 候选数、HTTP 进程总数），
再核对三处文档中的路由、端口与中文数量声明，防止文档再次落后于实现。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "localai/manifests/audio-cpp-models.json"
README_PATH = ROOT / "README.md"
AGENTS_PATH = ROOT / "AGENTS.md"
START_SH_PATH = ROOT / "start.sh"

# 非音频模型的固定 HTTP 服务；LocalAI(8400) 由 manager.py 单独管理，不在这 21 个进程之列。
FIXED_HTTP_SERVICES: tuple[tuple[str, int], ...] = (
    ("Control-plane 控制面", 8300),
    ("Easel bridge 网关", 8500),
    ("MOSS-Image-SFT 图片 API", 8501),
    ("FLUX.3 Klein 图片 API", 8502),
    ("Ornith MTP 9B API", 8504),
    ("Ornith llama.cpp backend（仅 loopback）", 8506),
)

# 清单允许的服务 runtime；出现新 runtime 时应先更新本脚本再合入。
_KNOWN_AUDIO_RUNTIMES = frozenset({"audio_cpp_runtime", "voicedesign"})
_CN_DIGITS = "零一二三四五六七八九"
_NUMERAL_RE = r"[一二三四五六七八九十]{1,3}"
_HTTP_PROCESSES_RE = re.compile(rf"({_NUMERAL_RE})\s*个\s*HTTP\s*进程")
_AUDIO_MODEL_RE = re.compile(rf"({_NUMERAL_RE})\s*个\s*音频模型")
_AUDIO_CPP_RE = re.compile(rf"({_NUMERAL_RE})\s*个\s*audio\.cpp")
_LOCALAI_CANDIDATE_RE = re.compile(rf"({_NUMERAL_RE})\s*个\s*(?:LocalAI|GGUF)\s*候选")
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)
# 服务表行以路由列为标志；README 的环境变量表等其他表格不纳入端口核对。
_SERVICE_ROW_MARK = "/v1/"
# LocalAI(8400) 等基础设施端口不属于仓库 API 服务表，但允许出现在文档中。
KNOWN_INFRA_PORTS = frozenset({8400})


class ServiceInventoryError(ValueError):
    """服务清单与文档口径不一致。"""


def int_to_cn(value: int) -> str:
    """把 1-99 的整数转换为中文数字，用于核对文档中的数量声明。"""
    if not 0 <= value <= 99:
        raise ValueError(f"仅支持 1-99 的中文数字转换：{value}")
    if value < 10:
        return _CN_DIGITS[value]
    tens, ones = divmod(value, 10)
    prefix = "十" if tens == 1 else f"{_CN_DIGITS[tens]}十"
    return prefix + (_CN_DIGITS[ones] if ones else "")


def cn_to_int(text: str) -> int | None:
    """把一到九十九的中文数字解析为整数；无法解析时返回 None。"""
    if not text or any(ch not in _CN_DIGITS + "十" for ch in text):
        return None
    if "十" not in text:
        return _CN_DIGITS.index(text)
    head, _, tail = text.partition("十")
    tens = 1 if head in ("", "一") else (_CN_DIGITS.index(head) if head else 1)
    if tail:
        return tens * 10 + _CN_DIGITS.index(tail)
    return tens * 10


@dataclass(frozen=True)
class ServiceEntry:
    """清单中一个音频模型服务的期望事实。"""

    key: str
    port: int
    route: str
    directory: str
    runtime: str


@dataclass(frozen=True)
class ExpectedInventory:
    """从清单与固定服务推导出的期望口径。"""

    audio_services: tuple[ServiceEntry, ...]
    audio_cpp_services: tuple[ServiceEntry, ...]
    voicedesign_services: tuple[ServiceEntry, ...]
    fixed_services: tuple[tuple[str, int], ...]
    total_http_processes: int


def build_expected_inventory(manifest: dict[str, Any]) -> ExpectedInventory:
    """解析 audio-cpp-models.json，推导服务数与 HTTP 进程总数。

    Args:
        manifest: ``localai/manifests/audio-cpp-models.json`` 的反序列化内容。

    Returns:
        期望口径汇总，包含全部音频服务、audio.cpp 子集与固定服务。

    Raises:
        ServiceInventoryError: 清单缺少 service 块、端口/路由重复或 runtime 未登记。
    """
    entries: list[ServiceEntry] = []
    seen_ports: dict[int, str] = {}
    seen_routes: dict[str, str] = {}
    for model in manifest.get("models", []):
        key = model.get("key", "")
        service = model.get("service") or {}
        runtime = service.get("runtime", "")
        if runtime not in _KNOWN_AUDIO_RUNTIMES:
            raise ServiceInventoryError(f"{key} 的 service.runtime={runtime!r} 未在本脚本登记")
        port = service.get("port")
        route = service.get("route", "")
        if not isinstance(port, int) or not route.startswith("/v1/"):
            raise ServiceInventoryError(f"{key} 缺少有效的 service.port/route")
        if port in seen_ports:
            raise ServiceInventoryError(f"端口 {port} 同时属于 {seen_ports[port]} 与 {key}")
        if route in seen_routes:
            raise ServiceInventoryError(f"路由 {route} 同时属于 {seen_routes[route]} 与 {key}")
        seen_ports[port] = key
        seen_routes[route] = key
        entries.append(
            ServiceEntry(
                key=key,
                port=port,
                route=route,
                directory=service.get("directory", ""),
                runtime=runtime,
            )
        )
    if not entries:
        raise ServiceInventoryError("清单 models 为空，无法推导期望口径")
    fixed_ports = {port for _, port in FIXED_HTTP_SERVICES}
    overlap = fixed_ports & set(seen_ports)
    if overlap:
        raise ServiceInventoryError(f"固定服务与音频模型端口冲突：{sorted(overlap)}")
    audio_cpp = tuple(e for e in entries if e.runtime == "audio_cpp_runtime")
    voicedesign = tuple(e for e in entries if e.runtime == "voicedesign")
    total = len(entries) + len(FIXED_HTTP_SERVICES)
    return ExpectedInventory(
        audio_services=tuple(entries),
        audio_cpp_services=audio_cpp,
        voicedesign_services=voicedesign,
        fixed_services=FIXED_HTTP_SERVICES,
        total_http_processes=total,
    )


def _check_count_claims(
    doc_name: str,
    text: str,
    pattern: re.Pattern[str],
    expected: int,
    label: str,
) -> list[str]:
    """核对文档中的中文数量声明是否全部等于期望值。"""
    errors: list[str] = []
    matches = pattern.findall(text)
    if not matches:
        return [f"{doc_name}：缺少可核对的“{label}”数量声明"]
    for numeral in matches:
        actual = cn_to_int(numeral)
        if actual != expected:
            errors.append(
                f"{doc_name}：“{int_to_cn(expected)}{label}”声明为“{numeral}{label}”"
                f"（解析为 {actual}），请按清单更新"
            )
    return errors


def _check_port_table(doc_name: str, text: str, allowed_ports: set[int]) -> list[str]:
    """核对 Markdown 服务表行中的端口都属于已知服务，识别过期表格行。"""
    errors: list[str] = []
    for row in _TABLE_ROW_RE.findall(text):
        if _SERVICE_ROW_MARK not in row:
            continue
        for cell_port in re.findall(r"\b(\d{4,5})\b", row):
            port = int(cell_port)
            if port not in allowed_ports and port not in KNOWN_INFRA_PORTS:
                errors.append(f"{doc_name}：服务表出现未登记端口 {port}，请核对是否过期")
    return errors


def check_document(
    doc_name: str,
    text: str,
    inventory: ExpectedInventory,
    *,
    require_table_ports: bool = True,
) -> list[str]:
    """核对一份文档（README/AGENTS）与服务清单的一致性。

    Args:
        doc_name: 文档名称，用于错误提示。
        text: 文档全文。
        inventory: 期望口径。
        require_table_ports: 是否要求每个音频端口出现在表格行中
            （README 与 AGENTS 的服务表都应逐项列出）。

    Returns:
        不一致清单；为空表示通过。
    """
    errors: list[str] = []
    for entry in inventory.audio_services:
        if entry.route not in text:
            errors.append(f"{doc_name}：缺少路由 {entry.route}")
        if str(entry.port) not in text:
            errors.append(f"{doc_name}：缺少端口 {entry.port}")
        elif require_table_ports and not re.search(
            rf"^\|[^\n]*\b{entry.port}\b", text, re.MULTILINE
        ):
            errors.append(f"{doc_name}：端口 {entry.port} 未出现在服务表中")
    for label, port in inventory.fixed_services:
        if str(port) not in text:
            errors.append(f"{doc_name}：缺少固定服务 {label}（端口 {port}）")
    allowed_ports = {e.port for e in inventory.audio_services} | {
        port for _, port in inventory.fixed_services
    }
    errors.extend(_check_port_table(doc_name, text, allowed_ports))
    errors.extend(
        _check_count_claims(
            doc_name, text, _HTTP_PROCESSES_RE, inventory.total_http_processes, "个 HTTP 进程"
        )
    )
    errors.extend(
        _check_count_claims(
            doc_name, text, _AUDIO_MODEL_RE, len(inventory.audio_services), "个音频模型"
        )
    )
    errors.extend(
        _check_count_claims(
            doc_name, text, _AUDIO_CPP_RE, len(inventory.audio_cpp_services), "个 audio.cpp"
        )
    )
    errors.extend(
        _check_count_claims(
            doc_name,
            text,
            _LOCALAI_CANDIDATE_RE,
            len(inventory.audio_services),
            "个 LocalAI/GGUF 候选",
        )
    )
    return errors


def check_start_sh(text: str, inventory: ExpectedInventory) -> list[str]:
    """核对 start.sh 是否登记了全部清单端口与路由。"""
    errors: list[str] = []
    for entry in inventory.audio_services:
        if f"PORT:-{entry.port}" not in text:
            errors.append(f"start.sh：缺少 {entry.key} 的端口默认值 {entry.port}")
        if entry.route not in text:
            errors.append(f"start.sh：缺少 {entry.key} 的路由 {entry.route}")
    for label, port in inventory.fixed_services:
        if f"PORT:-{port}" not in text:
            errors.append(f"start.sh：缺少固定服务 {label} 的端口默认值 {port}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """执行一致性核对并打印结果。

    Args:
        argv: 可选参数，目前仅支持传入仓库根目录用于测试。

    Returns:
        全部通过时返回 0，否则返回 1。
    """
    root = Path(argv[0]).resolve() if argv else ROOT
    manifest_path = root / "localai/manifests/audio-cpp-models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = build_expected_inventory(manifest)

    docs = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "AGENTS.md": (root / "AGENTS.md").read_text(encoding="utf-8"),
    }
    start_sh = (root / "start.sh").read_text(encoding="utf-8")

    errors: list[str] = []
    for doc_name, text in docs.items():
        errors.extend(check_document(doc_name, text, inventory))
    errors.extend(check_start_sh(start_sh, inventory))
    # 同一文档多处出现相同口径错误时只报告一次，避免重复刷屏。
    errors = list(dict.fromkeys(errors))

    voicedesign = ", ".join(e.key for e in inventory.voicedesign_services) or "无"
    print(
        f"音频模型服务：{len(inventory.audio_services)}（audio.cpp {len(inventory.audio_cpp_services)}、"
        f"独立 runtime：{voicedesign}）"
    )
    print(f"固定 HTTP 服务：{len(inventory.fixed_services)}")
    print(f"start.sh 管理的 HTTP 进程总数：{inventory.total_http_processes}")
    if errors:
        print(f"服务口径不一致 {len(errors)} 处：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("服务清单与 README、AGENTS、start.sh 口径一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
