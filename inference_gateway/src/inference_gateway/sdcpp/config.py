"""sd.cpp 外部二进制与共享 GPU 配置。"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    """拒绝非有限数值以及不符合边界的时限或资源值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} 必须是有限数值。")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} 必须是有限{'非负' if allow_zero else '正'}数。")
    return number


def local_path(value: str, base: Path | None = None) -> Path:
    """展开操作者配置的路径，拒绝未解析环境变量。"""
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("路径必须为非空字符串。")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"路径包含未定义的环境变量：{value}")
    path = Path(expanded)
    return ((base / path) if base is not None and not path.is_absolute() else path).resolve()


@dataclass(frozen=True)
class SdcppConfig:
    """外部 CLI 的身份、运行目录和独立超时设置。"""

    binary: Path
    binary_sha256: str
    lock_path: Path
    output_root: Path
    queue_timeout: float = 900
    run_timeout: float = 3600
    cleanup_timeout: float = 30
    release_delay: float = 2

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> SdcppConfig:
        """读取本机环境，要求显式指定与其他服务相同的 GPU 锁。"""
        values = os.environ if environment is None else environment
        digest = values.get("SDCPP_BIN_SHA256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("SDCPP_BIN_SHA256 必须是二进制 SHA-256。")
        runtime = values.get("RUNTIME_CACHE_DIR", "storage/.cache/runtime")
        return cls(
            binary=local_path(values.get("SDCPP_BIN", "")),
            binary_sha256=digest,
            lock_path=local_path(values.get("GPU_LOCK_FILE", "")),
            output_root=local_path(values.get("SDCPP_OUTPUT_DIR", str(Path(runtime) / "sdcpp"))),
            queue_timeout=positive_number(
                values.get("GPU_LOCK_WAIT_TIMEOUT", "900"), "GPU_LOCK_WAIT_TIMEOUT"
            ),
            run_timeout=positive_number(
                values.get("SDCPP_RUN_TIMEOUT", "3600"), "SDCPP_RUN_TIMEOUT"
            ),
            cleanup_timeout=positive_number(
                values.get("SDCPP_CLEANUP_TIMEOUT", "30"), "SDCPP_CLEANUP_TIMEOUT"
            ),
            release_delay=positive_number(
                values.get("CUDA_RELEASE_DELAY", "2"), "CUDA_RELEASE_DELAY", allow_zero=True
            ),
        )

    def validate_binary(self) -> None:
        """在锁外校验可执行文件摘要，不运行 CLI 或初始化 CUDA。"""
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ValueError("SDCPP_BIN 不是可执行文件。")
        with self.binary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != self.binary_sha256:
            raise ValueError("SDCPP_BIN 的 SHA-256 与配置不匹配。")
