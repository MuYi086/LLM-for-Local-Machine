"""读取并校验内部推理 provider 配置。"""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from .errors import GatewayConfigurationError


@dataclass(frozen=True)
class GatewayConfig:
    """单项服务的 provider、LocalAI 地址和有界超时。"""

    provider: str
    localai_enabled: bool
    base_url: str
    model: str
    api_key: str | None
    connect_timeout: float
    request_timeout: float
    cleanup_timeout: float
    gpu_lock_file: str
    gpu_lock_wait_timeout: float
    cuda_release_delay: float


def _positive_float(name: str, value: str) -> float:
    """读取有限正数环境值。"""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise GatewayConfigurationError(f"{name} 必须是正数。") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise GatewayConfigurationError(f"{name} 必须是有限正数。")
    return parsed


def _nonnegative_float(name: str, value: str) -> float:
    """读取有限非负秒数，供 CUDA worker 退出缓冲使用。"""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise GatewayConfigurationError(f"{name} 必须是非负数。") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise GatewayConfigurationError(f"{name} 必须是有限非负数。")
    return parsed


def _is_loopback(host: str) -> bool:
    """只允许 LocalAI 绑定在本机回环地址。"""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_base_url(value: str) -> str:
    """拒绝明文外网地址、凭据、查询和 URL fragment。"""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or not _is_loopback(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GatewayConfigurationError(
            "LLM_LOCALAI_BASE_URL 只允许不带凭据的本机 HTTP 回环地址。"
        )
    try:
        _ = parsed.port
    except ValueError as exc:
        raise GatewayConfigurationError("LLM_LOCALAI_BASE_URL 端口无效。") from exc
    return value.rstrip("/")


def load_gateway_config(
    *,
    service_provider_env: str,
    default_model: str,
    request_timeout: float,
    environ: dict[str, str] | None = None,
) -> GatewayConfig:
    """加载服务 provider 配置，并在启动时拒绝不安全值。"""
    values = os.environ if environ is None else environ
    enabled = values.get("LLM_LOCALAI_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    provider = (
        values.get(
            service_provider_env,
            values.get("LLM_LOCAL_INFERENCE_PROVIDER", "native"),
        )
        .strip()
        .lower()
    )
    if provider not in {"native", "localai"}:
        raise GatewayConfigurationError(f"{service_provider_env} 只支持 native 或 localai。")
    if provider == "localai" and not enabled:
        raise GatewayConfigurationError(
            f"{service_provider_env}=localai 需要显式设置 LLM_LOCALAI_ENABLED=true。"
        )

    base_url = _validate_base_url(values.get("LLM_LOCALAI_BASE_URL", "http://127.0.0.1:8400"))
    model = values.get(f"{service_provider_env.removesuffix('_PROVIDER')}_MODEL", default_model)
    if not model or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in model
    ):
        raise GatewayConfigurationError("LocalAI model 名称只允许字母、数字、点、下划线和短横线。")

    return GatewayConfig(
        provider=provider,
        localai_enabled=enabled,
        base_url=base_url,
        model=model,
        api_key=values.get("LLM_LOCALAI_API_KEY") or None,
        connect_timeout=_positive_float(
            "LLM_LOCALAI_CONNECT_TIMEOUT",
            values.get("LLM_LOCALAI_CONNECT_TIMEOUT", "5"),
        ),
        request_timeout=_positive_float(
            "LocalAI 请求超时",
            str(request_timeout),
        ),
        cleanup_timeout=_positive_float(
            "LLM_LOCALAI_CLEANUP_TIMEOUT",
            values.get("LLM_LOCALAI_CLEANUP_TIMEOUT", "30"),
        ),
        gpu_lock_file=values.get("GPU_LOCK_FILE", "storage/.cache/runtime/gpu-runtime.lock"),
        gpu_lock_wait_timeout=_positive_float(
            "GPU_LOCK_WAIT_TIMEOUT",
            values.get("GPU_LOCK_WAIT_TIMEOUT", "900"),
        ),
        cuda_release_delay=_nonnegative_float(
            "CUDA_RELEASE_DELAY",
            values.get("CUDA_RELEASE_DELAY", "2"),
        ),
    )
