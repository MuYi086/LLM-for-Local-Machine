"""核验后端实际启用了跨休眠 RAM 缓存，避免配置被旧二进制静默忽略。"""

from __future__ import annotations

import json
import urllib.request


def require_sleep_cache(base_url: str) -> None:
    """要求已通过 loopback 校验的 backend 报告缓存补丁已启用。

    Args:
        base_url: 启动助手已验证的 loopback 后端根地址。

    Raises:
        RuntimeError: 无法确认支持或启用状态，调用者须停止本次 backend。
    """
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base_url.rstrip("/") + "/props", timeout=5) as response:
            props = json.loads(response.read()) if response.status == 200 else None
    except (OSError, ValueError) as exc:
        raise RuntimeError("无法核实 sleep prompt cache；本次 backend 不得进入就绪状态。") from exc
    if not isinstance(props, dict) or any(
        props.get(key) is not True
        for key in ("sleep_prompt_cache_supported", "sleep_prompt_cache_enabled")
    ):
        raise RuntimeError(
            "backend 未支持或未启用 sleep prompt cache；请部署缓存补丁，"
            "或设置 ORNITH_9B_SLEEP_CACHE=0 回退。"
        )
