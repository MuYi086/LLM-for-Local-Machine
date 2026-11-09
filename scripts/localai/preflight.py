#!/usr/bin/env python3
"""校验固定 LocalAI/audio.cpp 安装、九个 GGUF 候选和服务 readiness。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference_gateway/src"))

from inference_gateway import (  # noqa: E402
    GatewayConfigurationError,
    LocalAIClient,
    LocalAIRequestError,
    LocalAIUnavailableError,
    load_gateway_config,
)
from scripts.localai.manager import LocalAIManager, LocalAIManagerError  # noqa: E402

MODEL_MANIFEST_PATH = ROOT / "localai/manifests/audio-cpp-models.json"


def main() -> int:
    """确认受管 LocalAI 与全部清单模型已就绪，不加载任何模型。"""
    if os.getenv("LLM_LOCALAI_ENABLED", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print("LocalAI 未启用，跳过 audio.cpp 候选预检。")
        return 0
    if os.getenv("LLM_LOCAL_AUDIO_CPP_PROVIDER", "native").strip().lower() != "localai":
        print("audio.cpp 服务固定使用 LocalAI provider。", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        candidates = [
            candidate
            for candidate in manifest["models"]
            if candidate["service"]["runtime"] in {"voicedesign", "audio_cpp_runtime"}
        ]
        if len(candidates) != 9:
            raise LocalAIManagerError(
                f"期望九个 audio.cpp GGUF 候选，清单含 {len(candidates)} 个。"
            )

        manager = LocalAIManager()
        manager.verify_audio_cpp_candidates()
        environment = dict(os.environ)
        environment.pop("LLM_LOCAL_AUDIO_CPP_MODEL", None)
        timeout = float(os.getenv("AUDIO_CPP_REQUEST_TIMEOUT", "900"))
        for index, candidate in enumerate(candidates):
            model_id = candidate["model_id"]
            config = load_gateway_config(
                service_provider_env="LLM_LOCAL_AUDIO_CPP_PROVIDER",
                default_model=model_id,
                request_timeout=timeout,
                environ=environment,
            )
            if config.model != model_id:
                raise LocalAIManagerError(f"LocalAI model ID 与候选清单不一致：{model_id}")
            client = LocalAIClient(config)
            if index == 0 and not client.health():
                raise LocalAIManagerError("LocalAI 未就绪；拒绝启动 audio.cpp 服务。")
            if not client.model_ready():
                raise LocalAIManagerError(f"LocalAI 未注册模型 {model_id}。")
    except (
        GatewayConfigurationError,
        LocalAIManagerError,
        LocalAIRequestError,
        LocalAIUnavailableError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"LocalAI/audio.cpp 候选预检失败：{exc}", file=sys.stderr)
        return 1

    print("LocalAI 已就绪，九个 audio.cpp 模型均已注册。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
