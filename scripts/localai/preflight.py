#!/usr/bin/env python3
"""校验已显式启用的 LocalAI provider 和模型注册状态。"""

from __future__ import annotations

import ipaddress
import os
import socket
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


def main() -> int:
    """检查显式选择 LocalAI 的候选模型，不启动或加载模型。"""
    providers = (
        (
            "LLM_LOCAL_QWEN3_TTS_PROVIDER",
            "llm-local.qwen3-tts.1p7b-base",
            float(os.getenv("QWEN3_TTS_REQUEST_TIMEOUT", "600")),
            "tts-base",
        ),
        (
            "LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER",
            "llm-local.qwen3-tts.1p7b-voicedesign",
            float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900")),
            "voice-design",
        ),
        (
            "LLM_LOCAL_SHEETSAGE2_PROVIDER",
            "llm-local.sheetsage2.worker.v1",
            float(os.getenv("SHEETSAGE2_REQUEST_TIMEOUT", "1800")),
            "sheetsage2-worker",
        ),
        (
            "LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER",
            "llm-local.mert-v2-fullsong.worker.v1",
            float(os.getenv("MERT_V2_FULLSONG_REQUEST_TIMEOUT", "900")),
            "mert-v2-fullsong-worker",
        ),
        (
            "LLM_LOCAL_YUE2_VAE_PROVIDER",
            "llm-local.yue2-vae.worker.v1",
            float(os.getenv("YUE2_VAE_REQUEST_TIMEOUT", "900")),
            "yue2-vae-worker",
        ),
        (
            "LLM_LOCAL_YUE2_3B_PROVIDER",
            "llm-local.yue2-3b.worker.v1",
            float(os.getenv("YUE2_3B_REQUEST_TIMEOUT", "1800")),
            "yue2-3b-worker",
        ),
        (
            "LLM_LOCAL_TIGER_DNR_PROVIDER",
            "llm-local.tiger-dnr.worker.v1",
            float(os.getenv("TIGER_DNR_REQUEST_TIMEOUT", "900")),
            "tiger-dnr-worker",
        ),
    )
    selected = [
        entry for entry in providers if os.getenv(entry[0], "native").strip().lower() == "localai"
    ]
    if not selected:
        print("LocalAI provider 未启用，保持 native 启动路径。")
        return 0
    manager = LocalAIManager()
    worker_tasks = {
        "sheetsage2-worker",
        "mert-v2-fullsong-worker",
        "yue2-vae-worker",
        "yue2-3b-worker",
        "tiger-dnr-worker",
    }
    if any(task in worker_tasks for _, _, _, task in selected):
        try:
            manager.verify_worker_bridge_candidates()
            verify_worker_bridge_listener(
                os.getenv("LLM_LOCALAI_BRIDGE_ADDRESS", "127.0.0.1:50052")
            )
        except LocalAIManagerError as exc:
            print(f"LocalAI worker bridge 预检失败：{exc}", file=sys.stderr)
            return 1
    for service_env, default_model, timeout, task in selected:
        try:
            if task == "tts-base":
                manager.verify_qwen3_tts_candidate()
            elif task in worker_tasks:
                if (
                    os.getenv(service_env.removesuffix("_PROVIDER") + "_MODEL", default_model)
                    != default_model
                ):
                    raise LocalAIManagerError(
                        f"{task} 只能使用 manifest 固定模型 ID {default_model}。"
                    )
            else:
                manager.verify_qwen3_voicedesign_candidate()
            config = load_gateway_config(
                service_provider_env=service_env,
                default_model=default_model,
                request_timeout=timeout,
            )
            client = LocalAIClient(config)
            if not client.health():
                raise LocalAIManagerError(
                    f"LocalAI 未就绪；拒绝启动显式选择 LocalAI 的 {task.upper()} 服务。"
                )
            if not client.model_ready():
                raise LocalAIManagerError(f"LocalAI 未注册模型 {config.model}。")
        except (
            GatewayConfigurationError,
            LocalAIManagerError,
            LocalAIRequestError,
            LocalAIUnavailableError,
        ) as exc:
            print(f"LocalAI {task} 候选模型预检失败：{exc}", file=sys.stderr)
            return 1
        print(f"LocalAI 已就绪，{task} 模型已注册：{config.model}")
    return 0


def verify_worker_bridge_listener(address: str) -> None:
    """确认 worker bridge 在本机回环地址监听，避免暴露远程 GPU worker。"""
    host, separator, port_text = address.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        port = int(port_text)
        is_loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError as exc:
        raise LocalAIManagerError("LLM_LOCALAI_BRIDGE_ADDRESS 必须是有效回环地址。") from exc
    if not separator or not is_loopback or not 1 <= port <= 65535:
        raise LocalAIManagerError("LLM_LOCALAI_BRIDGE_ADDRESS 必须是有效回环地址。")
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise LocalAIManagerError(
            f"LocalAI worker bridge 未在 {host}:{port} 监听；检查 managed bridge 或外部 gRPC backend。"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
