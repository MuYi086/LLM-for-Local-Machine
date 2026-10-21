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
            "UNITALE_QWEN3_ASR_PROVIDER",
            "unitale.qwen3-asr.1p7b",
            float(os.getenv("QWEN3_ASR_REQUEST_TIMEOUT", "900")),
            "asr",
        ),
        (
            "UNITALE_QWEN3_TTS_PROVIDER",
            "unitale.qwen3-tts.1p7b-base",
            float(os.getenv("QWEN3_TTS_REQUEST_TIMEOUT", "600")),
            "tts-base",
        ),
        (
            "UNITALE_QWEN3_VOICEDESIGN_PROVIDER",
            "unitale.qwen3-tts.1p7b-voicedesign",
            float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900")),
            "voice-design",
        ),
        (
            "UNITALE_DOTS_TTS_SOAR_PROVIDER",
            "unitale.dots-tts-soar.q8-0",
            float(os.getenv("DOTS_TTS_SOAR_REQUEST_TIMEOUT", "900")),
            "dots-tts-soar",
        ),
        (
            "UNITALE_VOXCPM2_PROVIDER",
            "unitale.voxcpm2.q8-0",
            float(os.getenv("VOXCPM2_REQUEST_TIMEOUT", "600")),
            "voxcpm2",
        ),
        (
            "UNITALE_CONFUCIUS4_TTS_PROVIDER",
            "unitale.confucius4-tts.orig",
            float(os.getenv("CONFUCIUS4_TTS_REQUEST_TIMEOUT", "900")),
            "confucius4-tts",
        ),
        (
            "UNITALE_MOSS_VOICEGENERATOR_PROVIDER",
            "unitale.moss-voicegenerator.bf16",
            float(os.getenv("MOSS_VOICEGENERATOR_REQUEST_TIMEOUT", "900")),
            "moss-voicegenerator",
        ),
        (
            "UNITALE_FIRERED_TTS3_CLONE_PROVIDER",
            "unitale.fireredtts3.base.q8-0",
            float(os.getenv("FIRERED_TTS3_REQUEST_TIMEOUT", "900")),
            "fireredtts3-base",
        ),
        (
            "UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER",
            "unitale.fireredtts3.instruct.q8-0",
            float(os.getenv("FIRERED_TTS3_REQUEST_TIMEOUT", "900")),
            "fireredtts3-instruct",
        ),
        (
            "UNITALE_ACESTEP_PROVIDER",
            "unitale.ace-step15.xl-turbo.q8dit",
            float(os.getenv("ACESTEP_REQUEST_TIMEOUT", "1800")),
            "ace-step",
        ),
        (
            "UNITALE_STABLE_AUDIO_3_MEDIUM_PROVIDER",
            "unitale.stable-audio3-medium.q8-0",
            float(os.getenv("STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT", "900")),
            "stable-audio3-medium",
        ),
        (
            "UNITALE_SHEETSAGE2_PROVIDER",
            "unitale.sheetsage2.worker.v1",
            float(os.getenv("SHEETSAGE2_REQUEST_TIMEOUT", "1800")),
            "sheetsage2-worker",
        ),
        (
            "UNITALE_MERT_V2_FULLSONG_PROVIDER",
            "unitale.mert-v2-fullsong.worker.v1",
            float(os.getenv("MERT_V2_FULLSONG_REQUEST_TIMEOUT", "900")),
            "mert-v2-fullsong-worker",
        ),
        (
            "UNITALE_YUE2_VAE_PROVIDER",
            "unitale.yue2-vae.worker.v1",
            float(os.getenv("YUE2_VAE_REQUEST_TIMEOUT", "900")),
            "yue2-vae-worker",
        ),
        (
            "UNITALE_SOULX_FLASHHEAD_PROVIDER",
            "unitale.soulx-flashhead.worker.v1",
            float(os.getenv("SOULX_FLASHHEAD_REQUEST_TIMEOUT", "1800")),
            "soulx-flashhead-worker",
        ),
        (
            "UNITALE_DITTO_PROVIDER",
            "unitale.ditto.worker.v1",
            float(os.getenv("DITTO_REQUEST_TIMEOUT", "1800")),
            "ditto-worker",
        ),
        (
            "UNITALE_YUE2_3B_PROVIDER",
            "unitale.yue2-3b.worker.v1",
            float(os.getenv("YUE2_3B_REQUEST_TIMEOUT", "1800")),
            "yue2-3b-worker",
        ),
        (
            "UNITALE_MOSS_SOUNDEFFECT_PROVIDER",
            "unitale.moss-soundeffect-v2.worker.v1",
            float(os.getenv("MOSS_SOUNDEFFECT_REQUEST_TIMEOUT", "600")),
            "moss-soundeffect-v2-worker",
        ),
        (
            "UNITALE_STEP_AUDIO_EDITX_PROVIDER",
            "unitale.step-audio-editx.worker.v1",
            float(os.getenv("STEP_AUDIO_EDITX_REQUEST_TIMEOUT", "900")),
            "step-audio-editx-worker",
        ),
        (
            f"UNITALE_MOSS_AUDIO_4B_{os.getenv('MOSS_AUDIO_4B_VARIANT', 'thinking').upper()}_PROVIDER",
            f"unitale.moss-audio-4b.{os.getenv('MOSS_AUDIO_4B_VARIANT', 'thinking').lower()}.worker.v1",
            float(os.getenv("MOSS_AUDIO_4B_REQUEST_TIMEOUT", "900")),
            f"moss-audio-4b-{os.getenv('MOSS_AUDIO_4B_VARIANT', 'thinking').lower()}-worker",
        ),
        (
            "UNITALE_TIGER_DNR_PROVIDER",
            "unitale.tiger-dnr.worker.v1",
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
        "soulx-flashhead-worker",
        "ditto-worker",
        "yue2-3b-worker",
        "moss-soundeffect-v2-worker",
        "step-audio-editx-worker",
        "moss-audio-4b-thinking-worker",
        "moss-audio-4b-instruct-worker",
        "tiger-dnr-worker",
    }
    if any(task in worker_tasks for _, _, _, task in selected):
        try:
            manager.verify_worker_bridge_candidates()
            verify_worker_bridge_listener(
                os.getenv("UNITALE_LOCALAI_BRIDGE_ADDRESS", "127.0.0.1:50052")
            )
        except LocalAIManagerError as exc:
            print(f"LocalAI worker bridge 预检失败：{exc}", file=sys.stderr)
            return 1
    for service_env, default_model, timeout, task in selected:
        try:
            if task == "asr":
                manager.verify_qwen3_asr_candidate()
            elif task == "tts-base":
                manager.verify_qwen3_tts_candidate()
            elif task == "dots-tts-soar":
                manager.verify_dots_tts_soar_candidate()
            elif task == "voxcpm2":
                manager.verify_voxcpm2_candidate()
            elif task == "confucius4-tts":
                manager.verify_confucius4_tts_candidate()
            elif task == "moss-voicegenerator":
                manager.verify_moss_voicegenerator_candidate()
            elif task == "fireredtts3-base":
                manager.verify_fireredtts3_base_candidate()
            elif task == "fireredtts3-instruct":
                manager.verify_fireredtts3_instruct_candidate()
            elif task == "ace-step":
                manager.verify_ace_step_candidate()
            elif task == "stable-audio3-medium":
                manager.verify_stable_audio_candidate()
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
        raise LocalAIManagerError("UNITALE_LOCALAI_BRIDGE_ADDRESS 必须是有效回环地址。") from exc
    if not separator or not is_loopback or not 1 <= port <= 65535:
        raise LocalAIManagerError("UNITALE_LOCALAI_BRIDGE_ADDRESS 必须是有效回环地址。")
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise LocalAIManagerError(
            f"LocalAI worker bridge 未在 {host}:{port} 监听；检查 managed bridge 或外部 gRPC backend。"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
