#!/usr/bin/env python3
"""确认 LocalAI backend 停止后解除共享 GPU 隔离。"""

from __future__ import annotations

import fcntl
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference_gateway/src"))
sys.path.insert(0, str(ROOT / "llm_local_runtime/src"))

from inference_gateway import LocalAIClient, load_gateway_config  # noqa: E402
from llm_local_runtime import gpu_quarantine_path  # noqa: E402
from scripts.localai.manager import LocalAIManager, LocalAIManagerError  # noqa: E402


def main() -> int:
    """持有 GPU 锁，停止隔离记录中的模型并原子清除标记。"""
    config = load_gateway_config(
        service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
        default_model="llm-local.qwen3-tts.1p7b-base",
        request_timeout=float(os.getenv("QWEN3_TTS_REQUEST_TIMEOUT", "600")),
    )
    if not config.localai_enabled:
        print("恢复前需设置 LLM_LOCALAI_ENABLED=true。", file=sys.stderr)
        return 2
    lock_path = Path(config.gpu_lock_file)
    marker_path = gpu_quarantine_path(lock_path)
    if not marker_path.is_file():
        print("没有待恢复的 GPU 隔离标记。")
        return 0

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU 仍有活动请求，暂不解除隔离。", file=sys.stderr)
            return 1
        try:
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"隔离标记无法读取：{exc}", file=sys.stderr)
                return 1
            model = marker.get("model") if isinstance(marker, dict) else None
            if not isinstance(model, str) or not model:
                print("隔离标记缺少有效 model，保留隔离状态。", file=sys.stderr)
                return 1
            client = LocalAIClient(replace(config, model=model))
            shutdown_error: Exception | None = None
            try:
                client.shutdown_model()
            except Exception as exc:
                shutdown_error = exc
            # 外部 bridge 的 Free 错误可能被固定 LocalAI 吞掉；曾经无法回收的
            # worker 必须由宿主机进程组检查确认，HTTP 200 不足以解除隔离。
            worker_cleanup_failed = marker.get("reason") == "worker_process_exit_unconfirmed"
            if shutdown_error is not None or worker_cleanup_failed:
                managed = os.getenv("LLM_LOCALAI_MANAGED", "false").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if not managed:
                    print(
                        f"LocalAI 未确认停止 {model}：{shutdown_error or 'worker 回收失败'}",
                        file=sys.stderr,
                    )
                    return 1
                try:
                    manager = LocalAIManager(
                        environ={**os.environ, "LLM_LOCALAI_WORKER_BRIDGE_ENABLED": "true"}
                        if worker_cleanup_failed
                        else None
                    )
                    manager.stop()
                    manager.verify_managed_localai_stopped()
                except LocalAIManagerError as verification_error:
                    print(
                        f"LocalAI backend 未确认停止 {model}：{shutdown_error or 'worker 回收失败'}；进程组退出复核失败："
                        f"{verification_error}",
                        file=sys.stderr,
                    )
                    return 1
                print("已停止受管 LocalAI 进程组并确认 worker 与 loopback 端口回收。")
            marker_path.unlink()
            directory_fd = os.open(marker_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            print(f"LocalAI backend 已确认停止，GPU 隔离已解除：{model}")
            return 0
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
