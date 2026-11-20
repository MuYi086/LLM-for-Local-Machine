#!/usr/bin/env python3
"""确认 GPU backend 停止后解除共享 GPU 隔离。"""

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

from inference_gateway.sdcpp.processes import recover_sdcpp  # noqa: E402

from inference_gateway import LocalAIClient, load_gateway_config  # noqa: E402
from llm_local_runtime import gpu_quarantine_path  # noqa: E402
from scripts.localai.manager import LocalAIManager, LocalAIManagerError  # noqa: E402


def _environment_lock_path() -> Path:
    """按启动脚本环境定位共享 GPU 锁文件。"""
    configured = os.getenv("GPU_LOCK_FILE")
    if configured:
        return Path(configured)
    runtime_cache = os.getenv("RUNTIME_CACHE_DIR")
    if runtime_cache:
        return Path(runtime_cache) / "gpu-runtime.lock"
    return Path("storage/.cache/runtime/gpu-runtime.lock")


def _sync_marker_directory(marker_path: Path) -> None:
    """同步隔离标记所在目录。"""
    descriptor = os.open(marker_path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_group_exists(pgid: int) -> bool:
    """检查 backend 进程组是否仍存在。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_command_line(pid: int) -> list[str] | None:
    """读取进程参数；进程已经退出时返回空值。"""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [argument.decode(errors="replace") for argument in raw.split(b"\0") if argument]


def _recover_localai(marker_path: Path, marker: dict[str, object]) -> int:
    """保留原有 LocalAI backend 恢复流程。"""
    config = load_gateway_config(
        service_provider_env="LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER",
        default_model="llm-local.qwen3-tts.1p7b-voicedesign",
        request_timeout=float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900")),
    )
    if not config.localai_enabled:
        print("恢复 LocalAI 前需设置 LLM_LOCALAI_ENABLED=true。", file=sys.stderr)
        return 2
    model = marker.get("model")
    if not isinstance(model, str) or not model:
        print("LocalAI 隔离标记缺少有效 model，保留隔离状态。", file=sys.stderr)
        return 1
    client = LocalAIClient(replace(config, model=model))
    shutdown_error: Exception | None = None
    try:
        client.shutdown_model()
    except Exception as exc:
        shutdown_error = exc
    if shutdown_error is not None:
        managed = os.getenv("LLM_LOCALAI_MANAGED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not managed:
            print(f"LocalAI 未确认停止 {model}：{shutdown_error}", file=sys.stderr)
            return 1
        try:
            manager = LocalAIManager()
            manager.stop()
            manager.verify_managed_localai_stopped()
        except LocalAIManagerError as verification_error:
            print(
                f"LocalAI backend 未确认停止 {model}：{shutdown_error}；进程组退出复核失败："
                f"{verification_error}",
                file=sys.stderr,
            )
            return 1
        print("已停止受管 LocalAI 进程组并确认 loopback API 端口关闭。")
    marker_path.unlink()
    _sync_marker_directory(marker_path)
    print(f"LocalAI backend 已确认停止，GPU 隔离已解除：{model}")
    return 0


def main() -> int:
    """持有共享 GPU 锁，按 marker provider 恢复 backend 并清除标记。"""
    lock_path = _environment_lock_path()
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
            # 必须在持锁后重读；等待锁期间旧请求可能已更新或删除标记。
            if not marker_path.exists():
                print("没有待恢复的 GPU 隔离标记。")
                return 0
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"隔离标记无法读取：{exc}", file=sys.stderr)
                return 1
            if not isinstance(marker, dict):
                print("隔离标记格式无效，保留隔离状态。", file=sys.stderr)
                return 1
            provider = marker.get("provider", "localai")
            if provider == "stable-diffusion.cpp":
                try:
                    recover_sdcpp(
                        marker_path, marker, float(os.getenv("SDCPP_CLEANUP_TIMEOUT", "30"))
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"sd.cpp 恢复失败，保留隔离状态：{exc}", file=sys.stderr)
                    return 1
                print("sd.cpp 进程组已确认停止，GPU 隔离已解除。")
                return 0
            if provider != "localai":
                print(f"隔离标记 provider 不受支持：{provider}。", file=sys.stderr)
                return 1
            return _recover_localai(marker_path, marker)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
