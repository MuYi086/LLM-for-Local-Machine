#!/usr/bin/env python3
"""使用已安装的 LocalAI 和无模型 worker 验证大文件传输及收尾。"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference_gateway/src"))
sys.path.insert(0, str(ROOT / "llm_local_runtime/src"))

from inference_gateway.execution import execute_localai_worker_file_reference  # noqa: E402

from inference_gateway import GatewayConfig  # noqa: E402
from scripts.localai.manager import LocalAIManager  # noqa: E402


def free_port() -> int:
    """为临时的 loopback 服务选择可用端口。"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def main() -> int:
    """隔离配置与产物，实际经过 HTTP → Go gRPC → Python bridge。"""
    installed = LocalAIManager()
    with tempfile.TemporaryDirectory(prefix="localai-bridge-smoke-") as temporary:
        root = Path(temporary)
        api_port, bridge_port = free_port(), free_port()
        while bridge_port == api_port:
            bridge_port = free_port()
        artifacts = root / "artifacts"
        environment = {
            **os.environ,
            "LLM_LOCALAI_ENABLED": "true",
            "LLM_LOCALAI_WORKER_BRIDGE_ENABLED": "true",
            "LLM_LOCALAI_DATA_DIR": str(root / "localai"),
            "LLM_LOCALAI_MODELS_DIR": str(root / "models"),
            "LLM_LOCALAI_BINARY": str(installed.binary_path.resolve()),
            "LLM_LOCALAI_BACKENDS_DIR": str(installed.backends_dir.resolve()),
            "LLM_LOCALAI_BASE_URL": f"http://127.0.0.1:{api_port}",
            "LLM_LOCALAI_BRIDGE_ADDRESS": f"127.0.0.1:{bridge_port}",
            "LLM_LOCALAI_BRIDGE_WORKERS": str(
                ROOT / "qa/tests/fixtures/localai_bridge_smoke_workers.json"
            ),
            "LLM_LOCALAI_ARTIFACT_DIR": str(artifacts),
            "STORAGE_DIR": str(root / "storage"),
            "RUNTIME_CACHE_DIR": str(root / "cache"),
            "GPU_LOCK_FILE": str(root / "gpu.lock"),
            "HF_MIRROR_DIR": str(Path.home() / "hf-mirror"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(ROOT / "llm_local_runtime/src"),
                    str(ROOT / "inference_gateway/src"),
                    str(ROOT / "localai_worker_bridge/src"),
                ]
            ),
        }
        manager = LocalAIManager(environ=environment)
        manager.verify_install()
        manager.models_dir.mkdir(parents=True, exist_ok=True)
        model = "llm-local.bridge.smoke-file"
        (manager.models_dir / f"{model}.yaml").write_text(
            f"name: {model}\nbackend: llm-local-worker\nparameters:\n  model: {model}\n",
            encoding="utf-8",
        )
        config = GatewayConfig(
            provider="localai",
            localai_enabled=True,
            base_url=environment["LLM_LOCALAI_BASE_URL"],
            model=model,
            api_key=os.getenv("LLM_LOCALAI_API_KEY"),
            connect_timeout=5,
            request_timeout=60,
            cleanup_timeout=15,
            gpu_lock_file=environment["GPU_LOCK_FILE"],
            gpu_lock_wait_timeout=5,
            cuda_release_delay=0,
        )
        try:
            manager.start()
            for size_mib in (41, 129):
                path, _ = execute_localai_worker_file_reference(
                    config=config,
                    operation="run",
                    payload={"worker_input": {"size_bytes": size_mib * 1024 * 1024}},
                    output_dir=root / "received",
                    artifact_dir=artifacts,
                    suffix=".wav",
                    media_type="audio/wav",
                )
                try:
                    with wave.open(str(path), "rb") as audio:
                        assert (
                            audio.getnchannels(),
                            audio.getsampwidth(),
                            audio.getframerate(),
                        ) == (2, 3, 48_000)
                        expected_frames = size_mib * 1024 * 1024 // 6
                        assert audio.getnframes() == expected_frames
                        frames_read = 0
                        while chunk := audio.readframes(16_384):
                            assert not any(chunk), "静音产物传输后内容不一致"
                            frames_read += len(chunk) // 6
                        assert frames_read == expected_frames
                    assert not list(artifacts.iterdir()), "共享 artifact 未消费"
                    assert not Path(f"{config.gpu_lock_file}.quarantine").exists()
                    print(
                        json.dumps(
                            {
                                "size_mib": size_mib,
                                "received_bytes": path.stat().st_size,
                                "verified": True,
                            }
                        )
                    )
                finally:
                    path.unlink(missing_ok=True)
        except Exception:
            log = manager.logs_dir / "localai.log"
            if log.exists():
                print(log.read_text(errors="replace")[-8000:], file=sys.stderr)
            raise
        finally:
            manager.stop()
            manager.verify_managed_localai_stopped()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
