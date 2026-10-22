"""bridge 收尾失败后不得仅凭 LocalAI HTTP 200 解除 GPU 隔离。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from inference_gateway import load_gateway_config
from scripts.localai import recover_quarantine
from scripts.localai.manager import LocalAIManagerError


class QuarantineRecoveryTests(unittest.TestCase):
    def test_worker_failure_requires_host_confirmation_even_after_http_success(self) -> None:
        for confirmed in (False, True):
            with self.subTest(confirmed=confirmed), tempfile.TemporaryDirectory() as temporary:
                lock_path = Path(temporary) / "gpu.lock"
                marker = Path(f"{lock_path}.quarantine")
                marker.write_text(
                    json.dumps(
                        {
                            "model": "llm-local.localai.worker-bridge",
                            "reason": "worker_process_exit_unconfirmed",
                        }
                    )
                )
                environment = {
                    "LLM_LOCALAI_ENABLED": "true",
                    "LLM_LOCALAI_MANAGED": "true",
                    "GPU_LOCK_FILE": str(lock_path),
                }
                manager = Mock()
                if not confirmed:
                    manager.verify_managed_localai_stopped.side_effect = LocalAIManagerError(
                        "worker alive"
                    )
                with (
                    patch.dict("os.environ", environment),
                    patch.object(recover_quarantine, "LocalAIClient") as client,
                    patch.object(recover_quarantine, "LocalAIManager", return_value=manager),
                ):
                    # 载入真实配置，确保恢复使用实际 GPU 锁路径。
                    config = load_gateway_config(
                        service_provider_env="LLM_LOCAL_QWEN3_TTS_PROVIDER",
                        default_model="llm-local.qwen3-tts.1p7b-base",
                        request_timeout=5,
                    )
                    with patch.object(
                        recover_quarantine, "load_gateway_config", return_value=config
                    ):
                        result = recover_quarantine.main()
                client.return_value.shutdown_model.assert_called_once()
                manager.stop.assert_called_once()
                manager.verify_managed_localai_stopped.assert_called_once()
                self.assertEqual(result, 0 if confirmed else 1)
                self.assertEqual(marker.exists(), not confirmed)
