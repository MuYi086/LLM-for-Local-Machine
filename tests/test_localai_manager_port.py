"""验证 LocalAI 重启时的 loopback 端口检查允许复用 TIME_WAIT。"""

from __future__ import annotations

import socket
import unittest
from unittest.mock import MagicMock, call, patch

from scripts.localai.manager import LocalAIManager


class LocalAIManagerPortTests(unittest.TestCase):
    def test_port_probe_sets_reuseaddr_before_binding(self) -> None:
        probe = MagicMock()
        with patch("scripts.localai.manager.socket.socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value = probe

            LocalAIManager._ensure_port_available(8400)

        self.assertEqual(
            probe.mock_calls,
            [
                call.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
                call.bind(("127.0.0.1", 8400)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
