"""启动前端口检查的无模型回归测试。"""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import io
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location(
    "startup_ports_for_test", Path(__file__).resolve().parents[1] / "main/check_ports.py"
)
assert SPEC and SPEC.loader
startup_ports = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = startup_ports
SPEC.loader.exec_module(startup_ports)
check_ports = startup_ports.check_ports
main = startup_ports.main


class StartupPortTests(unittest.TestCase):
    def test_available_ports_are_released_after_checking(self) -> None:
        listener = MagicMock()
        listener.__enter__.return_value = listener
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 8392))]
        with (
            patch("startup_ports_for_test.socket.getaddrinfo", return_value=addresses),
            patch("startup_ports_for_test.socket.socket", return_value=listener),
        ):
            self.assertEqual(check_ports([("Ditto", "0.0.0.0", 8392)]), [])
        listener.bind.assert_called_once_with(("0.0.0.0", 8392))
        listener.listen.assert_called_once()
        listener.__exit__.assert_called_once()

    def test_reports_every_conflict_and_releases_failed_probes(self) -> None:
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 8392))]
        with (
            patch("startup_ports_for_test.socket.getaddrinfo", return_value=addresses),
            patch("startup_ports_for_test.socket.socket", return_value=listener),
        ):
            errors = check_ports([("Ditto", "0.0.0.0", 8392), ("MOSS", "0.0.0.0", 8341)])
        self.assertEqual(len(errors), 2)
        self.assertIn("Ditto (0.0.0.0:8392)", errors[0])
        self.assertIn("MOSS (0.0.0.0:8341)", errors[1])
        self.assertEqual(listener.__exit__.call_count, 2)

    def test_ipv6_probe_uses_the_same_separate_address_space_as_uvicorn(self) -> None:
        listener = MagicMock()
        listener.__enter__.return_value = listener
        addresses = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 8392, 0, 0))]
        with (
            patch("startup_ports_for_test.socket.getaddrinfo", return_value=addresses),
            patch("startup_ports_for_test.socket.socket", return_value=listener),
        ):
            self.assertEqual(check_ports([("Ditto", "::", 8392)]), [])
        listener.setsockopt.assert_any_call(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)

    def test_invalid_ports_fail_before_any_socket_is_opened(self) -> None:
        with patch("startup_ports_for_test.socket.socket") as create_socket:
            for port in (0, -1, 65536):
                with self.subTest(port=port):
                    self.assertTrue(check_ports([("Ditto", "0.0.0.0", port)]))
            create_socket.assert_not_called()

    def test_cli_conflict_exits_with_service_details(self) -> None:
        output = io.StringIO()
        with (
            patch(
                "startup_ports_for_test.check_ports",
                return_value=["Ditto (0.0.0.0:8392): occupied"],
            ),
            contextlib.redirect_stderr(output),
        ):
            self.assertEqual(main(["Ditto", "0.0.0.0", "8392"]), 1)
        self.assertIn("Ditto (0.0.0.0:8392)", output.getvalue())
        self.assertIn("Ctrl+C", output.getvalue())

    def test_cli_rejects_malformed_port_configuration(self) -> None:
        with (
            contextlib.redirect_stderr(io.StringIO()),
            patch("startup_ports_for_test.check_ports") as probe,
        ):
            self.assertEqual(main(["Ditto", "0.0.0.0", "invalid"]), 2)
            self.assertEqual(main(["Ditto", "0.0.0.0"]), 2)
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
