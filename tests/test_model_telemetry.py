"""验证性能观测不改变响应、不保存正文，并在关闭时不创建文件。"""

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_local_runtime.telemetry import TelemetryMiddleware, Trace


class TelemetryTests(unittest.TestCase):
    def test_split_sse_preserves_bytes_and_records_only_metrics(self):
        async def run(directory):
            chunks = [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"secret"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"hel',
                b'lo private"}}]}\n\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":3},"timings":{"predicted_per_second":12.5}}\n\n',
                b"data: [DONE]\n\n",
            ]
            sent = []

            async def app(scope, receive, send):
                scope["model_telemetry"].mark("lease_acquired")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")],
                    }
                )
                for chunk in chunks:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})

            async def send(message):
                sent.append(message)

            middleware = TelemetryMiddleware(app, service="test", paths={"/chat"})
            await middleware(
                {"type": "http", "path": "/chat", "method": "POST", "headers": []}, None, send
            )
            self.assertEqual(b"".join(m.get("body", b"") for m in sent), b"".join(chunks))
            record = json.loads(next(directory.glob("requests-*.jsonl")).read_text())
            self.assertEqual(record["usage"]["prompt_tokens"], 20)
            self.assertEqual(record["timings"]["predicted_per_second"], 12.5)
            self.assertIsNotNone(record["first_content_ms"])
            self.assertIn("lease_acquired", record["stages_ms"])
            self.assertNotIn("private", json.dumps(record))
            self.assertNotIn("secret", json.dumps(record))

        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                "os.environ", {"MODEL_TELEMETRY_ENABLED": "true", "MODEL_TELEMETRY_DIR": root}
            ),
        ):
            asyncio.run(run(Path(root)))

    def test_invalid_numeric_extensions_do_not_break_stream(self):
        trace = Trace("test")
        trace.feed(b'data: {"timings":{"prompt_ms":' + b"9" * 1000 + b"}}\n\n")
        self.assertEqual(trace.timings, {})

    def test_disabled_does_not_touch_storage(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                "os.environ",
                {"MODEL_TELEMETRY_ENABLED": "false", "MODEL_TELEMETRY_DIR": root + "/missing"},
            ),
        ):
            trace = Trace("test")
            trace.finish()
            self.assertFalse(Path(root, "missing").exists())

    def test_tool_only_is_not_content_and_oversized_event_is_bounded(self):
        with patch.dict("os.environ", {"MODEL_TELEMETRY_ENABLED": "false"}):
            trace = Trace("test")
            trace.feed(
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"secret"}}]}}]}\n\n'
            )
            self.assertIsNone(trace.first_content_ms)
            self.assertIsNotNone(trace.first_tool_ms)
            trace.feed(b"x" * 300000)
            self.assertLessEqual(len(trace.buffer), 65536)

    def test_disk_failure_does_not_raise(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                "os.environ",
                {"MODEL_TELEMETRY_ENABLED": "true", "MODEL_TELEMETRY_DIR": root + "/file/child"},
            ),
        ):
            Path(root, "file").touch()
            Trace("test").finish()

    def test_cancellation_records_incomplete_request(self):
        async def run():
            async def app(scope, receive, send):
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                await TelemetryMiddleware(app, service="test", paths={"/chat"})(
                    {"type": "http", "path": "/chat", "headers": []}, None, None
                )

        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(
                "os.environ", {"MODEL_TELEMETRY_ENABLED": "true", "MODEL_TELEMETRY_DIR": root}
            ),
        ):
            asyncio.run(run())
            record = json.loads(next(Path(root).glob("requests-*.jsonl")).read_text())
            self.assertEqual(record["outcome"], "cancelled")
            self.assertIsNone(record["first_content_ms"])


class ResourceAndReportTests(unittest.TestCase):
    def test_memory_snapshot_uses_available_and_keeps_swap_separate(self):
        from llm_local_runtime.telemetry_resources import read_memory

        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "meminfo")
            path.write_text(
                "MemTotal: 8192 kB\nMemAvailable: 3072 kB\nSwapTotal: 4096 kB\nSwapFree: 1024 kB\n"
            )
            self.assertEqual(
                read_memory(path),
                {"total_mib": 8.0, "used_mib": 5.0, "available_mib": 3.0, "swap_used_mib": 3.0},
            )
            self.assertEqual(read_memory(Path(root, "missing")), {})

    def test_native_log_parser_does_not_save_other_log_text(self):
        from llm_local_runtime.telemetry_resources import parse_backend_line

        row = parse_backend_line(
            "1.42.282.758 I slot print_timing: id 0 | task 538 | prompt eval time = 15390.16 ms / 10331 tokens ( 1.49 ms per token, 671.27 tokens per second)"
        )
        self.assertEqual(row["task_id"], 538)
        self.assertEqual(row["phase"], "prompt")
        self.assertEqual(row["tokens"], 10331)
        self.assertEqual(row["tokens_per_second"], 671.27)
        self.assertIsNone(parse_backend_line("request body private password"))

    def test_report_filters_time_and_keeps_services_separate(self):
        from llm_local_runtime.telemetry_cli import build_report

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            rows = [
                {
                    "started_at": "2026-11-12T02:00:01+00:00",
                    "ended_at": "2026-11-12T02:00:04+00:00",
                    "service": "bridge",
                    "request_id": "a",
                    "duration_ms": 3000,
                    "first_content_ms": 1000,
                    "usage": {"completion_tokens": 10},
                },
                {
                    "started_at": "2026-11-12T02:00:01+00:00",
                    "ended_at": "2026-11-12T02:00:05+00:00",
                    "service": "ornith",
                    "request_id": "a",
                    "duration_ms": 4000,
                    "usage": {"completion_tokens": 10},
                },
                {
                    "started_at": "2026-11-12T03:00:00+00:00",
                    "service": "bridge",
                    "request_id": "b",
                    "usage": {"completion_tokens": 999},
                },
            ]
            (directory / "requests-test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n{partial"
            )
            report = build_report(
                directory, "2026-11-12T02:00:00+00:00", "2026-11-12T02:01:00+00:00"
            )
            self.assertEqual(len(report["requests"]), 2)
            self.assertEqual(report["services"]["bridge"]["completion_tokens"], 10)
            self.assertEqual(report["services"]["ornith"]["completion_tokens"], 10)
            self.assertTrue(report["warnings"])

    def test_begin_end_never_overwrite_existing_test(self):
        from llm_local_runtime.telemetry_cli import begin_test, end_test

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            test_id = begin_test(directory, "32k")
            end_test(directory, test_id)
            with self.assertRaises(FileExistsError):
                end_test(directory, test_id)

    def test_resource_summary_does_not_invent_pretest_baseline(self):
        from llm_local_runtime.telemetry_cli import build_report

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "resources.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-11-12T02:00:01+00:00",
                        "memory": {"used_mib": 100},
                        "gpu": {},
                        "processes": [],
                    }
                )
                + "\n"
            )
            result = build_report(
                directory, "2026-11-12T02:00:00+00:00", "2026-11-12T02:00:02+00:00"
            )
            self.assertIsNone(result["resources"]["memory_mib"]["used_mib"]["baseline"])
            self.assertIsNone(result["resources"]["memory_mib"]["used_mib"]["peak_delta"])

    def test_unfinished_call_is_reported_instead_of_disappearing(self):
        from llm_local_runtime.telemetry_cli import build_report

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "starts-test.jsonl").write_text(
                json.dumps(
                    {
                        "started_at": "2026-11-12T02:00:01+00:00",
                        "service": "ornith",
                        "request_id": "request",
                        "trace_id": "trace",
                    }
                )
                + "\n"
            )
            result = build_report(
                directory, "2026-11-12T02:00:00+00:00", "2026-11-12T02:01:00+00:00"
            )
            self.assertEqual(result["requests"][0]["outcome"], "unfinished")
            self.assertIsNone(result["services"]["ornith"]["completion_tokens"])
            self.assertTrue(any("结束记录" in text for text in result["warnings"]))

    def test_pretest_baseline_is_not_counted_as_task_peak(self):
        from llm_local_runtime.telemetry_cli import build_report

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            rows = [
                {"timestamp": "2026-11-12T02:00:00+00:00", "memory": {"used_mib": 900}},
                {"timestamp": "2026-11-12T02:00:02+00:00", "memory": {"used_mib": 100}},
            ]
            (directory / "resources.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            result = build_report(
                directory, "2026-11-12T02:00:01+00:00", "2026-11-12T02:00:03+00:00"
            )
            stats = result["resources"]["memory_mib"]["used_mib"]
            self.assertEqual(stats["baseline"], 900)
            self.assertEqual(stats["sampled_peak"], 100)
            self.assertEqual(stats["peak_delta"], -800)
            self.assertEqual(stats["valid_samples"], 1)

    def test_cli_begin_end_creates_readable_reports_without_model_calls(self):
        script = Path(__file__).resolve().parents[1] / "scripts/model_telemetry.py"
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, str(script), "--directory", root]
            begin = subprocess.run(
                [*command, "begin", "--label", "32k"], check=True, capture_output=True, text=True
            )
            test_id = begin.stdout.split("测试 ID：", 1)[1].strip()
            subprocess.run(
                [*command, "end", test_id, "--settle-seconds", "0"],
                check=True,
                capture_output=True,
                text=True,
            )
            markdown = next(Path(root).glob("report-*.md")).read_text()
            report = json.loads(next(Path(root).glob("report-*.json")).read_text())
            self.assertIn("缺失与口径", markdown)
            self.assertEqual(report["requests"], [])
            self.assertTrue(report["warnings"])

    def test_backend_pidfile_tracks_process_outside_startup_tree(self):
        import shutil

        from llm_local_runtime.telemetry_resources import read_processes

        executable = shutil.which("sleep")
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root, "backend.pid")
            process = subprocess.Popen([executable, "30"])
            try:
                pidfile.write_text(str(process.pid))
                with patch.dict(
                    "os.environ",
                    {"ORNITH_BACKEND_PID_FILE": str(pidfile), "LLAMA_SERVER_BIN": executable},
                ):
                    rows = read_processes(2147483647, pss=False)
                row = next((row for row in rows if row["pid"] == process.pid), None)
                self.assertIsNotNone(row)
                self.assertEqual(row["role"], "ornith_backend")
                self.assertGreaterEqual(row["rss_mib"], 0)
            finally:
                process.terminate()
                process.wait(timeout=5)


class ReportAttachmentTests(unittest.TestCase):
    def test_attach_replaces_only_matching_test_block_and_preserves_review(self):
        from llm_local_runtime.telemetry_cli import attach_result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            test_id = "a" * 32
            target = directory / "result.md"
            target.write_text(
                "# 本轮审查\n<!-- MODEL_TELEMETRY_BEGIN test_id="
                + test_id
                + " -->\n旧快照\n<!-- MODEL_TELEMETRY_END -->\n## 审查正文\n不能改变的结论\n"
            )
            result = {
                "test_id": test_id,
                "test_state": "ended",
                "window_start": "2026-11-12T02:00:00+00:00",
                "window_end": "2026-11-12T02:01:00+00:00",
                "window_seconds": 60,
                "services": {},
                "requests": [],
                "resources": {"memory_mib": {}, "gpu_used_mib": {}, "processes": []},
                "backend_unassigned": [],
                "warnings": [],
                "limitations": [],
            }
            attach_result(target, result, directory / "raw.json")
            text = target.read_text()
            self.assertIn("不能改变的结论", text)
            self.assertNotIn("旧快照", text)
            self.assertIn("结束窗口", text)
            self.assertIn(test_id, text)
            self.assertEqual(text.count("MODEL_TELEMETRY_BEGIN"), 1)
            self.assertTrue(text.endswith("## 审查正文\n不能改变的结论\n"))

    def test_wrong_id_missing_or_duplicate_marker_and_symlink_are_rejected(self):
        from llm_local_runtime.telemetry_cli import attach_result

        with tempfile.TemporaryDirectory() as root:
            target = Path(root, "result.md")
            valid = (
                "<!-- MODEL_TELEMETRY_BEGIN test_id="
                + "a" * 32
                + " -->\nx\n<!-- MODEL_TELEMETRY_END -->"
            )
            for contents in ("no marker", valid.replace("a" * 32, "b" * 32), valid + valid):
                target.write_text(contents)
                with self.assertRaises(ValueError):
                    attach_result(target, {"test_id": "a" * 32}, Path(root, "raw.json"))
                self.assertEqual(target.read_text(), contents)
            link = Path(root, "link.md")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                attach_result(link, {"test_id": "a" * 32}, Path(root, "raw.json"))

    def test_cli_report_then_end_updates_same_result_with_own_window(self):
        script = Path(__file__).resolve().parents[1] / "scripts/model_telemetry.py"
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            command = [sys.executable, str(script), "--directory", root]
            output = subprocess.run([*command, "begin"], check=True, capture_output=True, text=True)
            test_id = output.stdout.split("测试 ID：", 1)[1].strip()
            target = directory / "result.md"
            target.write_text(
                "# 审查\n<!-- MODEL_TELEMETRY_BEGIN test_id="
                + test_id
                + " -->\n待采集\n<!-- MODEL_TELEMETRY_END -->\n完整正文\n"
            )
            subprocess.run(
                [*command, "report", "--test-id", test_id, "--attach-result", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("阶段快照", target.read_text())
            subprocess.run(
                [*command, "end", test_id, "--settle-seconds", "0", "--attach-result", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            text = target.read_text()
            self.assertIn("结束窗口", text)
            self.assertNotIn("阶段快照", text)
            self.assertTrue(text.endswith("完整正文\n"))

    def test_atomic_replace_failure_keeps_original_report_and_removes_temp(self):
        from llm_local_runtime.telemetry_cli import attach_result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            target = directory / "result.md"
            original = (
                "<!-- MODEL_TELEMETRY_BEGIN test_id="
                + "a" * 32
                + " -->\nold\n<!-- MODEL_TELEMETRY_END -->"
            )
            target.write_text(original)
            report = {
                "test_id": "a" * 32,
                "test_state": "open",
                "window_start": "2026-11-12T02:00:00+00:00",
                "window_end": "2026-11-12T02:01:00+00:00",
                "window_seconds": 60,
                "services": {},
                "requests": [],
                "resources": {"memory_mib": {}, "gpu_used_mib": {}, "processes": []},
                "backend_unassigned": [],
                "warnings": [],
                "limitations": [],
            }
            with patch(
                "llm_local_runtime.telemetry_cli.os.replace", side_effect=OSError("disk failure")
            ):
                with self.assertRaises(OSError):
                    attach_result(target, report, directory / "raw.json")
            self.assertEqual(target.read_text(), original)
            self.assertEqual(list(directory.glob(".telemetry-*.tmp")), [])

    def test_attach_cli_rejects_implicit_latest_directory(self):
        script = Path(__file__).resolve().parents[1] / "scripts/model_telemetry.py"
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, "result.md")
            target.write_text("unchanged")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "report",
                    "--test-id",
                    "a" * 32,
                    "--attach-result",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--directory", result.stderr)
            self.assertEqual(target.read_text(), "unchanged")

    def test_report_does_not_label_earlier_snapshot_ended_if_end_appears_later(self):
        import io
        from contextlib import redirect_stdout

        from llm_local_runtime import telemetry_cli

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            test_id = telemetry_cli.begin_test(directory, "32k")
            original_build = telemetry_cli.build_report

            def build_then_end(*args):
                result = original_build(*args)
                telemetry_cli.end_test(directory, test_id)
                return result

            with (
                patch.object(
                    sys, "argv", ["telemetry", "--directory", root, "report", "--test-id", test_id]
                ),
                patch.object(telemetry_cli, "build_report", side_effect=build_then_end),
                redirect_stdout(io.StringIO()),
            ):
                telemetry_cli.main()
            result = json.loads(next(directory.glob("report-*.json")).read_text())
            self.assertEqual(result["test_state"], "open")

    def test_attachment_preserves_crlf_outside_the_statistics_block(self):
        from llm_local_runtime.telemetry_cli import attach_result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            target = directory / "result.md"
            prefix = b"# Review\r\n<!-- MODEL_TELEMETRY_BEGIN test_id=" + b"a" * 32 + b" -->"
            suffix = b"<!-- MODEL_TELEMETRY_END -->\r\nKeep this review.\r\n"
            target.write_bytes(prefix + b"\r\nold\r\n" + suffix)
            report = {
                "test_id": "a" * 32,
                "test_state": "open",
                "window_start": "2026-11-12T02:00:00+00:00",
                "window_end": "2026-11-12T02:01:00+00:00",
                "window_seconds": 60,
                "services": {},
                "requests": [],
                "resources": {"memory_mib": {}, "gpu_used_mib": {}, "processes": []},
                "backend_unassigned": [],
                "warnings": [],
                "limitations": [],
            }
            attach_result(target, report, directory / "raw.json")
            self.assertTrue(target.read_bytes().startswith(prefix))
            self.assertTrue(target.read_bytes().endswith(suffix))
