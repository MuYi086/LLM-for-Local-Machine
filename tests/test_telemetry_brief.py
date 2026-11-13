"""验证收尾简报不会把大型遥测记录送回模型上下文。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llm_local_runtime.telemetry_brief import MAX_BRIEF_BYTES, render_brief

ROOT = Path(__file__).resolve().parents[1]


class TelemetryBriefTests(unittest.TestCase):
    def test_large_resources_and_raw_records_do_not_escape_whitelist(self):
        report = {
            "test_id": "a" * 32,
            "test_state": "ended",
            "window_seconds": 12.3,
            "services": {"ornith_9b": {"requests": 3, "completion_tokens": 50}},
            "resources": {
                "sample_count": 8,
                "processes": [
                    {
                        "role": "service_tree",
                        "pid": i,
                        "secret": "RAW_RECORD",
                        "rss_mib": {"sampled_peak": 999999},
                    }
                    for i in range(5000)
                ]
                + [{"role": "ornith_backend", "rss_mib": {"sampled_peak": 1234}}],
                "gpu_used_mib": {"GPU-1": {"sampled_peak": 14500}},
                "memory_mib": {"used_mib": {"sampled_peak": 5000}},
            },
            "requests": ["RAW_RECORD"] * 10000,
            "backend_unassigned": ["RAW_RECORD"],
            "warnings": ["注意" * 10000] * 100,
        }
        result = render_brief(report, Path("data.json"), Path("result.md"))
        self.assertLessEqual(len(result.encode()), MAX_BRIEF_BYTES)
        self.assertNotIn("RAW_RECORD", result)
        brief = json.loads(result)
        self.assertTrue(brief["attached"])
        self.assertEqual(brief["resources"]["backend_rss_peak_mib"], 1234)
        self.assertEqual(brief["warnings_count"], 100)
        self.assertEqual(brief["services"]["ornith_9b"]["requests"], 3)

    def test_missing_metrics_stay_unknown(self):
        brief = json.loads(render_brief({"services": {"bridge": {}}}, Path("report.json")))
        self.assertIsNone(brief["resources"]["gpu_peak_mib"])
        self.assertIsNone(brief["services"]["bridge"]["weighted_decode_tokens_per_second"])
        self.assertFalse(brief["attached"])

    def test_control_characters_and_oversized_metrics_still_fit(self):
        brief = render_brief(
            {
                "test_id": "\x00" * 1000,
                "window_seconds": 10**1000,
                "services": {
                    "\x00" * 200 + str(i): {
                        "usage_coverage": "\x00" * 1000,
                        "native_speed_coverage": "\x00" * 1000,
                    }
                    for i in range(8)
                },
                "warnings": ["\x00" * 2000] * 5,
            },
            Path("中" * 1000),
        )
        self.assertLess(len(brief.encode()), MAX_BRIEF_BYTES)
        self.assertIsNone(json.loads(brief)["window_seconds"])

    def test_end_brief_attaches_markers_without_printing_full_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                str(ROOT / "scripts/model_telemetry.py"),
                "--directory",
                str(root),
            ]
            begin = subprocess.run(command + ["begin"], capture_output=True, text=True, check=True)
            test_id = begin.stdout.split("测试 ID：")[1].strip()
            target = root / "result.md"
            target.write_text(
                f"before\n<!-- MODEL_TELEMETRY_BEGIN test_id={test_id} -->\nwaiting\n<!-- MODEL_TELEMETRY_END -->\nafter\n"
            )
            result = subprocess.run(
                command
                + [
                    "end",
                    test_id,
                    "--settle-seconds",
                    "0",
                    "--attach-result",
                    str(target),
                    "--brief",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            brief = json.loads(result.stdout)
            self.assertLessEqual(len(result.stdout.encode()), MAX_BRIEF_BYTES)
            self.assertTrue(brief["attached"])
            self.assertEqual(brief["test_state"], "ended")
            self.assertTrue(target.read_text().endswith("after\n"))
            self.assertTrue(Path(brief["json_path"]).exists())
