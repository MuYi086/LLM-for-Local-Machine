"""验证 Ornith 9B 实测报告的流解析和资源汇总。"""

from __future__ import annotations

import unittest

from tests.ornith_1_5_9b_gguf.benchmark_live import (
    ResourceSample,
    build_request_payload,
    parse_backend_context_size,
    parse_sse_stream,
    summarize_resources,
    summarize_stream,
)


class LiveBenchmarkTests(unittest.TestCase):
    def test_report_context_is_read_from_backend_launch_arguments(self) -> None:
        arguments = [b"llama-server", b"--model", b"Ornith.gguf", b"--ctx-size", b"98304"]

        self.assertEqual(parse_backend_context_size(arguments), 98304)
        self.assertIsNone(parse_backend_context_size([b"llama-server", b"--ctx-size", b"bad"]))

    def test_live_request_disables_thinking_and_requests_usage(self) -> None:
        payload = build_request_payload("测试提示", max_tokens=96)

        self.assertEqual(payload["model"], "Ornith-1.5-9B-GGUF")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "测试提示"}])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["stream"], True)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["max_tokens"], 96)

    def test_stream_metrics_use_first_and_last_content_events(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":"\\u4f60\\u597d"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":"\\uff0c\\u4e16\\u754c\\u3002"},"finish_reason":"stop"}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":12}}\n',
            b"data: [DONE]\n",
        ]
        timestamps = iter((10.2, 11.2))

        result = parse_sse_stream(lines, clock=lambda: next(timestamps))
        metrics = summarize_stream(result, request_started_at=10.0, request_finished_at=12.0)

        self.assertEqual(result.text, "你好，世界。")
        self.assertEqual(result.completion_tokens, 12)
        self.assertEqual(result.finish_reasons, ["stop"])
        self.assertEqual(metrics["time_to_first_token_seconds"], 0.2)
        self.assertEqual(metrics["generation_interval_seconds"], 1.0)
        self.assertEqual(metrics["average_tokens_per_second"], 12.0)
        self.assertEqual(metrics["elapsed_seconds"], 2.0)

    def test_resource_summary_reports_baseline_peak_and_deltas(self) -> None:
        samples = [
            ResourceSample(1.0, 4000, 8192, 6.0, 6.1, 12000, 28000, 40000),
            ResourceSample(2.0, 7000, 8192, 12.0, 12.5, 19000, 21000, 40000),
            ResourceSample(3.0, 5000, 8192, 9.0, 12.5, 15000, 25000, 40000),
        ]

        summary = summarize_resources(samples, interval_seconds=0.25)

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["gpu_used_baseline_mib"], 4000)
        self.assertEqual(summary["gpu_used_peak_mib"], 7000)
        self.assertEqual(summary["gpu_used_peak_delta_mib"], 3000)
        self.assertEqual(summary["backend_rss_peak_mib"], 12.0)
        self.assertEqual(summary["host_memory_used_peak_mib"], 19000)
        self.assertEqual(summary["host_memory_available_min_mib"], 21000)
        self.assertEqual(summary["sample_interval_seconds"], 0.25)


if __name__ == "__main__":
    unittest.main()
