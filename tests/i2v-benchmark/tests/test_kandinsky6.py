"""Kandinsky 对照基准的无模型配置与请求条件回归。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BENCHMARK_DIR / "kandinsky6/benchmark.py"
SPEC = importlib.util.spec_from_file_location("kandinsky6_benchmark_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Kandinsky6BenchmarkTests(unittest.TestCase):
    """无需 Torch、CUDA 或模型即可验证对照矩阵和本地文件合同。"""

    def test_kandinsky_uses_same_inputs_and_video_conditions_as_hunyuan(self) -> None:
        hunyuan = yaml.safe_load((BENCHMARK_DIR / "config.yaml").read_text(encoding="utf-8"))
        kandinsky = benchmark.load_config()

        self.assertEqual(kandinsky["seeds"], hunyuan["seeds"])
        self.assertEqual(kandinsky["width"], hunyuan["width"])
        self.assertEqual(kandinsky["height"], hunyuan["height"])
        self.assertEqual(kandinsky["fps_target"], hunyuan["fps_target"])
        self.assertEqual(kandinsky["frames"], hunyuan["frames"])
        self.assertEqual(kandinsky["scenes"], hunyuan["scenes"])
        self.assertEqual(len(benchmark.trials(kandinsky)), 9)

    def test_model_specific_distilled_settings_and_video_only_mode(self) -> None:
        config = benchmark.load_config()

        self.assertEqual(config["num_inference_steps"], 10)
        self.assertEqual(config["guidance_scale"], 1.0)
        self.assertFalse(config["sample_audio"])
        self.assertFalse(config["expand_prompts"])

    def test_worker_payload_keeps_conditioning_local_and_output_portrait(self) -> None:
        config = benchmark.load_config()
        image_file = BENCHMARK_DIR / "assets/rainy_city.png"
        payload = benchmark.build_payload(config, "rainy_city", 1234, image_file)

        self.assertEqual(payload["width"], 480)
        self.assertEqual(payload["height"], 832)
        self.assertEqual(payload["num_frames"], 121)
        self.assertEqual(payload["frame_rate"], 24)
        self.assertEqual(payload["seed"], 1234)
        self.assertEqual(payload["prompt"], config["scenes"]["rainy_city"]["prompt"])
        self.assertTrue(payload["local_files_only"])
        self.assertFalse(payload["sample_audio"])

    def test_results_stay_under_ignored_benchmark_directory(self) -> None:
        config = benchmark.load_config()
        target = benchmark.output_root(config).relative_to(BENCHMARK_DIR)

        self.assertEqual(target, Path("results/kandinsky6"))
        ignored = (BENCHMARK_DIR / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/", ignored)


if __name__ == "__main__":
    unittest.main()
