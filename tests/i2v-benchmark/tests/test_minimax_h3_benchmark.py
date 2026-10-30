"""MiniMax H3 对照基准的无模型合同回归。"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BENCHMARK_DIR / "minimax_h3/benchmark.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class MiniMaxH3BenchmarkTests(unittest.TestCase):
    """无需 CUDA 或权重即可验证 H3 基准配置。"""

    def test_loads_independent_scenes_seeds_and_video_geometry(self) -> None:
        config = benchmark.load_config()
        source = json.loads((BENCHMARK_DIR / "minimax_h3/scenes.json").read_text(encoding="utf-8"))

        self.assertEqual(config["scenes"], source["scenes"])
        self.assertEqual(config["seeds"], source["seeds"])
        self.assertEqual(config["width"], source["width"])
        self.assertEqual(config["height"], source["height"])
        self.assertEqual(config["fps_target"], source["fps_target"])
        self.assertEqual(len(benchmark.trials(config)), 9)

    def test_keeps_h3_native_steps_duration_and_frame_grid(self) -> None:
        config = benchmark.load_config()

        self.assertEqual(config["native_steps"], 20)
        self.assertEqual(config["duration_seconds"], 5.0)
        self.assertEqual(benchmark.expected_frame_count(config), 124)
        self.assertTrue(config["include_audio"])
        self.assertEqual(config["expected_audio_streams"], 1)

    def test_request_uses_shared_prompt_seed_and_i2v_image(self) -> None:
        config = benchmark.load_config()
        payload = benchmark.build_request(
            config,
            "rainy_city",
            1234,
            "benchmark/rainy_city.png",
        )

        self.assertEqual(payload["mode"], "fl2va")
        self.assertEqual(payload["prompt"], config["scenes"]["rainy_city"]["prompt"])
        self.assertEqual(payload["first_frame_path"], "benchmark/rainy_city.png")
        self.assertEqual(payload["seed"], 1234)
        self.assertEqual(payload["width"], 480)
        self.assertEqual(payload["height"], 832)
        self.assertEqual(payload["duration_seconds"], 5.0)
        self.assertTrue(payload["include_audio"])

    def test_results_stay_under_ignored_i2v_benchmark_directory(self) -> None:
        config = benchmark.load_config()
        target = benchmark.output_root(config).relative_to(BENCHMARK_DIR)

        self.assertEqual(target, Path("results/minimax_h3"))
        ignored = (BENCHMARK_DIR / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/", ignored)

    def test_uploaded_image_cleanup_removes_empty_lock_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.png"
            lock = Path(temporary) / ".image.png.lock"
            image.write_bytes(b"test")
            lock.touch()

            benchmark._remove_uploaded_images([image])

            self.assertFalse(image.exists())
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
