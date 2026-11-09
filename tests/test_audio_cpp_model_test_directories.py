"""确保每个 audio.cpp 模型有独立的真实 API 冒烟入口。"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.audio_cpp import generate_examples

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "localai/manifests/audio-cpp-models.json"


class AudioCppModelTestDirectoryTests(unittest.TestCase):
    def test_single_model_runner_pins_the_requested_service(self) -> None:
        with patch.object(generate_examples, "generate_examples", return_value=[]) as generate:
            result = generate_examples.run_single_model(
                "midashenglm_gen",
                argv=["--timeout", "12"],
            )

        self.assertEqual(result, 0)
        generate.assert_called_once_with(
            model_keys=["midashenglm_gen"],
            host="127.0.0.1",
            timeout=12,
            reference_audio=None,
            reference_text=None,
            source_audio=None,
            output_dir=ROOT / "storage/audio_cpp_tests",
        )

    def test_each_audio_cpp_service_has_a_model_specific_generation_runner(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        candidates = [
            candidate
            for candidate in manifest["models"]
            if candidate["service"].get("runtime") == "audio_cpp_runtime"
        ]
        self.assertEqual(len(candidates), 8)

        for candidate in candidates:
            key = candidate["key"]
            directory_name = Path(candidate["service"]["directory"]).name
            test_directory = ROOT / "tests" / directory_name
            with self.subTest(model=key):
                readme = test_directory / "README.md"
                runner = test_directory / "generate_example.py"
                self.assertTrue(readme.is_file(), f"缺少模型测试说明：{readme}")
                self.assertTrue(runner.is_file(), f"缺少模型真实生成入口：{runner}")

                module_spec = importlib.util.spec_from_file_location(
                    f"model_api_smoke_{key}", runner
                )
                self.assertIsNotNone(module_spec)
                self.assertIsNotNone(module_spec.loader)
                module = importlib.util.module_from_spec(module_spec)
                with patch.object(generate_examples, "run_single_model", return_value=0) as run:
                    module_spec.loader.exec_module(module)
                    self.assertEqual(module.main(), 0)
                run.assert_called_once_with(key)


if __name__ == "__main__":
    unittest.main()
