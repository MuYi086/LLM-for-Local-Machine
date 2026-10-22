"""仓库 Python 代码质量基线的无模型回归测试。"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

# 配置测试确保各 uv 项目有锁文件、统一 Ruff 基线且不在启动时同步依赖。

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
MODEL_PROJECTS = (
    "qwen3_tts",
    "voxcpm2",
    "ace_step_1_5",
    "qwen3_voiceDesign",
    "Confucius4_TTS",
    "YuE2-3B",
    "SheetSage2",
    "YuE2-Vae",
    "MERT-v2-FullSong",
    "Qwen3_ASR_1.7B",
    "TIGER-DnR",
)
EXPECTED_RUFF_RULES = {"E", "F", "I", "UP", "B"}


class PythonQualityConfigurationTests(unittest.TestCase):
    def test_each_model_project_has_a_locked_ruff_baseline(self) -> None:
        for project_name in MODEL_PROJECTS:
            with self.subTest(project=project_name):
                project_dir = REPOSITORY_DIR / project_name
                configuration = tomllib.loads(
                    (project_dir / "pyproject.toml").read_text(encoding="utf-8")
                )

                self.assertEqual(configuration["project"]["requires-python"], "==3.12.13")
                self.assertFalse(configuration["tool"]["uv"]["package"])
                self.assertIn("ruff>=0.15.9", configuration["dependency-groups"]["dev"])
                self.assertEqual(
                    (project_dir / ".python-version").read_text(encoding="utf-8").strip(),
                    "3.12.13",
                )

                ruff_config = configuration["tool"]["ruff"]
                self.assertEqual(ruff_config["line-length"], 100)
                self.assertEqual(ruff_config["target-version"], "py312")
                self.assertEqual(
                    set(ruff_config["lint"]["select"]),
                    EXPECTED_RUFF_RULES,
                )
                self.assertIn("B008", ruff_config["lint"]["ignore"])
                self.assertIn("E501", ruff_config["lint"]["ignore"])
                self.assertIn(
                    'name = "ruff"', (project_dir / "uv.lock").read_text(encoding="utf-8")
                )

    def test_root_quality_rules_remain_configured(self) -> None:
        root_ruff = tomllib.loads((REPOSITORY_DIR / "ruff.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(root_ruff["lint"]["select"]), EXPECTED_RUFF_RULES)
        self.assertIn("E501", root_ruff["lint"]["ignore"])

    def test_startup_uses_locked_environments_without_runtime_sync(self) -> None:
        start_script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")

        self.assertEqual(start_script.count("setsid uv run --no-sync --project"), 12)
        self.assertNotIn("setsid uv run --project", start_script)
        self.assertEqual(start_script.count('for pid in "${pids[@]}"; do'), 3)
        self.assertNotIn('wait "$qwen3_tts_pid"', start_script)


if __name__ == "__main__":
    unittest.main()
