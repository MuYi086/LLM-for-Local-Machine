"""独立 ACE-Step 1.5 服务的静态迁移检查。"""

from __future__ import annotations

# 迁移测试确认 ACE-Step 已独立成 uv 服务，且控制面没有重新接管模型推理。
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "model-audio" / "ace_step_1_5"


class AceStepMigrationTests(unittest.TestCase):
    def test_service_files_and_locked_project_exist(self) -> None:
        for relative_path in (
            ".python-version",
            "README.md",
            "main.py",
            "runtime.py",
            "worker.py",
            "pyproject.toml",
            "uv.lock",
            "tests/test_api.py",
            "tests/test_runtime.py",
        ):
            self.assertTrue((SERVICE_DIR / relative_path).is_file(), relative_path)

    def test_start_script_keeps_ace_step_separate_and_offline_at_startup(self) -> None:
        script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'ACESTEP_PROJECT_DIR="${ACESTEP_PROJECT_DIR:-$PROJECT_DIR/ace_step_1_5}"', script
        )
        self.assertIn('export ACESTEP_PORT="${ACESTEP_PORT:-8313}"', script)
        self.assertIn('setsid uv run --no-sync --project "$ACESTEP_PROJECT_DIR"', script)
        self.assertIn('python "$ACESTEP_PROJECT_DIR/main.py"', script)
        self.assertNotIn("model-audio/ace_step_1_5/worker.py", script)

    def test_final_route_and_bgm_storage_contract_are_documented(self) -> None:
        readme = (REPOSITORY_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("/v1/aceStep/bgm", readme)
        self.assertIn("storage/bgm/", readme)
        self.assertIn("8313", readme)


if __name__ == "__main__":
    unittest.main()
