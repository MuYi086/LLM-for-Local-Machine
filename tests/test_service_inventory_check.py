"""验证服务清单一致性校验脚本的推导与核对逻辑。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_service_inventory.py"


def _load_module() -> Any:
    """以文件路径加载校验脚本，避免依赖 scripts 包结构。"""
    spec = importlib.util.spec_from_file_location("check_service_inventory", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ChineseNumeralTest(unittest.TestCase):
    """中文数字与整数的互转。"""

    def setUp(self) -> None:
        self.module = _load_module()

    def test_roundtrip_common_values(self) -> None:
        for value in (1, 4, 10, 11, 14, 15, 17, 21, 30, 99):
            with self.subTest(value=value):
                self.assertEqual(self.module.cn_to_int(self.module.int_to_cn(value)), value)

    def test_parse_known_wording(self) -> None:
        cases = {"十": 10, "十五": 15, "十七": 17, "二十一": 21, "十四": 14}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.module.cn_to_int(text), expected)

    def test_invalid_numeral_returns_none(self) -> None:
        self.assertIsNone(self.module.cn_to_int("百"))
        self.assertIsNone(self.module.cn_to_int(""))


class BuildInventoryTest(unittest.TestCase):
    """从清单 JSON 推导期望口径。"""

    def setUp(self) -> None:
        self.module = _load_module()
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "models": [
                {
                    "key": "a",
                    "service": {
                        "runtime": "audio_cpp_runtime",
                        "port": 8344,
                        "route": "/v1/audio-cpp/A",
                        "directory": "model-audio/A",
                    },
                },
                {
                    "key": "b",
                    "service": {
                        "runtime": "audio_cpp_runtime",
                        "port": 8345,
                        "route": "/v1/audio-cpp/B",
                        "directory": "model-audio/B",
                    },
                },
                {
                    "key": "vd",
                    "service": {
                        "runtime": "voicedesign",
                        "port": 8301,
                        "route": "/v1/qwen/VD",
                        "directory": "model-audio/vd",
                    },
                },
            ],
        }

    def test_counts_follow_manifest(self) -> None:
        inventory = self.module.build_expected_inventory(self.manifest)
        self.assertEqual(len(inventory.audio_services), 3)
        self.assertEqual(len(inventory.audio_cpp_services), 2)
        self.assertEqual(len(inventory.voicedesign_services), 1)
        self.assertEqual(
            inventory.total_http_processes,
            3 + len(self.module.FIXED_HTTP_SERVICES),
        )

    def test_duplicate_port_rejected(self) -> None:
        self.manifest["models"][1]["service"]["port"] = 8344
        with self.assertRaises(self.module.ServiceInventoryError):
            self.module.build_expected_inventory(self.manifest)

    def test_unknown_runtime_rejected(self) -> None:
        self.manifest["models"][0]["service"]["runtime"] = "future_runtime"
        with self.assertRaises(self.module.ServiceInventoryError):
            self.module.build_expected_inventory(self.manifest)


class CheckDocumentTest(unittest.TestCase):
    """文档核对逻辑能识别缺失项与过期数量声明。"""

    def setUp(self) -> None:
        self.module = _load_module()
        self.inventory = self.module.build_expected_inventory(
            {
                "models": [
                    {
                        "key": "a",
                        "service": {
                            "runtime": "audio_cpp_runtime",
                            "port": 8344,
                            "route": "/v1/audio-cpp/A",
                        },
                    }
                ]
            }
        )
        self.total = self.inventory.total_http_processes
        self.audio_cn = self.module.int_to_cn(len(self.inventory.audio_services))
        self.cpp_cn = self.module.int_to_cn(len(self.inventory.audio_cpp_services))
        self.total_cn = self.module.int_to_cn(self.total)

    def _baseline_doc(self) -> str:
        fixed_rows = "\n".join(
            f"| {label} | {port} | 说明 |" for label, port in self.inventory.fixed_services
        )
        return (
            f"共{self.total_cn}个 HTTP 进程由 start.sh 管理。\n"
            f"本仓库提供{self.audio_cn}个音频模型 HTTP API，{self.cpp_cn}个 audio.cpp 服务，"
            f"共{self.audio_cn}个 LocalAI 候选。\n"
            "| 服务 | 端口 | 接口 |\n| --- | ---: | --- |\n"
            "| A | 8344 | POST /v1/audio-cpp/A |\n"
            f"{fixed_rows}\n"
        )

    def test_baseline_doc_passes(self) -> None:
        errors = self.module.check_document("doc", self._baseline_doc(), self.inventory)
        self.assertEqual(errors, [])

    def test_stale_count_reported(self) -> None:
        text = self._baseline_doc().replace(f"{self.total_cn}个 HTTP 进程", "十七个 HTTP 进程")
        errors = self.module.check_document("doc", text, self.inventory)
        self.assertTrue(any("HTTP 进程" in error for error in errors))

    def test_missing_route_and_table_row_reported(self) -> None:
        # 删除服务表行且路由缺失，正文仍保留端口字样时，应报告端口不在服务表。
        text = self._baseline_doc().replace("| A | 8344 | POST /v1/audio-cpp/A |\n", "")
        text = text + "备用端口 8344。\n"
        errors = self.module.check_document("doc", text, self.inventory)
        self.assertTrue(any("缺少路由 /v1/audio-cpp/A" in error for error in errors))
        self.assertTrue(any("端口 8344 未出现在服务表中" in error for error in errors))

    def test_unknown_table_port_reported(self) -> None:
        text = self._baseline_doc() + "| 退役服务 | 8399 | POST /v1/audio-cpp/RETIRED |\n"
        errors = self.module.check_document("doc", text, self.inventory)
        self.assertTrue(any("8399" in error for error in errors))


class RepositoryConsistencyTest(unittest.TestCase):
    """当前仓库真实文件应通过服务口径校验（回归门禁）。"""

    def test_real_repository_passes(self) -> None:
        module = _load_module()
        manifest = json.loads(
            (ROOT / "localai/manifests/audio-cpp-models.json").read_text(encoding="utf-8")
        )
        inventory = module.build_expected_inventory(manifest)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        start_sh = (ROOT / "start.sh").read_text(encoding="utf-8")
        errors = (
            module.check_document("README.md", readme, inventory)
            + module.check_document("AGENTS.md", agents, inventory)
            + module.check_start_sh(start_sh, inventory)
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
