from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.localai.validate_capabilities import (
    CapabilityManifestError,
    validate_capabilities_manifest,
)


class LocalAICapabilityManifestTests(unittest.TestCase):
    def test_repository_capability_manifest_matches_fastapi_source(self) -> None:
        summary = validate_capabilities_manifest()

        self.assertEqual(summary.service_count, 24)
        self.assertEqual(summary.route_count, 79)
        self.assertEqual(summary.request_schema_count, 17)
        self.assertEqual(summary.request_field_count, 268)
        self.assertEqual(summary.uv_project_count, 23)
        self.assertEqual(summary.family_candidate_count, 13)
        self.assertEqual(summary.no_exact_family_count, 9)
        self.assertEqual(summary.out_of_scope_count, 2)

    def test_rejects_incomplete_audio_cpp_family_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self._write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["audio_cpp_family_screen"]["services"].clear()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                CapabilityManifestError, "family_screen 与服务清单覆盖范围不一致"
            ):
                validate_capabilities_manifest(manifest_path, root)

    def test_rejects_manifest_when_source_route_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_fixture(root)
            service_main = root / "svc/main.py"
            service_main.write_text(
                'from fastapi import FastAPI\napp = FastAPI()\n@app.post("/new")\ndef run(): pass\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CapabilityManifestError, "源码路由与快照不一致"):
                validate_capabilities_manifest(root / "capabilities.json", root)

    def test_rejects_unmapped_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self._write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["services"][0]["migration_field_mapping"]["routes"][0]["field_mappings"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CapabilityManifestError, "字段映射不完整"):
                validate_capabilities_manifest(manifest_path, root)

    def test_rejects_request_schema_requiredness_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service_path = root / "svc"
            service_path.mkdir()
            (service_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (service_path / "main.py").write_text(
                "from fastapi import FastAPI\n"
                "from pydantic import BaseModel, Field\n"
                "app = FastAPI()\n"
                "class Body(BaseModel):\n"
                "    text: str = Field(min_length=1)\n"
                "@app.post('/run')\n"
                "def run(request: Body): pass\n",
                encoding="utf-8",
            )
            field = {
                "name": "text",
                "type": "str",
                "default": None,
                "default_factory": None,
                "constraints": {"min_length": "1"},
                "required": True,
                "source": "svc/main.py",
                "line": 5,
            }
            route = {
                "method": "POST",
                "path": "/run",
                "source": "svc/main.py",
                "line": 6,
                "request_schemas": ["Body"],
            }
            mapped_route = {
                **route,
                "source_fields": [field],
                "field_mappings": [{"source_field": "text", "destination_field": "request.text"}],
            }
            manifest_path = root / "capabilities.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "service_count_with_uv_projects": 1,
                        "services": [
                            {
                                "project": "svc",
                                "request_schemas": {
                                    "Body": {
                                        "source": "svc/main.py",
                                        "line": 4,
                                        "inherits": [],
                                        "fields": [field],
                                    }
                                },
                                "routes": [route],
                                "migration_field_mapping": {"routes": [mapped_route]},
                            }
                        ],
                        "audio_cpp_family_screen": self._family_screen_fixture(),
                    }
                ),
                encoding="utf-8",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            request_parameter = {
                "name": "request",
                "type": "Body",
                "default": None,
                "required": True,
            }
            manifest["services"][0]["routes"][0]["parameters"] = [request_parameter]
            manifest["services"][0]["migration_field_mapping"]["routes"][0]["parameters"] = [
                request_parameter
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(
                validate_capabilities_manifest(manifest_path, root).request_field_count, 1
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["services"][0]["request_schemas"]["Body"]["fields"][0]["required"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CapabilityManifestError, "请求模型字段与源码不一致"):
                validate_capabilities_manifest(manifest_path, root)

    def test_rejects_fastapi_file_requiredness_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service_path = root / "svc"
            service_path.mkdir()
            (service_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (service_path / "main.py").write_text(
                "from fastapi import FastAPI, File, UploadFile\n"
                "app = FastAPI()\n"
                "@app.post('/upload')\n"
                "def upload(audio: UploadFile = File(...)): pass\n",
                encoding="utf-8",
            )
            route = {
                "method": "POST",
                "path": "/upload",
                "source": "svc/main.py",
                "line": 3,
                "parameters": [
                    {
                        "name": "audio",
                        "type": "UploadFile",
                        "default": "File(...)",
                        "required": True,
                    }
                ],
            }
            mapped_route = {
                **route,
                "source_fields": [{"name": "audio"}],
                "field_mappings": [{"source_field": "audio", "destination_field": "request.audio"}],
            }
            manifest_path = root / "capabilities.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "service_count_with_uv_projects": 1,
                        "services": [
                            {
                                "project": "svc",
                                "request_schemas": {},
                                "routes": [route],
                                "migration_field_mapping": {"routes": [mapped_route]},
                            }
                        ],
                        "audio_cpp_family_screen": self._family_screen_fixture(),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(validate_capabilities_manifest(manifest_path, root).route_count, 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["services"][0]["routes"][0]["parameters"][0]["required"] = False
            manifest["services"][0]["migration_field_mapping"]["routes"][0]["parameters"][0][
                "required"
            ] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CapabilityManifestError, "路由参数与源码不一致"):
                validate_capabilities_manifest(manifest_path, root)

    @staticmethod
    def _write_fixture(root: Path) -> Path:
        service_path = root / "svc"
        service_path.mkdir()
        (service_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (service_path / "main.py").write_text(
            'from fastapi import FastAPI\napp = FastAPI()\n@app.post("/run")\ndef run(): pass\n',
            encoding="utf-8",
        )
        route = {
            "method": "POST",
            "path": "/run",
            "source": "svc/main.py",
            "line": 3,
            "source_fields": [{"name": "text"}],
            "field_mappings": [{"source_field": "text", "destination_field": "text"}],
        }
        manifest_path = root / "capabilities.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "service_count_with_uv_projects": 1,
                    "services": [
                        {
                            "project": "svc",
                            "request_schemas": {},
                            "routes": [route],
                            "migration_field_mapping": {"routes": [route]},
                        }
                    ],
                    "audio_cpp_family_screen": LocalAICapabilityManifestTests._family_screen_fixture(),
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    @staticmethod
    def _family_screen_fixture() -> dict[str, object]:
        return {
            "source_revision": "9a02e61326aaaf9d462b584ca5e0daba22c0abfc",
            "source_readme": (
                "https://github.com/0xShug0/audio.cpp/blob/"
                "9a02e61326aaaf9d462b584ca5e0daba22c0abfc/README.md"
            ),
            "evidence_level": "E1",
            "interpretation": "固定上游 README family 声明，不代表运行时验证。",
            "services": [
                {
                    "project": "svc",
                    "status": "no_exact_family",
                    "family": None,
                    "declared_tasks": [],
                    "catalog_entry_line": None,
                    "assessment": "测试服务没有精确 family 候选。",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
