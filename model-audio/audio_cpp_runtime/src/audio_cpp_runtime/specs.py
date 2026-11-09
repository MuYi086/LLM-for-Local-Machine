"""从 LocalAI 候选清单读取受支持的音频模型与 HTTP 服务配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_DIR = Path(__file__).resolve().parents[4]
MODEL_MANIFEST_PATH = REPOSITORY_DIR / "localai/manifests/audio-cpp-models.json"


@dataclass(frozen=True)
class ModelSpec:
    """单个 audio.cpp GGUF 与其 HTTP 服务合同。"""

    key: str
    model_id: str
    source_directory: str
    required_file: str
    model_file_env: str
    family: str
    task: str
    directory: str
    port: int
    route: str
    kind: str
    description: str


def _read_model_specs() -> dict[str, ModelSpec]:
    """读取并检查共享候选清单中的八个模型服务。"""
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        candidates = manifest["models"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 audio.cpp 模型清单：{MODEL_MANIFEST_PATH}") from exc
    if manifest.get("schema_version") != 1 or not isinstance(candidates, list):
        raise RuntimeError("audio.cpp 模型清单版本或 models 格式无效。")

    specs: dict[str, ModelSpec] = {}
    routes: set[str] = set()
    ports: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("service"), dict):
            raise TypeError("audio.cpp 模型候选格式无效。")
        service = candidate["service"]
        if service.get("runtime") != "audio_cpp_runtime":
            continue
        values = {
            "key": candidate.get("key"),
            "model_id": candidate.get("model_id"),
            "source_directory": candidate.get("source_directory"),
            "required_file": candidate.get("required_file"),
            "model_file_env": candidate.get("model_file_env"),
            "family": candidate.get("family"),
            "task": candidate.get("task"),
            "directory": service.get("directory"),
            "port": service.get("port"),
            "route": service.get("route"),
            "kind": service.get("kind"),
            "description": service.get("description"),
        }
        if any(value is None for value in values.values()):
            raise RuntimeError(f"audio.cpp 候选字段缺失：{candidate.get('key')}")
        key = str(values["key"])
        route = str(values["route"])
        port = int(values["port"])
        if key in specs or route in routes or port in ports:
            raise RuntimeError(f"audio.cpp 服务 key、路由或端口重复：{key}")
        routes.add(route)
        ports.add(port)
        specs[key] = ModelSpec(**values)
    if len(specs) != 8:
        raise RuntimeError(f"当前 audio.cpp runtime 应登记 8 个模型服务，实际为 {len(specs)} 个。")
    return specs


MODEL_SPECS = _read_model_specs()


def model_manifest() -> dict[str, Any]:
    """读取完整清单，供候选校验和能力门禁使用。"""
    return json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
