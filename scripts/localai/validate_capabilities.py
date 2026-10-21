"""核对 LocalAI 能力清单与当前服务源码。"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "localai/manifests/capabilities.json"
_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "api_route"}
_AUDIO_CPP_SOURCE_REVISION = "9a02e61326aaaf9d462b584ca5e0daba22c0abfc"
_AUDIO_CPP_FAMILY_SCREEN_STATUSES = {
    "family_candidate",
    "partial_family_candidate",
    "no_exact_family",
    "excluded",
}


class CapabilityManifestError(ValueError):
    """能力清单结构或源码路由不一致。"""


@dataclass(frozen=True)
class CapabilityManifestSummary:
    """能力清单的已核对项目数和字段数。"""

    service_count: int
    uv_project_count: int
    route_count: int
    request_schema_count: int
    request_field_count: int
    family_candidate_count: int
    no_exact_family_count: int
    out_of_scope_count: int


def _decorated_routes(source_path: Path) -> set[tuple[str, str, int]]:
    """读取源码中以字符串常量声明的 FastAPI 路由及装饰器行号。"""
    try:
        source_tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise CapabilityManifestError(f"无法解析服务源码：{source_path}") from exc

    routes: set[tuple[str, str, int]] = set()
    for node in ast.walk(source_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.lower()
            if method not in _ROUTE_METHODS or not decorator.args:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                routes.add((method.upper(), path.value, decorator.lineno))
    return routes


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _fastapi_default_is_required(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return True
    if isinstance(node, ast.Call) and _call_name(node.func) in {"File", "Form"}:
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is Ellipsis:
            return True
        return any(
            keyword.arg == "default"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is Ellipsis
            for keyword in node.keywords
        )
    return False


def _route_parameter_facts(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict[str, Any]]:
    """提取路由函数的参数类型、默认值及 FastAPI 必填语义。"""
    positional = function.args.posonlyargs + function.args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(function.args.defaults))
    defaults.extend(function.args.defaults)
    facts: list[dict[str, Any]] = []
    for argument, default in zip(positional, defaults, strict=True):
        if argument.arg == "self":
            continue
        facts.append(
            {
                "name": argument.arg,
                "type": ast.unparse(argument.annotation) if argument.annotation else None,
                "default": ast.unparse(default) if default is not None else None,
                "required": _fastapi_default_is_required(default),
            }
        )
    for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True):
        facts.append(
            {
                "name": argument.arg,
                "type": ast.unparse(argument.annotation) if argument.annotation else None,
                "default": ast.unparse(default) if default is not None else None,
                "required": _fastapi_default_is_required(default),
            }
        )
    return facts


def _route_declarations(
    source_path: Path,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    """返回 FastAPI 路由到函数参数清单的静态索引。"""
    try:
        source_tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise CapabilityManifestError(f"无法解析服务源码：{source_path}") from exc

    declarations: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for node in ast.walk(source_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.upper()
            if method.lower() not in _ROUTE_METHODS or not decorator.args:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                declarations[(method, path.value, decorator.lineno)] = _route_parameter_facts(node)
    return declarations


def _pydantic_field_facts(node: ast.AnnAssign, source: str, project: str) -> dict[str, Any]:
    """提取 Pydantic 字段的类型、默认值、约束和必填性。"""
    if not isinstance(node.target, ast.Name):
        raise CapabilityManifestError(f"{project} 请求模型字段目标不是名称。")
    if node.value is None:
        default_node = None
        factory_node = None
        constraints: dict[str, str] = {}
        required = True
    elif isinstance(node.value, ast.Call) and _call_name(node.value.func) == "Field":
        keywords = {keyword.arg: keyword.value for keyword in node.value.keywords if keyword.arg}
        default_node = keywords.get("default")
        factory_node = keywords.get("default_factory")
        constraints = {
            name: ast.unparse(value)
            for name, value in keywords.items()
            if name not in {"default", "default_factory"}
        }
        required = default_node is None and factory_node is None
        if isinstance(default_node, ast.Constant) and default_node.value is Ellipsis:
            required = True
    else:
        default_node = node.value
        factory_node = None
        constraints = {}
        required = isinstance(default_node, ast.Constant) and default_node.value is Ellipsis

    return {
        "name": node.target.id,
        "type": ast.unparse(node.annotation),
        "default": ast.unparse(default_node) if default_node is not None else None,
        "default_factory": ast.unparse(factory_node) if factory_node is not None else None,
        "constraints": constraints,
        "required": required,
        "source": source,
        "line": node.lineno,
    }


def _pydantic_schema_facts(source_path: Path, project: str) -> dict[str, dict[str, Any]]:
    """提取服务 main.py 中声明的 Pydantic 类和字段。"""
    try:
        source = source_path.read_text(encoding="utf-8")
        source_tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise CapabilityManifestError(f"无法解析服务源码：{source_path}") from exc

    source_name = f"{project}/main.py"
    schemas: dict[str, dict[str, Any]] = {}
    for node in ast.walk(source_tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [
            _pydantic_field_facts(statement, source_name, project)
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        ]
        schemas[node.name] = {
            "line": node.lineno,
            "inherits": [
                ast.unparse(base)
                for base in node.bases
                if ast.unparse(base).split(".")[-1] != "BaseModel"
            ],
            "fields": fields,
        }
    return schemas


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapabilityManifestError(f"{label} 必须是 JSON object。")
    return value


def _validate_audio_cpp_family_screen(
    manifest: dict[str, Any],
    project_names: set[str],
) -> tuple[int, int, int]:
    """核对固定 audio.cpp family 筛选覆盖全部服务，且不混淆 E1 与运行证据。"""
    screen = _require_object(manifest.get("audio_cpp_family_screen"), "audio_cpp_family_screen")
    source_revision = screen.get("source_revision")
    source_readme = screen.get("source_readme")
    if source_revision != _AUDIO_CPP_SOURCE_REVISION:
        raise CapabilityManifestError("audio_cpp_family_screen 未指向 P0 固定 audio.cpp revision。")
    if (
        not isinstance(source_readme, str)
        or source_revision not in source_readme
        or not source_readme.endswith("/README.md")
    ):
        raise CapabilityManifestError(
            "audio_cpp_family_screen.source_readme 必须是固定 revision README。"
        )
    if screen.get("evidence_level") != "E1":
        raise CapabilityManifestError("audio.cpp README family 声明只能登记为 E1。")
    if not isinstance(screen.get("interpretation"), str) or not screen["interpretation"].strip():
        raise CapabilityManifestError("audio_cpp_family_screen.interpretation 不能为空。")

    entries = screen.get("services")
    if not isinstance(entries, list):
        raise CapabilityManifestError("audio_cpp_family_screen.services 必须是数组。")

    screened_projects: set[str] = set()
    candidate_count = 0
    no_exact_family_count = 0
    out_of_scope_count = 0
    for entry_value in entries:
        entry = _require_object(entry_value, "audio_cpp_family_screen.service")
        project = entry.get("project")
        if not isinstance(project, str) or not project or project in screened_projects:
            raise CapabilityManifestError(f"audio.cpp family 筛选项目名无效或重复：{project!r}")
        screened_projects.add(project)
        status = entry.get("status")
        if status not in _AUDIO_CPP_FAMILY_SCREEN_STATUSES:
            raise CapabilityManifestError(f"{project} 的 audio.cpp family 筛选状态无效：{status!r}")
        assessment = entry.get("assessment")
        if not isinstance(assessment, str) or not assessment.strip():
            raise CapabilityManifestError(f"{project} 的 audio.cpp family assessment 不能为空。")

        family = entry.get("family")
        tasks = entry.get("declared_tasks")
        line = entry.get("catalog_entry_line")
        if status in {"family_candidate", "partial_family_candidate"}:
            if (
                not isinstance(family, str)
                or not family
                or not isinstance(tasks, list)
                or not tasks
                or any(not isinstance(task, str) or not task for task in tasks)
                or len(set(tasks)) != len(tasks)
                or not isinstance(line, int)
                or line <= 0
            ):
                raise CapabilityManifestError(
                    f"{project} 的 audio.cpp 候选必须含 family、任务和 README 行号。"
                )
            candidate_count += 1
        elif status == "no_exact_family":
            if family is not None or tasks != [] or line is not None:
                raise CapabilityManifestError(
                    f"{project} 未匹配 family 时不得伪造候选 family、任务或行号。"
                )
            no_exact_family_count += 1
        else:
            if family is not None or tasks != [] or line is not None:
                raise CapabilityManifestError(
                    f"{project} 超出范围时不得登记 audio.cpp family 候选。"
                )
            out_of_scope_count += 1

    if screened_projects != project_names:
        raise CapabilityManifestError(
            "audio_cpp_family_screen 与服务清单覆盖范围不一致："
            f"缺筛选={sorted(project_names - screened_projects)}，"
            f"多余项目={sorted(screened_projects - project_names)}"
        )
    return candidate_count, no_exact_family_count, out_of_scope_count


def validate_capabilities_manifest(
    manifest_path: Path = MANIFEST_PATH,
    repository_root: Path = ROOT,
) -> CapabilityManifestSummary:
    """检查能力快照完整性、字段映射和源码中的 API 路由。"""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityManifestError(f"无法读取能力清单：{manifest_path}") from exc
    manifest = _require_object(manifest, "manifest")
    services = manifest.get("services")
    if not isinstance(services, list) or not services:
        raise CapabilityManifestError("manifest.services 必须是非空数组。")

    project_names: set[str] = set()
    schema_names: set[tuple[str, str]] = set()
    route_count = 0
    field_count = 0
    uv_project_count = 0
    problems: list[str] = []

    for service_value in services:
        service = _require_object(service_value, "service")
        project = service.get("project")
        if not isinstance(project, str) or not project or project in project_names:
            raise CapabilityManifestError(f"服务项目名无效或重复：{project!r}")
        project_names.add(project)

        project_path = (repository_root / project).resolve()
        if not project_path.is_relative_to(repository_root.resolve()):
            raise CapabilityManifestError(f"服务路径越出仓库根目录：{project}")
        source_path = project_path / "main.py"
        source_available = source_path.is_file()
        if not source_available:
            raise CapabilityManifestError(f"能力清单引用的服务源码不存在：{source_path}")
        if (project_path / "pyproject.toml").is_file():
            uv_project_count += 1

        request_schemas = service.get("request_schemas", {})
        if not isinstance(request_schemas, dict):
            raise CapabilityManifestError(f"{project}.request_schemas 必须是 object。")
        schema_names.update((project, name) for name in request_schemas)
        source_schemas = _pydantic_schema_facts(source_path, project) if source_available else {}
        schema_field_names: dict[str, set[str]] = {}
        for name, schema_value in request_schemas.items():
            schema = _require_object(schema_value, f"{project}.{name}")
            source_schema = source_schemas.get(name)
            if source_available and source_schema is None:
                raise CapabilityManifestError(f"{project} 请求模型源码不存在：{name}")
            expected_fields = schema.get("fields")
            if not isinstance(expected_fields, list) or any(
                not isinstance(field, dict) or not isinstance(field.get("name"), str)
                for field in expected_fields
            ):
                raise CapabilityManifestError(f"{project}.{name}.fields 必须是数组。")
            if (
                source_available
                and source_schema is not None
                and (
                    schema.get("source") != f"{project}/main.py"
                    or schema.get("line") != source_schema["line"]
                    or schema.get("inherits") != source_schema["inherits"]
                )
            ):
                problems.append(f"{project}.{name} 请求模型类或继承声明与源码不一致。")
            actual_fields = (
                source_schema["fields"] if source_schema is not None else expected_fields
            )
            if source_available and expected_fields != actual_fields:
                expected_names = {field["name"] for field in expected_fields}
                actual_names = {field["name"] for field in actual_fields}
                problems.append(
                    f"{project}.{name} 请求模型字段与源码不一致："
                    f"字段名变化={sorted(actual_names ^ expected_names)}；"
                    "另需核对类型、默认值、约束、必填性和行号。"
                )
            schema_field_names[name] = {field["name"] for field in expected_fields}

        routes = service.get("routes")
        mapping = _require_object(service.get("migration_field_mapping"), f"{project}.mapping")
        mapped_routes = mapping.get("routes")
        if not isinstance(routes, list) or not isinstance(mapped_routes, list):
            raise CapabilityManifestError(
                f"{project} 缺少 routes 或 migration_field_mapping.routes。"
            )

        expected_signatures: set[tuple[str, str, str, int]] = set()
        source_route_declarations = _route_declarations(source_path) if source_available else {}
        for route_value in routes:
            route = _require_object(route_value, f"{project}.route")
            method, path, source, line = (
                route.get("method"),
                route.get("path"),
                route.get("source"),
                route.get("line"),
            )
            if (
                not isinstance(method, str)
                or not isinstance(path, str)
                or source != f"{project}/main.py"
                or not isinstance(line, int)
            ):
                raise CapabilityManifestError(f"{project} 含有无效路由记录：{route!r}")
            signature = (method.upper(), path, source, line)
            if signature in expected_signatures:
                raise CapabilityManifestError(f"{project} 存在重复路由记录：{signature}")
            expected_signatures.add(signature)
            parameters = route.get("parameters", [])
            if not isinstance(parameters, list) or any(
                not isinstance(parameter, dict) for parameter in parameters
            ):
                raise CapabilityManifestError(f"{project} 路由 parameters 格式无效：{signature}")
            source_parameters = source_route_declarations.get((method.upper(), path, line))
            if source_available and source_parameters != parameters:
                problems.append(
                    f"{project} {method.upper()} {path} 路由参数与源码不一致："
                    "需核对参数名、类型、默认值及必填性。"
                )

        mapped_signatures: set[tuple[str, str, str, int]] = set()
        mapped_fields_by_signature: dict[tuple[str, str, str, int], set[str]] = {}
        for route_value in mapped_routes:
            route = _require_object(route_value, f"{project}.mapped_route")
            signature = (
                str(route.get("method", "")).upper(),
                route.get("path"),
                route.get("source"),
                route.get("line"),
            )
            if signature in mapped_signatures:
                raise CapabilityManifestError(f"{project} 存在重复字段映射路由：{signature}")
            mapped_signatures.add(signature)

            source_fields = route.get("source_fields", [])
            field_mappings = route.get("field_mappings", [])
            if not isinstance(source_fields, list) or not isinstance(field_mappings, list):
                raise CapabilityManifestError(f"{project} 路由字段清单格式无效：{signature}")
            if any(
                not isinstance(field, dict) or not isinstance(field.get("name"), str)
                for field in source_fields
            ):
                raise CapabilityManifestError(f"{project} 路由含有无效 source_fields：{signature}")
            if any(
                not isinstance(field, dict) or not isinstance(field.get("source_field"), str)
                for field in field_mappings
            ):
                raise CapabilityManifestError(f"{project} 路由含有无效 field_mappings：{signature}")
            expected_fields = {field["name"] for field in source_fields}
            mapped_fields = {field["source_field"] for field in field_mappings}
            mapped_fields_by_signature[signature] = expected_fields
            if expected_fields != mapped_fields:
                problems.append(
                    f"{project} {signature[0]} {signature[1]} 字段映射不完整："
                    f"未映射={sorted(expected_fields - mapped_fields)}，"
                    f"多余={sorted(mapped_fields - expected_fields)}"
                )
            field_count += len(source_fields)

        if expected_signatures != mapped_signatures:
            problems.append(
                f"{project} 路由与字段映射清单不一致："
                f"缺映射={sorted(expected_signatures - mapped_signatures)}，"
                f"多余映射={sorted(mapped_signatures - expected_signatures)}"
            )

        for route_value in routes:
            route = _require_object(route_value, f"{project}.route")
            schema_refs = route.get("request_schemas", [])
            if not isinstance(schema_refs, list):
                raise CapabilityManifestError(f"{project} 路由 request_schemas 格式无效。")
            if not schema_refs:
                continue
            try:
                schema_fields = set().union(*(schema_field_names[name] for name in schema_refs))
            except KeyError as exc:
                raise CapabilityManifestError(
                    f"{project} 路由引用未登记的请求模型：{exc.args[0]}"
                ) from exc
            signature = (
                str(route.get("method", "")).upper(),
                route.get("path"),
                route.get("source"),
                route.get("line"),
            )
            mapped_fields = mapped_fields_by_signature.get(signature, set())
            if schema_fields != mapped_fields:
                problems.append(
                    f"{project} {signature[0]} {signature[1]} 请求模型字段与迁移映射不一致："
                    f"未映射={sorted(schema_fields - mapped_fields)}，"
                    f"多余={sorted(mapped_fields - schema_fields)}"
                )

        if source_available:
            actual_routes = {
                (method, path, f"{project}/main.py", line)
                for method, path, line in _decorated_routes(source_path)
            }
            if expected_signatures != actual_routes:
                problems.append(
                    f"{project} 源码路由与快照不一致："
                    f"新增/变更={sorted(actual_routes - expected_signatures)}，"
                    f"已删除/过期={sorted(expected_signatures - actual_routes)}"
                )
        route_count += len(routes)

    declared_uv_projects = manifest.get("service_count_with_uv_projects")
    if declared_uv_projects != uv_project_count:
        problems.append(
            "service_count_with_uv_projects 与项目目录不一致："
            f"清单={declared_uv_projects!r}，实际={uv_project_count}"
        )
    family_candidate_count, no_exact_family_count, out_of_scope_count = (
        _validate_audio_cpp_family_screen(manifest, project_names)
    )
    if problems:
        raise CapabilityManifestError("\n".join(problems))

    return CapabilityManifestSummary(
        service_count=len(services),
        uv_project_count=uv_project_count,
        route_count=route_count,
        request_schema_count=len(schema_names),
        request_field_count=field_count,
        family_candidate_count=family_candidate_count,
        no_exact_family_count=no_exact_family_count,
        out_of_scope_count=out_of_scope_count,
    )


def main() -> int:
    """运行源码和能力快照一致性检查。"""
    try:
        summary = validate_capabilities_manifest()
    except CapabilityManifestError as exc:
        print(f"LocalAI 能力清单校验失败：\n{exc}", file=sys.stderr)
        return 1
    print(
        "LocalAI 能力清单校验通过："
        f"{summary.service_count} 个服务入口，{summary.uv_project_count} 个 uv 项目，"
        f"{summary.route_count} 条路由，{summary.request_schema_count} 个请求 schema，"
        f"{summary.request_field_count} 个请求字段；固定 audio.cpp family 筛选含 "
        f"{summary.family_candidate_count} 个模型族候选、{summary.no_exact_family_count} 个无精确候选、"
        f"{summary.out_of_scope_count} 个范围外服务。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
