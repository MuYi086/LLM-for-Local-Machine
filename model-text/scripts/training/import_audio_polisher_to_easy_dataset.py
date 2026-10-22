#!/usr/bin/env python3
"""把有声书文本润色 POC 来源和候选数据导入 Easy Dataset。"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def decide_dataset_import(current_total: int, expected_total: int) -> str:
    """判断应导入、跳过还是阻止可能产生重复的部分导入。"""
    if current_total == 0:
        return "import"
    if current_total == expected_total:
        return "skip"
    raise ValueError(
        f"Easy Dataset 当前有 {current_total} 条，预期为 0 或 {expected_total} 条；"
        "请先人工核对，避免重复导入"
    )


class EasyDatasetClient:
    """封装本地 Easy Dataset 所需的最小 HTTP API。"""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        path: str,
        *,
        payload: Any | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        method: str | None = None,
    ) -> Any:
        """发送请求并解析 JSON 响应。"""
        request_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)


def import_project(
    *,
    client: EasyDatasetClient,
    project_name: str,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """幂等创建项目、上传来源并导入未经确认的候选数据。"""
    manifest = json.loads((output_dir / "data" / "manifest.json").read_text(encoding="utf-8"))
    import_payload = json.loads(
        (output_dir / "easy_dataset_import.json").read_text(encoding="utf-8")
    )
    expected_total = len(import_payload["datasets"])

    projects = client.request("/api/projects")
    project = next((item for item in projects if item["name"] == project_name), None)
    if project is None:
        project = client.request(
            "/api/projects",
            payload={
                "name": project_name,
                "description": (
                    "A 原始小说片段到 B 纯净可朗读文本；自动 POC，"
                    "research-only，所有候选均待人工审核。"
                ),
            },
            method="POST",
        )
    project_id = str(project["id"])

    files = client.request(f"/api/projects/{project_id}/files?page=1&pageSize=100")
    existing_names = {item["fileName"] for item in files.get("data", [])}
    for source in manifest["sources"]:
        file_name = str(source["file_name"])
        if file_name in existing_names:
            continue
        source_path = source_dir / file_name
        client.request(
            f"/api/projects/{project_id}/files",
            data=source_path.read_bytes(),
            headers={
                "Content-Type": "application/octet-stream",
                "x-file-name": urllib.parse.quote(file_name),
            },
            method="POST",
        )

    datasets = client.request(f"/api/projects/{project_id}/datasets?page=1&size=1")
    decision = decide_dataset_import(int(datasets.get("total", 0)), expected_total)
    import_result: dict[str, Any] = {"decision": decision}
    if decision == "import":
        response = client.request(
            f"/api/projects/{project_id}/datasets/import",
            payload=import_payload,
            method="POST",
        )
        import_result.update(response)

    files = client.request(f"/api/projects/{project_id}/files?page=1&pageSize=100")
    datasets = client.request(
        f"/api/projects/{project_id}/datasets?page=1&size=100&status=unconfirmed"
    )
    return {
        "project_id": project_id,
        "project_name": project_name,
        "files": int(files.get("total", len(files.get("data", [])))),
        "unconfirmed_datasets": int(datasets.get("total", len(datasets.get("data", [])))),
        "import": import_result,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-url", default="http://127.0.0.1:1717")
    parser.add_argument("--project-name", default="audio-polisher-lora-poc")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    """执行幂等导入并输出摘要。"""
    args = parse_args()
    result = import_project(
        client=EasyDatasetClient(args.easy_url, args.timeout_seconds),
        project_name=args.project_name,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
