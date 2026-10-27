"""阻塞式 FastAPI 处理函数的回归检查。

模型请求可能启动一次性 worker、获取共享 GPU 文件锁，或执行阻塞式上游
HTTP 调用。FastAPI 必须在线程池中执行这些处理函数，避免它们阻塞事件循环，
导致健康检查或上传接口无法响应。
"""

from __future__ import annotations

# 静态检查阻塞路由是否交给线程池，避免模型请求卡住健康检查和上传。
import ast
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def route_handler_kinds(source_path: Path) -> dict[str, str]:
    """返回每个带装饰器的应用路由是同步函数还是异步函数。"""
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    handlers: dict[str, str] = {}

    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in {"get", "post"}:
                continue
            route = decorator.args[0]
            if isinstance(route, ast.Constant) and isinstance(route.value, str):
                handlers[route.value] = type(node).__name__
    return handlers


class FastApiExecutionModelTests(unittest.TestCase):
    """确保阻塞任务不占用 FastAPI 事件循环，同时不改变 API 契约。"""

    def test_blocking_routes_use_sync_handlers(self) -> None:
        expected_routes = {
            "main/main.py": {"/v1/health", "/v1/check/audio"},
            "model-audio/clone_qwen3-tts-1.7b-base/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/qwen/clone",
            },
            "model-audio/clone_voxcpm-2/main.py": {"/v1/health", "/v1/check/audio"},
            "model-audio/music_ace-step-1.5/main.py": {
                "/v1/health",
                "/v1/aceStep/bgm",
            },
            "model-audio/design_qwen3-tts-1.7b-voicedesign/main.py": {"/v1/health"},
            "model-audio/clone_confucius-4/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/confucius4TTS/generate",
            },
            "model-audio/separate_tiger-dnr/main.py": {"/v1/health"},
            "model-video/HunyuanVideo-1.5-Comfy/main.py": {
                "/v1/health",
                "/v1/check/image",
                "/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy",
            },
        }

        for relative_path, routes in expected_routes.items():
            with self.subTest(source=relative_path):
                handler_kinds = route_handler_kinds(REPOSITORY_DIR / relative_path)
                self.assertEqual(
                    {route: handler_kinds.get(route) for route in routes},
                    {route: "FunctionDef" for route in routes},
                )

        voxcpm2_source = (REPOSITORY_DIR / "model-audio/clone_voxcpm-2/main.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "return await run_in_threadpool(synthesize_voxcpm2_payload, data)", voxcpm2_source
        )

    def test_async_upload_routes_offload_filesystem_work(self) -> None:
        upload_sources = {
            "main/main.py",
            "model-audio/clone_qwen3-tts-1.7b-base/main.py",
            "model-audio/clone_voxcpm-2/main.py",
            "model-audio/clone_confucius-4/main.py",
            "model-video/HunyuanVideo-1.5-Comfy/main.py",
        }

        for relative_path in upload_sources:
            with self.subTest(source=relative_path):
                source = (REPOSITORY_DIR / relative_path).read_text(encoding="utf-8")
                upload_handler = (
                    "store_uploaded_image"
                    if relative_path == "model-video/HunyuanVideo-1.5-Comfy/main.py"
                    else "store_uploaded_audio"
                )
                self.assertIn(
                    f"return await run_in_threadpool({upload_handler},",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
