"""使用隔离配置和假模型验证 OpenCode 插件，不调用真实模型。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OpenCodeContextGuardTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "需要 Node 执行 OpenCode 插件单元测试")
    def test_plugin_hooks(self):
        result = subprocess.run(
            ["node", "--test", str(ROOT / "tests/opencode/context-guard.test.mjs")],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("opencode"), "需要本机 OpenCode 进行无模型集成验证")
    def test_installed_opencode_compacts_and_continues_with_fake_model(self):
        requests = []
        summary = "## Objective\n- Test context handling.\n## Requirements\n- Read in pages.\n## Decisions\n- Keep 32K.\n## Work State\n### Completed\n- Read start.sh lines 1-150.\n### Active\n- Finish.\n### Blocked\n- (none)\n## Next Move\n1. Reply done.\n## Relevant Files\n- start.sh\n## Important Context\n- Fake model only."

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                system = json.dumps(
                    [m for m in body.get("messages", []) if m.get("role") == "system"]
                )
                compact = "checkpoint, not performing" in system
                tools = any(m.get("role") == "tool" for m in body.get("messages", []))
                first = not any(
                    any(m.get("role") == "tool" for m in item.get("messages", []))
                    for item in requests[:-1]
                )
                if compact:
                    content, finish, usage = {"content": summary}, "stop", 1000
                elif tools or not first:
                    content, finish, usage = {"content": "done"}, "stop", 1000
                else:
                    content, finish, usage = (
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "read-one",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": json.dumps(
                                            {"path": str(project / "start.sh")}
                                        ),
                                    },
                                }
                            ]
                        },
                        "tool_calls",
                        30000,
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for payload in [
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "Ornith-1.5-35B-A3B-AD-Q5",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", **content},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "Ornith-1.5-35B-A3B-AD-Q5",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        "usage": {
                            "prompt_tokens": usage,
                            "completion_tokens": 40,
                            "total_tokens": usage + 40,
                        },
                    },
                ]:
                    self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")

        with tempfile.TemporaryDirectory() as root:
            temporary = Path(root)
            project = temporary / "project"
            project.mkdir()
            # 验证与真实仓库相同的自动发现路径，不依赖全局配置或显式 plugins 条目。
            shutil.copytree(
                ROOT / ".opencode/plugins/local-model-context",
                project / ".opencode/plugins/local-model-context",
            )
            (project / "start.sh").write_text(
                "".join(f"# line {index}\n" for index in range(1, 221))
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                config_dir = temporary / "config/opencode"
                config_dir.mkdir(parents=True)
                config = {
                    "model": "local-ornith/Ornith-1.5-35B-A3B-AD-Q5",
                    "providers": {
                        "local-ornith": {
                            "package": "@opencode/ai/providers/openai-compatible",
                            "settings": {
                                "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                                "apiKey": "fake",
                            },
                            "models": {
                                "Ornith-1.5-35B-A3B-AD-Q5": {
                                    "limit": {"context": 32768, "output": 4096},
                                    "capabilities": {
                                        "tools": True,
                                        "input": ["text"],
                                        "output": ["text"],
                                    },
                                }
                            },
                        }
                    },
                    "compaction": {"auto": True, "buffer": 8192, "keep": {"tokens": 4096}},
                }
                (config_dir / "opencode.json").write_text(json.dumps(config))
                environment = {
                    **os.environ,
                    "XDG_CONFIG_HOME": str(temporary / "config"),
                    "XDG_DATA_HOME": str(temporary / "data"),
                    "XDG_STATE_HOME": str(temporary / "state"),
                    "XDG_CACHE_HOME": str(temporary / "cache"),
                }
                for key in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "OPENCODE_CONFIG_DIR"):
                    environment.pop(key, None)
                result = subprocess.run(
                    [
                        "opencode",
                        "run",
                        "--standalone",
                        "--auto",
                        "--format",
                        "json",
                        "CONTEXT_BENCHMARK_PROMPT.md benchmark: read start.sh then reply done.",
                    ],
                    cwd=project,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=50,
                )
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-5000:])
                compact = [
                    body
                    for body in requests
                    if "checkpoint, not performing"
                    in json.dumps(
                        [m for m in body.get("messages", []) if m.get("role") == "system"]
                    )
                ]
                self.assertTrue(compact, (result.stdout + result.stderr)[-5000:])
                self.assertTrue(
                    all(
                        body.get("chat_template_kwargs", {}).get("enable_thinking") is False
                        for body in compact
                    )
                )
                self.assertTrue(
                    all(
                        0 < body.get("max_tokens", body.get("max_completion_tokens", 0)) <= 2048
                        for body in compact
                    )
                )
                self.assertTrue(any("lines 1-150" in json.dumps(body) for body in requests))
                self.assertIn("done", result.stdout)
                self.assertNotIn(
                    "did not match the required template", result.stdout + result.stderr
                )
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
