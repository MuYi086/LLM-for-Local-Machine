# 任务

## 1. 建立 change 与范围确认

- [x] 1.1 建立本 change 四件套（proposal/design/tasks/verification）与规格 delta，记录用户已确认的四项决策。

## 2. 删除模型服务与专用运行时

- [x] 2.1 删除 `model-text/Ornith-1.5-9B-MTP-GGUF/`（app.py、main.py、缓存）与 `model-text/llama_cpp_runtime/`。
- [x] 2.2 删除 `tests/ornith_1_5_9b_mtp_benchmarks/`、`tests/llama_cpp/`、`scripts/llama_cpp/`。
- [x] 2.3 删除 `model-text/README.md` 中 Ornith 段落，仅保留训练实验说明；按现状重写该文件。

## 3. 收缩 start.sh 与隔离恢复

- [x] 3.1 从 `start.sh` 删除 Ornith 目录/env/端口预检/llama-server 与 GGUF/mmproj 预检/预热启停/监控/清理接线，
      保留 LocalAI、sd.cpp、audio.cpp、图片服务、GPU 锁与 quarantine 恢复。
- [x] 3.2 从 `scripts/localai/recover_quarantine.py` 删除 llama.cpp provider 恢复分支与 `model-text` sys.path 依赖。
- [x] 3.3 从 `scripts/quality_gate.sh` 的 lint 清单删除两个已删目录。
- [x] 3.4 从 `scripts/check_service_inventory.py` 的 FIXED_HTTP_SERVICES 删除 8504/8506。

## 4. 网关与遥测收敛

- [x] 4.1 从 `easel_bridge/app.py` 删除聊天路由、模型 ID、URL/工厂参数、遥测 chat 路径；保留图片与语音合同。
- [x] 4.2 从 `llm_local_runtime/telemetry_resources.py` 删除 Ornith 配置/模型文件快照、backend 角色与 MTP 日志解析。
- [x] 4.3 从 `llm_local_runtime/telemetry_brief.py` 删除 `ornith_backend` 专属峰值字段。

## 5. OpenCode 配置与文档清理

- [x] 5.1 删除 `.opencode/plugins/local-model-context/`、`tests/opencode/`、`tests/test_opencode_context_guard.py`、
      `CONTEXT_BENCHMARK_PROMPT.md`、`scripts/prepare_context_benchmark.py`。
- [x] 5.2 修改仓库外 `~/.config/opencode/opencode.json`：移除 `local-ornith` provider 与默认 `model` 字段。
- [x] 5.3 删除仅描述 Ornith/llama.cpp 的文档与历史方案文件，更新 `docs/easel-bridge.md`、
      `docs/model-telemetry.md`、`docs/dev-workflow.md`、`docs/agent-memory/decisions.md`、
      `openspec/coverage.md`、`openspec/tools/template/tasks.md` 中的相关引用。
- [x] 5.4 更新 `README.md` 与 `AGENTS.md`：删除 Ornith 服务表行、API 示例、env 说明与 21 进程口径，改为 19 进程。

## 6. 测试同步

- [x] 6.1 重写 `tests/test_start_services.py`：移除 llama-server/GGUF/ORNITH_* 夹具与 Ornith 断言，
      验证 start.sh 不再启动或预检 Ornith，且其余服务与 19 进程口径完整。
- [x] 6.2 更新 `tests/test_localai_active_scope.py`：移除 llama_runtime 导入与 llama.cpp 隔离恢复相关用例。
- [x] 6.3 更新 `tests/test_easel_bridge.py`：移除聊天路由/CHAT_URL/chat 模型断言，验证 404 与图片/语音合同不变。
- [x] 6.4 更新 `tests/test_model_telemetry.py`、`tests/test_telemetry_brief.py`：移除 Ornith 服务/角色断言。
- [x] 6.5 确认 `tests/test_service_inventory_check.py` 通用口径仍通过（固定服务数量由脚本推导，无需硬编码 21）。

## 7. 验证与收敛

- [x] 7.1 静态核对：`bash -n start.sh`、`bash scripts/quality_gate.sh`（含 Ruff、能力清单与服务口径校验）。
- [x] 7.2 无模型回归：`uv run --project qa --locked python -m unittest discover -s tests -v` 全部通过。
- [x] 7.3 OpenSpec strict 校验：`npx -y @fission-ai/openspec validate --all --strict`。
- [x] 7.4 真实模型验证：本次为下线变更，无真实推理需求；如实记录未执行。
- [x] 7.5 归档 change，同步主规格（删除两个能力域、MODIFIED 其余）与 `openspec/coverage.md`，
      并把 verification.md 的实际结果回写到归档记录。

> 无模型回归通过不代表真实模型验证完成；本次无真实模型验证项，保持如实记录。
