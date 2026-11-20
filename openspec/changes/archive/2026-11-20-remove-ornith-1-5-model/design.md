# 设计

## Context

Ornith-1.5-9B-MTP 权重已从本机删除，但该模型的 API 适配器、专用 llama.cpp 运行时、Easel 聊天路由、
OpenCode 插件与固定上下文基准仍残留在源码、启动脚本、规格和测试中。start.sh 的预检会因缺失 GGUF 直接失败，
导致整个模型栈无法启动；保留死代码会持续污染服务口径文档与质量门禁。用户已确认：移除 local-ornith provider、
一并删除 llama_cpp_runtime、完全移除网关聊天路由，并按 OpenSpec 流程先行。

## Goals / Non-Goals

- Goals：
  - 本机不再存在任何 Ornith-1.5 服务入口、CLI 依赖或配置引用；`bash start.sh` 不再因缺失 Ornith 资产失败。
  - Easel 网关只剩图片与语音代理，服务口径文档与脚本一致（4 固定服务、19 HTTP 进程）。
  - 全局 opencode 配置不再指向不存在的本地聊天模型；OpenCode 使用其自身默认模型。
  - 无模型质量门禁全部通过，规格与 coverage 同步。
- Non-Goals：
  - 不改动十五个音频模型服务、LocalAI/audio.cpp 清单与 capabilities.json。
  - 不改动 LocalAI 与 sd.cpp 的 GPU 锁、quarantine 恢复与共享锁语义。
  - 不清理 `storage/` 运行数据、历史遥测与用户生成产物。
  - 不引入替代的本地聊天模型；不修改 Easel 自身的 Codex 执行引擎选择。

## Decisions

- **直接删除源码与目录，不做兼容 stub**：Ornith 已确定不再本地使用，保留 stub 只会留下坏接口与死配置。
  删除范围：`model-text/Ornith-1.5-9B-MTP-GGUF/`、`model-text/llama_cpp_runtime/`、
  `tests/ornith_1_5_9b_mtp_benchmarks/`、`tests/llama_cpp/`、`.opencode/plugins/local-model-context/`、
  `tests/opencode/`、`scripts/llama_cpp/` 及相关文档/提示词/准备脚本。
- **start.sh 做减法而非重写**：保留 LocalAI 注册/预检、sd.cpp 校验、audio.cpp 与图片服务启停、
  GPU 锁与 quarantine 恢复；只移除 Ornith 专属 env/端口预检/llama-server 预检/预热/backend 启停/监控。
  注意 GPU quarantine 检查与 `recover_quarantine.py` 仍服务于其他 GPU provider，必须保留。
- **网关聊天路由整体移除**：从 `easel_bridge/app.py` 删除 chat factory 参数、模型 ID、默认 URL、
  `/v1/chat/completions`、`/v1/models` 中的聊天条目与 `TelemetryMiddleware` 的 chat 路径；
  保留 Bearer 鉴权、图片与语音代理合同不变。
- **隔离恢复按 provider 收敛**：`recover_quarantine.py` 删除 llama.cpp 分支与其对 `model-text` 的 sys.path
  依赖；保留 LocalAI 与 stable-diffusion.cpp 分支。旧的 llama.cpp quarantine 标记（若有）需人工处理，
  但本模型已下线且正常退出不产生标记，不作为阻塞。
- **遥测去角色保留框架**：`telemetry_resources.py` 删除 `ORNITH_MTP_9B` 配置快照、模型文件记录、
  backend 角色推断与 MTP 原生解析；`telemetry_brief.py` 删除 `ornith_backend` 专属峰值字段。
  资源/请求/测试标记采集继续服务其余模型。
- **规格与口径同步**：移除 `ornith-text-model` 与 `opencode-context-and-benchmark` 两个能力域主规格，
  在 change delta 中声明 REMOVED；其余受影响规格按实际行为 MODIFIED；
  `check_service_inventory.py` 固定服务去掉 8504/8506，README/AGENTS 同步到 19 进程；
  `scripts/quality_gate.sh` 的 lint 清单去掉两个已删目录。
- **全局 opencode 配置（仓库外）**：删除 `local-ornith` provider、`model` 默认模型字段，
  保留 compaction 通用设置；网关 Key 文件 `easel-bridge-api-key` 仍由图片/语音配置保留。

## Risks / Trade-offs

- [历史 llama.cpp quarantine 标记残留可能导致 start.sh 预检失败] → 移除的是 llama.cpp provider 恢复分支；
  正常关闭不产生该标记。若存在遗留标记，用户需手动删除 `.quarantine`（一次性人工处理），不引入第二套恢复工具。
- [测试目录 `test_start_services.py` 与 `test_localai_active_scope.py` 与 Ornith 强耦合] → 按新 start.sh 行为重写断言：
  不再注入 llama-server/GGUF/mmproj 夹具与 ORNITH_* 变量，改验证「不出现 Ornith/llama 调用」与其余服务完整启动。
- [用户的 README/model-text README/coverage 已有未提交改动] → 在这些改动基础上继续编辑，不整体覆盖文件。
- [真实模型验证无法执行] → 本次为纯下线变更，无真实 GPU 推理需求；验证集中在静态核对与无模型回归，
  verification.md 明确标注未执行真实模型验证。
