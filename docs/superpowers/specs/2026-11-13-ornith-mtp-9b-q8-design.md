# Ornith MTP 9B Q8 API 设计

## 目标与背景

新增 `Ornith-1.5-9B-MTP-GGUF-Q8` Chat Completions API，监听外部端口 `8504`，路由为 `/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`。模型只读加载自 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/`，使用 MTP Q8 主模型和 BF16 vision projector。保留现有 `Ornith-1.5-9B-GGUF` API，避免破坏已有 OpenCode、Easel 和脚本客户端。

镜像目录包含 `Ornith-1.5-9B-MTP-Q8_0.gguf` 与 `mmproj-Ornith-1.5-9B-BF16.gguf`。元数据标识主模型为 `qwen35`、标称上下文为 262144，projector 带 vision encoder。服务默认仍遵循仓库当前 98304 上下文上限、共享 GPU 锁、sleep-idle 和 quarantine 生命周期。

## 范围

- 新服务适配器复用 `model-text/llama_cpp_runtime`，固定路由和 backend model alias。
- `start.sh` 检查两个只读权重文件、CUDA llama-server 能力和端口；以 `--mmproj`、`--spec-type draft-mtp` 启动 backend。
- 新 API 使用对外端口 `8504`；保留旧 API `8503`，旧 backend 改用 loopback `8505`，新 MTP backend 使用 loopback `8506`。
- 两个 backend 顺序预热、独立监视和收尾，所有启动、请求、关闭继续共用同一个 `GPU_LOCK_FILE`。
- 增加无模型 API 合同测试和显式真实模型 demo；demo 支持文本请求，并允许提供图片验证 projector 路径。
- 更新项目服务清单、部署变量、API 调用示例和测试说明。

## 非目标

- 不下载、复制或提交 GGUF 权重及 projector。
- 不将任何 llama-server backend 暴露到非 loopback 地址。
- 不在无模型质量门禁中加载模型、调用 CUDA 或发出真实推理请求。
- 不改变现有 API 路由、Easel 默认聊天上游或现有模型参数。

## 约束

- API 转发必须继续通过共享 GPU 锁和 `llama_cpp_runtime` 的休眠确认、隔离标记及清理逻辑。
- 模型和 projector 均从 `$HF_MIRROR_DIR` 只读加载。
- 新模型默认上下文不超过 `98304`；MTP 草稿长度沿用现有默认值 `4`。
- `start.sh` 管理的 HTTP 监听数由 15 增至 17，其中两个 llama-server 只绑定 loopback。

## 方案

1. **新增并行 API（采用）**：保留现有 8503 API 和模型，在 8504 新增 MTP API；将两个 loopback backend 分别放在 8505、8506。它满足任务指定端口和路由，也保留已有客户端行为，代价是需要管理第二个 backend 和额外端口。
2. **替换现有 API**：将现有 API 改为新模型并搬到 8504。它只需管理一个 backend，但会移除旧服务接口或使旧模型别名指向不同模型，与新增目录和独立路由的意图不符。
3. **沿用现有端口拓扑**：在 8503 提供新路由、8504 保留为 backend。改动最少，但违反任务明确指定的外部端口 8504。

采用方案 1，因为需求定义了一个新的模型名、服务目录、路由和监听端口；保留现有 API 是符合现有部署契约的默认做法。

## 验收

- 无模型测试确认新适配器只注册指定路由、固定 model alias，并将请求送往其 loopback backend。
- 启动脚本测试确认模型与 projector 缺失会在启动 LocalAI 前失败，端口规划不冲突，两个 backend 都带正确模型参数并经过共享 runtime 启动与关闭。
- demo 默认请求新 API 的文本生成；指定本地图片时发送 OpenAI-compatible image content。输出独占写入 `storage/llm_tests/`，不覆盖已有样例。
- Shell 语法、质量门禁和 Ornith 无模型测试通过；真实模型调用单独运行，不并入质量门禁。
