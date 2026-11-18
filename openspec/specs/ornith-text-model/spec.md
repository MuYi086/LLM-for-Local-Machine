# Ornith MTP 9B 文本模型服务

## Purpose

覆盖 Ornith-1.5-9B-MTP-GGUF-Q8 API（8504）与仅绑定 loopback 的 llama.cpp backend（8506）的
转发关系、权重只读加载、默认推理参数、跨 sleep 缓存约束，以及 OpenCode 压缩请求的上下文预算语义。

## 实现依据

- [model-text/Ornith-1.5-9B-MTP-GGUF/app.py](../../../model-text/Ornith-1.5-9B-MTP-GGUF/app.py)
- [model-text/llama_cpp_runtime/runtime.py](../../../model-text/llama_cpp_runtime/runtime.py)
- [model-text/llama_cpp_runtime/sleep_cache.py](../../../model-text/llama_cpp_runtime/sleep_cache.py)
- [model-text/llama_cpp_runtime/context_budget.py](../../../model-text/llama_cpp_runtime/context_budget.py)
- [docs/llama-cpp-sleep-cache.md](../../../docs/llama-cpp-sleep-cache.md)
- [docs/llama-cpp-sleep-guard.md](../../../docs/llama-cpp-sleep-guard.md)
- [docs/opencode-compaction-budget.md](../../../docs/opencode-compaction-budget.md)
- [docs/opencode-context.md](../../../docs/opencode-context.md)
- [tests/ornith_1_5_9b_mtp_benchmarks/performance.md](../../../tests/ornith_1_5_9b_mtp_benchmarks/performance.md)

## Requirements

### Requirement: API 转发到 loopback backend

Ornith MTP API（8504，`POST /v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`）MUST 把 OpenAI Chat
Completions 请求转发到仅绑定 127.0.0.1:8506 的 llama-server `/v1/chat/completions`，
自身不直接加载模型。

#### Scenario: 正常聊天请求

- **WHEN** 客户端提交 Chat Completions 请求
- **THEN** 适配器在共享 GPU 租约内转发请求并透传流式响应，不在 API 进程内推理

### Requirement: 权重只读加载且不注册候选

MTP Q8 主模型与 BF16 vision projector MUST 从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/`
只读加载（可用 `ORNITH_*` 环境变量覆盖位置）；MUST NOT 注册为 LocalAI 候选，MUST NOT 提交模型文件。

#### Scenario: 镜像目录缺权重

- **WHEN** 启动预检发现主模型或 projector 缺失
- **THEN** 启动失败并报告缺失路径，不回退到错误精度的权重

### Requirement: 默认推理参数保持固定口径

Ornith MTP API 默认使用 98304（96k）上下文、OpenCode buffer=16384、keep.tokens=10240；
backend 使用主 GGUF 内置 MTP、草稿长度 4、主/草稿 KV 为 F16、logical/physical batch 1024、
线程 6/8、load-mode none，MUST NOT 依赖独立草稿权重。

#### Scenario: 按默认参数启动

- **WHEN** start.sh 以默认环境变量预热 backend
- **THEN** backend 以上述参数加载，无需额外草稿权重文件

#### Scenario: 部署覆盖参数

- **WHEN** 部署通过环境变量调整上下文或线程参数
- **THEN** 实际生效值以启动输出为准，文档不把旧默认值当作当前事实

### Requirement: 跨 sleep 缓存有补丁与预算约束

backend 跨 sleep 缓存默认预算为 4096 MiB CPU RAM；MUST 部署
[llama-cpp-sleep-cache 后端补丁](../../../docs/llama-cpp-sleep-cache.md)，启动时核验实际能力；
缓存 MUST NOT 跨进程重启，超限回退冷算。

#### Scenario: 补丁能力核验

- **WHEN** backend 未打 sleep 缓存补丁或能力核验失败
- **THEN** 启动报告能力缺失并按冷算口径运行，不静默声称缓存生效

#### Scenario: 缓存超限

- **WHEN** 请求上下文超过缓存预算
- **THEN** 该轮回退冷算，不因缓存缺失失败

### Requirement: 压缩请求保持上下文预算语义

OpenCode 压缩请求的 `llm_context_guard` MUST 由适配器剥离；模板/分词 MUST 在既有 GPU 租约内执行，
按实际 backend 容量保留申请输出和安全余量；超限 MUST 返回标准
`context_length_exceeded`，MUST NOT 静默缩短摘要；普通请求 MUST NOT 增加预检调用。

#### Scenario: 压缩请求超限

- **WHEN** 压缩请求的模板加输出预算超过 backend 容量
- **THEN** 返回标准 context_length_exceeded 错误，客户端可感知并调整

#### Scenario: 普通请求

- **WHEN** 客户端发起非压缩聊天请求
- **THEN** 适配器不追加额外预检调用，请求延迟与基线一致

### Requirement: 启动预热与退出确认

start.sh MUST 通过共享 GPU 锁预热 Ornith MTP backend，确认模型进入 sleep-idle 后再启动控制面和
全部 API；任一服务退出时清理全部服务进程组并确认 backend 停止；停止状态不能确认时保留
quarantine 标记。

#### Scenario: 正常启动

- **WHEN** start.sh 预热阶段完成
- **THEN** backend 处于 sleep-idle，控制面与 API 才开始监听

#### Scenario: 异常退出

- **WHEN** backend 停止状态不能确认
- **THEN** 保留 quarantine 标记并按恢复流程处理，不假设进程已退出
