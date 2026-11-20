# 移除 Ornith-1.5-9B-MTP 模型与配套链路

## Why

本机已删除 `model-text/Ornith-1.5-9B-MTP-GGUF` 模型权重，不再本地使用 Ornith-1.5 模型。
当前源码仍保留该服务适配器、专用 llama.cpp 运行时、Easel 网关聊天路由、OpenCode 插件与固定上下文基准、
以及一整套启动/关闭/遥测/测试接线。继续保留会产生无法启动的死服务（start.sh 预检会因缺失 GGUF 直接失败）、
过期的服务口径文档和失效的本地 Provider 配置。依据 AGENTS.md「需求分级」，下线模型服务属于需规格先行的变更。

已确认依据：
- `localai/manifests/audio-cpp-models.json`：十五个音频候选不含任何 Ornith 条目；Ornith 从未注册为 LocalAI 候选。
- `scripts/check_service_inventory.py` 的 `FIXED_HTTP_SERVICES` 含 8504/8506 两个固定 HTTP 服务，
  合计 21 个 HTTP 进程；移除后应为 4 个固定服务、19 个 HTTP 进程。
- `start.sh`：Ornith backend 由 `python3 -m llama_cpp_runtime.runtime` 启停，预检要求 GGUF/mmproj/CUDA llama-server。
- `easel_bridge/app.py`：`/v1/chat/completions` 的唯一上游是 8504 的 Ornith 适配器。
- `~/.config/opencode/opencode.json`：默认模型为 `local-ornith/Ornith-1.5-9B-MTP-GGUF-Q8`，无替代本地聊天模型。

## What Changes

- 删除 `model-text/Ornith-1.5-9B-MTP-GGUF/` 服务目录与 `model-text/llama_cpp_runtime/` 共享运行时目录。
- 从 `start.sh` 删除 Ornith 端口/dir/env/预检/预热/启停/监控接线，不再要求 llama-server、GGUF、mmproj 与 sleep-cache 补丁。
- 从 `easel_bridge` 删除 `/v1/chat/completions`、`CHAT_MODEL_ID`、`chat_url`、`EASEL_BRIDGE_MTP_CHAT_URL`
  及聊天遥测路径；网关只保留图片与语音代理。
- 删除 llama.cpp 专用隔离恢复分支（含 `recover_quarantine.py` 中 llama.cpp 逻辑），
  保留 LocalAI 与 sd.cpp 隔离恢复；`llm_local_runtime` 遥测去掉 `ornith_backend`/`ornith_adapter`
  角色、Ornith 配置快照与 MTP 原生日志解析。
- 删除 OpenCode 本地插件 `.opencode/plugins/local-model-context/`、上下文基准提示词/准备脚本与对应测试，
  并删除全局 `~/.config/opencode/opencode.json` 中的 `local-ornith` provider 与默认模型字段（仓库外一次性配置）。
- 删除 Ornith/llama.cpp 专属规格、`docs/` 与历史调研中仅描述该模型的文档、基准结果归档入口与测试/样例目录。
- 同步服务口径：`check_service_inventory.py` 固定服务去掉 8504/8506；README、AGENTS.md、
  `scripts/quality_gate.sh` lint 清单、`openspec/coverage.md` 与主规格同步到 19 个 HTTP 进程口径。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-overview`：服务总数与固定服务口径从 21 调整为 19，移除 Ornith 条目。
- `service-lifecycle-and-startup`：删除「Ornith 预热先于全部 API」要求与 llama.cpp backend 启停语义。
- `easel-bridge-gateway`：删除聊天路由，网关只代理 MOSS 图片与 VoxCPM2 语音。
- `model-telemetry`：删除 Ornith 后端角色、Ornith 配置快照与 MTP 原生日志解析口径；遥测框架保留。
- `gpu-lock-and-quarantine`：删除 llama.cpp provider 的隔离恢复分支，LocalAI 与 sd.cpp 不变。
- `quality-gate-and-testing`：Ruff lint 清单与真实样例入口移除 Ornith/llama.cpp 目录；无模型回归继续保留。

### Removed Capabilities

- `ornith-text-model`：整个能力域下线（主规格与 change delta 均移除）。
- `opencode-context-and-benchmark`：整个能力域下线（主规格与 change delta 均移除）。

## Impact

- 代码/目录：删除 13 个源码/测试/插件目录与 10 余个文档/脚本文件；修改 start.sh、easel_bridge、
  recover_quarantine.py、telemetry_resources.py、telemetry_brief.py、多个测试文件与规格文档。
- 服务口径：固定 HTTP 服务 6→4，start.sh 管理 HTTP 进程 21→19；音频服务 15 不变；LocalAI 候选 15 不变。
  路由 8504/8506 与聊天路由 `/v1/chat/completions` 不再存在。
- 不确定事项：
  - 已确认：模型权重已由用户删除；无替代本地聊天模型；用户确认移除 provider 与聊天路由、一并删除共享运行时、先建 change。
  - 待对齐：全局 opencode 配置属仓库外用户配置，本次按确认直接修改 `~/.config/opencode/opencode.json`；
    若用户使用 Windows 原生 OpenCode，需自行同步对应配置与 Key 文件。
  - 硬阻塞：无。
- 边界：不删除任何音频服务、不改动 GPU 锁/LocalAI/sd.cpp 既有隔离机制、不清理运行数据目录（storage/）。
  历史 SDD 记录 `docs/superpowers/` 按 AGENTS.md 保留为历史，仅移除其中纯属 Ornith 接入方案的文件。
