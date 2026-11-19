# 项目总览

## Purpose

覆盖仓库整体能力口径（服务数量、端口分配、事实来源层级）与全部能力域规格的导航入口。
本规格是维护者与 agent 的第一入口；行为变化 MUST 同步对应能力域规格。

## 实现依据

- [README.md](../../../README.md)
- [AGENTS.md](../../../AGENTS.md)
- [localai/manifests/audio-cpp-models.json](../../../localai/manifests/audio-cpp-models.json)
- [start.sh](../../../start.sh)
- [scripts/check_service_inventory.py](../../../scripts/check_service_inventory.py)

## Requirements

### Requirement: 服务口径与清单一致

仓库维护十五个音频模型服务（十四个 audio.cpp 服务共用共享 runtime，VoiceDesign 使用独立 uv 项目）、
一个 Ornith MTP 文本模型 API、MOSS 与 FLUX.3 Klein 两个图片 API、一个轻量控制面、Steam Audio CPU
renderer 和 Easel/OpenCode API bridge；start.sh 管理二十一个 HTTP 进程（含仅绑定 loopback 的
Ornith llama.cpp backend）；LocalAI 固定版本由 `scripts/localai/manager.py` 在 127.0.0.1:8400 单独
管理，MUST NOT 计入上述进程数。服务数量、路由与端口口径 MUST 与清单一致，由
`scripts/check_service_inventory.py` 在质量门禁中核对。

#### Scenario: 查询当前服务清单

- **WHEN** 维护者阅读 README、AGENTS.md 或主规格确认服务口径
- **THEN** 三处与 `localai/manifests/audio-cpp-models.json` 及 start.sh 一致，否则门禁失败

#### Scenario: 新增或退役模型服务

- **WHEN** 清单、start.sh 或端口分配变化
- **THEN** README 服务表、AGENTS.md 服务表与对应能力域规格同步更新后归档

### Requirement: 事实来源分层固定

项目事实 MUST 按以下层级取用：模型清单与构建脚本（`localai/manifests/`、`sdcpp/manifests/`、
`start.sh`）是服务事实唯一权威；README 与 docs/ 描述部署与机制；AGENTS.md 维护协作规则；
openspec/specs/ 维护行为基线。文档与实现冲突时 MUST 先核对实现，再更新文档。

#### Scenario: 文档与源码冲突

- **WHEN** 文档描述与清单或源码行为不一致
- **THEN** 以实现为准修正文档并记录原因，不按旧文档推断行为

### Requirement: 能力域导航保持可用

每个能力域 MUST 有对应主规格且链接可达，完整清单见下方「能力域清单」；维护者 MUST 能通过
源码归属矩阵（[openspec/coverage.md](../../coverage.md)）定位实现依据。

#### Scenario: 定位某个服务的约束

- **WHEN** 维护者需要确认某服务的当前行为基线
- **THEN** 通过源码归属矩阵（[openspec/coverage.md](../../coverage.md)）找到对应规格与实现依据

#### Scenario: 区分服务能力与治理约定

- **WHEN** 询问遥测、Dagu 调度、音乐后处理或协作纪律是否属于当前能力
- **THEN** 按能力域导航定位对应主规格，服务事实仍以清单为准，治理约定不冒充已实现工具能力

### Requirement: 历史资料不推断当前能力

历史报告、任务草稿（task*.md）、SDD plans/ledgers 与已退役目录 MUST NOT 用于推断当前接口；
当前能力以清单、源码与主规格为准。

#### Scenario: 引用历史文档

- **WHEN** 历史方案与当前清单冲突
- **THEN** 以清单与主规格为准，历史文档仅作背景记录

## 能力域清单

服务与机制域：

- [audio.cpp 音频模型服务](../audio-cpp-services/spec.md)
- [Ornith 文本模型](../ornith-text-model/spec.md)
- [图片服务](../image-sdcpp-services/spec.md)
- [控制面与空间渲染](../control-plane-and-spatial-render/spec.md)
- [GPU 锁与隔离](../gpu-lock-and-quarantine/spec.md)
- [上传与保留策略](../storage-retention-and-upload/spec.md)
- [质量门禁与测试](../quality-gate-and-testing/spec.md)
- [Easel 网关](../easel-bridge-gateway/spec.md)
- [服务生命周期与启动](../service-lifecycle-and-startup/spec.md)

运行与治理域：

- [模型性能记录](../model-telemetry/spec.md)
- [OpenCode 上下文插件与固定上下文基准](../opencode-context-and-benchmark/spec.md)
- [Dagu 独立作业调度](../dagu-job-worker/spec.md)
- [音乐后处理编排规则](../music-postprocessing/spec.md)
- [清单版本锁定与外部二进制核验](../manifest-version-pinning/spec.md)
- [工程协作与验证纪律](../engineering-collaboration/spec.md)
