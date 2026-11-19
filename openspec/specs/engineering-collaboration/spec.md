# 工程协作与验证纪律

## Purpose

覆盖协作语言、需求分级、三层验证纪律、行为变化回写、提交与 Python 风格约定，以及经验沉淀的边界。
本规格描述治理规则，不替代任何能力域的行为基线；OpenSpec 流程见 [openspec/README.md](../../../openspec/README.md)。

## 实现依据

- [AGENTS.md](../../../AGENTS.md)
- [CONSTITUTION.md](../../../CONSTITUTION.md)
- [docs/dev-workflow.md](../../../docs/dev-workflow.md)
- [docs/agent-memory/index.md](../../../docs/agent-memory/index.md)
- [docs/python-engineering.md](../../../docs/python-engineering.md)
- [openspec/README.md](../../../openspec/README.md)

## Requirements

### Requirement: 协作语言与事实来源固定

面向项目所有者的协作与文档 MUST 优先使用简体中文；代码标识符、命令、路径、API 字段和第三方专有名词
MUST 保留英文。项目事实 MUST 以当前源码、清单和构建脚本为准；文档与实现冲突时 MUST 先核对实现，
再更新文档，MUST NOT 按旧文档推断行为。

#### Scenario: 文档与实现冲突

- **WHEN** 文档描述与清单或源码不一致
- **THEN** 以实现为准修正文档并记录原因，不按历史文档推断当前接口

### Requirement: 需求按风险分级处理

局部修复、文案与配置微调 MUST 直接修改并验证，不建 change；新增模型 API、修改服务合同或端口 MUST
建立 OpenSpec change（proposal/design/tasks/verification）；启动编排、GPU 锁、隔离恢复等高风险机制
MUST 规格先行，先确认规则与副作用再实施。

#### Scenario: 新增模型 API

- **WHEN** 需求涉及新增模型服务或修改服务合同/端口
- **THEN** 先建立 OpenSpec change 并以清单事实为依据，再推进实现与验证

### Requirement: 三层验证严格区分

验证 MUST 分三层记录：静态核对（`bash -n`、Ruff、能力清单与服务口径校验）、无模型回归
（`uv run --project qa --locked python -m unittest discover -s tests -v`）、真实模型验证（单独串行执行）。
无模型测试通过 MUST NOT 写成真实模型验证完成；未验证项 MUST NOT 勾选，必须在 verification.md 说明。

#### Scenario: 无模型门禁通过

- **WHEN** 静态核对与无模型回归全部通过
- **THEN** 只声明无模型层完成，真实 GPU 推理仍列为未完成或另行安排

### Requirement: 不确定事项按四态记录且不得编造事实

不确定事项 MUST 按「已确认 / 临时假设 / 待对齐 / 硬阻塞」标注：已确认 MUST 记录来源；临时假设
MUST 标记影响且保持可调整、不写入主规格；待对齐 MUST 记录责任方与临时处理；硬阻塞 MUST 暂停依赖
任务。任何状态都 MUST NOT 自行编造模型能力、上游契约或硬件行为，确认后 MUST 记录结果、来源与日期
并同步受影响的文档与规格。

#### Scenario: 上游契约未确认

- **WHEN** 实现依赖未知的上游响应字段或硬件行为
- **THEN** 记为待对齐或硬阻塞并暂停依赖任务，不猜测字段或行为后写成已确认能力

### Requirement: 修改保持最小且不覆盖用户变更

修改前 MUST 先阅读相邻实现与既有文档；MUST 保持最小改动、不重构无关代码、不覆盖用户已有变更。
MUST NOT 主动提交、推送或调用外部生产系统；本仓库没有前端 Mock 环节，MUST NOT 用合成响应冒充
真实模型验证结果。

#### Scenario: 发现无关代码问题

- **WHEN** 修改过程中发现与当前需求无关的缺陷
- **THEN** 记录该问题但不顺手重构，不覆盖用户已有改动

### Requirement: 提交内容有明确边界

MUST NOT 提交权重、第三方源码、上传素材、生成音视频、训练数据、虚拟环境、缓存、密钥或机器专用
绝对路径；归档 MUST 把 change 目录移入 `openspec/changes/archive/`，同步主规格与 `coverage.md`，
MUST NOT 先归档后补文档。

#### Scenario: 清理与提交前检查

- **WHEN** 准备提交或归档一次改动
- **THEN** 确认运行数据、密钥、缓存和机器专用路径未入库，规格与 coverage.md 已同步

### Requirement: 行为变化同步文档与规格

行为或契约变化后 MUST 同步 README、AGENTS.md、`localai/manifests/capabilities.json` 与对应规格；
服务数量、路由与端口口径 MUST 由 `scripts/check_service_inventory.py` 在质量门禁中兜底核对。归档前
MUST 核对全部任务与验证记录，无法完成的真实模型验证 MUST 明确列为未完成。

#### Scenario: 修改服务合同

- **WHEN** 请求参数、响应结构或错误语义变化
- **THEN** 同步 capabilities.json、校验脚本、README/AGENTS 与规格后再归档

### Requirement: 提交与 Python 风格固定

Commit subject MUST 使用简洁 Conventional Commit（如 `feat:`、`fix:`、`docs:`）；Python 注释与
Docstring MUST 默认使用简体中文，标识符、类型名、库/API 名与通用技术术语保留英文；公共函数、公共类
与重要业务入口 MUST 提供 Google Style / PEP 257 Docstring。复杂并发、进程生命周期、GPU/CUDA 与兼容
workaround MUST 说明原因，显然的赋值与简单条件 MUST NOT 写逐行注释。

#### Scenario: 审查一次提交

- **WHEN** 提交包含新增公共函数或复杂并发逻辑
- **THEN** subject 符合 Conventional Commit，Docstring 与必要的原因注释齐全且无逐行冗余注释

### Requirement: 经验沉淀只记录已确认结论

经验沉淀 MUST 只记录确认过且有来源的结论，MUST NOT 保存敏感信息或未确认假设；历史报告、任务草稿、
SDD 历史记录与退役目录 MUST NOT 用于推断当前接口，当前能力以清单、源码与主规格为准。

#### Scenario: 引用历史方案

- **WHEN** 历史方案与当前清单或主规格冲突
- **THEN** 以清单与主规格为准，历史资料仅作背景记录，不升级为产品行为
