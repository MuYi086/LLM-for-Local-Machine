# OpenSpec 规格体系

本目录是仓库的规格驱动开发（spec-driven）基线：主规格描述当前行为，change 跟踪进行中的变更，
归档 change 记录已收敛的历史变更。协作规则与需求分级见根目录 [AGENTS.md](../AGENTS.md)，
具体操作见 [开发工作流](../docs/dev-workflow.md)。

## 目录结构

| 路径 | 职责 |
| --- | --- |
| `config.yaml` | OpenSpec schema、项目上下文与各类 artifact 的写作规则 |
| `specs/<能力域>/spec.md` | 主规格：当前已确认的行为基线（Purpose / 实现依据 / Requirements / Scenario） |
| `changes/<change-id>/` | 进行中的变更：proposal → design → tasks → verification → specs delta |
| `tools/template/` | 四类 artifact 的写作模板（放在 changes/ 之外，避免被 CLI 识别为活动 change） |
| `changes/archive/<change-id>/` | 已归档变更，保留四件套供追溯 |
| `coverage.md` | 全部源码目录到规格的归属矩阵 |

## change 生命周期

1. **建立**：新增模型 API、修改服务合同或端口、高风险机制变更时，从 `openspec/tools/template/` 复制
   四件套，以 `<日期>-<名称>` 命名目录（如 `2026-11-19-xxx`）。涉及行为变化的 change 还需在
   `specs/<能力域>/spec.md` delta 中用 `## ADDED/MODIFIED/REMOVED Requirements` 头描述基线变化；
   纯工具、文档类 change 在 `.openspec.yaml` 写 `schema: spec-driven` 与 `skip_specs: true`
   显式声明零 delta。
2. **推进**：按 tasks 拆分实施；verification.md 记录每轮验证，未验证项不勾选，不确定事项用
   「已确认 / 临时假设 / 待对齐 / 硬阻塞」标注。
3. **收敛**：全部任务完成后运行无模型质量门禁；涉及真实模型的任务单独串行执行并如实记录。
4. **归档**：把 change 目录移入 `changes/archive/`，同步主规格（`openspec/specs/`）与
   `coverage.md`；服务口径由 `scripts/check_service_inventory.py` 在门禁中兜底核对。

## 主规格维护规则

- 主规格只写已确认的行为；行为变化时更新对应 Requirement/Scenario，不新增重复规格。
- 每个能力域的 `实现依据` 必须指向真实存在的源码或文档路径；引用失效视为规格漂移。
- 服务数量、路由与端口不在主规格中复述清单数值的地方口径出现冲突时，以
  `localai/manifests/` 清单为准（project-overview 规格声明了该层级）。

## 校验

- 仓库原生门禁：`bash scripts/quality_gate.sh`（含 `scripts/check_service_inventory.py` 与
  `scripts/localai/validate_capabilities.py`），不依赖 Node.js，是 CI 的唯一门禁。
- OpenSpec CLI（npm 包 `@fission-ai/openspec`，可用 `npx -y @fission-ai/openspec` 免安装运行）：
  - `openspec validate --all --strict` 校验主规格与活动 change（2026-11-19 实测 16/16 通过，主规格覆盖 16 个能力域）；
  - `openspec validate --all --strict --archived` 额外核对归档 change 任务全部完成；
  - `openspec list --specs` 查看规格清单，`openspec show <名称>` 查看单个规格或 change，
    `openspec doctor` 检查根配置健康。
- 主规格与 change 的格式遵循 spec-driven schema（`### Requirement:` 必须含 MUST 且至少一个
  `#### Scenario:` 的 WHEN/THEN）；规格或 change 落地后建议执行上述 strict 校验再提交。
