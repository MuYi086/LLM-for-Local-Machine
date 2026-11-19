# 无模型质量门禁与测试边界

## Purpose

覆盖无模型质量门禁的组成、CI 执行、测试边界（不下载权重、不依赖 CUDA、不调用真实模型）、
真实模型验证的独立入口，以及 uv 依赖可复现要求。

## 实现依据

- [scripts/quality_gate.sh](../../../scripts/quality_gate.sh)
- [scripts/check_service_inventory.py](../../../scripts/check_service_inventory.py)
- [scripts/localai/validate_capabilities.py](../../../scripts/localai/validate_capabilities.py)
- [qa/pyproject.toml](../../../qa/pyproject.toml)
- [ruff.toml](../../../ruff.toml)
- [.github/workflows/quality.yml](../../../.github/workflows/quality.yml)
- [docs/python-engineering.md](../../../docs/python-engineering.md)
- [docs/model-telemetry.md](../../../docs/model-telemetry.md)
- [docs/opencode-compaction-budget.md](../../../docs/opencode-compaction-budget.md)
- [scripts/model_telemetry.py](../../../scripts/model_telemetry.py)
- [config/model-telemetry.env](../../../config/model-telemetry.env)
- [llm_local_runtime/src/llm_local_runtime/telemetry.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_brief.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_brief.py)

## Requirements

### Requirement: 无模型质量门禁可重复执行

质量门禁 MUST 只依赖无模型检查：`bash -n start.sh`、Steam Audio core 测试、四个 uv 项目
（main、VoiceDesign、audio_cpp_runtime、qa）的 `uv lock --check`、覆盖全部维护目录的 Ruff
check/format、`scripts/localai/validate_capabilities.py` 能力清单校验、
`scripts/check_service_inventory.py` 服务口径校验，以及 `uv run --project qa --locked python -m unittest
discover -s tests -v` 无模型回归。

#### Scenario: 本地执行门禁

- **WHEN** 维护者按仓库命令执行 `bash scripts/quality_gate.sh`
- **THEN** 上述检查全部运行，任一失败即以非零退出终止

#### Scenario: 服务口径漂移

- **WHEN** 清单新增模型但 README、AGENTS.md 或 start.sh 未同步
- **THEN** `scripts/check_service_inventory.py` 报告缺失的路由、端口或过期数量声明，门禁失败

### Requirement: CI 与本地共用同一门禁

CI MUST 在 push 与 PR 时执行 `scripts/quality_gate.sh`，与本地门禁保持同一口径。

#### Scenario: 提交引入回归

- **WHEN** push 或 PR 触发 CI
- **THEN** CI 运行与本地相同的无模型门禁，结果一致

### Requirement: 测试不触碰真实模型

仓库测试 MUST NOT 下载权重、依赖 CUDA、调用外部模型服务或执行真实模型。

#### Scenario: 测试环境缺少 GPU

- **WHEN** 在无 CUDA 设备的环境运行 `python -m unittest discover -s tests`
- **THEN** 全部测试可执行并通过，不因缺少权重或 GPU 失败

### Requirement: 真实模型验证单独串行

真实模型生成 MUST 独立于无模型门禁单独串行执行：音频样例用
`tests/audio_cpp/generate_examples.py`，Ornith 文本样例用
`tests/ornith_1_5_9b_mtp_benchmarks/generate_example.py`，输出放入运行数据目录。

#### Scenario: 执行真实样例

- **WHEN** 维护者安排真实模型验证
- **THEN** 样例按共享 GPU 锁串行执行并输出到运行数据目录，结果记录不并入无模型回归

### Requirement: 依赖变化保持锁一致

依赖变化后 MUST 在对应 uv 项目执行 `uv lock`，提交 pyproject 与锁文件，并用
`uv lock --check` 验证。

#### Scenario: 更新项目依赖

- **WHEN** 任一维护项目的依赖声明变化
- **THEN** 锁文件同步更新并通过 `uv lock --check`，否则门禁失败

### Requirement: 基准归档使用简报

基准归档 MUST 使用 `end --attach-result --brief` 收尾，简报 stdout MUST NOT 超过 4096 UTF-8 字节；
MUST NOT 把完整遥测 JSON 或大型报告送回模型。`attached=true` 且 `test_state=ended` 即视为统计块确认，
MUST NOT 再读完整 JSON/Markdown；采集与写回细则见
[model-telemetry](../model-telemetry/spec.md)。

#### Scenario: 归档性能记录

- **WHEN** 一轮真实模型基准完成
- **THEN** 归档附件为受字节上限约束的简报，完整遥测保留在 `storage/.cache/runtime/telemetry/` 运行数据目录
