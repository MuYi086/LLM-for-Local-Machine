# OpenCode 上下文插件与固定上下文基准

## Purpose

覆盖 `.opencode/plugins/local-model-context/` 插件的生效范围与上下文修复行为，以及固定上下文基准
v6 的自动遥测绑定、源码分页读取与报告边界。模型服务侧的压缩预算语义见
[ornith-text-model](../ornith-text-model/spec.md)，采集与写回见
[model-telemetry](../model-telemetry/spec.md)。

## 实现依据

- [.opencode/plugins/local-model-context/index.js](../../../.opencode/plugins/local-model-context/index.js)
- [.opencode/plugins/local-model-context/package.json](../../../.opencode/plugins/local-model-context/package.json)
- [CONTEXT_BENCHMARK_PROMPT.md](../../../CONTEXT_BENCHMARK_PROMPT.md)
- [scripts/prepare_context_benchmark.py](../../../scripts/prepare_context_benchmark.py)
- [docs/opencode-context.md](../../../docs/opencode-context.md)
- [docs/opencode-compaction-budget.md](../../../docs/opencode-compaction-budget.md)
- [tests/opencode/context-guard.test.mjs](../../../tests/opencode/context-guard.test.mjs)
- [tests/test_opencode_context_guard.py](../../../tests/test_opencode_context_guard.py)

## Requirements

### Requirement: 插件只在本仓库的本地 provider 生效

OpenCode V2 本地插件 MUST 只对 `local-ornith` provider、仅在本仓库生效，无 npm 依赖；插件 MUST NOT
修改主任务 system 或持久化历史，MUST NOT 新增模型请求、直连 llama.cpp 后端或绕过共享 GPU 锁。
停用方式 MUST 通过 OpenCode 项目配置的 `-llm-local-model-context`，MUST NOT 修改打包的可执行文件。

#### Scenario: 非本地模型会话

- **WHEN** 会话使用非 `local-ornith` provider
- **THEN** 插件不改变该会话的 system、工具或请求参数

### Requirement: 源码读取强制分页且单次一个文件

插件 MUST 对执行固定上下文基准的会话在源码 read 执行前补齐/限制 `limit<=150`、`offset>=1`；
同一次模型回复中的第二个源码 read MUST 返回明确工具错误。其他文件与其他会话 MUST NOT 被限制。
固定任务 MUST NOT 用 `cat`、整文件 read 或合并命令绕过分页。

#### Scenario: 一次回复多次读取源码

- **WHEN** 同一模型回复发起第二个源码 read
- **THEN** 工具返回明确错误，模型在下一轮按实际未读行继续，不把拒绝当作任务失败

#### Scenario: 缺少分页参数

- **WHEN** 源码 read 未带 offset/limit 或 limit 超过 150
- **THEN** 插件在工具执行前补齐或限制为合法分页参数

### Requirement: 压缩请求隔离且不关闭自动压缩

压缩请求 MUST 使用精简摘要 system，将历史任务格式当作待总结数据并要求原生英文检查点标题；
只替换压缩请求的 system 和工具，MUST NOT 更改持久化历史或主任务 system。仅压缩请求 MUST 设置
`temperature=0`、输出上限最多 4096（保留更低既有预算），并在 HTTP 请求构造后设置
`chat_template_kwargs.enable_thinking=false`；正文、其他模板参数与鉴权头 MUST 保留。自动压缩
MUST NOT 被关闭，普通请求 MUST NOT 改变。

#### Scenario: 触发自动压缩

- **WHEN** OpenCode 对本地模型会话发起压缩请求
- **THEN** 摘要使用独立 system 与英文标题模板，thinking 关闭且输出受限，压缩后从下一未读行续读

### Requirement: 本轮数据由插件自动绑定

固定入口 MUST 为 `storage/.cache/runtime/telemetry-current`，插件 MUST 在首个基准请求前创建本轮测试
ID 与带上下文档位的结果文件，并将 `directory`、`test_id`、`context`、`report` 放入会话指令。固定任务
MUST 使用系统指令给出的绑定信息，MUST NOT 自行选择最新历史目录、旧报告或另建报告；绑定失败时
MUST 只报告具体缺项，MUST NOT 给出舒适度结论。

#### Scenario: 自动绑定成功

- **WHEN** 新会话提交固定任务且性能记录已开启
- **THEN** 插件提供本轮唯一目录、测试 ID、上下文档位与结果文件，模型在同一文件追加正文

#### Scenario: 自动绑定失败

- **WHEN** 缺少本轮目录、测试 ID 或上下文档位
- **THEN** 任务只报告缺失项，不从旧记录补数，也不给出通过/舒适结论

### Requirement: 固定任务报告不编造证据

固定任务 v6 MUST 按顺序完整阅读五个指定源码并保留真实行号，报告 MUST 使用简体中文、1200—1800 字、
严格六部分，证据表最多 8 行；未保留的行号或未观察到的压缩/性能项 MUST 写“未保留，待复核”或“未知”，
MUST NOT 循环回读凑证据或编造证据。除固定提示词、五个源码、本轮遥测标记/汇总与结果文件外，
MUST NOT 读取额外文件、历史、`.env`、密钥或原始日志。

#### Scenario: 报告阶段缺少精确行号

- **WHEN** 阅读时未保留某条证据的精确行号
- **THEN** 表格标注“未保留，待复核”，不为此反复回读源码

### Requirement: 验证不调用真实模型

插件行为 MUST 由 Node 单元测试与 Python 集成测试覆盖，集成测试使用临时 XDG 目录、临时源码与本机假
OpenAI SSE 服务；无 OpenCode/Node 的环境 MUST 明确跳过，MUST NOT 下载依赖、调用真实模型或依赖
CUDA/既有服务。单元与假模型测试通过 MUST NOT 写成真实模型长任务已通过。

#### Scenario: 无 GPU 环境运行插件测试

- **WHEN** 在无 OpenCode/Node 或无模型的环境执行插件测试
- **THEN** 相应测试跳过或通过，不触发真实推理，也不把跳过写成真实验证完成
