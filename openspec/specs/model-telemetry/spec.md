# 模型性能记录

## Purpose

覆盖 `bash start.sh` 默认启用的逐请求性能与资源记录：采集器不加载模型且不获取 GPU 锁、每次启动的独立目录、
`begin`/`end`/`report` 测试标记、简报出口与结果文件写回，以及指标口径与隐私边界。
固定上下文基准工作流见 [opencode-context-and-benchmark](../opencode-context-and-benchmark/spec.md)。

## 实现依据

- [llm_local_runtime/src/llm_local_runtime/telemetry.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_brief.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_brief.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_cli.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_cli.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_resources.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_resources.py)
- [scripts/model_telemetry.py](../../../scripts/model_telemetry.py)
- [config/model-telemetry.env](../../../config/model-telemetry.env)
- [docs/model-telemetry.md](../../../docs/model-telemetry.md)

## Requirements

### Requirement: 采集器不参与推理且不占用 GPU 锁

性能记录默认启用；采集器 MUST NOT 加载模型、调用推理 API 或获取 GPU 锁，MUST NOT 改变推理结果；
采集缺失、响应事件过大或磁盘写入失败 MUST NOT 中断推理，磁盘失败 MUST 告警。关闭记录 MUST NOT
关闭必要的控制台与故障日志。

#### Scenario: 默认启动与显式关闭

- **WHEN** 通过 `bash start.sh` 启动服务，或显式设置 `MODEL_TELEMETRY_ENABLED=false`
- **THEN** 记录按配置开启或关闭，显式环境变量优先于 `config/model-telemetry.env`，推理路径不变

#### Scenario: 采集失败

- **WHEN** 资源采样或磁盘写入失败
- **THEN** 推理继续执行，失败以告警体现，不以记录缺失为由中断请求

### Requirement: 每次启动独立目录且不清理历史

每次启动 MUST 创建独立性能目录（默认 `storage/.cache/runtime/telemetry/run-.../`），MUST NOT 清理
历史记录；单独启动 API 时使用 `MODEL_TELEMETRY_DIR` 或 `telemetry/standalone/`，目录可用
`MODEL_TELEMETRY_ROOT` 覆盖。原始采样与报告 MUST 视为运行数据，MUST NOT 提交到 Git。

#### Scenario: 多次启动

- **WHEN** 服务重启或单独启动某个 API
- **THEN** 各批次使用独立目录，历史采样保留，不混入上一批次数据

### Requirement: 简报出口只含固定指标

`end`/`report --brief` MUST 只输出固定服务指标、资源峰值、warnings 摘录与 attached 状态，stdout
MUST NOT 超过 4096 UTF-8 字节；完整数据 MUST 继续落盘。`attached=true` 且 `test_state=ended`
即视为统计块已确认，MUST NOT 再读取完整 JSON/Markdown。

#### Scenario: 归档简报

- **WHEN** 维护者用 `end --brief` 或 `report --brief` 汇总一轮测试
- **THEN** 输出受字节上限约束，不携带原始提示词、回答或完整遥测文件

### Requirement: 测试标记区分窗口且暴露未完成请求

`begin`/`end`/`report` MUST 使用测试 ID 区分多轮；`end` MUST 写入结束标记并最多等待 10 秒让最后一次
请求完成写盘，可用 `--settle-seconds` 调整；存在未结束请求时报告 MUST 列出 `unfinished`，MUST NOT
当作完整成功样本。写回合并 MUST 显式提供目录与测试 ID，MUST NOT 支持“自动最新批次”或任意时间范围。

#### Scenario: 多个测试轮次

- **WHEN** 同一服务批次执行多轮 begin/end
- **THEN** 每轮按独立测试 ID 聚合，跨窗口边界的请求保留完整数据并注明边界

#### Scenario: 仍有未完成请求

- **WHEN** `end` 时请求尚未完成卸载或写盘
- **THEN** 报告保留 `unfinished` 警告，需用相同目录与 ID 的 `report` 重新汇总

### Requirement: 结果写回只更新唯一统计块

`--attach-result` MUST 只更新匹配同一测试 ID 的唯一 `MODEL_TELEMETRY_BEGIN`/`MODEL_TELEMETRY_END`
统计块；目标 MUST 是已有普通 Markdown 文件，符号链接、缺少标记、重复标记和错误 ID MUST 拒绝；
写回 MUST 使用文件锁与原子替换，MUST NOT 修改块外正文或手动补录的客户端统计。

#### Scenario: 合法写回

- **WHEN** 结果文件存在唯一匹配测试 ID 的统计块
- **THEN** 阶段快照或最终统计原子写入该块，块外内容保持不变

#### Scenario: 非法写回目标

- **WHEN** 目标是符号链接、标记缺失/重复或 ID 不匹配
- **THEN** 写回被拒绝并报错，不强行覆盖任何文件

### Requirement: 记录最小且写入隔离

记录器 MUST NOT 保存提示词、回答、工具参数、鉴权头或进程命令行；每个 JSONL 文件 MUST 只由一个进程
写入，并在进程内用锁串行追加。不同观测层以共同 `request_id` 关联，各层 `trace_id` 独立；各层
token 与耗时 MUST NOT 相加。

#### Scenario: 记录敏感字段

- **WHEN** 请求包含提示词、工具参数或鉴权头
- **THEN** 这些字段不出现在任何遥测文件与简报中

### Requirement: 指标口径不冒充界面实测

`first_content_ms` MUST 按本层首次非空正文事件计算，MUST NOT 冒充 OpenCode 界面 TTFT；原生 decode
速度与 `prompt_n` MUST NOT 当作界面 tok/s 或完整输入 usage；无直接证据的加载耗时、OOM 与重试
MUST 保持未知，MUST NOT 编造。

#### Scenario: 报告性能结论

- **WHEN** 汇总首事件延迟、加权 decode 速度或缓存命中效果
- **THEN** 报告区分观测层与界面口径，缺失项写未知，不以小批次速度推断加速
