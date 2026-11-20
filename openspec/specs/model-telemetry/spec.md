# 模型性能记录

## Purpose

覆盖 `bash start.sh` 默认启用的逐请求性能与资源记录：采集器不加载模型且不获取 GPU 锁、每次启动的独立目录、
`begin`/`end`/`report` 测试标记、简报出口与结果文件写回，以及指标口径与隐私边界。

## 实现依据

- [llm_local_runtime/src/llm_local_runtime/telemetry.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_brief.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_brief.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_cli.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_cli.py)
- [llm_local_runtime/src/llm_local_runtime/telemetry_resources.py](../../../llm_local_runtime/src/llm_local_runtime/telemetry_resources.py)
- [scripts/model_telemetry.py](../../../scripts/model_telemetry.py)
- [config/model-telemetry.env](../../../config/model-telemetry.env)
- [docs/model-telemetry.md](../../../docs/model-telemetry.md)
- [tests/test_model_telemetry.py](../../../tests/test_model_telemetry.py)
- [tests/test_telemetry_brief.py](../../../tests/test_telemetry_brief.py)

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

收尾简报 MUST 只输出固定白名单指标，MUST NOT 携带请求正文、逐进程完整记录或任意额外字段。
资源指标保留采样数、整卡显存峰值、guest 内存峰值；MUST NOT 输出 `ornith_backend` 后端 RSS 峰值
或任何 llama.cpp 专属字段。简报 MUST 为可被 `json.loads` 解析的单行 JSON 且不超过 4096 UTF-8 字节。

#### Scenario: 归档简报

- **WHEN** 对含大量原始记录的报告执行 `end --brief`
- **THEN** 简报只含 `test_id`、`test_state`、`window_seconds`、服务固定指标与资源白名单，无原始记录泄漏
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
写回 MUST 使用文件锁与原子替换，MUST NOT 修改块外正文或手动补录的客户端统计。采集与写回行为
MUST 由 `tests/test_model_telemetry.py` 与 `tests/test_telemetry_brief.py` 无模型覆盖。

#### Scenario: 合法写回

- **WHEN** 结果文件存在唯一匹配测试 ID 的统计块
- **THEN** 阶段快照或最终统计原子写入该块，块外内容保持不变

#### Scenario: 非法写回目标

- **WHEN** 目标是符号链接、标记缺失/重复或 ID 不匹配
- **THEN** 写回被拒绝并报错，不强行覆盖任何文件
### Requirement: 记录最小且写入隔离

采集器 MUST NOT 记录请求正文、环境变量或命令行等敏感字段；进程角色仅用固定标签。
MUST NOT 记录 `ornith_backend`/`ornith_adapter` 角色、`ORNITH_MTP_9B_*` 配置快照、
Ornith 模型文件信息或 MTP 原生日志解析结果。LocalAI、sd.cpp 与其余模型的采集口径不变。

#### Scenario: 记录敏感字段

- **WHEN** 采集器读取进程、日志与环境变量
- **THEN** 不持久化 Ornith 专属角色/配置/模型文件字段，不把命令行或环境变量原文写入记录
### Requirement: 指标口径不冒充界面实测

`first_content_ms` MUST 按本层首次非空正文事件计算，MUST NOT 冒充 OpenCode 界面 TTFT；原生 decode
速度与 `prompt_n` MUST NOT 当作界面 tok/s 或完整输入 usage；无直接证据的加载耗时、OOM 与重试
MUST 保持未知，MUST NOT 编造。

#### Scenario: 报告性能结论

- **WHEN** 汇总首事件延迟、加权 decode 速度或缓存命中效果
- **THEN** 报告区分观测层与界面口径，缺失项写未知，不以小批次速度推断加速
