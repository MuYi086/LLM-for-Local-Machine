# 音乐后处理编排规则

## Purpose

覆盖 [AGENTS.md](../../../AGENTS.md) 声明的音乐后处理协作规则：角色职责边界、阶段隔离与禁止重复处理，
以及 AI 分离分轨的证据边界。六个被引用 skill 的定义文件缺失属于未决事项，本规格只记录协作约定，
MUST NOT 把它们描述为当前已实现能力。

## 实现依据

- [AGENTS.md](../../../AGENTS.md)
- [.agents/skills/](../../../.agents/skills/)

## Requirements

### Requirement: 角色职责按声明路由且不得越界

按 AGENTS.md 的协作约定，执行音乐后处理时各角色 MUST 只承担声明的职责：`audio-analyzer` 只分析音频并
输出 JSON、MUST NOT 修改源音频；`mix-engineer` 只做分轨 EQ、动态处理和混音、MUST NOT 执行最终母带
标准化；`mastering-engineer` 是唯一负责最终母带、响度及峰值控制的角色；`ffmpeg-skill` 只负责裁剪、
探测、格式转换和通用导出、MUST NOT 接管混音与母带；`music-evaluate` 只负责质量评估和报告、未经明确
授权 MUST NOT 重新处理音频；`batch-processor` 只负责批量任务执行、MUST NOT 决定音乐处理策略。
上述职责是协作约定；各角色是否可用以 `.agents/skills/` 中实际存在的定义为准（见「未决事项」）。

#### Scenario: 执行最终母带

- **WHEN** 任务进入最终母带、响度与峰值控制阶段
- **THEN** 只由 `mastering-engineer` 职责执行，`mix-engineer` 与 `ffmpeg-skill` 不接管该阶段

#### Scenario: 质量评估发现问题

- **WHEN** `music-evaluate` 在评估中发现需要重新处理
- **THEN** 只产出报告与建议，未经明确授权不触发重处理

### Requirement: 阶段隔离且禁止重复处理

输入源文件 MUST 只读；每个阶段 MUST 写入独立输出目录，前一阶段验证通过后 MUST NOT 执行下一阶段；
同一任务 MUST NOT 由两个 Skill 重复执行响度归一化；MUST NOT 多个进程同时写入同一输出文件；默认
MUST NOT 并行执行多个 GPU 音频模型，全部阶段完成后 MUST 执行最终质量检查。

#### Scenario: 前一阶段未验证

- **WHEN** 上一阶段产物尚未验证
- **THEN** 后续阶段不启动，不把未验证中间结果交给下一角色

#### Scenario: 重复响度归一化

- **WHEN** 同一任务已被一个 Skill 做过响度归一化
- **THEN** 第二个 Skill 不重复执行归一化，避免叠加处理

### Requirement: AI 分离分轨不得冒充原生分轨

源分离产物 MUST 明确标记为 AI 估计分轨；MUST NOT 把 AI 分离的估计分轨当作原生录音分轨用于混音或评估
结论。该边界与 [audio-cpp-services](../audio-cpp-services/spec.md) 的源分离合同一致。

#### Scenario: 使用分离结果

- **WHEN** 下游阶段使用源分离返回的 stems
- **THEN** 记录其 AI 估计来源，不按原生多轨录音描述或评估

### Requirement: 声明的角色未被定义时不得描述为可用

AGENTS.md 声明的六个 skill 在 `.agents/skills/` 中尚无对应定义文件；在核验到实际定义前，规格与文档
MUST NOT 把它们写成已可用的工具能力，MUST NOT 为凑齐角色而编造实现或上游行为。该冲突 MUST 按
「待对齐」记录并保留责任方与核验方式。

#### Scenario: 询问某个后处理 skill 是否可用

- **WHEN** 维护者或 agent 查询 `audio-analyzer` 等六个角色的可用性
- **THEN** 如实说明定义文件缺失、当前仅为 AGENTS.md 协作约定，不声称可直接调用

## 未决事项

| 事项 | 状态 | 依据 / 责任方 | 临时处理 | 确认结果 |
| --- | --- | --- | --- | --- |
| AGENTS.md 声明的 audio-analyzer、mix-engineer、mastering-engineer、ffmpeg-skill、music-evaluate、batch-processor 六个 skill 在 `.agents/skills/` 均无定义文件 | 待对齐 | AGENTS.md「Music Postprocessing Rules」与 `.agents/skills/` 目录现状；责任方：仓库维护者 | 仅记录协作约定，不写成可用能力；不擅自创建或删除 skill 文件 | 待维护者确认是补充 skill 定义还是修订 AGENTS.md；确认后更新本规格与 AGENTS.md |
