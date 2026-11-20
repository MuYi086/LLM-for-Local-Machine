# 上传、原子提交与保留策略

## Purpose

覆盖 `llm_local_runtime` 共享包的流式上传、校验、SHA-256 与原子提交，客户端路径标识的边界，
运行数据默认目录，以及保留策略的安全约束。

## 实现依据

- [llm_local_runtime/src/llm_local_runtime/storage.py](../../../llm_local_runtime/src/llm_local_runtime/storage.py)
- [llm_local_runtime/src/llm_local_runtime/gpu.py](../../../llm_local_runtime/src/llm_local_runtime/gpu.py)
- [main/storage_maintenance.py](../../../main/storage_maintenance.py)
- [AGENTS.md 存储与提交约束](../../../AGENTS.md)
- [README.md 存储与清理](../../../README.md)

## Requirements

### Requirement: 上传流式暂存并原子提交

上传 MUST 经过 `llm_local_runtime` 流式暂存、校验、SHA-256 与原子提交；提交结果要么完整可见，
要么不落盘，不留半成品。

#### Scenario: 提交合法上传

- **WHEN** 客户端上传通过校验的文件
- **THEN** 内容流式暂存、计算 SHA-256 并原子提交到目标存储，观测者不会看到部分写入状态

#### Scenario: 暂存或校验失败

- **WHEN** 暂存过程中断或校验不通过
- **THEN** 已暂存内容被清理，不产生可见的提交结果

### Requirement: 客户端路径只是逻辑标识

客户端提供的 `full_path` MUST 视为逻辑标识；服务 MUST NOT 直接按该值读取服务器任意路径。

#### Scenario: 提交引用逻辑路径

- **WHEN** 客户端以 `full_path` 引用既有上传
- **THEN** 服务按登记的 SHA-256 与暂存记录解析引用，不把该值当作文件系统路径打开

#### Scenario: 伪造或越界的路径标识

- **WHEN** `full_path` 指向存储目录之外或格式非法
- **THEN** 请求被拒绝，不触发对目标路径的读取

### Requirement: 共享包保持依赖边界

`llm_local_runtime` MUST NOT 导入 FastAPI、Torch 或模型包；由维护中的 uv 项目以 editable
依赖引用。

#### Scenario: 引入违规依赖

- **WHEN** 该包新增对 FastAPI、Torch 或任一模型包的导入
- **THEN** 依赖边界被破坏，审查不通过，需改为在调用方实现相关逻辑

### Requirement: 运行数据目录职责固定

默认目录 MUST 保持：VoiceDesign 音色位于 `storage/timbre/`；控制面与服务临时文件位于
`$RUNTIME_CACHE_DIR`；FLUX 编辑上传临时目录为 `$RUNTIME_CACHE_DIR/sdcpp_uploads` 且请求结束后删除；
新增 audio.cpp API MUST NOT 长期保存上传或输出。

#### Scenario: 请求级临时文件

- **WHEN** 图片编辑或音频处理请求产生临时上传
- **THEN** 临时内容写入 `$RUNTIME_CACHE_DIR` 下的专用目录并在请求结束后删除

#### Scenario: 真实模型样例输出

- **WHEN** 执行真实模型串行验证
- **THEN** 样例写入运行数据目录（如 `storage/llm_tests/`、`storage/audio_cpp_tests/`），不提交进仓库

### Requirement: 保留策略默认关闭且可预览

保留策略 MUST 默认关闭；显式设置时 MUST 先预览清理计划再执行；实现 MUST NOT 删除用户上传、
历史运行数据、生成结果或策略未覆盖目录；`storage/` 中的用户运行数据不由代码清理。

#### Scenario: 启用保留策略

- **WHEN** 部署显式配置保留策略
- **THEN** 系统先输出清理计划预览，确认后才执行对应清理

#### Scenario: 误配置的清理范围

- **WHEN** 策略试图覆盖用户上传或策略外目录
- **THEN** 清理不执行并报告风险，不以任何路径折叠方式删除未授权内容

#### Scenario: 预览后执行清理

- **WHEN** 显式设置 `STORAGE_RETENTION_HOURS` 或 `STORAGE_RETENTION_MAX_BYTES`
- **THEN** 先以 `python main/storage_maintenance.py` 预览计划，确认后才传 `--apply`；脚本只匹配
  `storage/timbre/` 下的 `qwen_voicedesign_*.wav`，不触及其他运行数据
