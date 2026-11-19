# audio.cpp 音频模型服务

## Purpose

覆盖十五个音频模型 HTTP API 的服务合同：服务事实的唯一权威清单、共享 runtime 边界、上传处理、
逐模型合同差异，以及源分离与强制对齐两类专属合同。GPU 锁与隔离语义由
[gpu-lock-and-quarantine](../gpu-lock-and-quarantine/spec.md) 单独约束。

## 实现依据

- [localai/manifests/audio-cpp-models.json](../../../localai/manifests/audio-cpp-models.json)
- [localai/manifests/capabilities.json](../../../localai/manifests/capabilities.json)
- [scripts/localai/validate_capabilities.py](../../../scripts/localai/validate_capabilities.py)
- [model-audio/audio_cpp_runtime/src/audio_cpp_runtime/service.py](../../../model-audio/audio_cpp_runtime/src/audio_cpp_runtime/service.py)
- [model-audio/audio_cpp_runtime/src/audio_cpp_runtime/specs.py](../../../model-audio/audio_cpp_runtime/src/audio_cpp_runtime/specs.py)
- [model-audio/（十五个服务入口 main.py）](../../../model-audio/)
- [docs/localai-audiocpp-upgrade.md](../../../docs/localai-audiocpp-upgrade.md)

## Requirements

### Requirement: 服务事实以候选清单为唯一权威

音频服务的 GGUF 文件名、family/task、LocalAI model ID、路由与端口 MUST 以
`localai/manifests/audio-cpp-models.json` 为准；文档与实现冲突时先核对清单再更新文档，
服务口径由 `scripts/check_service_inventory.py` 在质量门禁中核对。

#### Scenario: 新增或修改音频模型服务

- **WHEN** 清单中新增模型条目或调整端口、路由、family/task
- **THEN** start.sh 注册、README 服务表、AGENTS.md 服务表与 `localai/manifests/capabilities.json` 同步更新，
  否则无模型质量门禁失败

#### Scenario: 修改服务合同

- **WHEN** 调整请求参数、响应结构或错误语义
- **THEN** 同步 `localai/manifests/capabilities.json` 并运行
  `scripts/localai/validate_capabilities.py`，通过后才算完成

### Requirement: 共享 runtime 与依赖边界

十五个音频模型服务 MUST 全部通过 LocalAI/audio.cpp 推理（LocalAI 由
`scripts/localai/manager.py` 在 127.0.0.1:8400 单独管理，不属于仓库 API 服务）；其中 VoiceDesign
（8301，`/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign`）使用独立 uv 项目，其余十四个共用
`model-audio/audio_cpp_runtime/` 轻量项目且 MUST NOT 安装 Torch。

#### Scenario: 新增 audio.cpp 模型 API

- **WHEN** 在 `model-audio/<模型目录>/main.py` 接入新模型
- **THEN** 该入口只从清单选择对应的共享服务合同，不复制推理、上传或错误映射实现，
  也不引入 Torch 等重依赖

### Requirement: 健康检查不加载权重

模型 API 健康检查 MUST NOT 加载模型权重，避免探活请求触发显存占用。

#### Scenario: 服务就绪探活

- **WHEN** 控制面或启动预检调用任一音频模型健康接口
- **THEN** 响应只反映进程与配置状态，不触发权重加载或推理

### Requirement: 音频上传按块暂存且不长期保存

音频上传 MUST 按块暂存，默认单文件上限 64 MiB；实现 MUST NOT 整文件 `read()`、直接覆盖
sidecar 或将临时输入长期保存。

#### Scenario: 接收合法上传

- **WHEN** 客户端提交不超过默认上限的音频文件
- **THEN** 文件按块暂存、校验后进入推理流程，请求结束后不在服务目录长期保留临时输入

#### Scenario: 超限或非法上传

- **WHEN** 文件超过默认上限或校验失败
- **THEN** 请求被拒绝并清理已暂存内容，不留下部分写入的临时文件

### Requirement: 逐模型合同差异保持显式

各模型 MUST 按自身合同处理输入：Qwen3-TTS Base 通过 WAV 参考音频克隆且必须提供与音频逐字匹配的
`reference_text`，输入 `zh`/`zh-CN` 映射为 audio.cpp 接受的 `Chinese`；Confucius4 与
IndexTTS2.5 同为 WAV 参考克隆；SeedVC 需要源音频和目标 WAV；Qwen3-ASR 返回转写 JSON；
MiDashengLM-Gen、Stable Audio SFX、Stable Audio 3 Medium 与 MOSS-SoundEffect v2 返回文本提示
生成的音频/音效；VoxCPM2 允许无参考音频运行。

#### Scenario: 克隆请求缺少逐字稿

- **WHEN** Qwen3-TTS Base 请求的 `reference_text` 与参考音频不逐字匹配或缺失
- **THEN** 请求在推理前被拒绝，不消耗 GPU 租约

#### Scenario: ASR 转写

- **WHEN** 客户端向 Qwen3-ASR 提交音频
- **THEN** 服务返回转写 JSON，不额外生成或保存音频

#### Scenario: 文本提示生成请求

- **WHEN** 客户端向 MiDashengLM-Gen、Stable Audio SFX、Stable Audio 3 Medium 或
  MOSS-SoundEffect v2 提交文本提示
- **THEN** 服务返回生成的音频/音效，不要求参考音频

### Requirement: 源分离一次推理返回全部 stems

BS-RoFormer（8344）、Mel-Band-RoFormer（8346）与 HTDemucs（8347）MUST 经一次 transform 请求的
`X-Audio-Stems` 产物取回全部 stem 并返回 WAV stems ZIP；MUST NOT 通过多次 transform 逐个提取。
stem 清单缺失、格式错误、重复命名或包含越界引用时 MUST 拒绝该响应且不留请求文件。

#### Scenario: 正常分离请求

- **WHEN** 客户端上传受支持的 WAV 并请求分离
- **THEN** 服务在共享 GPU 锁内执行一次 transform，按响应头清单取回全部 stem，返回完整 ZIP

#### Scenario: stem 清单不完整或越界

- **WHEN** `X-Audio-Stems` 缺失、解析失败、重名或指向受限路径之外的产物
- **THEN** 服务拒绝不完整/越界响应并清理已暂存文件，不返回部分结果

#### Scenario: stem 下载中途失败

- **WHEN** 某个 stem 获取失败或内容不是 PCM WAV
- **THEN** API 返回错误并清理此前已暂存的所有文件

### Requirement: 强制对齐必须要求已知逐字稿

Qwen3-ForcedAligner（8345）MUST 要求客户端提供已知逐字稿，返回词级时间对齐 JSON；上游响应缺少
`words` 时间戳时 MUST 返回上游合同错误，不把普通转写文本误报为对齐结果。

#### Scenario: 正常对齐请求

- **WHEN** 客户端提供音频与对应逐字稿
- **THEN** 服务返回包含词级时间戳的对齐 JSON

#### Scenario: 上游缺少词级时间戳

- **WHEN** 上游响应没有 `words` 时间字段
- **THEN** 服务返回上游合同错误，不降级为纯转写文本响应
