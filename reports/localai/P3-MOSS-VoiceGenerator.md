# P3：MOSS VoiceGenerator 音色设计

- 状态：`implemented_unverified`
- 证据等级：E0 候选；固定 audio.cpp 源码声明达到 E1，LocalAI 请求和本机模型尚未验收。
- provider：新增 `UNITALE_MOSS_VOICEGENERATOR_PROVIDER`，默认仍为 `native`。
- 模型：`unitale.moss-voicegenerator.bf16`，`family:moss_voicegen`、`task:vdes`。
- 权重：`MOSS-VoiceGenerator-GGUF/moss_voicegen_bf16_codec_f16_decode.gguf`，固定 HF revision `83c5d96c03023ff5a7712570d057ce26c8769f98`。文件页约 6.02 GB，未取得固定 LFS SHA-256；无权重下载或真实推理。
- 固定 audio.cpp 源码：LocalAI v4.11.0 内嵌 `9a02e61326aaaf9d462b584ca5e0daba22c0abfc`；源码和模型包由用户提供于 `/home/muyi086/tts-depency/localai-audiocpp-upgrade-src`，无 Git metadata 的归档无法独立证明其 commit tree。

## 字段映射

| 服务字段 | audio.cpp/LocalAI 映射 | 边界 |
| --- | --- | --- |
| `voice_description` | `instructions` → session `instruction` | 不传参考音频；模型包自带 MOSS Audio Tokenizer v1 解码权重 |
| `text` | `input` | 标题标记按 native worker 规则移除，需 E2 对照确认 |
| `audio_temperature` | `params.temperature` | Family 默认 1.5 |
| `audio_top_p` / `audio_top_k` | `params.top_p` / `params.top_k` | Family 默认 0.6 / 50 |
| `audio_repetition_penalty` | `params.repetition_penalty` | Family 默认 1.1 |
| 无 seed 字段 | `params.seed` | 每个 LocalAI 请求生成新的 32-bit seed；audio.cpp 默认 0 会让同请求固定重放 |
| `max_chars_per_chunk` | provider 侧检查 | 需要 native Python 多段切分时返回 422 |
| `pause_ms` | 无 | 单段请求无段间静音；多段请求拒绝迁移 |
| `max_new_tokens` | 无等价项 | 只接受 native 默认 4096；非默认值返回 422 |
| `dtype` / `attn_implementation` | 无 | 只接受 `auto`；候选 GGUF 和 audio.cpp 控制 dtype/device |

语言选项不在当前 native API 合同内，provider 不传 `language`。Native 没有 seed 输入并使用未固定的 Torch 随机状态；LocalAI
provider 生成随机 seed 保持每次请求变化，但无法保证与 native 随机序列一致。输出仍返回 `audio/wav`，写入 `storage/timbre/` 并通过
`AudioReferenceStore.register_timbre_file` 建立跨服务引用。响应字节由标准库 LocalAI client 收集，当前代码没有额外验证
WAV 解码、时长或采样率；真实请求验收时必须补充这些检查证据。

## 变更与验证

- 增加独立服务 provider、请求转换、LocalAI client 方法、模型 YAML、固定版本清单候选、注册/校验 CLI 与启动前预检。
- 更新 `capabilities.json` 的 MOSS 路由/字段快照，所有请求字段都有显式映射、默认接受或拒绝规则。
- 无模型服务回归覆盖单段映射、分段/原生专属选项拒绝、LocalAI 路径绕过原生 worker、音色保存和引用注册。
- LocalAI manager 回归覆盖手动 GGUF 缺失、YAML 注册、revision 记录和本地 size/SHA-256 变化检测。
- `moss_voiceGenerator/pyproject.toml` 新增 editable `unitale-inference-gateway`；需更新并验证该项目 `uv.lock`。

## 未完成验收

- LocalAI v4.11.0 `audio-cpp` gRPC backend 已构建、安装并通过无模型启动检查；MOSS 权重与 E2/E3 推理、质量和生命周期验收仍待手动准备模型后执行。
- MOSS GGUF 由用户后续手动下载。需在固定 revision 下检查文件 hash、GGUF tensor 元数据和模型注册结果。
- 需在目标 GPU 实测中文/英文合成、音色指令遵循、文字可懂度、WAV 结构与输出采样率，记录 RTF/内存/显存。
- 需验证 LocalAI backend 成功、失败、超时、断连后的停止确认、进程退出与显存回落；mock 不代表 E2/E3。
- 上游文档描述长文本每段会创建独立音色且段间音量不一致；本试点因此拒绝 native 多段请求，之后如需支持需单独完成合同与拼接质量验收。

## 回滚

将 `UNITALE_MOSS_VOICEGENERATOR_PROVIDER` 设为 `native` 并重启服务。无需删除原 MOSS worker、editable 源码、模型目录或已保存音色。
