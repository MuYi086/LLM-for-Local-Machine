# P3：Qwen3-TTS Base 克隆代码试点

- 状态：`single_sample_verified_quality_unverified`
- 日期：2026-10-22
- 服务：`qwen3_tts` / 8321，`POST /v1/qwen/clone`
- 默认 provider：`native`
- LocalAI 候选：`llm-local.qwen3-tts.1p7b-base`，audio.cpp `family:qwen3_tts`、`task:tts`
- 模型制品：官方 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` safetensors；权重由用户手动准备，snapshot revision 与全量非缓存文件 hash 已按本地制品登记于 versions.json
- 证据：无模型 mock 覆盖；2026-10-22 通过一次真实中文短句、24 kHz WAV 及 shutdown；完整 E2/E3 未验收

## 字段映射

| LLM Local Backend 字段 | LocalAI/audio.cpp 处理 | 状态与限制 |
| --- | --- | --- |
| `text` | `/v1/audio/speech` 的 `input` | 保留原请求校验；长文本若按现行切段策略会产生多个片段则返回 422 |
| `audio_path` | 现有逻辑标识解析成 `voice` 本地音频路径 | 只使用已校验的引用，不接受任意服务器路径；使用合成参考完成一次 x-vector-only 调用；文本条件仍待扩展验证 |
| `backend` | 保留旧 API 固定值校验 | 不作为 provider 选择器，也不转发 |
| `prompt_text` | `params.reference_text` | 使用引用 sidecar 或显式请求转写 |
| `language` | `language` | 原值映射；audio.cpp 的具体语种表现待模型 smoke |
| `x_vector_only` | `params.x_vector_only_mode` | 没有参考转写时强制为 `true`，有转写时保留请求值 |
| `max_new_tokens` | `params.max_tokens` | 语义映射待固定 backend 实测 |
| `top_p`、`temperature` | 同名 `params` | temperature 非正值在进入 GPU 锁前拒绝 |
| `device_map`、`dtype`、`attn_implementation` | 不支持 | 请求显式提供非空值时返回 422，不静默忽略 |
| `max_chars_per_chunk`、`pause_ms` | 不支持跨片段合成/拼接 | 如果请求文本与切段策略会生成多个片段，则返回 422 |
| `trim_leading_silence*` | LocalAI WAV 返回后执行 CPU 后处理 | 复用原 worker 的去静音算法和参数 |
| 输出 | 校验 RIFF/WAVE，转单声道并沿用既有目录保存，返回 `audio/wav` | 24 kHz WAV 结构已通过一次调用验证；音质未验收 |

## 生命周期与候选注册

provider 通过 `LLM_LOCAL_QWEN3_TTS_PROVIDER=localai` 单独启用；native 仍是默认。LocalAI 请求沿用共享 GPU 文件锁，完成后调用 `/backend/shutdown`；停止未确认时写入隔离标记，阻止后续 GPU 推理。候选注册命令只检查手动准备的文件并生成匹配 YAML，不下载权重：

```bash
python3 scripts/localai/manager.py register-qwen3-tts-candidate
```

`QWEN3_TTS_MODEL_DIR` 可指向模型目录。注册会检查配置、生成配置、模型 safetensors、语音 tokenizer 和 tokenizer 文件存在且非空；本地 metadata revision 与非缓存文件 SHA-256 已记录；YAML 使用 models 内经过目标校验的相对 .hf 链接。

## 无模型验证

历史验证记录：本轮服务退役前的质量门禁通过；当时的项目与测试计数已过期。当前收尾质量门禁结果见 [升级记录](../../docs/localai-audiocpp-upgrade.md)。VoiceDesign 的 gateway/manager/provider 路由均由 mock 覆盖；这不证明固定 audio.cpp RPC、模型载入、WAV 质量、shutdown 后显存释放或并发真实表现。

## 待完成验收

1. 本地快照和 hash 已登记；上游 revision 身份的独立联网核对尚未完成。
2. 在 GPU 上验证 YAML 加载、参考 WAV/文本映射、语言、token/采样值、输出格式和音频质量。
3. 核验 `/backend/shutdown` 后 backend 子进程退出与显存回落，并执行失败断连和隔离恢复演练。
4. 对照 native 建立中英多长度样本与 CER/WER/盲听及延迟、峰值显存基线；满足第 9 节发布门禁后才考虑改变默认值。

---

## 同阶段：Qwen3 VoiceDesign 代码试点

- 状态：`single_sample_verified_quality_unverified`
- 服务：`qwen3_voiceDesign` / 8301，`POST /v1/qwen/timbre`
- 默认 provider：`native`
- LocalAI 候选：`llm-local.qwen3-tts.1p7b-voicedesign`，audio.cpp `family:qwen3_tts`、`task:vdes`
- 模型制品：官方 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` safetensors；权重由用户手动准备，snapshot revision 与全量非缓存文件 hash 已按本地制品登记于 versions.json
- 证据：mock 覆盖接口字段和切段存储；2026-10-22 通过一次真实中文短句、24 kHz WAV 及 shutdown；完整 E2/E3 未验收

### 字段映射和限制

| LLM Local Backend 字段 | LocalAI/audio.cpp 处理 | 状态与限制 |
| --- | --- | --- |
| `voice_description` | `/v1/audio/speech` 的 `instructions` | 固定 `task:vdes`，不附加 Base 克隆参考音频 |
| `text`、`language` | `input`、`language` | 保留现有文本和语言值 |
| `max_new_tokens`、`top_p`、`temperature` | `params.max_tokens`、同名采样参数 | 值映射 mock 已覆盖，真实生成待验收 |
| `max_chars_per_chunk` | 使用与 native worker 相同的标点/长度切段 | 在一次共享 GPU 租约内顺序合成；单次最多 32 段，超过时返回 422 |
| `pause_ms` | CPU 解码 WAV 后插入静音并重编码 | 复用 native worker 的分段停顿语义；采样率不一致或 WAV 无效时拒绝 |
| `device_map`、`dtype`、`attn_implementation` | LocalAI YAML 固定 CUDA 设备；拒绝非服务默认值 | 显式非默认配置返回 422；逐请求 PyTorch 覆盖不静默忽略 |
| 输出 | WAV 合并后保存至 `storage/timbre/` 并注册为可复用音色 | 一个短句输出 24 kHz WAV 且 shutdown 成功；音质及异常生命周期待扩展验证 |

VoiceDesign 注册命令只检查运维已准备的目录并写入独立 YAML：

```bash
python3 scripts/localai/manager.py register-qwen3-voicedesign-candidate
```

无模型测试覆盖候选缺文件、`task:vdes` 固定与配置篡改拒绝，以及设计指令、分段、静音合并入口和音色目录写入。WAV 解码/拼接依赖 VoiceDesign 服务的 NumPy/SoundFile，本次 QA 路由用 mock 替代；真实 backend 的一次短句、模型本地 revision/hash 与 shutdown 已验证；完整资源基准和质量对照仍待完成。

2026-10-22 修复候选绝对模型路径和 Base session 选项前缀；两条真实调用的样本大小、显存和验收边界见 [当前升级记录](../../docs/localai-audiocpp-upgrade.md#7-2026-10-22-复核修复与当前验证)。
