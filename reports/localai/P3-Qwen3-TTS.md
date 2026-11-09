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

历史验证记录：本轮服务退役前的质量门禁通过；当时的项目与测试计数已过期。当前收尾质量门禁结果见[历史升级归档](audiocpp-upgrade-history-2026-10-22.md)。Base 克隆的 gateway/provider 路由由 mock 覆盖；下方另记录 VoiceDesign GGUF 的真实请求与 WAV 输出校验。

## 待完成验收

1. 本地快照和 hash 已登记；上游 revision 身份的独立联网核对尚未完成。
2. 在 GPU 上验证 YAML 加载、参考 WAV/文本映射、语言、token/采样值、输出格式和音频质量。
3. 核验 `/backend/shutdown` 后 backend 子进程退出与显存回落，并执行失败断连和隔离恢复演练。
4. 对照 native 建立中英多长度样本与 CER/WER/盲听及延迟、峰值显存基线；满足第 9 节发布门禁后才考虑改变默认值。

---

## Qwen3 VoiceDesign GGUF 接入

- 状态：`single_sample_verified_quality_unverified`
- 日期：2026-11-08
- 服务：`qwen3_voiceDesign` / 8301，`POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign`
- provider：固定 `localai/audio.cpp`；start.sh 默认注册候选、启动受管 LocalAI 并预检
- 模型 ID：`llm-local.qwen3-tts.1p7b-voicedesign`，固定 `family:qwen3_tts`、`task:vdes` 与 CUDA
- 模型文件：`$HF_MIRROR_DIR/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf`
- SHA-256：`f12c79e8c492f006b04f5ba823bebe512c9d46e13a7837081f1d34803b4bac0e`
- 本机推理证据：最终代码 API 返回 24 kHz、单声道、3.36 秒、161324 字节 PCM WAV；共享 GPU 记录峰值 5522 MiB；`/backend/shutdown` 与 LocalAI 受管进程组停止均已确认
- 生成文件：`storage/timbre/qwen3-tts-voicedesign-56e80fd15f7e42f6abd865819d4ed83c.wav`
- 验收边界：单次短中文请求和 WAV 结构已验证；听感、长文本、音色描述效果及故障恢复演练未验收

### 字段映射

| VoiceDesign 请求字段 | LocalAI/audio.cpp 字段或处理 | 说明 |
| --- | --- | --- |
| `voice_description` | `/v1/audio/speech` 的 `instructions` | 固定使用 `task:vdes`，不加入 Base 克隆参考音频 |
| `text`、`language` | `input`、`language` | 语言为空时省略 |
| `max_new_tokens`、`top_p`、`temperature` | `params.max_tokens` 与同名采样字段 | 可选采样值为空时省略 |
| `max_chars_per_chunk` | CPU 按 Unicode 字符数切段 | 单次最多 32 段，超出返回 422 |
| `pause_ms` | CPU 合并 PCM WAV 时插入静音 | 分段采样参数必须一致 |
| 输出 | 验证 WAV 后写入 `storage/timbre/` 并登记 `AudioReferenceStore` | HTTP 输出与存储文件经逐字节比较一致 |

管理器只接受任务指定文件名的非空 GGUF；在 LocalAI models 目录建立受控相对 `.gguf` 链接，并拒绝覆盖未知 YAML。旧 safetensors YAML 仅在内容与已知旧配置完全相同时备份后迁移。真实请求经共享 GPU 锁排队并在响应后确认 backend 停止；无法确认时继续遵循 gateway quarantine 规则。

无模型回归覆盖 API 校验、WAV 存储与分段、候选注册、旧配置迁移和 start.sh 配置顺序。执行 `uv run --project qa --locked python tests/qwen3_tts_voicedesign/generate_example.py` 可在具备完整部署的主机重新生成并验证短句示例。2026-10-22 safetensors 快照 hash 保留于[历史版本快照](versions-history-2026-10-22.json)，不是当前候选；当前 GGUF 与推理结果登记于 [`versions.json`](../../localai/manifests/versions.json)。
