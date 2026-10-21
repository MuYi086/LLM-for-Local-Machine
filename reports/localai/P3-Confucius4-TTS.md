# P3：Confucius4-TTS LocalAI 代码试点

- 状态：`implemented_unverified`；候选权重和 LocalAI 运行仍为 E0。
- 服务：`Confucius4_TTS` / 8361，`POST /v1/confucius4TTS/generate`。
- 默认 provider：`native`；仅设置 `UNITALE_CONFUCIUS4_TTS_PROVIDER=localai` 才选择 LocalAI。
- LocalAI model ID：`unitale.confucius4-tts.orig`。
- audio.cpp：`family:confucius4_tts`、`task:clon`、CUDA device 0、`confucius4_tts.mem_saver=true`。
- 权重候选：`audio-cpp/audio.cpp-gguf` revision
  `83c5d96c03023ff5a7712570d057ce26c8769f98`，文件
  `Confucius4-TTS-GGUF/confucius4-tts-orig.gguf`。固定页面没有可用的文件 size/SHA-256；用户手动准备，本次没有下载。
- 上游依据：[固定 audio.cpp TTS 文档](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/backend/cpp/audio-cpp/audio.cpp/docs/tts.md)、
  [固定 HF 包目录](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98/Confucius4-TTS-GGUF)、
  [audio.cpp 固定版本模型许可清单](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/backend/cpp/audio-cpp/audio.cpp/docs/model_licenses.md)。

## 已实现字段映射

| Unitale 输入/配置 | LocalAI/audio.cpp | 当前限制 |
| --- | --- | --- |
| `audio_path` | 参考存储解析后传到 `voice` | 校验逻辑引用，保持同一上传 WAV。 |
| `lang` | `language` | 默认 `zh`；固定文档的 14 个语言代码可传，英文和中文为已验证 normalization 路径，其余 12 种为实验路径。 |
| `text` | `input` | 保留服务既有 1–12000 字符 API 校验。 |
| `temperature`、`top_p`、`top_k`、`num_beams`、`repetition_penalty` | 同名 `params` | 参数名/默认值按固定 audio.cpp 文档映射；实际数值影响待真实模型验证。 |
| `max_length` | `params.max_tokens` | 服务默认 1520。 |
| `n_timesteps`、`inference_cfg_rate` | `params.num_inference_steps`、`params.guidance_scale` | 保持 native 服务默认参数。 |
| `max_text_tokens_per_segment` | `params.text_chunk_size` | 同名语义映射；audio.cpp 自行分段并按请求 fade 值拼接。 |
| `cross_fade_duration`、`edge_fade_duration`、`edge_pad_duration` | 同名 `_sec` 的 `params` | 秒值原样传递。 |
| `prompt_text` sidecar | 不发送 | 与 Confucius4 原生 worker 一致，sidecar 只用于上传/检查元信息。 |
| `raw`、`verbose`、`vocalization_duration_seconds` | 无 LocalAI 映射 | 启用 raw/verbose 或指定感叹音时长时返回 422。 |
| `CONFUCIUS4_TTS_DEVICE` | 固定 YAML `device:0` | 非 `cuda` / `cuda:0` 配置返回 422。 |
| WAV 输出 | 网关验证 RIFF/WAVE，沿用 `storage/clone/` 持久化 | 输出采样率与真实服务完整合同尚未通过模型确认。 |

## 注册与验收

软件升级完成后，用户将权重手动放在 LocalAI models 目录的
`Confucius4-TTS-GGUF/confucius4-tts-orig.gguf`，再运行：

```bash
python3 scripts/localai/manager.py register-confucius4-tts-candidate
```

该命令不下载模型。它复制固定 YAML，并记录本地 size/SHA-256，用于检查本地文件是否变化；由于上游 LFS hash 不可得，
不能据此确认权重来源。启动 preflight 检查模型存在、记录和 YAML 与固定候选一致。

无模型回归覆盖字段映射、sidecar 不参与、unsupported 请求 422、LocalAI 路由不启动 native worker、输出存储和 manager 手动注册。
LocalAI gRPC backend 已构建、安装并通过无模型启动检查；候选权重与真实服务 smoke 尚未准备，因此没有 E2 请求/响应和 E3 质量、时延、显存及退出证据。
后续需验证中英文及其他声明语言、长文本分段、输出 WAV 格式、backend shutdown、混合 native/LocalAI GPU 锁互斥和异常隔离。
Native provider 保持默认，Confucius4 Python 模型依赖没有删除。
