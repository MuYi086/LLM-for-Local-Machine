# P3：FireRedTTS3 Base 克隆与 Instruct 音色设计

- 状态：`implemented_unverified`
- 证据等级：E0 候选；固定 audio.cpp 文档和源码声明达到 E1，LocalAI 请求与目标机模型尚未验收。
- provider：新增 `UNITALE_FIRERED_TTS3_CLONE_PROVIDER` 与 `UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER`，两个进程分别配置，默认均为 `native`。
- Base model：`unitale.fireredtts3.base.q8-0`，`family:fireredtts3`、`task:clon`。
- Instruct model：`unitale.fireredtts3.instruct.q8-0`，`family:fireredtts3`、`task:vdes`。
- 权重 revision：`83c5d96c03023ff5a7712570d057ce26c8769f98`。固定快照显示 Base Q8 约 4.18 GB、Instruct Q8 约 4.15 GB；尚未取得上游 LFS SHA-256。模型均留待用户手动准备。
- 固定实现依据：[audio.cpp FireRedTTS3 文档](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/backend/cpp/audio-cpp/audio.cpp/docs/models/fireredtts3.md)、[固定模型包快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)。

## 字段映射

| Unitale 字段 | Base `clon` | Instruct `vdes` |
| --- | --- | --- |
| 文本 | `input` | `input` |
| 参考音频 `audio_path` | 先经 `AudioReferenceStore` 解析，再传 `voice` 文件路径 | 不适用 |
| 参考转写 `prompt_text` | 显式值或已校验 sidecar 映射到 `params.reference_text` | 不适用 |
| 音色描述 `voice_description` | 不适用 | `instructions` + `params.template_name=voice_design` |
| `language` | 顶层 `language` | 顶层 `language` |
| `n_timesteps` | `params.num_inference_steps` | `params.num_inference_steps` |
| `inference_cfg` | `params.guidance_scale` | `params.guidance_scale` |
| `stop_threshold` / `seed` | 同名 `params` | 同名 `params` |
| `backend` | 保留合同字段；native worker 本身也不使用该字段 | 不适用 |

Base 和 Instruct 使用不同模型 ID、YAML、LocalAI provider 与候选登记命令。Timbre 模式输出仍写入
`storage/timbre/` 并注册 `AudioReferenceStore`；clone 模式仍写入原配置输出目录。LocalAI 选项映射和 native
业务预检在取得共享 GPU 锁之前执行；推理期间持锁并要求 backend shutdown 成功后才释放。

本次只迁移 Unitale 已暴露的 Base clone 和 Instruct voice design。audio.cpp Instruct semantic/acoustic edit
虽有 family 文档，但当前服务没有对应 HTTP 路由，因此没有被包装成新增或已迁移功能。`FIRERED_TTS3_USE_FASTTEXT`、
`USE_WETEXT`、`DO_TN` 与 PyTorch attention 环境覆盖无法由固定 LocalAI 请求等价控制；只接受服务默认文本规范化配置和
`attn_implementation=auto`，其他配置明确返回 422。

## 变更与无模型验证

- `firered_tts3/main.py` 增加按 mode 独立的 provider 选择、健康状态、请求映射和错误转换；native worker 保留原样。
- 增加标准库 LocalAI client 的 Base reference clone 与 Instruct voice design 请求方法。
- 增加两份固定 YAML、固定 HF revision 清单、人工注册/本地文件 hash 校验 CLI 与 `start.sh` 启动前预检。
- 更新 FireRed 字段/路由能力快照；服务与 LocalAI client 测试覆盖参数映射、两种 mode 的输出/存储、候选注册以及 native worker 未被调用。
- `firered_tts3/pyproject.toml` 添加 editable `unitale-inference-gateway`，需更新并验证项目锁文件。

## 未完成验收

- LocalAI 固定 `audio-cpp` gRPC backend 已构建、安装并通过无模型启动检查；FireRed 权重与 E2/E3 推理、质量和生命周期验收仍待手动准备模型后执行。
- 用户下载两个 Q8 文件后，需检查 GGUF 元数据和本地 hash，并运行候选注册与 LocalAI 模型预检。
- 需在目标 GPU 测试中文、英语和选定方言；核对参考转写影响、voice design 指令遵循、WAV 采样参数、文本准确度、音质及模型退出/显存回落。
- LocalAI 的文本前端与 native `WeText/DO_TN` 质量一致性需要真实样例核验；当前不升到 E2/E3，也不改变默认 provider。

## 回滚

分别将 `UNITALE_FIRERED_TTS3_CLONE_PROVIDER`、`UNITALE_FIRERED_TTS3_TIMBRE_PROVIDER` 设置为 `native` 并重启。
原 Base/Instruct worker、editable 源码与已保存 WAV 都保留。
