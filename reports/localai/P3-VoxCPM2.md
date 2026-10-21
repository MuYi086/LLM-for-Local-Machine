# P3：VoxCPM2 LocalAI 代码试点

- 状态：`implemented_unverified`，候选权重与运行时证据仍为 E0。
- 服务：`voxcpm2` / 8322，`POST /v1/voxcpm2/clone`。
- 默认 provider：`native`；仅设置 `UNITALE_VOXCPM2_PROVIDER=localai` 才选择 LocalAI。
- LocalAI model ID：`unitale.voxcpm2.q8-0`。
- audio.cpp：`family:voxcpm2`、`task:tts`、CUDA device 0；关闭 VoxCPM2 prompt cache。
- 权重候选：`audio-cpp/audio.cpp-gguf` revision
  `09fe073ba154561f4474162e8bd4ab233a848eca`，文件
  `VoxCPM2-GGUF/voxcpm2-q8_0.gguf`。HF revision 中列出该 Q8 文件，标示约 2.96 GB；固定 revision LFS SHA-256
  尚未取得。用户手动准备模型，本次没有下载。
- 上游依据：[固定 audio.cpp TTS 文档](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/backend/cpp/audio-cpp/audio.cpp/docs/tts.md)、
  [固定 HF 包目录](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/09fe073ba154561f4474162e8bd4ab233a848eca/VoxCPM2-GGUF)
  和[该 revision 的包 README](https://huggingface.co/audio-cpp/audio.cpp-gguf/blob/09fe073ba154561f4474162e8bd4ab233a848eca/README.md)。

## 已实现字段映射

| Unitale 输入/配置 | LocalAI/audio.cpp | 当前限制 |
| --- | --- | --- |
| `audio_path` | reference store 校验后传给 `voice` | 保持逻辑引用解析和音色 WAV 复用。 |
| `prompt_text` 或 sidecar | `params.reference_text` | controllable 模式继续省略参考转写与 sidecar。 |
| `text` | `input` | 接受单段；会触发 native 长文本切分的请求返回 422。 |
| `control_instruction` | 依据固定文档格式，前置为 `(instruction)text` | 仅用于 `clone_mode=controllable`；与 `prompt_text` 互斥规则不变。 |
| `cfg_value` | `params.guidance_scale` | 数值参数名按固定 audio.cpp 文档映射，语义和效果待真实模型验证。 |
| `inference_timesteps` | `params.num_inference_steps` | 数值参数名按固定 audio.cpp 文档映射，效果待真实模型验证。 |
| 单段限制 | `params.text_chunk_size=len(text)+1` | 防止 audio.cpp 默认 2048 字符切片改变一次性 native 请求；超过 native 分段阈值的请求仍拒绝。 |
| WAV 输出 | 原样返回并写入原 clone 输出目录 | LocalAI 响应由网关校验 RIFF/WAVE；真实采样率/质量待 smoke。 |

明确不支持的请求返回 422：`nonverbal_tags`、native `normalize`/`denoise`/`load_denoiser`/
`optimize`/`retry_badcase`、非随机 seed、CUDA device 0 以外设备以及会执行 native 分片的长文本。
当前服务默认 `VOXCPM2_RETRY_BADCASE=true`，因此调用方选 LocalAI 时需显式在请求中设置
`retry_badcase=false`；网关不会静默移除该 native 行为。

## 注册与验收

用户把权重放在 LocalAI models 目录的
`VoxCPM2-GGUF/voxcpm2-q8_0.gguf` 后运行：

```bash
python3 scripts/localai/manager.py register-voxcpm2-candidate
```

命令不下载权重；它写入固定 YAML 和本地文件 size/SHA-256 记录。该记录仅检测注册后变化，不能确认上游来源。
启动 preflight 会验证候选路径、配置与记录。服务回归覆盖 Ultimate/controllable 映射、sidecar 语义、明确拒绝
不支持选项、响应 WAV 路径和 LocalAI client 请求字段；候选登记测试只用临时假 GGUF。

尚未安装固定 `audio-cpp` gRPC backend，也没有用户模型权重，因此没有 E2 请求/响应或 E3 质量、延迟、显存、
进程退出证据。必须在模型准备和 backend 构建后完成 clone/controllable、中文/多语种、长文本拒绝、输出检查、
`/backend/shutdown`、native/LocalAI GPU 锁互斥以及异常清理 smoke。Native provider 仍为默认，模型依赖未删除。
