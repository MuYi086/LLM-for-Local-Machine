# P4 Stable Audio 3 Medium LocalAI 候选接入记录

更新日期：2026-10-21。状态：`implemented_unverified`（服务适配和无模型回归已实现，尚无真实模型 smoke）。

## 能力范围与字段

`POST /v1/stableAudio/soundEffect` 当前是文本提示词加时长的音乐/音效生成接口，不接收源音频。
`native` 仍为默认；LocalAI provider 调用 `/v1/sound-generation` 并保持原有 `audio/wav` 响应和
`storage/soundEffect/` 落盘位置。

| Unitale 字段 | LocalAI SoundGeneration | 处理方式 |
| --- | --- | --- |
| `prompt` | `text` | 直接映射 |
| `seconds` / `duration` | `duration_seconds` | 请求模型已将兼容别名归一到 `seconds` |
| `steps` | 无 | 仅接受服务默认 8，其他值返回 422 |
| `cfg_scale` | 无逐请求 guidance scale | 仅接受服务默认 1.0，其他值返回 422 |
| `seed` | 无 | 仅接受 `-1` 后端随机值，显式 seed 返回 422 |
| `dtype` | 无；候选为 Q8 GGUF | 显式 dtype 返回 422 |
| `device` | 模型 YAML 固定 `backend:cuda`、`device:0` | 当前接口只允许 `cuda`，不提供其他设备选择 |

Service request 不含 `audio`、mask 或 edit mode；LocalAI provider 不宣称迁移 Stable Audio 的
init-audio 或 inpaint。返回 WAV 使用 SoundFile 分块完整解码，检查帧数、采样率 44.1 kHz、双声道，
然后调用现有存储函数。

## 候选版本

固定 audio.cpp 源码文档声明 Stable Audio 3 Medium 使用 `family:stable_audio` 和 `task:gen`。
固定模型包 revision 为
[`audio-cpp/audio.cpp-gguf@09fe073ba154561f4474162e8bd4ab233a848eca`](https://huggingface.co/audio-cpp/audio.cpp-gguf/commit/09fe073ba154561f4474162e8bd4ab233a848eca)，README 列出
`Stable-Audio-3-Medium-GGUF/stable-audio-3-medium-q8_0.gguf`。没有登记可验证的文件 SHA-256
或大小；本地 hash 只用来判断文件登记后是否变化。原模型使用 Stability AI Community License，
商业分发或使用前应核对上游当前许可文本。

用户手动准备权重后运行：

```bash
python scripts/localai/manager.py register-stable-audio-candidate
python scripts/localai/manager.py verify-stable-audio-candidate
```

上述命令不会下载模型。`start.sh` 启动预检会再次检查手动制品、固定 YAML、LocalAI 健康和模型注册状态。

## 验证与剩余工作

无模型回归覆盖默认 provider、SoundGeneration 字段/响应、存储结果、unsupported sampling 参数拒绝、
输出 WAV 的完整分块读取、候选配置登记、本地 hash 变化和缺权重时不下载。

待完成：用户提供候选模型后验证文件来源及 SHA-256；在固定宿主机 audio-cpp backend 上验证 Q8
GGUF 加载、44.1 kHz 双声道输出、默认 guidance/seed 语义、输出时长、Q8 音质、backend 停止和 CUDA
回收。固定 backend 构建当前仍等待 Ubuntu 24.04 amd64 gRPC/protobuf 开发包。回滚方式是恢复
`UNITALE_STABLE_AUDIO_3_MEDIUM_PROVIDER=native` 并重启。
