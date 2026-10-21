# P4 ACE-Step LocalAI 候选接入记录

更新日期：2026-10-21。状态：`implemented_unverified`（代码和无模型回归已实现；尚无 LocalAI 模型实测）。

## 能力范围

服务 `POST /v1/aceStep/bgm` 输出 BGM WAV，不接收歌词、编辑或续写字段。请求省略歌词与 native
worker 的空歌词值一致，但模型是否生成任何歌声仍待实测。默认 provider 仍是
`native`。显式选 `UNITALE_ACESTEP_PROVIDER=localai` 后，服务调用 LocalAI
`POST /v1/sound-generation`，保持现有响应类型与 `storage/bgm/` 落盘语义。

| Unitale 字段 | LocalAI 字段 | 处理方式 |
| --- | --- | --- |
| `prompt` | `text` | 直接映射 |
| `seconds` | `duration_seconds` | 直接映射，后端实际时长待实测 |
| `bpm` | `bpm` | 可选，直接映射 |
| `keyscale` | `keyscale` | 可选，直接映射 |
| `timesignature` | `timesignature` | 可选，直接映射 |
| 服务不接收歌词 | 省略 `lyrics` | audio.cpp 将缺省歌词解析为空字符串，与 native worker 一致；HTTP 的 `instrumental` 虽可接收，但固定引擎源码没有读取该选项的实现，不把它当作无歌声保证 |
| `steps` | 无 | 仅接受当前默认值 8；其他值在获取 GPU 锁前返回 422 |
| `seed` | 无 | 仅接受 `-1` 随机值；显式 seed 返回 422 |

返回 WAV 需要能由 SoundFile 分块完整解码、帧数大于 0、采样率为 48 kHz、双声道。验证通过后使用
原有 `persist_audio_bytes` 写入 BGM 输出目录。由于 SoundGeneration 不报告实际随机 seed，
`X-ACE-Step-Seed: -1` 表示请求后端随机生成，不是实际 seed 回报。

## 模型候选与版本

固定 audio.cpp 源码文档声明 ACE-Step XL Turbo 的 `family:ace_step`、`task:gen`。
固定的通用 `audio-cpp-gguf` 快照没有 Q8 DiT 候选文件。候选登记使用
[CaptainArni/audio.cpp-gguf revision `ac4d11eda6855ddedb16e42632e1a9376bfcfa81`](https://huggingface.co/CaptainArni/audio.cpp-gguf/commit/ac4d11eda6855ddedb16e42632e1a9376bfcfa81)，文件路径为
`ACE-Step1.5-GGUF/xl-turbo-q8dit/ace-step-1.5-xl-turbo-q8dit.gguf`。其
[固定 revision README](https://huggingface.co/CaptainArni/audio.cpp-gguf/blob/ac4d11eda6855ddedb16e42632e1a9376bfcfa81/README.md)
将该 Q8 DiT 文件列为 proposed，并注明不在路径测试矩阵；其余 planner、text encoder、VAE
使用 BF16。页面没有可核实的文件 SHA-256，因此当前只能记录固定候选 revision 和相对路径，不能
声称手动文件已证明来自该 revision，也不能将登记升为 E1 以上制品证据。

权重由用户手动准备，升级代码和 manager 均不下载：

```bash
python scripts/localai/manager.py register-ace-step-candidate
python scripts/localai/manager.py verify-ace-step-candidate
```

manager 会检查固定相对路径，原子安装模板 YAML，并记录本地文件 size/hash。该 hash 只检测登记后
本地文件是否变化；不能核验上游来源。启动前的 preflight 需要候选文件、配置、LocalAI 和模型注册状态
都已存在。

## 已做验证与未完成项

无模型回归覆盖 SoundGeneration JSON 字段、响应 WAV 格式边界、服务 BGM 存储/响应、steps 和 seed
拒绝、候选文件缺失提示、候选 YAML 注册及登记后本地 hash 变化。能力清单从源码同步字段和路由行号。

真实模型加载与音质、实际 BPM/调性控制、时长偏差、LocalAI backend 每请求退出、CUDA 回收时间和
显存峰值尚未测试。LocalAI 固定 backend 构建仍受 Ubuntu 24.04 amd64 gRPC/protobuf 开发依赖阻挡；
安装包解决后，仍需用户手动准备模型及素材再执行 E2/E3 smoke。native 回滚为
`UNITALE_ACESTEP_PROVIDER=native` 并重启服务。
