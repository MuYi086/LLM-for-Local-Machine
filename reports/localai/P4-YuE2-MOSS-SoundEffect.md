# P4 YuE2 与 MOSS-SoundEffect 能力边界

- 状态：`implemented_unverified`（通过 P7 原生 one-shot worker bridge）
- 日期：2026-10-21
- LocalAI 默认与服务 provider：均保持 native
- 模型权重：本次升级未主动下载；当前 HF 镜像目录的必需文件已通过只读检查，真实推理未验收

## YuE2-3B

固定 audio.cpp 源码将 YuE2 注册为 `family:yue2`、`task:gen`，组件目录默认包含主模型
`yue2-3b-q8_0.gguf`、VAE `yue2-vae-f16.gguf` 和四个配置/tokenizer sidecar。请求解析器要求非空
`style`，并从 `lyrics` 或 text input 获取歌词；`cot`、`abc`、seed、guidance 与 AR/NAR 采样配置
通过专用 request options 读取。

LocalAI v4.11.0 的 `/v1/sound-generation` body 有 `text`、`lyrics` 等音乐字段，但没有 `style`、
`cot`、`abc`、seed、逐项 sampling 或 options map。固定 gRPC translator 只将 text 设为
`text_input`，把 lyrics 设为 `options[lyrics]`，没有构造 YuE2 所需的 style condition。因此即使
能返回 WAV，也不能保留当前 Unitale `style/tags`、默认 `cot=full`、seed、ABC 与采样合同；本次
不添加无法调用的 LocalAI provider，不把其标为部分迁移。

P7 已定义版本化 `unitale.yue2-3b.worker.v1`，完整发送现有 worker JSON（style、tags、歌词、cot、seed、ABC、cfg_scale 和两个采样对象）并返回 WAV。可选 ABC/metadata 成品仍按原 worker 合同处理；native 默认不变。LocalAI 路由和字段传递有无模型回归，真实推理仍未验收。

## MOSS-SoundEffect v2

固定 audio.cpp family screen 没有 `MOSS-SoundEffect-v2.0` 的精确 family/task。Stable Audio、ACE-Step
和其他音效 family 均不能证明权重兼容或模型原生效果等价，因此 8312 通过
`unitale.moss-soundeffect-v2.worker.v1` 调度原 MOSS worker；native 仍是默认 provider，不从其他模型转接。

## 未完成的验收

上述判断完成接口/family 静态差距分析和无模型 bridge 实施。由于未下载权重，未执行 E2 请求、E3
音质/显存/进程退出或失败清理验收。模型由用户手动准备；本次变更没有下载模型。
