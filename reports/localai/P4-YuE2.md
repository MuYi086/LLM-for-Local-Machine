# P4：YuE2-3B audio.cpp 能力边界

- 状态：`implemented_unverified`（通过 P7 one-shot worker bridge）
- 日期：2026-10-21
- LocalAI 与服务 provider 默认均保持 `native`
- 模型权重：本轮未下载；真实推理未验收

固定 audio.cpp 源码将 YuE2 注册为 `family:yue2`、`task:gen`，请求解析要求非空 `style`，并从
`lyrics` 或 text input 获取歌词；`cot`、`abc`、seed、guidance 与 AR/NAR 采样配置通过专用 request
options 读取。

LocalAI v4.11.0 的 `/v1/sound-generation` body 没有 `style`、`cot`、`abc`、seed、逐项 sampling
或 options map。固定 gRPC translator 不能构造 YuE2 所需的 style condition。因此，即使该请求能返回
WAV，也不能保留 LLM Local Backend 当前 `style/tags`、`cot`、seed、ABC 与采样合同；当前不通过 SoundGeneration
接入 YuE2。

P7 的 `llm-local.yue2-3b.worker.v1` 将现有 worker JSON 原样发送，包括 style、tags、歌词、cot、seed、ABC、
`cfg_scale` 和两个 sampling 对象，并返回 WAV。native 默认不变；真实请求、质量、资源和停止行为待验收。

无模型 bridge 测试只证明字段传递和产物协议，不证明上游模型行为。模型由用户手动准备，本轮没有下载模型。
