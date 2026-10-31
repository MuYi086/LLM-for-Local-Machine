# 固定 audio.cpp 模型族筛选

- 日期：2026-10-21
- 上游 revision：9a02e61326aaaf9d462b584ca5e0daba22c0abfc
- 证据等级：E1（固定 README 中的 family/task/variant 声明）
- 部署总范围：10 个模型服务与 8300 控制面，共 11 个 HTTP 入口
- 能力清单范围：8 个服务入口，含 7 个模型 uv 项目与 8300 控制面；此清单不覆盖全部部署服务
- 源码校验：localai/manifests/capabilities.json

本文只记录固定 audio.cpp README 的上游声明。family 匹配不证明当前存在模型制品、
provider 代码、请求字段映射、推理兼容性或质量验收。

## 仍有上游 family 对应关系的服务

| 服务 | 固定上游 family | 当前状态 |
| --- | --- | --- |
| qwen3_tts | qwen3_tts | Base 克隆 provider 保留；本机 HF 模型目录存在 |
| qwen3_voiceDesign | qwen3_tts | VoiceDesign provider 保留；本机 HF 模型目录存在 |

只有 Qwen3-TTS Base 与 VoiceDesign 当前保留直接 LocalAI audio.cpp provider。

## 当前没有保留直接候选的服务

| 服务 | 筛选结果 |
| --- | --- |
| Qwen3-ASR | audio.cpp 声明 qwen3_asr，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |
| Confucius4-TTS | 上游声明 confucius4_tts，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |

## 无精确模型族

| 服务 | 结论 |
| --- | --- |
| TIGER-DnR | 无同名三 Stem 分离模型；当前通过 worker bridge 保留原生 ZIP 合同 |

固定上游目录中的任务和模型变体仍按 E1 维护；能力清单用 excluded 标记未在镜像中准备模型制品的直接候选。
不应将 family 标签写成已实现 provider，也不应据此宣称 E2/E3。