# 固定 audio.cpp 模型族筛选

- 日期：2026-10-21
- 上游 revision：9a02e61326aaaf9d462b584ca5e0daba22c0abfc
- 证据等级：E1（固定 README 中的 family/task/variant 声明）
- 当前清单：12 个服务入口、11 个模型 uv 项目与 8300 控制面
- 源码校验：localai/manifests/capabilities.json

本文只记录固定 audio.cpp README 的上游声明。family 匹配不证明当前存在模型制品、
provider 代码、请求字段映射、推理兼容性或质量验收。

## 仍有上游 family 对应关系的服务

| 服务 | 固定上游 family | 当前状态 |
| --- | --- | --- |
| qwen3_tts | qwen3_tts | Base 克隆 provider 保留；本机 HF 模型目录存在 |
| qwen3_voiceDesign | qwen3_tts | VoiceDesign provider 保留；本机 HF 模型目录存在 |
| SheetSage2 | sheetsage2 | 部分候选；当前走固定 one-shot worker bridge |
| YuE2-3B | yue2 | 上游 family 声明保留作 E1 记录；当前走固定 one-shot worker bridge |

只有 Qwen3-TTS Base 与 VoiceDesign 当前保留直接 LocalAI audio.cpp provider。
SheetSage2 和 YuE2-3B 使用 llm-local.worker bridge，不映射为直接 audio.cpp 推理。

## 当前没有保留直接候选的服务

| 服务 | 筛选结果 |
| --- | --- |
| Qwen3-ASR | audio.cpp 声明 qwen3_asr，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |
| VoxCPM2 | 上游声明 voxcpm2，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |
| Confucius4-TTS | 上游声明 confucius4_tts，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |
| ACE-Step | 上游声明 ace_step，但 GGUF 不在 hf-mirror；直接 provider 已移除，原生服务保留 |

## 无精确模型族

| 服务 | 结论 |
| --- | --- |
| MERT-v2-FullSong | 无同名整曲表征模型；当前通过 worker bridge 保留原生 JSON/NPZ 合同 |
| TIGER-DnR | 无同名三 Stem 分离模型；当前通过 worker bridge 保留原生 ZIP 合同 |
| YuE2-Vae | 歌曲生成 family 不等同于独立 latent encode/decode；当前通过 worker bridge 保留原生合同 |

固定上游目录中的任务和模型变体仍按 E1 维护；能力清单用 excluded 标记未在镜像中准备模型制品的直接候选。
不应将 family 标签写成已实现 provider，也不应据此宣称 E2/E3。