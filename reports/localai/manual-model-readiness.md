# 本地模型镜像与外部依赖状态

- 检查日期：2026-10-21
- 检查根目录：~/hf-mirror
- 本轮未下载模型或安装 Python 项目依赖。
- 处理规则：完整模型目录不存在，或服务必需辅助模型缺失时，移除该服务及其运行引用；不保留待下载的缺失模型服务。
- 当前保留范围：11 个模型服务 uv 项目、1 个控制面，共 12 个 HTTP 进程。

## 当前服务模型根目录

下列 11 个服务模型根目录均存在于镜像中。目录存在不等于所有权重、Python 依赖或真实推理已经验收。

| 服务 | 镜像相对路径 | 目录检查 |
| --- | --- | --- |
| Qwen3-TTS Base | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 存在 |
| Qwen3 VoiceDesign | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | 存在 |
| Qwen3-ASR | Qwen/Qwen3-ASR-1.7B | 存在 |
| VoxCPM2 | openbmb/VoxCPM2 | 存在 |
| Confucius4-TTS | netease-youdao/Confucius4-TTS | 存在 |
| ACE-Step 1.5 XL Turbo | ACE-Step/acestep-v15-xl-turbo-diffusers | 存在 |
| YuE2-3B | m-a-p/YuE2-3B | 存在 |
| YuE2-Vae | m-a-p/YuE2-Vae | 存在 |
| SheetSage2 | m-a-p/SheetSage2 | 存在 |
| MERT-v2-FullSong | m-a-p/MERT-v2-FullSong | 存在 |
| TIGER-DnR | JusperLee/TIGER-DnR | 存在 |

## 已按规则退役

| 功能/制品 | 镜像检查 | 清理结果 |
| --- | --- | --- |
| Stable Audio 3 Medium | stabilityai/stable-audio-3-medium 不存在 | 服务目录、路由、worker、启动项、维护前缀和测试已移除 |
| Ditto | digital-avatar/ditto-talkinghead 不存在 | 服务代码、路由、worker、启动项、维护项和测试已移除；本机忽略的运行环境残余已归入 `model-video/ditto/` |
| Seed-VC | 主目录 Plachta/Seed-VC 存在，但必需的 openai/whisper-small 缺失；F0/SVC 另缺 44 kHz BigVGAN 和 RMVPE | 整个服务及 VC/SVC LocalAI 候选已移除 |
| SoulX-FlashHead | 主目录存在，但必需的 facebook/wav2vec2-base-960h 缺失 | 服务代码、图片上传 API、启动项和 worker bridge 已移除；本机忽略的运行环境残余已归入 `model-video/SoulX-FlashHead-1_3B/` |
| Qwen3-ASR LocalAI Q8 | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 Qwen3-ASR 服务保留 |
| VoxCPM2 LocalAI Q8 | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 VoxCPM2 服务保留 |
| Confucius4-TTS LocalAI GGUF | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 Confucius4-TTS 服务保留 |
| ACE-Step LocalAI Q8 DiT | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 ACE-Step 服务保留 |

已退役服务不再列入人工补模清单；重新启用必须先重新评估模型、依赖和维护范围。

## 当前保留的 LocalAI 路径

Qwen3-TTS Base 与 VoiceDesign 的官方 Hugging Face 模型目录存在，直接候选 YAML 与只读文件预检保留。
one-shot worker bridge 保留 SheetSage2、MERT-v2-FullSong、YuE2-Vae、YuE2-3B、TIGER-DnR 五个模型 ID。
这些是本地模型服务的执行后端，不额外要求 audio.cpp GGUF。

LocalAI v4.11.0 二进制和 CUDA audio-cpp backend 已在仓库忽略运行目录中存在。manager 的文件指纹核验和无模型启动/停止记录见升级报告。
无模型检查不验证 TTS 模型加载、音质、峰值显存或 backend shutdown 后的 GPU 回落。

## 外部依赖状态

| 依赖 | 检查结果 | 后续操作 |
| --- | --- | --- |
| Steam Audio SDK 4.8.1 | 已安装于 `~/.local/share/steam-audio/4.8.1`；renderer 已构建，控制面诊断可用 | Linux `start.sh` 默认使用该目录；自定义 SDK 位置时设置 `STEAM_AUDIO_SDK_DIR` |
| Confucius4-TTS 上游源码 | ~/LLM-depency/Confucius4-TTS 存在 | 无需补充 |
| TIGER-DnR 上游源码 | ~/.local/share/tiger-dnr/TIGER 存在 | 无需补充 |
| LocalAI v4.11.0 binary/backend | storage/.cache/localai 下均存在 | 无需补充 |
| Python 锁定依赖 | 11 个服务项目、worker bridge 与 QA 的离线 `uv sync --locked --dry-run` 均为 `Would make no changes` | 没有检测到待同步的 Python 包；检查未修改环境 |
| uv、FFmpeg、CMake、G++ | 命令可用 | 无需补充 |

当前没有因 hf-mirror 根目录缺失而等待用户下载的保留服务主模型，也没有已知待补充的外部依赖。Steam Audio SDK 是空间导出的外部开发依赖，不是模型权重，也不影响 LocalAI/audio.cpp backend。

## 验收边界

质量门禁只验证锁、静态能力清单、无模型测试、脚本和 C++ 核心逻辑；不会下载模型、执行 CUDA 推理或证明真实输出质量。
模型加载和端到端输出只在已有本地模型与依赖完整时单独运行。
