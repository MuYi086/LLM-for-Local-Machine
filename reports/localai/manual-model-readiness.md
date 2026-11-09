# 本地模型镜像与外部依赖状态

- 历史检查日期：2026-10-21；当前维护范围和权重目录复核：2026-11-01
- 检查根目录：~/hf-mirror
- 原始盘点未下载模型；后续接入使用已准备的本地模型，并同步独立项目的锁定依赖。
- 处理规则：完整模型目录不存在，或服务必需辅助模型缺失时，移除该服务及其运行引用；不保留待下载的缺失模型服务。
- 当前保留范围：9 个模型服务 uv 项目、1 个控制面，共 10 个 HTTP 进程。
- LongCat Avatar 与 MiniMax H3 已于 2026-11-02 退役，不再列入当前部署和模型准备清单。

## 当前服务模型根目录

下表列出当前 9 个服务的模型根目录。目录存在不等于所有权重、Python 依赖或真实推理已经验收。

| 服务 | 镜像相对路径 | 目录检查 |
| --- | --- | --- |
| Qwen3-TTS Base | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 存在 |
| Qwen3 VoiceDesign | audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF | BF16 GGUF 存在；SHA-256 已登记；一次短句推理及 WAV 校验通过，质量待评估 |
| Qwen3-ASR | Qwen/Qwen3-ASR-1.7B | 存在 |
| Confucius4-TTS | netease-youdao/Confucius4-TTS | 存在 |
| TIGER-DnR | JusperLee/TIGER-DnR | 存在 |
| MOSS-Music-8B-Instruct GGUF | OpenMOSS/moss-music | 存在 |
| IndexTTS-2.5 | IndexTeam/IndexTTS-2.5 | 2026-11-01 目录存在 |
| MOSS-SoundEffect v2.0 | OpenMOSS/MOSS-SoundEffect-v2.0 | 2026-11-01 目录存在 |
| MSST Mega 53 | oulianov/mvsep_mega_53 | 2026-11-01 目录存在 |

## 已按规则退役

| 功能/制品 | 镜像检查 | 清理结果 |
| --- | --- | --- |
| Stable Audio 3 Medium | stabilityai/stable-audio-3-medium 不存在 | 服务目录、路由、worker、启动项、维护前缀和测试已移除 |
| Seed-VC | 主目录 Plachta/Seed-VC 存在，但必需的 openai/whisper-small 缺失；F0/SVC 另缺 44 kHz BigVGAN 和 RMVPE | 整个服务及 VC/SVC LocalAI 候选已移除 |
| Qwen3-ASR LocalAI Q8 | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 Qwen3-ASR 服务保留 |
| Confucius4-TTS LocalAI GGUF | GGUF 不在 hf-mirror | 只移除 LocalAI provider、候选 YAML、预检和候选测试；原生 Confucius4-TTS 服务保留 |

已退役服务不再列入人工补模清单；重新启用必须先重新评估模型、依赖和维护范围。

## 当前保留的 LocalAI 路径

Qwen3-TTS Base 的官方 Hugging Face 模型目录存在；VoiceDesign 使用本地 audio.cpp BF16 GGUF 文件。两个直接候选均只读检查源模型制品，VoiceDesign 的真实短句与 WAV 输出校验已完成。
one-shot worker bridge 保留 TIGER-DnR 一个模型 ID。
该 bridge 是本地模型服务的执行后端，不额外要求 audio.cpp GGUF。

LocalAI v4.11.0 二进制和 CUDA audio-cpp backend 已在仓库忽略运行目录中存在。manager 的文件指纹核验和无模型启动/停止记录见升级报告。
无模型检查不验证 TTS 模型加载、音质、峰值显存或 backend shutdown 后的 GPU 回落。

## 外部依赖状态

| 依赖 | 检查结果 | 后续操作 |
| --- | --- | --- |
| Steam Audio SDK 4.8.1 | 已安装于 `~/.local/share/steam-audio/4.8.1`；renderer 已构建，控制面诊断可用 | Linux `start.sh` 默认使用该目录；自定义 SDK 位置时设置 `STEAM_AUDIO_SDK_DIR` |
| Confucius4-TTS 上游源码 | ~/LLM-depency/Confucius4-TTS 存在 | 无需补充 |
| TIGER-DnR 上游源码 | ~/.local/share/tiger-dnr/TIGER 存在 | 无需补充 |
| LocalAI v4.11.0 binary/backend | storage/.cache/localai 下均存在 | 无需补充 |
| Python 锁定依赖（2026-10-21 历史记录） | 当时 11 个服务项目、worker bridge 与 QA 的离线 `uv sync --locked --dry-run` 均为 `Would make no changes` | 没有检测到待同步的 Python 包；检查未修改环境 |
| uv、FFmpeg、CMake、G++ | 命令可用 | 无需补充 |

上表外部依赖检查是对应日期的记录，部署时仍须以服务健康响应核验。Steam Audio SDK 是空间导出的外部开发依赖，
不是模型权重，也不影响 LocalAI/audio.cpp backend。

## 验收边界

质量门禁只验证锁、静态能力清单、无模型测试、脚本和 C++ 核心逻辑；不会下载模型、执行 CUDA 推理或证明真实输出质量。
模型加载和端到端输出只在已有本地模型与依赖完整时单独运行。
