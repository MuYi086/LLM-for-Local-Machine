# 固定 audio.cpp 模型族筛选

- 日期：2026-10-20
- 上游 revision：`9a02e61326aaaf9d462b584ca5e0daba22c0abfc`
- 证据等级：E1（固定 README 中的 family/task/variant 声明）
- 服务覆盖：24 个仓库入口；13 个 family 候选、9 个无精确模型族、2 个范围外服务
- 机器清单：[`capabilities.json`](../../localai/manifests/capabilities.json) 中的 `audio_cpp_family_screen`

这次筛选只回答固定 audio.cpp README 是否声明了同一模型族及相关任务。它不证明 LocalAI 固定
backend 已暴露该 family、实际模型包可加载、请求参数可传递、输出符合现有 API，或本机质量达标。
所有候选仍需先完成 LocalAI E2 和本机 E3 验收，再考虑 provider 切换。

## 已列出的 family 候选

| 仓库服务 | 固定上游 family | README 声明任务 | 判断与待核对项 |
| --- | --- | --- | --- |
| `Confucius4_TTS` | `confucius4_tts` | Clone | 同名模型族候选；核对准确模型包、语言与请求/响应映射。 |
| `Qwen3_ASR_1.7B` | `qwen3_asr` | ASR | 声明 1.7B-hf 及 GGUF 16/Q8；权重 metadata、sidecar、LocalAI 加载和 context/token 映射待验证。 |
| `SheetSage2` | `sheetsage2` | MIDI | **部分候选**：README 声明音频转 ABC；不证明当前 MIDI、和弦/节拍/调性/结构与 ZIP 输出合同完整。 |
| `YuE2-3B` | `yue2` | Music | 声明 YuE2-3B 歌词生成和可选 ABC；完整模型包与采样参数映射待验证。 |
| `ace_step_1_5` | `ace_step` | Music, Edit | 声明 ACE-Step 1.5 XL Turbo；需核对当前服务模式及所有编辑参数。 |
| `dots_tts_soar` | `dots_tts` | TTS, Clone, Edit, Ctrl | 声明 DotTTS SOAR；跨语言克隆、参考条件和输出仍需逐字段验证。 |
| `firered_tts3` | `fireredtts3` | TTS, Clone, Design, Ctrl | 声明 Base 和 Instruct/VoiceDesign；两个运行模式、输出字段及权重包待核对。 |
| `moss_voiceGenerator` | `moss_voicegen` | Voice Design | 社区模型目录列出 MOSS-VoiceGenerator；模型包与设计流程映射待核对。 |
| `qwen3_tts` | `qwen3_tts` | TTS, Clone, Design, Ctrl | 声明 Base、CustomVoice、VoiceDesign 变体；当前克隆字段仍需逐项映射。 |
| `qwen3_voiceDesign` | `qwen3_tts` | TTS, Clone, Design, Ctrl | VoiceDesign 变体有明确声明；不得误映射到 CustomVoice。 |
| `seed-vc` | `seed_vc` | VC | **部分候选**：声明 SeedVC voice conversion；任务表没有证明 F0 条件 singing conversion (SVC)。 |
| `stable_audio_3_medium` | `stable_audio` | Music, SFX, Edit | 声明 Stable Audio 3 Medium；生成、补全和续写参数待逐项验证。 |
| `voxcpm2` | `voxcpm2` | TTS, Clone, Design, Ctrl | 声明 VoxCPM2-2B；可控克隆指令、标签限制及输出映射待验证。 |

目录中的任务标签和模型变体见[固定 revision README 的 speech 表](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/README.md#L67-L120)、
[音频处理表](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/README.md#L121-L139)、
[音乐表](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/README.md#L140-L151)
及[community model 表](https://github.com/0xShug0/audio.cpp/blob/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/README.md#L152-L188)。

## 固定目录没有精确模型族

| 仓库服务 | 结论 |
| --- | --- |
| `LongCat_AudioDiT_3.5B_bf16` | 没有 LongCat AudioDiT；其他 TTS/Clone family 不代表该扩散模型兼容。 |
| `MERT-v2-FullSong` | 没有 MERT-v2-FullSong 表征模型；其他 embedding 不等价于整曲/逐帧输出。 |
| `SoulX-FlashHead-1_3B` | 没有 SoulX 图像加音频说话视频模型。 |
| `Step_Audio_EditX` | 没有 Step-Audio-EditX；其他带 Edit 标签的模型不是该模型。 |
| `TIGER-DnR` | 没有 TIGER-DnR；其他分离 family 不保证对白/音效/音乐三路 Stem 语义。 |
| `YuE2-Vae` | YuE2 歌曲生成不声明独立 latent encode/decode 合同。 |
| `ditto` | 没有 Ditto 图像加音频说话视频模型。 |
| `moss_audio_4b_thinking` | 没有 MOSS-Audio 4B Thinking/Instruct；转写或其他音频理解模型不能替代。 |
| `moss_soundEffect` | 没有该 MOSS SoundEffect 模型；其他 SFX family 属于不同模型。 |

`main` 控制面/CPU 导出和 `mimo_tts` 云服务是两个范围外边界，不计入本地模型候选。
Ruff/源码门禁会检查能力清单对全部服务的覆盖，以及筛选结果总数；清单仍需随固定上游 revision
变更显式更新。运行 `uv run --project qa --locked python scripts/localai/validate_capabilities.py`。

## 下游执行顺序

先在可构建宿主机 backend 且有 GPU 访问的环境完成 P1 固定 LocalAI/backend 安装及生命周期实测，然后完成 P2 Qwen3-ASR
E2/E3。其后只对上表 family 候选逐个检查 LocalAI RPC/模型包和字段合同；部分候选必须先补出独立能力缺口，
无精确模型族的本地服务进入 P7 原生 worker bridge，不把不同模型的相同任务标签当作替换依据。
