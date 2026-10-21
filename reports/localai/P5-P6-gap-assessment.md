# P5/P6 能力缺口与执行状态

- 日期：2026-10-21
- 范围：P5 编辑/理解/分离/转换/S2ST 与 P6 结构化音乐任务
- 依据：仓库现有请求/worker 合同；固定 audio.cpp revision `9a02e61326aaaf9d462b584ca5e0daba22c0abfc` 的 family 清单
- 结论：完成候选/合同静态核对；除 Seed-VC 外没有新增错误的近似 family 接入。任何本机推理与输出均未据此宣称通过 E2/E3。

## P5

| 服务/能力 | 当前判断 | 依据与保持的行为 | 下一步门槛 |
| --- | --- | --- | --- |
| `Step_Audio_EditX` 编辑 | `implemented_unverified` | 固定 family 无 Step-Audio-EditX。P7 `unitale.step-audio-editx.worker.v1` 保留原 worker JSON、编辑字段和 WAV 输出；native 仍默认。 | 用户准备权重后验证编辑、重复请求、进程退出和 LocalAI shutdown。 |
| `moss_audio_4b_thinking` 理解 | `implemented_unverified` | 固定 family 无 MOSS-Audio Thinking/Instruct。P7 为两个变体分别固定 worker ID，保留音频输入与 JSON 文本响应；不以 ASR 代替理解。 | 用户准备权重后分别验证变体响应、时间问答、推理字段和 worker 清理。 |
| `TIGER-DnR` 分离 | `implemented_unverified` | 固定 family 无 TIGER-DnR。P7 以不透明随机文件引用传递原 worker 生成的三 Stem ZIP，服务端校验目录、大小、MIME 和固定文件名，再沿用原子批次保存；不使用 HTDemucs 替代。 | 用户准备权重后验证三 Stem 同步、长音频、4 GiB 边界、LocalAI shutdown 与异常清理。 |
| `seed-vc` VC/SVC | `implemented_unverified` 候选 | `seed_vc/main.py` 增加显式 LocalAI `AudioTransform` provider，分开 `task:vc`/`v2_vc` 与 `task:svc`/`v1_svc`，解析逻辑音频引用并保留 WAV 响应。固定 family README 声明 VC/SVC，但目录 revision 和制品 hash 未固定。native 仍为默认。 | 构建并运行固定 backend；用户准备模型后分别完成 VC/SVC E2/E3、输出采样属性、质量、关闭确认和 GPU 锁释放。 |

## P6

| 服务/能力 | 当前判断 | 依据与保持的行为 | 下一步门槛 |
| --- | --- | --- | --- |
| `SheetSage2` 转谱 | `partial_family_candidate` | 固定 family 仅声明音频转 ABC；Unitale 还返回 MIDI、和弦/节拍/调性/结构/时间标注 ZIP，并依赖固定 revision 的 MERT 父模型。P7 bridge 保留原生 ZIP 成品。 | 用户准备权重后验证父模型版本、完整 ZIP 内容和 E2/E3。 |
| `YuE2-Vae` latent encode/decode | `implemented_unverified` | `yue2` 歌曲生成不等于独立 VAE API；P7 bridge 保留 `[T,64]` NPY、FP32 latent 和官方分块 WAV decode。 | 用户准备权重后分别验证 48 kHz 立体声 encode、latent shape/dtype 与分块 decode。 |
| `MERT-v2-FullSong` 表征 | `implemented_unverified` | 固定 family 无 MERT-v2-FullSong；P7 bridge 保留 1024 维整曲 JSON/25 Hz 逐帧 NPZ，及 30–360 秒/24 kHz mono 预检。 | 用户准备权重后验证 masked mean、逐帧时间轴、长音频边界和 E2/E3。 |

## 共同阻塞与顺序

1. 用户提供的 Ubuntu 24.04 amd64 deb 已在隔离前缀解包，`grpc-server` 与 backend package 已构建并通过 `manager verify-install`；详细依赖闭包见 [P1/P2 报告](P1-P2.md)。
2. 不通过 Docker 构建，也没有使用 apt/dpkg 系统安装；backend 只放在忽略的本机运行目录。
3. bridge 集成状态不代表 P5/P6 真实模型验收完成；模型由用户手动准备，本轮未下载。
4. TIGER-DnR 已接 P7 file-reference ZIP bridge；仍需用户准备模型后验证三路输出、长音频与 E2/E3。
