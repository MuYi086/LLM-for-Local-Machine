# LocalAI 手动模型准备状态

- 检查日期：2026-10-21
- 权重下载：本次未执行
- 检查范围：当前 `LocalAI` audio.cpp 直接候选与 worker bridge 所需的原生服务模型

## 尚缺的直接候选制品

以下文件均未在 `storage/.cache/localai/models/`、`$HF_MIRROR_DIR`、`$HOME/tts-depency` 或 Hugging Face hub 缓存中找到。
下载后放进 `storage/.cache/localai/models/`，保留下列相对目录结构。

| 能力 | 上游固定版本 | 文件下载地址 | 目标相对路径 |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | `audio-cpp/audio.cpp-gguf@09fe073ba154561f4474162e8bd4ab233a848eca`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/09fe073ba154561f4474162e8bd4ab233a848eca)） | [qwen3-asr-1.7b-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/09fe073ba154561f4474162e8bd4ab233a848eca/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf) | `Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf` |
| dots.tts-soar | `audio-cpp/audio.cpp-gguf@83c5d96c03023ff5a7712570d057ce26c8769f98`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)） | [dots-tts-soar-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/83c5d96c03023ff5a7712570d057ce26c8769f98/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf) | `DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf` |
| VoxCPM2 | `audio-cpp/audio.cpp-gguf@09fe073ba154561f4474162e8bd4ab233a848eca`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/09fe073ba154561f4474162e8bd4ab233a848eca)） | [voxcpm2-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/09fe073ba154561f4474162e8bd4ab233a848eca/VoxCPM2-GGUF/voxcpm2-q8_0.gguf) | `VoxCPM2-GGUF/voxcpm2-q8_0.gguf` |
| Confucius4-TTS | `audio-cpp/audio.cpp-gguf@83c5d96c03023ff5a7712570d057ce26c8769f98`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)） | [confucius4-tts-orig.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/83c5d96c03023ff5a7712570d057ce26c8769f98/Confucius4-TTS-GGUF/confucius4-tts-orig.gguf) | `Confucius4-TTS-GGUF/confucius4-tts-orig.gguf` |
| MOSS VoiceGenerator | `audio-cpp/audio.cpp-gguf@83c5d96c03023ff5a7712570d057ce26c8769f98`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)） | [moss_voicegen_bf16_codec_f16_decode.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/83c5d96c03023ff5a7712570d057ce26c8769f98/MOSS-VoiceGenerator-GGUF/moss_voicegen_bf16_codec_f16_decode.gguf) | `MOSS-VoiceGenerator-GGUF/moss_voicegen_bf16_codec_f16_decode.gguf` |
| FireRedTTS3 Base | `audio-cpp/audio.cpp-gguf@83c5d96c03023ff5a7712570d057ce26c8769f98`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)） | [fireredtts3-base-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/83c5d96c03023ff5a7712570d057ce26c8769f98/FireRedTTS3-Base-GGUF/fireredtts3-base-q8_0.gguf) | `FireRedTTS3-Base-GGUF/fireredtts3-base-q8_0.gguf` |
| FireRedTTS3 Instruct | `audio-cpp/audio.cpp-gguf@83c5d96c03023ff5a7712570d057ce26c8769f98`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/83c5d96c03023ff5a7712570d057ce26c8769f98)） | [fireredtts3-instruct-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/83c5d96c03023ff5a7712570d057ce26c8769f98/FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-q8_0.gguf) | `FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-q8_0.gguf` |
| ACE-Step 1.5 XL Turbo | `CaptainArni/audio.cpp-gguf@ac4d11eda6855ddedb16e42632e1a9376bfcfa81`（[固定提交](https://huggingface.co/CaptainArni/audio.cpp-gguf/commit/ac4d11eda6855ddedb16e42632e1a9376bfcfa81)） | [ace-step-1.5-xl-turbo-q8dit.gguf](https://huggingface.co/CaptainArni/audio.cpp-gguf/resolve/ac4d11eda6855ddedb16e42632e1a9376bfcfa81/ACE-Step1.5-GGUF/xl-turbo-q8dit/ace-step-1.5-xl-turbo-q8dit.gguf) | `ACE-Step1.5-GGUF/xl-turbo-q8dit/ace-step-1.5-xl-turbo-q8dit.gguf` |
| Stable Audio 3 Medium | `audio-cpp/audio.cpp-gguf@09fe073ba154561f4474162e8bd4ab233a848eca`（[固定快照](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/09fe073ba154561f4474162e8bd4ab233a848eca)） | [stable-audio-3-medium-q8_0.gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/09fe073ba154561f4474162e8bd4ab233a848eca/Stable-Audio-3-Medium-GGUF/stable-audio-3-medium-q8_0.gguf) | `Stable-Audio-3-Medium-GGUF/stable-audio-3-medium-q8_0.gguf` |
| Seed-VC VC/SVC | 固定 audio.cpp 文档未固定模型包 revision 或文件 hash（[模型文档目录](https://github.com/0xShug0/audio.cpp/tree/9a02e61326aaaf9d462b584ca5e0daba22c0abfc/backend/cpp/audio-cpp/audio.cpp/docs)） | 需手动准备完整 `SeedVC-MLX` 模型目录 | `SeedVC-MLX/` |

Qwen3-ASR 的 Q8 文件在版本清单中固定为 2,473,010,048 bytes 和 SHA-256
`da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8`。其余 GGUF 文件尚无上游 LFS hash；
下载后注册会记录本地 hash，仅用于检测之后的文件变化。ACE-Step Q8 DiT 仍是 E0 候选，不能据此宣称
音频路径兼容已验证。

## 已准备且不需重复下载

- Qwen3-TTS Base 与 VoiceDesign 的官方 safetensors 所需文件已核验，候选 YAML 已登记在忽略的模型配置目录。
- `start.sh` 当前默认 HF 镜像路径检查为 36/36 存在。23 个服务的只读 health 检查未报告必需模型文件或源码路径缺失；
  这只表示文件/目录检查通过，不代表真实推理质量或硬件资源验收通过。
- P7 worker bridge 的 11 个固定模型 YAML 已登记并核验。`MOSS-Audio` Thinking/Instruct 共用一个服务项目。
- 仓库中的 25 个锁定 uv 项目已用 `uv sync --locked --offline` 同步，无缺失 wheel；
  `manager.py verify-install` 通过，当前不需要额外 deb。flash-attn、Conda 及报告里的运行参数 false 项为可选优化、
非采用环境或默认关闭的功能，不是本次缺失依赖。

模型放置后应先执行对应 `register-*-candidate` 与 `verify-*-candidate`；这些步骤不会下载权重。
之后还需在用户提供的样本上完成 E2/E3、模型加载后 backend shutdown/显存回落和 P8 观察，
因此软件侧完成不代表模型迁移已通过生产验收。
