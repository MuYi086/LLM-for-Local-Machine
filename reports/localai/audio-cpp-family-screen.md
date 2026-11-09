# audio.cpp 模型族与本地候选筛选

- 日期：2026-11-09
- 上游 revision：9a02e61326aaaf9d462b584ca5e0daba22c0abfc
- 证据等级：E1（固定上游 README 的 family/task/variant 声明）
- 当前范围：九个 LocalAI/audio.cpp GGUF 模型服务、8300 控制面与 Steam Audio CPU renderer
- 活跃 API 快照：localai/manifests/capabilities.json
- 本地候选清单：localai/manifests/audio-cpp-models.json

family/task 与本地 GGUF 文件存在性不证明模型请求已成功、输出质量已验收或峰值资源适合生产。
下表记录本机候选制品已做只读存在性检查；真实推理结果由逐模型生成记录补充。

## 当前候选

| 模型服务 | LocalAI family/task | 本机 GGUF | 服务端口 |
| --- | --- | --- | ---: |
| Qwen3-TTS VoiceDesign | qwen3_tts / vdes | qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf | 8301 |
| Confucius4-TTS | confucius4_tts / clon | confucius4-tts-orig.gguf | 8331 |
| IndexTTS2.5 | index_tts2 / clon | index-tts2_5-f16.gguf | 8312 |
| MiDashengLM-Gen | midashenglm_gen / gen | midashenglm-gen-q8_0.gguf | 8341 |
| Qwen3-ASR | qwen3_asr / asr | qwen3-asr-1.7b-f16.gguf | 8321 |
| Qwen3-TTS Base | qwen3_tts / tts | qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf | 8311 |
| SeedVC | seed_vc / vc | seed-vc-mlx-f16.gguf | 8342 |
| Stable Audio 3 Small SFX | stable_audio / gen | stable-audio-3-small-sfx-f16.gguf | 8343 |
| VoxCPM2 | voxcpm2 / tts | voxcpm2-q8_0.gguf | 8313 |

上述候选共用固定 LocalAI/audio.cpp 宿主和轻量 Python HTTP runtime。所有模型推理共用 GPU_LOCK_FILE，
并在每次请求结束时确认 LocalAI backend shutdown 后释放显存租约。

## 筛选范围以外

当前没有为 MOSS-Music、TIGER-DnR、MSST-Mega53 或退役视频模型保留 LocalAI/audio.cpp 服务入口。
Steam Audio 是纯 CPU renderer。不要从已退役报告、旧候选文件或上游 family 列表推断这些模型已实现。

## 验收边界

- 此筛选报告确认固定上游对照和本机候选文件名；不代表 LocalAI 服务已经启动。
- API 合同、候选注册和 readiness 校验由代码与无模型质量门禁覆盖。
- 真实模型生成逐个顺序运行，输出记录在运行目录 storage/audio_cpp_tests/，不纳入 Git。
- 新接入模型仍需检查实际 HTTP 响应、WAV/JSON 合同、backend shutdown 与 CUDA 释放；短样例不等于听感或性能验收。

真实生成命令与输入文件要求见 [audio.cpp 冒烟生成说明](../../tests/audio_cpp/README.md)。
