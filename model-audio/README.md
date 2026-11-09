# 音频模型与处理

当前音频运行面维护九个 GPU 模型 API 和一个 CPU 空间音频 renderer。八个新接入的 audio.cpp 服务共享
`audio_cpp_runtime` uv 项目，不安装 Torch；VoiceDesign 保持独立环境。所有 GPU 服务共用同一
`GPU_LOCK_FILE`，按跨进程队列串行推理。

| 目录 | 模型职责 | 默认端口 | 业务接口 |
| --- | --- | ---: | --- |
| `design_qwen3-tts-1.7b-voicedesign/` | Qwen3-TTS 文本音色设计 | 8301 | `POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign` |
| `tts_confucius4-tts/` | Confucius4 多语参考音频克隆 | 8331 | `POST /v1/netease/Confucius4-TTS-GGUF` |
| `tts_indextts2.5/` | IndexTTS2.5 参考音频克隆 | 8312 | `POST /v1/indexTeam/IndexTTS2.5-GGUF` |
| `gen_midashenglm/` | MiDashengLM 文本提示音频生成 | 8341 | `POST /v1/mi/MiDashengLM-Gen-GGUF` |
| `asr_qwen3-1.7b/` | Qwen3-ASR 语音转写 | 8321 | `POST /v1/qwen/Qwen3-ASR-1.7B-GGUF` |
| `tts_qwen3-1.7b-base/` | Qwen3-TTS Base 参考音频克隆 | 8311 | `POST /v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF` |
| `vc_seed-vc/` | SeedVC 参考音色转换 | 8342 | `POST /v1/SeedVC/SeedVC-MLX-GGUF` |
| `gen_stable-audio-3-small-sfx/` | Stable Audio 3 Small 音效生成 | 8343 | `POST /v1/stabilityai/Stable-Audio-3-Small-SFX-GGUF` |
| `tts_voxcpm2/` | VoxCPM2 TTS、音色设计与可选参考克隆 | 8313 | `POST /v1/OpenBMB/VoxCPM2-GGUF` |
| `spatial_steam-audio/` | Steam Audio C++17 双耳空间 renderer | 无 HTTP 端口 | 由 8300 控制面调用，纯 CPU |

八个共享运行时服务各自有薄型 `main.py` 入口；模型 ID、GGUF 文件名、LocalAI family/task、路由和端口
集中在 `localai/manifests/audio-cpp-models.json`。健康检查统一为 `GET /v1/health`，不会触发推理。

各服务请求格式、端口、部署步骤和全局 GPU 锁说明见根目录 [README](../README.md)。
真实模型串行生成示例见 [audio.cpp 冒烟生成说明](../tests/audio_cpp/README.md)。
