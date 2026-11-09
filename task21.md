你先阅读`model-audio/design_qwen3-tts-1.7b-voicedesign`这种实现，以及通过start.sh调用的方式，
再当前项目分别将以下模型引入model-audio目录，然后再tests目录完成对应模型的测试生成
`/home/muyi086/hf-mirror/audio.cpp-gguf/Confucius4-TTS-GGUF` 端口8331  路由地址`/v1/netease/Confucius4-TTS-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/IndexTTS2.5-GGUF` 端口8312  路由地址`/v1/indexTeam/IndexTTS2.5-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/MiDashengLM-Gen-GGUF` 端口8341  路由地址`/v1/mi/MiDashengLM-Gen-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/Qwen3-ASR-1.7B-GGUF` 端口8321  路由地址`/v1/qwen/Qwen3-ASR-1.7B-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-Base-GGUF` 端口8311  路由地址`/v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/SeedVC-MLX-GGUF` 端口8342  路由地址`/v1/SeedVC/SeedVC-MLX-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/Stable-Audio-3-Small-SFX-GGUF` 端口8343  路由地址`/v1/stabilityai/Stable-Audio-3-Small-SFX-GGUF`
`/home/muyi086/hf-mirror/audio.cpp-gguf/VoxCPM2-GGUF` 端口8313  路由地址`/v1/OpenBMB/VoxCPM2-GGUF`

要尽可能的统一项目中uv使用的python环境和torch版本，保证尽可能少下载额外依赖