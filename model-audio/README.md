# 音频模型与处理

此目录集中维护当前启用的音频模型服务，以及 CPU 空间音频 renderer。权重仍由
`$HF_MIRROR_DIR` 指向的仓库外目录提供；LocalAI 管理与分发代码保留在根目录 `localai/`。

当前模型服务包括 Qwen3-TTS Base/VoiceDesign、ACE-Step、YuE2-3B、SheetSage2、YuE2-Vae、
MERT-v2-FullSong、VoxCPM2、TIGER-DnR、Confucius4-TTS 和 Qwen3-ASR。Qwen3-ASR 按输入模态归入
音频模型。`model-audio/steam_audio_renderer/` 是 CPU 空间音频处理程序，不加载 AI 模型。
