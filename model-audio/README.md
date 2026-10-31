# 音频模型与处理

此目录集中维护当前启用的音频模型服务，以及 CPU 空间音频 renderer。权重仍由
`$HF_MIRROR_DIR` 指向的仓库外目录提供；LocalAI 管理与分发代码保留在根目录 `localai/`。

当前 9 个模型服务包括 MOSS-Music-8B-Instruct、MOSS-SoundEffect v2.0、Qwen3-TTS Base/VoiceDesign、IndexTTS-2.5、
MSST Mega 53、TIGER-DnR、Confucius4-TTS 和 Qwen3-ASR。Qwen3-ASR 按输入模态归入
音频模型。`model-audio/spatial_steam-audio/` 是 CPU 空间音频处理程序，不加载 AI 模型。

## 目录命名与功能速查

目录统一使用 `功能词_模型名-版本`，功能词是一个英文单词，模型部分使用连字符，优先使用小写；MOSS 目录按既有任务约定保留 `Instruct` 大小写。
保留可辨认的模型名、版本、规模及变体；模型包名、API 路径、环境变量和仓库外权重目录独立命名。

| 目录 | 功能词含义 | 主要用途 |
| --- | --- | --- |
| `analyse_moss-music-8b-Instruct/` | analyse：音乐理解 | 上传音乐并生成分析或问答文本 |
| `soundEffect-v2.0/` | generate：音效生成 | 根据文字提示词生成最长 30 秒环境音效 |
| `clone_confucius-4/` | clone：语音克隆 | 跨语言零样本 TTS |
| `clone_qwen3-tts-1.7b-base/` | clone：语音克隆 | 参考音频驱动的多语言 TTS |
| `indexTTS-2.5/` | clone：语音克隆 | 参考音频驱动的五语言 TTS 与情感控制 |
| `design_qwen3-tts-1.7b-voicedesign/` | design：音色设计 | 用文字描述音色并生成语音 |
| `separate_msst-mega53/` | separate：音源分离 | 按请求选择音乐声部并保留 residual |
| `separate_tiger-dnr/` | separate：音源分离 | 分离对白、音效和音乐 |
| `spatial_steam-audio/` | spatial：空间渲染 | CPU 空间音频处理程序 |
| `transcribe_qwen3-asr-1.7b/` | transcribe：语音转写 | 语种识别及语音转文字 |
