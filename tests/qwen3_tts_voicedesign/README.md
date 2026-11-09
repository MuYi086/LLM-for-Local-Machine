# Qwen3-TTS VoiceDesign 本机示例

该目录的 unittest 以 LocalAI mock 验证 HTTP 合同，不加载模型。`generate_example.py` 是单独的
真实推理冒烟命令：请求本机 `8301`，检查 RIFF/WAVE 与 PCM 采样帧，并确认服务已将同一文件保存在
`storage/timbre/`。它不复制或改写源模型，也不会在仓库中保存测试音频。

在本机完整模型环境中启动服务：

```bash
bash start.sh
```

启动脚本默认使用 `QWEN_VOICEDESIGN_MODEL_FILE` 指定的 BF16 GGUF，注册 `qwen3_tts/task:vdes`
候选，启动受管 LocalAI/audio.cpp，完成 readiness 和模型预检后启动 VoiceDesign HTTP 服务。
模型路径可在启动前覆盖；输入模型文件保持只读。

另开终端执行短句示例：

```bash
uv run --project qa --locked python tests/qwen3_tts_voicedesign/generate_example.py
```

默认请求的音色描述为“成年女性，声音清晰自然，温和亲切，语速中等”，朗读“你好，欢迎使用
Qwen3-TTS 音色设计。”。生成音频由服务写入 `storage/timbre/`，并登记到共享音色引用索引。
可通过 `--voice-description`、`--text`、`--base-url` 与 `--timeout` 调整冒烟请求。
