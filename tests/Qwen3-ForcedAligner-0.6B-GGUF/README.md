# Qwen3 ForcedAligner API 对齐测试

- 服务：`127.0.0.1:8345`
- 路由：`POST /v1/audio-cpp/Qwen3-ForcedAligner-0.6B-GGUF`
- 输入：multipart WAV `audio`、必填逐字稿 `text` 与语言 `language`
- 输出：含词级 `words[].start`、`words[].end` 和 `words[].text` 的 JSON，时间单位为秒
- 该模型只对齐已知稿件，不识别未知语音，也不会纠正稿件

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/Qwen3-ForcedAligner-0.6B-GGUF/generate_example.py \
  --input-audio /path/to/speech.wav \
  --transcript "需要与录音逐字匹配的稿件。" \
  --language Chinese
```

对齐结果默认写入 `storage/audio_cpp_tests/`。输入 WAV 只读。
