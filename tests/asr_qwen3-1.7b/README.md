# Qwen3-ASR 1.7B API 转写测试

- 服务：`127.0.0.1:8321`
- 路由：`POST /v1/qwen/Qwen3-ASR-1.7B-GGUF`
- 输入：本地 PCM WAV（只读）
- 输出：含非空 `text` 字段的 JSON，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/asr_qwen3-1.7b/generate_example.py \
  --reference-audio /path/to/speech.wav
```

该入口只请求此模型的真实 HTTP API，不运行模型时的质量门禁不会生成音频。
