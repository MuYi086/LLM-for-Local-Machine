# Qwen3-TTS Base API 生成测试

- 服务：`127.0.0.1:8311`
- 路由：`POST /v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF`
- 输入：本地 PCM WAV 参考音频、与音频逐字一致的 `--reference-text` 和固定中文文本
- 输出：校验过的 WAV，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务。参考文本必须准确匹配音频：

```bash
uv run --project qa --locked python tests/tts_qwen3-1.7b-base/generate_example.py \
  --reference-audio /path/to/reference.wav \
  --reference-text '参考音频中的逐字稿'
```

该入口只请求此模型的真实 HTTP API，不运行模型时的质量门禁不会生成音频。
