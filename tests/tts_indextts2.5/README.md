# IndexTTS 2.5 API 生成测试

- 服务：`127.0.0.1:8312`
- 路由：`POST /v1/indexTeam/IndexTTS2.5-GGUF`
- 输入：本地 PCM WAV 参考音频（只读）和固定中文文本
- 输出：校验过的 WAV，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/tts_indextts2.5/generate_example.py \
  --reference-audio /path/to/reference.wav
```

该入口只请求此模型的真实 HTTP API，不运行模型时的质量门禁不会生成音频。
