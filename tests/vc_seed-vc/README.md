# SeedVC API 音色转换测试

- 服务：`127.0.0.1:8342`
- 路由：`POST /v1/SeedVC/SeedVC-MLX-GGUF`
- 输入：源语音 WAV 和目标音色参考 WAV（均只读）
- 输出：校验过的转换 WAV，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/vc_seed-vc/generate_example.py \
  --source-audio /path/to/source.wav \
  --reference-audio /path/to/target-voice.wav
```

该入口只请求此模型的真实 HTTP API，不运行模型时的质量门禁不会生成音频。
