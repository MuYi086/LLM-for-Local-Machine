# Mel-Band RoFormer API 分离测试

- 服务：`127.0.0.1:8346`
- 路由：`POST /v1/audio-cpp/Mel-Band-RoFormer-GGUF`
- 输入：multipart `audio`，44.1 kHz 单声道或立体声 PCM WAV
- 输出：包含模型包返回全部命名 WAV stems 的 ZIP，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/Mel-Band-RoFormer-GGUF/generate_example.py \
  --input-audio /path/to/song-44100.wav
```

该入口只请求 Mel-Band RoFormer API。分离结果为模型估计声部，不是原生录音分轨。
