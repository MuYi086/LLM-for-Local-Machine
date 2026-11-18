# HTDemucs API 分离测试

- 服务：`127.0.0.1:8347`
- 路由：`POST /v1/audio-cpp/HTDemucs-GGUF`
- 输入：multipart `audio`，44.1 kHz 单声道或立体声 PCM WAV
- 输出：包含模型返回全部命名 WAV stems 的 ZIP；标准包通常包含 vocals、drums、bass 和 other
- 输出默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/HTDemucs-GGUF/generate_example.py \
  --input-audio /path/to/song-44100.wav
```

该入口只请求 HTDemucs API。分轨是 AI 估计结果，不能视作原生录音分轨。
