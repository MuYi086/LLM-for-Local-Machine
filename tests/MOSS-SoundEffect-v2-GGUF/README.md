# MOSS-SoundEffect v2 API 生成测试

- 服务：`127.0.0.1:8315`
- 路由：`POST /v1/openmoss/MOSS-SoundEffect-v2-GGUF-f16`
- 输入：测试脚本内置的音效提示词，不需要参考音频
- 模型文件：`moss-soundeffect-v2-bf16.gguf`（BF16）
- 输出：校验过的 WAV，默认写入 `storage/audio_cpp_tests/`

先准备模型文件和兼容的 audio.cpp backend，再通过 `bash start.sh` 启动服务，最后运行：

```bash
uv run --project qa --locked python tests/MOSS-SoundEffect-v2-GGUF/generate_example.py
```

脚本只调用此模型的真实 HTTP API，不会自行加载模型。无模型质量门禁不会生成音频。
