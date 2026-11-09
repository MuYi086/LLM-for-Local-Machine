# MiDashengLM-Gen API 生成测试

- 服务：`127.0.0.1:8341`
- 路由：`POST /v1/mi/MiDashengLM-Gen-GGUF`
- 输入：脚本内置的结构化文本提示词，不需要参考音频
- 输出：校验过的 WAV，默认写入 `storage/audio_cpp_tests/`

先通过 `bash start.sh` 启动服务，再运行：

```bash
uv run --project qa --locked python tests/gen_midashenglm/generate_example.py
```

该入口只请求此模型的真实 HTTP API，不运行模型时的质量门禁不会生成音频。
