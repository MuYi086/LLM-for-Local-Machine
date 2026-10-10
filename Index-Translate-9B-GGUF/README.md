# Index-Translate 9B 语音翻译服务

本目录提供独立的 FastAPI 项目和一次性 llama.cpp worker。服务接收共享 `AudioReferenceStore`
逻辑标识 `audio_path`，依次调用 Qwen3-ASR 转写、Index-Translate GGUF 翻译，再将原始参考音频
和译文交给 Confucius4-TTS 合成 WAV，以保留参考音频中的说话人音色。

业务路由为 `POST /v1/indexTranslate/translate`，统一启动端口为 `8385`。目标语言限定为
Confucius4-TTS 当前支持的 `zh`、`en`、`ja`、`ko`、`de`、`fr`、`th`、`id`、`vi`、`es`、
`pt`、`it`、`ru`、`ms`。可选的 `glossary`、`hard_constraints` 和 `soft_constraints` 使用
Index-Translate 的 instTrans 格式；常规翻译使用模型卡推荐提示词。

默认 GGUF 为 `$HF_MIRROR_DIR/IndexTeam/Index-Translate-9B-GGUF/Index-Translate-9B.Q8_0.gguf`。
服务通过 `llama-server` 启动一次性模型 worker；默认上下文为 32768、最大输出为 1024 token、
温度为 0 并关闭 thinking。部署时先安装本机兼容的 llama.cpp，再执行：

```bash
uv sync --project Index-Translate-9B-GGUF --locked
```

可通过 `INDEX_TRANSLATE_MODEL_DIR`、`INDEX_TRANSLATE_MODEL_PATH`、
`INDEX_TRANSLATE_LLAMA_SERVER_BIN`、`INDEX_TRANSLATE_CONTEXT_SIZE`、
`INDEX_TRANSLATE_MAX_NEW_TOKENS` 和 `INDEX_TRANSLATE_TEMPERATURE` 覆盖默认设置。
