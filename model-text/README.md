# 文本模型与处理

`Ornith-1.5-9B-MTP-GGUF/` 是唯一维护中的 Ornith 模型 API，监听8504，路由为
`POST /v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`；llama-server 仅绑定 `127.0.0.1:8506`。
`bash start.sh` 统一管理其启动、预热和关闭，请求受共享GPU锁及quarantine保护。

模型从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/Ornith-1.5-9B-MTP-Q8_0.gguf` 只读加载。
配置统一使用 `ORNITH_MTP_9B_*`，上下文默认98304（96k）；本机OpenCode输出4096、buffer16384、keep10240。
vision projector 可由 `ORNITH_MTP_9B_VISION_ENABLED` 控制，本机文本测试关闭。跨sleep RAM缓存默认4096MiB，
详见[缓存说明](../docs/llama-cpp-sleep-cache.md)。共享运行时在 `llama_cpp_runtime/`，不在HTTP适配器中加载模型。

真实调用示例见[模型示例](../tests/ornith_1_5_9b_mtp_gguf_q8/README.md)，性能基准工具在
`tests/ornith_1_5_9b_mtp_benchmarks/`。历史对比数据只用于追溯，不代表当前服务范围。

`scripts/training/` 与 `training/configs/` 保留纯文本LoRA实验；它们不是HTTP服务。
