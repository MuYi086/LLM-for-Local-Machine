# 文本模型与处理

`Ornith-1.5-9B-GGUF/` 与 `Ornith-1.5-9B-MTP-GGUF/` 提供 llama.cpp 聊天 API。服务从模型镜像目录只读加载 GGUF，通过各自的 loopback llama-server 推理；请求使用共享 GPU 锁，并在确认对应模型进入 sleep-idle 后释放。

`llama_cpp_runtime/` 是模型无关的共享运行时，提供 OpenAI Chat Completions 适配器、backend 启动与关闭、sleep-idle readiness 和 quarantine 管理。两个服务以固定路由和模型别名复用该运行时。

默认启用 [跨 sleep RAM 缓存](../docs/llama-cpp-sleep-cache.md)，在 4096 MiB 预算内保留主模型、MTP 草稿及循环检查点；每轮仍释放 CUDA，下一轮由原生 token 前缀匹配恢复。补丁能力会在启动时核验，设 `ORNITH_9B_SLEEP_CACHE=0` 可回退；旧运行进程须重启才会采用新构建。

`scripts/training/` 与 `training/configs/` 集中放置有声书文本润色的 LoRA 实验工具和配置；该实验处理文本，不是语音训练或生产服务。

仓库根目录的 `bash start.sh` 统一管理 API：原有 Ornith API 对外监听 8503、backend 只绑定 `127.0.0.1:8505`；MTP API 对外监听 8504、backend 只绑定 `127.0.0.1:8506`。两个 backend 顺序预热并共用 GPU 锁。LocalAI 使用 `127.0.0.1:8400`。

```bash
curl -X POST http://127.0.0.1:8503/v1/ornith-ai/Ornith-1.5-9B-GGUF \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":128}'
```

模型默认从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q8_0.gguf` 只读加载，可通过 `ORNITH_9B_MODEL_FILE` 覆盖。真实调用 demo 见 [`tests/ornith_1_5_9b_gguf/README.md`](../tests/ornith_1_5_9b_gguf/README.md)。

MTP Q8 默认从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/Ornith-1.5-9B-MTP-Q8_0.gguf` 只读加载，vision projector 默认是同目录的 `mmproj-Ornith-1.5-9B-BF16.gguf`。真实调用 demo 见 [`tests/ornith_1_5_9b_mtp_gguf_q8/README.md`](../tests/ornith_1_5_9b_mtp_gguf_q8/README.md)。

`tests/model_web_comparison/` 中的旧模型对比数据是历史实验归档；当前复测脚本只使用 Ornith 9B。
