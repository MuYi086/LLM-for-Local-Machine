# 文本模型与处理

`ornith_1_5_35b_a3b/` 提供唯一的文本模型 llama.cpp HTTP 适配器。Ornith 使用 CUDA，前 23 层 MoE 专家权重放在 CPU，通过 F16 mmproj 接收图片输入，并加载独立 Q8_0 MTP 文件启用 `draft-mtp`。每个请求复用共享 GPU 锁，请求后确认模型卸载才释放。

`scripts/training/` 与 `training/configs/` 集中放置有声书文本润色的 LoRA 实验工具和配置；该实验处理文本，不是语音训练或生产服务。

仓库根目录的 `bash start.sh` 统一管理 API：Ornith 对外监听 8401，backend 只绑定 `127.0.0.1:8403`。LocalAI 使用 `127.0.0.1:8400`。

```bash
curl -X POST http://127.0.0.1:8401/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":128}'
```

默认从 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/` 读取主 GGUF、mmproj 与 `mtp-Ornith-1.5-35B-A3B-Q8_0.gguf`。文件路径可分别通过 `ORNITH_MODEL_FILE`、`ORNITH_MMPROJ_FILE` 与 `ORNITH_MTP_FILE` 覆盖。运行参数与样例见 [`tests/ornith_1_5_35b_a3b/README.md`](../tests/ornith_1_5_35b_a3b/README.md)。

`tests/model_web_comparison/` 中保留的 Qwen 对比数据是历史实验归档；当前脚本只运行 Ornith。
