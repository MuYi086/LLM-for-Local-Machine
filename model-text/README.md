# 文本模型与处理

`ornith_1_5_35b_a3b/` 与 `qwen3_6_35b_a3b/` 提供 Ornith 和 Qwen3.6 的 llama.cpp HTTP 适配器。
Ornith 推理使用 CUDA，前 23 层 MoE 专家权重放在 CPU，通过 F16 mmproj 接收图片输入，并加载独立 Q8_0 MTP 文件启用 `draft-mtp`；Qwen 使用 CUDA0、
前 23 层 CPU MoE 和 `draft-mtp`。每个请求复用共享 GPU 锁，请求后确认对应模型卸载才释放。
`scripts/training/` 与 `training/configs/` 集中放置有声书文本润色的 LoRA 实验工具和配置；该实验处理文本，
不是语音训练或生产服务。

两套 API 由仓库根目录的 `bash start.sh` 统一管理。Ornith 对外监听 8401，backend 在 `127.0.0.1:8403`；
Qwen 对外监听 8402，backend 在 `127.0.0.2:8400`。LocalAI 使用 `127.0.0.1:8400`，可以同时运行：

```bash
curl -X POST http://127.0.0.1:8401/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":128}'
```

默认从 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/` 读取主 GGUF、mmproj 与 `mtp-Ornith-1.5-35B-A3B-Q8_0.gguf`。运行 `start.sh` 前需在 PATH
中安装支持 CUDA0、`--n-cpu-moe` 和 `--sleep-idle-seconds` 的 llama.cpp CUDA 构建 `llama-server`，
或通过 `LLAMA_SERVER_BIN` 指定可执行文件；文件路径可分别通过
`ORNITH_MODEL_FILE`、`ORNITH_MMPROJ_FILE` 与 `ORNITH_MTP_FILE` 覆盖。Ornith 使用
`--spec-type draft-mtp --spec-draft-model "$ORNITH_MTP_FILE"` 启用 MTP，启动前检查该文件非空。`ORNITH_SPEC_TYPE=none` 可关闭 Ornith MTP。
`bash start.sh` 默认采用实测配置：两模型 CPU MoE 均为 23 层，Ornith 线程 6/8、Qwen 线程 8/8，
主 KV 为 Q8、草稿 KV 为 F16、草稿长度 4、load-mode none。复测步骤与结论见
[`tests/model_web_comparison/README.md`](../tests/model_web_comparison/README.md)。真实模型生成样例见
[`tests/ornith_1_5_35b_a3b/README.md`](../tests/ornith_1_5_35b_a3b/README.md)。

Qwen GGUF 默认从 `$HF_MIRROR_DIR/unsloth/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` 加载，
通过 `QWEN36_MODEL_FILE` 覆盖。llama.cpp 需支持 `--spec-type draft-mtp`。Qwen API 路由为
`POST /v1/qwen/Qwen3.6-35B-A3B-MTP`；真实样例生成方式见
[`tests/qwen3_6_35b_a3b/README.md`](../tests/qwen3_6_35b_a3b/README.md)。

```bash
curl -X POST http://127.0.0.1:8402/v1/qwen/Qwen3.6-35B-A3B-MTP \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用中文简要解释 MTP。"}],"max_tokens":128}'
uv run --project qa --locked python tests/qwen3_6_35b_a3b/generate_example.py
```

运行实验脚本前先查看必填的来源目录和输出目录参数：

```bash
PYTHONPATH=model-text uv run --project qa --locked python -m scripts.training.build_audio_polisher_poc --help
```
