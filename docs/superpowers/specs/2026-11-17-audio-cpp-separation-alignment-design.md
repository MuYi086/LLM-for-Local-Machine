# 四个 audio.cpp 音频模型接入设计

## 意图与范围

把 `task10.md` 列出的 BS-RoFormer、Qwen3 ForcedAligner、Mel-Band RoFormer 和 HTDemucs 接入当前 LocalAI/audio.cpp 服务。每个模型使用独立的 `model-audio/<模型目录>` 入口，注册为共享 audio.cpp runtime 服务，并由 `start.sh` 在 8344–8347 启动。每个服务提供独立的真实 HTTP API demo。

## 目标与约束

- 路由使用 `/v1/audio-cpp/<模型目录名>`，端口按模型清单固定映射 8344、8345、8346、8347。
- 模型权重只读使用 `~/hf-mirror/audio.cpp-gguf` 中已有 GGUF，不复制或提交权重。
- 继续使用统一 LocalAI provider、共享 GPU 锁、隔离标记和 `audio_cpp_runtime` uv 环境，不增加 Torch。
- RoFormer 和 HTDemucs 通过 LocalAI `/audio/transform` 返回的 stem 产物提供完整 ZIP；ForcedAligner 通过 transcription API 的逐字稿 prompt 返回对齐 JSON。
- demo 写入 `storage/audio_cpp_tests/`，先校验输入和响应，不改写源 WAV；无模型质量门禁不访问真实模型。

## 考虑过的做法

- **扩展共享 router 和 LocalAI client（采用）**：沿用已有十个服务的启动、GPU 锁、上传暂存、错误映射与清单校验，按 `source_separation` 和 `forced_alignment` 两种专属合同补齐能力。
- 为四个模型新增独立推理 runtime：会绕开仓库要求的 LocalAI provider 与共享 GPU 锁，并重复依赖，因此不采用。
- 通过多次 LocalAI transform 请求逐个提取 stem：会重复运行昂贵分离。使用一次 transform 的 `X-Audio-Stems` 清单并逐个读取产物，减少重复推理。

## 实现责任

- 候选清单保存模型路径、family/task、model ID、端口、路由和 API 类型；`start.sh` 与 capabilities manifest 同步覆盖。
- 共享 LocalAI client 读取并校验 stem 响应头，再通过受限的 LocalAI 生成文件路径读取各 stem。
- 共享服务校验输入、在全局 GPU 锁内分离/对齐、清理上传与临时产物，并返回 ZIP 或 JSON。
- 每个测试目录提供 README 和只针对该模型的真实 API 运行脚本。

## 验收

- 四个 GGUF 在默认镜像目录下通过启动预检；四个端口与路由进入启动检查及能力清单。
- 合同单测覆盖候选、LocalAI 请求、zip 中全部 stem、forced alignment prompt 和结果校验。
- 四个 demo 脚本可被质量门禁发现；真实模型调用只由操作者单独运行。
