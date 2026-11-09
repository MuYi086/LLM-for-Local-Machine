# 活跃音频后端收敛设计

## 状态

本文件记录 task21 之前的退役范围，作为历史设计保留。后续 task21.md 已重新接入八个 audio.cpp 模型；
当前运行面以 AGENTS.md、README.md 和 localai/manifests/audio-cpp-models.json 为准。

## 目标

把当前仓库运行面收敛为 Qwen3-TTS 1.7B VoiceDesign（LocalAI/audio.cpp）、8300 控制面和
Steam Audio CPU 空间渲染器。移除原统一启动脚本中的其他废弃音频模型，以及它们专属的
provider、worker bridge、候选配置、端口、接口、依赖和测试夹具。

全局 GPU 文件锁是明确例外：保留现有共享实现、隔离/恢复语义和必要配置，并为后续多模型接入
建立独立文档与回归覆盖。不得清理 `storage/` 中可能属于用户的运行数据。

## 活跃架构

- 8300 控制面：健康与控制信息、CPU 音频导出、Steam Audio 空间渲染及进度查询。
- 8301 VoiceDesign：固定 LocalAI/audio.cpp GGUF 候选及现有 JSON 合同。
- Steam Audio renderer：独立 C++17 CPU 程序；不加载模型，也不获取 GPU 锁。
- `llm_local_runtime`：保留流式上传、存储/容量工具和跨进程 GPU 锁；删除仅供语音克隆使用的
  参考音频映射功能。
- `inference_gateway` 与 LocalAI 管理脚本：仅保留 VoiceDesign provider、LocalAI 宿主机生命
  周期、候选校验及 GPU 隔离恢复；删除旧模型和 worker bridge 的执行路径。

`start.sh` 只启动 8300 与 8301 两个 HTTP 服务，并按现有规则管理 LocalAI 子进程组。控制面
应使用自身轻量 uv 项目，不再借用已废弃的 Qwen3-TTS Base 环境。端口预检、退出时进程组清理、
LocalAI 停止确认与隔离标记行为继续有效。

## API 与配置

保留 `GET /v1/health`、`GET /v1/control`、`POST /v1/audio/export`、
`POST /v1/audio/spatial/render`、`GET /v1/audio/spatial/render/progress/{job_id}`，以及
VoiceDesign 的健康和合成路由。

移除只为克隆/参考音频服务的 `POST /v1/upload_audio`、`GET /v1/check/audio`，克隆存储环境变量、
Base/TIGER provider 配置，以及废弃服务的端口、路径和启动输出。空间渲染继续使用 Manifest v1
和重复 `assets` multipart 字段。

## 全局 GPU 锁存档

保留 `llm_local_runtime.gpu_runtime_lock`、`gpu_quarantine_path`、超时/隔离异常、显存指标采样、
VoiceDesign 的锁接入和 LocalAI quarantine 恢复脚本。新增 `docs/global-gpu-lock.md`，记录共享锁
文件合同、正数排队时限、`finally` 释放、失败隔离、人工恢复步骤、指标含义及未来 provider 的
接入要求。空间渲染仍明确标为 CPU 路径。

锁合同以实现为准，不在本次清理中改写锁算法或放宽故障恢复条件。

## 移除范围

- `start.sh` 中 MOSS-Music、MOSS-SoundEffect、Qwen3-TTS Base、IndexTTS、TIGER-DnR、
  Confucius4、Qwen3-ASR、MSST 的项目、模型变量、端口、PID 和启动命令。
- TIGER one-shot worker bridge：`localai_worker_bridge/`、其 manifest/candidate、LocalAI host
  管理/回收分支、gateway 文件协议封装、smoke runner、QA fixture 和专属依赖。
- 其他废弃模型的候选 YAML、能力清单条目、质量门禁项目、克隆上传/检查接口、专属存储路径，
  以及只服务于旧模型的构建脚本。
- 旧服务的报告、设计文档、任务记录、操作说明和退役源代码保留作追溯，不属于当前运行说明。
  当前 LocalAI/audio.cpp 部署说明改为 VoiceDesign 专用文档；活动版本和能力清单只列出现役服务。

保留与本次服务收敛无关的文本训练、视频归档和调研资料；保留 VoiceDesign 任务/实现记录、
Steam Audio 设计资料和空间音频说明。旧模型对应的运行数据不由代码改动删除。

## 文档和质量门禁

同步 README、`AGENTS.md`、`model-audio/README.md`、Python 工程说明、LocalAI 能力清单和版本
清单：准确描述一个模型 API、一个控制面 API 与一个 CPU renderer；uv 锁检查只覆盖仍维护的
项目与 QA。保留 Steam Audio 的无 SDK C++ 核心检查和现有空间音频约束。

## 回归覆盖

- 检查 `start.sh` 只预检并启动 8300/8301，且没有废弃项目路径或变量。
- 检查 LocalAI manager/preflight 只注册和验证 VoiceDesign，不引用 bridge/Base/TIGER。
- 检查控制面保留的路由、空间上传合同、VoiceDesign 输出行为。
- 覆盖 GPU 锁互斥、等待超时、隔离标记阻止推理、释放锁与恢复合同；不启动 CUDA 或真实模型。
- 运行仓库约定的 shell 语法、uv 锁检查、Ruff、无模型 QA 和 Steam Audio 核心门禁。

## 验收条件

仓库启动配置、能力清单、当前说明文档和门禁只列出上述活跃运行面；废弃音频模型的运行集成已移除，
历史报告、设计资料和退役源代码作为归档保留；GPU 锁实现和恢复机制仍被 VoiceDesign 使用且可由文档独立查阅；空间
音频路由与 CPU renderer 合同保持有效。
