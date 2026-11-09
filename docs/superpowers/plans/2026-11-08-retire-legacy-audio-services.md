# Retire Legacy Audio Services Implementation Plan

> 历史计划：按当时范围退役旧模型；后续 `task21.md` 重新接入八个 audio.cpp 模型。当前运行状态以
> `AGENTS.md`、`README.md` 与 `localai/manifests/audio-cpp-models.json` 为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛启动、接口、LocalAI、QA 和维护文档到 VoiceDesign、控制面及 Steam Audio CPU renderer，并保留可供未来所有 GPU 模型共用的全局串行锁。

**Architecture:** 控制面与 VoiceDesign 使用独立 uv 项目；LocalAI 管理只服务于 VoiceDesign。共享运行时继续提供跨进程 `flock`、有限排队、quarantine 与指标，空间音频保持纯 CPU。删除旧克隆引用映射、one-shot worker bridge 和其余退役音频模型集成。

**Tech Stack:** Python 3.12.13、FastAPI、uv、bash、LocalAI/audio.cpp、Steam Audio C++17、unittest、Ruff。

**Spec:** `docs/superpowers/specs/2026-11-08-active-audio-scope-design.md`

## Global Constraints

- 运行面只保留 8300 控制面、8301 VoiceDesign 和 Steam Audio CPU renderer。
- 所有 GPU 模型通过同一个 `GPU_LOCK_FILE` 跨进程串行运行；默认等待 900 秒，部署保持正数超时。
- 锁必须在 `finally` 释放；backend 停止无法确认时保留 `.quarantine`，人工恢复前不得继续 GPU 推理。
- 空间渲染不获取 GPU 锁；正式空间导出继续使用 Manifest v1 与重复 `assets` 字段。
- 输入源和 `storage/` 运行数据不因本次清理而被删除。
- QA 不下载权重、不依赖 CUDA、不调用真实模型服务。

## Review Focus

- 控制面启动仍依赖已退役的 Base 环境：由隔离的启动脚本合同测试覆盖。
- 清理旧模型配置误删 VoiceDesign LocalAI 候选或恢复命令：由 manager CLI、candidate 和 preflight 合同覆盖。
- 单一 GPU 模型暂时掩盖锁不再全局串行：由两个独立锁使用者的互斥测试覆盖。
- 移除克隆引用映射意外改变 VoiceDesign 返回或 WAV 持久化：由真实 HTTP 测试覆盖。
- 清理历史接口误删空间 multipart 上传或进度路由：由控制面 HTTP 路由测试覆盖。

## File Map

- `main/main.py` 与 `main/pyproject.toml`：控制面路由和独立轻量运行环境。
- `model-audio/design_qwen3-tts-1.7b-voicedesign/`：VoiceDesign 输出、依赖锁和 API 合同。
- `llm_local_runtime/src/llm_local_runtime/`：共享上传/存储与 GPU 文件锁；移除克隆引用映射。
- `scripts/localai/`、`inference_gateway/src/inference_gateway/`、`localai/manifests/`：固定 VoiceDesign 候选、宿主进程生命周期和安全恢复。
- `start.sh`、`scripts/quality_gate.sh`、`qa/`：只启动和检查当前维护项目。
- `README.md`、`AGENTS.md`、`model-audio/README.md`、`docs/python-engineering.md` 及 LocalAI 操作文档：同步当前能力。
- `tests/`：控制面、启动合同和共享 GPU 锁无模型回归。

## Tasks

### Task 1: 收敛控制面路由并移除克隆引用映射

**Files:**
- Modify: `main/main.py`, `main/storage_maintenance.py`
- Modify: `model-audio/design_qwen3-tts-1.7b-voicedesign/main.py`
- Modify: `llm_local_runtime/src/llm_local_runtime/storage.py`, `llm_local_runtime/src/llm_local_runtime/__init__.py`
- Create: `main/pyproject.toml`, `main/uv.lock`, `main/.python-version`
- Create: `tests/test_control_plane_contract.py`
- Modify: `tests/qwen3_tts_voicedesign/test_service.py`

**Interfaces:** 保留 `/v1/health`、`/v1/control`、`/v1/audio/export`、空间 render/progress 和 VoiceDesign 现有路由；移除克隆上传/检查端点及 `AudioReferenceStore` sidecar 行为。VoiceDesign 仍把有效 WAV 原子保存到 `TIMBRE_STORAGE_DIR` 并返回文件。

- [x] **Step 1: 写控制面与 VoiceDesign 行为测试。** `POST /v1/upload_audio` 和 `GET /v1/check/audio` 返回 404；空间路由仍注册；VoiceDesign 成功后只保存 WAV、不创建 `.references`。
- [x] **Step 2: 运行对应 unittest 并确认旧路由/sidecar 断言失败。** 分别运行 `uv run --project qa --locked python -m unittest discover -s tests -p 'test_control_plane_contract.py' -v` 与 `uv run --project qa --locked python -m unittest discover -s tests/qwen3_tts_voicedesign -p 'test_service.py' -v`。
- [x] **Step 3: 删除克隆专用端点、配置和引用 sidecar，保留空间上传与 VoiceDesign 原子输出。** 增加独立 main uv 项目并为 multipart 处理声明 `python-multipart`。
- [x] **Step 4: 生成 main 的 uv lock 并重跑上述两个 unittest 命令。** 两组控制面与 VoiceDesign 行为通过。

### Task 2: 启动面只保留控制面和 VoiceDesign

**Files:**
- Modify: `start.sh`, `tests/qwen3_tts_voicedesign/test_start_config.py`
- Create: `tests/test_start_services.py`
- Modify: `main/check_ports.py` only if its existing argument contract needs adaptation.

**Interfaces:** 默认端口为 8300/8301。控制面用 `main/` uv 项目，VoiceDesign 用自己的 uv 项目；LocalAI 子进程仍由 manager 管理。保留整组端口预检、LocalAI readiness、进程组退出清理及共享 `GPU_LOCK_FILE` 环境。

- [x] **Step 1: 写启动脚本行为测试。** 用临时 PATH 中的 `uv`/`python3` stub 启动脚本，检查实际调用只有 main、VoiceDesign 和 LocalAI 管理命令，端口预检只收到 8300/8301；旧模型项目不会被访问。
- [x] **Step 2: 运行该测试并确认当前脚本因旧项目和额外端口而失败。** 运行 `uv run --project qa --locked python -m unittest discover -s tests -p 'test_start_services.py' -v`。
- [x] **Step 3: 精简 `start.sh` 的变量、端口、日志、PID、启动调用和退出监控列表；控制面切换到 `main/`。** 保留音色输出、上传限制、LocalAI 生命周期、GPU 锁和空间渲染配置。
- [x] **Step 4: 运行启动行为测试和 VoiceDesign 启动合同测试。** 两组启动合同通过，启动脚本仅编排两个 HTTP 进程及受管 LocalAI。

### Task 3: 固化共享 GPU 串行锁合同

**Files:**
- Keep: `llm_local_runtime/src/llm_local_runtime/gpu.py` and public exports.
- Create: `docs/global-gpu-lock.md`, `tests/test_gpu_runtime.py`.
- Modify: `AGENTS.md`, `README.md`, `docs/python-engineering.md`.

**Interfaces:** 保持 `gpu_runtime_lock(lock_path, label, timeout_seconds=None)`、`gpu_quarantine_path`、`GpuLockTimeoutError`、`GpuQuarantinedError` 与 `GpuRuntimeMetrics` API 不变。每个未来 GPU provider 使用同一路径取得锁；Steam Audio 不取得该锁。

- [x] **Step 1: 写共享锁行为测试。** 覆盖两个不同模型标签竞争同一锁、有限等待超时、隔离标记拒绝新请求、异常退出后锁可再次取得。
- [x] **Step 2: 运行 `uv run --project qa --locked python -m unittest discover -s tests -p 'test_gpu_runtime.py' -v`。** 现有实现通过串行合同；未更改锁算法。
- [x] **Step 3: 编写独立存档文档。** 说明统一锁路径、跨进程 `flock`、正数超时、`finally` 释放、quarantine 恢复步骤、显存指标范围和新 provider 的强制接入要求。
- [x] **Step 4: 在 README、AGENTS 和工程说明中链接该文档。** 锁算法未更改，CPU 空间路径不进入 GPU 队列。

### Task 4: LocalAI 与 gateway 移除退役模型和 worker bridge

**Files:**
- Modify: `scripts/localai/manager.py`, `scripts/localai/preflight.py`, `scripts/localai/host.py`, `scripts/localai/recover_quarantine.py`
- Modify: `inference_gateway/src/inference_gateway/__init__.py`, `execution.py`, `localai_client.py`
- Delete: `inference_gateway/src/inference_gateway/worker_protocol.py`, `localai_worker_bridge/`, `scripts/localai/smoke_worker_bridge.py`
- Delete: `localai/manifests/qwen3-tts-base-candidate.yaml`, `tiger-dnr-worker-candidate.yaml`, `worker-bridge.json`
- Modify: `localai/manifests/capabilities.json`, `versions.json`
- Create: `tests/test_localai_active_scope.py`

**Interfaces:** manager 只接受 VoiceDesign 候选注册/校验及 LocalAI 安装、启停、状态、watch；恢复 quarantine 使用 `LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER` 和固定 VoiceDesign model ID。保留 VoiceDesign 请求的共享 GPU 锁和 LocalAI shutdown 确认。

- [x] **Step 1: 写外部可见合同测试。** manager `--help` 只列 VoiceDesign 候选命令；preflight 只校验 VoiceDesign；活跃能力清单只含控制面与 VoiceDesign；恢复脚本默认使用 VoiceDesign provider。
- [x] **Step 2: 运行 LocalAI 活跃范围合同测试。** 先在旧管理命令和 bridge 文件存在时确认失败，再在清理后通过。
- [x] **Step 3: 移除 Base/TIGER 与 bridge 管理路径。** 删除 worker bridge 运行路径和源码；保留模型 shutdown、锁及 quarantine。
- [x] **Step 4: 更新 QA 依赖和锁，删除 gRPC/protobuf/bridge fixtures，重建活跃 manifest。** LocalAI scope、候选和 VoiceDesign 测试通过。

### Task 5: 清理质量门禁和废弃音频资料

**Files:**
- Modify: `scripts/quality_gate.sh`, `qa/pyproject.toml`, `qa/uv.lock`, `model-audio/README.md`, `README.md`, `AGENTS.md`, `docs/python-engineering.md`
- Rewrite: `docs/localai-audiocpp-upgrade.md` as current VoiceDesign LocalAI operations guide.
- Delete: `scripts/build_moss_music.sh` and obsolete runtime-only build integrations.
- Keep: active VoiceDesign and spatial documents; unrelated video/text archives and user storage data.
- Keep: `reports/localai/`, retired audio-service plans/specs, task records, historical operation notes, and retired model source for traceability. Label old operational material as archived where it could be mistaken for current guidance.

**Interfaces:** quality gate checks `main/`、VoiceDesign、QA 的 uv locks，保留 Steam Audio 核心测试、Ruff、能力清单校验和无模型 unittest。文档说明两条 HTTP 端口、LocalAI 配置、空间导出和 GPU lock 存档。历史报告、设计和退役源代码保留，不纳入运行和质量门禁。

- [x] **Step 1: 添加活跃服务面回归断言，确保能力/API 文档只列当前服务。** 覆盖能力/版本清单、QA 依赖、门禁项目和 LocalAI 管理接口。
- [x] **Step 2: 运行新断言并处理旧能力条目、LocalAI 候选、bridge 依赖和过期活动说明。** 历史报告和设计资料按用户要求保留。
- [x] **Step 3: 更新入口文档、LocalAI 当前操作文档、质量门禁及版本/能力清单；移除旧音频运行集成并归档易混淆的操作说明。** 历史报告、设计记录、退役模型源码、独立文本训练、视频归档、VoiceDesign 计划和空间材料均保留。
- [x] **Step 4: 执行完整门禁。** `bash scripts/quality_gate.sh` 通过 shell 语法、Steam Audio 核心、锁检查、Ruff、能力校验及 22 项 unittest。

## Completion Contract

- `start.sh` 仅预检并启动 8300/8301，主控制面不引用已退役模型环境。
- 仓库运行代码、候选/能力清单和门禁中不存在旧音频服务与 worker bridge 集成。
- 控制面保留标准音频导出及 Steam Audio 空间上传/进度合同。
- VoiceDesign 继续使用全局 GPU 锁；锁行为被无模型回归测试覆盖并有独立存档文档。
- 历史报告、设计资料、任务记录和退役模型源码保留为归档，不再作为当前服务说明；`storage/` 用户数据保留。
- 本次不创建 git commit。
