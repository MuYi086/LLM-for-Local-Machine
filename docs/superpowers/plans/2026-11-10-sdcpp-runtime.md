# sd.cpp Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task by task.

**Goal:** 在未选定模型时完成受共享锁保护、可恢复、可验证的图片/视频 CLI 执行层。

**Architecture:** stdlib-only provider 位于 inference_gateway；独立运维 CLI 读取空默认清单。
新进程组配合启动许可和原子 quarantine 实现可确认的生命周期；复用既有恢复入口。

**Tech Stack:** Python 3.12、Linux /proc、fcntl.flock、unittest、Ruff。

**Spec:** docs/superpowers/specs/2026-11-10-sdcpp-runtime-design.md

## Global Constraints

- 所有 GPU provider 共用 GPU_LOCK_FILE；不安装 Torch，不执行真实模型，不提交机器绝对路径或模型。
- schema_version=1；排队/运行/回收时限有限正数；CUDA 延迟有限非负。
- status 与 validate 不加载模型；start.sh 仍管理十四个 HTTP 进程。
- 用户已批准架构并要求开始执行，本会话直接实施；改动留在 feature branch 供审阅。

## Review Focus

- 父进程在 Popen 和 marker 更新之间崩溃：worker 未获许可不能加载模型。
- 根进程退出但后代仍活动：确认并回收整个进程组后才能解锁。
- PID 被复用或 marker 被替换：不能误杀或删除其他请求的标记。
- 无效 NaN/布尔数值、未知字段及越界任务：预检拒绝且不启动 backend。
- 成功退出但没有有效产物：不得发布，不能覆盖用户历史输出。

### Task 1: 清单与配置合同

**Files:** inference_gateway/src/inference_gateway/sdcpp/{__init__,config,models}.py;
sdcpp/manifests/{models,versions}.json; tests/test_sdcpp_config.py

**Interfaces:** SdcppConfig.from_env(); load_models(path) -> dict[str, ModelSpec]; GenerationRequest。

- [x] 写清单、参数限制、环境校验和无模型状态测试并确认失败。
- [x] 实现固定参数映射与 CPU 文件预检。
- [x] 运行 `uv run --project qa --locked python -m unittest discover -s tests -p 'test_sdcpp_config.py' -v`，预期全部通过。

### Task 2: 进程执行与隔离恢复

**Files:** inference_gateway/src/inference_gateway/sdcpp/{processes,worker,execution}.py;
scripts/localai/recover_quarantine.py; tests/test_sdcpp_execution.py

**Interfaces:** generate(config, model, request, cancel_event=None) -> GenerationResult;
recover_sdcpp(marker_path, marker, timeout) -> None。

- [x] 使用伪 CLI 写成功/失败/超时/取消/共享锁/后代进程/恢复测试并确认失败。
- [x] 实现 stdin 启动许可、原子 marker、进程身份检查、进程组回收和独立输出发布。
- [x] 恢复入口持锁后重读 marker，再按 provider 分派 sd.cpp；保持既有 backend 行为。
- [x] 运行 `uv run --project qa --locked python -m unittest discover -s tests -p 'test_sdcpp*.py' -v`，预期全部通过。

### Task 3: 运维入口与文档

**Files:** scripts/sdcpp/manager.py; docs/sdcpp-runtime.md; README.md; AGENTS.md;
docs/global-gpu-lock.md; scripts/quality_gate.sh; tests/test_sdcpp_cli.py

**Interfaces:** status/validate/generate CLI；默认空清单；generate 返回产物和运行指标 JSON。

- [x] 写 CLI 无模型状态、配置错误和伪模型生成端到端测试并确认失败。
- [x] 实现入口、文档和质量门禁覆盖。
- [x] 运行新测试与 `bash scripts/quality_gate.sh`，预期通过；独立代码审查并处理重要发现。


## 执行结果

- 2026-11-10：在 `feat/sdcpp-runtime` 分支实施，改动保留未提交状态供审阅。
- 新增 30 项 sd.cpp 无模型测试；执行器使用临时伪 CLI，不运行真实模型或 CUDA。
- 首次根目录基线 68 项通过；最终质量门禁通过，根目录 98 项、Ornith 单独 discovery 16 项通过。
- Shell、无 SDK C++ 核心、四个 uv lock、Ruff 和能力清单校验均通过。
- 独立审查的两个 Important 问题均已 RED→GREEN：视频单帧预检、组回收前保留根进程身份。
- 选型、真实模型兼容性与性能、二进制构建来源由后续部署验证；默认模型清单仍为空。
- 取舍：本阶段仅核验 PNG/AVI 文件头；后代若主动脱离 session 不在当前进程组合同内。
