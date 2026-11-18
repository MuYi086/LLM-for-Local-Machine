# 接入四个 audio.cpp 分离与对齐模型

## Why

[task10.md](../../../../task10.md) 提出把 `~/hf-mirror/audio.cpp-gguf` 中的
BS-RoFormer-ep368-GGUF、Qwen3-ForcedAligner-0.6B-GGUF、Mel-Band-RoFormer-GGUF、HTDemucs-GGUF
四个模型经 start.sh 接入 audio.cpp，在 `model-audio/<模型目录>` 提供服务，以 8344–8347 对外暴露，
路由为 `/v1/audio-cpp/<模型名称>`，并在 tests 目录完成 demo 测试。需求设计与实施计划见
[设计](../../../../docs/superpowers/specs/2026-11-17-audio-cpp-separation-alignment-design.md)与
[计划](../../../../docs/superpowers/plans/2026-11-17-audio-cpp-separation-alignment.md)。

## What Changes

- 新增四个 LocalAI/audio.cpp 候选并注册到 `localai/manifests/audio-cpp-models.json`
  （bs_roformer 8344、qwen3_forced_aligner 8345、mel_band_roformer 8346、htdemucs 8347）。
- 新增 `model-audio/` 四个服务入口：BS-RoFormer、Mel-Band-RoFormer、HTDemucs 走一次
  transform 的 `X-Audio-Stems` 源分离合同；Qwen3-ForcedAligner 走逐字稿 prompt 的强制对齐合同。
- start.sh 把四个端口、路由、目录纳入启动监督与 readiness 预检；
  `localai/manifests/capabilities.json` 同步快照。
- 每个模型新增 tests 目录真实 API demo（generate_example.py 与 README）。

## Capabilities

### New Capabilities

无（能力并入既有域）。

### Modified Capabilities

- `audio-cpp-services`：新增源分离与强制对齐两类合同及四个服务条目。
- `service-lifecycle-and-startup`：启动预检与监督范围扩至 8344–8347。
- `quality-gate-and-testing`：Ruff lint 覆盖新增服务目录（本 change 归档时补齐）。

## Impact

- 代码：`localai/manifests/audio-cpp-models.json`、`localai/manifests/capabilities.json`、
  `start.sh`、`model-audio/audio_cpp_runtime/`、四个 `model-audio/<模型>/main.py`、
  四个 `tests/<模型>/`。
- 服务口径：音频模型服务 11→15，HTTP 进程 17→21；README 与 AGENTS.md 服务表同步。
- 边界：权重只读镜像目录；不新增 Torch；共享 GPU 锁与隔离机制不变。
