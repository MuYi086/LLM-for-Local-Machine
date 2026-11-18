# 设计

## Context

原始需求见 [task10.md](../../../../task10.md)；完整设计与取舍已沉淀于
[2026-11-17 设计文档](../../../../docs/superpowers/specs/2026-11-17-audio-cpp-separation-alignment-design.md)，
实施步骤见[实施计划](../../../../docs/superpowers/plans/2026-11-17-audio-cpp-separation-alignment.md)。
本 change 归档时按主规格基线复核，设计要点与该文档一致。

## Goals / Non-Goals

- Goals：四个分离/对齐模型复用共享 audio.cpp runtime、LocalAI provider、GPU 锁与隔离机制，
  以 8344–8347 独立服务暴露；每模型提供真实 API demo。
- Non-Goals：不新增独立推理 runtime；不重复实现上传/锁/错误映射；不引入 Torch；
  不改动其余十一个既有音频服务的合同。

## Decisions

- 源分离采用一次 transform 请求并以 `X-Audio-Stems` 响应头清单取回全部 stem（多次 transform
  会重复昂贵分离，故不采用）。
- 强制对齐把已知逐字稿作为 transcription prompt 传递，保留 verbose JSON 的词级时间字段；
  无 `words` 时间戳即返回上游合同错误。
- 服务合同、端口、路由全部登记进 `localai/manifests/audio-cpp-models.json` 与
  `localai/manifests/capabilities.json`，不在服务目录硬编码。

## Risks / Trade-offs

- stem 清单异常或中途下载失败 → 统一拒绝并清理已暂存文件，不返回部分结果。
- 文档口径曾落后清单 → 归档时以 `scripts/check_service_inventory.py` 机器校验收敛（见
  [verification.md](verification.md)）。
