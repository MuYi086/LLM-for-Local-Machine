# 已确认的技术决策与实测结论

每条经验注明来源与确认时间；实现变化后同步更新本文件，过时结论明确标注。

## GPU 与隔离

- 所有 GPU provider 必须共用同一 `GPU_LOCK_FILE`，任何绕过都会造成跨进程显存 OOM；
  隔离标记只能经 `scripts/localai/recover_quarantine.py` 在确认进程退出后解除。
  来源：[全局 GPU 锁说明](../global-gpu-lock.md)。
- Ornith 请求持锁后的顺序（`/props.is_sleeping` → 推理 → 卸载确认 → `CUDA_RELEASE_DELAY` →
  finally 释放）不可省略；异常路径也必须释放锁。来源：[AGENTS.md 架构约束](../../AGENTS.md)。

## Ornith 9B 上下文与压缩

- 96k（98304）是同机实测的舒适区上限：32k–96k 内吞吐稳定，128k 及以上明显劣化，
  不建议按上限口径配置。来源：[上下文档位实测](../../ornith-1.5-9B-Q8模型上下文测试舒适度区间.md)、
  [96k 性能分析](../../tests/ornith_1_5_9b_mtp_benchmarks/performance.md)（2026-11 前后多轮实测）。
- OpenCode 压缩请求的失败链是「输入挤占输出」：模板/分频必须在既有 GPU 租约内执行预算，
  超限返回标准 `context_length_exceeded`，静默缩短摘要会持续污染归档。
  来源：[压缩预算机制](../opencode-compaction-budget.md)。
- 跨 sleep 缓存上限 4096 MiB CPU RAM 是当前默认；缓存不跨进程重启，超限回退冷算，
  启动时会核验后端补丁能力，缺失时不得声称缓存生效。来源：[sleep 缓存文档](../llama-cpp-sleep-cache.md)。

## 服务口径与门禁

- OpenSpec CLI 的 npm 包名是 `@fission-ai/openspec`（不是 `openspec`，后者是无关占位包）；
  change 模板放在 `openspec/tools/template/`，不放在 `changes/` 下，避免 `openspec list`
  把模板识别为活动 change；确实需要零 delta 的 change（纯工具/文档类）时在其 `.openspec.yaml`
  声明 `schema: spec-driven` + `skip_specs: true`。来源：2026-11-19 工具接入实测，
  `openspec validate --all --strict` 全部通过。
- 服务事实的唯一权威是 `localai/manifests/audio-cpp-models.json`；文档曾落后实现
  （AGENTS 写 17 进程/11 音频服务，实际 21/15），因此引入
  `scripts/check_service_inventory.py` 纳入门禁，口径漂移会直接使门禁失败。
  来源：2026-11-19 校验脚本落地时的漂移记录（12 处不一致全部命中）。
- `scripts/quality_gate.sh` 的 lint 路径曾漏掉 6 个含 main.py 的服务目录（含 4 个分离模型），
  修复方式是把全部 audio.cpp 服务目录显式列入；新增服务目录时同步补该清单。
  来源：2026-11-19 门禁核对。
- 无模型回归（unittest discover）与真实模型验证是两层：前者通过不能证明真实模型可用，
  真实样例必须单独串行执行并如实记录。来源：[AGENTS.md 验证纪律](../../AGENTS.md)、
  [开发工作流](../dev-workflow.md)。

## audio.cpp 接入

- 新增 audio.cpp 模型的标准路径：清单登记 → start.sh 监督 → 共享 runtime 合同 →
  capabilities 快照 → 逐模型 demo；源分离用一次 transform 的 `X-Audio-Stems`，
  多次 transform 会重复昂贵分离。来源：
  [分离/对齐设计](../superpowers/specs/2026-11-17-audio-cpp-separation-alignment-design.md)。
- audio.cpp 的上游 family 声明不等于模型能力，接入前需按仓库
  [audio-cpp 分析](../../audio-cpp分析.md) 与实测确认。来源：[AGENTS.md 口径规则](../../AGENTS.md)。
