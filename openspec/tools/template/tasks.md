# 任务

## 1. <阶段一标题>

- [ ] 1.1 <任务；标注实现文件与对应测试>
- [ ] 1.2 <任务>

## 2. <阶段二标题>

- [ ] 2.1 <任务>
- [ ] 2.2 <任务>

## 3. 验证与收敛

- [ ] 3.1 无模型质量门禁：bash -n start.sh、bash scripts/quality_gate.sh、
      uv run --project qa --locked python -m unittest discover -s tests -v 全部通过
- [ ] 3.2 服务口径校验：清单、start.sh、README、AGENTS 同步（check_service_inventory.py 通过）
- [ ] 3.3 能力清单校验：合同变化后 capabilities.json 同步并通过 validate_capabilities.py
- [ ] 3.4 真实模型验证（如适用）：单独串行执行，样例入口 tests/audio_cpp/generate_examples.py
      或 tests/ornith_1_5_9b_mtp_benchmarks/generate_example.py，结果记录到 verification.md
- [ ] 3.5 归档前同步主规格与 coverage.md

> 无模型回归通过不能勾选为真实模型验证完成；未验证项保持未勾选并在 verification.md 说明。
