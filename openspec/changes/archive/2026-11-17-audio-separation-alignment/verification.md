# 验证记录与未完成项

本 change 于 2026-11-17 由
[实施计划](../../../../docs/superpowers/plans/2026-11-17-audio-cpp-separation-alignment.md) 落地；
本记录为 2026-11-19 归档时按当前源码重新核验的结果，严格区分静态核对、无模型回归与真实模型验证三层。

## 已执行检查（2026-11-19 归档复核）

- `python3 scripts/check_service_inventory.py` 通过：十五个音频服务、二十一个 HTTP 进程口径与
  清单一致（本次归档同时修正了 AGENTS.md 落后于清单的 12 处漂移，并把该脚本纳入质量门禁）。
- `bash -n start.sh` 通过。
- `uv run --project qa --locked ruff check/format` 通过，含本次补入 lint 清单的 6 个服务目录
  （修复 MOSS-SoundEffect-v2 main.py 的 import 分隔；其余 5 个目录本已合规）。
- `scripts/localai/validate_capabilities.py` 能力清单校验通过（含四个新服务的合同快照）。
- 无模型回归 `python -m unittest discover -s tests`：232 个用例中 199 个通过。
  33 个失败全部位于 `test_start_services.StartServicesTests`：本机当时有真实服务运行
  （start.sh 于 2026-11-17 启动，21 个端口监听中），该类用例要求端口空闲，端口预检
  报 `Address already in use`。属环境冲突而非回归；CI（`.github/workflows/quality.yml`）在
  干净环境执行同一门禁。停服后本地复跑 `python -m unittest test_start_services` 可复核。

## 已知问题

- `scripts/quality_gate.sh` 的 lint 清单此前遗漏 6 个含 main.py 的服务目录（含本 change 的 4 个），
  已在归档时补齐并写入 [经验索引](../../../../docs/agent-memory/decisions.md)。
- AGENTS.md 服务口径曾落后清单（17 进程/11 音频服务），已修正并引入机器校验防止复发。

## 真实模型验证

- 归档复核未重跑四个模型的真实 demo；真实样例入口为 `tests/BS-RoFormer-ep368-GGUF/`、
  `tests/Qwen3-ForcedAligner-0.6B-GGUF/`、`tests/Mel-Band-RoFormer-GGUF/`、`tests/HTDemucs-GGUF/`
  各自的 `generate_example.py`，需在服务运行时单独串行执行，输出写入运行数据目录。
- 无模型回归通过不代表真实模型验证完成；真实结果以各测试目录的运行记录为准。

## 待完成

- 停服后本地复跑 `test_start_services` 全部用例，确认 33 个环境冲突用例在干净端口下通过
  （CI 已覆盖同一门禁）。
