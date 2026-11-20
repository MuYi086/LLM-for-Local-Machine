# 验证记录与未完成项

本记录严格区分静态核对、无模型回归与真实模型验证三层；未验证项不写成已通过。
本次为模型下线变更，不引入新推理路径。

## 已执行检查（2026-11-20）

- `bash -n start.sh`：通过。
- `bash scripts/quality_gate.sh`：通过。包含 Steam Audio renderer core 测试、4 个 uv 项目锁校验、
  Ruff check/format（131 个文件）、`scripts/localai/validate_capabilities.py`、
  `scripts/check_service_inventory.py`、以及无模型 unittest。
- `python3 scripts/check_service_inventory.py`：通过。输出为「音频模型服务：15（audio.cpp 14、
  独立 runtime：qwen3_tts_voicedesign）／固定 HTTP 服务：4／start.sh 管理的 HTTP 进程总数：19」，
  README、AGENTS、start.sh 口径一致，无 8504/8506 残留。
- `uv run --project qa --locked python -m unittest discover -s tests -v`：179 个用例全部通过（OK）。
- `npx -y @fission-ai/openspec validate --all --strict`：改动后 15/15 通过；删除两个能力域主规格并归档后
  再次执行仍全部通过（见下方归档复核）。
- 全局 `~/.config/opencode/opencode.json`：已移除 `local-ornith` provider 与默认 `model` 字段；
  Key 文件 `~/.config/opencode/easel-bridge-api-key` 保留供图片/语音配置使用。原文件备份为
  `opencode.json.bak-20261120-remove-ornith`。

## 不确定事项

| 事项 | 状态 | 依据 / 责任方 | 临时处理 | 影响 / 确认结果 |
| --- | --- | --- | --- | --- |
| 全局 opencode 配置为仓库外文件 | 已确认 | 用户提供路径 `~/.config/opencode/opencode.json` | 直接编辑该文件，删除 local-ornith 与默认 model | 已执行并备份；Windows 原生 OpenCode 需用户自行同步对应配置与 Key 文件 |
| 历史 llama.cpp quarantine 标记 | 临时假设 | 正常关闭不产生该标记 | 删除 llama.cpp 恢复分支；不新增恢复工具 | 若存在遗留标记需人工删除 `.quarantine`，一次性处理，不阻塞 |
| 用户已有未提交改动 | 已确认 | `git status` 显示 README.md、model-text/README.md、openspec/coverage.md 已修改 | 在这些改动基础上增量编辑，不整体覆盖 | 已保留用户对旧测试目录重命名的清理，并在其基础上完成 Ornith 移除 |
| compaction 的 buffer/keep 数值 | 临时假设 | 原为 96k 本地模型调参 | 保留 `auto=true`、`buffer=16384`、`keep.tokens=10240` | 不影响正确性；若默认模型上下文不同，可由用户再调整 |

## 已知问题

- 无。`docs/superpowers/` 与 `tests/model_web_comparison/` 中仍含 Ornith 字样，按 AGENTS.md 属历史记录，
  仅作追溯，不推断当前接口。

## 真实模型验证

- 本次为模型下线变更，不引入新推理路径，无真实 GPU 验证需求。
- 未执行任何 `generate_example.py`；无模型回归通过不代表真实模型验证完成。

## 待完成

- 无。change 已归档，主规格与 coverage.md 已同步。
