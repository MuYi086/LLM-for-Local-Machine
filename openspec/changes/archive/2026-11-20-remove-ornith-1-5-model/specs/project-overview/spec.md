## MODIFIED Requirements

### Requirement: 服务口径与清单一致

start.sh 管理的 HTTP 进程数量 MUST 与 `localai/manifests/audio-cpp-models.json` 推导值一致，
固定 HTTP 服务为控制面（8300）、Easel bridge（8500）、MOSS-Image-SFT（8501）、FLUX.3 Klein（8502），
合计四个；十五个音频模型服务加四个固定服务共十九个 HTTP 进程。README.md 与 AGENTS.md 的服务表、
数量声明 MUST 与该口径一致，由 `scripts/check_service_inventory.py` 在质量门禁中核对。
端口 8504 与 8506 及 `/v1/ornith-ai/...` 路由 MUST NOT 出现在服务清单与文档中。

#### Scenario: 查询当前服务清单

- **WHEN** 核对 README、AGENTS、start.sh 与清单
- **THEN** 十五个音频服务、四个固定服务、十九个 HTTP 进程的口径全部一致，无 8504/8506 残留

#### Scenario: 新增或退役模型服务

- **WHEN** 音频候选增减或固定服务变化
- **THEN** 清单、start.sh、README、AGENTS 同步更新，并通过 `scripts/check_service_inventory.py`
