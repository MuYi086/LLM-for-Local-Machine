# MOSS 与 FLUX.3 Klein 图片服务

## Purpose

覆盖 stable-diffusion.cpp 两个图片 API（8501/8502）的固定生成参数、上传与临时文件处理、
按请求启动的 CLI provider 边界，以及隔离恢复入口。

## 实现依据

- [inference_gateway/src/inference_gateway/sdcpp/execution.py](../../../inference_gateway/src/inference_gateway/sdcpp/execution.py)
- [inference_gateway/src/inference_gateway/sdcpp/worker.py](../../../inference_gateway/src/inference_gateway/sdcpp/worker.py)
- [scripts/sdcpp/manager.py](../../../scripts/sdcpp/manager.py)
- [sdcpp/manifests/models.json](../../../sdcpp/manifests/models.json)
- [docs/sdcpp-runtime.md](../../../docs/sdcpp-runtime.md)

## Requirements

### Requirement: MOSS-Image-SFT 固定生成合同

MOSS-Image-SFT API（8501，`POST /v1/openmoss/MOSS-Image-SFT-Q8_0`）MUST 固定使用
text-to-image、官方 tokenizer、CFG 3 和 16 像素尺寸对齐；请求 MUST 在共享 GPU 锁内生成单张 PNG；
worker 未确认退出时 MUST 保留 quarantine。

#### Scenario: 正常文生图请求

- **WHEN** 客户端提交文生图请求
- **THEN** 服务在 GPU 锁内生成单张 PNG 并返回，生成完成后 worker 可确认退出

#### Scenario: worker 退出未确认

- **WHEN** 生成后无法确认 CLI worker 进程退出
- **THEN** 服务保留 quarantine 标记，后续 GPU 请求被阻止

### Requirement: FLUX.3 Klein 默认参数与编辑路由

FLUX.3 Klein API（8502，`POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8` 与 `/edit`）
MUST 使用 1024×1024、16 步、CFG 1.0、Euler 和 CPU offload 默认值；编辑路由接收 PNG/JPEG
multipart 上传，默认上限 64 MiB，按块暂存到 `$RUNTIME_CACHE_DIR/sdcpp_uploads` 并在请求结束后删除。

#### Scenario: 编辑请求

- **WHEN** 客户端上传不超过上限的 PNG/JPEG 并请求编辑
- **THEN** 上传按块暂存到 `sdcpp_uploads`，生成完成后临时文件被删除

#### Scenario: 上传超限或格式不受支持

- **WHEN** 编辑上传超过 64 MiB 或不是 PNG/JPEG
- **THEN** 请求在推理前被拒绝，临时内容被清理

### Requirement: 参考图仅由清单配置

仅使用 `--ref-image` 时 MUST 由模型清单明确配置；实现 MUST NOT 默认启用参考图路径。

#### Scenario: 清单未配置参考图

- **WHEN** 模型清单未声明参考图组件
- **THEN** 服务不接受需要 `--ref-image` 的请求路径

### Requirement: CLI provider 按请求启动且预检完备

inference_gateway.sdcpp MUST 作为按请求启动的 sd.cpp CLI provider 运行，无 HTTP 端口；
`sdcpp/manifests/` MUST 记录外部版本与 MOSS、FLUX.3 Klein 清单；`scripts/sdcpp/manager.py`
负责本机状态、预检和生成，预检包括 sd.cpp 可执行文件摘要与各模型组件核验。

#### Scenario: 启动预检

- **WHEN** start.sh 或 manager 执行预检
- **THEN** sd.cpp 可执行文件摘要与清单声明的模型组件一致才继续，否则启动失败

### Requirement: 隔离恢复统一入口

CLI provider 的隔离恢复 MUST 按 PID/starttime 与进程组核验，并统一使用既有
`scripts/localai/recover_quarantine.py` 入口。

#### Scenario: 需要恢复隔离

- **WHEN** CLI worker 异常退出且状态待确认
- **THEN** 维护者通过 `recover_quarantine.py` 核验进程身份后解除隔离，不使用第二套工具

### Requirement: 视频模型不得描述为可用

视频模型尚未接入当前启动项；规格与文档 MUST NOT 将其描述为当前可用能力。

#### Scenario: 询问视频生成能力

- **WHEN** 客户端或文档引用视频模型
- **THEN** 当前能力口径不包含视频生成，历史源码只作归档说明
