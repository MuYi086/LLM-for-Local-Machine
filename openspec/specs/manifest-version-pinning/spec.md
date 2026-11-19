# 清单版本锁定与外部二进制核验

## Purpose

覆盖仓库外第三方二进制与源码 revision 的锁定和核验：LocalAI v4.11.0 与 audio.cpp backend revision、
stable-diffusion.cpp commit、task enum 补丁前置、sd.cpp 摘要校验，以及 LocalAI 候选注册方式。
服务数量、路由与 GGUF 文件名仍以 [project-overview](../project-overview/spec.md) 的清单层级为准。

## 实现依据

- [localai/manifests/versions.json](../../../localai/manifests/versions.json)
- [localai/manifests/qwen3-tts-voicedesign-candidate.yaml](../../../localai/manifests/qwen3-tts-voicedesign-candidate.yaml)
- [sdcpp/manifests/versions.json](../../../sdcpp/manifests/versions.json)
- [scripts/localai/manager.py](../../../scripts/localai/manager.py)
- [scripts/localai/patches/audio-cpp-task-kinds.patch](../../../scripts/localai/patches/audio-cpp-task-kinds.patch)
- [scripts/sdcpp/manager.py](../../../scripts/sdcpp/manager.py)
- [docs/localai-audiocpp-upgrade.md](../../../docs/localai-audiocpp-upgrade.md)
- [README.md](../../../README.md)

## Requirements

### Requirement: 版本与 commit 以锁文件为准

LocalAI MUST 固定使用 v4.11.0（source commit `58830f7ac508845a6f4efa32cfca06af422d4d82`）与 CUDA 12
variant 的 audio.cpp backend，backend MUST 固定 revision `0382d37fe6e1a817b6b8fed5dcaab0e11fd1e8e6`；
stable-diffusion.cpp MUST 固定 commit `a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf`。版本事实 MUST 以
`localai/manifests/versions.json` 与 `sdcpp/manifests/versions.json` 为准，revision 变化 MUST 重新部署
并重新核验，MUST NOT 以旧指纹冒充新版本。

#### Scenario: 升级 backend revision

- **WHEN** 部署新的 audio.cpp backend revision
- **THEN** 更新锁定文件、重新构建并核验后再运行，不沿用旧 revision 的指纹记录

#### Scenario: 查询当前版本

- **WHEN** 维护者确认第三方二进制版本
- **THEN** 以两份 versions.json 的 commit/版本字段为准，不以历史报告推断

### Requirement: 指纹核验先于运行

`scripts/localai/manager.py verify-install` MUST 只核对并记录本地制品指纹，MUST NOT 下载权重或构建
backend；sd.cpp CLI 摘要 MUST 由 `SDCPP_BIN_SHA256` 显式设置，启动预检 MUST 校验该摘要与模型组件，
不一致 MUST NOT 继续启动。模型权重与第三方源码 MUST 位于仓库外，MUST NOT 提交进仓库。

#### Scenario: 摘要不匹配

- **WHEN** `sd-cli` 实际摘要与 `SDCPP_BIN_SHA256` 不一致
- **THEN** 预检失败并阻止启动，不用未核验的二进制生成

#### Scenario: 核验缺失权重

- **WHEN** 镜像目录缺少锁定的 GGUF 或模型组件
- **THEN** 核验失败并报告缺失路径，不自动下载或回退到错误精度权重

### Requirement: task enum 补丁是构建前置条件

LocalAI v4.11.0 的 backend adapter 未镜像 audio.cpp 新 revision 的 `TurnDetection`、`WakeWord` 与
`AudioClassification` task；构建 backend 前 MUST 应用
`scripts/localai/patches/audio-cpp-task-kinds.patch`，同步 task enum、名称与双向映射，并保留
`-Werror=switch` 检查。源码 revision 变更时 manager MUST NOT 仅刷新旧 backend 的指纹，MUST 要求重新
部署后再核验。

#### Scenario: 未打补丁直接构建

- **WHEN** backend 在应用补丁前构建
- **THEN** 不视为合规部署，需打补丁后重建并重新 verify-install

### Requirement: 候选注册由 manager 受控完成

十五个 LocalAI 候选 MUST 由 `scripts/localai/manager.py` 注册：在 LocalAI models 目录建立指向本地 GGUF
的受控符号链接并原子写入 YAML，MUST NOT 复制权重、MUST NOT 手工写入模型目录；注册后的 readiness
检查 MUST NOT 触发推理或权重加载。`verify-audio-cpp-candidates` MUST 可用于核对已注册候选。

#### Scenario: 注册候选

- **WHEN** start.sh 或 manager 注册十五个候选
- **THEN** 只建立受控符号链接与原子 YAML，权重保持在镜像目录，readiness 不加载模型

#### Scenario: 手工改动模型目录

- **WHEN** 绕过 manager 手工增删 LocalAI models 目录条目
- **THEN** 该状态不受受控注册合同保障，需用 manager 重新核对后再运行

### Requirement: 候选 YAML 模板不落绝对路径

候选 YAML 模板 MUST 使用占位符（如 `__QWEN3_VOICEDESIGN_MODEL_PATH__`）表达受控相对 GGUF 引用，
由 manager 在注册时替换；模板中的 `family`、`task`、`backend`、`device` MUST 与候选清单一致，
MUST NOT 在仓库中提交解析后的机器专用绝对路径或密钥。

#### Scenario: 注册 VoiceDesign 候选

- **WHEN** manager 注册 `llm-local.qwen3-tts.1p7b-voicedesign`
- **THEN** 占位符被替换为清单声明的受控相对引用，family/task 与清单一致，仓库中不留绝对路径
