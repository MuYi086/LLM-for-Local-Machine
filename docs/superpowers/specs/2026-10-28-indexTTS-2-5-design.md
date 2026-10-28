# IndexTTS-2.5 HTTP 服务设计

## 意图与目标

在 `model-audio/indexTTS-2.5/` 增加本地 IndexTTS-2.5 语音克隆服务，由 `start.sh` 在 8323 端口暴露 `POST /v1/index-team/indexTTS-2.5`。调用方先上传参考音频并取得逻辑 `audio_path`，再提交文本、语言及可选情感向量/语速因子，服务返回并保存 WAV。

## 范围与非目标

- HTTP 进程只负责请求校验、共享音频引用、健康检查、GPU 排队和一次性 worker 生命周期。
- worker 从 `indextts.infer_v2_5` 导入官方 `IndexTTS2`，调用 `infer(...)`，支持 ZH/EN/JA/ES/AR、8 维情感向量和 `duration_factor`。
- 不启用 Qwen 情感文本分类、不接收情感参考音频、不加入训练或 WebUI。
- 请求过程不下载模型或辅助权重，不修改 hf-mirror 源文件。

## 约束

- 仓库运行环境固定 Python `3.12.13`；服务使用独立 uv 项目，模型、Torch 与 CUDA 依赖不进入共享运行时。
- 官方 IndexTTS 源码放在仓库外的 `$HOME/LLM-depency/index-tts`，固定 `v2.5.0`；服务自行声明推理所需依赖，不安装其 `requires-python <3.12` 的打包入口。
- 主模型来自 `$HF_MIRROR_DIR/IndexTeam/IndexTTS-2.5`；Wav2Vec2-BERT、CAMPPlus、BigVGAN 使用已有镜像目录。每个 worker 在运行缓存中创建只读符号链接视图，避免向模型目录写入 `hf_cache/`。
- 共用 `GPU_LOCK_FILE`，使用一次性 worker 和进程组清理；默认锁等待上限 900 秒，推理超时默认 900 秒。
- 输入参考音频按共享 `AudioReferenceStore` 逻辑标识读取，生成 WAV 保存到 `storage/clone/`。
- 不提交权重、外部源码或运行缓存；代码/API 变化同步 README、质量门禁和无模型 QA。

## 方案

1. 直接把 `indextts` 作为 Python 包依赖：上游包声明 Python `<3.12`，与仓库固定运行时冲突，放弃。
2. 把官方源码复制进服务目录：会把第三方源码带入仓库，违反开发边界，放弃。
3. 从仓库外的固定官方源码目录导入推理入口，并在服务 uv 项目显式安装推理依赖：保留 Python `3.12.13`、上游实现和独立运行环境，选用此方案。若真实导入或锁解析发现不可兼容，按错误修正最小依赖/兼容点，不改变仓库 Python 版本。

## 留给实现者的决定

按现有克隆服务合同设计 JSON 请求字段和错误码；将模型文件/依赖缺失反映在健康状态及请求预检中。对外只公开任务指定的主路由及共享上传、音频检查、健康路由。
