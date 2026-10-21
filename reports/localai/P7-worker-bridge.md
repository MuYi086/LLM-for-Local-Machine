# P7：LocalAI one-shot worker bridge 实施记录

- 状态：`implemented_unverified`（10 个服务项目、11 个 model ID 已接入；真实模型推理未验收）
- 日期：2026-10-21
- 本地服务默认 provider：`native`
- 模型权重：依用户要求未下载

## 已实现

- LocalAI v4.11.0 completion 请求承载版本化 `unitale.worker` v1 JSON envelope，由固定
  `unitale-worker` loopback gRPC backend 分发。
- `localai/manifests/worker-bridge.json` 固定 11 个 worker ID、项目目录、入口脚本、输出参数/类型、
  超时和环境变量白名单。请求不能指定其他 Python 模块、shell 命令或 worker 路径。
- 每次 RPC 使用目标服务自己的 `uv run --project <project> --no-sync` 启动 one-shot worker；
  服务调用侧持有共享 GPU 锁，并在 LocalAI backend shutdown 后才释放锁。
- worker 失败、超时或取消时清理临时输入/产物并终止进程组。受管 LocalAI 停止流程会验证
  bridge request marker 并回收脱离宿主机进程组的 worker。
- 十个项目分别为 SheetSage2、MERT-v2-FullSong、YuE2-Vae、SoulX-FlashHead、Ditto、YuE2-3B、
  MOSS-SoundEffect、Step-Audio-EditX、MOSS-Audio-4B、TIGER-DnR。MOSS-Audio 的 Thinking 与
  Instruct 变体共用一个项目，各有固定 model ID。
- 输出映射保留现有合同：SheetSage2 ZIP；MERT JSON/NPZ；YuE2-Vae NPY/WAV；人像视频 MP4；
  YuE2-3B、MOSS-SoundEffect、Step-Audio-EditX WAV；MOSS-Audio JSON 文本；TIGER 三 Stem ZIP。
  TIGER 大文件通过受控目录中的随机文件引用传递，服务校验 ID、路径、MIME、大小和固定 ZIP 内容后删除引用。
- 各服务均可通过对应 `UNITALE_*_PROVIDER=native` 保持原 provider；LocalAI provider 需显式开启。

## 候选登记

在忽略目录 `storage/.cache/localai/models/` 登记并验证 worker bridge YAML。登记不包含权重路径，
也不会下载或读取模型权重：

```bash
python3 scripts/localai/manager.py register-worker-bridge-candidates
python3 scripts/localai/manager.py verify-worker-bridge-candidates
```

受管启动还需 `UNITALE_LOCALAI_ENABLED=true`、`UNITALE_LOCALAI_MANAGED=true`、
`UNITALE_LOCALAI_WORKER_BRIDGE_ENABLED=true`。不受管运行时须自行启动 bridge，并将
`unitale-worker:127.0.0.1:50052` 加入 `LOCALAI_EXTERNAL_GRPC_BACKENDS`。

## 验证证据与限制

`UV_OFFLINE=1 bash scripts/quality_gate.sh` 已通过：根目录 363 项、ACE-Step 12 项、
Stable Audio 10 项；同时 Ruff、格式、锁文件和 capability manifest 检查通过。LocalAI 能力快照
覆盖 24 个服务入口、23 个 uv 服务项目、79 条路由、17 个请求 schema 和 268 个请求字段。
worker bridge 候选登记与核验、Qwen3-TTS Base/VoiceDesign 文件核验、`manager.py verify-install`、
`bash -n start.sh` 与 `git diff --check` 也通过。无模型测试只证明代码路径、协议与产物合同，
不提供真实 LocalAI gRPC 模型请求证据。
25 个有锁项目已用 `uv sync --locked --offline` 同步；没有下载 Python 包或模型权重。

尚未验收权重加载、真实模型输出质量、峰值 GPU 显存、最长输入、模型加载后显存回落、真实故障恢复，
也没有开始 P8 切流及观察期。模型权重和符合约定的测试样本由用户准备；本次没有下载模型。

## 回滚和下一阶段

把对应 `UNITALE_*_PROVIDER` 设为 `native` 即回到现有服务 worker，无需删除候选 YAML。用户准备权重后，
逐模型执行真实请求、输出合同、backend shutdown、GPU 资源和异常清理验收；E3 通过后再按 P8 门槛放量。
