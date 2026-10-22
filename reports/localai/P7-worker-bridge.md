# P7：LocalAI one-shot worker bridge 实施记录

- 状态：`implemented_unverified`（5 个服务项目、5 个固定 model ID 已接入；真实模型推理未验收）
- 日期：2026-10-21
- 本地服务默认 provider：`native`
- 模型权重：依用户要求未下载

## 已实现

- LocalAI v4.11.0 completion 请求承载版本化 `llm-local.worker` v1 envelope，由固定的
  `llm-local-worker` loopback gRPC backend 分发。
- [`worker-bridge.json`](../../localai/manifests/worker-bridge.json) 固定 5 个 worker ID、项目目录、入口、
  输出类型、超时和环境变量白名单。请求不能指定其他 Python 模块、shell 命令或 worker 路径。
- 每次 RPC 使用目标服务自己的 `uv run --project <project> --no-sync` 启动 one-shot worker；
  服务调用侧持有共享 GPU 锁，并在 backend shutdown 后才释放锁。
- worker 失败、超时或取消时清理临时输入/产物并终止进程组。受管 LocalAI 停止流程会验证 request marker，
  回收脱离宿主机进程组的 worker。

| 服务 | 固定 model ID | 原生输出合同 |
| --- | --- | --- |
| SheetSage2 | `llm-local.sheetsage2.worker.v1` | ZIP |
| MERT-v2-FullSong | `llm-local.mert-v2-fullsong.worker.v1` | embedding JSON / frames NPZ |
| YuE2-Vae | `llm-local.yue2-vae.worker.v1` | encode NPY / decode WAV |
| YuE2-3B | `llm-local.yue2-3b.worker.v1` | WAV |
| TIGER-DnR | `llm-local.tiger-dnr.worker.v1` | 三 Stem ZIP 文件引用 |

候选登记和核验不含模型文件，不会下载权重：

```bash
python3 scripts/localai/manager.py register-worker-bridge-candidates
python3 scripts/localai/manager.py verify-worker-bridge-candidates
```

受管启动还需 `LLM_LOCALAI_ENABLED=true`、`LLM_LOCALAI_MANAGED=true`、
`LLM_LOCALAI_WORKER_BRIDGE_ENABLED=true`。非受管运行时须自行启动 bridge，并将
`llm-local-worker:127.0.0.1:50052` 加入 `LOCALAI_EXTERNAL_GRPC_BACKENDS`。

## 验证边界

仓库无模型测试覆盖协议、worker allowlist、路由、输出类型和文件清理。它不验证模型加载、真实输出质量、
峰值 GPU 显存、最长输入、backend 停止后的显存回落或故障恢复。上述 E2/E3 验收须等用户准备权重后逐项执行。
完整质量门禁的本次结果记录于 [`localai-audiocpp-upgrade.md`](../../docs/localai-audiocpp-upgrade.md)。

provider 设为 `native` 即回到原服务 worker；LocalAI 候选 YAML 不影响 native 路径。

## 2026-10-22 复核修复

五个文件成品合同现均使用最大 4 GiB 的共享 file_ref，避免固定 LocalAI 50 MiB gRPC 接收限制。
41 MiB 与 129 MiB 静音 WAV 已经真实 LocalAI HTTP/Go gRPC/无模型 worker 链路验证，内容与消费清理通过。
进程组统一收尾、Free 失败隔离、目录迁移和恢复流程均已补回归；详见 [升级记录](../../docs/localai-audiocpp-upgrade.md#7-2026-10-22-复核修复与当前验证)。
此记录不代表五个 worker 的真实权重推理或输出质量验收完成。
