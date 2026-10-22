# Qwen3-ASR 原生短样本基线

- 采集日期：2026-10-20
- 服务：镜像内 `Qwen3_ASR_1.7B` 原生 Transformers worker
- 状态：单次短样本记录；没有参考文本，不构成质量或性能结论。
- 当前 LocalAI 裸机部署与模型镜像维护范围见[升级记录](../../docs/localai-audiocpp-upgrade.md)。

## 样本与结果

使用工作区内未纳入 Git 的本机 WAV 样本，通过 `POST /v1/qwen3/asr` 处理一次。输入为 11.008 秒、
48 kHz、双声道；响应为 HTTP 200，返回语言 `Chinese` 和 41 个转写字符。报告不保存转写正文，
没有可确认对齐的参考文本，因此没有计算 CER/WER。

| 指标 | 结果 | 口径 |
| --- | --- | --- |
| API `elapsed_seconds` | 3.863 秒 | worker 内模型计时，不含上传和 GPU 排队 |
| Worker 进程时间 | 8.62 秒 | 含 Python 启动与权重加载 |
| 持锁时间 | 10.628 秒 | 含 2 秒 CUDA 回收等待 |
| 全卡已用显存峰值 | 6,350 MiB | `nvidia-smi` 全卡汇总；开始 1,640 MiB，结束 1,611 MiB |
| 上传大小 | 2,113,614 字节 | 原始 staging 输入 |

该缓存未固定 Hub revision，仅为当时本机 native 运行记录。下列文件 hash 用于描述该次运行环境，
不代表完整模型快照锁定：

| 文件 | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6` |
| `model-00002-of-00002.safetensors` | `6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc` |
| `config.json` | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| `generation_config.json` | `1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c` |

## 限制

单次短样本、无参考文本、无 CER/WER，也没有 Q8 对照或重复测量。该结果不能用于证明模型质量、
不同 provider 等价或部署容量。LocalAI backend 的本机构建、安装和无模型启停结果见
[`P1-P2.md`](P1-P2.md)；真实 LocalAI 模型请求尚未运行。
