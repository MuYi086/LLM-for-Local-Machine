# LongCat-Video-Avatar-1.5 量化搭配 WanGP 最佳实践

> **历史记录：** LongCat-Video-Avatar-1.5 集成已退役，本文不再表示当前受支持的服务或运行建议。
> 记录保留 2026-11-02 的本机配置和基准结果，源代码和本地模型文件未因此删除。

本文记录本项目在单张 16 GiB RTX 4070 Ti SUPER 上，通过 WanGP 推理
LongCat-Video-Avatar-1.5 Quanto 权重的已验证配置、配置变更和基准结果。
本次测试当时选择 **MMGP Profile 0 + 手动 9216 MiB 视频预加载 + SDPA**。

服务 API、权重路径和请求方式见本目录的 [README.md](README.md)。8390 使用
WanGP 推荐的 `shared.api.init()` / `session.submit_task()` API；WanGP 自带 MMGP，
无需在 8390 的 uv 环境中另装 MMGP。

## 测试时运行配置（历史）

测试时 WanGP 配置文件位于 `$HOME/LLM-depency/WanGP/wgp_config.json`。以下字段构成本次
视频推理配置：

| 字段 | 锁定值 | 用途 |
| --- | --- | --- |
| `profile` | `0` | 使用 MMGP Profile 0 |
| `video_profile` | `0` | LongCat 视频任务的显存管理档位 |
| `video_preload_mode` | `"manual"` | 使用手动预加载额度 |
| `video_preload_in_VRAM` | `9216` | 手动预加载目标，单位 MiB，即 9 GiB |
| `attention_mode` | `"sdpa"` | 使用 PyTorch SDPA attention |

这次实测启动时的 `image_profile` 为 `4`，`image_preload_mode` 和
`audio_preload_mode` 为 `"default"`。这些字段不是 LongCat 视频预加载额度。
8390 的 `configure_wangp.py` 会在启动时按 `LONGCAT_AVATAR_MEMORY_PROFILE`
同步 `profile`、`video_profile` 和 `image_profile`，并设置 SDPA；它不会改写
`video_preload_mode` 或 `video_preload_in_VRAM`。因此已有配置文件中的 manual
9216 MiB 设置会在常规服务重启时保留。若删除或重新生成 WanGP 配置文件，应在
配置脚本完成后重新确认上述字段。

只需修改配置 JSON 中的相关字段，保留其余 WanGP 配置：

```json
{
  "profile": 0,
  "video_profile": 0,
  "video_preload_mode": "manual",
  "video_preload_in_VRAM": 9216,
  "attention_mode": "sdpa"
}
```

`video_preload_in_VRAM` 是模型块的预加载目标，不是整个推理进程的显存硬上限。
WanGP/MMGP 仍需为激活、VAE、缓存和运行时工作区留出显存；实际占用由生成阶段的
工作负载决定。8390 的空闲显存门槛是在请求开始前做容量预检，也不是推理期间的
显存限制。

## 配置变更记录

此前已将 LongCat 切到 Profile 0 和 SDPA，并将视频预加载模式设为 `manual`。
本次 9 GiB 试验在手动 10 GiB 配置基础上，只将：

```text
video_preload_in_VRAM: 10240 → 9216
```

五段测试完成后，当时将 `9216` 保留为本机选定值。WanGP 配置由服务外部维护，
项目启动配置脚本保留该手动预加载字段；本次没有修改推理代码或服务 API。

## 基准测试方法

- 硬件：NVIDIA GeForce RTX 4070 Ti SUPER，显存总量 16,376 MiB（约 16 GiB）。
- 工作流：同一 20 秒任务拆成五段，按顺序串行请求 8390；不重跑其他模型工作流。
- 9 GiB 轮次输入音频和五张参考图的 SHA-256 与 8 GiB 轮次一致。
- 时间：分别统计五次 HTTP 推理请求耗时之和，以及包含上传、合成和 QC 的完整
  runner 测量时间。
- GPU：`nvidia-smi` 整卡指标，1 秒采样；表中均值覆盖完整 runner 采样窗，峰值为
  设备总显存占用，包含运行前基线。
- 主机内存：`/proc/meminfo` 和 8390/worker 进程树，1 秒采样；最低
  `MemAvailable`、worker RSS/PSS 峰值和 swap 增量按五段推理时间窗统计。RSS 可能
  重复计算共享页，PSS 会分摊共享页。
- 生成结果：五段均成功；两个最终 MP4 均通过 FFmpeg 严格解码检查。完整运动画面和
  音频的人工复核仍需单独完成。

## 各档测试结果

时间列为“五次推理请求合计 / 完整 runner 测量时间”。内存容量均为 GiB；
`MemAvailable` 列括号中为相对该轮首段前基线的下降量，用于对照各轮不同的起始状态。

| 配置 | 请求合计 / 全程（秒） | GPU 均值 | 显存均值 / 峰值 | MemAvailable 最低（基线下降） | worker RSS / PSS 峰值 | 系统 swap 增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Profile 4 | 1361.1 / 1373.0 | 81.7% | 4.52 / 5.50 | 9.67（22.43） | 31.15 / 未记录 | 4.04 |
| Profile 0 默认 | 2024.8 / 2036.0 | 56.1% | 12.31 / 15.35 | 15.33（17.86） | 31.63 / 未记录 | 2.71 |
| Profile 0，manual 4096 MiB | 1349.7 / 1363.4 | 82.6% | 7.42 / 9.61 | 11.37（19.88） | 31.90 / 31.89 | 4.49 |
| Profile 0，manual 8192 MiB | 1340.9 / 1352.0 | 82.8% | 10.93 / 13.63 | 15.45（16.20） | 30.34 / 30.31 | 3.61 |
| **Profile 0，manual 9216 MiB（当前选定）** | **1329.6 / 1340.9** | **83.6%** | **10.95 / 13.96** | **17.16（15.52）** | **29.10 / 29.09** | **3.06** |
| Profile 0，manual 10240 MiB | 2207.4 / 2218.7 | 59.2% | 13.13 / 15.36 | 17.91（14.58） | 29.94 / 29.94 | 3.75 |

Profile 4 和 Profile 0 默认轮次的历史记录没有 PSS 数据。显存为整卡统计；9 GiB
轮次的显存峰值为 14,294 MiB（13.96 GiB），相对开始前 1,375 MiB 基线增加
12,919 MiB（12.62 GiB），峰值时约剩 2.03 GiB。

### 8、9、10 GiB 各段请求耗时

| 段 | manual 8192 MiB（秒） | manual 9216 MiB（秒） | manual 10240 MiB（秒） | 9 GiB GPU 均值 |
| --- | ---: | ---: | ---: | ---: |
| shot01 | 327.2 | 332.4 | 342.7 | 85.3% |
| shot02 | 340.5 | 337.8 | 1145.9 | 84.4% |
| shot03 | 195.9 | 186.5 | 215.0 | 85.0% |
| shot04 | 248.0 | 254.7 | 285.0 | 83.0% |
| shot05 | 229.4 | 218.1 | 218.7 | 82.9% |

## 结果解读与操作建议

- 9 GiB 相比 8 GiB 请求合计快 11.3 秒，GPU 均值高 0.8 个百分点；平均显存几乎
  相同，峰值多约 0.33 GiB。worker RSS/PSS 峰值低约 1.24/1.22 GiB，系统 swap
  增量低约 0.55 GiB。单轮差异不代表稳定的统计优势，但整体没有出现明显回退。
- 10 GiB 轮次第二段耗时 1145.9 秒、该段 GPU 均值 40.4%，整轮 GPU 均值降至
  59.2%；9 GiB 对应第二段为 337.8 秒、84.4%，未复现该长停顿。现有数据支持保留
  9 GiB，暂不继续提高到 10 GiB。
- 9 GiB 峰值仍保留约 2 GiB 显存余量。它把显存均值从 4 GiB 档的 7.42 GiB 提高到
  10.95 GiB，同时比 10 GiB 档避开了本次异常慢段。对于本机当前工作流，9 GiB 是
  显存利用率、运行时间和主机内存表现之间的已实测折中。
- Profile 0 默认和 Profile 4 的轮次总耗时分别为 2036.0 秒和 1373.0 秒；手动
  9216 MiB 轮次为 1340.9 秒、GPU 均值 83.6%。每档目前只有一次完整轮次，且模型
  seed 会变化，因此结论是基于实测选档，不是对所有工作负载的普遍保证。
- 在该硬件和该工作流下，测试结论曾选择 `video_preload_mode="manual"`、
  `video_preload_in_VRAM=9216`、Profile 0 和 SDPA。由于集成现已退役，此设置仅作历史
  复现参考，不构成本项目当前运行配置。

## 记录索引

9 GiB 轮次 ID：`20261102T114124-d909d4`。详细分段指标、GPU 样本、主机内存原始
采样和生成产物保存在本次 LongCat 对比测试工作目录的
`comparison/optimized-profile0-9216/` 与 `profile0-manual9216/` 下；本仓库只记录
可复用的配置与汇总数据，不纳入机器专属绝对路径、模型权重或生成媒体文件。
