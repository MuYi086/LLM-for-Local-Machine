# P5/P6 能力缺口与执行状态

- 日期：2026-10-21
- 范围：当前保留的分离服务与结构化音乐任务
- 依据：现行请求/worker 合同及固定 audio.cpp revision 9a02e61326aaaf9d462b584ca5e0daba22c0abfc
- 结论：五个 one-shot worker bridge 的代码和无模型合同检查已完成；不代表真实模型 E2/E3 已验收。

## P5：音源分离

| 服务/能力 | 当前路径 | 保留的行为 | 模型验收 |
| --- | --- | --- | --- |
| TIGER-DnR | llm-local.tiger-dnr.worker.v1 bridge | 随机文件引用传输三 Stem ZIP；服务校验目录、大小、MIME 和固定文件名，再原子保存批次 | 仍需用本地权重验证同步输出、长音频、资源上限、shutdown 与异常清理 |

缺少镜像模型或必需辅助权重的音色转换服务已按镜像清理规则退役，不再保留待下载的转换候选。

## P6：结构化音乐任务

| 服务/能力 | 当前路径 | 保留的合同 | 模型验收 |
| --- | --- | --- | --- |
| SheetSage2 转谱 | llm-local.sheetsage2.worker.v1 bridge | ZIP、可用时包含 ABC、MIDI、事件和窗口输出；依赖配置固定 revision 的 MERT 父模型 | 仍需验证父模型 revision、ZIP 内容与结果语义 |
| YuE2-Vae latent encode/decode | llm-local.yue2-vae.worker.v1 bridge | [T,64] NPY、FP32 latent 与官方分块 WAV decode | 仍需验证 48 kHz 输入、shape/dtype 与 decode |
| MERT-v2-FullSong 表征 | llm-local.mert-v2-fullsong.worker.v1 bridge | 1024 维整曲 JSON、25 Hz 逐帧 NPZ；预检 30–360 秒/24 kHz mono | 仍需验证 masked mean、时间轴与长音频边界 |
| YuE2-3B 歌曲生成 | llm-local.yue2-3b.worker.v1 bridge | 原生请求参数和 WAV 输出 | 仍需验证完整模型调用与输出质量 |

## 共同边界

worker bridge 使用固定 LocalAI model ID 调用各服务环境的一次性 worker。模型由用户维护在本地镜像中；
本次未下载模型或安装 Python 依赖。无模型测试不代替真实推理、显存、shutdown 和异常恢复验收。