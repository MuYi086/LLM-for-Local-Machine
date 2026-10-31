# 音源分离能力缺口与执行状态

- 日期：2026-10-21
- 范围：当前保留的分离服务
- 依据：现行请求/worker 合同及固定 audio.cpp revision 9a02e61326aaaf9d462b584ca5e0daba22c0abfc
- 结论：一个 one-shot worker bridge 的代码和无模型合同检查已完成；不代表真实模型 E2/E3 已验收。

## P5：音源分离

| 服务/能力 | 当前路径 | 保留的行为 | 模型验收 |
| --- | --- | --- | --- |
| TIGER-DnR | llm-local.tiger-dnr.worker.v1 bridge | 随机文件引用传输三 Stem ZIP；服务校验目录、大小、MIME 和固定文件名，再原子保存批次 | 仍需用本地权重验证同步输出、长音频、资源上限、shutdown 与异常清理 |

缺少镜像模型或必需辅助权重的音色转换服务已按镜像清理规则退役，不再保留待下载的转换候选。

## 共同边界

worker bridge 使用固定 LocalAI model ID 调用各服务环境的一次性 worker。模型由用户维护在本地镜像中；
本次未下载模型或安装 Python 依赖。无模型测试不代替真实推理、显存、shutdown 和异常恢复验收。