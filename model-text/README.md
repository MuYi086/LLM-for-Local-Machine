# 文本模型与处理

本目录当前不提供对外 HTTP 文本模型 API。此前维护的 Ornith-1.5-9B-MTP 服务及其 llama.cpp 共享运行时已下线，
本机不再使用 Ornith-1.5 模型；对话需求由 Easel 自身的执行引擎承担。

`scripts/training/` 与 `training/configs/` 保留纯文本 LoRA 实验；它们不是 HTTP 服务，
也不自动取得服务 GPU 锁，执行真实训练或预测需单独安排 GPU。
