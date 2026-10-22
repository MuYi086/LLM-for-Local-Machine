# 文本模型与处理

当前没有独立的文本模型 HTTP 推理服务。`model-text/scripts/training/` 与 `model-text/training/configs/` 集中放置
有声书文本润色的 LoRA 实验工具和配置；该实验处理文本，不是语音训练或生产服务。

运行实验脚本前先查看必填的来源目录和输出目录参数：

```bash
PYTHONPATH=model-text uv run --project qa --locked python -m scripts.training.build_audio_polisher_poc --help
```
