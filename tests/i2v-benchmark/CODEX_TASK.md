# HunyuanVideo I2V 基准任务

目标：在 RTX 4070 Ti SUPER 16GB + WSL Ubuntu 上复现并维护 HunyuanVideo 1.5 480p I2V 基线。

1. 使用项目独立的 `uv` 环境；不覆盖已有 ComfyUI，也不安装全局 Python 包。
2. 检查 FFmpeg/ffprobe、`nvidia-smi`、WSL CUDA，以及 ComfyUI `/object_info` 和 `/system_stats`。
3. 从已准备的本地模型目录加载 Step-Distilled FP8 主权重及其必需文本编码器、视觉编码器和 VAE；请求时不得下载权重。
4. 用官方 UI 模板生成 API 工作流，检查主模型、12 步、CFG=1、flow shift=7 和图片输入节点。
5. 预检通过后，用 3 张真实竖屏 PNG、固定提示词和 seed 42、1234、2026 运行 9 条记录，验证输出尺寸、帧率、帧数和 MP4。
6. 保存 ComfyUI 版本、权重文件名与字节数、运行时间、全卡显存峰值、WSL 内存和失败信息。
7. 盲评表仅用于人工记录画质；没有完成评分时保持“待测”，不推断画质分数。
