# 视频模型

当前维护三个独立视频服务：SoulX-FlashHead 人像视频生成、HunyuanVideo-1.5 Comfy 图生视频和
Wan2.2 TI2V GGUF 文生/图生视频。
服务由根目录 `start.sh` 统一启动，各自使用独立 uv 项目和一次性 worker。

| 服务 | 端口 | 主路由 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| SoulX-FlashHead | 8391 | `POST /v1/soulX/flashHead` | 人像图片与驱动音频 | 带音轨 MP4 |
| HunyuanVideo-1.5 Comfy | 8393 | `POST /v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy` | 参考图片与文本提示词 | 848×480、24 FPS MP4 |
| Wan2.2 TI2V 5B GGUF | 8392 | `POST /v1/QuantStack/Wan2.2-TI2V-5B-GGUF` | 文本提示词与可选参考图片 | MP4 视频 |

## SoulX-FlashHead

主模型默认位于 `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B/`，音频编码器位于
`$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h/`，官方源码位于
`$HOME/LLM-depency/SoulX-FlashHead/`。完整依赖、字段和调用示例见
[SoulX-FlashHead 服务说明](avatar_soulx-flashhead-1.3b/README.md)。

## HunyuanVideo-1.5 Comfy

模型快照默认位于 `$HF_MIRROR_DIR/HunyuanVideo-1.5-Comfy/`，ComfyUI 源码默认位于
`$HOME/LLM-depency/ComfyUI/`。图生视频模型与 Qwen、ByT5、SigCLIP 和 VAE 辅助权重都从本地
快照加载；服务不会下载权重。请求期间启动 loopback ComfyUI API，worker 退出时一并回收。

详细依赖、API 字段和安装步骤见
[HunyuanVideo-1.5 Comfy 服务说明](HunyuanVideo-1.5-Comfy/README.md)。

## Wan2.2 TI2V 5B GGUF

模型权重默认位于 `$HF_MIRROR_DIR/QuantStack/Wan2.2-TI2V-5B-GGUF/`，ComfyUI 与 ComfyUI-GGUF
源码分别位于 `$HOME/LLM-depency/ComfyUI/` 和 `$HOME/LLM-depency/ComfyUI-GGUF/`。服务每个请求
启动一次性 ComfyUI，并只加载 GGUF 自定义节点。详细依赖、API 字段和显存设置见
[Wan2.2 TI2V 服务说明](Wan2.2-TI2V-5B-GGUF/README.md)。
