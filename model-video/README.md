# 视频模型

当前维护两个独立视频服务：SoulX-FlashHead 人像视频生成和 MiniMax H3 图生视频。
服务由根目录 `start.sh` 统一启动，各自使用独立 uv 项目和一次性 worker。

| 服务 | 端口 | 主路由 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| SoulX-FlashHead | 8391 | `POST /v1/soulX/flashHead` | 人像图片与驱动音频 | 带音轨 MP4 |
| MiniMax H3 16G HiFi | 8392 | `POST /v1/minimax/H3-16G-HiFi` | FL2VA 首尾帧或 Ref2VA 参考图及文本 | 默认无音轨 MP4；`include_audio=true` 时带原生音轨 |

## SoulX-FlashHead

主模型默认位于 `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B/`，音频编码器位于
`$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h/`，官方源码位于
`$HOME/LLM-depency/SoulX-FlashHead/`。完整依赖、字段和调用示例见
[SoulX-FlashHead 服务说明](avatar_soulx-flashhead-1.3b/README.md)。

## MiniMax H3 16G HiFi

模型默认位于 `$HF_MIRROR_DIR/minimax/MiniMax-H3-16G-HiFi/`。请求的 `include_audio` 默认 `false`，无需音频 VAE 权重即可生成静音视频；设为 `true` 会启用 H3 原生音轨。服务调用仓库外的 ComfyUI 与模型随附
`ComfyUI-sol-attn-main.zip`，通过一次性 worker 启动本地 ComfyUI；默认限制为 960×544、5 秒输入时长、20 步，
并要求提交前至少有 13,000 MiB 空闲显存。模型、ComfyUI 版本/环境和 API 用法见
[MiniMax H3 服务说明](MiniMax-H3-16G-HiFi/README.md)。
