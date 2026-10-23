# 视频模型

当前维护 `avatar_soulx-flashhead-1.3b/`，提供 SoulX-FlashHead 图片与音频驱动的人像视频生成。
默认端口 `8391`，业务接口 `POST /v1/soulX/flashHead`，由根目录 `start.sh` 统一启动。

## 本地依赖与运行

| 组件 | 默认位置 |
| --- | --- |
| Lite/Pro 主模型与对应 VAE | `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B/` |
| facebook/wav2vec2-base-960h 音频编码器 | `$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h/` |
| 官方源码 | `$HOME/LLM-depency/SoulX-FlashHead/` |

`HF_MIRROR_DIR` 默认 `~/hf-mirror`。模型和音频编码器均在仓库外，服务不自动下载。
HTTP 进程只做上传、校验和生命周期管理；模型使用独立 uv 环境，在每个请求的一次性 worker 中加载。

目录使用 `功能词_模型名-版本`，`avatar` 表示数字人或说话头像生成。上传素材、生成 MP4、
虚拟环境和缓存属于本机运行数据，不随 Git 提交。

完整的依赖、安装、接口字段、配置与测试说明见 [SoulX-FlashHead 服务文档](avatar_soulx-flashhead-1.3b/README.md)。
