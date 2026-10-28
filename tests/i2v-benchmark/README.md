# HunyuanVideo 1.5 I2V 本地基准

此目录保留 HunyuanVideo 1.5 480p I2V 的本地运行脚本、真实结果和分级时长验证。正式基准使用
3 个场景 × 3 个 seed，共 9 次；固定输入图片、英文提示词、480×832、24 FPS、121 帧，使用
Step-Distilled FP8、12 步、CFG=1、flow shift=7。

## 环境与权重

WSL Ubuntu、Python 3.10+、`uv`、ComfyUI（支持 HunyuanVideo 1.5）、FFmpeg/ffprobe 和 `nvidia-smi`。
ComfyUI 通过内置节点运行，不需要额外的 GGUF 节点。

需要在 ComfyUI 模型目录准备以下文件：

- `models/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors`
- `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `models/text_encoders/byt5_small_glyphxl_fp16.safetensors`
- `models/clip_vision/sigclip_vision_patch14_384.safetensors`
- `models/vae/hunyuanvideo15_vae_fp16.safetensors`

模型来源：[Hugging Face](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged)、
[ModelScope](https://modelscope.cn/models/Comfy-Org/HunyuanVideo_1.5_repackaged)。

## 运行命令

先启动 ComfyUI 并确认 `http://127.0.0.1:8188` 可访问，然后执行：

```bash
cd tests/i2v-benchmark
uv sync
uv run i2vbench fetch-templates
uv run i2vbench prepare
uv run i2vbench preflight
uv run i2vbench run --limit 1
uv run i2vbench run
uv run i2vbench review-sheet
uv run i2vbench report
```

`prepare` 根据当前 ComfyUI `/object_info` 编译官方 UI 工作流。若节点结构变化，先检查转换器和实时
节点 schema；`preflight` 通过后再推理。正式输入图片放在 `assets/rainy_city.png`、
`assets/windy_portrait.png`、`assets/calm_mountains.png`；`make-demo-images` 只用于流程检查。

## 输出和当前记录

- `results/runs.csv`：单条试验状态、耗时、显存、内存与视频规格。
- `results/videos/`：生成的原始视频。
- `results/raw/` 和 `results/gpu/`：ComfyUI 请求记录及采样监控。
- `results/tiers/`：49、81、121 帧的 Hunyuan 分级验证。
- `results/report.md`：根据真实运行记录生成的单模型基线。
- `results/benchmark_notes.md`：运行环境、权重指纹、口径与复现限制。

报告只记录性能和人工画质评分，不计算跨模型排名。GPU 峰值来自 `nvidia-smi` 全卡占用；壁钟时间
包含上传、排队、推理、解码和拉取视频。雨夜、人物和风景三类输入仅用于本地业务观察，不是公共
标准评测集。
