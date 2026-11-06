# stable-diffusion.cpp 模型推理本机显存筛选

更新：2026-11-05。筛选基于 [stable-diffusion.cpp 当前 README 的 Supported models](https://github.com/leejet/stable-diffusion.cpp#features) 及各模型说明/权重目录。README 有 33 个图像生成支持链接，按不同具体 checkpoint/变体拆成下表 42 行；另有 7 个图像编辑支持链接、5 个视频支持链接，以及 README 列出的附加模型能力。此处只把能通过量化、CPU 参数卸载/分段执行，在本机显存预算内推理的条目列为符合；高分辨率、多帧、并发模型会改变峰值。

## 结论与机器基线

| 项目 | 实测/采用值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，`nvidia-smi` 报告 16,376 MiB（约 16.0 GiB）总显存 |
| 当前 WSL GPU 占用 | 4,865 MiB，进程为 Xwayland；空闲 11,511 MiB（约 11.24 GiB）。这是 2026-11-05 的一次快照，会随桌面/驱动状态变化 |
| CPU / 主机内存 | Core i5-13600KF；用户提供主机内存 48 GB。当前 WSL 可见 36.9 GB，`MemAvailable` 33.7 GB |
| 常规单次推理预算 | 进程管理显存上限 9 GiB；需较大计算图的条目最多用 10 GiB。按目前 Xwayland 占用，系统合计约 13.8–14.8 GiB，给 16 GiB 卡保留约 1.2–2.2 GiB |
| 并发假设 | 一次只推理一个模型，不与其他 GPU 模型同时运行；图片 batch=1。视频从 17–33 帧、约 480p 起步 |

**符合的模型：**下表列出的图像生成、图像编辑和视频模型均可在上述部署预算内运行，条件是采用所列量化并按需将文本编码器/大权重放 CPU；大模型需 stable-diffusion.cpp 的分段计算。大模型“能运行”不等于适合高分辨率或高帧数长视频，表内条件属于推荐的起步设置。附加能力依附于表中的基础模型，单独不能生成。

## 显存口径和限制

本文把“推理峰值”按**实际部署时单卡总量**理解：模型在 GPU 上的量化权重/当前计算段 + VAE/条件模块驻留部分 + 激活与计算工作区。不能只用权重文件大小代替推理显存。表中范围是按权重规模、上游运行示例和推理工作区估算的配置峰值，不是本机逐模型实测值；本轮没有启动逐模型推理基准，因此不把估算值写成实测结果。

官方性能文档说明：计算缓冲区日志不含权重和缓存；Flash Attention 在示例中可为 Flux 768×768 减约 600 MB、SD2 768×768 减约 1,400 MB；`--offload-to-cpu` 将参数留在主机内存并按需搬入 GPU，图切分可在权重和完整工作区不能同时驻留时分段执行；`--max-vram` 可限制进程的显存预算，但不覆盖驱动上下文及其他程序占用。[性能与显存说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/performance.md)

建议 CUDA 推理从以下参数起步，模型文档规定的 `--diffusion-model`、编码器和 VAE 参数另行补齐：

```bash
--backend cuda0 --offload-to-cpu --max-vram 9 --flash-attn on
```

显存紧张时把 `--max-vram` 调到 8；复杂模型/常规高分辨率最多尝试 10，并先确认 `nvidia-smi` 空闲显存至少 11 GiB。VAE 可用 `--vae-tiling`；超大模型可将扩散参数改为 disk backend，但会增加读取延迟。CPU offload 需要主机内存，disk offload 会显著变慢。[后端放置与分段执行](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/backend.md)；[量化格式](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)

> 表中“模型峰值”是建议部署上限下的完整推理显存，不是 GGUF 文件体积。量化文件体积只用于核对模型权重是否匹配，并帮助估计 CPU 内存需求。所有值按 GiB 近似，分辨率和视频帧数增加时需要重新测量。

## 高精度优先：Q8、INT8、FP8 与 F16

**Q4 不代表画质一定会明显变差。**它主要减少权重的表示精度；扩散模型的可见变化因架构、量化方法、提示词和采样设置而异。Q4 有时与高精度结果很接近，有时细节、文字或提示词遵循会有差别。通常 Q8 的量化误差更小，更接近原始 F16/BF16，但没有对本表每个模型做同提示词的 A/B 实测，因此不承诺每张图都能看出提升。Q8 权重通常接近 Q4 的两倍体积，速度取决于后端和模型；CPU/磁盘卸载会进一步增加延迟。[上游量化格式说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)

建议先把高精度用在**扩散主模型**。文本编码器可继续放 CPU，按需要使用 Q4/Q5；它与把扩散主模型量化到 Q4 是两件事。下表的 GB 是模型文件大小，未包含 VAE、文本编码器和工作区，也不等于 GPU 显存占用。

### 图像：优先试这些 Q8 / F16

| 推荐模型 | 主模型权重 | 本机运行建议 |
| --- | --- | --- |
| FLUX.2 Klein 4B | [Q8_0，4.3 GB](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/blob/main/flux-2-klein-4b-Q8_0.gguf) | 体积和画质折中最好入门；文本编码器放 CPU，单图 1024 级起步。 |
| MOSS-Image-Base | [Q8_0，5.71 GB](https://huggingface.co/leejet/MOSS-Image-Base-GGUF/tree/main) | 需匹配 FLUX.2 VAE；Qwen3.5 编码器放 CPU。 |
| FLUX.3 Image Klein | [Q8_0，6.15 GB](https://huggingface.co/leejet/flux-3-image-klein-GGUF/tree/main) | Qwen3-VL 编码器放 CPU；单图起步，编辑时减少参考图数量。 |
| Z-Image Turbo | [Q8_0，6.58 GB](https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main) | Qwen 编码器放 CPU；保留 Turbo 对应采样设置。 |
| Qwen Image 2.1 | [Q8_0，7.69 GB](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/main) | 更适合中文提示和画面文字；Qwen3-VL 8B 放 CPU，先用单张、1024 级和 VAE tiling。 |
| SD 1.5 / SD 2.1 / SDXL | [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)、[SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)、[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) F16 权重 | 可直接保留 F16，不必量化；SD 1.5/2.1 显存余量充足，SDXL 建议文本编码器放 CPU。 |

这几种 Q8 主权重文件本身都不超过约 8 GB，配合 9 GiB 显存预算、文本编码器 CPU 参数放置和 batch=1，较适合作为本机高精度起点。遇到显存不足时可开启分段计算或提高 `--max-vram` 至 10 GiB（前提是运行前至少有 11 GiB 空闲显存）。体积更大的原始 Qwen Image Q8_0 约 21.8 GB、FLUX.1-dev Q8_0 约 12.8 GB，也有权重，但要 CPU 参数卸载；原始 Qwen Image 还可能需要磁盘卸载，速度会低很多。

### 视频：可跑的更高精度配置

| 推荐模型 | 主模型权重 | 本机运行建议 |
| --- | --- | --- |
| Wan2.2 TI2V 5B | [Q8_0，5.4 GB](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main) | **优先推荐。**文本编码器放 CPU，从 480p、17 帧、单任务开始；权重目录还包含其他量化等级。[sd.cpp Wan 说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md) |
| HunyuanVideo 1.5 480p T2V / I2V | [distilled FP8 scaled，约 8.33 GB](https://huggingface.co/Comfy-Org/Hunyuan-Video-1.5/tree/main/split_files/diffusion_models) | 选择对应 480p 的 distilled FP8 文件；Qwen2.5-VL 与 ByT5 放 CPU，低帧数起步。默认 patched GGML 构建优先；`SD_USE_UPSTREAM_GGML=ON` 会将 FP8 safetensors 在加载时转成 F16，运行内存约翻倍。 |
| Wan2.1 T2V 14B | [Q8_0，约 15.9 GB](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/tree/main) | 可尝试 CPU 参数卸载；从 480p、17 帧开始。若 WSL 主机可用内存不足，再用 disk 参数后端，读取延迟会明显增加。 |
| LTX-2.5 22B Dev | [Q8_0，约 23.6 GB](https://huggingface.co/vantagewithai/LTX-2.5-GGUF/tree/main/dev) | 属于慢速尝试项：扩散参数放 disk、Gemma 编码器放 CPU，先用低分辨率和短视频。主模型文件本身已接近本机 WSL 可用内存上限。 |
| MiniMax-H3 | [pruned INT8 convrot，约 21 GB](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) | 可替换原表的 Q4 主模型；Qwen32B 编码器放 CPU 并选小量化，扩散参数按需放 disk，帧数从 17 起步。须使用默认 patched GGML 构建；`SD_USE_UPSTREAM_GGML=ON` 会拒绝此 INT8 文件。 |

FP8/INT8 文件也要看运行时如何加载：`SD_USE_UPSTREAM_GGML=ON` 的兼容构建会把 FP8 safetensors 展开成 F16；INT8 Convrot 则要求默认 patched GGML 扩展并以 INT8 直接执行。前者会让运行内存约翻倍，后者在 upstream GGML 构建中会被拒绝。[构建与量化格式兼容性](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)；[INT8 Convrot 说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/int8_convrot.md)。`--params-backend cpu` 将参数放主机内存；`diffusion=disk,te=cpu,vae=cpu` 可降低主机内存占用，但被换出的权重需要从文件重读，速度会更慢。[CPU/disk 参数卸载说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/performance.md)

大型主模型可按需试这组参数放置选项（模型本身的权重、提示词和视频参数仍按对应文档提供）：

```bash
--backend cuda0 --params-backend diffusion=disk,te=cpu,vae=cpu --max-vram 9 --flash-attn on
```

## 符合条件的图像生成模型

| 上游模型 | 推理权重/推荐精度 | 输入、输出与用途 | 估算推理峰值（单卡）及条件 |
| --- | --- | --- | --- |
| SD v1.4 | [CompVis 原始权重](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)；F16 | 英文文本生成或图生图，512 级插画/概念图 | **2.5–4 GiB**；原生 F16，通常无需卸载 |
| SD v1.5 / v2.1 | [v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)、[v2.1](https://huggingface.co/Manojb/stable-diffusion-2-1-base)；F16 | 文生图、图生图；v1.5 生态微调/LoRA 多，v2.1 常见 768 尺寸 | **3–5 GiB**；F16 即可，叠加 ControlNet 时走 CPU offload |
| SD-Turbo | [Stability AI 权重](https://huggingface.co/stabilityai/sd-turbo)；F16 | 少步数文生图快速预览 | **3–5 GiB**；遵循 Turbo 的低步数设置 |
| SDXL / SDXL-Turbo | [Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)、[Turbo](https://huggingface.co/stabilityai/sdxl-turbo)；F16 或 Q8_0，VAE 可用 CPU/tiling | 文生图/图生图，常用 1024×1024；Turbo 快速预览 | **6–9 GiB**；VAE tiling；叠加条件模块时保留 CPU offload |
| SSD-1B、Vega、tiny SD1/SD2 | [SSD-1B](https://huggingface.co/segmind/SSD-1B)、[Vega](https://huggingface.co/segmind/Segmind-Vega)、[tiny SD](https://huggingface.co/segmind/tiny-sd)、[BK-SDM tiny](https://huggingface.co/nota-ai/bk-sdm-tiny)、[SD2 tiny](https://huggingface.co/nota-ai/bk-sdm-v2-tiny)；F16 | 轻量文生图和草图；权重格式依模型而异 | **3–6 GiB**；部分 Diffusers 权重需按[蒸馏模型说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/distilled_sd.md)转换；LoRA 不可当独立底模 |
| SDXS-512-DreamShaper | [官方权重](https://huggingface.co/akleine/sdxs-512)、[GGUF](https://huggingface.co/concedo/sdxs-512-tinySDdistilled-GGUF)；F16/GGUF | 一步 512 文生图 | **3–5 GiB**；使用 `--steps 1 --cfg-scale 1` |
| SDXS-512-0.9 | [官方权重](https://huggingface.co/akleine/sdxs-09)；F16 | 一步 512 文生图 | **3–5 GiB**；使用 `--steps 1 --cfg-scale 1` |
| Stable Diffusion 3 Medium | [官方权重](https://huggingface.co/stabilityai/stable-diffusion-3-medium)；扩散权重 Q4_0/Q8_0，T5-XXL 放 CPU | 多文本编码器文生图，常见 1024 尺寸 | **6–9 GiB**；配齐 CLIP-L/G、T5-XXL 和 VAE；文本编码器 CPU offload |
| Stable Diffusion 3.5 Large | [主权重](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)、[配套文本编码器](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/tree/main/text_encoders)；扩散权重 Q4_0，文本编码器 CPU | 高质量 1024 文生图 | **7–9 GiB**；主权重需量化，文本编码器 CPU offload |
| FLUX.1-dev | [官方 GGUF](https://huggingface.co/leejet/FLUX.1-dev-gguf)；Q4_0 主权重约 6.4 GB，Q8_0 约 12.1 GB | 质量优先的文本生成图像 | **7–9 GiB**；CLIP-L、T5-XXL 放 CPU；上游权重占用数据见 [Flux 指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md) |
| FLUX.1-schnell | [官方 GGUF](https://huggingface.co/leejet/FLUX.1-schnell-gguf)；Q4_0 | 少步数快速文生图 | **6–9 GiB**；CLIP-L、T5-XXL 放 CPU，使用 Schnell 采样配置 |
| FLUX.2-dev | [GGUF 权重](https://huggingface.co/city96/FLUX.2-dev-gguf)：Q2_K 12.9 GB；Q4_0 已约 19.3 GB。Mistral Small 3.2 24B 另配 Q4 GGUF | 文生图及参考图编辑 | **8–10 GiB**；只建议 Q2_K 主权重，Mistral 编码器放 CPU；1024 以下起步。Q4 主权重文件已大于显存，不能全 GPU 常驻；需 `--offload-to-cpu`，主机权重驻留约 25–30 GB。[运行说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md) |
| FLUX.2-klein 4B / 4B Base | [蒸馏版 GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF)、[Base GGUF](https://huggingface.co/leejet/FLUX.2-klein-base-4B-GGUF)；Q4_0 文件各约 2.46 GB | 文生图、参考图编辑；Base 与蒸馏版采样设置不同 | **4–7 GiB**；Qwen3 编码器可 CPU offload；4B Q4 有较多余量 |
| FLUX.2-klein 9B / 9B Base | [蒸馏版 GGUF](https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF)、[Base GGUF](https://huggingface.co/leejet/FLUX.2-klein-base-9B-GGUF)；Q4_0 文件约 5.62 GB | 文生图、参考图编辑 | **6–9 GiB**；文本编码器 CPU offload，1024 级起步 |
| FLUX.3 Image Klein / Klein Base | [Klein GGUF](https://huggingface.co/leejet/flux-3-image-klein-GGUF)、[Base GGUF](https://huggingface.co/leejet/flux-3-image-klein-base-GGUF)；Klein Q4_K 约 3.51 GB，Base 用 Q4 | 文生图和最多多张参考图编辑 | **5–9 GiB**；Qwen3-VL 4B 编码器放 CPU；Klein 与 Base 的步数/CFG 不同。[模型说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md) |
| MOSS-Image-Base / SFT | [Base GGUF](https://huggingface.co/leejet/MOSS-Image-Base-GGUF)、[SFT GGUF](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF)；Base Q4_K 约 3.33 GB，SFT 选 Q4；Qwen3.5-9B 编码器 CPU | 1024 文生图；Base/SFT 分别偏基础生成/指令跟随 | **5–8 GiB**；FLUX.2 VAE 与 tokenizer 需匹配。[上游组件说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/moss_image.md) |
| Fibo 2.0 Dev | [官方模型/Comfy 权重](https://huggingface.co/Comfy-Org/Fibo-2.0/tree/main/diffusion_models)；扩散权重转换 Q4_0，Qwen3-VL 4B 用 Q4 GGUF 并放 CPU | 文生图、多参考图编辑 | **6–9 GiB**；配标准 FLUX.2 VAE；使用官方 Fibo preset |
| Lens / Lens Turbo | [Comfy 权重](https://huggingface.co/Comfy-Org/Lens/tree/main/diffusion_models)；扩散权重 Q4_0；[GPT-OSS-20B Q4 GGUF](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) 放 CPU | Lens 细节生成，Turbo 少步快速生成 | **6–9 GiB**；文本编码器是主要主机内存负担，需 CPU offload |
| Chroma | [模型权重](https://huggingface.co/lodestones/Chroma)、[GGUF](https://huggingface.co/silveroxides/Chroma-GGUF)；Q4/Q8，T5-XXL CPU | Flux 系文生图 | **4–7 GiB**；需 FLUX VAE；上游文档称低显存卡也可运行。[指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/chroma.md) |
| Chroma1-Radiance | [模型权重](https://huggingface.co/lodestones/Chroma1-Radiance)、[GGUF](https://huggingface.co/silveroxides/Chroma1-Radiance-GGUF)；Q4，T5-XXL CPU | Chroma 系文生图 | **5–8 GiB**；需匹配 VAE/文本编码器 |
| Qwen Image | [GGUF 权重](https://huggingface.co/QuantStack/Qwen-Image-GGUF)；Q3_K_S 约 8.95 GB，Q4_0 约 11.9 GB；Qwen2.5-VL 7B 编码器 CPU | 中文/英文提示、画面文字生成 | **8–10 GiB**；建议 Q3_K_S 或 Q4_0 + CPU 参数卸载，VAE tiling；1024 级单图。Q4_K_M 文件为 13.1 GB，不可当作完整推理显存。[模型说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image.md) |
| Qwen Image 2.1 | [GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF)：Q4_K 约 4.2 GB；Qwen3-VL-8B Q4 放 CPU | 文生图、参考图编辑、透明背景 PNG/WebP | **5–8 GiB**；配套 2.1 VAE；尺寸取 32 倍数。[使用说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md) |
| PiD / PiD 1.5 | [PiD/PixelDiT 权重](https://huggingface.co/Comfy-Org/PixelDiT/tree/main/diffusion_models)、[NVIDIA PiD checkpoints](https://huggingface.co/nvidia/PiD/tree/main/checkpoints)；扩散权重 Q4_0，Gemma 2 2B 编码器 CPU | 参考图条件的像素空间解码/放大，不是纯文生图 | **6–9 GiB**；高分辨率像素解码需 tiled VAE/较小输出起步 |
| LongCat Image | [Comfy 权重](https://huggingface.co/Comfy-Org/LongCat-Image/tree/main/diffusion_models)、[GGUF](https://huggingface.co/vantagewithai/LongCat-Image-GGUF/tree/main/comfy)；Q4，Qwen2.5-VL CPU | 文生图 | **6–9 GiB**；FLUX VAE 与视觉编码器需配齐 |
| Z-Image / Z-Image-Turbo | [Turbo GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF)：Q4_K 3.86 GB；[Base GGUF](https://huggingface.co/unsloth/Z-Image-GGUF) 用 Q4；Qwen3 编码器 CPU | 文生图；Turbo 快速生成 | **4–8 GiB**；上游说明可在 4 GB VRAM 运行；Base/Turbo 分别遵循自身采样参数。[指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md) |
| Z-Image L2P | [L2P 模型卡/权重](https://huggingface.co/zhen-nan/L2P)；Q4_0，文本编码器 CPU | L2P 变体文生图 | **5–9 GiB**；按上游 L2P tokenizer 和参数加载。[指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image_l2p.md) |
| MiniT2I | [MiniT2I 主权重](https://huggingface.co/MiniT2I/MiniT2I/tree/main/minit2i-b-16/transformer) F16；[Flan-T5-Large](https://huggingface.co/google/flan-t5-large) CPU | 文生图 | **3–6 GiB**；体量较小，主模型无需量化 |
| P-H3-Image | [扩散权重、VAE 与 Qwen3-VL-4B connector](https://huggingface.co/PrunaAI/P-H3-Image-Comfy/tree/main)；扩散 Q4，编码器 CPU | 文生图及参考图编辑 | **6–9 GiB**；不包含未接入的 prompt enhancement |
| SenseNova U1.5 8B MoT | [官方模型](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)；Q4_0，整套仓库含 tokenizer | 2048×2048 RGB 像素空间文生图，不需单独文本编码器/VAE | **8–10 GiB**；2048 原生画幅属高占用边界，使用 10 GiB 档并避免其他 GPU 程序；调低尺寸会离开训练分辨率桶。[上游说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sensenova_u1.md) |
| Ovis-Image-7B | [GGUF](https://huggingface.co/leejet/Ovis-Image-7B-GGUF) Q4_0；[Ovis 2.5 编码器](https://huggingface.co/Comfy-Org/Ovis-Image/tree/main/split_files/text_encoders) CPU | 文生图 | **6–9 GiB**；VAE 和文本编码器需配齐 |
| Anima / Anima2 | [基础权重](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)、[Anima GGUF](https://huggingface.co/Bedovyy/Anima-GGUF/tree/main)；Q4_K_M 示例文件约 1.42 GB | 文生图；也可加载兼容的 Anima edit LoRA | **4–7 GiB**；Qwen3-0.6B 编码器，遵守模型许可 |
| ERNIE-Image / Turbo | [普通版 GGUF](https://huggingface.co/unsloth/ERNIE-Image-GGUF/tree/main)、[Turbo GGUF](https://huggingface.co/unsloth/ERNIE-Image-Turbo-GGUF/tree/main)；Q4_K_M 普通版约 5.02 GB | 普通版质量生成、Turbo 快速生成 | **4–7 GiB**；官方集成说明称 4 GB 或更低显存可运行；配套 VAE。[指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/ernie_image.md) |
| Boogu Image Base | [Comfy 扩散权重](https://huggingface.co/Comfy-Org/Boogu-Image/tree/main/diffusion_models)；主模型 Q4，Qwen3-VL-8B Q4 CPU | 文生图及 `z_image_omni` 参考图编辑 | **6–9 GiB**；Qwen3-VL 与 FLUX VAE 属完整推理组件 |
| Krea2 Raw / Turbo | [Raw](https://huggingface.co/krea/Krea-2-Raw)、[Turbo](https://huggingface.co/krea/Krea-2-Turbo)、[GGUF](https://huggingface.co/realrebelai/KREA-2_GGUFs/tree/main)；Q4，Qwen3-VL-4B CPU | 文生图；社区编辑 LoRA 可用于编辑 | **6–9 GiB**；配 Wan2.1 VAE，Turbo 与 Raw 的采样策略不同 |
| Mage-Flow Base / Turbo | [扩散权重/VAE](https://huggingface.co/Comfy-Org/Mage-Flow/tree/main)；4B 主模型 Q4，Qwen3-VL-4B Q4 CPU | 原生分辨率文生图和指令式图像编辑 | **5–8 GiB**；需 128-channel Mage VAE；编辑走官方 preset |
| SeFi-Image 1B/2B/5B（Base/RL/Turbo） | [Hugging Face 模型集合](https://huggingface.co/SeFi-Image)；1B/2B 选 Q4 + Qwen3-VL-2B，5B 选 Q4 + Qwen3-VL-4B；编码器 CPU | 多尺度文生图；Turbo 快速生成 | **4–9 GiB**；Hugging Face 权重受访问门控，需先获准下载 |
| HiDream-O1-Image / Dev | [官方 Comfy checkpoints](https://huggingface.co/Comfy-Org/HiDream-O1-Image/tree/main/checkpoints)；主模型 Q4，文本编码器 Q4/Q8 放 CPU | 高质量文生图/编辑 | **8–10 GiB**；多个文本编码器不可漏配，启用 offload 与 VAE tiling |
| Ideogram4 | [FP8 transformer 与 unconditional 权重](https://huggingface.co/ideogram-ai/ideogram-4-fp8/tree/main)；扩散权重转 Q4_0，Qwen3-VL-8B Q4 CPU | 高质量文生图、画面文字 | **8–10 GiB**；配 FLUX.2 VAE，1024 级起步；FP8 文件不可全部驻留当前可用显存 |
| LLaDA-Image Base / Turbo | [GGUF 转换集合与 connectors](https://huggingface.co/fszontagh/LLaDA-Image-GGUF)、[官方模型](https://huggingface.co/inclusionAI/LLaDA-Image)；主扩散模型 Q4，文本编码器 Q8_0/Q4 CPU | 文生图；编辑版还需 edit connectors 和 SigVQ encoder | **5–9 GiB**；Base/Turbo 分别配正确 connectors。[指南](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/llada_image.md) |
| Ming-Image 0.1 Design | [DiT/VAE/编码器组件](https://huggingface.co/Comfy-Org/Ming-Image/tree/main)；6B DiT Q4，Ling-mini-2.0 编码器 CPU | 文生图，支持 RGBA 输出 | **5–8 GiB**；完整 tokenizer、VAE 和文本编码器按说明配齐 |
| PixArt-α / PixArt-Σ | [α 权重](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/tree/main/transformer)、[Σ 权重](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main/transformer)；DiT F16/Q8，T5-XXL CPU | 1024 级文生图 | **4–7 GiB**；两代 VAE 结构不同，需用匹配组件 |
| Iris-3B | [Iris-3B](https://huggingface.co/speridlabs/iris-3b)；Q4_0，Qwen3-VL-4B-Instruct Q4 CPU | RGB 像素空间文生图，无 VAE | **5–8 GiB**；像素空间高分辨率会抬高激活峰值 |

## 符合条件的图像编辑模型

编辑模型复用其主模型的量化权重、文本编码器和 VAE；下表的峰值已包括单张参考图的条件处理，不含多图批处理。

| 上游编辑模型 | 具体权重/量化 | 用途 | 估算推理峰值（单卡）及条件 |
| --- | --- | --- | --- |
| FLUX.1-Kontext-dev | [官方权重](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[GGUF](https://huggingface.co/leejet/FLUX.1-Kontext-dev-GGUF)；Q4_0 | 指令式参考图编辑 | **7–9 GiB**；T5/CLIP CPU，单参考图 |
| Qwen Image Edit / Edit 2509 / Edit 2511 | [Comfy 主权重](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models)、[GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF)；Q3_K_S/Q4_0，Qwen2.5-VL CPU | 中文指令图像编辑 | **8–10 GiB**；Q4 权重可高于空闲显存，必须参数卸载；单图 1024 级 |
| LongCat Image Edit Turbo | [官方权重](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo)、[GGUF](https://huggingface.co/vantagewithai/LongCat-Image-Edit-GGUF)；Q4 | 参考图指令编辑 | **7–9 GiB**；Qwen2.5-VL CPU，单参考图 |
| Boogu Image Edit | [Boogu 扩散组件](https://huggingface.co/Comfy-Org/Boogu-Image/tree/main/diffusion_models)；Q4 主模型、Qwen3-VL CPU | 参考图编辑 | **6–9 GiB**；走 `z_image_omni` preset |
| Mage-Flow-Edit | [Mage-Flow 权重](https://huggingface.co/Comfy-Org/Mage-Flow/tree/main/diffusion_models)；Q4 主模型、Qwen3-VL CPU | 指令式编辑 | **6–9 GiB**；走 `mage_flow` preset，限制 VLM 输入尺寸 |
| LLaDA-Image Edit Base / Turbo | [LLaDA GGUF](https://huggingface.co/fszontagh/LLaDA-Image-Turbo-GGUF)；Q4 扩散模型 + edit connectors/SigVQ | 文本指令编辑 | **6–9 GiB**；encoder CPU，单参考图 |
| Fibo 2.0 Edit Turbo | [官方模型卡](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)、[Comfy 权重目录](https://huggingface.co/Comfy-Org/Fibo-2.0/tree/main/diffusion_models)；Q4 主模型、Qwen3-VL CPU | 快速指令式编辑/多参考编辑 | **6–9 GiB**；低分辨率单图起步，使用 Fibo preset |
| Qwen-Image-Layered（扩展编辑页） | [官方 BF16 权重](https://huggingface.co/Qwen/Qwen-Image-Layered)；转换/量化 Q3_K_S，文本组件 CPU | 分层图像编辑与图层输出 | **8–10 GiB**；在[编辑支持列表](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/edit.md#supported-models)中单独列出，不是 README 七类以外的独立生成模型 |
| Krea2 / Anima 社区编辑 LoRA | [Krea2 LoRA](https://huggingface.co/conradlocke/krea2-identity-edit)、[Anima 基础权重](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/diffusion_models)；LoRA 原精度，底模 Q4 | 身份保持/参考图编辑 | **底模峰值 + 约 0.2–1 GiB**；只加载与主模型架构匹配的 LoRA |
| InstructPix2Pix / CosXL-Edit | [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix)、[CosXL 权重](https://huggingface.co/stabilityai/cosxl/tree/main)；前者 SD1.x F16，后者 SDXL Q8/F16 | 参考图加文本指令编辑 | **4–9 GiB**；CosXL 的许可有非商业研究限制，SDXL 使用 VAE tiling |

## 符合条件的视频模型

| 上游模型 | 具体权重/推荐精度 | 用途 | 估算推理峰值（单卡）及条件 |
| --- | --- | --- | --- |
| Wan2.1 / Wan2.2 | [Wan2.1/2.2 权重与 GGUF](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/wan.md)；1.3B F16/Q8；TI2V 5B 可用 Q8；14B 通常 Q4/Q5，Q8 需 CPU/disk 参数卸载；文本编码器 CPU | T2V、I2V、首尾帧、VACE、S2V | **5–10 GiB**；用 `-M vid_gen`；17–33 帧、480p 起步；配相应版本 VAE/CLIP/audio encoder，必要时用 TAE |
| MiniMax-H3 | [Comfy 权重](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)、[GGUF 量化集合](https://huggingface.co/leejet/MiniMax-H3-GGUF)；T2VA 选 pruned Q4_K_M 扩散约 11.4 GB + Qwen32B Q2_K_M 编码器约 13.1 GB；Ref2VA 可选 pruned Q2_K_M 扩散约 6.72 GB | 同步生成视频与立体声音频；支持首帧、首尾帧及多模态参考 | **8–10 GiB**；短片 17 帧起步，Qwen32B 必须 CPU，扩散权重启用 offload；建议组合主/文本权重约 24.5 GB（未计 VAE 和运行开销），WSL 下可容纳但需避免并发；Q2 文本编码器会有精度损失 |
| LTX-2.3 / LTX-2.5 | [LTX-2.3 权重/GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)、[LTX-2.5 官方权重](https://huggingface.co/Lightricks/LTX-2.5/tree/main/diffusion_models)、[2.5 GGUF](https://huggingface.co/vantagewithai/LTX-2.5-GGUF)；主模型 Q4，Gemma 编码器 CPU | T2V、I2V、首尾帧、音视频生成 | **8–10 GiB**；使用 17 帧/较小画布，按版本配 Gemma 3/4、video/audio VAE；长视频和空间放大会超预算 |
| HunyuanVideo 1.5 | [Comfy 组件](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files)；扩散 Q4，Qwen2.5-VL 7B/ByT5 CPU | 文本生成视频 | **8–10 GiB**；17 帧、低分辨率起步，开启 CPU offload/temporal tiling |
| LingBot-Video Dense 1.3B / MoE 30B-A3B | [Dense 权重](https://huggingface.co/robbyant/lingbot-video-dense-1.3b/tree/main/transformer)、[MoE 权重](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b/tree/main/transformer)；Dense F16/Q8；MoE Q4，Qwen3-VL-4B CPU | T2V、I2V | Dense **5–8 GiB**；MoE **8–10 GiB**（强制 CPU 参数卸载、9 帧/低分辨率起步）；帧数需满足模型时序压缩约束 |

## 附加能力：与基础模型组合后可运行

这些支持项不是独立生成模型。下表显存是相对基础模型的附加开销范围；启用后仍需让**基础模型 + 附件**总峰值留在 10 GiB 应用预算内。

| 附加能力 | 权重入口/量化 | 作用和增量显存 |
| --- | --- | --- |
| PhotoMaker v1 / v2 | [v1](https://huggingface.co/bssrdf/PhotoMaker)、[v2](https://huggingface.co/bssrdf/PhotoMakerV2)；SDXL 必需 | 人像身份保持；约 **+0.5–1.5 GiB**，v2 的 id embeds 可复用 |
| IP-Adapter / Plus | [官方 SD1.5/SDXL 权重](https://huggingface.co/h94/IP-Adapter/tree/main)；匹配 SD1.5 或 SDXL | 参考图外观/主体条件；CLIP Vision 与 adapter 约 **+0.5–1.5 GiB** |
| ControlNet（SD1.5） | 需下载与架构匹配的 ControlNet checkpoint；上游 README 未指定单一权重 | 姿势/边缘/深度条件；单 ControlNet 约 **+1–3 GiB**，超过余量时 CPU offload |
| ADetailer / YOLOv8 detection | [Ultralytics YOLOv8 权重](https://github.com/ultralytics/assets/releases)，先按上游步骤转换 | 检测后局部重绘；检测器放 CPU 时额外 GPU 开销小于 **1 GiB**，局部重绘继续消耗基础模型的峰值 |
| LoRA / LCM-LoRA | [LoRA 使用说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/lora.md)、[LCM 权重/说明](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/lcm.md) | 风格/角色适配或少步采样；通常 **+0.1–0.8 GiB**，必须匹配底模，非独立模型 |
| TAESD / TAEHV / TAEQI2.1 | [TAESD](https://huggingface.co/madebyollin/taesd)、[TAEHV](https://github.com/madebyollin/taehv)、[TAEQI2.1](https://huggingface.co/madebyollin/taeqi2_1) | 低显存快速解码；约 **+0.1–0.5 GiB**，可降低 VAE 解码峰值但有画质损失 |
| ESRGAN / MoSR | [RealESRGAN x4 Anime 6B](https://github.com/xinntao/Real-ESRGAN/releases)、[MoSR 模型示例](https://github.com/Phhofm/models/releases) | 生成后放大；约 **1–3 GiB**，单独阶段运行，超大输入需分块 |

## 不单独计入模型数的条目

- `Z-Image Omni` 是 `docs/edit.md` 的参考图 preset，表中按其实际关联的 Boogu 等底模计算；它没有独立 checkpoint。
- SD1.x/SDXL 的社区 LoRA、LCM-LoRA、ControlNet、IP-Adapter、PhotoMaker、ADetailer、TAESD 和 ESRGAN 均是已有主模型的扩展或后处理，不重复计为独立扩散模型。
- 同一个模型同时支持文生图和图像编辑时，编辑表仅表示另一种推理输入/条件流程，不重复计算部署时的模型显存总和。

## 单卡部署总显存

- 本机实际推理时应按**一次运行一个模型**核算，不把 42+7+5 个条目的峰值相加。建议常规进程显存预算 **9 GiB**；最高占用候选为 SenseNova、HiDream、Ideogram4、Qwen Image/编辑，以及视频大模型，按 **10 GiB** 上限和低帧/低分辨率配置运行。
- 目前系统背景占用约 **4.75 GiB**，因此最高部署总显存预算约为 **4.75 + 10 = 14.75 GiB / 16.0 GiB**；这给当前 WSL/显示环境留下约 1.25 GiB 余量。若运行前 `nvidia-smi` 的空闲显存低于 11 GiB，应先停掉其他 GPU 推理任务或降低 `--max-vram`，不要照搬 10 GiB 档。
- 如果“总大小”指全量权重文件而非显存，答案会是数十至数百 GB，且主要在磁盘/主机内存；它不能与单卡 VRAM 相加。完整主/辅助组件见各行权重链接。
- 此表证明的是按建议量化/卸载与工作负载能纳入本机显存预算的部署配置，不承诺未经实测的吞吐量、画质或任意分辨率/帧数。视频模型如果要提升至 720p、49/81 帧，需逐项重新采集 `nvidia-smi` 峰值。

## 上游资料

- [stable-diffusion.cpp README：支持模型、权重格式与后端](https://github.com/leejet/stable-diffusion.cpp)
- [显存优化、CPU/disk offload、分段推理和 VAE tiling](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/performance.md)
- [Backend/参数放置](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/backend.md)
- [GGUF 与量化类型](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)
- [图像编辑 preset 覆盖](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/edit.md)
