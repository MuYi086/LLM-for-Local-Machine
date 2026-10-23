# MOSS-Music-8B-Instruct 音乐理解服务

默认端口 `8310`，业务路由 `POST /v1/openMoss/moss-music`。上传音乐并用文字提问，返回
音乐描述或问答文本；例如分析风格、乐器、人声、情绪、节奏、编曲，也可以要求转写歌词。

本服务使用本机已准备的 [MOSS-Music-8B-Instruct GGUF 转换](https://huggingface.co/scragnog/MOSS-Music-8B-Instruct-GGUF)，
原模型为 [OpenMOSS-Team/MOSS-Music-8B-Instruct](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct)。
GGUF 需要 HOT-Step-CPP 的音频编码器与 deepstack（多层音频特征注入）实现，不能仅交给 llama.cpp。
服务不依赖 Torch 或 Transformers，HTTP 进程不加载模型。每个请求启动一个 Python worker，
通过 `exec` 替换为原版 `ace-caption`，完成或异常后回收整个进程组。

## 本地模型和构建

默认权重目录为 `$HF_MIRROR_DIR/OpenMOSS/moss-music`，`HF_MIRROR_DIR` 默认 `~/hf-mirror`。
两份文件必须同时存在；tokenizer 已嵌入语言模型 GGUF，无需另行下载：

| 文件 | 完整制品大小 |
| --- | --- |
| `moss-lm-q8_0.gguf` | 8,709,513,792 bytes |
| `moss-aud-f16.gguf` | 1,724,480,640 bytes |

服务在排队前校验 GGUF v3 文件头和大小，不在健康检查读取完整权重或执行推理。
当前仅维护这组 Q8 语言模型与 F16 音频编码器；推理不自动下载权重。

上游源码保留在仓库外，固定版本为：

- [HOT-Step-CPP](https://github.com/scragnog/HOT-Step-CPP)：`2db8c0d5380ee5703e23affbcab74f4ef8e4fba7`。
- `engine/ggml` 子模块：`8eb519bd4f2ad18fa9fc9ff1e46def9fd493fed0`。

准备外部源码时执行以下命令；已有目录时直接核对 revision，避免覆盖本地改动：

```bash
git clone https://github.com/scragnog/HOT-Step-CPP.git "$HOME/LLM-depency/HOT-Step-CPP"
git -C "$HOME/LLM-depency/HOT-Step-CPP" checkout 2db8c0d5380ee5703e23affbcab74f4ef8e4fba7
git -C "$HOME/LLM-depency/HOT-Step-CPP" submodule update --init engine/ggml
```

构建需要 CMake 3.21+、C++17、CUDA Toolkit 与兼容的宿主 C++ 编译器。从本仓库根目录执行：

```bash
bash scripts/build_moss_music.sh
uv sync --project model-audio/analyse_moss-music-8b-Instruct --locked
```

本仓库的 CMake 入口直接编译上游原版 CLI 及 GGML，不引入上游完整应用的 VST3 或其他模型。
构建产物默认位于 `~/.local/share/moss-music/build`。默认 CUDA 架构为 `89-real`；其他显卡通过
`MOSS_MUSIC_CUDA_ARCHITECTURES` 调整。CUDA 12.0 配合 GCC 12 的构建示例：

```bash
bash scripts/build_moss_music.sh -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
```

## 启动与调用

全量启动使用根目录 `bash start.sh`，会预检 `8310` 并在退出时清理该服务进程组。
单独启动时执行：

```bash
uv run --project model-audio/analyse_moss-music-8b-Instruct --no-sync \
  python model-audio/analyse_moss-music-8b-Instruct/main.py
curl -fsS http://127.0.0.1:8310/v1/health
```

健康接口 HTTP 200 表示 API 可以响应；`available.ready` 才表示文件、二进制、FFmpeg 和设备检查通过，
仍不代表真实模型输出质量已经验收。调用示例（另开终端，音频路径替换为自己的文件）：

```bash
curl -fsS http://127.0.0.1:8310/v1/openMoss/moss-music \
  -F 'audio=@song.wav' \
  -F 'prompt=请分析这段音乐的乐器、人声与情绪。' \
  -F 'max_new_tokens=1024' \
  -F 'temperature=0'
```

请求为 `multipart/form-data`：

| 字段 | 默认值与边界 | 含义 |
| --- | --- | --- |
| `audio` | 必填，默认最大 64 MiB | WAV、MP3、OGG、FLAC、M4A、AAC 或 WebM；实际内容须能被 FFmpeg 解码 |
| `prompt` | 默认中文综合音乐分析问题；1–16000 字符，拒绝纯空白 | 音乐描述、具体问答或歌词转写指令 |
| `max_new_tokens` | 1024；1–4096 | 输出 token 上限 |
| `temperature` | 0；0–2，有限数 | 0 使用贪心解码，正值使用上游采样器 |

返回 JSON 字段为 `model`、`runtime`、`text`、`duration_seconds` 和 `audio_sha256`。
`text` 是模型的自由文本输出，不保证特定结构或 BPM/调性等估计的精度。

CPU 预检在 GPU 排队前完成：验证实际音频时长为 `(0, 420]` 秒，转成 16 kHz 单声道 PCM16，
再复核解码时长。超长输入直接拒绝；不静默截断。上传、规范化 WAV、请求 JSON、提示词和输出 TXT
都只暂存于请求目录，成功、失败及超时后删除；响应不落盘到长期业务存储。

错误码：字段/音频校验失败 `422`，超过上传上限 `413`，缺失依赖/权重或 GPU 排队超时 `503`，
CPU 转换或 worker 超时 `504`，worker 推理失败 `500`。

## 配置

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MOSS_MUSIC_HOST` / `MOSS_MUSIC_PORT` | `0.0.0.0` / `8310` | 监听地址；也接受共享 `HOST` / `PORT` |
| `MOSS_MUSIC_MODEL_DIR` | `$HF_MIRROR_DIR/OpenMOSS/moss-music` | 本地 GGUF 目录 |
| `MOSS_MUSIC_CAPTION_BIN` | `~/.local/share/moss-music/build/ace-caption` | C++ 推理入口 |
| `MOSS_MUSIC_BACKEND` | `CUDA0` | 固定设备，也接受 `CPU` 或其他 `CUDA` 序号 |
| `MOSS_MUSIC_REQUEST_TIMEOUT` | `900` | worker 超时秒数，须为正数 |
| `MOSS_MUSIC_MAX_AUDIO_SECONDS` | `420` | 音频时长上限，可降低，不能高于 420 |
| `MOSS_MUSIC_WORKER_TMP_DIR` | `$RUNTIME_CACHE_DIR/moss_music` | 每请求临时目录的父目录 |
| `MOSS_MUSIC_FFMPEG_BIN` / `MOSS_MUSIC_FFPROBE_BIN` | `ffmpeg` / `ffprobe` | CPU 预检工具 |
| `UPLOAD_MAX_BYTES` | `67108864` | 共享流式上传字节上限 |
| `GPU_LOCK_FILE` | `$RUNTIME_CACHE_DIR/gpu-runtime.lock` | 共享 GPU 串行队列 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` | GPU 排队超时秒数，须为正数 |
| `CUDA_RELEASE_DELAY` | `2` | 持锁期间的 CUDA 释放等待秒数 |

构建脚本另接受 `MOSS_MUSIC_SOURCE_DIR`、`MOSS_MUSIC_BUILD_DIR`、`MOSS_MUSIC_BUILD_JOBS` 和
`MOSS_MUSIC_CUDA_ARCHITECTURES`，CMake 其他参数可以直接附在命令末尾。

## 无模型验证

```bash
uv run --project qa --locked python -m unittest discover -s tests -p test_moss_music_service.py -v
bash scripts/quality_gate.sh
```

测试 mock 模型进程，不下载权重、不依赖 GPU、不执行真实推理；覆盖上传、音频预检、字段校验、
队列与进程组生命周期、超时及临时文件清理。真实模型的描述质量和歌词准确性需要独立手动验收。
