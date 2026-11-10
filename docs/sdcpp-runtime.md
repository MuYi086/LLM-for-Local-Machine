# stable-diffusion.cpp 执行层

当前提供 stdlib-only CLI provider，并已接入 MOSS-Image-SFT-Q8_0 和 FLUX-3-Image-Klein-GGUF-Q8 两个图片 API。
它们分别默认监听 8501 与 8502，由 `start.sh` 管理。真实推理验收状态以
[`versions.json`](../sdcpp/manifests/versions.json) 为准；目前只有 MOSS 标记为真实推理验证。`start.sh` 当前管理十六个 HTTP 进程；
`model-video/` 中退役代码仍为归档，尚未接入视频服务。

## 已实现边界

- `inference_gateway.sdcpp`：严格清单、结构化参数、组件预检、共享 GPU 锁和独立 CLI 进程组。
- `scripts/sdcpp/manager.py`：`status`、`validate`、`generate`。
- `scripts/localai/recover_quarantine.py`：统一分派 LocalAI、llama.cpp 和 sd.cpp 隔离恢复。
- PNG 图片和 AVI 视频按 UUID 写入独立目录，成功后原子发布；失败仅清理当前请求。
- `generation.json` 保存请求与二进制摘要；返回 JSON 包含排队、worker 总耗时和共享显存采样指标。
  worker 总耗时包括加载、生成和回收，尚不拆分模型加载与采样时间；显存指标不是模型独占显存。
  输出验证目前检查常规文件和格式头，不代替解码、画质或视频时序评估。

provider 不提供独立 HTTP 任务队列或 server 常驻复用。两个图片 API 接受同步请求，通过全局 GPU 锁串行生成。
FLUX 编辑接口接收受限大小的 PNG/JPEG 文件并临时暂存，不接受客户端路径；该接口不提供编辑 mask、LoRA 或多阶段工作流。
`image-to-image` 仅供清单明确声明并配置了参考图参数的模型使用，不代表所有模型都支持图片编辑。

## 外部运行时

版本基线见 [`versions.json`](../sdcpp/manifests/versions.json)：

```text
repository=https://github.com/leejet/stable-diffusion.cpp
commit=a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf
executable=sd-cli
```

在仓库外按该 commit 的官方构建指南准备 CUDA 版 sd-cli；模型、VAE、文本编码器也保持在仓库外。
当前工具不会下载源码、模型或编译二进制。操作者需核对构建来源，`SDCPP_BIN_SHA256` 只能确认文件内容，
不能证明二进制由指定 commit 编译。升级时同时更新版本清单并重新验证参数与模型。

```bash
export SDCPP_BIN="/path/to/external/build/bin/sd-cli"
sha256sum "$SDCPP_BIN"
export SDCPP_BIN_SHA256="<上一步输出的 64 位摘要>"

# 必须与 start.sh 和所有模型服务完全相同。
export RUNTIME_CACHE_DIR="$PWD/storage/.cache/runtime"
export GPU_LOCK_FILE="$RUNTIME_CACHE_DIR/gpu-runtime.lock"
export SDCPP_OUTPUT_DIR="$RUNTIME_CACHE_DIR/sdcpp"

python3 scripts/sdcpp/manager.py status
```

`status` 不执行 sd-cli、不计算二进制摘要、不加载权重；`configured` 仅表示环境配置语法有效。
`validate` 校验二进制摘要和组件文件是否存在、非空、可读，不能证明模型结构可用。
这些命令及隔离恢复只需 Python 3.12 与仓库源码；也可通过 `uv run --project qa --locked python ...` 执行。

## 模型清单与图片 API

默认 [`models.json`](../sdcpp/manifests/models.json) 注册 MOSS 与 FLUX.3 Klein 两个模型。MOSS 声明
`text-to-image` 任务和四个只读组件；文件默认位于
`$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF/`：MOSS Q8_0 diffusion GGUF、
Qwen3.5-9B-Q6_K GGUF、FLUX.2 VAE safetensors 和官方 `tokenizer.json`。启动前会检查所有组件和
sd-cli 摘要，不下载权重或改写模型文件。FLUX.3 Klein 声明 `text-to-image` 与 `image-to-image`，四个只读组件
位于 `$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF/`：`flux3-image-klein-Q8_0.gguf`、
`Qwen3VL-4B-Instruct-Q8_0.gguf`、`mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` 和 `ae.safetensors`。

MOSS 默认生成参数为 1024×1024、20 步、CFG 3、Euler，runtime 使用 CUDA0、CPU 参数后端和 CPU offload；
尺寸上限为 1024×1024、总像素不超过 1,048,576，宽高都必须是 16 的倍数，步数上限为 50。
MOSS registry 的 flow shift 默认值为 5，API 不开放覆盖项。

FLUX.3 Klein 默认使用 1024×1024、16 步、CFG 1.0、Euler、CUDA0、CPU 参数后端和 CPU offload；最大分辨率为
1024×1024，宽高必须是 16 的倍数，最大 50 步。上游 FLUX.3 Klein registry 的 flow shift 默认值为 6.93，
清单不传 `--flow-shift`，沿用该默认值。图像编辑通过 sd.cpp 的 `--ref-image` 传入一张上传参考图。
新增其他模型时需先验证实际任务与组件组合，再通过 `SDCPP_MODEL_MANIFEST` 或 `--manifest` 指定本机清单。

MOSS API 由 `start.sh` 管理，默认路由为 `POST http://127.0.0.1:8501/v1/openmoss/MOSS-Image-SFT-Q8_0`，
健康检查为 `GET http://127.0.0.1:8501/v1/health`。健康检查只检查 sd-cli 可执行文件、组件文件和 quarantine，
不会计算摘要或加载模型。请求字段包括 `prompt`、可选 `negative_prompt`、`size`、`steps`、`cfg_scale` 和 `seed`；
每次只生成一张 PNG，并以 OpenAI Images 风格的 `b64_json` 返回。

```bash
curl -X POST http://127.0.0.1:8501/v1/openmoss/MOSS-Image-SFT-Q8_0 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一只坐在窗边的猫，柔和晨光","size":"1024x1024","steps":20,"cfg_scale":3,"seed":42}'

uv run --project qa --locked python tests/MOSS-Image-SFT-Q8_0/generate_example.py
```

HTTP demo 会把 PNG 以唯一文件名保存到 `storage/llm_tests/MOSS-Image-SFT-Q8_0/`；API worker 中间产物位于
`$RUNTIME_CACHE_DIR/sdcpp/<request-id>/`。这两处都属于运行数据，不应提交或由本集成清理。

FLUX 文生图与编辑 API 由 `start.sh` 管理，路由分别为：
`POST http://127.0.0.1:8502/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8` 和
`POST http://127.0.0.1:8502/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8/edit`。
编辑使用 multipart 字段 `prompt`、`image`、可选 `size`、`steps`、`cfg_scale` 与 `seed`；参考图只接受 PNG/JPEG，
默认最多 64 MiB，临时文件存于 `$RUNTIME_CACHE_DIR/sdcpp_uploads` 并在请求结束后删除。

```bash
curl -X POST http://127.0.0.1:8502/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red bicycle beside a quiet canal at sunset","size":"1024x1024","steps":16,"cfg_scale":1,"seed":42}'

curl -X POST http://127.0.0.1:8502/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8/edit \
  -F 'prompt=Change the bicycle to bright blue' \
  -F 'image=@reference.png;type=image/png' \
  -F 'size=1024x1024' -F 'steps=16' -F 'cfg_scale=1' -F 'seed=43'

uv run --project qa --locked python tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py
```

`FLUX-3-Image-Klein-GGUF-Q8` 尚未由真实模型运行验证；单元测试只使用伪生成器，不初始化 CUDA。

以下为新增模型时可参考的通用字段示意，模型名和路径是占位值：

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "chosen-image-model",
      "tasks": ["text-to-image"],
      "components": {
        "diffusion_model": "${HF_MIRROR_DIR}/chosen/model.gguf",
        "vae": "${HF_MIRROR_DIR}/chosen/vae.safetensors",
        "t5xxl": "${HF_MIRROR_DIR}/chosen/text-encoder.gguf",
        "tokenizer": "${HF_MIRROR_DIR}/chosen/tokenizer.json"
      },
      "defaults": {
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 1.0,
        "sampler": "euler"
      },
      "limits": {
        "max_width": 1024,
        "max_height": 1024,
        "max_pixels": 1048576,
        "max_steps": 30,
        "max_frames": 1,
        "width_multiple": 16,
        "height_multiple": 16
      },
      "runtime": {
        "backend": "cuda0",
        "params_backend": "cpu",
        "flash_attn": "auto",
        "vae_tiling": true,
        "temporal_tiling": false,
        "offload_to_cpu": false,
        "max_vram_gib": 14
      }
    }
  ]
}
```

- 组件允许 `model` 或 `diffusion_model` 二选一，并可配 `high_noise_diffusion_model`、`vae`、
  `clip_l`、`clip_g`、`t5xxl`、`llm`、`llm_vision`、`clip_vision`、`tokenizer`。具体必需组件由选定模型确定。
- 相对组件路径基于清单所在目录；支持环境变量和 `~`，未定义变量直接报错。
- 任务允许 `text-to-image`、`image-to-image`、`text-to-video`、`image-to-video`，只能调用清单声明的任务。
- 视频至少生成 2 帧，需明确覆盖 `defaults.frames` 和 `limits.max_frames`；按模型需要配置 `frame_multiple`、
  `frame_offset`，帧数须满足 `(frames - frame_offset) % frame_multiple == 0`。
  `defaults.fps` 默认为 16，可覆盖，上限 120；这是请求参数，不保证模型原生帧率可调。
- 采样器允许 `euler`、`euler_a`、`heun`、`dpm2`、`dpm++2m`、`lcm`，仅使用模型验证过的组合。
- 运行后端目前允许 `cuda0`、`cpu`；参数放置允许 `cuda0`、`cpu`。即使配置 CPU 计算，本 provider
  仍保守地走全局锁。14 GiB 预算是初始配置建议，不能保证每个模型、分辨率或帧数组合都可运行。
- `offload_to_cpu` 为布尔选项，对应 sd-cli 的 `--offload-to-cpu`；仅在模型验证过的配置中启用。
- `runtime.image_to_image_flag` 可为 `init-img` 或 `ref-image`，只影响 `image-to-image`；`image-to-video` 保持使用 `--init-img`。
- 不接受任意 CLI flags，不在每次加载时自动量化；提前准备与固定版本兼容的权重。

```bash
export SDCPP_MODEL_MANIFEST="/path/to/local-models.json"
python3 scripts/sdcpp/manager.py validate --model chosen-image-model
python3 scripts/sdcpp/manager.py generate \
  --model chosen-image-model --task text-to-image \
  --prompt '一只坐在窗边的猫' --seed 42

# 仅当本地清单已配置并验证对应视频任务时使用。
python3 scripts/sdcpp/manager.py generate \
  --model chosen-video-model --task image-to-video \
  --init-image /path/to/reference.png --prompt '镜头缓慢推进' \
  --width 832 --height 480 --frames 33 --fps 16
```

CLI 的文件路径来自受信任的本机操作者。HTTP API 不接受客户端提供的文件路径或服务器 `full_path`；
FLUX 编辑先把上传内容存入服务管理的临时文件，再将该路径传给 provider。
输入文件只读，推理参数以 argv 传递，不经 shell 展开。

## 超时、取消与恢复

| 环境变量 | 默认值 | 合同 |
| --- | --- | --- |
| `GPU_LOCK_WAIT_TIMEOUT` | 900 秒 | 有限正数，等待共享 GPU 锁 |
| `SDCPP_RUN_TIMEOUT` | 3600 秒 | 有限正数，从许可启动后开始，包括加载和生成 |
| `SDCPP_CLEANUP_TIMEOUT` | 30 秒 | 有限正数，停止进程组并确认 |
| `CUDA_RELEASE_DELAY` | 2 秒 | 有限非负，确认退出后继续持锁等待 |

程序调用 `generate(..., cancel_event=event)` 可请求取消；排队期间仍由排队超时限制，取得锁后检查取消。
CLI 的 Ctrl+C 会进入进程回收；SIGKILL 等无法执行 finally 的终止会保留 quarantine。

worker 先等待父进程通过 stdin 授权。只有原子记录 PID、PGID 和 Linux starttime 后才加载权重。
退出码为 0 也会确认并回收同组后代。正常执行以 `waitid(WNOWAIT)` 观察退出，在组回收前保留根进程，
避免数字 PGID 提前被复用。不能确认停止、身份被复用或隔离标记被替换时，保留隔离和任务目录。

```bash
python3 scripts/localai/recover_quarantine.py
```

恢复入口取得共享锁后重新读取标记；sd.cpp worker CLI 没有 HTTP 端口，因此只核对进程身份与进程组退出。
若根进程已经退出但仍有无法核对身份的后代，脚本会保留隔离，需要操作者核查进程，再重新执行恢复入口。
恢复不删除历史产物或遗留任务目录；不要手动移除 quarantine。

## MOSS 推理验收记录

2026-11-10 使用固定 stable-diffusion.cpp commit 与 MOSS Q8_0 本地组件，通过 8501 HTTP API 完成
1024×1024、20 步、CFG 3、Euler、seed 13579 的单张生成。PNG 为有效 1024×1024 RGB，API 返回 200；
worker 总耗时约 35.98 秒，观测峰值 GPU 显存约 7951 MiB，进程组正常退出且未留下 quarantine。
该记录验证了组件组合、API 路由和一次真实推理，不代表系统性画质评估。

接入其他图片或视频模型前，仍需按模型记录固定 commit、二进制摘要、权重摘要、量化、采样器、seed、尺寸、
步数、帧数和 FPS；比较冷启动耗时、峰值显存、系统内存、输出质量与退出状态，再开放相应 API 能力。
