# stable-diffusion.cpp 执行层

当前提供 stdlib-only CLI provider，供本机操作者以及后续模型 API 复用。默认模型清单为空，
尚未完成真实模型兼容性、性能和质量验证。它不增加 HTTP 端口，`start.sh` 仍管理原有十四个 HTTP 进程。
`model-video/` 中退役代码仍为归档。

## 已实现边界

- `inference_gateway.sdcpp`：严格清单、结构化参数、组件预检、共享 GPU 锁和独立 CLI 进程组。
- `scripts/sdcpp/manager.py`：`status`、`validate`、`generate`。
- `scripts/localai/recover_quarantine.py`：统一分派 LocalAI、llama.cpp 和 sd.cpp 隔离恢复。
- PNG 图片和 AVI 视频按 UUID 写入独立目录，成功后原子发布；失败仅清理当前请求。
- `generation.json` 保存请求与二进制摘要；返回 JSON 包含排队、worker 总耗时和共享显存采样指标。
  worker 总耗时包括加载、生成和回收，尚不拆分模型加载与采样时间；显存指标不是模型独占显存。
  输出验证目前检查常规文件和格式头，不代替解码、画质或视频时序评估。

本阶段尚不提供 HTTP 任务队列、远程上传、server 常驻复用、编辑 mask、LoRA 或多阶段工作流。
`image-to-image` 只映射普通参考图与文本条件，不代表所有模型支持图片编辑。

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

## 本地模型清单

默认 [`models.json`](../sdcpp/manifests/models.json) 为 `{"schema_version": 1, "models": []}`。
选型后在仓库外创建本地清单，通过 `SDCPP_MODEL_MANIFEST` 或全局 `--manifest` 指定。
以下是字段示意，名称和路径均为占位值，不代表已经验证某个模型：

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
        "t5xxl": "${HF_MIRROR_DIR}/chosen/text-encoder.gguf"
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
        "max_vram_gib": 14
      }
    }
  ]
}
```

- 组件允许 `model` 或 `diffusion_model` 二选一，并可配 `high_noise_diffusion_model`、`vae`、
  `clip_l`、`clip_g`、`t5xxl`、`llm`、`clip_vision`。具体必需组件由选定模型确定。
- 相对组件路径基于清单所在目录；支持环境变量和 `~`，未定义变量直接报错。
- 任务允许 `text-to-image`、`image-to-image`、`text-to-video`、`image-to-video`，只能调用清单声明的任务。
- 视频至少生成 2 帧，需明确覆盖 `defaults.frames` 和 `limits.max_frames`；按模型需要配置 `frame_multiple`、
  `frame_offset`，帧数须满足 `(frames - frame_offset) % frame_multiple == 0`。
  `defaults.fps` 默认为 16，可覆盖，上限 120；这是请求参数，不保证模型原生帧率可调。
- 采样器允许 `euler`、`euler_a`、`heun`、`dpm2`、`dpm++2m`、`lcm`，仅使用模型验证过的组合。
- 运行后端目前允许 `cuda0`、`cpu`；参数放置允许 `cuda0`、`cpu`。即使配置 CPU 计算，本 provider
  仍保守地走全局锁。14 GiB 预算是初始配置建议，不能保证每个模型、分辨率或帧数组合都可运行。
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

CLI 的文件路径来自受信任的本机操作者。未来 HTTP 入口必须使用上传和逻辑引用，不能直接把远程请求的
`full_path` 传给本接口。输入文件只读，推理参数以 argv 传递，不经 shell 展开。

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

恢复入口取得共享锁后重新读取标记；sd.cpp CLI 没有 HTTP 端口，因此只核对进程身份与进程组退出。
若根进程已经退出但仍有无法核对身份的后代，脚本会保留隔离，需要操作者核查进程，再重新执行恢复入口。
恢复不删除历史产物或遗留任务目录；不要手动移除 quarantine。

## 后续模型验收

按模型分别记录固定 commit、二进制摘要、权重摘要、量化、采样器、seed、尺寸、步数、帧数和 FPS。
比较冷启动总耗时、峰值显存、系统内存、输出质量以及退出状态；图片先行，视频逐级提升资源参数。
只有完成真实生成与能力核验，才把对应任务接入模型 API 和当前能力清单。
