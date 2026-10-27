# MSST Mega 53 音乐声部分离服务

默认端口 **8380**；`POST /v1/msst/separate` 接收歌曲和调用方明确指定的声部，返回 ZIP。
`stems` 必填，模型层不固定业务配器，不默认分离 53 轨，不限制为某几个推荐乐器。
`GET /v1/health` 的 `supported_stems` 来自本地模型 YAML，当前权重有 53 类。
类别名严格按该清单填写，例如当前模型使用 `vocal`，不是 `vocals`。

## 安装与外部文件

在仓库根目录由运维手动执行（启动脚本不会安装）：

```bash
uv sync --project model-audio/separate_msst-mega53 --locked
uv run --no-sync --project model-audio/separate_msst-mega53 python model-audio/separate_msst-mega53/main.py
```

此服务拥有独立 uv 环境；已安装外部 MSST 的 `.venv` 不能代替服务环境同步。
服务复用同一个 uv Python 3.12.13、Torch 2.12.0 / TorchAudio 2.11.0 CUDA 13.0 wheel 缓存，
不从外部 `.venv` 导入包，也不安装另一份名为 `msst` 的 PyPI 包。
Torch 只在 worker 中导入。依赖与权重准备后的健康检查仍不能替代真实推理验收。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MSST_HOST` / `MSST_PORT` | `0.0.0.0` / `8380` | 监听地址 |
| `MSST_SOURCE_DIR` | `~/LLM-depency/Music-Source-Separation-Training` | 本地上游源码，不复制进仓库 |
| `MSST_MODEL_DIR` | `$HF_MIRROR_DIR/oulianov/mvsep_mega_53` | checkpoint 与 YAML |
| `MSST_OUTPUT_DIR` | `$STORAGE_DIR/msst` | 持久保存 ZIP，不由 WAV 保留策略自动清理 |
| `MSST_DEVICE` | `cuda:0` | 允许 `cpu` 或 `cuda:N` |
| `MSST_REQUEST_TIMEOUT` | `900` | worker 推理超时秒数 |
| `MSST_MAX_AUDIO_SECONDS` | `600` | 最大输入时长，CPU 阶段校验 |
| `MSST_MAX_OUTPUT_BYTES` | `2147483648` | 预估分轨成品总量上限；超过时在排队前拒绝 |
| `UPLOAD_MAX_BYTES` | `67108864` | 单次流式上传上限，大型歌曲 WAV 可由运维提高 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` | 共享 GPU 队列等待时限，必须大于零 |
| `CUDA_RELEASE_DELAY` | `2` | worker 退出后持锁等待秒数 |
| `MSST_FFMPEG_BIN` / `MSST_FFPROBE_BIN` | `ffmpeg` / `ffprobe` | 宿主机媒体工具 |
| `MSST_WORKER_TMP_DIR` / `MSST_UPLOAD_DIR` | runtime cache 下 `msst_worker` / `msst_uploads` | 请求临时目录 |

模型目录须同时包含 `mvsep_mega_model_bs_roformer_53_stems_v1.ckpt` 与
`mvsep_mega_model_bs_roformer_53_stems.yaml`。模型和源码只读取本地文件，请求不下载。
源代码、模型配置和权重需要一起维护兼容性；当前本地 checkout 与模型构造接口已核对，真实模型推理尚未验收。

## 请求与返回

```bash
curl -fsS http://127.0.0.1:8380/v1/health
curl --fail-with-body http://127.0.0.1:8380/v1/msst/separate \
  -F 'audio=@song.wav' \
  -F 'stems=guitar,strings,synth' \
  -F 'chunk_seconds=10' \
  -o stems.zip
```

- `audio`：multipart 音频文件，遵循共享上传扩展名/媒体类型和大小校验。
- `stems`：必填的英文逗号分隔类别，按模型支持清单选择任意非空子集；拒绝空项、未知名称和重复。
  仅校验模型支持范围，不在服务中强制业务推荐或排除具有包含关系的类别。
- `chunk_seconds`：`5`、`10`、`20`，默认 `10`；batch size 固定为 1，overlap 固定为 2。
  不做静默 OOM 重试，不启用 TTA 或 torch.compile。缩小分块可能影响边界效果，应实测。

成功返回 `application/zip`，包括每个请求声部的 `<stem>.wav`、`residual.wav` 和 `manifest.json`。
顺序与请求一致；所有 WAV 为 **44.1 kHz 双声道 FLOAT**，样本数相同，不逐轨响度归一化。
HTTP 头 `X-MSST-Batch-ID` 标识本次归档，`X-MSST-Audio-SHA256` 是原始上传内容哈希。
无独立上传、任意路径读取、后台任务查询或持久下载 API；业务方须接收并保存这次 ZIP 响应。

manifest 包含输入/模型/配置 SHA-256、上游 Git revision（可获取时）、所选类别、采样率、声道、样本数、实际分块参数、每轨文件名与
SHA-256/peak/rms_dbfs，以及 `reconstruction_max_abs_error` 和警示。
这是分轨元数据，**不是** 8300 的 Render Manifest v1，业务方需要另行编译渲染计划。

先在 CPU 完整加载 checkpoint 并严格匹配，再只保留请求声部的输出头，随后移动到 GPU。
未选择头不进入 GPU；共享 Transformer 主干仍需计算，不能承诺耗时按轨数同比例下降。
最多可请求模型全部声部，但显存、主机内存、时限和输出容量仍是实际约束。

## residual 与空间化

`residual = 44.1 kHz 标准化输入 - sum(选定声部)`。将所选轨道原样加回 residual，
应在浮点误差内重建标准化输入；不会严格逐字节重建上游原始 48 kHz 文件。
FLOAT WAV 可保存超出 ±1 的值，不能先转整型或逐轨截幅。

数学 residual 不会消除串音或类别重叠，也不是独立提取的“其他乐器”。
例如 strings/violin、vocal/lead-vocal/back-vocal、drums/kick 等可能包含同一声音。
改变空间位置或增益后，原来的相位抵消关系会变化，必须试听。
丢弃某目标轨前须将该轨干声回并到底轨；不能把原曲额外叠加到这些轨道上。

按声部生成计划后，以独立资产提交到 8300 `/v1/audio/spatial/render`，由它完成标准化到
48 kHz、空间处理、混合、loudnorm 和编码；禁止浏览器预混或在 Steam Audio 后追加 Haas/aecho。
无需再让其他模型逐一处理歌曲，音乐分析可按需调用。

错误：参数/音频无效 `422`；上传超限 `413`；依赖/模型/工具不可用、GPU 排队超时或隔离 `503`；
worker 超时 `504`；推理或归档失败 `500`。成功、失败和超时均清理请求与 worker 临时文件和进程组。
客户端超时应覆盖 CPU 预检、GPU 排队、worker 推理和 ZIP 传输，不要每次网络超时立即重复提交 GPU 任务。
