# MiniMax H3 16G HiFi I2V 基准

独立使用 `scenes.json` 中的 3 张图、提示词、seed、480×832 分辨率和 24 FPS，运行
3 场景 × 3 seed。H3 使用原生 20 步、5 秒请求和原生音频；`17k+5` 帧网格使输出为
124 帧（约 5.17 秒）。

基准通过本地 H3 API（默认 `http://127.0.0.1:8392`）逐条请求，测量每次完整服务请求耗时，
包括 ComfyUI 子进程启动、权重加载、推理和服务清理。基准使用一次性 worker。图片上传和视频
输出都写入 H3 结果目录或在完成后清理；输入图片保持只读。显存与 WSL 内存按 0.5 秒采样，
GPU 门槛为至少 13,000 MiB 空闲显存。

## 运行

先确认 H3 服务在 `8392` 健康可用，并关闭其它 GPU 推理。从仓库根目录运行：

```bash
uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py preflight
uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py run --limit 1
uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py run
uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py report
```

Smoke run 是 9 条计划中的第一条；完整运行会跳过已成功样本并接着执行剩余条目。任何生成失败
都会停止后续请求，以免重复触发显存或运行故障。CSV 遥测、模型清单、报告、视频画廊和视频
写入 Git 忽略的 `tests/i2v-benchmark/results/minimax_h3/`。

每条请求均通过一次性 worker 冷启动，因此测得的是当前 API 的完整请求成本，不是纯采样时间。
