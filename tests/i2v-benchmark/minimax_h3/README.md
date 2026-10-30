# MiniMax H3 16G HiFi I2V 对照基准

复用 `kandinsky6/config.json` 中的 3 张图、提示词、seed、480×832 分辨率和 24 FPS，运行
3 场景 × 3 seed = 9 条。H3 保持服务原生 20 步、5 秒请求和原生音频；H3 的帧网格要求
`17k+5`，所以实际输出为 124 帧（约 5.17 秒），Kandinsky 输出 121 帧（约 5.04 秒）。

基准通过正在运行的 H3 API（默认 `http://127.0.0.1:8392`）逐条请求，测量每次完整服务请求的
耗时，包括 ComfyUI 子进程启动、权重加载、推理和服务清理。每次请求仍按服务设计使用一个新 worker。
图片上传和生成视频都写入测试结果目录或在完成后清理；输入图片保持只读。显存与 WSL 内存使用
0.5 秒采样，GPU 门槛为 H3 服务设置的至少 13,000 MiB 空闲显存。

## 运行

先确认 H3 服务在 `8392` 健康可用，并关闭其它 GPU 推理。命令从仓库根目录执行：

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

Smoke run 是 9 条计划中的第一条，完整运行会跳过已成功的样本并接着执行剩余 8 条。任何生成失败
都会停止后续请求，以免重复触发显存或运行故障。结果、CSV 遥测、模型清单、报告和同图同 seed
的 Kandinsky/H3 对比页写入 Git 忽略的 `tests/i2v-benchmark/results/minimax_h3/`。

Kandinsky 在单个进程内驻留 pipeline，H3 每次通过一次性 worker 冷启动，因此时间对比代表当前
服务的实际请求成本，不是两个模型的纯采样时间。两者的步数、帧数和音轨也按各自原生设置保留。
