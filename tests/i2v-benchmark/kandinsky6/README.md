# Kandinsky 6 Lite I2V 本地基准

这是一个仅供验证、可整目录删除的对照测试，所有代码、配置和结果都留在 `tests/i2v-benchmark/`。
它复用 Hunyuan 基准的 3 张真实竖屏图、3 段固定英文提示词、seed `42/1234/2026`、480×832、
121 帧、24 FPS，以及 GPU/WSL 内存采样和人工盲评流程。Kandinsky 的模型参数遵循 Lite 蒸馏版：
BF16、10 步、`guidance_scale=1.0`，并关闭音频采样。

测试直接调用本地 Kandinsky worker pipeline，使画幅、帧数和提示词可与 Hunyuan 基准对照；
正式 HTTP 接口的默认画幅仍是 864×480。benchmark 在一次命令中驻留 pipeline，记录首次冷启动和后续
热运行；正式 HTTP 服务每次请求使用一次性 worker，因此这里的壁钟时间不代表 HTTP 接口耗时。
两模型仍使用各自蒸馏 checkpoint 的官方步数，不能把 Hunyuan 的 12 步和 flow shift=7 套到 Kandinsky。

## 运行

先安装仓库中新建的 Kandinsky 独立环境，再从仓库根目录执行：

```bash
uv sync --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --locked
uv run --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --no-sync \
  python tests/i2v-benchmark/kandinsky6/benchmark.py preflight
uv run --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --no-sync \
  python tests/i2v-benchmark/kandinsky6/benchmark.py run --limit 1
uv run --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --no-sync \
  python tests/i2v-benchmark/kandinsky6/benchmark.py run
uv run --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --no-sync \
  python tests/i2v-benchmark/kandinsky6/benchmark.py review-sheet
uv run --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --no-sync \
  python tests/i2v-benchmark/kandinsky6/benchmark.py report
```

`--limit 1` 只做单条 smoke run。完整 9 条矩阵建议在确认结果目录为空后一次运行，避免 smoke run 和
正式批次分属两个进程、分别冷加载模型。重新开始时删除 `tests/i2v-benchmark/results/kandinsky6/`。

`preflight` 检查 29 项本地权重文件、Python/Diffusers、BF16/CUDA、ffprobe 和三张真实图片；不加载
pipeline、不生成视频。测试不联网下载模型。运行前应关闭其它 GPU 任务。输出包含每次视频、逐次遥测、
`runs.csv`、模型文件大小清单、盲评视频/评分 CSV 和 `report.md`，位于已忽略的 `results/kandinsky6/`。

在仓库根目录运行 `cd tests/i2v-benchmark && uv run --locked python -m unittest discover -s tests -v`，
可执行不加载模型的 benchmark 配置回归。验证完成后可以删除 `tests/i2v-benchmark/kandinsky6/` 和对应配置测试文件；
不需要改服务代码、`start.sh` 或质量门禁。
