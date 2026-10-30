# 扩展 I2V 五场景对比

该批次将同一张 480×832 输入图、同一段提示词和 seed 42 分别交给 Kandinsky 6 Lite 与 MiniMax H3。两个模型均按共享配置输出 480×832、24 FPS；Kandinsky 按蒸馏版 10 步生成 121 帧无音轨视频，H3 按原生 20 步、5 秒生成 124 帧带音轨视频。H3 服务默认 960×544，但本批 API 请求显式覆盖为共享的 480×832。

所有文件与报告都位于 `tests/i2v-benchmark/`；新结果写入 `results/extended5/`，不覆盖旧基准。图片源文件及裁切后的测试图片被 `.gitignore` 忽略。

场景提示词见 `kandinsky6.json`。两模型通过共享条件配置使用完全相同的图片、提示词与 seed。

| 场景 | 图像来源 | 作者 / 许可 |
|---|---|---|
| 行人 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Person_walking_across_street.jpg) | Keith Vaughton，CC BY 2.0 |
| 狗 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:RunningDog.jpg) | Billreef，公有领域 |
| 车辆 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Car_driving_along_a_road_surrounded_by_trees.jpg) | Etha / Pexels，CC0 |
| 海浪 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Waves_at_ocean_beach.jpg) | Jon Sullivan，公有领域 |
| 山谷云雾 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Landscape_with_clouds.jpg) | Aammell，CC0 |

源图已按主体位置裁切并缩放到 480×832；来源原图和处理后 PNG 存放于 `assets/extended5/`。

## 执行

从仓库根目录运行。先执行 Kandinsky `preflight` 和生成；H3 的前置检查还会确认 Kandinsky 对照视频全部完成。GPU 不并行：先完成 Kandinsky，再完成 H3。

```bash
KANDINSKY6_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/kandinsky6.json \
  model-video/Kandinsky-6.0-Lite-Distill-BF16/.venv/bin/python \
  tests/i2v-benchmark/kandinsky6/benchmark.py preflight

MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  python tests/i2v-benchmark/minimax_h3/benchmark.py preflight
```

结果和对照报告：

```bash
KANDINSKY6_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/kandinsky6.json \
  model-video/Kandinsky-6.0-Lite-Distill-BF16/.venv/bin/python \
  tests/i2v-benchmark/kandinsky6/benchmark.py run
KANDINSKY6_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/kandinsky6.json \
  model-video/Kandinsky-6.0-Lite-Distill-BF16/.venv/bin/python \
  tests/i2v-benchmark/kandinsky6/benchmark.py report

MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  python tests/i2v-benchmark/minimax_h3/benchmark.py run
MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  python tests/i2v-benchmark/minimax_h3/benchmark.py report
```
