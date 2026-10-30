# MiniMax H3 I2V 五场景扩展基准

该批次使用 `shared_conditions.json` 中独立维护的 5 张图、提示词和 seed 42，分辨率为
480×832、24 FPS。H3 使用原生 20 步、5 秒请求、124 帧，并封装原生音轨。
新结果写入 `tests/i2v-benchmark/results/extended5/minimax_h3/`，不覆盖三场景基准。

输入图已按主体位置裁切并缩放到 480×832；图片保存在 `assets/extended5/`。

| 场景 | 图像来源 | 作者 / 许可 |
|---|---|---|
| 行人 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Person_walking_across_street.jpg) | Keith Vaughton，CC BY 2.0 |
| 狗 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:RunningDog.jpg) | Billreef，公有领域 |
| 车辆 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Car_driving_along_a_road_surrounded_by_trees.jpg) | Etha / Pexels，CC0 |
| 海浪 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Waves_at_ocean_beach.jpg) | Jon Sullivan，公有领域 |
| 山谷云雾 | [Wikimedia Commons 文件页](https://commons.wikimedia.org/wiki/File:Landscape_with_clouds.jpg) | Aammell，CC0 |

## 执行

先确认 H3 服务在 `8392` 健康可用，并关闭其它 GPU 推理。从仓库根目录运行：

```bash
MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py preflight

MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py run

MINIMAX_H3_BENCHMARK_CONFIG=tests/i2v-benchmark/extended5/minimax_h3.json \
  uv run --project model-video/MiniMax-H3-16G-HiFi --no-sync \
  python tests/i2v-benchmark/minimax_h3/benchmark.py report
```
