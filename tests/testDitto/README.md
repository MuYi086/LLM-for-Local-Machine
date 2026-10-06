# Ditto 数字人演示

启动 `8392` Ditto 服务后，进入本目录执行：

```bash
uv run ./testDitto.py
```

在仓库根目录也可以执行：

```bash
uv run tests/testDitto/testDitto.py
```

脚本参考 `tests/testSoulX_FlashHead/testFlashHead.py`，上传同目录的 `1.png` 和 `demo.wav`，
调用 `POST /v1/ditto/talkingHead`，将带声音的视频保存到 `dist/demo.mp4`。
输入文件默认根据脚本位置定位，可从其他目录调用。PEP 723 内联元数据声明
Python `3.12.13` 和 `httpx==0.28.1`，uv 会自动准备独立运行环境。

目录内 `demo.wav` 时长约 38.35 秒，采样率为 24 kHz。本次生成的 `dist/demo.mp4`
时长 38.35 秒、分辨率 1024×1024、帧率 25 fps，包含 H.264 视频与 AAC 声轨，
已通过 FFmpeg 完整解码检查。

默认 seed 为 42、motion diffusion 采样步数为 50、图片最大处理尺寸为 1920，
与 Ditto 服务默认参数一致。请求超时为 3600 秒，包含 GPU 排队与推理时间；
服务端另有自己的 worker 超时。成功时原子替换输出 MP4，失败时保留已有视频并清理临时文件。

```bash
uv run ./testDitto.py --sampling-timesteps 50 --seed 42 --output ./dist/demo.mp4
```

可用 `--image`、`--audio` 指定其他素材，`--server-url`、`--timeout` 调整连接设置，
`--max-size` 调整输入图片处理尺寸。服务安装和接口说明见 [Ditto 服务说明](../../ditto/README.md)。

这是手动执行的真实 GPU 演示，常规无模型质量门禁不会自动运行。
本目录图片、音频和 `dist/` 成品已加入 Git 忽略规则。
