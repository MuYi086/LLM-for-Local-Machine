# SoulX-FlashHead 数字人演示

`demo.wav` 使用 `1.wav` 的音色和 `朗读文本.md` 的天气播报文本，通过
FireRedTTS3 Base 的 `8325` 克隆接口生成，时长约 38.35 秒，采样率 24 kHz。
参考音频对应的文本先通过 Qwen3-ASR 转写，再传给克隆接口。

启动 `8391` SoulX 服务后，进入本目录执行：

```bash
uv run ./testFlashHead.py
```

脚本通过 PEP 723 内联元数据声明 Python `3.12.13` 和 `httpx==0.28.1`，
uv 会自动准备独立运行环境。在仓库根目录也可以执行
`uv run tests/testSoulX_FlashHead/testFlashHead.py`。

脚本上传同目录的 `1.png` 和已有的 `demo.wav`，调用
`POST /v1/soulX/flashHead`，将带声音的视频保存到 `dist/demo.mp4`。
本次生成的 MP4 时长 38.35 秒、分辨率 512×512、帧率 25 fps，包含 H.264 视频和 AAC 音轨，
已通过 FFmpeg 完整解码检查。
输入文件默认根据脚本位置定位，因此也可以从其他目录调用。
默认使用 Lite、seed 42，关闭人脸裁剪；适用于当前卡通图片的直接缩放和中心裁剪。
重新运行成功时会替换 `dist/demo.mp4`，失败时保留已有视频并清理临时文件。

```bash
# 切换 Pro，或指定其他输入与输出
uv run ./testFlashHead.py --model-type pro --output ./dist/demo-pro.mp4
```

服务源码、权重及环境变量见 [SoulX 服务说明](../../SoulX-FlashHead-1_3B/README.md)。
此脚本是手动执行的真实 GPU 演示，常规质量门禁不会执行推理。
本目录图片、音频和 `dist/` 成品已加入 Git 忽略规则。
