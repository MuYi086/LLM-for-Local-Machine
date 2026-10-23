# LLM Local Backend Steam Audio Renderer

这是独立于模型 worker 的 CPU 命令行渲染器。它读取 48 kHz 规范 WAV 与 Render Manifest v1，
为每个点声源创建独立的 Steam Audio Direct/Binaural Effect，逐帧应用距离、空气吸收、HRTF
及简单线性轨迹，再流式混入 48 kHz 双声道 float WAV pre-master。BGM 和需要保留的环境床不做
HRTF；最终 loudnorm、true peak 和 WAV/MP3 编码由 8300 控制面完成。

## 外部依赖

- Steam Audio SDK 4.8.1（仓库不提交 SDK、`phonon.dll` 或 `libphonon.so`）；
- CMake 3.17+ 和支持 C++17 的编译器；
- 运行控制面时还需要 FFmpeg。

从 Steam Audio 官方发布页取得 SDK 并解压到 `$HOME/.local/share/steam-audio/4.8.1`。
Linux 启动脚本、控制面与构建脚本默认使用该路径，不需要每次设置 `STEAM_AUDIO_SDK_DIR`；其他安装位置可覆盖此变量。SDK 目录应包含
`include/phonon.h`，以及 Linux 的 `lib/linux-x64/libphonon.so` 或 Windows 的
`lib/windows-x64/phonon.dll`。

Linux：

```bash
bash scripts/build_steam_audio_renderer.sh
export STEAM_AUDIO_RENDERER_BIN="$PWD/model-audio/spatial_steam-audio/build/steam-audio-render"
```

Windows PowerShell：

```powershell
$env:STEAM_AUDIO_SDK_DIR = "C:\sdk\steamaudio"
.\scripts\build_steam_audio_renderer.ps1
$env:STEAM_AUDIO_RENDERER_BIN = "$PWD\model-audio\spatial_steam-audio\build\Release\steam-audio-render.exe"
$env:PATH = "$env:STEAM_AUDIO_SDK_DIR\lib\windows-x64;$env:PATH"
```

构建脚本只读取本地 SDK，不下载依赖。Linux 可通过 RPATH 找到 SDK 动态库；Windows 必须让
`phonon.dll` 位于 renderer 同目录或 `PATH` 中。

## CLI 合同

单文件 PoC：

```bash
steam-audio-render \
  --input samples/whisper.wav \
  --output out/whisper-spatial.wav \
  --position '0.42,0,0.42' \
  --listener '0,0,0' \
  --sample-rate 48000 \
  --hrtf default \
  --interpolation bilinear
```

生产批量渲染：

```bash
steam-audio-render \
  --manifest runtime/render-manifest.json \
  --assets-dir runtime/assets \
  --output runtime/pre-master.wav \
  --threads 4
```

成功时 stdout 只有一行 JSON；诊断日志写 stderr。退出码为 `2`（参数/Manifest）、`3`（WAV
输入输出）、`4`（Steam Audio）或 `5`（其他未分类输出异常）。输入资产不能是符号链接，也不能越过
`--assets-dir`。renderer 会补零处理最后一个不足 frame size 的输入块，并把成品严格裁剪到
Manifest 总时间线。

当前阶段使用内置或显式 SOFA HRTF、距离和空气吸收，不启用 reflections、occlusion、
transmission 或 GPU 射线追踪。语义字段已经保留，后续能力必须通过独立开关和测试增加。
