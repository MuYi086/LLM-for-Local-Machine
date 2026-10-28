# IndexTTS-2.5

独立 FastAPI 服务使用官方 IndexTTS-2.5 `v2.5.0` 推理源码，为参考音频生成多语言语音。
API 进程不导入 Torch；每次生成在独立 worker 中加载模型，并通过共享 GPU 文件锁串行执行。

## 本地准备

主模型默认读取 `$HF_MIRROR_DIR/IndexTeam/IndexTTS-2.5`。运行还需要本机已有的
Wav2Vec2-BERT、CAMPPlus 和 BigVGAN 资源，默认位置与 Confucius4-TTS 共用：

| 资源 | 默认路径 | 覆盖变量 |
| --- | --- | --- |
| IndexTTS-2.5 | `$HF_MIRROR_DIR/IndexTeam/IndexTTS-2.5` | `INDEXTTS25_MODEL_DIR` |
| 官方推理源码 | `$HOME/LLM-depency/index-tts-2.5.0`，检出 `v2.5.0` | `INDEXTTS25_CODE_PATH` |
| Wav2Vec2-BERT | `$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0` | `INDEXTTS25_W2V_BERT_MODEL_DIR` |
| CAMPPlus | `$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin` | `INDEXTTS25_CAMPPLUS_PATH` |
| BigVGAN | `$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x` | `INDEXTTS25_BIGVGAN_MODEL_DIR` |

在仓库外准备固定版本源码：

```bash
git clone --depth 1 --branch v2.5.0 \
  https://github.com/index-tts/index-tts.git "$HOME/LLM-depency/index-tts-2.5.0"
uv sync --project model-audio/indexTTS-2.5 --locked
```

官方项目的打包元数据限定 Python `<3.12`。本服务遵循仓库的 Python `3.12.13` 部署环境，
不安装官方包，而是从上述源码目录导入 `indextts.infer_v2_5`；只声明该推理入口所需的运行依赖。
请求时设置 Hugging Face 与 ModelScope 离线模式，不下载缺失模型。`GET /v1/health` 会分别报告
源码、主模型、辅助模型、Python 依赖和 CUDA 是否就绪。

## API

先上传参考音频并记下返回的逻辑 `audio_path`：

```bash
curl -fsS http://127.0.0.1:8323/v1/upload_audio \
  -F 'audio=@reference.wav;type=audio/wav' \
  -F 'full_path=reference.wav'
```

再提交 JSON，响应为 `audio/wav`，生成文件保存到 `storage/clone/`（可用
`INDEXTTS25_OUTPUT_DIR` 覆盖）：

```bash
curl -fsS http://127.0.0.1:8323/v1/index-team/indexTTS-2.5 \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","audio_path":"reference.wav","lang":"ZH"}' \
  -o index-tts-25.wav
```

`lang` 支持 `ZH`、`EN`、`JA`、`ES`、`AR`；`duration_factor` 取值 0.5–2.0。可选
`emo_vector` 是 8 个 `[0, 1.2]` 数值，顺序为 happy、angry、sad、afraid、disgusted、
melancholic、surprised、calm；`emo_alpha` 取值 0–1。情感文本模型不加载。

默认服务端口为 `8323`，可通过 `INDEXTTS25_HOST`、`INDEXTTS25_PORT` 和
`INDEXTTS25_REQUEST_TIMEOUT` 覆盖。源权重保持只读；worker 的模型符号链接视图和缓存位于
`storage/.cache/runtime/index_tts25_worker/`，请求结束后清理。
