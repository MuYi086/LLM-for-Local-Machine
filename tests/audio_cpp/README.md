# audio.cpp 模型 API 冒烟测试

十四个服务分别提供固定模型的真实 API 入口：

| 模型 | 独立测试目录 |
| --- | --- |
| Confucius4-TTS | [`tests/tts_confucius4-tts/`](../tts_confucius4-tts/) |
| IndexTTS 2.5 | [`tests/tts_indextts2.5/`](../tts_indextts2.5/) |
| MiDashengLM-Gen | [`tests/gen_midashenglm/`](../gen_midashenglm/) |
| Qwen3-ASR 1.7B | [`tests/asr_qwen3-1.7b/`](../asr_qwen3-1.7b/) |
| Qwen3-TTS Base | [`tests/tts_qwen3-1.7b-base/`](../tts_qwen3-1.7b-base/) |
| SeedVC | [`tests/vc_seed-vc/`](../vc_seed-vc/) |
| Stable Audio 3 Small SFX | [`tests/gen_stable-audio-3-small-sfx/`](../gen_stable-audio-3-small-sfx/) |
| Stable Audio 3 Medium | [`tests/Stable-Audio-3-Medium-GGUF/`](../Stable-Audio-3-Medium-GGUF/) |
| MOSS-SoundEffect v2 | [`tests/MOSS-SoundEffect-v2-GGUF/`](../MOSS-SoundEffect-v2-GGUF/) |
| VoxCPM2 | [`tests/tts_voxcpm2/`](../tts_voxcpm2/) |
| BS-RoFormer ep368 | [`tests/BS-RoFormer-ep368-GGUF/`](../BS-RoFormer-ep368-GGUF/) |
| Qwen3 ForcedAligner 0.6B | [`tests/Qwen3-ForcedAligner-0.6B-GGUF/`](../Qwen3-ForcedAligner-0.6B-GGUF/) |
| Mel-Band RoFormer | [`tests/Mel-Band-RoFormer-GGUF/`](../Mel-Band-RoFormer-GGUF/) |
| HTDemucs | [`tests/HTDemucs-GGUF/`](../HTDemucs-GGUF/) |

每个目录的 `generate_example.py` 只请求对应模型，复用这里的分块 multipart 上传、响应校验和原子保存逻辑。分离模型需要 `--input-audio` 提供 44.1 kHz WAV；ForcedAligner 还需要 `--transcript` 和 `--language`。多个模型请求不应并发执行；模型共享全局 GPU 锁。

先通过 `bash start.sh` 启动服务，再运行某个模型目录 README 中的命令。生成 WAV 和 ASR JSON 默认保存在 `storage/audio_cpp_tests/`。`bash scripts/quality_gate.sh` 运行无模型合同测试，不会请求模型。
