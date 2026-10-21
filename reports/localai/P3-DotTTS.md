# P3：dots.tts-soar LocalAI 代码试点

- 状态：`implemented_unverified`
- 服务：`dots_tts_soar` / 8324，`POST /v1/dotsTTS/clone`
- 默认 provider：`native`
- LocalAI model ID：`unitale.dots-tts-soar.q8-0`
- audio.cpp 配置：`family:dots_tts`、`task:tts`、CUDA、`dots_tts.reference_cache_slots:0`
- 候选权重：`audio-cpp/audio.cpp-gguf` revision `83c5d96c03023ff5a7712570d057ce26c8769f98`，
  `DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf`
- 权重由用户手动准备；本次没有下载模型。
- 证据：字段转换、候选注册、服务 route、共享锁/shutdown 编排由 mock 测试覆盖；没有 LocalAI/backend/模型实测，非 E2/E3。

## 已实现映射

| Unitale 字段 | LocalAI/audio.cpp | 当前限制 |
| --- | --- | --- |
| `text` | `input` | 只接受一段 native 文本；会由原生服务分片的请求返回 422。请求时设 `text_chunk_size` 高于文本长度，关闭框架二次切段。 |
| `audio_path` | 通过 `AudioReferenceStore` 校验后传入 `voice` | 不接受客户端任意服务器路径；同内容设计音色仍引用 timbre WAV。 |
| `prompt_text` | `params.reference_text` | 显式文本优先，缺省使用原 sidecar。 |
| `language` | `language` | 保留服务原值；实际语种行为待 smoke。 |
| `seed`、`guidance_scale`、`speaker_scale` | 同名 DotTTS request params | 固定音频后端参数，mock 已核对。 |
| `num_steps`、`ode_method` | `num_inference_steps`、`sampler_mode` | 使用固定 audio.cpp 文档参数名。 |
| `max_generate_length` | `max_tokens` | 映射已 mock 覆盖。 |
| `template_name` | `template_name` | 当前 schema 的四个模板候选；Edit 不属于 SOAR endpoint。 |
| `max_chars_per_chunk`、`pause_ms` | 单段 preflight | 超过 native 默认或请求设置的切段长度时拒绝；多段和停顿仍走 native。 |
| `precision` | 不支持 | Q8 精度由候选 GGUF 固定；显式请求返回 422。 |
| `normalize_text=true`、`profile_inference=true` | 不支持 | 固定文档没有相同请求选项，显式启用返回 422。 |
| 输出 | WAV，要求 48 kHz | 请求携带 `sample_rate=48000`；输出声道、采样率及有效内容尚未用真实 backend 检验。 |

LocalAI 请求经过现有 `execute_localai_operation`，使用与 native 服务同一 GPU 锁、请求隔离标记、`/backend/shutdown` 和 `CUDA_RELEASE_DELAY`。失败停止时保留隔离标记；LocalAI 仍可按显式 provider 单服务切回 native。

## 手动制品准备

LocalAI models 目录下的期望位置为：

```text
DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf
```

权重手动放好后注册：

```bash
python3 scripts/localai/manager.py register-dots-tts-soar-candidate
```

注册会原子写入固定 YAML 和本地 size/SHA-256 记录，`start.sh` preflight 再核对文件、配置与本地 hash。该本机记录只检测注册之后的内容变化，不验证文件来自上游指定 commit。

## 制品与运行边界

固定 audio.cpp docs 选择独立 SOAR Q8 GGUF，并文档化 `reference_text`、`language`、`guidance_scale`、
`max_tokens`、`seed`、模板、采样器、步数及 speaker scale。HF revision 页面列出了目标 Q8 文件，但该 revision 的 README
量化表述没有同步列出 SOAR Q8；上游 LFS SHA-256 / 精确字节长度本次未能核实，因此版本清单将 hash 留空并保持 E0。
注册后计算的本地 hash 不等于上游 hash。该模型包说明中的 Apache-2.0 也不替代原始模型许可检查。

## 下一步验收

1. 从固定 HF revision 手动准备 Q8 文件；补齐权重真实大小和上游 LFS SHA-256 后再更新版本清单。
2. 等待 LocalAI audio-cpp gRPC backend 构建完成并安装固定文件树。
3. 运行模型加载、clone WAV、sidecar/显式转写、中文/英文、输出 48 kHz 单声道和 `/backend/shutdown` smoke。
4. 对比 native 与 LocalAI 的文本准确、盲听、延迟、峰值显存、并发排队、失败回收和进程退出。
5. 未达到 E3 前保持 native 默认；长文本、多段停顿与未映射选项继续走 native。
