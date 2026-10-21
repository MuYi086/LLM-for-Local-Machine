# P5 Seed-VC LocalAI AudioTransform 候选

## 状态

- 代码：`implemented_unverified`，只允许显式选择 `UNITALE_SEED_VC_PROVIDER=localai`；默认仍使用 native worker。
- 固定能力依据：仓库外用户提供的 LocalAI v4.11.0 / audio.cpp 固定源码中，Seed-VC 文档和 session 支持 `vc`、`svc` 两类 `AudioTransform`，包含本报告列出的参数。
- E2/E3：未完成。LocalAI gRPC backend 已构建、安装并通过无模型启动检查；完整 SeedVC-MLX 目录、真实响应、音质、采样率和 GPU/backend 生命周期均未实测。
- 权重：没有下载。模型目录由用户手动准备，manager 仅记录本地目录树 hash，不证明上游来源。

## 已实现映射

| Unitale 字段/能力 | LocalAI AudioTransform 候选 | 说明 |
| --- | --- | --- |
| 源音频 | multipart `audio` | 先以 `AudioReferenceStore` 解析客户端逻辑路径 |
| 目标音色 | multipart `reference` | 保留现有目标参考引用语义 |
| `diffusion_steps` | `params[num_inference_steps]` | 固定 session 在 V1/V2 均解析 |
| `length_adjust` | `params[length_adjust]` | 固定 Seed-VC 请求选项 |
| `inference_cfg_rate`，VC | V2 intelligibility 和 similarity 两个 scale 都使用同一数值 | 保留 Unitale 单一滑杆；两项独立效果尚未对比 |
| `inference_cfg_rate`，SVC | V1 `inference_guidance_scale` | 单字段直接映射 |
| `f0_condition=false` | `task:vc`, `route:v2_vc` | 使用 VC 模型配置 |
| `f0_condition=true` | `task:svc`, `route:v1_svc`, `f0_condition=true` | 切换独立 SVC 模型配置 |
| `auto_f0_adjust` | `params[auto_f0_adjust]` | 只在 F0 模式传递 |
| `semi_tone_shift` | `params[semitone_shift]` | Unitale 历史字段名转换为固定引擎字段名，只在 F0 模式传递 |
| `fp16=false` | HTTP 422 | 候选 YAML 固定 `seed_vc.weight_type:f16`；不静默丢弃精度要求 |
| 返回值 | RIFF/WAVE `audio/wav` | 原样原子保存到既有 `storage/clone/` 输出目录 |

两个 LocalAI ID 必须注册，因为固定 backend 的 task 在模型配置中固定：
`unitale.seed-vc.vc.f16` 与 `unitale.seed-vc.svc.f16`。它们复用用户手动准备的
`$UNITALE_LOCALAI_MODELS_DIR/SeedVC-MLX/` 目录。注册命令仅复制固定 YAML 并保存本地 tree hash：

```bash
python scripts/localai/manager.py register-seed-vc-candidate
python scripts/localai/manager.py verify-seed-vc-candidate
```

## 尚待完成

- 完成 P7 worker bridge 注册和 Seed-VC AudioTransform 路由；对 vc、svc 的真实响应、音质、采样率及 GPU/backend 生命周期仍待用户准备权重后实测。
- 用户手动准备完整 SeedVC-MLX 模型目录后登记；上游模型包 revision/SHA-256 目前没有被固定源码提供，因此只能记录本机 tree hash。
- 对 vc、svc 分别运行真实请求，核对 WAV 解码、采样率、单声道行为、音质和请求参数。
- 用真实 backend 检查 `backend/shutdown`、共享 GPU 锁释放、显存回收、超时及故障 quarantine。
- 根据两种路线的真实结果决定是否保留映射和是否进入 E2/E3；native 回滚仍可用。
