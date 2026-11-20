# 源码归属矩阵

本矩阵把仓库全部源码范围映射到 [主规格](specs/project-overview/spec.md) 的能力域，
用于防止行为基线与源码脱节。目录归属不等于逐行行为、全部 API 分支和真实模型验证均已覆盖；
源码与清单是核对依据，主规格与 change 需要随行为变化更新。

## 代码目录归属

| 源码范围 | 主规格及内容 |
| --- | --- |
| main/main.py、gpu_runtime.py、spatial_*.py、steam_audio_render.py、storage_maintenance.py、check_ports.py、render_progress.py | control-plane-and-spatial-render：健康/控制、CPU 导出、空间渲染编排、按 job_id 进度轮询与不加载模型边界；check_ports 同时支撑 service-lifecycle-and-startup 的端口预检 |
| llm_local_runtime/src/llm_local_runtime/storage.py | storage-retention-and-upload：流式暂存、SHA-256、原子提交与 full_path 逻辑标识 |
| llm_local_runtime/src/llm_local_runtime/gpu.py | gpu-lock-and-quarantine：跨进程 GPU 锁；storage-retention-and-upload 声明该包的依赖边界 |
| llm_local_runtime/src/llm_local_runtime/telemetry*.py | model-telemetry：采集、资源指标、简报与 CLI；quality-gate-and-testing 引用其 `end --attach-result --brief` 归档出口 |
| inference_gateway/src/inference_gateway/execution.py、localai_client.py、config.py、errors.py | gpu-lock-and-quarantine：LocalAI provider 的锁执行、backend shutdown、启动配置校验与稳定错误分类；audio-cpp-services 引用其推理路径 |
| inference_gateway/src/inference_gateway/sdcpp/ | image-sdcpp-services：按请求启动的 sd.cpp CLI provider、worker 与进程核验 |
| model-audio/audio_cpp_runtime/ | audio-cpp-services：十四个服务共享的轻量 uv 项目与合同实现（service.py、specs.py） |
| model-audio/design_qwen3-tts-1.7b-voicedesign/ | audio-cpp-services：VoiceDesign 独立 uv 项目（8301） |
| model-audio/tts_confucius4-tts/、tts_indextts2.5/、tts_qwen3-1.7b-base/ | audio-cpp-services：WAV 参考克隆合同 |
| model-audio/asr_qwen3-1.7b/ | audio-cpp-services：转写 JSON 合同 |
| model-audio/gen_midashenglm/、gen_stable-audio-3-small-sfx/、Stable-Audio-3-Medium-GGUF/、MOSS-SoundEffect-v2-GGUF/ | audio-cpp-services：文本提示音频/音效生成合同 |
| model-audio/vc_seed-vc/ | audio-cpp-services：音色转换（源音频 + 目标 WAV） |
| model-audio/tts_voxcpm2/ | audio-cpp-services：VoxCPM2 可选参考克隆 |
| model-audio/BS-RoFormer-ep368-GGUF/、Mel-Band-RoFormer-GGUF/、HTDemucs-GGUF/ | audio-cpp-services：源分离 `X-Audio-Stems` 一次推理合同（8344/8346/8347） |
| model-audio/Qwen3-ForcedAligner-0.6B-GGUF/ | audio-cpp-services：强制对齐合同（8345） |
| model-audio/spatial_steam-audio/ | control-plane-and-spatial-render：CPU renderer 本体，不取 GPU 锁 |
| model-image/MOSS-Image-SFT-Q8_0/、FLUX-3-Image-Klein-GGUF-Q8/ | image-sdcpp-services：8501/8502 图片 API 入口（main.py、service.py） |
| easel_bridge/ | easel-bridge-gateway：8500 loopback 网关与白名单转发 |
| scripts/localai/ | service-lifecycle-and-startup、gpu-lock-and-quarantine：宿主管理、预检与 recover_quarantine 唯一恢复入口；manifest-version-pinning：候选受控注册与 task enum 补丁前置 |
| scripts/sdcpp/ | image-sdcpp-services：本机状态、预检与生成；manifest-version-pinning：sd.cpp 摘要与版本锁定核验 |
| scripts/dagu/ | dagu-job-worker：可选 Dagu 调度层的 HTTP worker、作业状态机与无模型联调；不改变 service-lifecycle-and-startup 的服务边界 |
| scripts/quality_gate.sh、check_service_inventory.py | quality-gate-and-testing：无模型门禁与服务口径校验 |
| scripts/model_telemetry.py | model-telemetry：begin/end/report 与 `--attach-result` 写回；quality-gate-and-testing 引用简报归档 |
| scripts/build_steam_audio_renderer.{sh,ps1}、test_steam_audio_renderer_core.sh | control-plane-and-spatial-render：Steam Audio renderer 的构建与 core 无模型测试（测试同时进入 quality_gate.sh） |
| qa/ | quality-gate-and-testing：无模型回归 uv 项目与 Ruff 环境 |
| tests/ | quality-gate-and-testing：无模型回归；tests/test_dagu_jobs.py、tests/dagu_smoke.py 归 dagu-job-worker，tests/test_model_telemetry.py、test_telemetry_brief.py 归 model-telemetry；tests/audio_cpp/、tests/MOSS-Image-SFT-Q8_0/、tests/FLUX-3-Image-Klein-GGUF-Q8/ 为真实样例入口，tests/model_web_comparison/ 为历史对比入口，都不是无模型回归的一部分 |
| start.sh | service-lifecycle-and-startup：预检、进程组清理与启动日志同步；服务口径同时受 quality-gate-and-testing 门禁核对 |
| localai/manifests/、sdcpp/manifests/ | project-overview：服务事实唯一权威；manifest-version-pinning：版本/commit 锁定、指纹核验与 VoiceDesign 候选 YAML 占位符合同；audio.cpp 合同变化须同步 capabilities.json |
| config/model-telemetry.env | model-telemetry：性能记录开关与采样间隔配置 |
| config/（其他） | 运行配置，不视为独立业务能力 |
| dagu/ | dagu-job-worker：队列并发（gpu-heavy=1、cpu-post=3）与 MOSS、VoxCPM2 DAG 定义 |
| localai_worker_bridge/（已删除） | P7 阶段 worker bridge 实验残留（仅剩虚拟环境与缓存），已清理；历史结论见 reports/localai/P7-worker-bridge.md |
| reports/、测试结果/、空间音频讨论/ | 历史记录与调研资料，不据此推断当前接口（project-overview 声明该边界）；测试结果/用于基准归档，不作为能力口径 |
| model-video/、model-text/training/、model-text/scripts/training/ | 退役或实验性代码：视频模型未接入启动项；LoRA 实验不自动获取服务 GPU 锁 |
| storage/ | 运行数据目录：保留策略与清理边界见 storage-retention-and-upload；不由代码清理 |
| docs/ | 机制文档按域对应：dev-workflow.md、agent-memory/ 归 engineering-collaboration；model-telemetry.md 归 model-telemetry；dagu.md 归 dagu-job-worker；localai-audiocpp-upgrade.md 归 manifest-version-pinning；其余机制文档归对应服务域；model-audio/README.md、model-image/README.md、model-text/README.md 为各目录说明文档；行为基线仍以 openspec/specs/ 为准 |
| docs/superpowers/ | 既有 SDD 历史记录，只描述当时状态，不作为当前规格依据 |

## 非运行源码和外部边界

| 范围 | 处理方式 |
| --- | --- |
| openspec/、README.md、AGENTS.md、CLAUDE.md、CONSTITUTION.md | 协作规则与规格基线；AGENTS.md 的协作/需求分级/验证纪律归 engineering-collaboration，音乐后处理规则归 music-postprocessing；主规格描述当前工作区，未归档 change 单独跟踪 |
| task1.md、task3.md、task10.md、audio-cpp分析.md、stable-diffusion-cpp分析.md、stable-diffusion相关模型推理满足本机配置的筛选.md、stable-diffusion.cpp与WanGP视频及MV工作流对比报告.md、LongCat-Video-Avatar-1.5量化搭配WanGP最佳实践.md | 历史需求与调研输入；以实现及对应 change 为准，不自动升级为产品行为 |
| .agents/、.github/ | AGENTS.md 声明的六个音乐后处理 skill 在 `.agents/skills/` 无定义，归 music-postprocessing 并标注待对齐；其余为 agent 工具集成与 CI，不证明应用已运行 |
| seed-vc/、configuration/（已删除） | 残留/空目录，已清理 |
| .env.easel-bridge | 真实网关 Key（至少 32 字节随机值）：不读取内容、不提交、不入规格；仅 `.env.easel-bridge.example` 模板入库并由 easel-bridge-gateway 规格引用 |
| llm_local_runtime/uv.lock、inference_gateway/pyproject.toml | 前者存在但未纳入 `quality_gate.sh` 的四个项目锁校验，后者无锁文件；作为未决事项记录在 quality-gate-and-testing 规格 |
| LICENSE、.gitignore、ruff.toml | LICENSE 为法律文件；`.gitignore` 定义收录边界（权重、密钥、缓存、生成产物不入库）；`ruff.toml` 是 quality-gate-and-testing 的格式口径配置 |
| 权重、仓库外二进制与 Steam Audio SDK | 仓库外资产，不提交；文档只描述加载与核验方式，版本锁定见 manifest-version-pinning |

## 已确认清单与未完成项

- 十九个 HTTP 进程、十五个音频模型服务（十四个 audio.cpp + VoiceDesign）由
  `scripts/check_service_inventory.py` 对照 `localai/manifests/audio-cpp-models.json` 持续核对。
- 主规格共 14 份：project-overview 总览 + 13 个能力域（原有九个服务/机制域加 model-telemetry、
   dagu-job-worker、music-postprocessing、manifest-version-pinning、engineering-collaboration）。
- 真实模型验证按任务单独串行执行，通过与否记录在对应 change 的 verification.md；
  归档 change 见 [changes/archive](changes/archive/)。
- diffuse、非 none 遮挡与几何反射未实现；视频模型未接入；两者都不写入可用能力口径。
- AGENTS.md 声明的六个音乐后处理 skill 在 `.agents/skills/` 均无定义文件，属「待对齐」未决事项，
  记录于 [music-postprocessing 规格](specs/music-postprocessing/spec.md)，不写成可用工具能力。
