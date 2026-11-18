# 任务

> 任务结构与 [实施计划](../../../../docs/superpowers/plans/2026-11-17-audio-cpp-separation-alignment.md)
> 一致；本 change 在 2026-11-17 由该计划落地，归档时按计划任务核对完成状态。

## 1. 注册候选与受监督服务

- [x] 1.1 在 `localai/manifests/audio-cpp-models.json` 登记四个固定候选（权重文件名、family/task、
      model ID、端口、路由、API 类型）。
- [x] 1.2 start.sh 纳入四个服务目录、主机、端口与路由，扩展启动合同测试
      （`tests/test_start_services.py`、`tests/test_audio_cpp_candidate_registration.py`）。

## 2. 扩展 LocalAI client 支持对齐与命名 stems

- [x] 2.1 读取并校验 transform 响应的 `X-Audio-Stems`（缺失、格式错误、重复名、越界引用拒绝）。
- [x] 2.2 强制对齐通过 transcription prompt 传递逐字稿，保留 verbose JSON 的词级时间字段。

## 3. 新增源分离与强制对齐服务合同

- [x] 3.1 三个分离服务：一次 transform 后按响应头清单取回全部 stem，返回 WAV stems ZIP；
      中途失败清理已暂存文件。
- [x] 3.2 对齐服务：无 `words` 时间戳返回上游合同错误；输入 WAV/采样率/声道不合法在推理前拒绝。

## 4. 四个真实 demo 与口径同步

- [x] 4.1 每个模型新增 `tests/<模型>/generate_example.py` 与 README，输出写入运行数据目录。
- [x] 4.2 `localai/manifests/capabilities.json` 同步并通过能力清单校验。
- [x] 4.3 README 与 AGENTS.md 服务口径同步至 15 个音频服务、21 个 HTTP 进程，
      并由 `scripts/check_service_inventory.py` 持续核对（2026-11-19 归档时补齐 AGENTS 漂移与
      quality_gate.sh 的 lint 缺口）。
