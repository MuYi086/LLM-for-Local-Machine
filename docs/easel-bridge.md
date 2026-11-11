# Easel 本地模型网关

`easel_bridge` 提供 OpenAI-compatible API，把 Easel 的图片、语音请求转发到本仓库已经运行的 MOSS-Image-SFT 与
VoxCPM2 服务。它也可以把 Qwen3.6 暴露为兼容聊天 API，但不会更改 Easel 当前选择的 Codex 执行引擎。
所有推理继续由现有模型 API、共享 GPU 锁和 sleep-idle 机制执行。

## 接口

网关默认只监听 `127.0.0.1:8500`，所有 `/v1` 路由都要求相同的 `Authorization: Bearer <key>`：

| 路由 | 模型 ID | 上游与限制 |
| --- | --- | --- |
| `GET /v1/models` | — | 返回网关支持的模型列表 |
| `POST /v1/images/generations` | `MOSS-Image-SFT-Q8_0` | 映射到 MOSS；`n` 最多 4，每张依次生成；尺寸使用 `WIDTHxHEIGHT`，例如 `1024x1024` |
| `POST /v1/audio/speech` | `OpenBMB/VoxCPM2-GGUF` | 映射到 VoxCPM2 multipart `text`/`instruction`；接受 `voice=alloy`，仅支持默认 `speed=1`，输出 WAV 或 MP3 |
| `POST /v1/chat/completions` | `Qwen3.6-35B-A3B-MTP` | 原样转发 JSON 与 SSE 流到现有 Qwen 适配器 |

网关不会向上游转发客户端 API Key。上游 URL 只能由服务端环境变量指定，HTTP 重定向关闭。请求体默认上限为 12 MiB，
响应体上限为 64 MiB；上游访问超时默认为 900 秒。MP3 由本机 FFmpeg 从 VoxCPM2 的 WAV 响应转换，不写临时音频文件。

## 配置和启动

首次部署时生成 Key、创建仅当前用户可读的本地配置：

```bash
cp .env.easel-bridge.example .env.easel-bridge
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
chmod 600 .env.easel-bridge
```

`start.sh` 的 MOSS、Qwen3.6 与 VoxCPM2 API 默认监听 `127.0.0.1`，其余已有服务配置不变。若此前已经启动了旧版本，
需在维护窗口重新启动模型栈后，新的 loopback 默认绑定才会生效。确保 WSL 中的 FFmpeg 可运行：

```bash
ffmpeg -version
```

把生成的随机值写入 `.env.easel-bridge` 的 `EASEL_BRIDGE_API_KEY` 行。该文件被 Git 忽略。
已有配置时保留原 Key，不要重新复制模板。之后每次只需从仓库根目录运行：

```bash
bash start.sh
```

`start.sh` 自动读取 `.env.easel-bridge`，在启动模型前拒绝缺失、过短或占位符 Key，
检查 8500 端口，并将网关纳入进程监控。Ctrl+C 或任一服务退出时，网关与模型服务一起停止。
`EASEL_BRIDGE_ENV_FILE` 可指定其他配置文件，`EASEL_BRIDGE_PORT` 可覆盖默认端口；配置文件中的值覆盖同名环境变量。
网关固定绑定 `127.0.0.1`。日志只打印地址，不打印 Key。
上游 URL 未显式配置时，`start.sh` 使用 `127.0.0.1` 并跟随 Qwen、MOSS、VoxCPM2 的受管 API 端口。
需要其他绑定地址时，或旧配置文件已经写死 URL 时，通过对应 `EASEL_BRIDGE_*_URL` 显式配置。

排查网关时仍可单独启动；此时不要同时用 `start.sh` 启动同端口网关：

```bash
set -a
source .env.easel-bridge
set +a
uv run --project main --locked python -m uvicorn easel_bridge.app:app \
  --app-dir . --host 127.0.0.1 --port 8500
```

先在 WSL 中验证 Key：

```bash
curl -i http://127.0.0.1:8500/v1/models
curl http://127.0.0.1:8500/v1/models \
  -H "Authorization: Bearer $EASEL_BRIDGE_API_KEY"
```

第一条应返回 `401`，第二条应返回 MOSS、VoxCPM2、Qwen3.6 三个模型 ID。Easel 网页由 Windows 浏览器打开时，WSL2 通常可通过
Windows 的 `localhost` 转发访问 WSL 端口；如果 Easel 服务在 Windows 原生环境运行，应从 Easel 所在进程实际发起连接测试，
并保持桥接服务只监听 WSL loopback，不要为方便而将 GPU 上游端口重新绑定到所有网卡。

## OpenCode 一次配置

OpenCode v2 的本地 Provider 使用 `providers`、`package`、`settings`，Base URL 为
`http://127.0.0.1:8500/v1`，模型 ID 为 `Qwen3.6-35B-A3B-MTP`，上下文设置为 32768。
不要直接连接 llama.cpp 后端，否则会绕过上游 API 的共享 GPU 锁。

在运行 OpenCode 的 WSL 用户下，把 Key 保存为独立文件，供新终端和后台服务读取：

```bash
set -a
source .env.easel-bridge
set +a
mkdir -p ~/.config/opencode
(umask 077; printf '%s' "$EASEL_BRIDGE_API_KEY" > ~/.config/opencode/easel-bridge-api-key)
chmod 600 ~/.config/opencode/easel-bridge-api-key
```

将已有 `opencode.json` 的 `providers.local-qwen.settings.apiKey` 改为文件引用：

```json
"apiKey": "{file:~/.config/opencode/easel-bridge-api-key}"
```

该步骤只做一次；更换网关 Key 时再同步密钥文件。以后运行 `bash start.sh`，等待服务就绪，
另开终端进入业务项目后运行 `opencode --standalone`，不再需要 `source` 或额外启动网关。
`--standalone` 让 OpenCode 使用本次进程的配置；退出模型栈不会关闭另一个终端中的 OpenCode 界面。
Windows 原生 OpenCode 使用 Windows 用户目录中的配置与密钥文件，需要单独配置。

## Easel 填写项

在 Easel 的图片模型配置中选择 OpenAI-compatible：

| 字段 | 值 |
| --- | --- |
| Base URL | `http://127.0.0.1:8500/v1` |
| API Key | `.env.easel-bridge` 中的 `EASEL_BRIDGE_API_KEY` |
| Model | `MOSS-Image-SFT-Q8_0` |

若通过 Easel 项目的 `.env` 配置图片脚本，可合并以下字段（API Key 使用同一份随机值）：

```dotenv
IMG_BASE_URL=http://127.0.0.1:8500/v1
IMG_MODEL=MOSS-Image-SFT-Q8_0
IMG_API_KEY=填入EASEL_BRIDGE_API_KEY的值
```

在语音模型配置中选择 `openai-compatible`：

| 字段 | 值 |
| --- | --- |
| Base URL | `http://127.0.0.1:8500/v1` |
| API Key | 同一份 `EASEL_BRIDGE_API_KEY` |
| Model | `OpenBMB/VoxCPM2-GGUF` |
| Voice | `alloy`（兼容占位值；实际音色描述可走 instructions） |
| Instruction mode | `field` |

对应的 Easel `.env` 字段为：

```dotenv
VOICE_PROVIDER=openai-compatible
VOICE_BASE_URL=http://127.0.0.1:8500/v1
VOICE_API_KEY=填入EASEL_BRIDGE_API_KEY的值
VOICE_MODEL=OpenBMB/VoxCPM2-GGUF
VOICE_NARRATOR_VOICE_ID=alloy
VOICE_INSTRUCT_MODE=field
```

Easel 的音频脚本默认请求 MP3，因此系统需要安装 FFmpeg；也可以使用 WAV 输出。VoxCPM2 当前不接受速度调节，Easel 侧将速度保持为 `1`。
当前 Codex 运行时可以继续作为“对话与脚本”的执行引擎；图片和语音调用单独走以上模型配置。若以后希望直接调用本地 Qwen，
聊天 Provider 使用相同的 Base URL 与 Key，模型 ID 为 `Qwen3.6-35B-A3B-MTP`。

## 故障定位

- `401`：检查 API Key 是否一致，以及 shell 是否通过 `source .env.easel-bridge` 加载。
- `502`：确认 `bash start.sh` 中对应的模型服务健康，并查看网关与上游日志。
- `503`：MOSS/VoxCPM2 上游返回的忙碌或资源错误会保留状态码；不要通过直接访问后端来绕开 GPU 锁。
- MP3 转换报错：在网关所在 WSL 环境安装 FFmpeg，或设置 `EASEL_BRIDGE_FFMPEG_BIN` 指向其可执行文件。
- Windows Easel 无法连接：分别在 WSL 和 Easel 运行环境测试 `localhost:8500`；不要将上游模型端口开放到局域网。

网关测试不加载模型、不使用 CUDA，也不访问外部服务：

```bash
uv run --project qa --locked python -m unittest discover -s tests -p 'test_easel_bridge.py' -v
```
