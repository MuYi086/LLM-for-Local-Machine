# stable-diffusion.cpp 通用执行层

用户已确认专用 sd.cpp backend、共享 GPU 锁、CLI 首版与模型后选的架构。本次交付先完成可调用的执行层，
不把尚未选择或验证的模型声明为当前 HTTP 服务，也不恢复退役视频代码。

## 方案与边界

采用 stdlib-only `inference_gateway.sdcpp` 与运维入口 `scripts/sdcpp/manager.py`。
与常驻 sd-server、Python 原生绑定相比，按请求启动 CLI 更容易确认进程退出；代价是每次重新加载权重。
以后可以在相同 provider 边界增加 server 实现。当前工作区干净，使用独立 feature branch 在当前目录实施。

默认模型清单为空。操作者明确配置外部 sd-cli、SHA-256、GPU_LOCK_FILE 和本地清单后，才能执行生成。
源版本固定为上游 `a3c49b13be5a64a0b4da2b576af3a17e8ae1ccbf`，二进制来源由操作者校验；
不能由文件摘要推断二进制一定由指定 commit 编译。权重、第三方源码和运行产物均不提交。

## 合同

- 清单使用 schema_version=1，模型按 ID 唯一；任务限定 text-to-image、image-to-image、text-to-video、
  image-to-video。每个模型配置组件、生成默认值、尺寸/步数/帧数限制和明确验证的任务。
- 参数只通过结构化字段映射到固定 argv，不提供额外任意 flags；预检在锁外完成。
- 队列默认 900 秒、执行默认 3600 秒、回收默认 30 秒，均要求有限正数；CUDA_RELEASE_DELAY 默认 2 秒，有限非负。
- 使用唯一共享 GPU_LOCK_FILE，持锁覆盖启动、生成、进程组退出确认和释放延迟。
- 启动前原子写 quarantine。子进程先等待 stdin 启动许可，父进程持久化 PID/PGID、Linux starttime 后才许可 exec，
  关闭 Popen 与记录身份之间的崩溃窗口。父进程消失且未许可时，子进程因 EOF 退出，不加载权重。
- 失败、超时、取消和后代残留均先回收整个新 session 进程组。停止无法确认或 marker 被替换则保留隔离。
  恢复仅由既有 recover_quarantine.py 分派；检查 PID 复用，拒绝误杀。
- 正常执行使用 Linux waitid(WNOWAIT) 观察退出，保留根进程直到进程组完成回收，避免 PGID 被复用。
- 输出使用请求 UUID 独立目录，成功后原子改名发布；失败只清理当前请求。未确认进程停止时保留目录。
  单文件图片为 PNG，视频为 AVI；不增加编码模型或额外 GPU 后处理。
- 视频 frames 至少为 2；固定版本单帧 vid_gen 实际输出 PNG，因此必须在预检时拒绝。
- 运维 CLI 提供 status、validate、generate；status 不执行二进制、不读权重内容、不初始化 CUDA。
  CLI 的路径来自受信任本机操作者，不是面向远程用户的文件读取接口。
- 本阶段不增加 HTTP 端口，不修改 start.sh 的十四进程合同。未来 HTTP 入口需异步任务、流式上传和引用解析。

## 验证

使用临时文件和伪 CLI 覆盖结构化参数、预检、共享锁、成功发布、非零退出、超时、取消、子进程组残留、
启动许可、marker 替换、隔离恢复和 PID 复用；测试不调用模型、CUDA、nvidia-smi 或外部服务。
完成后运行仓库质量门禁，并记录真实模型性能尚未验证。
