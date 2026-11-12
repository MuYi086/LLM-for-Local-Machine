# llama.cpp HTTP 请求与 sleep-idle 竞态补丁

## 范围

补丁基于 llama.cpp commit `7d7f0287530bc8981690f361d8c1627dbae1c99a`，保存在
`scripts/llama_cpp/sleep-request-guard.patch`。只保存局部补丁，不将第三方源码、构建产物或权重纳入仓库。

HTTP 线程在 wake 后处理模板/分词时还没有向推理队列投递任务。旧队列可能在 1 秒 idle 后释放模型及词表，导致 HTTP 线程访问已释放资源，表现为 reasoning_budget_start_tag 越界，甚至可能崩溃。
补丁用 mutex_tasks 保护的引用计数覆盖需要模型的 response generator 生命周期：先保留引用再唤醒，引用未清零时不休眠，最后释放后重新计时。仅访问缓存的 bypass_sleep 健康/props 路径不持有引用。嵌套 generator 分别获取/释放，不提前解除保护。

## 应用与重建

`LLAMA_SOURCE_DIR`、`LLAMA_BUILD_DIR` 指向仓库外现有源码与已配置的 CUDA 构建目录。先查看该目录的 git diff，保留其他本地改动。停止服务后执行：

```bash
git -C "$LLAMA_SOURCE_DIR" apply --check "$PWD/scripts/llama_cpp/sleep-request-guard.patch"
git -C "$LLAMA_SOURCE_DIR" apply "$PWD/scripts/llama_cpp/sleep-request-guard.patch"
cmake --build "$LLAMA_BUILD_DIR" --target llama-server -j 4
bash scripts/llama_cpp/test_sleep_request_guard.sh "$LLAMA_SOURCE_DIR" "$LLAMA_BUILD_DIR" fixed
```

已应用补丁时 apply --check 会拒绝重复应用；可用 `git apply --reverse --check` 确认。升级上游版本不能盲目套补丁，应重新核对上游是否已有等效修复。

确认 start.sh 打印的 LLAMA_SERVER_BIN 来自这个构建，再启动服务。本机默认 build-cuda 已应用并重建；旧进程继续使用旧库，必须正常退出再启动。此前库已在构建目录保留 `.before-sleep-request-guard` 副本。若使用自定义 LLAMA_SERVER_BIN，需要修复其对应构建，不能认为默认构建修复会自动覆盖自定义程序。

## 无模型验证

回归程序链接现有 server-context 和基础库，不加载权重、不初始化 GPU、不调用 HTTP。
它先等待休眠、模拟请求唤醒与超过 idle deadline 的预处理，验证引用期间不休眠，嵌套释放不会提前休眠，最后释放后自动休眠。
旧构建运行第三个参数 `original` 可复现 `protected_request=0`；补丁构建使用 `fixed` 必须通过。

普通 Python 质量门禁不依赖仓库外 llama.cpp 构建，所以此测试独立执行。真实长上下文任务的端到端验证必须在重启后另跑；不要把队列测试通过记为真实 Qwen 基准成功。
