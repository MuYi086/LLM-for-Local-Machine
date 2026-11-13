# Ornith Sleep Cache Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在每请求释放 GPU 的约束下复用历史状态，降低下一轮 prefill 与首 token 延迟。

**Architecture:** 用户已批准先验证跨卸载状态保存/恢复，再接入服务。沿用 llama.cpp 原生有界 RAM prompt cache（包含主模型、草稿及循环检查点），在 sleep 前保存空闲 slot，唤醒后保留缓存并由原生 token 前缀匹配恢复。补丁保存在本仓库，第三方源码和构建在仓库外。用独立链接的后端和端口实测，保留当前运行实例。

**Tech Stack:** C++17、llama.cpp、Python 标准库、现有 GPU 生命周期助手。

## 约束和验收

- 权重与 GGUF、96k 上限、MTP/batch 参数不改变；必须保持同一个 GPU_LOCK_FILE。
- 保存与恢复全部发生在 GPU 租约内；保存后必须释放 CUDA 并报告 sleeping。
- RAM 缓存必须有正数容量上限、进程退出即消失；缓存未命中或超限回退完整 prefill。
- 覆盖相同前缀追加、不相关提示、前缀分叉、禁用缓存、容量不足、无 MTP 和 MTP。
- 比较原生 cache_n、prompt_n、prefill、首 token、请求及休眠收尾、RSS/guest RAM/显存；不能只比较 decode。
- 不自动提交、下载权重或重启用户正在运行的服务。

## 执行

- [x] 1. 维护真实多轮实验脚本，原二进制跨 sleep 复现 cache_n=0。
- [x] 2. 制作 opt-in RAM 缓存补丁和独立构建脚本；保留既有 sleep-request-guard。
- [x] 3. 运行无模型容量测试和真实多轮对照；按结果修复及收敛实现。
- [x] 4. 启动配置和 readiness 检查补丁能力，维护回退参数和文档。
- [x] 5. 执行相关无模型测试、静态检查、最终评审，核实实验端口及 quarantine。

## 执行记录

- 已核实：原生 prompt_save/load 已涵盖 target、draft 与 checkpoint，sleep 路径未保存 slot，load_model 无条件重建 RAM cache。
- Ruling: 采用后端 RAM 缓存补丁替代 HTTP 文件保存 — 避免 SSE 完成到 sleep 的竞态及磁盘持久化；代价是需要部署本机后端补丁，缓存不跨进程重启。
- 真模型：无 MTP/MTP 的追加与分叉通过；MTP 长输入工具轮次关闭缓存需 prefill 61935 token/39.220 秒，最终补丁命中 61893 token，只算 42 token/0.479 秒；首事件 42.577→4.269 秒，含保存/卸载的持锁时间 46.052→9.516 秒。
- 失效与容量：修改历史前部标签后 cache_n=0 且采用新标签；无关提示不命中；1 MiB 预算全部安全冷算。原生无模型容量/前缀/淘汰检查通过，相关 Python 45 项通过。
- 评审修复：不再使用 CMake target 递归构建依赖，改为 compile_commands 单文件私有编译和 archive 替换；快照异常回退冷算；补足前部变更断言与整个 lease 资源采样。复查无阻断问题。
- 最终验证：quality_gate.sh 通过，161 项无模型测试通过；原生缓存预算与 sleep-request-guard 回归通过，Ruff/format/git diff --check 通过。18504/28504 无监听，无 quarantine；旧 8504 进程保持 98304 上下文且 sleeping。未自动重启用户服务，完整 OpenCode 固定任务尚待新构建启动后复测。
