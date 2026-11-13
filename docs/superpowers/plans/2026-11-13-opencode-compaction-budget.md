# OpenCode Compaction Budget Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** 修复已批准的归档收尾膨胀与摘要输出空间不足问题，保留 96k 和已有 GPU 锁/休眠缓存协议。

**Architecture:** CLI 增加固定白名单、最多 4096 UTF-8 字节的 `--brief` 输出；基准插件禁止直接 read 遥测 JSON 和整份报告，结束后仅使用简报。只有 compaction HTTP 请求携带可剥离的预算检查标记。适配器在 GPU 租约内调用实际 backend 的 apply-template/tokenize，核验实际输入+申请输出+256 安全 token，超限返回标准 context_length_exceeded，让 OpenCode 缩减摘要历史；普通请求保持原路径。

## 已批准约束

- 不修改已有历史报告、会话数据库、权重或活动服务；不降低 96k 上限。
- tokenizer 与模板调用必须在同一 GPU 锁内，错误或取消后仍等待卸载；不能新建直连公网或绕过网关的压缩调用。
- 不通过截断用户消息或减小摘要输出掩盖预算不足。
- 固定任务维持源码 150 行分页，限制只作用于本地基准会话。

## 任务

- [x] 1. CLI 简报白名单/长度上限及 end/report 回归。
- [x] 2. 插件和固定任务收尾规则，压缩专用标记、Node 单元与隔离 OpenCode 集成。
- [x] 3. API 实际 tokenizer 预算检查，正确 400 与 finally 清理测试；真实后端文本计数核对。
- [x] 4. 文档、质量门禁、范围评审及实验清理。

## 验证记录

- 本轮历史 443219 字节汇总经简报白名单变成 1011 字节；原报告和数据库未修改。
- 真实适配器预检 16 token 与 backend usage=16 一致；实际 97012 输入/4096 输出拒绝，未提交生成，GPU 正常卸载；证据 context-budget-20261113T073904598805Z。
- 范围评审无阻断问题；补齐了改名符号链接保护和 tokenization 取消后标记/锁释放测试。
- Ruling: 隔离 OpenCode 使用 32k fixture、先返回可容纳的主请求 usage，再读取大工具结果触发压缩 — 原 16k fixture 的工具定义本身就会提前触发，若伪造 overflow，则没有可再缩减的历史；这不是实际保护逻辑的失败。
- 最终验证：quality_gate.sh 通过（170 项无模型测试），Node 插件 9 项通过；Ruff/format/git diff --check 通过。真实 tokenizer/拒绝证据在上述运行目录。当前 8500/8503/8504 及实验 18504/28504 无监听；原服务日志记录收到停止信号并正常退出，无 quarantine。未自动启动服务，完整 v6 长任务尚待重新启动后的新会话验证。
