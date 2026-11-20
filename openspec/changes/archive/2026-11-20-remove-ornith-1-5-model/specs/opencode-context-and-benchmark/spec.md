## REMOVED Requirements

### Requirement: 插件只在本仓库的本地 provider 生效

**Reason**: 全局 opencode 配置已移除 `local-ornith` provider，本地插件 `.opencode/plugins/local-model-context/` 已删除。

### Requirement: 源码读取强制分页且单次一个文件

**Reason**: 该约束由已删除的本地插件实现；固定源码阅读基准已下线。

### Requirement: 压缩请求隔离且不关闭自动压缩

**Reason**: 插件压缩钩子与请求改写逻辑已删除；OpenCode 使用其自身默认压缩行为。

### Requirement: 本轮数据由插件自动绑定

**Reason**: `CONTEXT_BENCHMARK_PROMPT.md` 与 `scripts/prepare_context_benchmark.py` 已删除，不再建立本轮性能绑定。

### Requirement: 固定任务报告不编造证据

**Reason**: 固定基准任务合同随提示词与插件删除而下线。

### Requirement: 验证不调用真实模型

**Reason**: 插件测试 `tests/opencode/` 与 `tests/test_opencode_context_guard.py` 已删除，无模型基准验证入口不再存在。
