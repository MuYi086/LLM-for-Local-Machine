## REMOVED Requirements

### Requirement: API 转发到 loopback backend

**Reason**: Ornith-1.5-9B-MTP 模型权重已从本机删除，8504 适配器与 8506 loopback backend 随之下线，不再提供聊天 API。

### Requirement: 适配器只转发 loopback 且超时有界

**Reason**: 适配器目录 `model-text/Ornith-1.5-9B-MTP-GGUF/` 已删除，无对应请求路径。

### Requirement: 请求租约与隔离标记按请求收尾

**Reason**: 适配器与共享 llama.cpp 运行时已删除，Ornith 请求租约不再存在。

### Requirement: 权重只读加载且不注册候选

**Reason**: 镜像目录权重已删除，start.sh 不再预检 GGUF/mmproj，服务不再启动。

### Requirement: 默认推理参数保持固定口径

**Reason**: `ORNITH_MTP_9B_*` 默认参数从 start.sh 移除，llama-server MTP 参数口径不再适用。

### Requirement: 跨 sleep 缓存有补丁与预算约束

**Reason**: sleep 缓存补丁构建脚本与 llama.cpp 运行时已删除，不再有跨 sleep 缓存预算。

### Requirement: 后端补丁部署且旧配置不再生效

**Reason**: 请求守卫补丁与相关脚本已删除。

### Requirement: 压缩请求保持上下文预算语义

**Reason**: 该语义由已删除的 OpenCode 插件与 llama.cpp 适配器实现，随能力域一并移除。

### Requirement: 启动预热与退出确认

**Reason**: start.sh 不再预热 Ornith backend，也不再执行其停止确认；llama.cpp quarantine 恢复分支已删除。
