# MiniMax H3 可选原生音轨

## 目标

H3 API 增加每请求 `include_audio` 布尔字段，默认 `false`。用户传 `true` 时生成并封装 H3 原生音轨；传 `false` 或省略时输出无音轨 MP4，便于后续用专业 TTS 配音。

## 设计

请求字段由 HTTP 层透传到一次性 worker。静音模式不加载音频 VAE、不执行 `VAEDecodeAudio`，且 `CreateVideo` 不接音频输入；带音轨模式保留当前工作流。worker 校验 MP4 的音轨是否与请求相符。H3 DiT 仍采样联合音视频 latent，因此静音模式只跳过音频解码和封装，不承诺缩短 DiT 采样或移除全部音频计算。

H3 benchmark 显式传 `include_audio: true`，维持已有原生音频对比合同。服务 README 将记录默认静音和显式开启方式。

## 范围与约束

- 不新增依赖，不改变 GPU admission gate、模型步数、帧数或服务生命周期。
- API 字段必须为布尔值；默认关闭原生音轨。
- 两种输出都必须是有效 H.264 MP4，音轨存在性必须与 `include_audio` 精确匹配。
- 验证使用无模型测试，不启动真实 GPU 推理。
