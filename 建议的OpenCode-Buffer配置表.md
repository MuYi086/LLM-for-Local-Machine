建议的OpenCode-Buffer配置表
| llama.cpp 上下文 | Buffer | Keep Tokens | 名义压缩触发位置 |
| ---: | ---: | ---: | ---: |
| 32K | 8K | 4K | 约 24K |
| 48K | 10K | 6K | 约 38K |
| 64K | 12K | 8K | 约 52K |
| 96K | 16K | 10K | 约 80K |
| 128K | 20K | 12K | 约 108K |
| 256K | 32K | 16K | 约 224K |

“名义触发位置”仅为 context−buffer，不是压缩请求实际输入量的保证；工具批量返回可一次跨过阈值，估算与模型 tokenizer 也可能不同。`keep.tokens` 是压缩后保留的近期对话预算。模板错误需要修复摘要格式，不能仅靠调大 buffer 处理。仓库已增加针对 OpenCode V2/local-ornith 的分页与压缩插件，详见 [排查说明](docs/opencode-context.md)。
