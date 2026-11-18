
你先阅读`~/hf-mirror/audio.cpp-gguf`
中的
`BS-RoFormer-ep368-GGUF`
`Qwen3-ForcedAligner-0.6B-GGUF`
`Mel-Band-RoFormer-GGUF`
`HTDemucs-GGUF`
目录结构，然后通过start.sh调用的方式，
使用audio.cpp的接入，在当前目录`model-audio/${模型名称}`中实现${模型名称}的调用，并暴露在`start.sh`中以端口8344，8345，8346，8347对外暴露服务，路由地址设置为`/v1/${模型组织名称}/${模型名称}`。然后再tests目录中完成该模型的demo测试
