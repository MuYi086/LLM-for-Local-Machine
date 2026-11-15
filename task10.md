
你先阅读`/home/muyi086/hf-mirror/audio.cpp-gguf/Stable-Audio-3-Medium-GGUF`目录结构，然后通过start.sh调用的方式，
使用audio.cpp的接入，在当前目录`model-audio/Stable-Audio-3-Medium-GGUF`中实现Stable-Audio-3-Medium-GGUF-f16模型的调用，并暴露在`start.sh`中以端口8314对外暴露服务，路由地址设置为`/v1/stabilityai/Stable-Audio-3-Medium-GGUF-f16`。然后再tests目录中完成该模型的demo测试
