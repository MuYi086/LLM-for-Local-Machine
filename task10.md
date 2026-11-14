
你先阅读`/home/muyi086/hf-mirror/ornith-ai/Ornith-1.5-9B-MTP-GGUF`目录结构，然后通过start.sh调用的方式，
使用llama.cpp的接入，在当前目录`model-text/Ornith-1.5-9B-MTP-GGUF`中实现Ornith-1.5-9B-MTP-GGUF-Q8模型的调用，并暴露在`start.sh`中以端口8504对外暴露服务，路由地址设置为`/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`。然后再tests目录中完成该模型的demo测试
