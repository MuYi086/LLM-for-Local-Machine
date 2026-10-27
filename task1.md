
你先阅读`Confucius4_TTS`这种python项目结构和实现，以及通过start.sh调用的方式，
然后阅读`/home/muyi086/hf-mirror/QuantStack/Wan2.2-TI2V-5B-GGUF`目录,
学习Confucius4_TTS的调用方式，在当前目录`model-video/Wan2.2-TI2V-5B-GGUF`中实现Wan2.2-TI2V-5B-GGUF模型的调用，并暴露在`start.sh`中以端口8392对外暴露服务，路由地址设置为`/v1/QuantStack/Wan2.2-TI2V-5B-GGUF`