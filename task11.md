
你先阅读`Confucius4_TTS`这种python项目结构和实现，以及通过start.sh调用的方式，
然后阅读`/home/muyi086/hf-mirror/HunyuanVideo-1.5-Comfy`目录,
学习Confucius4_TTS的调用方式，在当前目录`model-video/HunyuanVideo-1.5-Comfy`中实现HunyuanVideo-1.5-Comfy模型的调用，并暴露在`start.sh`中以端口8393对外暴露服务，路由地址设置为`/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy`