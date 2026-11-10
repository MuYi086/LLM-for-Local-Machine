
你先阅读`/home/muyi086/hf-mirror/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF`目录结构（具体使用可以参考`https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md`），然后通过start.sh调用的方式，
使用stable-diffusion-cpp的接入，在当前目录`model-image/FLUX-3-Image-Klein-GGUF-Q8`中实现FLUX-3-Image-Klein-GGUF-Q8模型的调用，并暴露在`start.sh`中以端口8502对外暴露服务，路由地址设置为`/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8`。然后再tests目录中完成该模型的调用测试demo，包含图片生成和图片编辑
