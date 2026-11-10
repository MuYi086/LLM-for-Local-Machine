
你先阅读`~/hf-mirror/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF`目录结构（具体使用可以参考`https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/moss_image.md`），然后通过start.sh调用的方式，
使用stable-diffusion-cpp的接入，在当前目录`model-image/MOSS-Image-SFT-Q8_0`中实现MOSS-Image-SFT-GGUF模型的调用，并暴露在`start.sh`中以端口8501对外暴露服务，路由地址设置为`/v1/openmoss/MOSS-Image-SFT-Q8_0`。然后再tests目录中完成该模型的调用测试demo
