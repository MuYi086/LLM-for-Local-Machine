
你先阅读`Confucius4_TTS`这种python项目结构和实现，以及通过start.sh调用的方式，
然后阅读`/home/muyi086/hf-mirror/thewintersun/ditto-talkinghead`目录,
将Confucius4_TTS的调用在当前项目`ditto`中实现，并暴露在`start.sh`中以端口8392对外暴露服务，路由地址设置为`/v1/ditto/talkingHead`