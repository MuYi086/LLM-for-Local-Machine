我的设备情况：win11+wsl、4070tiSuper、48g内存、13600kf。我测试大模型一般在wsl的ubuntu体验
你先访问
https://github.com/leejet/stable-diffusion.cpp
1. 找出链接中所有Supported models的链接a, 然后遍历a，
2. 依次找出所有模型具体的地址b，查看地址b,从中提炼出b的用途和使用场景，补全到`stable-diffusion-cpp分析.md`的内部表格内，这部分不要省略，不然我看到介绍无法知晓模型介绍以及主要支持什么功能。每个模型都需要说明输入、输出、主要功能和使用边界，以及我的机器配置推荐使用哪个量化版本。
3. 你先只遍历找出第一条模型，我看看效果先，再决定要不要全部执行


请读取 CONTEXT_BENCHMARK_PROMPT.md，只执行“固定任务”一节。