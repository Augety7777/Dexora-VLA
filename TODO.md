进行了许多移动，主要是把脚本从根目录放到scripts里，所以现在每个stage的sh脚本应该都是跑不了的；需要修正sh脚本、scripts中脚本内、和readme中的路径。
三个main脚本现在在train里，考虑到它们其实是三个train脚本加了argparse，可以合并到train中。
stats文件都被删掉了，用户可以自行从数据计算；如果要开源权重，可以放在examples下。
几个requirement.txt放到了pyproject.toml中，需要检查这样写对不对。
ref开头的参考代码、tools和lerobot都删掉了，应该没有依赖它们的东西。
dataprocess下脚本太乱了，需要整理；data和deploy下也差不多。
可以新建一个baseline文件夹，放pi0和gr00t模型的训练配置。
test意义可能不大，也许可以删掉。github ci已经删了。
整个repo里还有很多rdt相关内容，需要删掉。
许多注释提到了“paper”，并不合适，这看起来好像代码是根据论文写的一样



遥操（真机）
推理 （  ）、开环推理 （rdt-1b ）
训练 
readme一致
dataload 直接能跑
仿真 数据传hf




