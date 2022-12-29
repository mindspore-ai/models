
# 开发中

# 目录

- [GPT描述](#gpt描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [脚本及样例代码](#脚本及样例代码)
- [ModelZoo主页](#modelzoo主页)

# [GPT描述](#目录)

GPT网络由OpenAI提出，分为GPT、GPT2和GPT3三个版本。2020年7月提出的最新版GPT3是一个相当大的语言模型，有1750亿个参数。GPT3堆叠了Transformer的许多解码器结构，并提供了大量的训练数据。如此强大的语言模型，甚至不需要微调过程。正如论文标题所说，语言模型是小样本学习。GPT3证明，有了大型且训练有素的模型，我们可以获得与微调方法相媲美的性能。

[论文](https://arxiv.org/abs/2005.14165):  Tom B.Brown, Benjamin Mann, Nick Ryder et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). arXiv preprint arXiv:2005.14165

# [模型架构](#目录)

GPT3堆叠了Transformer的多层解码器。根据层数和嵌入大小，GPT3又分为几个版本。最大的模型包含96层，嵌入大小为12288，总参数为1750亿。

# [数据集](#目录)

- OpenWebText被用作训练数据，训练目标是预测每个位置的下一个token。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash

# 运行单机训练示例

bash scripts/run_standalone_train.sh 0 10 /path/dataset

# 运行分布式训练示例

bash scripts/run_distribute_training.sh /path/dataset /path/hccl.json 8

# 运行评估示例，目前支持Lambada和WikiText103的准确性和困惑度

bash scripts/run_evaluation.sh lambada /your/ckpt /your/data acc

```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。
请按照以下链接中的说明操作：
https:gitee.com/mindspore/models/tree/master/utils/hccl_tools.

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```shell
.
└─gpt
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh                 # Ascend单机训练shell脚本
    ├─run_distribut_train.sh                  # Ascend分布式训练shell脚本
    └─run_evaluation.sh                       # Ascend评估shell脚本
  ├─src
    ├─gpt_wrapper.py                          # 网络骨干代码
    ├─gpt.py                                  # 网络骨干代码
    ├─dataset.py                              # 数据预处理
    ├─inference.py                            # 评估函数
    ├─utils.py                                # util函数
  ├─train.py                                  # 训练阶段的训练网络
  └─eval.py                                   # 评估阶段的评估网络
```

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
