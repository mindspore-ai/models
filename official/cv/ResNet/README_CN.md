# 目录

<!-- TOC -->

- [目录](#目录)
- [ResNet描述](#resnet描述)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [GPU处理器环境运行](#gpu处理器环境运行)
            - [运行参数服务器模式训练](#运行参数服务器模式训练)
            - [训练时推理](#训练时推理)
    - [迁移训练过程](#迁移训练过程)
        - [迁移数据集处理](#迁移数据集处理)
        - [迁移训练Ckpt获取](#迁移训练ckpt获取)
        - [用法](#用法-1)
        - [结果](#结果)
    - [迁移训练推理过程](#迁移训练推理过程)
        - [用法](#用法-2)
    - [续训过程](#续训过程)
        - [用法](#用法-3)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
        - [结果](#结果-1)
    - [评估过程](#评估过程)
        - [用法](#用法-4)
            - [Ascend处理器环境运行](#ascend处理器环境运行-2)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
        - [结果](#结果-2)
    - [预测过程](#预测过程)
        - [预测](#预测)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [ONNX的导出与推理](#onnx的导出与推理)
        - [执行推理](#执行推理)
        - [结果](#结果-3)
- [应用MindSpore Golden Stick模型压缩算法](#应用mindspore-golden-stick模型压缩算法)
    - [训练过程](#训练过程-1)
        - [GPU处理器环境运行](#gpu处理器环境运行-2)
        - [Ascend处理器环境运行](#ascend处理器环境运行-3)
    - [评估过程](#评估过程-1)
        - [GPU处理器环境运行](#gpu处理器环境运行-3)
        - [Ascend处理器环境运行](#ascend处理器环境运行-4)
        - [结果](#结果-4)
            - [GPU结果](#gpu结果)
            - [Ascend结果](#ascend结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CIFAR-10上的ResNet18](#cifar-10上的resnet18)
            - [ImageNet2012上的ResNet18](#imagenet2012上的resnet18)
            - [CIFAR-10上的ResNet50](#cifar-10上的resnet50)
            - [ImageNet2012上的ResNet50](#imagenet2012上的resnet50)
            - [ImageNet2012上的ResNet34](#imagenet2012上的resnet34)
            - [flower\_photos上的ResNet34](#flower_photos上的resnet34)
            - [ImageNet2012上的ResNet101](#imagenet2012上的resnet101)
            - [ImageNet2012上的ResNet152](#imagenet2012上的resnet152)
            - [ImageNet2012上的SE-ResNet50](#imagenet2012上的se-resnet50)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
- [FAQ](#faq)

<!-- /TOC -->

# ResNet描述

## 概述

残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

如下为MindSpore使用CIFAR-10/ImageNet2012数据集对ResNet18/ResNet50/ResNet101/ResNet152/SE-ResNet50进行训练的示例。ResNet50和ResNet101可参考[论文1](https://arxiv.org/pdf/1512.03385.pdf)，SE-ResNet50是ResNet50的一个变体，可参考[论文2](https://arxiv.org/abs/1709.01507)和[论文3](https://arxiv.org/abs/1812.01187)。使用8卡Ascend 910训练SE-ResNet50，仅需24个周期，TOP1准确率就达到了75.9%（暂不支持用CIFAR-10数据集训练ResNet101以及用用CIFAR-10数据集训练SE-ResNet50）。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf)：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

2. [论文](https://arxiv.org/abs/1709.01507)：Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu."Squeeze-and-Excitation Networks"

3. [论文](https://arxiv.org/abs/1812.01187)：Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li."Bag of Tricks for Image Classification with Convolutional Neural Networks"

# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：共10个类、60,000个32*32彩色图像
    - 训练集：50,000个图像
    - 测试集：10,000个图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```text
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─validation_preprocess # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

> - <font size=2>训练时，如果使用CIFAR-10数据集, DATASET_PATH={CIFAR-10路径}/cifar-10-batches-bin;</font>
>   <font size=2>如果使用ImageNet2012数据集, DATASET_PATH={ImageNet2012路径}/train</font>
> - <font size=2>评估和推理时，如果使用CIFAR-10数据集, DATASET_PATH={CIFAR-10路径}/cifar-10-verify-bin;</font>
>   <font size=2>如果使用ImageNet2012数据集, DATASET_PATH={ImageNet2012路径}/validation_preprocess</font>

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH]  [CONFIG_PATH] [RESUME_CKPT]（可选）

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

```

- GPU处理器环境运行

```text
# 分布式训练示例
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 单机训练示例
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 推理示例
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]  [CONFIG_PATH]
```

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 在网页上设置 "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'"
# (2) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (3) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 在网页上设置 "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'"
# (2) 把训练好的模型地方到桶的对应位置。
# (3) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (4) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──resnet
  ├── README.md
  ├── config                              # 参数配置
    ├── resnet18_cifar10_config.yaml
    ├── resnet18_cifar10_config_gpu.yaml
    ├── resnet18_imagenet2012_config.yaml
    ├── resnet18_imagenet2012_config_gpu.yaml
    ├── resnet34_imagenet2012_config.yaml
    ├── resnet50_cifar10_config.yaml
    ├── resnet34_cpu_config.yaml
    ├── resnet50_imagenet2012_Boost_config.yaml     # 高性能版本：性能提高超过10%而精度下降少于1%
    ├── resnet50_imagenet2012_Ascend_Thor_config.yaml
    ├── resnet50_imagenet2012_config.yaml
    ├── resnet50_imagenet2012_GPU_Thor_config.yaml
    ├── resnet101_imagenet2012_config.yaml
    ├── resnet152_imagenet2012_config.yaml
    ├── se-resnet50_imagenet2012_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_parameter_server_train.sh      # 启动Ascend参数服务器训练(8卡)
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├── run_distribute_train_gpu.sh        # 启动GPU分布式训练（8卡）
    ├── run_parameter_server_train_gpu.sh  # 启动GPU参数服务器训练（8卡）
    ├── run_eval_gpu.sh                    # 启动GPU评估
    ├── run_standalone_train_gpu.sh        # 启动GPU单机训练（单卡）
    └── cache_util.sh                      # 使用单节点緩存的帮助函数
  ├── src
    ├── data_split.py                      # 切分迁移数据集脚本（cpu）
    ├── dataset.py                         # 数据预处理
    ├── logger.py                          # 日志处理
    ├── callback.py                        # 训练时推理回调函数
    ├── util.py                            # 定义基础功能
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet骨干网络，包括ResNet50、ResNet101和SE-ResNet50
    ├── model_utils
       ├── config.py                       # 参数配置
       ├── device_adapter.py               # 设备配置
       ├── local_adapter.py                # 本地设备配置
       └── moxing_adapter.py               # modelarts设备配置
  ├── fine_tune.py                         # 迁移训练网络（cpu）
  ├── quick_start.py                       # quick start演示文件（cpu）
  ├── requirements.txt                     # 第三方依赖
  ├── eval.py                              # 评估网络
  ├── predict.py                           # 预测网络
  └── train.py                             # 训练网络
```

## 脚本参数

在配置文件中可以同时配置训练参数和评估参数。

- 配置ResNet18、ResNet50和CIFAR-10数据集。

```text
"class_num":10,                  # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":10,        # 只保留最后一个keep_checkpoint_max检查点
"warmup_epochs":5,               # 热身周期数
"lr_decay_mode":"poly”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.1,                    # 最大学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

- 配置ResNet18、ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

- 配置ResNet34和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":1,         # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2,               # 热身周期数
"optimizer":"Momentum",          # 优化器
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":1.0,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

- 配置ResNet101和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":120,                # 训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":2,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr":0.1                         # 基础学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

- 配置ResNet152和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":140,                # 训练周期大小
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":2,               # 热身周期数  
"lr_decay_mode":"steps",         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr":0.1,                        # 基础学习率
"lr_end":0.0001,                 # 最终学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

- 配置SE-ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":28,                 # 创建学习率的周期大小
"train_epoch_size":24            # 实际训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":4,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":3,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0.0,                   # 初始学习率
"lr_max":0.3,                    # 最大学习率
"lr_end":0.0001,                 # 最终学习率
"save_graphs":False,             # 是否保存图编译结果
"save_graphs_path":"./graphs",   # 图编译结果保存路径
"has_trained_epoch":0,           # 加载已经训练好的模型的epoch大小；实际训练周期大小等于epoch_size减去has_trained_epoch
"has_trained_step":0,            # 加载已经训练好的模型的step大小；实际训练周期大小等于step_size减去has_trained_step
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 运行评估示例
用法：bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

运行单卡用例时如果想更换运行卡号，可以通过设置环境变量 `export DEVICE_ID=x` 或者在context中设置 `device_id=x`指定相应的卡号。

#### GPU处理器环境运行

```text
# 分布式训练示例
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 单机训练示例
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）

# 推理示例
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

#### 运行参数服务器模式训练

- Ascend参数服务器训练示例

```text
bash run_parameter_server_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）
```

- GPU参数服务器训练示例

```text
bash run_parameter_server_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]（可选）
```

#### 训练时推理

```bash
# Ascend 分布式训练时推理示例:
cd scripts/
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# Ascend 分布式断点训练时推理示例:
cd scripts/
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH] [RESUME_CKPT]

# Ascend 单机训练时推理示例:
cd scripts/
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# Ascend 单机断点训练时推理示例:
cd scripts/
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH] [RESUME_CKPT]

# GPU 分布式训练时推理示例:
cd scripts/
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# GPU 单机训练时推理示例:
cd scripts/
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]
```

训练时推理需要在设置`RUN_EVAL`为True，与此同时还需要设置`EVAL_DATASET_PATH`。此外，当设置`RUN_EVAL`为True时还可为python脚本设置`save_best_ckpt`, `eval_start_epoch`, `eval_interval`等参数。

默认情况下我们将启动一个独立的缓存服务器将推理数据集的图片以tensor的形式保存在内存中以带来推理性能的提升。用户在使用缓存前需确保内存大小足够缓存推理集中的图片（缓存ImageNet2012的推理集大约需要30GB的内存，缓存CIFAR-10的推理集约需要使用6GB的内存）。

在训练结束后，可以选择关闭缓存服务器或不关闭它以继续为未来的推理提供缓存服务。

## 迁移训练过程

### 迁移数据集处理

[根据提供的数据集链接下载数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz ),将切分数据集脚本data_split.py放置在下载好的flower_photos目录下，运行后会生成train文件夹及test文件夹，将train文件夹及test文件夹保存到新建文件夹datasets里.

### 迁移训练Ckpt获取

[根据提供的Ckpt链接下载数据集](https://download.mindspore.cn/models/r1.5/ ),Ckpt文件名称为“resnet34_ascend_v170_imagenet2012_official_cv_top1acc73.61_top5acc91.74.ckpt”，下载后存放在fine_tune.py同路径下。

### 用法

您可以通过python脚本开始训练：

```shell
python fine_tune.py --config_path ./config/resnet34_cpu_config.yaml
```

### 结果

- 使用flower_photos据集训练ResNet34

```text
# 迁移训练结果（CPU）
2023-02-17 11:43:14,405:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 1.487435, per step time: 3096.616 ms
2023-02-17 11:43:37,711:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 1.251532, per step time: 751.433 ms
2023-02-17 11:44:41,012:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 1.079233, per step time: 481.662 ms
2023-02-17 11:43:44,326:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 0.981760, per step time: 462.071 ms
2023-02-17 11:43:47,646:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 0.898887, per step time: 426.740 ms
2023-02-17 11:43:50,943:INFO:epoch: [1/10] step: [10/85], lr: 0.001000, loss: 0.803308, per step time: 450.666 ms
...
```

## 迁移训练推理过程

### 用法

您可以通过python脚本开始推理(需要先到resnet34_cpu_config.yaml配置文件中将ckpt_path设为最好的ckpt文件路径):

```shell
python eval.py --config_path ./cpu_default_config.yaml --data_path ./dataset/flower_photos/test
```

## 续训过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]

# 单机训练
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]
```

### 结果

- 使用CIFAR-10数据集训练ResNet18

```text
# 分布式训练结果（8P）
2023-02-17 14:27:29,405:INFO:epoch: [1/90] loss: 1.082604, epoch time: 40.559 s, per step time: 207.995 ms
2023-02-17 14:27:31,711:INFO:epoch: [2/90] loss: 1.045892, epoch time: 2.413 s, per step time: 12.377 ms
2023-02-17 14:27:34,012:INFO:epoch: [3/90] loss: 0.729006, epoch time: 2.486 s, per step time: 12.750 ms
2023-02-17 14:27:36,326:INFO:epoch: [4/90] loss: 0.766412, epoch time: 2.443 s, per step time: 12.529 ms
2023-02-17 14:27:39,646:INFO:epoch: [5/90] loss: 0.655058, epoch time: 2.851 s, per step time: 14.621 ms
...
```

- 使用ImageNet2012数据集训练ResNet18

```text
# 分布式训练结果（8P）
2023-02-17 15:30:06,405:INFO:epoch: [1/90] loss: 5.023574, epoch time: 154.658 s, per step time: 247.453 ms
2023-02-17 15:31:45,711:INFO:epoch: [2/90] loss: 4.253309, epoch time: 99.524 s, per step time: 159.239 ms
2023-02-17 15:33:18,012:INFO:epoch: [3/90] loss: 3.703176, epoch time: 92.655 s, per step time: 148.248 ms
2023-02-17 15:34:34,326:INFO:epoch: [4/90] loss: 3.458283, epoch time: 76.299 s, per step time: 122.078 ms
2023-02-17 15:35:59,646:INFO:epoch: [5/90] loss: 3.603806, epoch time: 84.435 s, per step time: 135.097 ms
...
```

- 使用ImageNet2012数据集训练ResNet34

```text
# 分布式训练结果（8P）
2023-02-20 09:47:10,405:INFO:epoch: [1/90] loss: 5.044510, epoch time: 139.308 s, per step time: 222.893 ms
2023-02-20 09:48:30,711:INFO:epoch: [2/90] loss: 4.194771, epoch time: 79.498 s, per step time: 127.196 ms
2023-02-20 09:49:53,012:INFO:epoch: [3/90] loss: 3.736507, epoch time: 83.387 s, per step time: 133.419 ms
2023-02-20 09:51:17,326:INFO:epoch: [4/90] loss: 3.417167, epoch time: 83.253 s, per step time: 133.204 ms
2023-02-20 09:52:41,646:INFO:epoch: [5/90] loss: 3.444441, epoch time: 83.931 s, per step time: 134.290 ms
...
```

- 使用CIFAR-10数据集训练ResNet50

```text
# 分布式训练结果（8P）
2023-02-20 10:14:13,405:INFO:epoch: [1/90] loss: 1.519848, epoch time: 63.275 s, per step time: 324.489 ms
2023-02-20 10:14:16,711:INFO:epoch: [2/90] loss: 1.497206, epoch time: 3.305 s, per step time: 16.950 ms
2023-02-20 10:14:19,012:INFO:epoch: [3/90] loss: 1.097057, epoch time: 3.315 s, per step time: 17.002 ms
2023-02-20 10:14:23,326:INFO:epoch: [4/90] loss: 0.852322, epoch time: 3.322 s, per step time: 17.036 ms
2023-02-20 10:14:27,646:INFO:epoch: [5/90] loss: 0.896606, epoch time: 4.432 s, per step time: 22.730 ms
...
```

- 使用ImageNet2012数据集训练ResNet50

```text
# 分布式训练结果（8P）
2023-02-20 10:01:18,405:INFO:epoch: [1/90] loss: 5.282135, epoch time: 183.647 s, per step time: 588.613 ms
2023-02-20 10:03:02,711:INFO:epoch: [2/90] loss: 4.446517, epoch time: 103.711 s, per step time: 332.408 ms
2023-02-20 10:04:41,012:INFO:epoch: [3/90] loss: 3.916948, epoch time: 99.554 s, per step time: 319.804 ms
2023-02-20 10:06:15,326:INFO:epoch: [4/90] loss: 3.510729, epoch time: 94.192 s, per step time: 301.897 ms
2023-02-20 10:07:43,646:INFO:epoch: [5/90] loss: 3.402662, epoch time: 87.943 s, per step time: 281.867 ms
...
```

- 使用ImageNet2012数据集训练ResNet101

```text
# 分布式训练结果（8P）
2023-02-20 10:52:57,405:INFO:epoch: [1/90] loss: 5.139862, epoch time: 218.528 s, per step time: 43.671 ms
2023-02-20 10:55:18,711:INFO:epoch: [2/90] loss: 4.252709, epoch time: 140.305 s, per step time: 28.039 ms
2023-02-20 10:57:38,012:INFO:epoch: [3/90] loss: 4.101140, epoch time: 140.267 s, per step time: 28.031 ms
2023-02-20 10:59:58,326:INFO:epoch: [4/90] loss: 3.468216, epoch time: 140.142 s, per step time: 28.006 ms
2023-02-20 11:02:20,646:INFO:epoch: [5/90] loss: 3.155962, epoch time: 140.167 s, per step time: 28.411 ms
...
```

- 使用ImageNet2012数据集训练ResNet152

```text
# 分布式训练结果（8P）
2023-02-20 11:29:43,405:INFO:epoch: [1/90] loss: 4.546348, epoch time: 308.530 s, per step time: 61.657 ms
2023-02-20 11:33:08,711:INFO:epoch: [2/90] loss: 4.020557, epoch time: 205.175 s, per step time: 41.002 ms
2023-02-20 11:36:34,012:INFO:epoch: [3/90] loss: 3.691725, epoch time: 205.198 s, per step time: 41.007 ms
2023-02-20 11:39:59,326:INFO:epoch: [4/90] loss: 3.230466, epoch time: 205.363 s, per step time: 41.040 ms
2023-02-20 11:43:27,646:INFO:epoch: [5/90] loss: 2.961051, epoch time: 208.493 s, per step time: 41.665 ms
...
```

- 使用ImageNet2012数据集训练SE-ResNet50

```text
# 分布式训练结果（8P）
2023-02-20 11:57:34,405:INFO:epoch: [1/90] loss: 4.478792, epoch time: 185.971 s, per step time: 37.164 ms
2023-02-20 11:59:22,711:INFO:epoch: [2/90] loss: 4.082346, epoch time: 107.408 s, per step time: 21.464 ms
2023-02-20 12:01:09,012:INFO:epoch: [3/90] loss: 4.116436, epoch time: 107.551 s, per step time: 21.493 ms
2023-02-20 12:02:58,326:INFO:epoch: [4/90] loss: 3.494506, epoch time: 108.719 s, per step time: 21.726 ms
2023-02-20 12:04:45,646:INFO:epoch: [5/90] loss: 3.412843, epoch time: 107.505 s, per step time: 21.484 ms
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 评估
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# 评估示例
bash run_eval.sh ~/cifar10-10-verify-bin  /resnet50_cifar10/train_parallel0/resnet-90_195.ckpt config/resnet50_cifar10_config.yaml
```

> 训练过程中可以生成检查点。

#### GPU处理器环境运行

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用CIFAR-10数据集评估ResNet18

```bash
result: {'top_5_accuracy': 0.9988420294494239, 'top_1_accuracy': 0.9369917221518} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- 使用ImageNet2012数据集评估ResNet18

```bash
result: {'top_5_accuracy': 0.89609375, 'top_1_accuracy': 0.7056089743589744} ckpt=train_parallel0/resnet-90_625.ckpt
```

- 使用CIFAR-10数据集评估ResNet50

```bash
result: {'top_5_accuracy': 0.99879807699230679, 'top_1_accuracy': 0.9372996794891795} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- 使用ImageNet2012数据集评估ResNet50

```bash
result: {'top_5_accuracy': 0.930090206185567, 'top_1_accuracy': 0.764074581185567} ckpt=train_parallel0/resnet-90_625.ckpt
```

- 使用ImageNet2012数据集评估ResNet34

```bash
result: {'top_5_accuracy': 0.9166866987179487, 'top_1_accuracy': 0.7379497051282051} ckpt=train_parallel0/resnet-90_625.ckpt
```

- 使用ImageNet2012数据集评估ResNet101

```bash
result:{'top_5_accuracy':0.9429417413572343, 'top_1_accuracy':0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

- 使用ImageNet2012数据集评估ResNet152

```bash
result: {'top_5_accuracy': 0.9438420294494239, 'top_1_accuracy': 0.78817221518} ckpt= resnet152-140_5004.ckpt
```

- 使用ImageNet2012数据集评估SE-ResNet50

```bash
result:{'top_5_accuracy':0.9342589628681178, 'top_1_accuracy':0.768065781049936} ckpt=train_parallel0/resnet-24_5004.ckpt

```

## 预测过程

### 预测

在运行以下命令之前，请检查用于评估的检查点路径和图片路径。

```shell
python predict.py --checkpoint_file_path [CKPT_PATH] --config_path [CONFIG_PATH] --img_path [IMG_PATH] > log.txt 2>&1 &  
```

示例如下：

```bash
python predict.py --checkpoint_file_path train_parallel0/resnet-90_625.ckpt --config_path config/resnet18_imagenet2012_config_gpu.yaml --img_path test.png > log.txt 2>&1 &  
```

您可以通过log.txt文件查看结果。预测结果和平均预测时间如下：

```bash
Prediction res: 5
Prediction avg time: 5.360 ms
```

如果你想调用MindSpore Lite后端进行推理，你可以直接设置接口 `predict` 的参数 `backend` 为'lite'，这是一个实验性质的特性，相应的运行示例如下：

```bash
predict.py --checkpoint_file_path [CKPT_PATH] --config_path [CONFIG_PATH] --img_path [IMG_PATH] --enable_predict_lite_backend True > log.txt 2>&1 &  
```

或者你可以调用MindSpore Lite Python接口进行推理，示例如下，具体细节参考[使用Python接口执行云侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_python.html) 。

```bash
predict.py --checkpoint_file_path [CKPT_PATH] --config_path [CONFIG_PATH] --img_path [IMG_PATH] --enable_predict_lite_mindir True > log.txt 2>&1 &  
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#contents)

导出mindir模型

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH] --batch_size 1
```

参数checkpoint_file_path为必填项，
`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

ModelArts导出mindir

```python
# (1) 在网页上设置 "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'"
# (2) 把训练好的模型地方到桶的对应位置。
# (3) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./resnet'"参数在yaml文件。
#          设置 "file_format='MINDIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./resnet'"参数在modearts的界面上。
#          设置 "file_format='MINDIR'" 参数在modearts的界面上。
# (4) 在modelarts的界面上设置代码的路径 "/path/resnet"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

### ONNX的导出与推理

- 导出ONNX模型

    ```bash
    python export.py --checkpoint_file_path=[CKPT_PATH] --device_target="GPU" --file_format="ONNX" --config_path=[CONFIG]

    其中：
        CKPT_PATH：ckpt所在路径
        CONFIG：参数配置文件的路径

    ```

- 启动推理：python方式

    ```bash
    python eval_onnx.py --net_name=[NETWORK] --dataset=[DATASET] --dataset_path=[DATAPATH] --onnx_path=[ONNX_PATH]

    #example: python eval_onnx.py --net_name="resnet18" dataset="imagenet2012" --dataset_path=/hy-tmp/data/imagenet2012/val onnx_path=/hy-tmp/resnet/hardnet.onnx
    ```

- 启动推理：bash(脚本)方式

    ```bash
    bash script/run_eval_onnx_gpu.sh [NETWORK] [DATASET] [DATAPATH] [ONNXPATH]

    其中：
        NETWORK：网络模型名称(如resnet18)
        DATASET：数据集(如imagenet2012、cifar10)
        DATAPATH：数据集路径
        ONNXPATH：onnx路径

    #example: bash scripts/run_onnx_eval_gpu.sh "resnet18" "imagenet2012" /hy-tmp/data/imagenet2012/val /hy-tmp/resnet/hardnet.onnx
    ```

- onnx推理结果将存放在 eval_onnx.log 中

### 执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

- `NET_TYPE` 选择范围：[resnet18, resnet34, se-resnet50, resnet50, resnet101, resnet152]。
- `DATASET` 选择范围：[cifar10, imagenet]。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

- 使用CIFAR-10数据集评估ResNet18

```bash
Total data: 10000, top1 accuracy: 0.9426, top5 accuracy: 0.9987.
```

- 使用ImageNet2012数据集评估ResNet18

```bash
Total data: 50000, top1 accuracy: 0.70668, top5 accuracy: 0.89698.
```

- 使用CIFAR-10数据集评估ResNet50

```bash
Total data: 10000, top1 accuracy: 0.9310, top5 accuracy: 0.9980.
```

- 使用ImageNet2012数据集评估ResNet50

```bash
Total data: 50000, top1 accuracy: 0.7696, top5 accuracy: 0.93432.
```

- 使用ImageNet2012数据集评估ResNet34

```bash
Total data: 50000, top1 accuracy: 0.7367.
```

- 使用ImageNet2012数据集评估ResNet101

```bash
Total data: 50000, top1 accuracy: 0.7871, top5 accuracy: 0.94354.
```

- 使用ImageNet2012数据集评估ResNet152

```bash
Total data: 50000, top1 accuracy: 0.78625, top5 accuracy: 0.94358.
```

- 使用ImageNet2012数据集评估SE-ResNet50

```bash
Total data: 50000, top1 accuracy: 0.76844, top5 accuracy: 0.93522.
```

# 应用MindSpore Golden Stick模型压缩算法

MindSpore Golden Stick是MindSpore的模型压缩算法集，我们可以在模型训练前应用MindSpore Golden Stick中的模型压缩算法，从而达到压缩模型大小、降低模型推理功耗，或者加速推理过程的目的。

针对ResNet50，MindSpore Golden Stick提供了SimQAT和SCOP算法，SimQAT是一种量化感知训练算法，通过引入伪量化节点来训练网络中的某些层的量化参数，从而在部署阶段，模型得以以更小的功耗或者更高的性能进行推理。SCOP算法提出一种可靠剪枝方法，通过构建一种科学控制机制减少所有潜在不相关因子的影响，有效的按比例进行节点删除，从而实现模型小型化。

针对ResNet18，MindSpore Golden Stick引入了华为自研量化算法SLB，SLB是一种基于权值搜索的低比特量化算法，利用连续松弛策略搜索离散权重，训练时优化离散权重的分布，最后根据概率挑选离散权重实现量化。与传统的量化算法相比，规避了不准确的梯度更新过程，在极低比特量化中更有优势。

## mindspore_gs环境安装[参考gloden-stick](https://toscode.gitee.com/kevinkunkun/golden-stick)

## 训练过程

| **算法**  | SimQAT | SCOP | SLB |
| --------- | ------ | --- | ---- |
| **支持的后端**  | GPU | GPU、Ascend | GPU |
| **是否支持预训练** | 支持加载预训练ckpt | 必须提供预训练ckpt | 算法原理上无法复用原ckpt，无法加载预训练ckpt |
| **是否支持续训练** | 支持 | 支持 | 支持 |
| **是否支持多卡训练** | 支持 | 支持 | 支持 |

- 预训练是指先不应用算法，先训练收敛一个全精度的网络。预训练获得的checkpoint文件被用于后续应用算法后的训练。
- 续训练是指应用算法后训练网络，在训练过程中中断训练，后续从中断处的ckpt继续进行训练。

### GPU处理器环境运行

```text
# 分布式训练
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'train.py'脚本所在的目录。
bash run_distribute_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# 分布式训练示例（应用SimQAT算法并从头开始量化训练）
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset

# 分布式训练示例（应用SimQAT算法并加载预训练的全精度checkoutpoint，进行量化训练）
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# 分布式训练示例（应用SimQAT算法并加载之前训练的checkoutpoint，继续进行量化训练）
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt

# 单机训练
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'train.py'脚本所在的目录。
bash run_standalone_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# 单机训练示例（应用SimQAT算法并从头开始量化训练）
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset

# 单机训练示例（应用SimQAT算法并加载预训练的全精度checkoutpoint，并进行量化训练）
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# 单机训练示例（应用SimQAT算法并加载上次量化训练的checkoutpoint，继续进行量化训练）
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt

# 针对不同的算法，只需替换PYTHON_PATH和CONFIG_FILE即可，比如需要应用SLB算法并使用单卡训练：
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/slb/ ../quantization/slb/resnet18_cifar10_config.yaml /path/to/dataset
# 比如需要应用SCOP算法并使用多卡训练：
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../pruner/scop/ ../pruner/scop/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt
```

### Ascend处理器环境运行

```text
# 分布式训练
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'train.py'脚本所在的目录。
bash run_distribute_train.sh [RANK_TABLE_FILE] [PYTHON_PATH] [CONFIG_PATH] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# 分布式训练示例(SCOP算法使用多卡训练)
bash run_distribute_train.sh /path/to/rank_table_file ../pruner/scop/ ../pruner/scop/resnet50_cifar10_config.yaml /path/to/dataset

# 单机训练
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'train.py'脚本所在的目录。
bash run_standalone_train.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# 单机训练示例(SCOP算法使用单卡训练)
bash run_standalone_train_ascend.sh ../pruner/scop/ ../pruner/scop/resnet50_cifar10_config.yaml /path/to/dataset
```

## 评估过程

### GPU处理器环境运行

```text
# 评估
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'eval.py'脚本所在的目录。
bash run_eval_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]
```

```text
# 评估示例
cd ./golden_stick/scripts/
bash run_eval_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset /path/to/ckpt

# 针对不同的量化算法，只需替换PYTHON_PATH和CONFIG_FILE即可，比如需要评估应用SLB算法后的resnet18网络精度：
bash run_eval_gpu.sh ../quantization/slb/ ../quantization/slb/resnet18_cifar10_config.yaml /path/to/dataset /path/to/ckpt
```

### Ascend处理器环境运行

```text
# 评估
cd ./golden_stick/scripts/
# PYTHON_PATH 表示需要应用的算法的'eval.py'脚本所在的目录。
bash run_eval_ascend.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]
```

```text
# 评估示例
cd ./golden_stick/scripts/
bash run_eval_gpu.sh ../pruner/scop/ ../pruner/scop/resnet50_cifar10_config.yaml /path/to/dataset /path/to/ckpt
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

#### GPU结果

- 使用SimQAT算法量化ResNet50，并使用CIFAR-10数据集评估：

```text
result:{'top_1_accuracy': 0.9354967948717948, 'top_5_accuracy': 0.9981971153846154} ckpt=~/resnet50_cifar10/train_parallel0/resnet-180_195.ckpt
```

- 使用SimQAT算法量化ResNet50，并使用ImageNet2012数据集评估：

```text
result:{'top_1_accuracy': 0.7254057298335468, 'top_5_accuracy': 0.9312684058898848} ckpt=~/resnet50_imagenet2012/train_parallel0/resnet-180_6672.ckpt
```

- 使用SCOP算法剪枝ResNet50，并使用CIFAR-10数据集评估：

```text
result:{'top_1_accuracy': 0.9273838141025641} prune_rate=0.45 ckpt=~/resnet50_cifar10/train_parallel0/resnet-400_390.ckpt
```

- 使用SLB算法对ResNet18做W4量化，并使用CIFAR-10数据集评估，W4表示权重量化为4bit：

```text
result:{'top_1_accuracy': 0.9534254807692307, 'top_5_accuracy': 0.9969951923076923} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W4量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W4表示权重量化为4bit：

```text
result:{'top_1_accuracy': 0.9537259230480767, 'top_5_accuracy': 0.9970251907601913} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W4A8量化，并使用CIFAR-10数据集评估，W4表示权重量化为4bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9493423482907600, 'top_5_accuracy': 0.9965192030237169} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W4A8量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W4表示权重量化为4bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9502425480769207, 'top_5_accuracy': 0.99679551926923707} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W2量化，并使用CIFAR-10数据集评估，W2表示权重量化为2bit：

```text
result:{'top_1_accuracy': 0.9503205128205128, 'top_5_accuracy': 0.9966947115384616} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W2量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W2表示权重量化为2bit：

```text
result:{'top_1_accuracy': 0.9509508250132057, 'top_5_accuracy': 0.9967347384161105} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W2A8量化，并使用CIFAR-10数据集评估，W2表示权重量化为2bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9463205184161728, 'top_5_accuracy': 0.9963947115384616} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W2A8量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W2表示权重量化为2bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9473382052115128, 'top_5_accuracy': 0.9964718041530417} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W1量化，并使用CIFAR-10数据集评估，W1表示权重量化为1bit：

```text
result:{'top_1_accuracy': 0.9485176282051282, 'top_5_accuracy': 0.9965945512820513} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W1量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W1表示权重量化为1bit：

```text
result:{'top_1_accuracy': 0.9491012820516176, 'top_5_accuracy': 0.9966351282059453} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W1A8量化，并使用CIFAR-10数据集评估，W1表示权重量化为1bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9450068910250512, 'top_5_accuracy': 0.9962450312382200} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W1A8量化，开启BatchNorm层矫正功能，并使用CIFAR-10数据集评估，W1表示权重量化为1bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.9466145833333334, 'top_5_accuracy': 0.9964050320512820} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- 使用SLB算法对ResNet18做W4量化，并使用ImageNet2012数据集评估，W4表示权重量化为4bit：

```text
result:{'top_1_accuracy': 0.6858173076923076, 'top_5_accuracy': 0.8850560897435897} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W4量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W4表示权重量化为4bit：

```text
result:{'top_1_accuracy': 0.6865184294871795, 'top_5_accuracy': 0.8856570512820513} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W4A8量化，并使用ImageNet2012数据集评估，W4表示权重量化为4bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6809975961503861, 'top_5_accuracy': 0.8819477163043847} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W4A8量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W4表示权重量化为4bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6816538461538406, 'top_5_accuracy': 0.8826121794871795} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W2量化，并使用ImageNet2012数据集评估，W2表示权重量化为2bit：

```text
result:{'top_1_accuracy': 0.6840144230769231, 'top_5_accuracy': 0.8825320512820513} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W2量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W2表示权重量化为2bit：

```text
result:{'top_1_accuracy': 0.6841746794871795, 'top_5_accuracy': 0.8840344551282051} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W2A8量化，并使用ImageNet2012数据集评估，W2表示权重量化为2bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6791516410250210, 'top_5_accuracy': 0.8808693910256410} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W2A8量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W2表示权重量化为2bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6805694500104102, 'top_5_accuracy': 0.8814763916410150} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W1量化，并使用ImageNet2012数据集评估，W1表示权重量化为1bit：

```text
result:{'top_1_accuracy': 0.6652945112820795, 'top_5_accuracy': 0.8690705128205128} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W1量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W1表示权重量化为1bit：

```text
result:{'top_1_accuracy': 0.6675184294871795, 'top_5_accuracy': 0.8707516025641026} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W1A8量化，并使用ImageNet2012数据集评估，W1表示权重量化为1bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6589927884615384, 'top_5_accuracy': 0.8664262820512820} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- 使用SLB算法对ResNet18做W1A8量化，开启BatchNorm层矫正功能，并使用ImageNet2012数据集评估，W1表示权重量化为1bit，A8表示激活量化为8bit：

```text
result:{'top_1_accuracy': 0.6609142628205128, 'top_5_accuracy': 0.8670873397435898} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

#### Ascend结果

- 使用SCOP算法剪枝ResNet50，并使用CIFAR-10数据集评估：

```text
result:{'top_1_accuracy': 0.928385416666666} prune_rate=0.45 ckpt=~/resnet50_cifar10/train_parallel0/resnet-400_195.ckpt
```

# 模型描述

## 性能

### 评估性能

#### CIFAR-10上的ResNet18

| 参数                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| 模型版本              | ResNet18                                                | ResNet18 |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  | PCIE V100-32G        |
| 上传日期              | 2021-02-25                          | 2021-07-23     |
| MindSpore版本          | 1.1.1                                                       | 1.3.0                |
| 数据集                    | CIFAR-10                                                    | CIFAR-10           |
| 训练参数        | epoch=90, steps per epoch=195, batch_size = 32             | epoch=90, steps per epoch=195, batch_size = 32  |
| 优化器                  | Momentum                                                         | Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵 |
| 输出                    | 概率                                                 | 概率 |
| 损失                       | 0.0002519517                                                   | 0.0015517382    |
| 速度                      | 13毫秒/步（8卡）                     | 29毫秒/步（8卡）       |
| 总时长                 | 4分钟                          | 11分钟       |
| 参数(M)             | 11.2                                                         | 11.2                         |
| 微调检查点 | 86（.ckpt文件）                                         |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### ImageNet2012上的ResNet18

| 参数                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| 模型版本              | ResNet18                                               | RESNET18 |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |  PCIE V100-32G        |
| 上传日期              | 2020-04-01  ;                        | 2021-07-23 |
| MindSpore版本          | 1.1.1                                                       | 1.3.0 |
| 数据集                    | ImageNet2012                                                    | ImageNet2012           |
| 训练参数        | epoch=90, steps per epoch=626, batch_size = 256             |  epoch=90, steps per epoch=625, batch_size = 256  |
| 优化器                  | Momentum                                                         |  Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵 |
| 输出                    | 概率                                                 |  概率 |
| 损失                       | 2.15702                                                       | 2.168664 |
| 速度                      | 110毫秒/步（8卡） (可能需要在datasetpy中增加set_numa_enbale绑核操作)                    | 107毫秒/步（8卡） |
| 总时长                 | 110分钟                          | 130分钟       |
| 参数(M)             | 11.7                                                         | 11.7 |
| 微调检查点| 90M（.ckpt文件）                                         |  90M（.ckpt文件） |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### CIFAR-10上的ResNet50

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet50-v1.5                                                |ResNet50-v1.5|
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  | GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05                          | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0   |
| 数据集                    | CIFAR-10                                                    | CIFAR-10
| 训练参数        | epoch=90, steps per epoch=195, batch_size = 32             |epoch=90, steps per epoch=195, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    | 概率                                                 |  概率          |
| 损失                       | 0.000356                                                    | 0.000716  |
| 速度                      | 18.4毫秒/步（8卡）                     |69毫秒/步（8卡）|
| 总时长                 | 6分钟                          | 20.2分钟|
| 参数(M)             | 25.5                                                         | 25.5 |
| 微调检查点 | 179.7M（.ckpt文件）                                         | 179.7M（.ckpt文件） |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### ImageNet2012上的ResNet50

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet50-v1.5                                                |ResNet50-v1.5|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |  GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05  ;                        | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0      |
| 数据集                    | ImageNet2012                                                    | ImageNet2012|
| 训练参数        | epoch=90, steps per epoch=626, batch_size = 256             |epoch=90, steps per epoch=5004, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    | 概率                                                 |  概率          |
| 损失                       | 1.8464266                                                    | 1.9023  |
| 速度                      | 118毫秒/步（8卡）                     |67.1毫秒/步（8卡）|
| 总时长                 | 114分钟                          | 500分钟|
| 参数(M)             | 25.5                                                         | 25.5 |
| 微调检查点| 197M（.ckpt文件）                                         | 197M（.ckpt文件）     |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### ImageNet2012上的ResNet34

| 参数                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| 模型版本              | ResNet34                                               |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期              | 2021-05-08  ;                        |
| MindSpore版本          | 1.1.1                                                       |
| 数据集                    | ImageNet2012                                                    |
| 训练参数        | epoch=90, steps per epoch=625, batch_size = 256             |
| 优化器                  | Momentum                                                         |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 1.9575993                                                       |
| 速度                      | 111毫秒/步（8卡）                     |
| 总时长                 | 112分钟                          |
| 参数(M)             | 20.79                                                         |
| 微调检查点| 166M（.ckpt文件）                                         |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config)|

#### flower_photos上的ResNet34

| 参数                 | CPU                                                                           |
| -------------------------- |-------------------------------------------------------------------------------|
| 模型版本              | ResNet34                                                                      |
| 资源                   | CPU 3.40GHz，4核；内存 8G；系统 win7                                                  |
| 上传日期              | 2022-08-30                                                                    |
| MindSpore版本          | 1.8.1                                                                         |
| 数据集                    | flower_photos                                                                 |
| 训练参数        | epoch=10, steps per epoch=85, batch_size = 32                                 |
| 优化器                  | Momentum                                                                      |
| 损失函数              | Softmax交叉熵                                                                    |
| 输出                    | 概率                                                                            |
| 损失                       | 0.32727173                                                                    |
| 速度                      | 6859毫秒/步                                                                      |
| 总时长                 | 70分钟                                                                          |
| 参数(M)             | 20.28                                                                         |
| 微调检查点| 81M（.ckpt文件）                                                                  |
| 配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### ImageNet2012上的ResNet101

| 参数                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| 模型版本              | ResNet101                                                |ResNet101|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  |  GPU(Tesla V100 SXM2)；CPU：2.1GHz，24核；内存：128G
| 上传日期              | 2021-07-05  ;                        | 2021-07-05
| MindSpore版本          | 1.3.0                                                       |1.3.0         |
| 数据集                    | ImageNet2012                                                    | ImageNet2012|
| 训练参数        | epoch=120, steps per epoch=5004, batch_size = 32             |epoch=120, steps per epoch=5004, batch_size = 32  |
| 优化器                  | Momentum                                                         |Momentum|
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵           |
| 输出                    |概率                                                 |  概率          |
| 损失                       | 1.6453942                                                    | 1.7023412  |
| 速度                      | 30.3毫秒/步（8卡）                     |108.6毫秒/步（8卡）|
| 总时长                 | 301分钟                          | 1100分钟|
| 参数(M)             | 44.6                                                        | 44.6 |
| 微调检查点| 343M（.ckpt文件）                                         | 343M（.ckpt文件）     |
|配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

#### ImageNet2012上的ResNet152

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | ResNet152  |
| 资源  |   Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期  |2021-02-10 ;  |
| MindSpore版本  | 1.0.1 |
| 数据集  |  ImageNet2012 |
| 训练参数  | epoch=140, steps per epoch=5004, batch_size = 32  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 1.7375104  |
|速度|47.47毫秒/步（8卡） |
|总时长   |  577分钟 |
|参数(M)   | 60.19 |
|  微调检查点 | 462M（.ckpt文件）  |
| 配置文件  | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config)  |

#### ImageNet2012上的SE-ResNet50

| 参数                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| 模型版本              | SE-ResNet50                                               |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  |
| 上传日期              | 2021-07-05  ；                        |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet2012                                                |
| 训练参数        | epoch=24, steps per epoch=5004, batch_size = 32             |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 1.754404                                                    |
| 速度                      | 24.6毫秒/步（8卡）                     |
| 总时长                 | 49.3分钟                                                  |
| 参数(M)             | 25.5                                                         |
| 微调检查点 | 215.9M （.ckpt文件）                                         |
|配置文件                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet/config) |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

- **Q: 如何使用`boost`功能获取最优的性能？**

  **A**： 我们在`Model`中提供了`boost_level`的入参，当你将其设置为O1或者O2模式时，框架会自动对网络的性能进行优化。当前这个模式已在resnet50上充分验证，你可以使用`resnet50_imagenet2012_Boost_config.yaml`来体验该模式。同时，在O1或者O2模式下，建议设置以下环境变量:`export  ENV_FUSION_CLEAR=1;export DATASET_ENABLE_NUMA=True;export ENV_SINGLE_EVAL=1;export SKT_ENABLE=1;`来获取更好的性能。

- **Q: 如何使用对ImageNet2012数据集进行预处理？**

  **A**： 建议参考https://bbs.huaweicloud.com/forum/thread-134093-1-1.html