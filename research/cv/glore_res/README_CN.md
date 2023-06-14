
# 目录

<!-- TOC -->

- [目录](#目录)
- [Glore_resnet描述](#glore_resnet描述)
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
    - [训练结果](#训练结果)
    - [推理过程](#推理过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
    - [推理结果](#推理结果)
    - [onnx模型导出与推理](#onnx模型导出与推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [ImageNet2012上的Glore_resnet50](#imagenet2012上的glore_resnet50)
            - [ImageNet2012上的Glore_resnet101](#imagenet2012上的glore_resnet101)
            - [ImageNet2012上的Glore_resnet200](#imagenet2012上的glore_resnet200)
        - [推理性能](#推理性能)
            - [ImageNet2012上的Glore_resnet50](#imagenet2012上的glore_resnet50)
            - [ImageNet2012上的Glore_resnet101](#imagenet2012上的glore_resnet101)
            - [ImageNet2012上的Glore_resnet200](#imagenet2012上的glore_resnet200)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Glore_resnet描述

## 概述

卷积神经网络擅长提取局部关系，但是在处理全局上的区域间关系时显得低效，且需要堆叠很多层才可能完成，而在区域之间进行全局建模和推理对很多计算机视觉任务有益。为了进行全局推理，facebook research、新加坡国立大学和360 AI研究所提出了基于图的全局推理模块-Global Reasoning Unit，可以被插入到很多任务的网络模型中。glore_res200是在ResNet200的Stage2, Stage3中分别均匀地插入了2和3个全局推理模块的用于图像分类任务的网络模型。

如下为MindSpore使用ImageNet2012数据集对glore_res50进行训练的示例。glore_res50可参考[论文1](https://arxiv.org/pdf/1811.12814v1.pdf)

## 论文

1.[论文](https://arxiv.org/abs/1811.12814):Yunpeng Chenyz, Marcus Rohrbachy, Zhicheng Yany, Shuicheng Yanz, Jiashi Fengz, Yannis Kalantidisy

# 模型架构

glore_res的总体网络架构如下：
[链接](https://arxiv.org/pdf/1811.12814v1.pdf)

glore_res200网络模型的backbone是ResNet200, 在Stage2, Stage3中分别均匀地插入了了2个和3个全局推理模块。全局推理模块在Stage2和Stage 3中插入方式相同.

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像  
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下:

```text
└─dataset
    ├─train                # 训练数据集
    └─val                  # 评估数据集
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 分布式训练
用法:bash run_distribute_train.sh [TRAIN_DATA_PATH] [RANK_TABLE] [CONFIG_PATH] [EVAL_DATA_PATH]

# 单机训练
用法:bash run_standalone_train.sh [TRAIN_DATA_PATH] [DEVICE_ID] [CONFIG_PATH] [EVAL_DATA_PATH]

# 运行评估示例
用法:bash run_eval.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

- GPU处理器环境运行

```bash
# 分布式训练
用法:bash run_distribute_train_gpu.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [RANK_SIZE] [CONFIG_PATH]

# 单机训练
用法:bash run_standalone_train.sh [TRAIN_DATA_PATH] [DEVICE_ID] [CONFIG_PATH] [EVAL_DATA_PATH]

# 运行评估示例
用法:bash run_eval.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```shell
.
└──Glore_resnet
  ├── README.md
  ├── config
    ├── config_resnet50_ascend.yaml            # Ascend glore_resnet50配置
    ├── config_resnet50_gpu.yaml            # GPU glore_resnet50配置
    ├── config_resnet101_gpu.yaml            # GPU glore_resnet101配置
    ├── config_resnet200_ascend.yaml            # Ascend glore_resnet200配置
    └── config_resnet200_gpu.yaml            # GPU glore_resnet200配置
  ├── script
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_distribute_train_gpu.sh        # 启动GPU分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend、GPU推理（单卡）
    └── run_standalone_train_gpu.sh        # 启动Ascend、GPU单机训练（单卡）
  ├── src
    ├── _init_.py
    ├── config.py                   #参数配置
    ├── dataset.py                  # 加载数据集
    ├── autoaugment.py                     # AutoAugment组件与类
    ├── lr_generator.py             # 学习率策略
    ├── loss.py                            # ImageNet2012数据集的损失定义
    ├── save_callback.py                   # 训练时推理并保存最优精度下的参数
    ├── glore_resnet200.py          # glore_resnet200网络
    ├── glore_resnet50.py          # glore_resnet50网络
    ├── transform.py                # 数据增强
    └── transform_utils.py          # 数据增强
  ├── eval.py                       # 推理脚本
  ├── export.py                     # 将checkpoint导出
  └── train.py                      # 训练脚本
```

## 脚本参数

- 配置Glore_resnet50在ImageNet2012数据集参数(Ascend)。

```text
"class_num":1000,                # 数据集类数
"batch_size":128,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":120,                # 此值仅适用于训练；应用于推理时固定为1
"pretrained": False,             # 加载预训练权重
"pretrain_epoch_size": 0,        # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.05,      # 标签平滑因子
"weight_init": "xavier_uniform",      # 权重初始化方式,可选"he_normal", "he_uniform", "xavier_uniform"
"use_autoaugment": True,         # 是否应用AutoAugment方法
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置Glore_resnet50在ImageNet2012数据集参数(GPU)。

```text
"class_num":1000,                # 数据集类数
"batch_size":128,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":130,                # 此值仅适用于训练；应用于推理时固定为1
"pretrained": False,             # 加载预训练权重
"pretrain_epoch_size": 0,        # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.05,      # 标签平滑因子
"weight_init": "xavier_uniform",      # 权重初始化方式,可选"he_normal", "he_uniform", "xavier_uniform"
"use_autoaugment": True,         # 是否应用AutoAugment方法
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置Glore_resnet101在ImageNet2012数据集参数(GPU)。

```text
"class_num":1000,                # 数据集类数
"batch_size":64,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.08,                 # 动量优化器
"weight_decay":0.0002,           # 权重衰减
"epoch_size":150,                # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"poly",          # 用于生成学习率的衰减模式
"lr_init":0.1,                   # 初始学习率
"lr_max":0.4,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置Glore_resnet200在ImageNet2012数据集参数(Ascend)。

```text
"class_num":1000,                # 数据集类数
"batch_size":80,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.08,                 # 动量优化器
"weight_decay":0.0002,           # 权重衰减
"epoch_size":150,                # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"poly",          # 用于生成学习率的衰减模式
"lr_init":0.1,                   # 初始学习率
"lr_max":0.4,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置Glore_resnet200在ImageNet2012数据集参数(GPU)。

```text
"class_num":1000,                # 数据集类数
"batch_size":64,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.08,                 # 动量优化器
"weight_decay":0.0002,           # 权重衰减
"epoch_size":150,                # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"poly",          # 用于生成学习率的衰减模式
"lr_init":0.1,                   # 初始学习率
"lr_max":0.4,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法:bash run_distribute_train.sh [TRAIN_DATA_PATH] [RANK_TABLE] [CONFIG_PATH] [EVAL_DATA_PATH]

# 单机训练
用法:bash run_standalone_train.sh [TRAIN_DATA_PATH] [RANK_TABLE] [CONFIG_PATH] [EVAL_DATA_PATH]

# 运行推理示例
用法:bash run_eval.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

#### GPU处理器环境运行

```text
# 分布式训练
用法:bash run_distribute_train_gpu.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [RANK_SIZE] [CONFIG_PATH]
示例:bash run_distribute_train_gpu.sh ~/Imagenet_Original/train/ ~/Imagenet_Original/val/ 8 ../config/config_resnet50_gpu.yaml

# 单机训练
用法:bash run_standalone_train.sh [TRAIN_DATA_PATH] [CONFIG_PATH] [EVAL_DATA_PATH]

# 运行推理示例
用法:bash run_eval.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

## 训练结果

- 使用ImageNet2012数据集训练Glore_resnet50（8 pcs）

```text
# 分布式训练结果（8P）
epoch:1 step:1251, loss is 5.074506
epoch:2 step:1251, loss is 4.339285
epoch:3 step:1251, loss is 3.9819345
epoch:4 step:1251, loss is 3.5608528
epoch:5 step:1251, loss is 3.3024906
...
```

- 使用ImageNet2012数据集训练Glore_resnet101（8 pcs）

```text
# 分布式训练结果（8P）
epoch:1 step:5004, loss is 4.7398486
epoch:2 step:5004, loss is 4.129058
epoch:3 step:5004, loss is 3.5034246
epoch:4 step:5004, loss is 3.4452052
epoch:5 step:5004, loss is 3.148675
...
```

- 使用ImageNet2012数据集训练Glore_resnet200（8 pcs）

```text
# 分布式训练结果（8P）
epoch:1 step:1251, loss is 6.0563216
epoch:2 step:1251, loss is 5.3812423
epoch:3 step:1251, loss is 4.782114
epoch:4 step:1251, loss is 4.4079633
epoch:5 step:1251, loss is 4.080069
...
```

## 推理过程

### 用法

#### Ascend处理器环境运行

```bash
# 推理
Usage: bash run_eval.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# 推理示例
bash run_eval.sh ~/Imagenet_Original/val/ 0 ~/glore_resnet200-150_1251.ckpt ../config/config_resnet50_gpu.yaml
```

#### GPU处理器环境运行

```bash
# 推理
Usage: bash run_eval_gpu.sh [EVAL_DATA_PATH] [DEVICE_ID] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# 推理示例
bash run_eval.sh ~/Imagenet/val/  ~/glore_resnet200-150_2502.ckpt ../config/config_resnet50_gpu.yaml
```

## 推理结果

```text
result:{'top_1 acc':0.802303685897436}
```

## onnx模型导出与推理

- 导出 ONNX:  

  ```shell
  python export.py --config_path /path/to/glore.yaml --ckpt_url /path/to/glore_res50.ckpt --file_name /path/to/glore_res50 --batch_size 1 --file_format ONNX --device_target CPU
  ```

- 运行推理-python方式:

  ```shell
  python eval_onnx.py --config_path /path/to/glore.yaml --data_path /path/to/image_val/ --onnx_path /path/to/.onnx --batch_size 1 --device_target GPU > output.eval.log 2>&1
  ```

- 运行推理-bash方式:

  ```shell
  # 需要修改对应yaml配置文件的配置项
  bash scripts/run_eval_onnx.sh /path/to/glore.yaml
  ```

- 推理结果将存放在 output.eval.log 中.

# 模型描述

## 性能

### 训练性能

#### ImageNet2012上的Glore_resnet50

| 参数                 | Ascend 910                                   |          GPU                       |
| -------------------------- | -------------------------------------- |------------------------------------|
| 模型版本              | Glore_resnet50                            |Glore_resnet50                     |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：2048G |GPU-V100 PCIE 32G                     |
| 上传日期              | 2021-03-21                                  |2021-09-22                         |
| MindSpore版本          | r1.1                                  |1.3.0                          |
| 数据集                    | ImageNet2012                             | ImageNet2012                      |
| 训练参数        | epoch=120, steps per epoch=1251, batch_size = 128  |epoch=130, steps per epoch=1251, batch_size = 128 |
| 优化器                  | Momentum                                       | Momentum                                           |
| 损失函数              | SoftmaxCrossEntropyExpand                    |SoftmaxCrossEntropyExpand          |
| 输出                    | 概率                                       |概率                               |
| 损失                       |1.8464266                                |1.7463021                        |
| 速度                      | 263.483毫秒/步（8卡）                     |655 毫秒/步（8卡）             |
| 总时长                 | 10.98小时                                   |58.5 小时                          |
| 参数(M)             | 30.5                                            |30.5                          |
| 微调检查点| 233.46M（.ckpt文件）                                      |233.46M（.ckpt文件）                          |
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/r2.0/research/cv/glore_res) |

#### ImageNet2012上的Glore_resnet101

| 参数                 |          GPU                       |
| --------------------------|------------------------------------|
| 模型版本              |Glore_resnet101                     |
| 资源                   |GPU-V100 PCIE 32G                     |
| 上传日期              |2021-10-22                         |
| MindSpore版本          | r1.5                                  |1.5.0                          |
| 数据集                    | ImageNet2012                      |
| 训练参数        |epoch=150, steps per epoch=5004, batch_size = 32 |
| 优化器                  | NAG                                           |
| 损失函数              |SoftmaxCrossEntropyExpand          |
| 输出                    |概率                               |
| 损失                       |1.7463021                        |
| 速度                      |33 毫秒/步（8卡）             |
| 总时长                 |30 小时                          |
| 参数(M)             |57                          |
| 微调检查点|579.06M（.ckpt文件）                          |
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/r2.0/research/cv/glore_res) |

#### ImageNet2012上的Glore_resnet200

| 参数                 | Ascend 910                                   |          GPU                       |
| -------------------------- | -------------------------------------- |------------------------------------|
| 模型版本              | Glore_resnet200                             |Glore_resnet200                     |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：2048G |GPU-V100(SXM2)                     |
| 上传日期              | 2021-03-34                                   |2021-05-25                         |
| MindSpore版本          | 1.3.0                                   |1.2.0                          |
| 数据集                    | ImageNet2012                             | ImageNet2012                      |
| 训练参数        | epoch=150, steps per epoch=2001, batch_size = 80  |epoch=150, steps per epoch=2502, batch_size = 64 |
| 优化器                  | NAG                                        | NAG                                           |
| 损失函数              | SoftmaxCrossEntropyExpand                    |SoftmaxCrossEntropyExpand          |
| 输出                    | 概率                                       |概率                               |
| 损失                       |0.8068262                                |0.55614954                        |
| 速度                      | 400.343毫秒/步（8卡）                     |912.211 毫秒/步（8卡）             |
| 总时长                 | 33时35分钟                                   |94时08分                          |
| 参数(M)             | 70.6                                           |70.6                          |
| 微调检查点| 807.57M（.ckpt文件）                                      |808.28(.ckpt)                          |
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/r2.0/research/cv/glore_res) |

### 推理性能

#### ImageNet2012上的Glore_resnet50

| 参数          | Ascend                      |   GPU                        |
| ------------------- | ----------------------|------------------------------|
| 模型版本       | Glore_resnet50              |  Glore_resnet50          |
| 资源            | Ascend 910                |   GPU-V100 PCIE 32G                        |
| 上传日期       | 2021-03-21                  |2021-09-22                    |
| MindSpore版本   | r1.1                 |1.3.0                    |
| 数据集             | ImageNet2012测试集(6.4GB)              | ImageNet2012测试集(6.4GB)                   |
| batch_size          | 128                   |128                          |
| 输出             | 概率                     |概率                         |
| 准确性            | 8卡: 78.44%             |8卡：78.50%                 |

#### ImageNet2012上的Glore_resnet101

| 参数          | GPU                      |
| ------------------- | ----------------------|
| 模型版本       | Glore_resnet101              |
| 资源            | GPU-V100(SXM2)                |
| 上传日期       | 2021-10-22                  |
| MindSpore版本   | 1.5.0                 |
| 数据集             | ImageNet2012测试集(6.4GB)             |
| batch_size          | 32                   |
| 输出             | 概率                     |
| 准确性            | 8卡: 79.663%            |

#### ImageNet2012上的Glore_resnet200

| 参数          | Ascend                      |   GPU                        |
| ------------------- | ----------------------|------------------------------|
| 模型版本       | Glore_resnet200              |  Glore_resnet200           |
| 资源            | Ascend 910                |   GPU-V100(SXM2)                       |
| 上传日期       | 2021-3-24                  |2021-05-25                    |
| MindSpore版本   | 1.3.0                 |1.2.0                    |
| 数据集             | ImageNet2012测试集(6.4GB)             | ImageNet2012测试集(6.4GB)                   |
| batch_size          | 80                   |64                          |
| 输出             | 概率                     |概率                         |
| 准确性            | 8卡: 80.23%             |8卡：80.603%                 |

# 随机情况说明

transform_utils.py中使用数据增强时采用了随机选择策略，train.py中使用了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models/)
