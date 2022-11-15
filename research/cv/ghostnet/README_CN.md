# 目录

<!-- TOC -->

- [目录](#目录)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本结构与说明](#脚本结构与说明)
- [脚本参数](#脚本参数)
- [训练过程](#训练过程)
    - [用法](#用法)
        - [Ascend处理器环境运行](#Ascend处理器环境运行)
    - [结果](#结果)
- [评估过程](#评估过程)
    - [用法](#用法-1)
        - [Ascend处理器环境运行](#Ascend处理器环境运行-1)
    - [结果](#结果-1)
- [推理过程](#推理过程)
    - [导出MindIR](#导出MindIR)
    - [在Ascend310执行推理](#在Ascend310执行推理)
    - [结果](#结果)
- [迁移训练过程](#迁移训练过程)
    - [迁移数据集处理](#迁移数据集处理)
    - [迁移训练Ckpt获取](#迁移训练Ckpt获取)
    - [用法](#用法)
    - [结果](#结果)
- [迁移训练推理过程](#迁移训练推理过程)
    - [用法](#用法)
    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GhostNet描述

## 概述

GhostNet由华为诺亚方舟实验室在2020年提出，此网络提供了一个全新的Ghost模块，旨在通过廉价操作生成更多的特征图。基于一组原始的特征图，作者应用一系列线性变换，以很小的代价生成许多能从原始特征发掘所需信息的“幻影”特征图（Ghost feature maps）。该Ghost模块即插即用，通过堆叠Ghost模块得出Ghost bottleneck，进而搭建轻量级神经网络——GhostNet。该架构可以在同样精度下，速度和计算量均少于SOTA算法。

如下为MindSpore使用ImageNet2012数据集对GhostNet进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1911.11907.pdf): Kai Han, Yunhe Wang, Qi Tian."GhostNet: More Features From Cheap Operations"

# 模型架构

GhostNet的总体网络架构如下：[链接](https://arxiv.org/pdf/1911.11907.pdf)

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

```text
└─dataset
    ├─ilsvrc                  # 训练数据集
    └─validation_preprocess   # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```Shell
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法：sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 运行评估示例
用法：sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

# 脚本说明

## 脚本结构与说明

```text
└──ghostnet
  ├── README.md
  ├── ascend310_infer                      # ascend310推理
    ├── inc
      └──  utils.h                         # ascend310推理
    ├── src
      ├── build.sh                         # ascend310推理
      ├── CMakeLists.txt                   # ascend310推理
      ├── main.cc                          # ascend310推理
      └──  utils.cc                        # ascend310推理
    ├── scripts
      ├── run_distribute_train.sh          # 启动Ascend分布式训练（8卡）
      ├── run_eval.sh                      # 启动Ascend评估
      ├── run_eval_onnx.sh                 # 启动ONNX评估
      ├── run_infer_310.sh                 # 启动Ascend310推理
      └── run_standalone_train.sh          # 启动Ascend单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── CrossEntropySmooth.py              # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    ├── dense.py                           # 调整预训练模型全连接层
    ├── data_split.py                      # 切分迁移数据集脚本
    ├── ghostnet600.py
    ├── launch.py
    └── ghostnet.py                        # ghostnet网络
  ├── eval.py                              # 评估网络
  ├── eval_onnx.py                         # ONNX评估
  ├── create_imagenet2012_label.py         # 创建ImageNet2012标签
  ├── export.py                            # 导出MindIR模型
  ├── postprocess.py                       # 310推理的后期处理
  ├── requirements.txt                     # 需求文件
  ├── fine_tune.py                         # 迁移训练网络
  └── train.py                             # 训练网络
```

# 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置GhostNet和ImageNet2012数据集。

```Python
"num_classes": 1000,           # 数据集类数
"batch_size": 128,             # 输入张量的批次大小
"epoch_size": 500,             # 训练周期大小
"warmup_epochs": 20,           # 热身周期数
"lr_init": 0.1,                # 基础学习率
"lr_max": 0.4,                 # 最大学习率
'lr_end': 1e-6,                # 最终学习率
'lr_decay_mode': 'cosine',     # 用于生成学习率的衰减模式
"momentum": 0.9,               # 动量优化器
"weight_decay": 4e-5,          # 权重衰减
"label_smooth": 0.1,           # 标签平滑因子
"loss_scale": 128,             # 损失等级
"use_label_smooth": True,      # 标签平滑
"label_smooth_factor": 0.1,    # 标签平滑因子
"save_checkpoint": True,       # 是否保存检查点
"save_checkpoint_epochs": 20,  # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max": 10,     # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path": "./",  # 检查点相对于执行路径的保存路径
```

# 训练过程

## 用法

### Ascend处理器环境运行

```Shell
# 分布式训练
用法:sh run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 单机训练
用法:sh run_standalone_train.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## 结果

- 使用ImageNet2012数据集训练GhostNet

```text
# 分布式训练结果（8P）
epoch: 1 step: 1251, loss is 5.001419
epoch time: 457012.100 ms, per step time: 365.317 ms
epoch: 2 step: 1251, loss is 4.275552
epoch time: 280175.784 ms, per step time: 223.961 ms
epoch: 3 step: 1251, loss is 4.0788813
epoch time: 280134.943 ms, per step time: 223.929 ms
epoch: 4 step: 1251, loss is 4.0310946
epoch time: 280161.342 ms, per step time: 223.950 ms
epoch: 5 step: 1251, loss is 3.7326777
epoch time: 280178.602 ms, per step time: 223.964 ms
...
```

# 评估过程

## 用法

### Ascend处理器环境运行

```Shell
# 评估
Usage: sh run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

```Shell
# 评估示例
sh  run_eval.sh  /data/dataset/ImageNet/imagenet_original  ghostnet-500_1251.ckpt
```

训练过程中可以生成检查点。

## 结果

评估结果保存在示例路径中，文件夹名为“eval”。您可在此路径下的日志找到如下结果：

- 使用ImageNet2012数据集评估GhostNet

```text
result: {'top_5_accuracy': 0.9162371134020618, 'top_1_accuracy': 0.739368556701031}
ckpt = /home/lzu/ghost_Mindspore/scripts/device0/ghostnet-500_1251.ckpt
```

# 推理过程

## [导出MindIR](#contents)

```shell
python export.py --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT] --checkpoint_path [CKPT_PATH]
```

参数ckpt_file为必填项，
`FILE_FORMAT` 必须在 ["AIR", "ONNX", "MINDIR"]中选择。

## 在Ascend310执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理前， mindir文件必须通过export.py脚本导出。以下展示了使用mindir模型执行推理的示例。目前仅支持batch_Size为1的推理。

```shell
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- DEVICE_ID 可选，默认值为0。

## 结果

推理结果保存在脚本执行的当前路径， 你可以在 acc.log 中看到以下精度计算结果。

- 使用 ImageNet2012 数据集评估 ghostnet

```shell
Total data: 50000, top1 accuracy: 0.73816, top5 accuracy: 0.9178.
```

# 迁移训练过程

## 迁移数据集处理

[根据提供的数据集链接下载数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz ),下载数据集后，将切分数据集脚本src.data_split.py放置在解压好的flower_photos目录下，运行data_split.py后会生成train文件夹及test文件夹，将train文件夹及test文件夹保存到fine_tune.py同级目录的新建文件夹dataset中。

## 迁移训练Ckpt获取

[根据提供的Ckpt链接下载预训练模型文件](https://download.mindspore.cn/model_zoo/research/cv/ghostnet/ ),下载“ghostnet_1x_pets.ckpt”、“ghostnet_nose_1x_pets.ckpt”以及“ghostnet600M_pets.ckpt”文件，保存在fine_tune.py同级目录新建pre_ckpt文件夹中。

## 用法

整理好文件后激活环境，开始训练。
因为ckpt仓有三个ckpt预训练文件，故迁移训练三个网络。

导入预训练模型“ghostnet_1x_pets.ckpt”，训练如下

```shell
python fine_tune.py --pre_trained ./pre_ckpt/ghostnet_1x_pets.ckpt
```

导入预训练模型“ghostnet_nose_1x_pets.ckpt”，训练如下

```shell
python fine_tune.py --pre_trained ./pre_ckpt/ghostnet_nose_1x_pets.ckpt
```

导入预训练模型“ghostnet600M_pets.ckpt”，训练如下

```shell
python fine_tune.py --pre_trained ./pre_ckpt/ghostnet600M_pets.ckpt
```

## 结果

- 使用flower_photos据集训练GhostNet

```text
# ghostnet_1x迁移训练结果
epoch: 1 step: 21, loss is 1.0636098384857178
Train epoch time: 295952.507 ms, per step time: 14092.977 ms
epoch: 2 step: 21, loss is 1.007066011428833
Train epoch time: 20480.770 ms, per step time: 975.275 ms
epoch: 3 step: 21, loss is 0.9204861521720886
Train epoch time: 20673.888 ms, per step time: 984.471 ms
...
epoch: 498 step: 21, loss is 0.5347862839698792
Train epoch time: 19795.049 ms, per step time: 942.621 ms
epoch: 499 step: 21, loss is 0.49817660450935364
Train epoch time: 19959.692 ms, per step time: 950.462 ms
epoch: 500 step: 21, loss is 0.5028425455093384
Train epoch time: 20185.629 ms, per step time: 961.220 ms
```

```text
# ghostnet_nose_1x迁移训练结果
epoch: 1 step: 21, loss is 1.1746268272399902
Train epoch time: 94845.916 ms, per step time: 4516.472 ms
epoch: 2 step: 21, loss is 1.0321934223175049
Train epoch time: 37248.247 ms, per step time: 1773.726 ms
epoch: 3 step: 21, loss is 0.9764260053634644
Train epoch time: 37365.344 ms, per step time: 1779.302 ms
...
epoch: 498 step: 21, loss is 0.5118361711502075
Train epoch time: 36716.475 ms, per step time: 1748.404 ms
epoch: 499 step: 21, loss is 0.5035715103149414
Train epoch time: 37642.484 ms, per step time: 1792.499 ms
epoch: 500 step: 21, loss is 0.49066391587257385
Train epoch time: 36474.781 ms, per step time: 1736.894 ms
```

```text
# ghostnet_600m迁移训练结果
epoch: 1 step: 21, loss is 1.2935304641723633
Train epoch time: 296802.766 ms, per step time: 14133.465 ms
epoch: 2 step: 21, loss is 1.356112003326416
Train epoch time: 44871.251 ms, per step time: 2136.726 ms
epoch: 3 step: 21, loss is 1.1128544807434082
Train epoch time: 45124.813 ms, per step time: 2148.801 ms
...
epoch: 498 step: 21, loss is 0.4896056652069092
Train epoch time: 45314.303 ms, per step time: 2157.824 ms
epoch: 499 step: 21, loss is 0.5079032182693481
Train epoch time: 45675.234 ms, per step time: 2175.011 ms
epoch: 500 step: 21, loss is 0.5031487345695496
Train epoch time: 45935.200 ms, per step time: 2187.390 ms
```

# 迁移训练推理过程

## 用法

设置好ckpt文件路径，通过python脚本开始推理。

ghostnet_1x网络推理

```shell
python eval.py --device_target CPU --checkpoint_path ./ckpt/ghostnet_1x_3-500_21.ckpt
```

ghostnet_nose_1x网络推理

```shell
python eval.py --device_target CPU --checkpoint_path ./ckpt/ghostnet_nose_1x-500_21.ckpt
```

ghostnet_600mx网络推理

```shell
python eval.py --device_target CPU --checkpoint_path ./ckpt/ghostnet600m_1-500_21.ckpt
```

## 结果

- 各个网络结构使用flower_photos数据集推理效果

```text
result: {'top_5_accuracy': 1.0, 'top_1_accuracy': 0.9207589285714286} ckpt= ./ckpt/ghostnet_1x_3-500_21.ckpt
```

```text
result: {'top_1_accuracy': 0.9252232142857143, 'top_5_accuracy': 1.0} ckpt= ./ckpt/ghostnet_nose_1x-500_21.ckpt
```

```text
result: {'top_1_accuracy': 0.9308035714285714, 'top_5_accuracy': 1.0} ckpt= ./ckpt/ghostnet_600m_1-500_21.ckpt
```

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | GhostNet |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2021-06-22 ;  |
| MindSpore版本  | 1.2.0 |
| 数据集  |  ImageNet2012 |
| 训练参数  | epoch=500, steps per epoch=1251, batch_size = 128  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 1.7887309  |
|速度|203.92毫秒/步（8卡） |
|总时长   |  39小时 |
|参数(M)   | 5.18 |
|  微调检查点 | 42.05M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet)  |

| 参数 | GPU; CPU  |
|:---:|:---:|
| 模型版本  | GhostNet |
| 资源  |  GeForce RTX 3090；CPU：3.60GHz，4核；内存：8G |
| 上传日期  |2022-09-05 ;  |
| MindSpore版本  | 1.8.1 |
| 数据集  |  flower_photos |
| 训练参数  | epoch=500, steps per epoch=21, batch_size = 128  |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
| 输出  | 概率 |
|  损失 | 0.5028425455093384  |
|速度|20185.629毫秒/步 |
|总时长   |  2.79小时 |
|参数(M)   | 5.18 |
|  微调检查点 | 29.9M（.ckpt文件）  |
| 脚本  | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet)  |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。