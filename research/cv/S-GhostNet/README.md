# 目录

<!-- TOC -->

- [目录](#目录)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本结构与说明](#脚本结构与说明)
- [训练过程](#训练过程)
    - [用法](#用法)
        - [Ascend处理器环境运行](#ascend处理器环境运行)
- [评估过程](#评估过程)
    - [用法](#用法-1)
        - [Ascend处理器环境运行](#ascend处理器环境运行-1)
    - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GhostNet描述

## 概述

S-GhostNet由华为诺亚方舟实验室在2021年提出，此网络在GhostNet的基础上，通过网络结构搜索方法探索了大模型的构建方法。旨在更低计算代价下，提供更优的性能。该架构可以在同样计算量下，精度优于SOTA算法。

如下为MindSpore使用ImageNet2012数据集对GhostNet进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/2108.00177.pdf): Chuanjian Liu, Kai Han, An Xiao, Yiping Deng, Wei Zhang, Chunjing Xu, Yunhe Wang."Greedy Network Enlarging"

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
    ├─imagenet
        ├─train                  # 训练数据集
        └─val                    # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本结构与说明

```text
└──S-GhostNet
  ├── README.md
  ├── script
    ├── ma-pre-start.sh                    # Modelarts训练日志保存
    ├── train_distributed_ascend.sh        # 单机分布式训练脚本
  ├── src
    ├── autoaug.py                         # 数据自动增强
    ├── dataset.py                         # 数据预处理
    ├── bignet.py                          # S-GhostNet网络定义
    ├── callback.py                        # 模型参数滑动平均
    ├── eval_callback.py                   # 训练过程中对模型测试
    ├── loss.py                            # 模型训练损失定义
    ├── utils.py
    └── ghostnet.py                        # ghostnet网络
  ├── eval_b1.py                           # 评估网络S-GhostNet-b1
  ├── eval_b4.py                           # 评估网络S-GhostNet-b4
  └── train.py                             # 训练网络
  └── compute_acc.py                       # 统计准确率
```

# 训练过程

## 用法

### ascend处理器环境运行

```Shell
# 分布式训练
用法:bash train_distributed_ascend.sh [bignet] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)

# 单机训练
用法:python train.py --model big_net --data_path path-to-imagent --drop 0.2 --drop-path 0.2 --large --layers 2,4,5,12,6,14 --channels 36,68,108,164,216,336 --batch-size 4

Modelarts训练
python train.py --model big_net --amp_level=O0 --autoaugment --batch-size=32 --channels=36,60,108,168,232,336 --ckpt_save_epoch=20 --cloud= --data_url=obs://path-to-imagenet --decay-epochs=2.4 --decay-rate=0.97 --device_num=8 --distributed --drop=0.3 --drop-path=0.1 --ema-decay=0.97 --epochs=450 --input_size=384 --large --layers=3,5,5,12,6,14 --loss_scale=1024 --lr=0.0001 --lr_decay_style=cosine --lr_end=1e-6 --lr_max=0.6 --model=big_net --opt=momentum --opt-eps=0.0001 --sync_bn --warmup-epochs=20 --warmup-lr=1e-6 --weight-decay=2e-5 --workers=8

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

# 评估过程

## 用法

### ascend处理器环境运行

```Shell
# Modelarts评估
Usage: python eval_b4.py --data_url=obs://path-to-imagenet --large= --model=big_net --test_mode=ema_best --train_url=obs://path-to-pretrained-model --trained_model_dir=s3://path-to-output
```

训练过程中可以生成检查点。

## 结果

评估结果保存在日志文件中。您可通过compute_acc.py统计分布式训练下不同checkpoint的结果：

# 模型描述

S-GhostNet_b1 --channels=28,44,72,140,196,280 --layers=1,2,3,4,3,6 --input_size=240 --large
Top-1=79.844
S-GhostNet_b4 --channels=36,60,108,168,232,336 --layers=3,5,5,12,6,14 --input_size=384 --large
Top-1=83.024

## 性能

### 评估性能

| 参数 | Ascend 910  |
|---|---|
| 模型版本  | S-GhostNet |
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期  |2022-04-29 ;  |
| MindSpore版本  | 1.5.1 |
| 数据集  |  ImageNet2012 |
| 训练参数  | epoch=450 |
| 优化器  | Momentum  |
| 损失函数  |Softmax交叉熵  |
|总时长   |  S-GhostNet_b4 32卡 112小时 |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。
