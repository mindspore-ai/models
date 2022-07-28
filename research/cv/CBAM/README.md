# 目录

- [cbam说明](#cbam说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#训练用法)
    - [评估过程](#评估过程)
        - [用法](#评估用法)
        - [结果](#评估结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# CBAM说明

CBAM(Convolutional Block Attention Module)是一种轻量级注意力模块的提出于2018年，它可以在空间维度和通道维度上进行Attention操作。

[论文](https://arxiv.org/abs/1807.06521)：  Sanghyuan Woo, Jongchan Park, Joon-Young Lee, In So Kweon. CBAM: Convolutional Block Attention Module.

# 模型架构

CBAM整体网络架构如下：

[链接](https://arxiv.org/abs/1807.06521)

# 数据集

使用的数据集：[RML2016.10A](https://www.xueshufan.com/publication/2562146178)

- 数据集大小：共611M，总共有22万条样本。
    - 训练集：110000条样本。
    - 测试集：取一个信噪比下的数据，含有样本5500条。
- 数据格式：IQ（In-phaseand Quadrature）：2*128。
    - 注：数据在src/data中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```path
.
└─CBAM
  ├─src
    ├─data.py                       # 数据集处理
    ├─model.py                      # CBAM网络定义
    ├─get_lr.py                     # 生成学习率
    ├─model_utils
      ├─config.py                   # 参数配置
      ├─device_adapter.py           # 适配云上或线下
      ├─local_adapter.py            # 线下配置
      ├─moxing_adapter.py           # 云上配置
  ├──eval.py                        # 评估网络
  ├──train.py                       # 训练网络
  ├──default_config.yaml            # 参数配置
  └──README.md                      # README文件
```

## 脚本参数

在default_config.yaml中可以同时配置训练和评估参数。

```python
"batch_size":32,                   # 输入张量的批次大小
"epoch_size":70,                   # 训练周期大小
"lr_init":0.001,                   # 初始学习率
"save_checkpoint":True,            # 是否保存检查点
"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":5,                 # 热身周期
```

## 训练过程

### 训练用法

首先需要在`default_config.yaml`中设置好超参数。

您可以通过华为云等资源开始训练，其中配置如下所示：

```shell
Ascend:
   训练输入：data_url = /cbam/dataset,
   训练输出：train_url = /cbam/train_output,
   输出日志：/cbam/train_logs
```

### 训练结果

Ascend评估结果保存在`/cbam/train_logs`下。您可以在日志中找到类似以下的结果。

```log
epoch: 1 step: 3437, loss is 0.7258548
epoch: 2 step: 3437, loss is 0.6980165
epoch: 3 step: 3437, loss is 0.6887816
epoch: 4 step: 3437, loss is 0.7017617
epoch: 5 step: 3437, loss is 0.694684
```

## 评估过程

### 评估用法

与训练相同，在`default_config.yaml`中设置好超参数，通过华为云平台进行训练：

```shell
Ascend:
训练输入：data_url = /cbam/dataset,
训练输入：ckpt_file = /cbam/train_output/cbam_train-70_3437.ckpt,
输出日志：/cbam/eval_logs
```

### 评估结果

Ascend评估结果保存在`/cbam/eval_logs`下。您可以在日志中找到类似以下的结果。

```log
result: {'Accuracy': 0.8494152046783626}
```

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend 910                                                  |
| -------------------------- | ---------------------------------------------------------- |
| 资源                       | Ascend 910                                                  |
| 上传日期                   | 2022-5-31                                                    |
| MindSpore版本              | 1.5.1                                                       |
| 数据集                     | RML2016.10A                                                 |
| 训练参数                   | default_config.yaml                                          |
| 优化器                     | Adam                                                         |
| 损失函数                   | BCEWithLogitsLoss                                             |
| 损失                       |  0.6702158                                                    |
| 准确率                     | 84.9%                                                         |
| 总时长                     | 41分钟 （1卡）                                              |
| 调优检查点                 | 5.80 M（.ckpt文件）                                              |

# 随机情况说明

在train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

