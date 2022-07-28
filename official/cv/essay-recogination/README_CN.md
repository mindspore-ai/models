# 目录

<!-- TOC -->

- [目录](#目录)
    - [Eassay-Recognition描述](#eassay-recognition描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)

<!-- /TOC -->

## Eassay-Recognition描述

本模型实现了对Origaminet模型的几个改进:

1. 基于gate与并行多分支的特征提取模块
2. 基于reshape的2d降维
3. order-align策略解决对齐策略环节的旋转问题

[原论文](https://arxiv.org/abs/2006.07491): Mohamed Yousef, Tom E. Bishop, "OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page Text Recognition by learning to unfold, " arXiv:2006.07491

## 模型架构

示例：在MindSpore上使用自制初高中作文数据集基于预训练模型训练出模型进行作文整篇识别

## 数据集

[训练样例数据及字符集文件](https://github.com/IntuitionMachines/OrigamiNet#IAM)

下载解压后将/data_set文件夹放在项目根目录下，将alph.gc放在项目的parameters/文件夹中

## 环境要求

- 硬件 (GPU)
    - 使用GPU来搭建硬件环境
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

- 执行训练或评估脚本：
    - GPU环境运行

    ```bash
    #GPU单机训练示例：
    bash run_standalone_train.sh GPU

    #GPU评估示例
    bash run_eval.sh GPU
    ```

## 脚本说明

### 脚本及样例代码

```text
.
└──eassy-recognition
  ├── README.md                         # 文档说明
  ├── README_CN.md                      # 中文文档说明
  ├── script
    ├── run_eval.sh                     # 启动评估
    └── run_standalone_train.sh         # 启动单机训练（1卡）
  ├── src
    ├── cnv_model.py                     # 模型结构定义文件
    ├── ds_load.py             # 数据预处理及自定义数据集
    ├── util.py              # 字符下标转换以及order-align序列实现
    └── cnv_for_train.py             # 带梯度、CTC损失的自定义单步训练模型
  ├── model_ckpt
    └── origaminet.ckpt                     # 预训练模型
  ├── parameters
    ├── hwdb.gin                     # 训练参数配置
    ├── infer.gin                     # 评估参数配置
    ├── train.gc                     # 训练数据文件列表
    └── test.gc                     # 评估数据文件列表
  ├── eval.py                           # 评估网络
  └── train.py                          # 训练网络
```

### 参数配置

在parameters/hwdb.gin中配置训练参数

```text
train.train_data_list = 'parameters/train.gc'                 #训练数据集文件名列表
train.train_data_path = 'data_set/train'                      #训练数据集路径

train.train_batch_size = 1                                    #输入张量批次大小。
train.lr = 0.01                                               #初始学习率
train.save_model_path = './saved_models_finetune/'            #检查点保存位置
train.model_prefix = "model_finetune_"                        #检查点名称前缀
train.continue_model = 'model_ckpt/origaminet.ckpt'           #预训练模型位置
train.valInterval = 100                                       #边训练边推理的间隔epoch数
```

## 训练过程

- 在`parameters/hwdb.gin`中设置选项，包括学习率和网络超参数。单击[MindSpore加载数据集教程](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset.html)，了解更多信息。

### 训练

- 在GPU上运行`run_standalone_train.sh`进行非分布式训练。

``` bash
bash run_standalone_train.sh [TRAIN_DATA_DIR] [DEVICE_TARGET]
```

## 评估过程

### 评估

- 运行`run_eval.sh`进行评估。

``` bash
bash run_eval.sh [TEST_DATA_DIR] [DEVICE_TARGET]
```

## 模型描述

### 性能

#### 训练性能

| 参数                 |   GPU |
| -------------------------- |---------------------------------- |
| 模型版本              | v1.0 |
| 资源                   | GPU(GeForce RTX 3090)，CPU 2.9GHz 64核，内存： 256G
| 上传日期              | 2022-01-07 |
| MindSpore版本          | 1.5.0rc1       |
| 数据集                    | 自制数据集 |
| 训练参数        | epoch=100, steps per epoch=100, batch_size = 1  |
| 优化器                  | Adam |
| 损失函数             | CTCLoss |
| 输出                    | 概率 |
| 损失                       | 11.842  |
| 速度                      | 5450毫秒/步（卡）|
| 总时长                 | 15小时(1pcs)|
| 参数(M)             | 12.6 |
| 微调检查点 | 10.59M (.ckpt文件) |
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/essay-recogination) |

#### 评估性能

| 参数          | essay-recognition                     |
| ------------------- | --------------------------- |
| 模型版本       | V1.0                        |
| 资源            |GPU；系统 Ubuntu 18.04.6 LTS                 |
| 上传日期       | 2022-01-07 |
| MindSpore版本   | 1.5.0rc1                 |
| 数据集             | 自制数据集                     |
| batch_size          | 1                          |
| 输出             | nCER                         |
| 准确率            | 0.146937                     |
| 推理模型 | 10.59M (.ckpt文件)          |

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
