目录

<!-- TOC -->

- [目录](#目录)
    - [网络描述](#网络描述)
    - [数据集](#数据集)
        - [Flash-MNIST数据集](#Flash-MNIST数据集)
    - [环境要求](#环境要求)
    - [快速开始](#快速开始)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [训练](#训练)
            - [训练结果](#训练结果)
        - [评估过程](#评估过程)
            - [评估](#评估)
            - [评估结果](#评估结果)
        - [导出过程](#导出过程)
            - [导出](#导出)
        - [推理过程](#推理过程)
            - [推理](#推理)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
            - [推理性能](#推理性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## 网络描述

Attention Cluster模型为ActivityNet Kinetics Challenge 2017中最佳序列模型。该模型通过带Shifting Opeation的Attention Clusters处理已抽取好的RGB、Flow、Audio特征数据。Shifting Operation通过对每一个attention单元的输出添加一个独立可学习的线性变换处理后进行L2-normalization，使得各attention单元倾向于学习特征的不同成分，从而让Attention Cluster能更好地学习不同分布的数据，提高整个网络的学习表征能力。

详细内容请参考[Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550)

## 数据集

### Flash-MNIST数据集

假设存放视频模型代码库的主目录为: Code\_Root

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)

- 数据集大小：52.4M，共10个类，6万张 28*28图像
    - 训练集：6万张图像
    - 测试集：5万张图像
- 数据格式：二进制文件
- 原始MNIST数据集目录结构:

```bash
.
└── mnist_dataset_dir
     ├── t10k-images-idx3-ubyte
     ├── t10k-labels-idx1-ubyte
     ├── train-images-idx3-ubyte
     └── train-labels-idx1-ubyte
```

通过如下方式准备数据:

```bash
    bash $Code_Root/scripts/make_dataset.sh
```

- 用于模型训练的数据集目录结构如下，也可以参考原作者仓库中的[实现](<https://github.com/longxiang92/Flash-MNIST.git>)：

```bash
.
└─ Datasets
    ├─ feature_train.pkl
    └─ feature_test.pkl
```

## 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 安装[MindSpore](https://www.mindspore.cn/install)
- 安装相关依赖 pip install -r requirements.txt
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速开始

准备好数据集后，即可按顺序依次进行模型训练与评估操作：

- 训练：

```shell
# 单卡训练
bash run_standalone_train.sh [FC][NATT][EPOCHS][DATASET_DIR][RESULT_DIR][DEVICE_ID]

# Ascend多卡训练
bash bash run_distribution_train.sh [RANK_TABLE][RANK_SIZE][DEVICE_START][FC][NATT][EPOCHS][DATASET_DIR][RESULT_DIR]
```

- 评估：

```shell
# 评估
bash bash run_eval.sh [FC][NATT][DATASET_DIR][CHECKPOINT_PATH][DEVICE_ID]
```

## 脚本说明

### 脚本及样例代码

```text
.
└─AttentionCluster
  ├─scripts
  | ├─make_dataset.sh             # 构建数据集脚本
  | ├─run_eval.sh                 # 评估运行脚本
  | ├─run_distribution_train.sh   # 多卡训练脚本
  | └─run_standalone_train.sh     # 单卡训练脚本
  |
  ├─src
  | ├─datasets
  | | ├─mnist.py                  # 读取mnist数据集
  | | ├─mnist_feature.py          # 提取合并后的mnist图像的特征
  | | ├─mnist_flash.py            # 将多张mnist图像合并
  | | ├─mnist_noisy.py            # 从原始mnist图像生成含有噪声的图像
  | | └─mnist_sampler.py          # 从原始mnist数据集中进行采样
  | |
  | ├─models
  | | └─attention_cluster.py      # AttentionCluster模型定义
  | |
  | └─utils
  |   └─config.py                 # 参数配置
  |
  ├─README_CN.md                  # 中文指南
  ├─requirements.txt              # 依赖包
  ├─eval.py                       # 评估
  ├─export.py                     # 模型导出
  ├─train.py                      # 训练
  ├─postprocess.py                # 计算准确率
  ├─preprocess.py                 # 数据集转换
  └─make_dataset.py               # 构建数据集
```

### 脚本参数

训练参数可以在`config.py`中配置。

```text
"lr": 0.001,                      # 学习率
"epochs": 200,                    # 训练轮次
"fc": 1,                          # 使用的全连接层的类型
"weight_decay": 0,                # 权重衰减
"batch-size": 64,                 # batch大小
"natt": 1,                        # attention头的数量
```

如需查阅更多参数信息，请参阅`config.py`脚本内容。

### 训练过程

#### 训练

```shell
# 单卡训练
bash run_standalone_train.sh 1 1 200 '../data' '../results' 0

# Ascend多卡训练（8P）
bash bash run_distribution_train.sh './rank_table_8pcs.json' 8 0 1 1 200 '../data' '../results'
```

其中，Ascend多卡训练还需要将相应`RANK_TABLE_FILE`文件的放置目录输入脚本（如`./rank_table_8pcs.json`），`RANK_TABLE_FILE`可按[此方法](#https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成。

#### 训练结果

训练时，当前训练轮次数，模型损失值，每轮次运行时间等信息会以如下形式显示：

```text
epoch: 1 step: 1600, loss is 6.558913
epoch time: 14510.303 ms, per step time: 9.069 ms
epoch: 2 step: 1600, loss is 6.3387423
epoch time: 10240.037 ms, per step time: 6.400 ms
epoch: 3 step: 1600, loss is 6.1619167
epoch time: 10197.713 ms, per step time: 6.374 ms
epoch: 4 step: 1600, loss is 6.1232996
epoch time: 10144.251 ms, per step time: 6.340 ms
epoch: 5 step: 1600, loss is 6.1846876
epoch time: 10158.274 ms, per step time: 6.349 ms
epoch: 6 step: 1600, loss is 6.040742
epoch time: 10141.502 ms, per step time: 6.338 ms
epoch: 7 step: 1600, loss is 6.1569977
epoch time: 10206.018 ms, per step time: 6.379 ms
epoch: 8 step: 1600, loss is 6.0411224
epoch time: 10160.674 ms, per step time: 6.350 ms
epoch: 9 step: 1600, loss is 6.008434
epoch time: 10316.776 ms, per step time: 6.448 ms
……
```

### 评估过程

#### 评估

在完成训练流程的基础上，评估流程将自动从checkpoints目录加载对应任务的最优检查点用于模型评估：

```shell
bash run_eval.sh 1 1 '../data' '../results/attention_cluster-200_1600.ckpt' 0
```

#### 评估结果

评估后模型在验证集上的相关评估结果将以如下形式显示：

```text
result: {'top_1_accuracy': 0.85205078125, 'top_5_accuracy': 0.9771484375} ckpt: ../results/attention_cluster-200_1600.ckpt
```

### 导出过程

#### 导出

将checkpoint文件导出成mindir格式模型：

```shell
Python export.py  --fc [FC] --natt [NATT] --ckpt [CHECKPOINT_PATH]
```

### 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

#### 推理

在导出模型后我们可以进行推理，以下展示了使用mindir模型执行推理的示例：

```shell
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

## 模型描述

### 性能

#### 评估性能

| 参数                 | Ascend                                                            |
| -------------------------- | -------------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；操作系统 Euler2.8                                            |
| 上传日期              | 2021-10-15                                    |
| MindSpore版本          | 1.3.0                                                     |
| 数据集                    | MNIST                                                 |
| 优化器                 | Adam                                                           |
| 损失函数              | Softmax交叉熵                                          |
| 脚本                    | [AttentionCluster](https://gitee.com/mindspore/models/tree/master/research/cv/AttentionCluster) |

|  fc类型 | natt数量 |  epoches | 学习率  | weight decay | t-1 accuracy(%) |
|-------|--------|----------|------|--------------|-----------------|
| 1     | 1      | 200      | 1e-4 | 0            | 34.5            |
| 1     | 2      | 200      | 1e-4 | 0            | 65.8            |
| 1     | 4      | 200      | 1e-4 | 0            | 72.7            |
| 1     | 8      | 200      | 1e-4 | 0            | 80.5            |
| 1     | 16     | 200      | 5e-4 | 1e-4            | 84.6            |
| 1     | 32     | 200      | 5e-4 | 1e-4           | 84.7            |
| 1     | 64     | 200      | 5e-4 | 1e-4            | 85.0            |
| 1     | 128    | 200      | 5e-4 | 1e-4            | 85.2            |
| 2     | 1      | 200      | 1e-4 | 0            | 54.1            |
| 2     | 2      | 200      | 1e-4 | 0            | 64.7            |
| 2     | 4      | 200      | 1e-4 | 0            | 73.8            |
| 2     | 8      | 200      | 5e-4 | 1e-4            | 76.4            |
| 2     | 16     | 200      | 5e-4 | 1e-4            | 86.6            |
| 2     | 32     | 200      | 5e-4 | 1e-4            | 87.0            |
| 2     | 64     | 200      | 5e-4 | 1e-4            | 87.3            |
| 2     | 128    | 200      | 5e-4 | 1e-4            | 87.4            |

#### 推理性能

| 参数                 | Ascend                                                            |
| -------------------------- | -------------------------------------------------------------- |
| 模型版本              | fc 1；natt 8                                          |
| 资源                   | Ascend 310                                            |
| 上传日期              | 2021-11-10                                    |
| MindSpore版本          | 1.3.0                                                     |
| 数据集                    | MNIST                                                 |
| 准确率                    | 80.5% |

## 随机情况说明

train.py和eval.py脚本中使用mindspore.common.set_seed()对随机种子进行了固定，可在对应的args中进行修改即可。

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。