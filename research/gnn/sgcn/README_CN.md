# 目录

<!-- TOC -->

- [目录](#目录)
    - [符号图卷积网络描述](#符号图卷积网络描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速开始](#快速开始)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
        - [评估过程](#评估过程)
        - [MINDIR模型导出过程](#MINDIR模型导出过程)
    - [模型描述](#模型描述)
        - [性能](#性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## 符号图卷积网络描述

符号图卷积网络（SGCN）于2018年提出，旨在对符号图结构数据进行学习。作者重新设计了GCN模型，根据平衡理论，定义平衡路径，维护‘friend’表示和‘enemy’表示，一起作为顶点的表达。

> [论文](https://arxiv.org/abs/1808.06354):  Signed Graph Convolutional Network. Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.

## 模型架构

SGCN根据正负连接分别包含三个图卷积层。每一层都以相应的正向/负向连接边作为输入数据。网络的loss计算由三部分组成，分别是只考虑正向连接的损失，只考虑负向连接的损失以及两者都考虑的回归损失。

## 数据集

实验基于两个与Bitcoin相关取自真实世界的[数据集：Bitcoin-Alpha和Bitcoin-OTC](https://github.com/benedekrozemberczki/SGCN/tree/master/input)，这两个数据集都来自致力于建立开放市场的网站，用户可以在这些网站上使用比特币买卖东西。由于比特币账户是匿名的，网站的用户为了安全，都开通了网络信任权限。这让用户可以积极(或消极)评价他们信任(或不信任)的人，这有助于解决交易中可能存在的欺诈问题。实验中，划分其中80%作为训练集，20%作为测试集。

| 数据集  | 用户数量 | 正向连接 | 负向连接 |
| -------  | ---------------:|-----:| ----:|
| Bitcoin-Alpha    |3784 | 12729  | 1416  |
| Bitcoin-OTC| 5901 |18390  | 3132  |

## 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 安装[MindSpore](https://www.mindspore.cn/install)
- 安装相关依赖 pip install -r requirements.txt
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 下载数据集Bitcoin-Alpha和Bitcoin-OTC，[可点此下载](https://github.com/benedekrozemberczki/SGCN/tree/master/input)并放到根目录下input文件夹中。

## 快速开始

通过官方指南安装[MindSpore](https://www.mindspore.cn/install)后，下载[数据集](https://github.com/benedekrozemberczki/SGCN/tree/master/input)，将下载好的数据集按如下目录结构进行组织，也可按此结构自行添加数据集：

```text
.
└─input
    ├─bitcoin_alpha.csv
    └─bitcoin_otc.csv
```

准备好数据集后，即可按顺序依次进行模型训练与评估操作：

- 训练：

```shell
# 单卡训练
bash ./scripts/run_standalone_train.sh [DEVICE_ID]

# Ascend多卡训练
bash ./scripts/run_distributed_train.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATA_PATH] [DISTRIBUTED]
```

示例：

```shell
# 单卡训练
bash ./scripts/run_standalone_train.sh 0

# Ascend多卡训练（8P）
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

- 评估：

```shell
# 评估
bash ./scripts/run_eval.sh [checkpoint_auc] [checkpoint_f1]
```

示例：

```shell
# 评估
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt
```

## 脚本说明

### 脚本及样例代码

```text
.
└─sgcn
  ├─README_CN.md                  # 中文指南
  |
  ├─scripts
  | ├─run_export.sh               # 模型导出运行脚本
  | ├─run_eval.sh                 # 评估运行脚本
  | ├─run_distributed_train.sh    # 多卡训练脚本
  | └─run_standalone_train.sh     # 单卡训练脚本
  |
  ├─src
  | ├─param_parser.py               # 参数配置
  | ├─ms_utils.py                   # 功能函数定义
  | ├─sgcn.py                       # SGCN骨干
  | ├─signedsageconvolution.py      # 定义图卷积层
  | └─metrics.py                    # 计算损失和反向传播
  |
  ├─requirements.txt                # 依赖包
  ├─train.py                        # 训练
  ├─eval.py                         # 评估
  └─export.py                       # 模型导出
```

### 脚本参数

训练参数可以在`param_parser.py`中配置。

```text
"learning-rate": 0.01,            # 学习率
"epochs": 500,                    # 训练轮次
"lamb": 1.0,                      # Embedding正则化参数
"weight_decay": 1e-5,             # 第一图卷积层参数的权重衰减
"test-size": 0.2,                 # 测试集比例
```

如需查阅更多参数信息，请参阅`param_parser.py`脚本内容。

### 训练过程

#### 运行

```shell
# 单卡训练
bash ./scripts/run_standalone_train.sh 0

# Ascend多卡训练（8P）
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

其中，Ascend多卡训练还需要将相应`RANK_TABLE_FILE`文件的放置目录输入脚本（如`./rank_table_8pcs.json`），`RANK_TABLE_FILE`可按[此方法](#https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成。

#### 结果

训练时，当前训练轮次数，模型损失值，每轮次运行时间等信息会以如下形式显示，且运行日志将保存至`./logs/train.log`：

```text
Epoch: 0001 train_loss= 1.2315195 time= 58.43749475479126 auc= 0.6405850837700487 f1= 0.0
Best checkpoint has been saved.
Epoch: 0002 train_loss= 0.9297038 time= 4.103950023651123 auc= 0.6986874053927796 f1= 0.0
Best checkpoint has been saved.
Epoch: 0003 train_loss= 0.8729827 time= 3.8731346130371094 auc= 0.711222902736952 f1= 0.9205789804908747
Best checkpoint has been saved.
...
Epoch: 0496 train_loss= 0.74630475 time= 3.708709239959717 auc= 0.8478284704192771 f1= 0.7298772169167804
Epoch: 0497 train_loss= 0.72714406 time= 3.710022211074829 auc= 0.8591252171659588 f1= 0.8952187182095626
Epoch: 0498 train_loss= 0.7302228 time= 3.708075761795044 auc= 0.8628629201232294 f1= 0.9127118644067798
Epoch: 0499 train_loss= 0.7248177 time= 3.70865535736084 auc= 0.8643315665373965 f1= 0.877109860823216
Epoch: 0500 train_loss= 0.73416984 time= 3.7085561752319336 auc= 0.8536205015932364 f1= 0.6415094339622641
Training fished! The best AUC and F1-Score is: 0.8777582622736414 0.9402768622280818 Total time: 1913.0554835796356
******************** finish training! ********************
```

### 评估过程

#### 运行

在完成训练流程的基础上，评估流程将自动从checkpoints目录加载对应任务的最优检查点用于模型评估。

```shell
# 评估
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt
```

#### 结果

这里以Bitcoin-OTC数据集为例，评估后模型在验证集上的相关评估结果将以如下形式显示，且运行日志将保存至`./logs/eval.log`：

```text
=====Evaluation Results=====
AUC: 0.866983
F1-Score: 0.930903
============================
```

### MINDIR模型导出过程

#### 运行

在完成训练流程的基础上，MINDIR模型导出流程将自动从checkpoints目录加载对应任务的最优检查点用于相应MINDIR模型导出。

```shell
# MINDIR模型导出
bash ./scripts/run_export.sh 0
```

#### 结果

若模型导出成功，程序将显示如下，且运行日志将保存至`./logs/export.log`：

```text
==========================================
sgcn.mindir exported successfully!
==========================================
```

## 模型描述

### 性能

| 参数                 | SGCN                                                            |
| -------------------------- | -------------------------------------------------------------- |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；操作系统 Euler2.8                                            |
| 上传日期              | 2021-10-01                                    |
| MindSpore版本          | 1.3.0                                                     |
| 数据集                    | Bitcoin-OTC / Bitcoin-Alpha                                                 |
| 训练参数        | epoch=500, lr=0.01, weight_decay=1e-5                                                      |
| 优化器                 | Adam                                                           |
| 损失函数              | Softmax交叉熵                                          |
| AUC                   | 0.8663 / 0.7979                                                      |
| F1-Score             | 0.9309 / 0.9527                                                    |
| 脚本                    | [SGCN](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |

## 随机情况说明

train.py和eval.py脚本中使用mindspore.set_seed()对全局随机种子进行了固定，可在对应的parser中进行修改即可。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
