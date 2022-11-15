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
        - [Ascend310推理过程](#Ascend310推理过程)
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
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
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
bash ./scripts/run_standalone_train.sh [DEVICE_ID] [dataset]

# Ascend多卡训练
bash ./scripts/run_distributed_train.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATA_PATH] [DISTRIBUTED]
```

示例：

```shell
# 单卡训练
bash ./scripts/run_standalone_train.sh 0 ./input/bitcoin_otc.csv

# Ascend多卡训练（8P）
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

- 评估：

```shell
# 评估
bash ./scripts/run_eval.sh [checkpoint_auc] [checkpoint_f1] [dataset]
```

示例：

```shell
# 评估
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt ./input/bitcoin_otc.csv
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
  | ├─run_infer_310.sh            # Ascend310推理脚本
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
  ├─preprocess.py                   # 预处理
  ├─postprocess.py                  # 后处理
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
bash ./scripts/run_standalone_train.sh 0 ./input/bitcoin_otc.csv

# Ascend多卡训练（8P）
bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0 ./input/bitcoin_otc.csv True
```

其中，Ascend多卡训练还需要将相应`RANK_TABLE_FILE`文件的放置目录输入脚本（如`./rank_table_8pcs.json`），`RANK_TABLE_FILE`可按[此方法](#https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成。

#### 结果

训练时，当前训练轮次数，模型损失值，每轮次运行时间等信息会以如下形式显示，且运行日志将保存至`./logs/train.log`：

```text
=========================================================================
Epoch: 0494 train_loss= 0.6885321 time= 0.3938899040222168 auc= 0.8568888790661333 f1= 0.8595539481615432
Epoch: 0494 sample_time= 0.09816598892211914 train_time= 0.03619980812072754 test_time= 0.2595179080963135 save_time= 0.0002493858337402344
=========================================================================
Epoch: 0495 train_loss= 0.6892552 time= 0.3953282833099365 auc= 0.8591453682601632 f1= 0.8528564934080921
Epoch: 0495 sample_time= 0.1002199649810791 train_time= 0.03614640235900879 test_time= 0.2589559555053711 save_time= 0.00026345252990722656
=========================================================================
Epoch: 0496 train_loss= 0.6864389 time= 0.3973879814147949 auc= 0.8581971834403941 f1= 0.7663798808735937
Epoch: 0496 sample_time= 0.09870719909667969 train_time= 0.03621697425842285 test_time= 0.26245594024658203 save_time= 0.0003509521484375
=========================================================================
Epoch: 0497 train_loss= 0.68468577 time= 0.3998579978942871 auc= 0.8540750929442135 f1= 0.6958808063102541
Epoch: 0497 sample_time= 0.10423851013183594 train_time= 0.03621530532836914 test_time= 0.2593989372253418 save_time= 0.00024199485778808594
=========================================================================
Epoch: 0498 train_loss= 0.6862765 time= 0.3946268558502197 auc= 0.8611026245391791 f1= 0.8313908313908315
Epoch: 0498 sample_time= 0.10092616081237793 train_time= 0.03557133674621582 test_time= 0.25812458992004395 save_time= 0.00023102760314941406
=========================================================================
Epoch: 0499 train_loss= 0.6885195 time= 0.3965325355529785 auc= 0.8558373386341545 f1= 0.8473539308657082
Epoch: 0499 sample_time= 0.10099625587463379 train_time= 0.03545022010803223 test_time= 0.26008152961730957 save_time= 0.00026535987854003906
=========================================================================
Epoch: 0500 train_loss= 0.692978 time= 0.3948099613189697 auc= 0.8620984786140553 f1= 0.8376332457902056
Epoch: 0500 sample_time= 0.10086441040039062 train_time= 0.03542065620422363 test_time= 0.25852012634277344 save_time= 0.0002288818359375
=========================================================================
Training fished! The best AUC and F1-Score is: 0.8689866859770485 0.9425843754201964 Total time: 41.48991870880127
******************** finish training! ********************
```

### 评估过程

#### 运行

在完成训练流程的基础上，评估流程将自动从checkpoints目录加载对应任务的最优检查点用于模型评估。

```shell
# 评估
bash ./scripts/run_eval.sh sgcn_otc_auc.ckpt sgcn_otc_f1.ckpt ./input/bitcoin_otc.csv
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
bash ./scripts/run_export.sh 0 ./input/bitcoin_otc.csv
```

#### 结果

若模型导出成功，程序将显示如下，且运行日志将保存至`./logs/export.log`：

```text
==========================================
sgcn.mindir exported successfully!
==========================================
```

### Ascend310推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

#### 运行

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。注意，对不同数据集的精度指标进行推理时，需要修改`postprocess.py`文件中对应的`checkpoint`参数。

```shell
# Ascend310 推理
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` 表示数据集名称，取值范围： ['bitcoin-alpha', 'bitcoin-otc']。
- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围：'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

#### 结果

推理结果保存在脚本执行的当前路径，你可以在`acc.log`中看到以下精度计算结果，这里以bitcoin-otc数据集为例。

```text
==========================================
Test set results: auc= 0.87464 f1= 0.93635
==========================================
```

## 模型描述

### Ascend910性能

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

### Ascend310性能

| 参数          | SGCN                                                         |
| ------------- | ------------------------------------------------------------ |
| 资源          | Ascend 310服务器                                             |
| 上传日期      | 2021-11-01                                                   |
| MindSpore版本 | 1.3.0                                                        |
| 数据集        | Bitcoin-OTC / Bitcoin-Alpha                                  |
| 训练参数      | epoch=500, lr=0.01, weight_decay=1e-5                        |
| 优化器        | Adam                                                         |
| 损失函数      | Softmax交叉熵                                                |
| AUC           | 0.8746 / 0.8227                                              |
| F1-Score      | 0.9363 / 0.9543                                              |
| 脚本          | [SGCN](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |

## 随机情况说明

train.py和eval.py脚本中使用mindspore.set_seed()对全局随机种子进行了固定，可在对应的parser中进行修改即可。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
