# 目录

- [T-GCN概述](#T-GCN概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
        - [运行](#运行)
        - [结果](#结果)
    - [评估流程](#评估流程)
        - [运行](#运行)
        - [结果](#结果)
    - [MINDIR模型导出流程](#MINDIR模型导出流程)
        - [运行](#运行)
        - [结果](#结果)
    - [Ascend310推理流程](#Ascend310推理流程)
        - [运行](#运行)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
    - [评估性能](#评估性能)
    - [Ascend310推理性能](#Ascend310推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

# [T-GCN概述](#目录)

时间图卷积网络（Temporal Graph Convolutional Network，T-GCN）模型，简称T-GCN模型，是Zhao L等人提出的一种适用于城市道路交通预测的模型。所谓交通预测，即基于道路历史交通信息，对一定时期内的交通信息进行预测，包括但不限于交通速度、交通流量、交通密度等信息。

[论文](https://arxiv.org/pdf/1811.05320.pdf)：Zhao L, Song Y, Zhang C, et al. T-gcn: A temporal graph convolutional network for traffic prediction[J]. IEEE Transactions on Intelligent Transportation Systems, 2019, 21(9): 3848-3858.

# [模型架构](#目录)

T-GCN模型主要由两大模块构成，分别为图卷积网络（Graph Convolutional Network，GCN）与门控循环单元（Gated Recurrent Unit，GRU）。

模型整体处理流程如下：输入n组历史时间序列数据，利用图卷积网络捕获城市路网拓扑结构，以获取数据的空间特征。再将得到的具有空间特征的数据输入门控循环单元，利用单元间的信息传递捕获数据的动态变化，以获取数据的时间特征。最后，经过全连接层，输出最终预测结果。

其中，GCN模块通过在傅里叶域中构造一个作用于图数据的节点及其一阶邻域的滤波器来捕获节点间的空间特征，之后在其上叠加多个卷积层来实现。GCN模块可对城市中心道路与其周围道路间的拓扑结构及道路属性实现编码，捕获数据的空间相关性。而GRU模块则是作为一种经典的递归神经网络变体来捕获交通流量数据中的时间相关性。该模块使用门控机制来记忆尽可能多的长期信息，且结构相对简单，参数较少，训练速度较快，可以在捕获当前时刻交通信息的同时，仍然保持历史交通信息的变化趋势，具有捕获数据的时间相关性的能力。

# [数据集](#目录)

- 数据集：实验基于两大由现实采集的[SZ-taxi数据集](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data)与[Los-loop数据集](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data)。

（1）SZ-taxi数据集选取深圳市罗湖区的156条主要城市道路为研究区域，记录了2015年1月1日至1月31日的出租车运行轨迹。该数据集主要包含两个部分，一是记录了城市道路间拓扑关系的一个156*156大小的邻接矩阵，其中每行代表一条道路，矩阵中的值表示道路间的连接。二是记录了每一条道路上速度值随时间变化的特征矩阵，其中每行代表一条道路，每列为不同时间段道路上的交通速度，每15分钟记录一次。

（2）Los-loop数据集由洛杉矶高速公路上共计207个环形探测器于2012年3月1日至2012年3月7日实时采集得到，数据每5分钟记录一次。与SZ-taxi数据集相似，该数据集主要包含邻接矩阵与特征矩阵两个部分，邻接矩阵中的值由探测器之间的距离计算得到。由于该数据集中存在数据缺失，因此论文作者采用线性插值的方法进行了缺失值填充。

- 数据处理：输入数据被归一化至[0,1]区间，并划分其中的80%作训练集，20%作测试集，来分别预测未来15分钟、30分钟、45分钟、60分钟的交通速度。

# [环境要求](#目录)

- 硬件（Ascend / GPU）
    - 需要准备具有Ascend或GPU处理能力的硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速开始](#目录)

通过官方指南[安装MindSpore](https://www.mindspore.cn/install)后，下载[数据集](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data)，将下载好的数据集按如下目录结构进行组织，也可按此结构自行添加数据集：

```python
.
└─tgcn
  ├─data
    ├─SZ-taxi          # SZ-taxi数据集
        ├─adj.csv      # 邻接矩阵
        └─feature.csv  # 特征矩阵
    ├─Los-loop         # Los-loop数据集s
        ├─adj.csv      # 邻接矩阵
        └─feature.csv  # 特征矩阵
...
```

组织好数据集后，即可按顺序依次进行模型训练与评估/导出等操作：

- 训练：

```python
# 单卡训练
bash ./scripts/run_standalone_train.sh [DEVICE_ID]

# Ascend多卡训练
bash ./scripts/run_distributed_train_ascend.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATA_PATH]
```

示例：

  ```python
  # 单卡训练
  bash ./scripts/run_standalone_train.sh 0

  # Ascend多卡训练（8卡）
  bash ./scripts/run_distributed_train_ascend.sh ./rank_table_8pcs.json 8 0 ./data
  ```

- 评估：

```python
# 评估
bash ./scripts/run_eval.sh [DEVICE_ID]
```

示例：

  ```python
  # 评估
  bash ./scripts/run_eval.sh 0
  ```

- MINDIR模型导出：

```python
# MINDIR模型导出
bash ./scripts/run_export.sh [DEVICE_ID]
```

示例：

```python
# MINDIR模型导出
bash ./scripts/run_export.sh 0
```

- Ascend310推理：

```python
# Ascend310推理
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

示例：

```python
# Ascend310推理
bash ./scripts/run_infer_310.sh ./outputs/SZ-taxi_1.mindir ./data y Ascend 0
```

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```python
.
└─tgcn
  ├─README_CN.md                        # 中文指南
  ├─README.md                           # 英文指南
  ├─requirements.txt                    # pip依赖文件
  ├─scripts
    ├─run_distributed_train_ascend.sh   # Ascend多卡训练运行脚本
    ├─run_distributed_train_gpu.sh      # GPU多卡训练运行脚本
    ├─run_eval.sh                       # 评估运行脚本
    ├─run_export.sh                     # MINDIR模型导出运行脚本
    ├─run_infer_310.sh                  # 310推理运行脚本
    └─run_standalone_train.sh           # 单卡训练运行脚本
  ├─ascend310_infer                     # 实现310推理源代码
  ├─src
    ├─model
        ├─__init__.py
        ├─graph_conv.py                 # 图卷积计算
        ├─loss.py                       # 自定义损失函数
        └─tgcn.py                       # T-GCN模型架构
    ├─__init__.py
    ├─callback.py                       # 自定义回调函数
    ├─config.py                         # 模型参数设定
    ├─dataprocess.py                    # 数据处理模块
    ├─metrics.py                        # 模型评估指标
    └─task.py                           # 监督预测任务
  ├─eval.py                             # 评估
  ├─export.py                           # MINDIR模型导出
  ├─preprocess.py                       # 310推理预处理
  ├─postprocess.py                      # 310推理后处理
  └─train.py                            # 训练
```

## [脚本参数](#目录)

- 训练、评估、MINDIR模型导出、Ascend310推理等操作相关任务参数皆在`./src/config.py`脚本中设定：

```python
class ConfigTGCN:
    device = 'Ascend'
    seed = 1
    dataset = 'SZ-taxi'
    hidden_dim = 100
    seq_len = 4
    pre_len = 1
    train_split_rate = 0.8
    epochs = 3000
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1.5e-3
    data_sink = True
```

如需查阅相关参数信息说明，请参阅`./src/config.py`脚本内容，也可参阅论文原文。

## [训练流程](#目录)

### [运行](#目录)

开始训练前，请确认已在`./src/config.py`脚本中完成相关训练参数设定。在同一任务下，后续评估、MINDIR模型导出、Ascend310推理等流程请保持参数一致。

```python
# 单卡训练
# 用法：
bash ./scripts/run_standalone_train.sh [DEVICE_ID]
# 示例：
bash ./scripts/run_standalone_train.sh 0

# Ascend多卡训练
# 用法：
bash ./scripts/run_distributed_train_ascend.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATA_PATH]
# 示例（8卡）：
bash ./scripts/run_distributed_train_ascend.sh ./rank_table_8pcs.json 8 0 ./data
```

单卡训练中`[DEVICE_ID]`为训练所调用卡的卡号。

Ascend多卡训练中`[RANK_TABLE]`为相应RANK_TABLE_FILE文件路径（如8卡训练使用的`./rank_table_8pcs.json`），RANK_TABLE_FILE可按[此方法](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成。`[RANK_SIZE]`为训练所调用卡的数量，`[DEVICE_START]`为起始卡号，`[DATA_PATH]`为数据集存放根目录。

### [结果](#目录)

训练时，当前训练轮次数，模型损失值，每轮次运行时间等有关信息会以如下形式展示：

  ```python
  ==========Training Start==========
  epoch: 1 step: 37, loss is 47.07869
  epoch time: 20385.370 ms, per step time: 550.956 ms
  RMSE eval: 8.408103
  Best checkpoint saved!
  epoch: 2 step: 37, loss is 26.325077
  epoch time: 607.063 ms, per step time: 16.407 ms
  RMSE eval: 6.355909
  Best checkpoint saved!
  epoch: 3 step: 37, loss is 24.1607
  epoch time: 606.936 ms, per step time: 16.404 ms
  RMSE eval: 6.126811
  Best checkpoint saved!
  epoch: 4 step: 37, loss is 23.835127
  epoch time: 606.999 ms, per step time: 16.405 ms
  RMSE eval: 6.077283
  Best checkpoint saved!
  epoch: 5 step: 37, loss is 23.536343
  epoch time: 606.879 ms, per step time: 16.402 ms
  RMSE eval: 6.035936
  Best checkpoint saved!
  epoch: 6 step: 37, loss is 23.218105
  epoch time: 606.861 ms, per step time: 16.402 ms
  RMSE eval: 5.993234
  Best checkpoint saved!
  ...
  ```

单卡训练将会把上述信息以运行日志的形式保存至`./logs/train.log`，且模型会以覆盖的形式自动保存最优检查点（.ckpt 文件）于`./checkpoints`目录下，供后续评估与模型导出流程加载使用（如`./checkpoints/SZ-taxi_1.ckpt`）。

Ascend多卡训练与单卡训练所展示信息的形式基本一致，运行日志及最优检查点将保存在以对应卡号ID命名的`./device{ID}`目录下（如`./device0/logs/train.log`与`./device0/checkpoints/SZ-taxi_1.ckpt`）。

## [评估流程](#目录)

### [运行](#目录)

在完成训练流程的基础上，评估流程将基于`./src/config.py`脚本中的参数设定自动从`./checkpoints`目录加载对应任务的最优检查点（.ckpt 文件）用于模型评估。

```python
# 评估
# 用法：
bash ./scripts/run_eval.sh [DEVICE_ID]
# 示例：
bash ./scripts/run_eval.sh 0
```

### [结果](#目录)

训练后模型在验证集上的相关指标评估结果将以如下形式展示，且以运行日志的形式保存至`./logs/eval.log`：

  ```python
  =====Evaluation Results=====
  RMSE: 4.083120
  MAE: 2.730229
  Accuracy: 0.715577
  R2: 0.847140
  Var: 0.847583
  ============================
  ```

## [MINDIR模型导出流程](#目录)

### [运行](#目录)

在完成训练流程的基础上，MINDIR模型导出流程将基于`./src/config.py`脚本中的参数设定自动从`./checkpoints`目录加载对应任务的最优检查点（.ckpt 文件）用于对应MINDIR模型导出。

```python
# MINDIR模型导出
# 用法：
bash ./scripts/run_export.sh [DEVICE_ID]
# 示例：
bash ./scripts/run_export.sh 0
```

### [结果](#目录)

若模型导出成功，程序将以如下形式展示，且以运行日志的形式保存至`./logs/export.log`：

```python
==========================================
SZ-taxi_1.mindir exported successfully!
==========================================
```

同时MINDIR模型文件将导出至`./outputs`目录下（如`./outputs/SZ-taxi_1.mindir`），供后续Ascend310推理使用等。

## [Ascend310推理流程](#目录)

### [运行](#目录)

在完成MINDIR模型导出的基础上，基于`./src/config.py`脚本中的参数设定，Ascend310推理流程将加载对应任务导出的MINDIR模型（.mindir 文件）用于对应Ascend310推理任务。

- 请注意：目前仅支持batch_size为1的Ascend310推理功能，即MINDIR模型必须基于batch_size=1的训练过程导出得到。

```python
# Ascend310推理
# 用法：
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
# 示例：
bash ./scripts/run_infer_310.sh ./outputs/SZ-taxi_1.mindir ./data y Ascend 0
```

### [结果](#目录)

若Ascend310推理执行成功，可通过`cat ./acc.log`查看推理精度有关信息，如：

```python
RMSE 4.4604 | MAE 3.2427 | ACC 0.6882 | R_2 0.8137 | VAR 0.8166
```

同时也可通过`cat ./time_Result/test_perform_static.txt`查看推理性能有关信息，如：

```python
NN inference cost average time: 3.51738 ms of infer_count 591
```

# [模型说明](#目录)

## [训练性能](#目录)

- 下表中训练性能由T-GCN模型基于SZ-taxi数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取1、2、3、4）的交通速度得到，相关指标为4组训练任务平均值：

| 参数 | Ascend |
| ------------------- | -------------------|
| 模型名称 | T-GCN |
| 运行环境 | 操作系统 Euler 2.8；Ascend 910；处理器 2.60GHz，192核心；内存，755G |
| 上传日期 | 2021-09-30 |
| MindSpore版本 | 1.3.0 |
| 数据集 | SZ-taxi（hidden_dim=100；seq_len=4） |
| 训练参数 | seed=1；epoch=3000；batch_size = 64；lr=0.001；train_split_rate = 0.8；weight_decay = 1.5e-3 |
| 优化器 | Adam with Weight Decay |
| 损失函数 | 自定义损失函数 |
| 输出 | 交通速度预测值 |
| 平均检查点（.ckpt 文件）大小 | 839 KB |
| 平均性能 | 单卡：23毫秒/步，871毫秒/轮；8卡：25毫秒/步，101毫秒/轮 |
| 平均总耗时 | 单卡：49分19秒；8卡：11分35秒 |
| 脚本 | [训练脚本](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn/train.py) |

- 下表中训练性能由T-GCN模型基于Los-loop数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取3、6、9、12）的交通速度得到，相关指标为4组训练任务平均值：

| 参数 | Ascend |
| ------------------- | ------------------- |
| 模型名称 | T-GCN |
| 运行环境 | 操作系统 Euler 2.8；Ascend 910；处理器 2.60GHz，192核心；内存，755G |
| 上传日期 | 2021-09-30 |
| MindSpore版本 | 1.3.0 |
| 数据集 | Los-loop（hidden_dim=64；seq_len=12） |
| 训练参数 | seed=1；epoch=3000；batch_size = 64；lr=0.001；train_split_rate = 0.8；weight_decay = 1.5e-3 |
| 优化器 | Adam with Weight Decay |
| 损失函数 | 自定义损失函数 |
| 输出 | 交通速度预测值 |
| 平均检查点（.ckpt 文件）大小 | 993KB |
| 平均性能 | 单卡：44毫秒/步，1066毫秒/轮；8卡：46毫秒/步，139毫秒/轮 |
| 平均总耗时 | 单卡：1时00分40秒；8卡：15分05秒 |
| 脚本 | [训练脚本](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn/train.py) |

## [评估性能](#目录)

- 下表中评估性能由T-GCN模型基于SZ-taxi数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取1、2、3、4）的交通速度得到，相关指标为4组评估任务平均值：

| 参数 | Ascend |
| ------------------- | ------------------- |
| 模型名称 | T-GCN |
| 运行环境 | 操作系统 Euler 2.8；Ascend 910；处理器 2.60GHz，192核心；内存，755G |
| 上传日期 | 2021-09-30 |
| MindSpore版本 | 1.3.0 |
| 数据集 | SZ-taxi（hidden_dim=100；seq_len=4；batch_size = 64） |
| 输出 | 交通速度预测值 |
| 均方根误差（RMSE）平均值 | 4.1003 |
| 平均绝对误差（MAE）平均值 | 2.7498 |
| 预测准确率（Accuracy）平均值 | 0.7144 |
| R平方（$R^2$）平均值 | 0.8458 |
| 可释方差（Explained Variance）平均值 | 0.8461 |
| 脚本 | [评估脚本](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn/eval.py) |

- 下表中评估性能由T-GCN模型基于Los-loop数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取3、6、9、12）的交通速度得到，相关指标为4组评估任务平均值：

| 参数 | Ascend |
| ------------------- | ------------------- |
| 模型名称 | T-GCN |
| 运行环境 | 操作系统 Euler 2.8；Ascend 910；处理器 2.60GHz，192核心；内存，755G |
| 上传日期 | 2021-09-30 |
| MindSpore版本 | 1.3.0 |
| 数据集 | Los-loop（hidden_dim=64；seq_len=12；batch_size = 64） |
| 输出 | 交通速度预测值 |
| 均方根误差（RMSE）平均值 | 6.1869 |
| 平均绝对误差（MAE）平均值 | 3.8552 |
| 预测准确率（Accuracy）平均值 | 0.8946 |
| R平方（$R^2$）平均值 | 0.8000 |
| 可释方差（Explained Variance）平均值 | 0.8002 |
| 脚本 | [评估脚本](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn/eval.py) |

## [Ascend310推理性能](#目录)

- 下表中Ascend310推理性能由T-GCN模型基于SZ-taxi数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取1、2、3、4）的交通速度得到，相关指标为4组评估任务平均值：

| 参数 | Ascend |
| ------------------- | ------------------- |
| 模型名称 | T-GCN |
| 运行环境 | Ascend 310 |
| 上传日期 | 2021-11-03 |
| MindSpore版本 | 1.3.0 |
| 数据集 | SZ-taxi（hidden_dim=100；seq_len=4；batch_size = 1） |
| 输出 | 交通速度预测值 |
| 均方根误差（RMSE）平均值 | 4.2559 |
| 平均绝对误差（MAE）平均值 | 3.0077 |
| 预测准确率（Accuracy）平均值 | 0.7025 |
| R平方（$R^2$）平均值 | 0.8307 |
| 可释方差（Explained Variance）平均值 | 0.8334 |

- 下表中Ascend310推理性能由T-GCN模型基于Los-loop数据集分别预测未来15分钟、30分钟、45分钟、60分钟（即pre_len分别取3、6、9、12）的交通速度得到，相关指标为4组评估任务平均值：

| 参数 | Ascend |
| ------------------- | ------------------- |
| 模型名称 | T-GCN |
| 运行环境 | Ascend 310 |
| 上传日期 | 2021-11-03 |
| MindSpore版本 | 1.3.0 |
| 数据集 | Los-loop（hidden_dim=64；seq_len=12；batch_size = 1） |
| 输出 | 交通速度预测值 |
| 均方根误差（RMSE）平均值 | 6.5038 |
| 平均绝对误差（MAE）平均值 | 4.2996 |
| 预测准确率（Accuracy）平均值 | 0.8854 |
| R平方（$R^2$）平均值 | 0.6388 |
| 可释方差（Explained Variance）平均值 | 0.6517 |

# [随机情况说明](#目录)

`./train.py`脚本中使用`mindspore.set_seed()`对全局随机种子进行了固定（默认值为1），可在`./src/config.py`脚本中进行修改。

# [ModelZoo主页](#目录)

 [T-GCN](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn)