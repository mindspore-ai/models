# 目录

- [目录](#目录)
- [SDNE概述](#IBN-Net概述)
- [SDNE示例](#IBN-Net示例)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [训练](#训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
        - [导出mindir模型](#导出mindir模型)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# SDNE概述

网络嵌入(Network Embedding)是学习网络顶点低维表示的一种重要方法，其目的是捕获和保存网络结构。与现有的网络嵌入不同，论文给出了一种新的深度学习网络架构SDNE，它可以有效的捕获高度非线性网络结构，同时也可以保留原先网络的全局以及局部结构。这项工作主要有三个贡献。（1）作者提出了结构化深度网络嵌入方法，可以将数据映射到高度非线性潜在空间中。（2）作者提出了一种新的半监督学习架构，可以同时学习到稀疏网络的全局和局部结构。（3）作者使用该方法在5个数据集上进行了评估，并将其应用到4个应用场景中，效果显著。

[论文](https://dl.acm.org/doi/10.1145/2939672.2939753)： Wang D ,  Cui P ,  Zhu W. Structural Deep Network Embedding[C]// Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. August 2016.

# SDNE示例

# 数据集

使用的数据集：[Wiki](https://github.com/shenweichen/GraphEmbedding/tree/master/data/wiki)
节点数: 2405个
边数: 17981条

# 环境要求

- 硬件：Ascend
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 单机训练运行示例
bash scripts/run_standalone_train.sh /path/Wiki_edgelist.txt /path/ckpt

# 运行评估示例
bash scripts/run_eval.sh  /path/Wiki_edgelist.txt /path/checkpoint
```

## 脚本说明

## 脚本和样例代码

```path
└── SDNE  
 ├── README_CN.md                    // SDNE相关描述
 ├── scripts
  ├── run_standalone_train.sh        // 用于单机训练的shell脚本
  └── run_eval.sh                    // 用于评估的shell脚本
 ├── src
  ├── __init__.py
  ├── loss.py                        // 损失函数
  ├── config.yaml                    // 参数配置
  ├── dataset.py                     // 创建数据集
  ├── sdne.py                        // SDNE架构
  └── utils.py                       // 功能性函数
 ├── classify.py                     // 分类任务脚本
 ├── export.py
 ├── eval.py                         // 测试脚本
 └── train.py                        // 训练脚本

```

## 脚本参数

```python
train.py中主要参数如下：

-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- data_url：数据集路径。
-- ckpt_url：存放checkpoint的路径。
-- dataset：使用的数据集。
-- epochs：迭代数。
-- pretrained：是否要使用预训练参数。

```

## 训练过程

### 训练

- 在Ascend环境训练

```shell
bash scripts/run_standalone_train.sh /path/Wiki_edgelist.txt /path/ckpt
```

上述shell脚本将运行训练。可以通过`train.log`文件查看结果。
采用以下方式达到损失值：

```log
...
epoch: 38 step: 1, loss is 27.804518
epoch time: 1104.996 ms, per step time: 1104.996 ms
epoch: 39 step: 1, loss is 26.283232
epoch time: 1105.147 ms, per step time: 1105.147 ms
epoch: 40 step: 1, loss is 24.820133
epoch time: 1105.217 ms, per step time: 1105.217 ms

```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

```bash
bash scripts/run_eval.sh  /path/Wiki_edgelist.txt /path/checkpoint
```

上述命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```bash
Reconstruction Precision K  [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000]
Precision@K(1)=  1.0
Precision@K(10)= 1.0
Precision@K(20)= 1.0
Precision@K(100)=        1.0
Precision@K(200)=        1.0
Precision@K(1000)=       1.0
Precision@K(2000)=       1.0
Precision@K(6000)=       0.9986666666666667
Precision@K(8000)=       0.991375
Precision@K(10000)=      0.966
MAP :  0.6673926527718224
```

## 导出mindir模型

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

# 模型描述

## 性能

### 训练性能

| 参数          | SDNE                                         |
| ------------- | ----------------------------------------------- |
| 模型版本      | SDNE                                  |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期      | 2021-12-31                                     |
| MindSpore版本 | 1.5.0                          |
| 数据集        | wiki                                       |
| 训练参数      | lr=0.002                     |
| 优化器        | Adam                                             |
| 损失函数      | SDNE Loss Function                       |
| 输出          | 概率                                            |
| 损失          | 24.8                                            |
| 速度 | 1卡：1105毫秒/步 |
| 总时间 | 1卡：44秒 |
| 参数(M) | 1.30 |
| 微调检查点 | 15M （.ckpt file） |
| 脚本 | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |

### 评估性能

| 参数          | SDNE            |
| ------------- | ------------------ |
| 模型版本      | SDNE     |
| 资源          | Ascend 910         |
| 上传日期      | 2021/12/31        |
| MindSpore版本 | MindSpore-1.3.0-c78      |
| 数据集        | wiki          |
| 输出          | 概率               |
| 准确性        | 1卡：66.74% |

# 随机情况说明

在train.py中，我们固定了python、numpy和mindspore的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。

