
# 目录

[View English](./README.md)

<!-- TOC -->

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
            - [用法](#用法)
            - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# SDNE概述

网络嵌入(Network Embedding)是学习网络顶点低维表示的一种重要方法，其目的是捕获和保存网络结构。与现有的网络嵌入不同，论文给出了一种新的深度学习网络架构SDNE，它可以有效的捕获高度非线性网络结构，同时也可以保留原先网络的全局以及局部结构。这项工作主要有三个贡献。（1）作者提出了结构化深度网络嵌入方法，可以将数据映射到高度非线性潜在空间中。（2）作者提出了一种新的半监督学习架构，可以同时学习到稀疏网络的全局和局部结构。（3）作者使用该方法在5个数据集上进行了评估，并将其应用到4个应用场景中，效果显著。

[论文](https://dl.acm.org/doi/10.1145/2939672.2939753) ： Wang D ,  Cui P ,  Zhu W. Structural Deep Network Embedding[C]// Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. August 2016.

# SDNE示例

# 数据集

使用的数据集：

- [WIKI](https://github.com/shenweichen/GraphEmbedding/tree/master/data/wiki/) 节点数: 2405个 边数: 17981条

- [CA-GRQC](https://github.com/suanrong/SDNE/tree/master/GraphData/) 节点数: 5242个 边数: 11496条

# 环境要求

- 硬件：Ascend
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Other
    - networkx
    - numpy
    - tqdm

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 单机训练运行示例
bash scripts/run_standalone_train.sh dataset_name /path/data /path/ckpt epoch_num 0

# GPU单机训练运行示例
bash scripts/run_standalone_train_gpu.sh [dataset_name] [data_path] [ckpt_path] [epoch_num] [device_id]

# 运行评估示例
bash scripts/run_eval.sh dataset_name /path/data /path/ckpt 0

# GPU运行评估示例
bash scripts/run_eval_gpu.sh  [dataset_name] [data_path] [ckpt_path] [device_id]
```

## 脚本说明

## 脚本和样例代码

```text
└── SDNE  
 ├── README_CN.md                    // SDNE相关描述
 ├── scripts
  ├── run_310_infer.sh               // 310推理脚本
  ├── run_standalone_train.sh        // 用于单机训练的shell脚本
  ├── run_eval.sh                    // 用于评估的shell脚本
  ├── run_standalone_train_gpu.sh    // 在GPU中用于单机训练的shell脚本
  └── run_eval_gpu.sh                // 在GPU中评估的shell脚本
 ├── ascend310_infer                 // 310推理相关代码
  ├── inc
   └── utils.h
  ├── src
   ├── main.cc
   └── utils.cc
  ├── build.sh
  ├── convert_data.py                // 转换数据脚本
  └── CMakeLists.txt
 ├── src
  ├── __init__.py
  ├── loss.py                        // 损失函数
  ├── config.py                      // 参数配置
  ├── dataset.py                     // 创建数据集
  ├── sdne.py                        // SDNE架构
  ├── initializer.py                 // 初始化脚本
  ├── optimizer.py                   // 优化器脚本
  └── utils.py                       // 功能性函数
 ├── export.py
 ├── eval.py                         // 测试脚本
 └── train.py                        // 训练脚本
```

## 脚本参数

```text
train.py中主要参数如下：

-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- device_target：['Ascend', 'GPU']
-- data_url：数据集路径。
-- ckpt_url：存放checkpoint的路径。
-- dataset：使用的数据集。['WIKI', 'GRQC']
-- epochs：迭代数。
-- pretrained：是否要使用预训练参数。
```

## 训练过程

### 训练

- 在Ascend环境训练WIKI数据集

```shell
bash scripts/run_standalone_train.sh WIKI /path/wiki /path/ckpt 40 0
```

- 在GPU环境训练

```shell
bash scripts/run_standalone_train_gpu.sh WIKI /path/wiki /path/ckpt 40 0
```

上述shell脚本将运行训练。可以通过`train.log`文件查看结果。
采用以下方式达到损失值：

```text
...
epoch: 36 step: 1, loss is 31.026050567626953
epoch time: 1121.593 ms, per step time: 1121.593 ms
epoch: 37 step: 1, loss is 29.539968490600586
epoch time: 1121.818 ms, per step time: 1121.818 ms
epoch: 38 step: 1, loss is 27.804513931274414
epoch time: 1120.751 ms, per step time: 1120.751 ms
epoch: 39 step: 1, loss is 26.283227920532227
epoch time: 1121.551 ms, per step time: 1121.551 ms
epoch: 40 step: 1, loss is 24.820133209228516
epoch time: 1123.054 ms, per step time: 1123.054 ms
```

- 在Ascend环境训练GRQC数据集

```shell
bash scripts/run_standalone_train.sh GRQC /path/grqc /path/ckpt 2 0
```

- 在GPU环境训练

```shell
bash scripts/run_standalone_train_gpu.sh GRQC /path/grqc /path/ckpt 2 0
```

上述shell脚本将运行训练。可以通过`train.log`文件查看结果。
采用以下方式达到损失值：

```text
...
epoch: 2 step: 157, loss is 607002.3125
epoch: 2 step: 158, loss is 638598.0625
epoch: 2 step: 159, loss is 485911.40625
epoch: 2 step: 160, loss is 774514.1875
epoch: 2 step: 161, loss is 733589.0625
epoch: 2 step: 162, loss is 504986.1875
epoch: 2 step: 163, loss is 416679.625
epoch: 2 step: 164, loss is 524830.75
epoch time: 14036.608 ms, per step time: 85.589 ms
```

## 评估过程

### 评估

- 在Ascend环境运行时评估WIKI数据集

```bash
bash scripts/run_eval.sh WIKI /path/wiki /path/ckpt 0
```

- 在GPU环境运行评估

```bash
bash scripts/run_eval_gpu.sh WIKI /path/wiki /path/ckpt 0
```

上述命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```text
Reconstruction Precision K  [1, 10, 20, 100, 200, 1000, 2000, 6000, 8000, 10000]
Precision@K(1)= 1.0
Precision@K(10)=        1.0
Precision@K(20)=        1.0
Precision@K(100)=       1.0
Precision@K(200)=       1.0
Precision@K(1000)=      1.0
Precision@K(2000)=      1.0
Precision@K(6000)=      0.9986666666666667
Precision@K(8000)=      0.991375
Precision@K(10000)=     0.966
MAP :  0.6673926856547066
```

- 在Ascend环境运行时评估GRQC数据集

```bash
bash scripts/run_eval.sh GRQC /path/grqc /path/ckpt 0
```

- 在GPU环境运行评估

```bash
bash scripts/run_eval_gpu.sh GRQC /path/grqc /path/ckpt 0
```

上述命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```text
Reconstruction Precision K  [10, 100]
getting similarity...
Precision@K(10)=        1.0
Precision@K(100)=       1.0
```

## 导出mindir模型

```bash
python export.py --dataset [NAME] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`dataset`是数据集名称，例如`GRQC`，FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

### 用法

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理之前，需要通过`export.py`导出`mindir`文件。

```bash
# Ascend310 推理
bash run_310_infer.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
```

`MINDIR_PATH`为`mindir`文件路径，`DATASET_NAME`为数据集名称，`DATASET_PATH`表示数据集文件的路径（例如`/datapath/sdne_wiki_dataset/WIKI/Wiki_edgelist.txt`）。

### 结果

推理结果保存在当前路径，可在`acc.log`中看到最终精度结果。

# 模型描述

## 性能

### 训练性能

| 参数          | SDNE                                         | GPU     |
| ------------- | ----------------------------------------------- | ------- |
| 模型版本      | SDNE                                  | SDNE     |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| 上传日期      | 2021-12-31                                     | 2022-02-18        |
| MindSpore版本 | 1.5.0                          | 1.5.0 |
| 数据集        | WIKI                               | WIKI                               |
| 训练参数      | lr=0.002                     | lr=0.002, epoch=40 |
| 优化器        | Adam                                             | Adam        |
| 损失函数      | SDNE Loss Function                       | SDNE Loss Function          |
| 输出          | 概率                                            | 概率                       |
| 损失          | 24.8                                          | 24.87                      |
| 速度 | 1卡：1105毫秒/步 | 1卡：15毫秒/步 |
| 总时间 | 1卡：44秒 | 1卡：44秒 |
| 参数(M) | 1.30 | 1.30 |
| 微调检查点 | 15M （.ckpt file） | 15M （.ckpt file） |
| 脚本 | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |

| 参数          | SDNE                                         |
| ------------- | ----------------------------------------------- |
| 模型版本      | SDNE                                  |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期      | 2022-4-7                                     |
| MindSpore版本 | 1.5.0                          |
| 数据集        | CA-GRQC                                       |
| 训练参数      | lr=0.01                     |
| 优化器        | RMSProp                                             |
| 损失函数      | SDNE Loss Function                       |
| 输出          | 概率                                            |
| 损失          | 736119.18                                            |
| 速度 | 1卡：86毫秒/步 |
| 总时间 | 1卡：28秒 |
| 参数(M) | 1.05 |
| 微调检查点 | 13M （.ckpt file） |
| 脚本 | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |

### 评估性能

| 参数          | SDNE                | GPU            |
| ------------- | ------------------ | ------------------ |
| 模型版本      | SDNE                | SDNE     |
| 资源          | Ascend 910         | Ubuntu 18.04.6, GF RTX3090, CPU 2.90GHz, 64cores, RAM 252GB |
| 上传日期      | 2021/12/31        | 2022/02/18       |
| MindSpore版本 | MindSpore-1.3.0-c78      | 1.5.0      |
| 数据集        | WIKI          | WIKI          |
| 输出          | 概率               | 概率          |
| 准确性        | 1卡：66.74% | 1卡：66.73% |

| 参数          | SDNE            |
| ------------- | ------------------ |
| 模型版本      | SDNE     |
| 资源          | Ascend 910         |
| 上传日期      | 2022/4/7        |
| MindSpore版本 | MindSpore-1.3.0-c78      |
| 数据集        | CA-GRQC          |
| 输出          | 概率               |
| 准确性        | 1卡：1 |

# 随机情况说明

在train.py中，我们固定了python、numpy和mindspore的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。

