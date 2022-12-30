# 目录

- [目录](#目录)
- [NRI-MPM描述](#nri-mpm描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [运行](#运行-1)
        - [结果](#结果-1)
    - [MINDIR模型导出过程](#mindir模型导出过程)
        - [运行](#运行-2)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
- [随机情况说明](#随机情况说明)
- [modelzoo主页](#modelzoo主页)

# NRI-MPM描述

NRI-MPM是一种具有高效消息传递机制的神经关系推断模型，用于推断群体动力系统中个体间的交互关系，并预测个体的未来状态。

NRI-MPM模型及相应的数据集均来自AAAI 2021的论文，"[Neural Relational Inference with Efficient Message Passing Mechanism](https://www.aaai.org/AAAI21Papers/AAAI-5078.ChenS.pdf)"。

# 模型架构

NRI-MPM模型是一种变分自编码器，编码器根据个体的历史状态推断其交互关系，解码器根据历史状态和推断出的交互关系预测个体的未来状态。详情如下：

（1）编码器：首先使用注意力机制和一维卷积神经网络把历史状态序列映射为节点表示，然后使用图神经网络根据节点表示推断出节点间不同类型连边的概率，根据概率采样得到连边；

（2）解码器：对于节点及其连边构成的交互图，使用图神经网络聚合节点表示，使用循环神经网络进行自回归的序列预测。

# 数据集

NRI-MPM模型在3个物理仿真数据集上验证，分别是弹簧系统Springs、带电粒子系统Charged和耦合振子系统Kuramoto。

每个样本包含固定数目节点的状态序列及反映交互关系的邻接矩阵，详情如下：

|  数据集   | 节点数目 | 输入维度 | 边类别数 | 观测步数 | 预测步数 |
| :------: | :---: | :----:   | :----: | :-----: | :-----:|
| Springs  |   5   |   4      | 1      |  49     |   50   |
| Charged  |   5   |   4      | 2      |  49     |   50   |
| Kuramoto |   5   |   3      | 1      |  49     |   50   |

每个数据集的训练、验证、测试样本数均为50k、10k、10k。在输入模型之前，节点的原始特征均会做min-max归一化。所有数据集测试的指标均为节点。

从[这里](https://pan.baidu.com/s/1tWug_iEsRtDfLL_h27akIg?pwd=pe8v)下载数据集后，将下载好的数据集按如下目录结构进行组织:

```shell
.
└─nri-mpm
  └─data
    ├─charged.pkl
    ├─kuramoto.pkl
    └─spring.pkl
```

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore1.5.0](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```bash
.
└─nri-mpm
  |
  ├─scripts
  | ├─run_eval_gpu.sh      # GPU启动评估
  | ├─run_export_gpu.sh    # GPU导出模型
  | ├─run_train_8p_gpu.sh  # GPU启动多卡训练
  | └─run_train_gpu.sh     # GPU启动训练
  |
  ├─models
  | ├─base.py              # 基础模块
  | ├─nri.py               # 主干网络模块
  | └─train_and_eval.py    # 训练和评估模块
  |
  ├─utils
  | ├─helper.py            # 工具函数和类
  | ├─logger.py            # 日志输出
  | └─metrics.py           # 网络评估函数
  |
  ├─README_CN.md           # 中文指南
  ├─requirements.txt       # 依赖包
  ├─export.py              # 导出模型
  ├─eval.py                # 评估
  └─train.py               # 训练
```

## 脚本参数

```python
train.py中主要参数如下
--dataset           # 数据集
--dim               # 输入维度
--epochs            # 训练轮数
--skip              # 是否在训练是采用跳过机制
--parallel          # 是否分布式训练
```

## 训练过程

### 运行

```bash
# GPU单卡训练：bash run_train_gpu.sh [DATASET] [EPOCHS] [DEVICE_ID]
# 以在spring数据集训练为例：
bash ./scripts/run_train_gpu.sh spring 500 0

# GPU多卡训练：bash run_train_8p_gpu.sh [DATASET] [EPOCHS] [DEVICE_NUM]
# 以在spring数据集训练为例：
bash ./scripts/run_train_8p_gpu.sh spring 500 8
```

### 结果

```text
...
2022-08-08 13:51:18,831 - INFO: ******************** spring epoch 495 ********************
2022-08-08 13:51:18,832 - INFO: train_loss: 0.000049
2022-08-08 13:51:18,832 - INFO: eval_step: 10, mse: 2.167e-06, acc: 0.9970, eval_loss: 0.088405
2022-08-08 13:51:35,999 - INFO: ******************** spring epoch 496 ********************
2022-08-08 13:51:36,000 - INFO: train_loss: 0.000084
2022-08-08 13:51:36,000 - INFO: eval_step: 10, mse: 2.196e-06, acc: 0.9966, eval_loss: 0.097650
2022-08-08 13:51:53,468 - INFO: ******************** spring epoch 497 ********************
2022-08-08 13:51:53,469 - INFO: train_loss: 0.000004
2022-08-08 13:51:53,469 - INFO: eval_step: 10, mse: 2.343e-06, acc: 0.9965, eval_loss: 0.105367
2022-08-08 13:52:10,531 - INFO: ******************** spring epoch 498 ********************
2022-08-08 13:52:10,532 - INFO: train_loss: 0.000004
2022-08-08 13:52:10,532 - INFO: eval_step: 10, mse: 2.114e-06, acc: 0.9965, eval_loss: 0.104949
2022-08-08 13:52:28,355 - INFO: ******************** spring epoch 499 ********************
2022-08-08 13:52:28,355 - INFO: train_loss: 0.000036
2022-08-08 13:52:28,356 - INFO: eval_step: 10, mse: 2.682e-06, acc: 0.9954, eval_loss: 0.108074
2022-08-08 13:52:45,330 - INFO: ******************** spring epoch 500 ********************
2022-08-08 13:52:45,331 - INFO: train_loss: 0.000019
2022-08-08 13:52:45,331 - INFO: eval_step: 10, mse: 2.881e-06, acc: 0.9958, eval_loss: 0.100078
2022-08-08 13:52:46,067 - INFO: ********************** Time Statistics **********************
2022-08-08 13:52:46,067 - INFO: The training process took 8745055.961 ms.
2022-08-08 13:52:46,067 - INFO: Each step took an average of 364.377 ms
2022-08-08 13:52:46,067 - INFO: ************************** End Training **************************
```

## 评估过程

### 运行

```bash
# GPU评估：bash run_eval_gpu.sh [DATASET] [DEVICE_ID]
# 以在spring数据集评估为例：
bash ./scripts/run_eval_gpu.sh spring 0
```

### 结果

```text
2022-08-08 14:04:17,301 - INFO: ************************** Evaluating **************************
2022-08-08 14:04:47,715 - INFO: test_step: 20, mse: 1.576e-04, acc: 0.9963
2022-08-08 14:04:47,716 - INFO: multi_mse: 7.336e-08, 2.077e-07, 4.608e-07, 8.180e-07, 1.278e-06, 1.839e-06, 2.502e-06, 3.265e-06, 4.129e-06, 5.093e-06, 6.155e-06, 7.317e-06, 8.580e-06, 9.943e-06, 1.141e-05, 1.299e-05, 1.467e-05, 1.647e-05, 1.839e-05, 2.044e-05
2022-08-08 14:04:47,716 - INFO: ************************** End Evaluation **************************
```

## MINDIR模型导出过程

### 运行

```bash
# GPU模型导出：bash run_export_gpu.sh [DATASET] [CKPT_FILE]
# 以导出在spring数据集上训练的模型为例：
bash scripts/run_export_gpu.sh spring checkpoints/spring.ckpt
```

### 结果

```text
=========================================
mindir/spring.mindir exported successfully!
=========================================
```

# 模型描述

## 性能

| 参数           | NRI-MPM                        |
| -------------  | -------------------------------|
| 资源           | GPU                            |
| 上传日期       | 2022-8-21                      |
| MindSpore版本  | 1.5.0                          |
| 数据集         | spring / charged / kuramoto     |
| 训练参数       | lr=0.001 / 0.002 / 0.001, epochs=500, batch_size=128            |
| 优化器         | Adam                                            |
| 损失函数       | NRI-MPM Loss Function                           |
| 准确率         | 99.63 / 92.91 / 96.65                           |
| 速度 (per step)| 364.377ms / 364.479ms / 316.452ms               |
| 总时间        | 8745055.961ms / 8747495.238ms / 7594855.069ms    |
| 脚本 | [NRI-MPM脚本](https://gitee.com/mindspore/models/tree/master/research/gnn/nri-mpm) |

# 随机情况说明

在该项目中没有设置随机种子。

# modelzoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
