# 目录

- [目录](#目录)
- [SNUH描述](#snuh描述)
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

# SNUH描述

SNUH是一种进行文档二进制表征学习的方法。该模型发表在自然语言处理顶会ACL 2021中。该模型首次将文本的语义信息和邻居信息统一在了同一个生成模型框架中。具体地，通过使用基于图诱导的高斯先验分布，该模型有效地将文档之间的关联关系建模到生成模型中。为了加速训练过程，该模型进一步提出了使用生成树的近似方法，从而使得模型可以在大规模数据场景下进行高效地训练。此方法在三个公开的数据集中取得了sota的效果。

# 模型架构

SNUH的模型结构包括3个部分：语义编码器，关系编码器和解码器。语义编码器将单个文本作为输入，输出文档的语义表征，关系编码器将一对文档对作为输入，输出该文档对的关系表征。最后解码器将这两个表征作为输入，输出重构的原始文档向量。

# 数据集

[Reuters](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset)

- 数据集大小：6.8M，训练集，验证集和测试集文档个数划分为：7752, 967, 964.

[TMC](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset)

- 数据集大小：28.3M，训练集，验证集和测试集文档个数划分为：21286, 3498, 3498.

[20Newsgroups](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset)

- 数据集大小：18.7M，训练集，验证集和测试集文档个数划分为： 11016, 3667, 3668.为节点。

从[这里](https://pan.baidu.com/s/1ltZQ-EpUOu2U3-gfFCd9BA?pwd=79xf)下载数据集后，将下载好的数据集按如下目录结构进行组织:

```shell
.
└─snuh
  └─data
    ├─reuters.tfidf.mat
    ├─tmc.tfidf.mat
    └─ng20.tfidf.mat
```

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore1.8.0](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```bash
.
└─snuh
  |
  ├─scripts
  | ├─run_export_gpu.sh               # GPU导出模型
  | ├─run_standalone_eval_gpu.sh      # GPU启动单卡评估
  | └─run_standalone_train_gpu.sh     # GPU启动单卡训练
  |
  ├─models
  | ├─snuh.py                         # 主干网络模块
  | └─train_and_eval.py               # 训练和评估模块
  |
  ├─utils
  | ├─args.py                         # 获取参数
  | ├─data.py                         # 数据处理
  | ├─logger.py                       # 日志输出
  | └─evaluation.py                   # 网络评估函数
  |
  ├─README_CN.md                      # 中文指南
  ├─requirements.txt                  # 依赖包
  ├─export.py                         # 导出模型
  ├─eval.py                           # 评估
  └─train.py                          # 训练
```

## 脚本参数

```python
train.py中主要参数如下
model_path              # 数据集名称
data_path               # 数据集路径
--num_features          # 哈希码位数
--num_neighbors         # 邻居个数
--batch_size            # batch大小
--lr                    # 学习率
--epochs                # 训练轮数
--num_trees             # 生成树数目
--temperature           # 构造图所使用的温度系数
--alpha                 # 哈希码二值化的温度系数
--beta                  # 损失函数中KL项的权重
```

## 训练过程

### 运行

```bash
# GPU单卡训练：bash run_standalone_train_gpu.sh [DATASET] [DEVICE_ID]
# 以在Reuters数据上进行哈希码位数为64的训练为例：
bash ./scripts/run_standalone_train_gpu.sh reuters64 0
```

### 结果

```text
...
************************************************************
End of epoch 20, val perf: 85.27
Best model so far, save model!
Train epoch time: 4735.294 ms, per step time: 19.567 ms
************************************************************
End of epoch 21, val perf: 85.26
Bad epoch 1.
Train epoch time: 4665.378 ms, per step time: 19.278 ms
************************************************************
End of epoch 22, val perf: 85.16
Bad epoch 2.
Train epoch time: 4511.529 ms, per step time: 18.643 ms
************************************************************
End of epoch 23, val perf: 85.08
Bad epoch 3.
Train epoch time: 4748.075 ms, per step time: 19.620 ms
************************************************************
End of epoch 24, val perf: 85.19
Bad epoch 4.
Train epoch time: 4618.619 ms, per step time: 19.085 ms
************************************************************
End of epoch 25, val perf: 84.97
Bad epoch 5.
Train epoch time: 4691.999 ms, per step time: 19.388 ms
************************************************************
End of epoch 26, val perf: 85.14
Bad epoch 6.
Train epoch time: 4636.309 ms, per step time: 19.158 ms
************************************************************
End of epoch 27, val perf: 85.18
Bad epoch 7.
Train epoch time: 4638.733 ms, per step time: 19.168 ms
********************** Time Statistics **********************
The training process took 134252.348 ms.
Each step took an average of 20.547 ms
************************** End Training **************************
```

## 评估过程

### 运行

```bash
# GPU评估：bash run_standalone_eval_gpu.sh [DATASET] [DEVICE_ID]
# 以评估Reuters数据集上64位哈希码的精度为例：
bash ./scripts/run_standalone_eval_gpu.sh reuters64 0
```

### 结果

```text
...
************************** Evaluating **************************
val precision: 85.27
test precision: 85.51
************************** End Evaluation **************************
```

## MINDIR模型导出过程

### 运行

```bash
# GPU模型导出：bash run_export_gpu.sh [DATASET]
# 以导出在Reuters数据上进行64位哈希码训练的模型为例：
bash scripts/run_export_gpu.sh reuters64
```

### 结果

```text
...
=========================================
mindir/reuters64.mindir exported successfully!
=========================================
```

# 模型描述

## 性能

| 参数           | SNUH                        |
| -------------  | -------------------------------|
| 资源           | GPU                            |
| 上传日期       | 2022-9-21                      |
| MindSpore版本  | 1.8.0                          |
| 数据集         | Reuters 128bits / TMC 128bits / 20Newsgroups 128bits|
| 训练参数       | 具体参数详见scripts/run_standalone_train_gpu.sh      |
| 优化器         | Adam                                                |
| 损失函数       | SNUH Loss Function                                  |
| 准确率         | 85.44 / 77.10 / 67.51                               |
| 速度 (per step)| 35.442ms / 135.676ms / 105.942ms                    |
| 总时间         | 145810.402ms / 1036025.649ms / 391772.699ms         |
| 脚本 | [SNUH脚本](https://gitee.com/mindspore/models/tree/master/research/nlp/snuh) |

在所有实验中的模型精度如下：

|         | 16bits | 32bits | 64bits | 128bits |
| :-----: | :----: | :----: | :----: | :-----: |
| Reuters | 80.69  | 83.47  | 85.51  |  85.44  |
|   TMC   | 71.51  | 74.63  | 76.66  |  77.10  |
|  NG20   | 54.59  | 62.49  | 65.13  |  67.51  |

# 随机情况说明

在该项目中没有设置随机种子。

# modelzoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
