# 目录

- [目录](#目录)
- [DialogXL描述](#dialogxl描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [下载预训练参数](#下载预训练参数)
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

# DialogXL描述

DialogXL模型用于解决对话语句分类任务，其通过增强记忆来存储更长的历史上下文，并通过对话感知的自注意力机制来解决对话语料中的层次结构。

# 模型架构

DialogXL以预训练语言模型XLNet作为模型的基本架构，XLNet为多层transformer堆叠结构，使用了片段循环机制，对于一段对话，每次循环输入一个对话语句Utterance，得到该语句的编码隐向量后，将隐向量保存在memory中，接着输入下一个对话语句。每个对话语句在transformer计算隐向量的过程中，memory都会参与到当前对话语句的key和value计算，这使得当前对话语句能够捕捉到历史对话语句的信息。同时DialogXL引入了对话感知的注意力机制，调整了transofmer计算注意力时的注意力掩码，使得不同的注意力头能过关注到对话上下文中的不同内容，增加对话编码的多样性。

# 数据集

DialogXL使用了MELD数据集作为网络模型验证的数据集，该数据集为一个英文对话语句情感判断任务，任务要求分类对话中每个对话语句的情感极性。MELD训练集有1038个对话，共87170个对话语句，验证集1000个对话，8069个对话语句，测试集1000个对话，7740个对话语句。

从[这里](https://pan.baidu.com/s/1NX8X-JTQ0i03AGSGr18e1g?pwd=t1un)下载数据集后，将下载好的数据集按如下目录结构进行组织:

```shell
.
└─dialogXL
  └─data
    ├─MELD_trainsets.pkl
    ├─MELD_valsets.pkl
    └─MELD_testsets.pkl
```

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore1.8.1](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```bash
.
└─dialogxl
  |
  ├─scripts
  | ├─run_eval_gpu.sh                 # GPU启动评估
  | └─run_standalone_train_gpu.sh     # GPU启动单卡训练
  |
  ├─src
  | ├─config.py                       # 模型参数
  | ├─data.py                         # 数据处理
  | └─dialogXL.py                     # 模型架构
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
--lr                    # 学习率
--output_dropout        # 输出层dropout
--epochs                # 训练轮数
```

## 训练过程

### 下载预训练参数

DialogXL模型在训练前需要先加载预训练好的XLNet模型参数，[这里](https://pan.baidu.com/s/1eD8TJQ7vsEoLIu6RVsHTXQ?pwd=fgku)提供了适配的MindSpore版本的XLNet预训练参数，将其下载后按如下目录结构组织。

```shell
.
└─dialogXL
  └─pretrained
    └─xlnet.ckpt
```

### 运行

```bash
# GPU单卡训练：bash run_standalone_train_gpu.sh [EPOCHS] [DEVICE_ID]
bash ./scripts/run_standalone_train_gpu.sh 5 0
```

### 结果

```text
********************** epoch 1 **********************
train_loss: 1.4824, train_acc: 50.09, train_fscore: 41.78.
Train epoch time: 5339611.185 ms, per step time: 2121.419 ms.
eval_acc: 53.74, eval_fscore: 48.86.
Eval epoch time: 96663.569 ms, per step time: 322.212 ms.
Best model saved!
********************** epoch 2 **********************
train_loss: 1.1707, train_acc: 61.65, train_fscore: 57.6.
Train epoch time: 5198738.013 ms, per step time: 2065.45 ms.
eval_acc: 57.89, eval_fscore: 54.3.
Eval epoch time: 94772.179 ms, per step time: 315.907 ms.
Best model saved!
********************** epoch 3 **********************
train_loss: 1.0268, train_acc: 66.31, train_fscore: 63.3.
Train epoch time: 5265046.015 ms, per step time: 2091.794 ms.
eval_acc: 58.52, eval_fscore: 55.21.
Eval epoch time: 95593.346 ms, per step time: 318.644 ms.
Best model saved!
********************** epoch 4 **********************
train_loss: 0.8983, train_acc: 71.5, train_fscore: 69.45.
Train epoch time: 5308438.759 ms, per step time: 2109.034 ms.
eval_acc: 58.34, eval_fscore: 55.48.
Eval epoch time: 96419.316 ms, per step time: 321.398 ms.
Best model saved!
********************** epoch 5 **********************
train_loss: 0.7718, train_acc: 76.34, train_fscore: 74.94.
Train epoch time: 5277837.927 ms, per step time: 2096.876 ms.
eval_acc: 58.16, eval_fscore: 55.32.
Eval epoch time: 96306.122 ms, per step time: 321.02 ms.
********************** End Training **********************
```

## 评估过程

### 运行

```bash
# GPU评估：bash run_eval_gpu.sh [DEVICE_ID]
bash ./scripts/run_eval_gpu.sh 0
```

### 结果

```text
********************** Evaluating **********************
test_acc: 61.76, test_fscore: 59.28
********************** End Evaluation **********************
```

## MINDIR模型导出过程

### 运行

```bash
python export.py
```

### 结果

```text
========================================
dialogXL.mindir exported successfully!
========================================
```

# 模型描述

## 性能

| 参数           | DialogXL                       |
| -------------  | -------------------------------|
| 资源           | GPU                            |
| 上传日期       | 2022-10-26                     |
| MindSpore版本  | 1.8.1                          |
| 数据集         | MELD                           |
| 训练参数       | lr=1e-6, epochs=5, output_dropout=0.0 |
| 优化器         | AdamW                          |
| 损失函数       | CrossEntropyLoss               |
| 精度           | 59.28                          |
| 速度 (per step)| 2109.034 ms                    |
| 脚本 | [DialogXL脚本](https://gitee.com/mindspore/models/tree/master/research/nlp/dialogxl) |

# 随机情况说明

在该项目中没有设置随机种子。

# modelzoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
