
# 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
    - [论文](#论文)
    - [简要介绍](#简要介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [数据集描述](#数据集描述)
    - [下载链接](#下载链接)
    - [数据集目录](#数据集目录)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本结构与说明](#脚本结构与说明)
- [脚本参数](#脚本参数)
    - [网络训练参数](#网络训练参数)
    - [知识蒸馏参数](#知识蒸馏参数)
- [网络训练过程](#网络训练过程)
    - [用法](#用法)
    - [结果](#结果)
- [知识蒸馏训练过程](#知识蒸馏训练过程)
    - [用法](#用法)
    - [结果](#结果)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 概述

## 论文

NeurIPS 2022 Paper: Asymmetric Temperature Scaling Makes Larger Networks Teach Well Again
[论文链接](https://openreview.net/forum?id=K3efgD7QzVp)

## 简要介绍

知识蒸馏技术可以将性能较好的神经网络（教师网络）的能力传授给性能较差的神经网络（学生网络），从而提升学生网络的性能。然而，并不是教师网络越大/越复杂/性能越好教地越好。相反地，一个非常好的教师不一定能够教出较好的学生。实验分析是因为大神经网络容易置信度过高，导致经过传统单一温度下的softmax之后，错误类别概率之间的差异较小，不能提供足够有价值的指导信息。为了增大错误类别概率的差异，提出了非对称温度缩放技术ATS。

# 模型架构

- 该示例代码实现了教师网络为ResNet14/ResNet110，学生网络为VGG8，在CIFAR100上的ATS算法过程。
- 使用传统的Temperature Scaling，即softmax和单一温度系数tp_tau=t_tau，使用ResNet110辅助下的VGG8的性能差于ResNet14辅助下的VGG8性能。
- 使用Asymmetric Temperature Scaling，对正确类别使用tp_tau，对错误类别使用t_tau，tp_tau>t_tau，即可提升ResNet110辅助下的VGG8的性能。

# 数据集

## 数据集描述

CIFAR100是图像分类任务经典数据集，包括5w训练图像，1w测试图像，每张图像大小为三通道32x32。

## 下载链接

[https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

## 数据目录

- --cifar-100-python
- |--train
- |--test
- |--meta
- |--file.txt~

# 环境要求

- 操作系统：Windows 10
- 硬件平台：CPU
- python版本：3.7.3
- anaconda版本：Anaconda3-2020.02-Windows-x86
- python第三方库：见'requirements.txt'文件
- Windows平台类Linux终端仿真器：cmder，下载链接[cmder](https://cmder.app/)

# 快速入门

在`scripts`目录下，打开cmder可运行以下脚本，脚本文件中是执行`python xxx.py`运行文件的命令行：

- 训练教师脚本：`bash run_teacher.sh`
- 训练学生脚本：`bash run_student.sh`
- KD+TS脚本：`bash run_ts.sh`
- KD+ATS脚本：`bash run_ats.sh`
- 导出MINDIR模型脚本：`bash run_export.sh`

# 脚本说明

## 脚本结构与说明

```text
└──ats
  ├── README.md
  ├── scripts
    ├── run_teacher.sh                     # 训练教师网络
    ├── run_student.sh                     # 训练学生网络
    ├── run_ts.sh                          # KD+TS
    ├── run_ts.sh                          # KD+ATS
    └── run_export.sh                      # 导出MindIR模型
  ├── src
    ├── data.py                            # 数据加载
    ├── vgg.py                             # VGG神经网络
    ├── resnet.py                          # ResNet神经网络
    ├── classify.py                        # 封装的训练模型类
    ├── distill.py                         # 封装的知识蒸馏类
    └── utils.py                           # 辅助工具
  ├── run_classify.py                      # 训练神经网络运行入口
  ├── run_distill.py                       # 知识蒸馏运行入口
  ├── requirements.txt                     # 需求文件
  └── export.py                            # 导出MindIR模型
```

# 脚本参数

## 网络训练参数

```python run_classify.py
"dataset": 'cifar100',                     # 数据集名称
"data_dir": './data',                      # 数据集目录
"download": True,                          # 数据集是否下载
"n_classes": 100,                          # 数据集类别数目
"net": 'VGG',                              # 网络结构系列
"n_layer": 8,                              # 网络结构深度
"net_name": 'VGG8',                        # 网络结构名称
"epoches": 240,                            # 训练轮数
"lr": 0.03,                                # 训练学习率
"momentum": 0.9,                           # 训练SGD动量
"batch_size": 128,                         # 训练批大小
"log_dir": './logs',                       # 日志保存目录
"log_name": 'student-nokd.log',            # 日志保存路径
"ckpt_dir": './ckpts',                     # 模型保存目录
"ckpt_name": 'cifar100-VGG8.ckpt',         # 模型保存路径
```

## 知识蒸馏参数

```python run_classify.py
"dataset": 'cifar100',                     # 数据集名称
"data_dir": './data',                      # 数据集目录
"download": True,                          # 数据集是否下载
"n_classes": 100,                          # 数据集类别数目
"t_net": 'ResNet',                         # 教师网络系列
"t_n_layer": 14,                           # 教师网络结构深度
"t_net_name": 'ResNet14',                  # 教师网络结构名称
"net": 'VGG',                              # 学生网络结构系列
"n_layer": 8,                              # 学生网络结构深度
"net_name": 'VGG8',                        # 学生网络结构名称
"kd_way": "ATS",                           # 知识蒸馏温度缩放方法：TS或者ATS
"tp_tau": 4.0,                             # 教师网络正确类别的温度系数
"t_tau": 2.0,                              # 教师网络错误类别的温度系数
"s_tau": 1.0,                              # 学生网络所有类别的温度系数
"lamb": 0.5,                               # 学生网络损失的平衡因子
"epoches": 240,                            # 训练轮数
"lr": 0.03,                                # 训练学习率
"momentum": 0.9,                           # 训练SGD动量
"batch_size": 128,                         # 训练批大小
"log_dir": './logs',                       # 日志保存目录
"log_name": 'student-kd-ats.log',          # 日志保存路径
"ckpt_dir": './ckpts',                     # 模型保存目录
"ckpt_name": 'cifar100-RES14-VGG8.ckpt',   # 模型保存路径
```

# 网络训练过程

## 用法

```Shell
echo "Train teacher network ResNet14 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 14 --net_name ResNet14 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet14.ckpt

echo "Train teacher network ResNet110 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 110 --net_name ResNet110 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet110.ckpt
```

## 结果

```text
[Epoch:1] [Loss:4.603412429491678] [TrAcc:0.011067708333333334] [TeAcc:0.0107421875]
[Epoch:2] [Loss:4.6048067808151245] [TrAcc:0.010323660714285714] [TeAcc:0.008634868421052632]
...
```

# 知识蒸馏过程

## 用法

```Shell
echo "Distill konwledge of teacher network ResNet14 to student network VGG8 using TS (tp_tau = t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 14 --t_net_name ResNet14 --net VGG --n_layer 8 --net_name VGG8 --kd_way TS --lamb 0.5 --tp_tau 4.0 --t_tau 4.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet14-VGG8-TS.ckpt

echo "Distill konwledge of teacher network ResNet110 to student network VGG8 using TS (tp_tau = t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 110 --t_net_name ResNet110 --net VGG --n_layer 8 --net_name VGG8 --kd_way TS --lamb 0.5 --tp_tau 4.0 --t_tau 4.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet110-VGG8-TS.ckpt

echo "Distill konwledge of teacher network ResNet14 to student network VGG8 using ATS (tp_tau > t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 14 --t_net_name ResNet14 --net VGG --n_layer 8 --net_name VGG8 --kd_way ATS --lamb 0.5 --tp_tau 4.0 --t_tau 2.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet14-VGG8-ATS.ckpt

echo "Distill konwledge of teacher network ResNet110 to student network VGG8 using ATS (tp_tau > t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 110 --t_net_name ResNet110 --net VGG --n_layer 8 --net_name VGG8 --kd_way ATS --lamb 0.5 --tp_tau 4.0 --t_tau 2.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet110-VGG8-ATS.ckpt
```

## 结果

```text
[Epoch:1] [Loss:4.607254505157471] [TrAcc:0.008831521739130434] [TeAcc:0.021577380952380952]
[Epoch:2] [Loss:4.593390011787415] [TrAcc:0.03515625] [TeAcc:0.0390625]
...
```

# 随机情况说明

可设置随机种子，不设置随机种子时候最终模型收敛性能标准差大概为0.5%。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。
