# 目录

- [目录](#目录)
- [HRNet_cls描述](#HRNet_cls描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

# HRNet_cls描述

HRNet是一个全能型的CV骨干网络，可用于图像分类、语义分割、面部识别等多种CV任务的特征提取阶段。该网络通过在整个处理过程中连接从高到低分辨率的卷积来维持高分辨率表示，并通过重复地融合不同成并行分支的卷积来产生强分辨率表示。

如下为MindSpore使用ImageNet数据集对HRNetW48进行训练的示例，完成图像分类任务。W48表示网络宽度（一号分支特征图的通道数）为48。

[论文](https://arxiv.org/pdf/1908.07919.pdf) ：Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao.

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```text
├─ HRNet_cls
│   ├─ README_CN.md                     # HRNet_cls相关描述
│   ├─ scripts
│   │   ├─ run_standalone_train.sh      # 用于单卡训练的shell脚本
│   │   ├─ run_distribute_train.sh      # 用于八卡训练的shell脚本
│   │   └─ run_eval.sh                  # 用于评估的shell脚本
│   ├─ src
│   │   ├─ model_utils                  # modelarts训练适应脚本
│   │   │   └─ moxing_adapter.py
│   │   ├─ callback.py                  # 参数配置
│   │   ├─ cls_hrnet.py                 # HRNet_cls架构
│   │   ├─ config.py                    # 配置参数
│   │   ├─ dataset.py                   # 创建数据集
│   │   ├─ loss.py                      # 损失函数
│   │   └─ utils.py                     # 工具函数脚本
│   ├─ eval.py                          # 评估脚本
│   ├─ export.py                        # 模型格式转换脚本
│   └─ train.py                         # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```text
'train_url': None,                      # 训练输出路径（桶）
'train_path': None,                     # 训练输出路径
'data_url': None,                       # 训练数据集路径（桶）
'data_path': None,                      # 训练数据集路径
'checkpoint_url': None,                 # checkpoint路径（桶）
'checkpoint_path': None,                # checkpoint路径
'eval_data_url': None,                  # 推理数据集路径（桶）
'eval_data_path': None,                 # 推理数据集路径
'eval_interval': 10,                    # 训练时推理的时间间隔
'modelarts': False,                     # 是否使用modelarts
'run_distribute': False,                # 是否多卡训练
'device_target': 'Ascend',              # 训练平台
'begin_epoch': 0,                       # 开始训练周期
'end_epoch': 120,                       # 结束训练周期
'total_epoch': 120,                     # 总训练周期数
'dataset': 'imagenet',                  # 数据集名称
'num_classes': 1000,                    # 数据集类别数
'batchsize': 16,                        # 输入批次大小
'input_size': 224,                      # 输入尺寸大小
'lr_scheme': 'linear',                  # 学习率衰减方案
'lr': 0.01,                             # 最大学习率
'lr_init': 0.0001,                      # 初始学习率
'lr_end': 0.00001,                      # 最终学习率
'warmup_epochs': 2,                      # 热身周期数
'use_label_smooth': True,               # 是否使用label smooth
'label_smooth_factor': 0.1,             # 标签平滑因子
'conv_init': 'TruncatedNormal',         # 卷积层初始化方式
'dense_init': 'RandomNormal',           # 全连接层初始化方式
'optimizer': 'rmsprop',                 # 优化器
'loss_scale': 1024,                     # loss scale
'opt_momentum': 0.9,                    # 动量参数
'wd': 1e-5,                             # 权重衰减率
'eps': 0.001                            # epsilon
'save_ckpt': True,                      # 是否保存ckpt文件
'save_checkpoint_epochs': 1,            # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 10,              # 保存ckpt文件的最大数量
'model': {...}                          # HRNet模型结构参数
```

## 训练过程

### 启动

```bash
# 训练示例
# 单卡训练
bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH](optional) [BEGIN_EPOCH](optional) [EVAL_DATASET_PATH](optional)
# 多卡训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH](optional) [BEGIN_EPOCH](optional) [EVAL_DATASET_PATH](optional)
# CHECKPOINT_PATH & BEGIN_EPOCH 用于从指定周期恢复训练
# EVAL_DATASET_PATH 用于启动训练时推理
```

### 结果

ckpt文件将存储在自定义路径下，训练日志将被记录到 `log` 中。训练日志部分示例如下：

```text
epoch: [ 1/120], epoch time: 2404040.882, steps: 10009, per step time: 240.188, avg loss: 4.093, lr:[0.005]
epoch: [ 2/120], epoch time: 827142.272, steps: 10009, per step time: 82.640, avg loss: 4.234, lr:[0.010]
epoch: [ 3/120], epoch time: 825985.514, steps: 10009, per step time: 82.524, avg loss: 3.057, lr:[0.010]
epoch: [ 4/120], epoch time: 825988.881, steps: 10009, per step time: 82.525, avg loss: 3.093, lr:[0.010]
```

## 评估过程

### 启动

```bash
# 评估示例
bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

### 结果

可以在 `eval_log` 查看评估结果。

```text
{'Loss': 1.9160713648223877, 'Top_1_Acc': 0.79358, 'Top_5_Acc': 0.9456}
```

# 模型说明

## 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | HRNet                          |
| 模型版本                    | W48-cls                           |
| 运行环境                    | HUAWEI CLOUD Modelarts                     |
| 上传时间                    | 2021-11-21                             |
| 数据集                      | imagenet                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | RMSProp                              |
| 损失函数                    | CrossEntropySmooth         |
| 最终损失                    | 1.67                                 |
| 精确度 (8p)                 | Top1[79.4%], Top5[94.6%]               |
| 训练总时间 (8p)             | 28.7h                                    |
| 评估总时间                  | 5min                                    |
| 参数量 (M)                 | 296M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_cls) |

# 随机情况的描述

我们在 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models)。
