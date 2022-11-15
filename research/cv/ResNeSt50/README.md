# 目录

- [ResNeSt说明](#resnest说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#训练用法)
        - [样例](#训练样例)
    - [评估过程](#评估过程)
        - [用法](#评估用法)
        - [样例](#评估样例)
        - [结果](#评估结果)
    - [推理过程](#推理过程)
        - [模型导出](#模型导出)
        - [用法](#推理用法)
        - [样例](#推理样例)
        - [结果](#推理结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# ResNeSt说明

ResNeSt是一个高度模块化的图像分类网络架构。ResNeSt的设计为统一的、多分支的架构，该架构仅需设置几个超参数。此策略提供了一个新维度，我们将其称为“基数”（转换集的大小），它是深度和宽度维度之外的一个重要因素。

[论文](https://arxiv.org/abs/2004.08955)：  Hang Zhang, Chongruo Wu, Alexander Smola et al. ResNeSt: Split-Attention Networks. 2020.

# 模型架构

ResNeSt整体网络架构如下：

[链接](https://arxiv.org/abs/2004.08955)

# 数据集

使用的数据集：[ImageNet](http://www.image-net.org/)

- 数据集大小：共1000个类，包含128万张彩色图像
    - 训练集：120G，128万张图像
    - 测试集：5G，5万张图像
- 数据格式：RGB图像。
    - 注：数据在src/datasets中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```path
.
└─ResNeSt50
  ├─scripts
    ├─run_train.sh
    ├─run_eval.sh
    ├─run_train_gpu.sh
    ├─run_eval_gpu.sh
    ├─run_distribute_train.sh              # 启动Ascend分布式训练（8卡）
    ├─run_distribute_eval.sh               # 启动Ascend分布式评估（8卡）
    ├─run_distribute_train_gpu.sh              # 启动GPU分布式训练（8卡）
    └─run_infer_310.sh                     # 启动310推理
  ├─src
    ├─datasets
      ├─autoaug.py                  # 随机数据增强方法
      ├─dataset.py                  # 数据集处理
    ├─models
      ├─resnest.py                  # ResNeSt50网络定义
      ├─resnet.py                   # 主干网络
      ├─splat.py                    # split-attention
      ├─utils.py                    # 工具函数：网络获取、加载权重等
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─eval_callback.py                # 推理信息打印
    ├─logging.py                      # 日志记录
  ├──eval.py                          # 评估网络
  ├──train.py                         # 训练网络
  ├──export.py                        # 导出Mindir接口
  ├──create_imagenet2012_label.py     # 创建数据集标签用于310推理精度验证
  ├──postprocess.py                   # 后处理
  └──README.md                        # README文件
```

## 脚本参数

在config.py中可以同时配置训练和评估参数。

```python
"net_name": 'resnest50'                   # 网络选择
"root": "/home/mindspore/dataset/imagenet_original"   # 数据集路径
"num_classes": 1000,                      # 数据集类数
"base_size": 224,                         # 图像大小
"crop_size": 224,                         # crop大小
"label_smoothing": 0.1,                   # 标签平滑
"batch_size": 64,                         # 输入张量的批次大小，不能超过64
"test_batch_size": 64,                    # 测试批次大小
"last_gamma": True,                       # zero bn last gamma
"final_drop": 1.0,                        # final_drop
"epochs": 270,                            # epochs
"start_epoch": 0,                         # start epochs
"num_workers": 64,                        # num_workers
"lr": 0.025,                              # 基础学习率,多卡训练乘以卡数
"lr_scheduler": 'cosine_annealing',       # 学习率模式
"lr_epochs": '30,60,90,120,150,180,210,240,270',            # LR变化轮次
"lr_gamma": 0.1,                          # 减少LR的exponential lr_scheduler因子
"eta_min": 0,                             # cosine_annealing调度器中的eta_min
"T_max": 270,                             # cosine_annealing调度器中的T-max
"max_epoch": 270,                         # 训练模型的最大轮次数量
"warmup_epochs" : 5,                      # 热身轮次
"weight_decay": 0.0001,                   # 权重衰减
"momentum": 0.9,                          # 动量
"is_dynamic_loss_scale": 0,               # 动态损失放大
"loss_scale": 1024,                       # 损失放大
"disable_bn_wd": True,                    # batchnorm no weight decay
```

## 训练过程

### 训练用法

首先需要在`src/config.py`中设置好超参数以及数据集路径等参数，接着可以通过脚本或者.py文件进行训练

您可以通过python脚本开始训练：

```shell
Ascend:
   python train.py --outdir ./output --device_target [device]
GPU:
   python train.py --outdir ./output --device_target [device]
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash run_distribute_train.sh RANK_TABLE_FILE
    # 单机训练
    bash run_train.sh OUTPUT_DIR
GPU:
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM]
    # 单机训练
    bash scripts/run_train_gpu.sh
```

### 训练样例

```shell
# Ascend分布式训练示例（8卡）
bash run_distribute_train.sh RANK_TABLE_FILE
# Ascend单机训练示例
bash run_train.sh OUTPUT_DIR
# GPU分布式训练示例（8卡）
bash scripts/run_distribute_train_gpu.sh 8
# GPU单机训练示例
bash scripts/run_train_gpu.sh
```

您可以在日志中找到检查点文件和结果。

## 评估过程

### 评估用法

您可以通过python脚本开始评估：

```shell
Ascend:
python eval.py --outdir ./output --pretrained_ckpt_path ~/resnest50-270_2502.ckpt
GPU:
python eval.py --outdir ./output --pretrained_ckpt_path ~/resnest50-270_2502.ckpt --device_target “GPU”
```

或通过shell脚本开始评估：

```shell
# 评估
Ascend:
bash scripts/run_eval.sh [OUT_DIR] [PRETRAINED_CKPT_PATH]
GPU:
bash scripts/run_eval_gpu.sh [OUT_DIR] [PRETRAINED_CKPT_PATH]
```

### 评估样例

```shell
# 检查点评估
Ascend:
bash scripts/run_eval.sh OUT_DIR PRETRAINED_CKPT_PATH
GPU:
bash scripts/run_eval_gpu.sh OUT_DIR PRETRAINED_CKPT_PATH

#或者直接使用脚本运行
python eval.py --outdir ./output --pretrained_ckpt_path ~/resnest50-270_2502.ckpt
```

### 评估结果

Ascend评估结果保存在脚本路径`/scripts/EVAL_LOG/`下。您可以在日志中找到类似以下的结果。

```log
acc=80.90%(TOP1)
acc=95.51%(TOP5)
```

GPU评估结果保存在脚本路径`/output1/valid下。您可以在日志中找到类似以下的结果。

```log
2022-01-30 12:10:33,478:INFO:Inference Performance: 379.23 img/sec
2022-01-30 12:10:33,478:INFO:before results=[[40525], [47716], [49984]]
2022-01-30 12:10:33,479:INFO:after results=[[40525],[47716],[49984]]
2022-01-30 12:10:33,479:INFO:after allreduce eval: top1_correct=40525, tot=49984,acc=81.08%(TOP1)
2022-01-30 12:10:33,479:INFO:after allreduce eval: top5_correct=47716, tot=49984,acc=95.46%(TOP5)
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在Ascend310执行推理，执行推理之前，需要通过`export.py`文件导出MINDIR模型

### 模型导出

```shell
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH] --net_name [NET_NAME] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"].

### 推理用法

通过shell脚本编译文件并在310上执行推理

```shell
# 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

PLATFORM is Ascend310, default is Ascend310.

### 推理样例

```shell
# 直接使用脚本运行
bash run_infer_310.sh /home/stu/lds/mindir/resnest50.mindir /home/MindSpore_dataset/ImageNet2012/val 0
```

### 推理结果

评估结果保存在脚本路径`/scripts/`下。您可以在`acc.log`找到精度结果，在`infer.log`中找到性能结果

```log
acc=0.8088(TOP1)
acc=0.9548(TOP5)
```

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend 910                                                  |  GPU                                                  |
| -------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| 资源                       | Ascend 910；CPU：2.60GHz，192核；内存：755GB | GeForce RTX 3090 ；CPU 2.90GHz，16cores；内存，252G        |
| 上传日期                   | 2021-11-09                                                  | 2022-2-15                                                  |
| MindSpore版本              | 1.3                                                        | 1.5                                                        |
| 数据集                     | ImageNet                                                   | ImageNet                                                   |
| 训练参数                   | src/config.py                                              | src/config.py                                              |
| 优化器                     | Momentum                                                   | Momentum                                                   |
| 损失函数                   | Softmax交叉熵                                              | Softmax交叉熵                                              |
| 损失                       | 1.466                                                     | 1.5859                                                     |
| 准确率                     | 80.9%(TOP1)                                               | 81.08%(TOP1)                                               |
| 总时长                     | 84h21m39s （8卡）                                        | 66h42m42s185（8卡）                                        |
| 调优检查点                 | 223 M（.ckpt文件）                                         | 212 M（.ckpt文件）                                         |

### 推理性能

| 参数                       |  Ascend 910   |  GPU             |
| -------------------------- | -------------------- | -------------------- |
| 资源                       | Ascend 910           | GeForce RTX 3090     |
| 上传日期                   | 2021-11-09           |2022-2-15            |
| MindSpore版本              | 1.3                  | 1.5                  |
| 数据集                     | ImageNet， 5万       | ImageNet， 5万       |
| batch_size                 | 1                    | 1                    |
| 输出                       | 分类准确率           | 分类准确率           |
| 准确率                     | acc=80.9%(TOP1)      | acc=81.08%(TOP1)      |

# 随机情况说明

dataset.py中设置了“ImageNet”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

