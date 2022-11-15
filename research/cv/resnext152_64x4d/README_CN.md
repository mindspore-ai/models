# 目录

- [目录](#目录)
- [ResNeXt152说明](#resnext152说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [样例](#样例)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [样例](#样例-1)
            - [结果](#结果)
    - [模型导出](#模型导出)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# ResNeXt152说明

ResNeXt是一个简单、高度模块化的图像分类网络架构。ResNeXt的设计为统一的、多分支的架构，该架构仅需设置几个超参数。此策略提供了一个新维度，我们将其称为“基数”（转换集的大小），它是深度和宽度维度之外的一个重要因素。

[论文](https://arxiv.org/abs/1611.05431)：  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

# 模型架构

ResNeXt整体网络架构如下：

[链接](https://arxiv.org/abs/1611.05431)

# 数据集

使用的数据集：[ImageNet](http://www.image-net.org/)

- 数据集大小：约125G, 共1000个类，224*224彩色图像
    - 训练集：120G，共1281167张图像
    - 测试集：5G，共50000张图像
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```python
.
└─resnext152_64x4d
  ├─ascend310_infer                   # 310的推理代码
    ├─inc
      ├─utils.h                       # 工具库头文件
    ├─src
      ├─build.sh                      # 运行脚本
      ├─CMakeLists.txt                # cmake文件
      ├─main_preprocess.cc            # 预处理
      ├─main.cc                       # 主函数入口
      ├─utils.cc                      # 工具库函数实现
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # 启动Ascend单机训练（单卡）
    ├─run_standalone_train_gpu.sh     # launch standalone training for gpu (1p)
    ├─run_distribute_train.sh         # 启动Ascend分布式训练（8卡）
    ├─run_distribute_train_gpu.sh     # launch distributed training for gpu (8p)
    ├─run_eval.sh                     # 启动评估
    └─run_eval_gpu.sh                 # launch evaluating for gpu
  ├─src
    ├─backbone
      ├─_init_.py                     # 初始化
      ├─resnet.py                     # ResNeXt152骨干
    ├─model_utils
      ├─config.py                     # 相关参数
      ├─device_adapter.py             # Device adapter for ModelArts
      ├─local_adapter.py              # Local adapter
      ├─moxing_adapter.py             # Moxing adapter for ModelArts
    ├─utils
      ├─_init_.py                     # 初始化
      ├─auto_mixed_precision.py       # 混合精度
      ├─cunstom_op.py                 # 网络操作
      ├─logging.py                    # 打印日志
      ├─optimizers_init_.py           # 获取参数
      ├─sampler.py                    # 分布式采样器
      ├─var_init_.py                  # 计算增益值
    ├─_init_.py                       # 初始化
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─dataset.py                      # 数据预处理
    ├─eval_callback.py                # 训练时推理
    ├─head.py                         # 常见头
    ├─image_classification.py         # 获取ResNet
    ├─metric.py                       # 推理
    ├─linear_warmup.py                # 线性热身学习率
    ├─warmup_cosine_annealing.py      # 每次迭代的学习率
    ├─warmup_step_lr.py               # 热身迭代学习率
  ├─create_imagenet2012_label.py      # 创建标签
  ├─default_config.yaml               # 参数
  ├─eval.py                           # 评估网络
  ├─export.py                         # export mindir script
  ├─postprocess.py                    # 310的后期处理
  ├─train.py                          # 训练网络
  ├─requirements.txt                  # 需要的python库
  ├─README.md                         # Documentation in English
  ├─README_CN.md                      # Documentation in Chinese
```

## 脚本参数

在config.py中可以同时配置训练和评估参数。

```config
"image_size": '224,224'                   # 图像大小
"num_classes": 1000,                      # 数据集类数
"per_batch_size": 128,                    # 输入张量的批次大小
"lr": 0.05,                               # 基础学习率
"lr_scheduler": 'cosine_annealing',       # 学习率模式
"lr_epochs": '30,60,90,120',              # LR变化轮次
"lr_gamma": 0.1,                          # 减少LR的exponential lr_scheduler因子
"eta_min": 0,                             # cosine_annealing调度器中的eta_min
"T_max": 150,                             # cosine_annealing调度器中的T-max
"max_epoch": 150,                         # 训练模型的最大轮次数量
"backbone": 'resnext152',                 # 骨干网络
"warmup_epochs" : 1,                      # 热身轮次
"weight_decay": 0.0001,                   # 权重衰减
"momentum": 0.9,                          # 动量
"is_dynamic_loss_scale": 0,               # 动态损失放大
"loss_scale": 1024,                       # 损失放大
"label_smooth": 1,                        # 标签平滑
"label_smooth_factor": 0.1,               # 标签平滑因子
"ckpt_interval": 2000,                    # 检查点间隔
"ckpt_path": 'outputs/',                  # 检查点保存位置
"is_save_on_master": 1,
"rank": 0,                                # 分布式本地进程序号
"group_size": 1                           # 分布式进程总数
```

## 训练过程

### 用法

您可以通过python脚本开始训练：

```shell
python train.py --data_dir ~/imagenet/train/ --platform Ascend --is_distributed 0
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    bash run_standalone_train.sh DEVICE_ID DATA_PATH
```

### 样例

```shell
# Ascend分布式训练示例（8卡）
bash scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
# Ascend单机训练示例
bash scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
```

您可以在日志中找到检查点文件和结果。

## 评估过程

### 用法

您可以通过python脚本开始训练：

```shell
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained resnext.ckpt
```

或通过shell脚本开始训练：

```shell
# 评估
bash run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH PLATFORM
```

PLATFORM is Ascend, default is Ascend.

#### 样例

```shell
# 检查点评估
bash scripts/run_eval.sh DEVICE_ID PRETRAINED_CKPT_PATH PLATFORM

#或者直接使用脚本运行
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained ~/best_acc_0.ckpt
```

#### 结果

评估结果保存在脚本路径下。您可以在日志中找到类似以下的结果。

```log
acc=80.08%(TOP1)
acc=94.71%(TOP5)
```

Example for the GPU evaluation:

```text
...
[DATE/TIME]:INFO:load model /path/to/checkpoints/ckpt_0/0-148_10009.ckpt success
[DATE/TIME]:INFO:Inference Performance: 218.14 img/sec
[DATE/TIME]:INFO:before results=[[39666], [46445], [49984]]
[DATE/TIME]:INFO:after results=[[39666] [46445] [49984]]
[DATE/TIME]:INFO:after allreduce eval: top1_correct=39666, tot=49984,acc=79.36%(TOP1)
[DATE/TIME]:INFO:after allreduce eval: top5_correct=46445, tot=49984,acc=92.92%(TOP5)
```

## 模型导出

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "ONNX", "MINDIR"].

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 用法

在执行推理之前，需要通过export.py导出mindir文件。目前仅可处理batch_size为1。

```shell
#Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`MINDIR_PATH`为生成的mindir的路径，`DATA_PATH`为imagenet的数据集路径，`DEVICE_ID`可选，默认值为0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```shell
Total data: 50000, top1 accuracy: 0.79174, top5 accuracy: 0.94178.
```

# 模型描述

## 性能

### 训练性能

| 参数       | ResNeXt152                                    | ResNeXt152                                   |
| ---------- | --------------------------------------------- | -------------------------------------------- |
| 资源       | Ascend 910, cpu:2.60GHz 192cores, memory:755G | 8x V100, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| 上传日期   | 06/30/2021                                    | 06/30/2021                                   |
| 版本信息   | 1.3                                           | 1.5.0 (docker build, CUDA 11.1)              |
| 数据集     | ImageNet                                      | ImageNet                                     |
| 训练参数   | src/config.py                                 | src/config.py; lr=0.05, per_batch_size=16    |
| 优化器     | Momentum                                      | Momentum                                     |
| 损失函数   | SoftmaxCrossEntropy                           | SoftmaxCrossEntropy                          |
| 损失       | 1.28923                                       | 2.172222                                     |
| 准确率     | 80.08%(TOP1)                                  | 79.36%(TOP1) (148 epoch, early stopping)     |
| 总时长     | 7.8 h 8ps                                     | 2 days 45 minutes (8P, processes)            |
| 调优检查点 | 192 M(.ckpt file)                             | -                                            |

#### 推理性能

| 参数       |                  |                  |                  |
| ---------- | ---------------- | ---------------- | ---------------- |
| 资源       | Ascend 910       | GPU V100         | Ascend 310       |
| 上传日期   | 06/20/2021       | 2021-10-27       | 2021-10-27       |
| 版本信息   | 1.2              | 1.5.0, CUDA 11.1 | 1.3.0            |
| 数据集     | ImageNet, 1.2W   | ImageNet, 1.2W   | ImageNet, 1.2W   |
| batch_size | 1                | 32               | 1                |
| outputs    | probability      | probability      | probability      |
| 准确率     | acc=80.08%(TOP1) | acc=79.36%(TOP1) | acc=79.34%(TOP1) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
