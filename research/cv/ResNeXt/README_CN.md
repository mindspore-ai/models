# 目录

- [目录](#目录)
- [ResNeXt说明](#resnext说明)
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
    - [迁移训练过程](#迁移训练过程)
        - [迁移数据集处理](#迁移数据集处理)
        - [用法](#用法-1)
    - [评估过程](#评估过程)
        - [用法](#用法-2)
            - [样例](#样例)
            - [结果](#结果)
    - [迁移训练推理过程](#迁移训练推理过程)
        - [用法](#用法-3)
    - [模型导出](#模型导出)
    - [推理过程](#推理过程)
        - [用法](#用法-4)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# ResNeXt说明

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

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2) 在modelarts的界面上设置代码的路径 "/path/ResNeXt"。
# (3) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (4) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (5) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 在modelarts的界面上设置代码的路径 "/path/ResNeXt"。
# (4) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的推理。
```

# 脚本说明

## 脚本及样例代码

```path
.
└─ResNeXt
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train.sh         # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh         # 启动Ascend分布式训练（8卡）
    ├─run_standalone_train_for_gpu.sh # 启动GPU单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh # 启动GPU分布式训练（8卡）
    ├─run_eval.sh                     # 启动评估
    └─run_onnx_eval.sh                # 启动Onnx评估
  ├─src
    ├─backbone
      ├─_init_.py                     # 初始化
      ├─resnet.py                     # ResNeXt骨干
    ├─utils
      ├─_init_.py                     # 初始化
      ├─cunstom_op.py                 # 网络操作
      ├─logging.py                    # 打印日志
      ├─optimizers_init_.py           # 获取参数
      ├─sampler.py                    # 分布式采样器
      ├─var_init_.py                  # 计算增益值
    ├─_init_.py                       # 初始化
    ├─config.py                       # 参数配置
    ├─crossentropy.py                 # 交叉熵损失函数
    ├─dataset.py                      # 数据预处理
    ├─head.py                         # 常见头
    ├─image_classification.py         # 获取ResNet
    ├─linear_warmup.py                # 线性热身学习率
    ├─warmup_cosine_annealing.py      # 每次迭代的学习率
    ├─warmup_step_lr.py               # 热身迭代学习率
  ├─model_utils
    ├──config.py                      # 参数配置
    ├──device_adapter.py              # 设备配置
    ├──local_adapter.py               # 本地设备配置
    ├──moxing_adapter.py              # modelarts设备配置
  ├──eval.py                          # 评估网络
  ├──eval_onnx.py                     # onnx评估网络
  ├──train.py                         # 训练网络
  ├──fine_tune.py                     # 迁移训练网络（cpu）
  ├──cpu_default_config.yaml          # 训练配置参数（cpu）
  ├──quick_start.py                   # quick start演示文件（cpu）
  ├──data_split.py                    # 切分迁移数据集脚本（cpu）
  ├──requirements.txt                 # 第三方依赖
  ├──mindspore_hub_conf.py            # MindSpore Hub接口
```

## 脚本参数

在config.py中可以同时配置训练和评估参数。

```python
"image_height": '224,224'                 # 图像大小
"num_classes": 1000,                      # 数据集类数
"per_batch_size": 128,                    # 输入张量的批次大小
"lr": 0.05,                               # 基础学习率
"lr_scheduler": 'cosine_annealing',       # 学习率模式
"lr_epochs": '30,60,90,120',              # LR变化轮次
"lr_gamma": 0.1,                          # 减少LR的exponential lr_scheduler因子
"eta_min": 0,                             # cosine_annealing调度器中的eta_min
"T_max": 150,                             # cosine_annealing调度器中的T-max
"max_epoch": 150,                         # 训练模型的最大轮次数量
"backbone": 'ResNeXt',                  # 骨干网络
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
python train.py --data_path ~/imagenet/train/ --device_target Ascend --run_distribute 0
```

或通过shell脚本开始训练：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    bash run_standalone_train.sh DEVICE_ID DATA_PATH
GPU:
    # 分布式训练示例（8卡）
    bash run_distribute_train_for_gpu.sh DATA_PATH
    # 单机训练
    bash run_standalone_train_for_gpu.sh DEVICE_ID DATA_PATH
GPU_ResNext101:
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train_for_gpu_resnext101.sh DATA_PATH CONFIG_PATH
    # 单机训练
    bash scripts/run_standalone_train_for_gpu_resnext101.sh DEVICE_ID DATA_PATH CONFIG_PATH
```

## 迁移训练过程

### 迁移数据集处理

[根据提供的数据集链接下载数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz ),下载后的数据集保存到新建文件夹dataset里，将切分数据集脚本data_split.py放置在flower_photos目录下，运行后会生成train文件夹及test文件夹。

### 用法

您可以通过python脚本开始训练：

```shell
python fine_tune.py --config_path ./cpu_default_config.yaml
```

## 评估过程

### 用法

您可以通过python脚本开始训练：

```shell
python eval.py --data_path ~/imagenet/val/ --device_target Ascend --checkpoint_file_path resnext.ckpt
```

```shell
# Onnx评估
python eval_onnx.py --data_path ~/imagenet/val/ --device_target GPU --onnx_file_path resnext.onnx
```

或通过shell脚本开始训练：

```shell
# GPU_ResNext101评估
bash scripts/run_eval_for_gpu_resnext101.sh DEVICE_ID DATA_PATH CHECKPOINT_FILE_PATH CONFIG_PATH
# 评估
bash scripts/run_eval.sh DEVICE_ID DATA_PATH CHECKPOINT_FILE_PATH PLATFORM
```

```shell
# Onnx评估
bash scripts/run_onnx_eval.sh DEVICE_ID DATA_PATH ONNX_FILE_PATH PLATFORM
```

DEVICE_TARGET is Ascend or GPU, default is Ascend.

#### 样例

```shell
# 检查点评估
bash scripts/run_eval.sh 0 /opt/npu/datasets/classification/val /ResNeXt_100.ckpt Ascend
# GPU_ResNext101评估
bash scripts/run_eval_for_gpu_resnext101.sh 0 ~/ImageNet2012/val ~/outputs_demo/best_acc.ckpt ./resnext101_config.yaml
```

#### 结果

评估结果保存在脚本路径下。您可以在日志中找到类似以下的结果。

## 迁移训练推理过程

### 用法

您可以通过python脚本开始训练(需要先到cpu_default_config.yaml配置文件中将ckpt_path设为最好的ckpt文件路径):

```shell
python eval.py --config_path ./cpu_default_config.yaml --data_path ./dataset/flower_photos/test
```

```resnext50
acc=78.16%(TOP1)
acc=93.88%(TOP5)
```

```resnext101
acc=79.86%(TOP1)
acc=94.72%(TOP5)
```

## 模型导出

本地导出mindir

```shell
python export.py --device_target [PLATFORM] --checkpoint_file_path [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`checkpoint_file_path` 参数为必填项。
`EXPORT_FORMAT` 可选 ["AIR", "MINDIR", "ONNX"]。

ModelArts导出mindir

```python
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./resnext50'"参数在yaml文件。
#          设置 "file_format='MINDIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./resnext50'"参数在modearts的界面上。
#          设置 "file_format='MINDIR'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/resnext50"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

## [推理过程](#contents)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 用法

在执行推理之前，需要通过export.py导出mindir文件。
目前仅可处理batch_Size为1。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [MODEL_DEVICE_TYPE] [INFER_DEVICE_TYPE] [DATA_PATH] [DEVICE_ID]
```

`MODEL_DEVICE_TYPE` 生成MindIR的环境, 在'Ascend'，'GPU'中选择。
`INFER_DEVICE_TYPE` 执行推理的环境, 在'Ascend'，'GPU'和'CPU'中选择。
`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```resnext50
Total data:50000, top1 accuracy:0.78462, top5 accuracy:0.94182
```

```resnext101
Total data:50000, top1 accuracy:0.79858, top5 accuracy:0.94716
```

# 模型描述

## 性能

### 训练性能

| 参数 | ResNeXt50 | |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8              | NV SMX2 V100-32G          |
| 上传日期              | 2021-7-05                                           | 2021-07-05      |
| MindSpore版本          | 1.3.0                                                      | 1.3.0                     |
| 数据集 | ImageNet | ImageNet |
| 训练参数        | src/config.py                                           | src/config.py          |
| 优化器                  | Momentum                                                        | Momentum                 |
| 损失函数             | Softmax交叉熵 | Softmax交叉熵 |
| 损失                       | 1.76592 | 1.8965 |
| 准确率 | 78%(TOP1)                                                  | 77.8%(TOP1)               |
| 总时长                 | 7.8小时 （8卡） | 21.5小时 （8卡） |
| 调优检查点 | 192 M（.ckpt文件） | 192 M（.ckpt文件） |

| 参数          | ResNeXt101                                        |
| ------------- | ------------------------------------------------- |
| 资源          | GeForce RTX 3090；CPU：3.50GHz，64核；内存：264GB |
| 上传日期      | 2022-6-30                                         |
| MindSpore版本 | 1.7.0                                             |
| 数据集        | ImageNet                                          |
| 训练参数      | src/config.py                                     |
| 优化器        | Momentum                                          |
| 损失函数      | Softmax交叉熵                                     |
| 损失          | 1.2427                                            |
| 准确率        | 78.04%(TOP1)                                      |
| 总时长        | 50小时 （8卡）                                    |
| 调优检查点    | 638 M（.ckpt文件）                                |

#### 推理性能

| 参数                 |ResNeXt50                            |                           |                      |
| -------------------------- | ----------------------------- | ------------------------- | -------------------- |
| 资源                   | Ascend 910；系统 Euler2.8                    | NV SMX2 V100-32G          | Ascend 310           |
| 上传日期              | 2021-7-05                                           | 2021-07-05   | 2021-07-05      |
| MindSpore版本         | 1.3.0                         | 1.3.0                     | 1.3.0                |
| 数据集 | ImageNet， 1.2万 | ImageNet， 1.2万 | ImageNet， 1.2万 |
| batch_size                 | 1                             | 1                         | 1                    |
| 输出 | 概率 | 概率 | 概率 |
| 准确率 | acc=78.16%(TOP1)              | acc=78.05%(TOP1)          |                      |

| 参数                | ResNeXt101                      |                       |
| ------------------- | --------------------------- | --------------------------- |
| 资源            | Ascend 310; OS Euler2.8     | GeForce RTX 3090 |
| 上传日期        | 06/22/2021 (month/day/year) | 2022-6-30 |
| MindSpore版本   | 1.2.0                       | 1.7.0                  |
| 数据集             | ImageNet                    | ImageNet            |
| batch_size          | 1                           | 1                          |
| 输出             | 概率                    | 概率                  |
| 准确率            | TOP1: 79.85%, TOP5: 94.71%  | 78.04%(TOP1) |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
