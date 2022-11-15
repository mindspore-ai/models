# 目录

<!-- TOC -->

- [目录](#目录)
- [PNASNet概述](#PNASNet概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PNASNet概述

[论文](https://arxiv.org/abs/1712.00559v3): Chenxi Liu, etc. Progressive Neural Architecture Search. 2018.

# 模型架构

PNASNet总体网络架构如下：

[链接](https://arxiv.org/abs/1712.00559v3)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2百万张彩色图像
    - 训练集：120G，共1.2百万张图像
    - 测试集：5G，共5万张图像

- 数据集结构：

  ```bash
  └─dataset
    ├─train                              # 训练集目录
    ├─val                                # 验证集目录
  ```

- 数据格式：RGB

    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件：Ascend
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```text
.
└─pnasnet
  ├─README.md
  ├─scripts
    ├─run_standalone_train_for_ascend.sh   # 使用Ascend平台启动单机训练（单卡）
    ├─run_distribute_train_for_ascend.sh   # 使用Ascend平台启动单机训练（8卡）
    ├─run_standalone_train_for_gpu.sh      # 使用GPU平台启动单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh      # 使用GPU平台启动分布式训练（8卡）
    ├─run_eval_for_ascend                  # 使用Ascend平台进行启动评估
    └─run_eval_for_gpu.sh                  # 使用GPU平台进行启动评估
  ├─src
    ├─ model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       └──moxing_adapter.py                # moxing adapter
    ├─CrossEntropySmooth.py                # 自定义交叉熵损失函数
    ├─dataset.py                           # 数据预处理
    ├─lr_generator.py                      # 学习率生成器
    └─pnasnet_mobile.py                  # 网络定义
  ├─default_config.yaml                    # 参数配置
  ├─export.py                              # 转换检查点
  ├─eval.py                                # 评估网络
  └─train.py                               # 训练网络
```

## 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

```default_config.yaml
'random_seed': 1,                          # 固定随机种子
'rank': 0,                                 # 分布式训练进程序号
'group_size': 1,                           # 分布式训练分组大小
'work_nums': 8,                            # 数据读取人员数
'epoch_size': 600,                         # 总周期数
'keep_checkpoint_max': 5,                 # 保存检查点最大数
'checkpoint_path': './checkpoint/',        # 检查点保存路径
'train_batch_size': 32,                    # 训练输入批次大小
'val_batch_size': 125,                     # 评估输入批次大小
'num_classes': 1000,                       # 数据集类数
'aux_factor': 0.4,                         # 副对数损失系数
'lr_init': 0.04*8,                         # 启动学习率
'lr_decay_rate': 0.97,                     # 学习率衰减率
'num_epoch_per_decay': 2.4,                # 衰减周期数
'weight_decay': 0.00004,                   # 权重衰减
'momentum': 0.9,                           # 动量
'opt_eps': 1.0,                            # epsilon参数
'rmsprop_decay': 0.9,                      # rmsprop衰减
'loss_scale': 1,                           # 损失规模
'cutout': True,                            # 训练时是否要对输入数据进行截断
'coutout_leng': 56,                        # 输入数据的截断长度
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  # Ascend单机训练
  bash run_standalone_train_for_ascend.sh [DEVICE_ID] [DATASET_PATH]
  ```

  ```bash
  # Ascend单机训练示例
  bash run_standalone_train_for_ascend.sh 0 /dataset/train
  ```

- GPU处理器环境运行

  ```bash
  # GPU单机训练
  bash run_standalone_train_for_gpu.sh [DEVICE_ID] [DATASET_PATH]
  ```

  ```bash
  # GPU单机训练示例
  bash run_standalone_train_for_gpu.sh 0 /dataset/train
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  # Ascend分布式训练（8卡）
  bash run_distribute_train_for_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
  ```

  ```bash
  # Ascend分布式训练示例（8卡）
  bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset/train
  ```

- GPU处理器环境运行

  ```bash
  # GPU分布式训练（8卡）
  bash run_distribute_train_for_gpu.sh [DATASET_PATH]
  ```

  ```bash
  # GPU分布式训练示例（8卡）
  bash run_distribute_train_for_gpu.sh /dataset/train
  ```

### 结果

可以在日志中找到检查点文件及结果。

## 评估过程

- Ascend处理器环境运行

  ```bash
  # 评估
  bash run_eval_for_ascend.sh [DATASET_PATH] [CHECKPOINT]
  ```

  ```bash
  # 评估示例
  bash run_eval_for_ascend.sh /dataset/val ./checkpoint/pnasnet-a-mobile-rank0-600_10009.ckpt
  ```

- GPU处理器环境运行

  ```bash
  # 评估
  bash run_eval_for_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
  ```

  ```bash
  # 评估示例
  bash run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/pnasnet-a-mobile-rank0-600_10009.ckpt
  ```

> 训练过程中可以生成检查点。

### 结果

评估结果保存在脚本路径下。路径下的日志中，可以找到如下结果：

- Ascend处理器环境运行

  acc=74.5%(TOP1)

- GPU处理器环境运行

  acc=74.3%(TOP1)

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#contents)

导出mindir模型

```bash
python export.py --device_target [PLATFORM] --checkpoint [CHECKPOINT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

参数CHECKPOINT_FILE为必填项，`PLATFORM` 必须在 ["Ascend", "GPU", "CPU"]中选择。`FILE_FORMAT` 必须在 ["AIR", "ONNX", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` MINDIR模型的路径和文件名。
- `DATASET_NAME` 必须是imagenet2012。
- `DATASET_PATH` 是imagenet2012数据集中val的路径。
- `NEED_PREPROCESS` 可以是 y 或 n。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，可以在acc.log中看到以下精度计算结果。
Top1 acc:  0.74484
Top5 acc:  0.91976

# 模型描述

## 性能

### 训练性能

| 参数           | Ascend 910                    | GPU                           |
| -------------- | ----------------------------- | ----------------------------- |
| 模型           | PNASNet                       | PNASNet                       |
| 资源           | Ascend 910                    | Tesla V100-PCIE               |
| 上传日期       | 11/07/2021 (month/day/year)   | 12/22/2021 (month/day/year)   |
| MindSpore版本  | 1.2.0                         | 1.5.0                         |
| 数据集         | ImageNet                      | ImageNet                      |
| 训练参数       | default_config.yaml           | default_config.yaml           |
| 优化器         | RMSProp                       | RMSProp                       |
| 损失函数       | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| 损失值         | 1.0660                        | 1.9632                        |
| 总时间         | 576 h 8ps                     | 193 h 8ps                     |
| 检查点文件大小 | 97 M(.ckpt file)              | 91M(.ckpt file)               |

### 评估性能

| 参数          | Ascend 910                  | GPU                         |
| ------------- | --------------------------- | --------------------------- |
| 模型          | PNASNet                     | PNASNet                     |
| 资源          | Ascend 910                  | Tesla V100-PCIE             |
| 上传日期      | 11/07/2021 (month/day/year) | 12/22/2021 (month/day/year) |
| MindSpore版本 | 1.2.0                       | 1.5.0                       |
| 数据集        | ImageNet                    | ImageNet                    |
| 批次大小      | 125                         | 125                         |
| 输出          | 概率                        | 概率                        |
| 精确度        | acc=74.5%(TOP1)             | acc=74.3%(TOP1)             |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。