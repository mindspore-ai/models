# 目录

<!-- TOC -->

- [目录](#目录)
- [NASNet概述](#NASNet概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# NASNet概述

[论文](https://arxiv.org/abs/1707.07012): Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. 2017.

# 模型架构

NASNet总体网络架构如下：

[链接](https://arxiv.org/abs/1707.07012v4)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、1.2百万张彩色图像
        - 训练集：120G，共1.2百万张图像
        - 测试集：5G，共5万张图像
- 数据格式：RGB
        * 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件：Ascend/GPU
    - 使用Ascend处理器来搭建硬件环境。
    - 使用GPU处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```text
.
└─nasnet
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train_for_ascend.sh   # 使用Ascend平台启动单机训练（单卡）
    ├─run_distribute_train_for_ascend.sh   # 使用Ascend平台启动分布式训练（8卡）
    ├─run_standalone_train_for_gpu.sh      # 使用GPU平台启动单机训练（单卡）
    ├─run_distribute_train_for_gpu.sh      # 使用GPU平台启动分布式训练（8卡）
    └─run_eval_for_ascend                  # 使用Ascend平台进行启动评估
    └─run_eval_for_gpu.sh                  # 使用GPU平台进行启动评估
  ├─src
    ├─config.py                            # 参数配置
    ├─dataset.py                           # 数据预处理
    ├─loss.py                              # 自定义交叉熵损失函数
    ├─lr_generator.py                      # 学习率生成器
├─nasnet_a_mobile.py                       # 网络定义
├─eval.py                                  # 评估网络
├─export.py                                # 转换检查点
└─train.py                                 # 训练网络
```

## 脚本参数

在config.py中可以同时Ascend和GPU配置训练参数和评估参数。

```python
'random_seed':1,                # 固定随机种子
'rank':0,                       # 分布式训练进程序号
'group_size':1,                 # 分布式训练分组大小
'work_nums':8,                  # 数据读取人员数
'epoch_size':600,               # 总周期数
'keep_checkpoint_max':30,       # 保存检查点最大数
'ckpt_path':'./',               # 检查点保存路径
'is_save_on_master':1           # 在rank0上保存检查点，分布式参数
'train_batch_size':32,          # 训练时输入批次大小
'val_batch_size':32,            # 评估时输入批次大小
'image_size' : 224,             # 图片大小为224*224
'num_classes':1000,             # 数据集类数
'label_smooth_factor':0.1,      # 标签平滑因子
'aux_factor':0.4,               # 副对数损失系数
'lr_init':0.04*8,               # 启动学习率
'lr_decay_rate':0.97,           # 学习率衰减率
'num_epoch_per_decay':2.4,      # 衰减周期数
'weight_decay':0.00004,         # 重量衰减
'momentum':0.9,                 # 动量
'opt_eps':1.0,                  # epsilon参数
'rmsprop_decay':0.9,            # rmsprop衰减
'loss_scale':1,                 # 损失规模
'cutout': True,                 # 训练时是否对输入数据截断
'coutout_length': 56,           # 当cutout=True时，训练数据的截断长度
```

```python
'random_seed':1,                # 固定随机种子
'rank':0,                       # 分布式训练进程序号
'group_size':1,                 # 分布式训练分组大小
'work_nums':8,                  # 数据读取人员数
'epoch_size':600,               # 总周期数
'keep_checkpoint_max':100,      # 保存检查点最大数
'ckpt_path':'./checkpoint/',    # 检查点保存路径
'is_save_on_master':1           # 在rank0上保存检查点，分布式参数
'train_batch_size':32,          # 训练时输入批次大小
'val_batch_size':32,            # 评估时输入批次大小
'image_size' : 224,             # 图片大小为224*224
'num_classes':1000,             # 数据集类数
'label_smooth_factor':0.1,      # 标签平滑因子
'aux_factor':0.4,               # 副对数损失系数
'lr_init':0.04*8,               # 启动学习率
'lr_decay_rate':0.97,           # 学习率衰减率
'num_epoch_per_decay':2.4,      # 衰减周期数
'weight_decay':0.00004,         # 重量衰减
'momentum':0.9,                 # 动量
'opt_eps':1.0,                  # epsilon参数
'rmsprop_decay':0.9,            # rmsprop衰减
'loss_scale':1,                 # 损失规模
'cutout': False,                # 训练时是否对输入数据截断
'coutout_length': 56,           # 当cutout=True时，训练数据的截断长度
```

## 训练过程

### 用法

```bash
# 分布式训练示例（8卡）
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
bash run_distribute_train_for_gpu.sh [DATASET_PATH]
# 单机训练
bash run_standalone_train_for_ascend.sh [DEVICE_ID] [DATASET_PATH]
bash run_standalone_train_for_gpu.sh [DEVICE_ID] [DATASET_PATH]
```

### 运行

```bash
# 分布式训练示例（8卡）
bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset
bash /run_distribute_train_for_gpu.sh /dataset
# 单机训练示例
bash run_standalone_train_for_ascend.sh 0 /dataset
bash run_standalone_train_for_gpu.sh 0 /dataset
```

### 结果

可以在日志中找到检查点文件及结果。

## 评估过程

### 用法

```bash
# 评估
bash run_eval_for_ascend.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
bash run_eval_for_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
```

### 启动

```bash
# 检查点评估
bash run_eval_for_ascend.sh 0 /dataset ./ckpt_0/nasnet-a-mobile-rank0-248_10009.ckpt
bash run_eval_for_gpu.sh 0 /dataset ./ckpt_0/nasnet-a-mobile-rank0-248_10009.ckpt
```

> 训练过程中可以生成检查点。

### 结果

评估结果保存在./eval路径下。路径下的日志eval.log中，可以找到如下结果：
acc=74.39%(TOP1,Ascend)
acc=73.5%(TOP1,GPU)

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#contents)

导出mindir模型

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

参数CHECKPOINT_FILE为必填项，
`PLATFORM` 必须在 ["Ascend", "GPU", "CPU"]中选择。
`FILE_FORMAT` 必须在 ["AIR", "ONNX", "MINDIR"]中选择。

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
- `DEVICE_ID`  可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。
Top1 acc: 0.74376
Top5 acc: 0.91598

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend 910                    | GPU                           |
| -------------------------- | ----------------------------- |-------------------------------|
| 模型                       | NASNet                        | NASNet                        |
| 资源                       | Ascend 910                    | NV SMX2 V100-32G              |
| 上传日期                   | 2021-11-01                    | 2020-09-24                    |
| MindSpore版本              | 1.3.0                         | 1.0.0                         |
| 数据集                     | ImageNet                      | ImageNet                      |
| 训练参数                   | src/config.py                 | src/config.py                 |
| 优化器                     | RMSProp                       | RMSProp                       |
| 损失函数                   | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| 损失值                     | 1.9617                        | 1.8965                        |
| 总时间                     | 8卡运行约403个小时            | 8卡运行约144个小时            |
| 检查点文件大小             | 89 M(.ckpt文件)               | 89 M(.ckpt文件)               |

### 评估性能

| 参数                       | Ascend 910                    | GPU                           |
| -------------------------- | ----------------------------- |-------------------------------|
| 模型                       | NASNet                        | NASNet                        |
| 资源                       | Ascend 910                    | NV SMX2 V100-32G              |
| 上传日期                   | 2021-11-01                    | 2020-09-24                    |
| MindSpore版本              | 1.3.0                         | 1.0.0                         |
| 数据集                     | ImageNet                      | ImageNet                      |
| batch_size                 | 32                            | 32                            |
| 输出                       | 概率                          | 概率                          |
| 精确度                     | acc=74.39%(TOP1)              | acc=73.5%(TOP1)               |

# ModelZoo主页

注意：此模型将在r1.8版本移动到`/models/research/`目录下。

请浏览官网[主页](https://gitee.com/mindspore/models)。
