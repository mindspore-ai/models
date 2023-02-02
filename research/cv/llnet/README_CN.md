# 目录

<!-- TOC -->

- [目录](#目录)
- [LLNet概述](#LLNet概述)
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

# LLNet概述

作者提出了一种基于深度自编码器的方法，可以从低光图像中识别信号特征，在高动态范围内自适应地使图像变亮，而不会过度放大图像中较亮的部分（即图像像素不饱和）。该网络可以成功应用于自然弱光环境和/或硬件老化所产生的带噪声的图像。

[论文](https://arxiv.org/abs/1511.03995v3): Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar: LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement. 2015.

# 模型架构

LLNet总体网络架构如下：

[链接](https://arxiv.org/abs/1511.03995v3)

# 数据集

使用的数据集：[dbimagenes](http://decsai.ugr.es/cvg/dbimagenes/)

- 数据集大小：1.1G，170张灰度图像
        - 训练集：526.3M，163张灰度图像，每张图片随机产生1250个17*17像素增加噪声和暗化后的图块
        - 验证集：526.3M，163张灰度图像，每张图片随机产生1250个17*17像素增加噪声和暗化后的图块
        - 测试集：1.3M，5张灰度图像， 每张图片29241 个17*17像素图块，随机增加噪声和暗化，然后进行恢复。
- 数据格式：灰度图像
        * 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件：Ascend
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```text
.
└─llnet
  ├─ README.md
  ├─ README_CN.md
  ├─── ascend310_infer
  │    ├── build.sh                            # 编译可执行文件的脚本
  │    ├── CMakeLists.txt                      # CMakeLists
  │    ├── inc
  │    │   └── utils.h                         # ascend310_infer的头文件
  │    └── src
  │        ├── main.cc                         # ascend310_infer的主函数
  │        └── utils.cc                        # ascend310_infer的辅助函数
  ├─── scripts
  │    ├─ run_standalone_train.sh              # 使用GPU、Ascend 910平台启动单机训练（单卡）
  │    ├─ run_distribute_train_for_ascend.sh   # 使用Ascend 910平台启动单机多卡训练（8卡）
  │    ├─ run_infer_310.sh                     # 使用Ascend 310平台进行验证
  │    └─ run_eval_for_gpu.sh                  # 模型CPU、GPU、Ascend 910验证
  ├─ src
  │    ├─ model_utils
  │    │  ├─ config.py                         # 处理参数配置
  │    │  ├─ device_adapter.py                 # ModelArts的设备适配器
  │    │  ├─ local_adapter.py                  # 本地适配器
  │    │  └─ moxing_adapter.py                 # ModelArts的模型适配器
  │    ├─ dataset.py                           # 数据读取
  │    ├─ lr_generator.py                      # 学习率生成器
  │    └─ llnet.py                             # 网络定义
  ├─ default_config.yaml                       # 默认参数设置
  ├─ eval.py                                   # 评估网络
  ├─ export.py                                 # 模型导出，用于ascend310_infer推理
  ├─ postprocess.py                            # ascend310_infer的后处理
  ├─ preprocess.py                             # ascend310_infer的预处理
  ├─ requirements.txt                          # 需要的pyaml包
  ├─ test.py                                   # 测试网络
  ├─ train.py                                  # 训练网络
  └─ write_mindrecords.py                      # 生成mindrecord格式数据集，用于训练、评估、测试。

```

## 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

```text
'random_seed': 1,                              # 固定随机种子
'rank': 0,                                     # 分布式训练卡号
'group_size': 1,                               # 分布式训练分组大小
'work_nums': 8,                                # 数据读取线程数
'pretrain_epoch_size': 5,                      # 预训练轮数
'finetrain_epoch_size': 300,                   # 精训轮数
'keep_checkpoint_max': 20,                     # 保存检查点最大数
'save_ckpt_path': './',                        # 检查点保存路径
'train_batch_size': 500,                       # 训练输入批次大小
'val_batch_size': 1250,                        # 评估输入批次大小
'lr_init': [0.01, 0.01, 0.001, 0.001],         # 初始学习率，前面3个用于预训练前面3层，最后1个用于精训。
'weight_decay': 0.0,                           # 权重衰减
'momentum': 0.9,                               # 动量
```

## 训练过程

### 用法

```bash
# Ascend分布式训练（8卡）
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
# Ascend单机训练
bash run_standalone_train.sh [DEVICE_ID] [DATASET_PATH]
```

### 运行

```bash
# Ascend分布式训练示例（8卡）
bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset
# Ascend单机训练示例
bash run_standalone_train.sh 0 /dataset
```

### 结果

可以在日志中找到检查点文件及结果。

## 评估过程

### 用法

```bash
# 评估
bash run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
```

### 启动

```bash
# 检查点评估
bash run_eval.sh 5 ../dataset ./ckpt_5/llnet-rank5-286_408.ckpt
```

### 结果

评估结果保存在脚本路径下。路径下的日志中，可以找到如下结果：
PSNR=21.593(dB) SSIM=0.617

## 推理过程

### [导出MindIR](#contents)

导出mindir模型

```python
python export.py --device_target [PLATFORM] --device_id [DEVICE_ID] --checkpoint [CHECKPOINT_FILE] --file_format [FILE_FORMAT] --file_name [FILE_NAME]
```

参数CHECKPOINT_FILE为必填项，
`PLATFORM` 必须在 ["Ascend", "GPU", "CPU"]中选择，缺省为Ascend。
`DEVICE_ID` 必须在 [0-7],缺省为0
`FILE_FORMAT` 必须在 ["AIR", "ONNX", "MINDIR"]中选择，缺省为MINDIR。
`FILE_NAME` 导出模型文件的基本名，缺省为llnet。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` MINDIR模型的路径和文件名。
- `DATASET_PATH` 是数据集的路径。
- `NEED_PREPROCESS` 可以是 y 或 n。第1次运行本脚本时需要为y。
- `DEVICE_ID`  可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。
PSNR:  21.582 (dB)
SSIM:   0.604

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend 910                    |
| -------------------------- | ----------------------------- |
| 模型                       | LLNet                         |
| 资源                       | Ascend 910                    |
| 上传日期                   | 07/23/2022 (month/day/year)   |
| MindSpore版本              | 1.5.1                         |
| 数据集                     | dbimagenes                    |
| 训练参数                   | default_config.yaml           |
| 优化器                     | Adam                          |
| 损失函数                   | MSE                           |
| 损失值                     | 0.0105                        |
| 总时间                     | 0 小时 17 分 21 秒 2卡        |
| 检查点文件大小             | 48.5 M(.ckpt file)            |

### 评估性能

| 参数                       | Ascend 910                    |
| -------------------------- | ----------------------------- |
| 模型                       | LLNet                         |
| 资源                       | Ascend 910                    |
| 上传日期                   | 07/23/2022 (month/day/year)   |
| MindSpore版本              | 1.5.1                         |
| 数据集                     | dbimagenes                    |
| 批次大小                   | 1250                          |
| 输出                       | 289个恢复后的像素             |
| 精确度                     | PSNR = 21.593  SSIM = 0.617   |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
