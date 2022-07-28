## 目录

- [目录](#目录)
- [网络描述](#网络描述)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [脚本参数](#脚本参数)
- [脚本使用](#脚本使用)
    - [训练脚本用法](#训练脚本用法)
    - [评估脚本用法](#评估脚本用法)
    - [导出脚本用法](#导出脚本用法)
- [模型描述](#模型描述)
- [随机情况说明](#随机情况说明)
- [官方主页](#官方主页)

## 网络描述

### 概述

高分辨率数字图像通常被缩小以适应各种显示屏幕或节省存储和带宽成本，同时采用后放大来恢复原始分辨率或放大图像中的细节。

然而，典型的图像降尺度由于高频信息的丢失是一种非注入映射，这导致逆升尺度过程的不适定问题，并对从降尺度的低分辨率图像中恢复细节提出了巨大的挑战。简单地使用图像超分辨率方法进行放大会导致无法令人满意的恢复性能。

可逆重缩放网络 (IRN)从新的角度对缩小和放大过程进行建模来解决这个问题，即可逆双射变换，在缩小过程中使用遵循特定分布的潜在变量来捕获丢失信息的分布，这可以在很大程度上减轻图像放大的不适定性质,以生成视觉上令人愉悦的低分辨率图像。通过这种方式，通过网络将随机绘制的潜在变量与低分辨率图像反向传递，从而使放大变得易于处理。

本示例主要针对IRN提出的深度神经网络架构以及训练过程进行了实现。

### 论文

Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He, Guolin Ke, Jiang Bian, Zhouchen Lin, and Tie-Yan Liu. 2020. Invertible Image Rescaling. In European Conference on Computer Vision (ECCV).

## 模型架构

![1](./figures/architecture.png)

## 数据集

本示例使用[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，其目录结构如下：

```bash
data/
    ├── DIV2K_train_HR/                 # 训练集高分辨率数据
    └── DIV2K_valid_HR/                 # 测试集高分辨率数据
```

## 环境要求

- 硬件
    - Ascend处理器
    - GPU
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

完成计算设备和框架环境的准备后，开发者可以运行如下指令对本示例进行训练和评估。

- GPU环境运行

```bash
# 单卡训练
# 用法：bash run_standalone_train_gpu.sh [SCALE] [DATASET_GT_PATH]
bash run_standalone_train_gpu.sh 4 /home/nonroot/IRN/data/DIV2K_train_HR

# 分布式训练
# 用法：bash run_distribute_train_gpu.sh [DEVICE_NUM] [SCALE] [DATASET_PATH]
# 样例：DEVICE_NUM等于2、4、8，分别对应2、4、8卡分布式
bash run_distribute_train_gpu.sh 8 4 /home/nonroot/IRN/data/DIV2K_train_HR

# 单卡评估
# 用法：bash run_eval_gpu.sh [SCALE] [DATASET_PATH] [CHECKPOINT_PATH]
bash run_eval_gpu.sh 4 /home/nonroot/IRN/data/DIV2K_valid_HR /home/nonroot/IRN/ckpt/latest.ckpt

```

- Ascend处理器环境运行

```bash
# 单卡训练
# 用法：bash run_standalone_train_ascend.sh [SCALE] [DATASET_GT_PATH]
bash run_standalone_train_ascend.sh 4 /home/nonroot/IRN/data/DIV2K_train_HR

# 8卡分布式训练
# 用法：bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [SCALE] [DATASET_PATH]
bash run_distribute_train_ascend.sh rank_table_file.json 4 /home/nonroot/IRN/data/DIV2K_train_HR

# 单卡评估
# 用法：bash run_eval_ascend.sh [SCALE] [DATASET_PATH] [CHECKPOINT_PATH]
bash run_eval_ascend.sh 4 /home/nonroot/IRN/data/DIV2K_valid_HR /home/nonroot/IRN/ckpt/latest.ckpt

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，请参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

## 脚本说明

```bash
.
├── README.md                               # 说明文档
├── scripts
│   ├── run_distribute_train_ascend.sh      # Ascend处理器环境多卡训练脚本
│   ├── run_distribute_train_gpu.sh         # GPU处理器环境多卡训练脚本
│   ├── run_eval_ascend.sh                  # Ascend处理器环境评估脚本
│   ├── run_eval_gpu.sh                     # GPU处理器环境评估脚本
│   ├── run_standalone_train_ascend.sh      # Ascend处理器环境单卡训练脚本
│   └── run_standalone_train_gpu.sh         # GPU处理器环境单卡训练脚本
├── src
│   ├── data
│   │   ├── dataset.py                      # 数据集处理
│   │   └── util.py                         # 数据集读取图片缩放等
│   ├── network
│   │   ├── Invnet.py                       # IRN网络定义
│   │   ├── net_with_loss.py                # 自定义loss
│   │   └── util.py                         # 网络初始化等
│   ├── optim
│   │   ├── adam_clip.py                    # 梯度裁剪
│   │   ├── warmup_cosine_annealing_lr.py   # 余弦退火学习率算法
│   │   └── warmup_multisteplr.py           # 多步学习率算法
│   ├── options
│   │   ├── options.py                      # 配置文件读取
│   │   ├── test
│   │   │   ├── test_IRN_x2.yml             # 2倍缩放测试配置文件
│   │   │   └── test_IRN_x4.yml             # 4倍缩放测试配置文件
│   │   └── train
│   │       ├── train_IRN_x2.yml            # 2倍缩放训练配置文件
│   │       └── train_IRN_x4.yml            # 4倍缩放训练配置文件
│   └── utils
│       └── util.py                         # 评价指标计算
├── train.py                                # 训练网络
├── export.py                               # 导出网络
├── requirements.txt                        # 环境需求文件
└── val.py                                  # 测试网络
```

## 脚本参数

在[/src/options/train/train_IRN_x4.yml](./src/options/train/train_IRN_x4.yml)中可以配置训练参数、数据集路径等参数。

```python
datasets:
  train:
    name: DIV2K
    mode: LQGT
    batch_size: 8

#### network structures
network_G:
  which_model_G:
      subnet_type: DBNet
  in_nc: 3
  out_nc: 3
  block_num: [8, 8]
  scale: 4
  init: xavier

#### training settings: learning rate scheme, loss
train:
  lr_G: !!float 2e-4
  beta1: 0.9
  beta2: 0.999
  epochs: 5001
  warmup_iter: -1  # no warm up

  lr_scheme: MultiStepLR
  lr_steps: [100000, 200000, 300000, 400000]
  lr_gamma: 0.5


  pixel_criterion_forw: l2
  pixel_criterion_back: l1

  manual_seed: 10

  val_freq: !!float 5e3

  lambda_fit_forw: 16.
  lambda_rec_back: 1
  lambda_ce_forw: 1
  weight_decay_G: !!float 1e-5
  gradient_clipping: 10

```

## 脚本使用

### 训练脚本用法

```bash
# python train.py -h
usage: train.py [--scale {2,4}] [--dataset_GT_path {path of intended GT dataset}] [--dataset_LQ_path {path of intended LQ dataset}] [--resume_state {path of the checkpoint}] [--device_target {Ascend,GPU,CPU}] [--device_num DEVICE_NUM] [--run_distribute RUN_DISTRIBUTE]


IRN for image rescaling.

optional arguments:
  -h, --help            show this help message and exit
  --scale {2,4}
                        Rescaling parameter
  --dataset_GT_path
                        Path of intended GT dataset
  --dataset_LQ_path None
                        Path of intended LQ dataset
  --resume_state None
                        Path of the checkpoint
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
  --device_num DEVICE_NUM
                        The number of device(s) to be used for training.
  --run_distribute RUN_DISTRIBUTE
                        Whether to train the model in distributed mode or not.
```

### 评估脚本用法

对训练好的模型进行精度评估：

```bash
# python test.py -h
usage: eval.py  [--scale {2,4}] [--dataset_GT_path {path of intended GT dataset}] [--dataset_LQ_path {path of intended LQ dataset}] [--resume_state {path of the checkpoint}] [--device_target {Ascend,GPU,CPU}]

IRN for image rescaling.

optional arguments:
  -h, --help            Show this help message and exit
  --scale {2,4}
                        Rescaling parameter
  --dataset_GT_path
                        Path of intended GT dataset
  --dataset_LQ_path None
                        Path of intended LQ dataset
  --resume_state
                        Path of the checkpoint
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
```

### 导出脚本用法

将训练好的模型导出为AIR、ONNX或MINDIR格式：

```bash
# python export.py -h
usage: export.py [-h] [--scale {2,4}] [--device_id DEVICE_ID] --checkpoint_path
                 CHECKPOINT_PATH [--file_name FILE_NAME]
                 [--file_format {AIR,ONNX,MINDIR}]
                 [--device_target {Ascend,GPU,CPU}]

IRN with AutoAugment export.

optional arguments:
  -h, --help            Show this help message and exit
  --scale {2,4}
                        ResIcaling parameter
  --device_id DEVICE_ID
                        Device id.
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path.
  --file_name FILE_NAME
                        Output file name.
  --file_format {AIR,ONNX,MINDIR}
                        Export format.
  --device_target {Ascend,GPU,CPU}
                        Device target.

```

## 模型描述

| 参数          | 单卡GPU                 | 4卡GPU                  | 单卡Ascend 910         | 8卡Ascend 910          |
| :------------ | :---------------------- | ----------------------- | :--------------------- | ---------------------- |
| 资源          | NVIDIA V100 | NVIDIA GeForce RTX 3090 | Ascend 910             | Ascend 910             | V |
| 上传日期      | 2022.6.13               | 2022.6.13               | 2021.09.25             | 2021.11.01             |
| MindSpore版本 | 1.6.1                   | 1.6.1                   | 1.3.0                  | 1.3.0                  |
| 训练数据集    | DIV2K                   | DIV2K                   | DIV2K                  | DIV2K                  |
| 优化器        | Adam                    | Adam                    | Adam                   | Adam                   |
| 输出          | Reconstructed HR image  | Reconstructed HR image  | Reconstructed HR image | Reconstructed HR image |
| PSNR          | NaN                     | 34.53                   | 34.11                  | 33.88                  |
| SSIM          | NaN                     | 0.9246                  | 0.9206                 | 0.9167                 |
| 速度          | 836ms/step              | 1417 ms/step            | 271 ms/step            | 409 ms/step            |
| 总时长        | NaN                     | 2952mins                | 2258 mins              | 409 mins               |
| 微调检查点    | 50.1M（.ckpt文件）      | 50.1M（.ckpt文件)       | 50.1M（.ckpt文件)      | 50.1M（.ckpt文件)      |
| 脚本          | [IRN](./)               | [IRN](./)               | [IRN](./)              | [IRN](./)              |

## 随机情况说明

[train.py](./train.py)中设置了随机种子，以确保训练的可复现性。

## 官方主页

请浏览官网[主页](https://gitee.com/mindspore/models)。