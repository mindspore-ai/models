# 目录

- [目录](#目录)
- [MobileNetV3描述](#mobilenetv3描述)
- [模型架构](#模型架构)
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

# MobileNetV3描述

MobileNetV3结合硬件感知神经网络架构搜索（NAS）和NetAdapt算法，已经可以移植到手机CPU上运行，后续随新架构进一步优化改进。（2019年11月20日）

[论文](https://arxiv.org/pdf/1905.02244)：Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

# 模型架构

MobileNetV3总体网络架构如下：

[链接](https://arxiv.org/pdf/1905.02244)

# 数据集

使用的数据集：[ImageNet(ILSVRC2012)](http://www.image-net.org/)

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

## 脚本和示例代码

```bash
├── mobileNetv3_large
  ├── README_CN.md                 # MobileNetV3相关描述
  ├── scripts
  │   ├──run_standalone_train.sh   # 用于单卡训练的shell脚本
  │   ├──run_distribute_train.sh   # 用于八卡训练的shell脚本
  │   └──run_eval.sh               # 用于评估的shell脚本
  ├── src
  │   ├──config.py                 # 参数配置
  │   ├──dataset.py                # 创建数据集
  │   ├──lr_generator.py           # 配置学习率
  │   ├──mobilenetV3.py            # MobileNetV3架构
  ├── eval.py                      # 评估脚本
  ├── export.py                    # 模型格式转换脚本
  └── train.py                     # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
    "num_classes": 1000,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 150,
    "epoch_size": 370,
    "warmup_epochs": 4,
    "lr_init": 0.0,
    "lr_end": 0.00,
    "lr": 1.54,
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "weight_init": "he_uniform",
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 50,
    "save_checkpoint_path": "./",
    "export_format": "MINDIR",
    "export_file": "mobilenetv3_large",
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
  python:
      Ascend单卡训练示例: python train.py --train_dataset_path [TRAIN_DATA_DIR] --eval_dataset_path [EVAL_DATA_DIR] --device_id [DEVICE_ID] --run_distribute False

  shell:
      Ascend单卡训练示例: bash ./scripts/run_standalone_train.sh [DEVICE_ID] [TRAIN_DATA_DIR] [EVAL_DATA_DIR]
      Ascend八卡并行训练:
          bash ./run_distribute_train.sh [RANK_TABLE_FILE] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [TRAIN_DATA_DIR] [EVAL_DATA_DIR]
```

### 结果

ckpt文件将存储在 `./ckpts_rank_0` 路径下，训练日志将被记录到 `log.txt` 中。训练日志部分示例如下：

```shell
epoch 1: epoch time: 553262.126, per step time: 518.521, avg loss: 5.270
epoch 2: epoch time: 151033.049, per step time: 141.549, avg loss: 4.529
epoch 3: epoch time: 150605.300, per step time: 141.148, avg loss: 4.101
epoch 4: epoch time: 150638.805, per step time: 141.180, avg loss: 4.014
epoch 5: epoch time: 150594.088, per step time: 141.138, avg loss: 3.607
```

## 评估过程

### 启动

您可以使用python或shell脚本进行评估。

```shell
# 评估示例
  python:
      python eval.py --device_id [DEVICE_ID] --checkpoint_path [PATH_CHECKPOINT] --dataset_path [DATA_DIR]
  shell:
      bash ./scripts/run_eval.sh [DEVICE_ID] [PATH_CHECKPOINT] [EVAL_DATA_DIR]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `evaluation_ascend/eval.log` 查看评估结果。

```shell
metric:  {'Loss': 6.346325377444248, 'Top1-Acc': 0.744024024024024, 'Top5-Acc': 0.916956956956957}
```

# 模型说明

## 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | mobilenetv3_large                          |
| 模型版本                    | v1.0                        |
| 运行环境                    | HUAWEI CLOUD Modelarts                     |
| Mindspore版本              | 1.3.0                                |
| 上传时间                    | 2021-9-17                           |
| 数据集                      | imagenet                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | Momentum                            |
| 损失函数                    | CrossEntropyWithLabelSmooth         |
| 最终损失                    | 2.01                                 |
| 精确度 (8p)                 | Top1[74.4%], Top5[91.7%]               |
| 训练总时间 (8p)             | 40h                                    |
| 评估总时间                  | 2min                                    |
| 参数量 (M)                 | 43.4M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetv3_large) |

# 随机情况的描述

我们在 `dataset.py` 和 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models)。