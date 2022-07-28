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

使用的数据集：[imagenet](http://www.image-net.org/)

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件（(Ascend/GPU)
    - 使用Ascen或GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── MobileNetV3
  ├── README_CN.md     # MobileNetV3相关描述
  ├── scripts
  │   ├──run_standalone_train_ascend.sh   # 用于单卡训练的shell脚本
  │   ├──run_standalone_train_gpu.sh
  │   ├──run_distribute_train_ascend.sh   # 用于八卡训练的shell脚本
  │   ├──run_distribute_train_gpu.sh
  │   ├──run_eval_ascend.sh    # 用于评估的shell脚本
  │   └──run_eval_gpu.sh
  ├── src
  │   ├──config.py      # 参数配置
  │   ├──dataset.py     # 创建数据集
  │   ├──loss.py        # 损失函数
  │   ├──lr_generator.py     # 配置学习率
  │   ├──mobilenetV3.py      # MobileNetV3架构
  │   └──monitor.py          # 监控网络损失和其他数据
  ├── argparser.py
  ├── eval.py       # 评估脚本
  ├── export.py     # 模型格式转换脚本
  └── train.py      # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'num_classes': 1000,                       # 数据集类别数
'image_height': 224,                       # 输入图像高度
'image_width': 224,                        # 输入图像宽度
'batch_size': 256,                         # 数据批次大小
'epoch_size': 370,                         # 模型迭代次数
'warmup_epochs': 4,                        # warmup epoch数量
'lr': 0.05,                                # 学习率
'momentum': 0.9,                           # 动量参数
'weight_decay': 4e-5,                      # 权重衰减率
'label_smooth': 0.1,                       # 标签平滑因子
'loss_scale': 1024,                        # loss scale
'save_checkpoint': True,                   # 是否保存ckpt文件
'save_checkpoint_epochs': 1,               # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 5,                  # 保存ckpt文件的最大数量
'save_checkpoint_path': "./checkpoint",    # 保存ckpt文件的路径
'export_file': "mobilenetv3_small",        # export文件
'export_format': "MINDIR",                 # export格式
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

- Ascend

```shell
# 训练示例
  # python:
    # Ascend单卡训练示例:
      python train.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --device_target="Ascend" &> log &

  # shell:
    # Ascend单卡训练示例:
      bash ./scripts/run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
    # Ascend八卡并行训练:
      cd ./scripts/
      bash ./run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

- GPU

```shell
# 训练示例
  # python:
    # Ascend单卡训练示例:
      python train.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --device_target="GPU" &> log &

  # shell:
    # Ascend单卡训练示例:
      bash ./scripts/run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID]
    # Ascend八卡并行训练:
      cd ./scripts/
      bash ./run_distribute_train_gpu.sh [DATASET_PATH] [DEVICE_NUM]
```

### 结果

ckpt文件将存储在 `./checkpoint` 路径下，训练日志将被记录到 `log.txt` 中。训练日志部分示例如下：

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

- Ascend

```shell
# 评估示例
  # python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --checkpoint_path [PATH_CHECKPOINT] --device_target="Ascend" &> eval.log &

  # shell:
      bash ./scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

- GPU

```shell
# 评估示例
  # python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATASET_PATH] --checkpoint_path [PATH_CHECKPOINT] --device_target="GPU" &> eval.log &

  # shell:
      bash ./scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `eval_log.txt` 查看评估结果。

```shell
result: {'Loss': 2.3101649037352554, 'Top_1_Acc': 0.6746546546546547, 'Top_5_Acc': 0.8722122122122122} ckpt= ./checkpoint/model_0/mobilenetV3-370_625.ckpt
```

# 模型说明

## 训练性能

| 参数                        | Ascend                                | GPU Tesla V100 (1 pcs)                                          | GPU Tesla V100 (6 pcs) |
| -------------------------- | ------------------------------------- |-----------------------------------------------------------------|-----------------------------------------------------------------------|
| 模型名称                    | mobilenetV3                          | mobilenetV3 small                                               | mobilenetV3 small                                                     |
| 运行环境                    | HUAWEI CLOUD Modelarts                     | Ubuntu 18.04.6, Tesla V100, CPU 2.70GHz, 32 cores, RAM 258 GB   | Ubuntu 18.04.6, Tesla V100 (6 pcs), CPU 2.70GHz, 32 cores, RAM 258 GB |
| 上传时间                    | 2021-3-25                             | 10/29/2021 (month/day/year)                                     | 10/29/2021 (month/day/year)                                           |
| MindSpore版本              | 1.3.0                                 | 1.5.0                                           | 1.5.0                                                     |
| 数据集                      | imagenet                              | imagenet                              | imagenet                              |
| 训练参数                    | src/config.py                         | epoch_size=370, batch_size=600, **lr=0.005**, other configs: src/config.py | epoch_size=370, batch_size=600, **lr=0.05**, other configs: src/config.py |
| 优化器                      | RMSProp                              | RMSProp                            | RMSProp                            |
| 损失函数                    | CrossEntropyWithLabelSmooth         | CrossEntropyWithLabelSmooth         | CrossEntropyWithLabelSmooth         |
| 精确度 (8p)                 | Top1[67.5%], Top5[87.2%]               | Accuracy: Top1[67.3%], Top5[87.1%]                              | Accuracy: Top1[67.3%], Top5[87.1%] |
| 最终损失                    | 2.31                                  | 2.32                                                            | 2.32                               |
| 速度                       |                                       | 430 ms/step                                                     | 3500 ms/step                       |
| 训练总时间 (8p)             | 16.4h                                    | 80h (1 pcs)                                                     | 117h (6 pcs)                       |
| 参数量 (M)                 | 36M                                   | 36M                                   | 36M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) |

# 随机情况的描述

我们在 `dataset.py` 和 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models)。