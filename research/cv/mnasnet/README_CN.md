# 目录

- [目录](#目录)
- [Mnasnet描述](#Mnasnet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#ckpt模型评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
    - [导出ONNX模型](#导出ONNX模型)
    - [评估过程](#onnx模型评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

# Mnasnet描述

MnasNet是以MobileNet为backbone，在这个基础结构上搜索block架构以替代bottleneck结构。MnasNet搜索出来的网络计算量更少，延时更小，精度更高。（2018年）

[论文](https://arxiv.org/abs/1807.11626)：Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le. MnasNet: Platform-Aware Neural Architecture Search for Mobile 2018.

# 模型架构

Mnasnet总体网络架构如下：

[链接](https://arxiv.org/abs/1807.11626)

# 数据集

使用的数据集：[ImageNet-2012](http://www.image-net.org/)

- 数据集大小: 146G, 1330k 1000类彩色图像
    - 训练: 140G, 1280k张图片
    - 测试: 6G, 50k张图片
- 数据格式：RGB
    - 注：数据在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或者GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```bash
├── mnasnet
  ├── README_CN.md                 # MNasNet架构相关描述
  ├── scripts
  │   ├──run_standalone_train.sh   # 用于Ascend单卡训练的shell脚本
  │   ├──run_distribute_train.sh   # 用于Ascend八卡训练的shell脚本
  │   ├──run_eval.sh               # 用于Ascend评估的shell脚本
  │   ├──run_standalone_train_gpu.sh   # 用于GPU单卡训练的shell脚本
  │   ├──run_distribute_train_gpu.sh   # 用于GPU八卡训练的shell脚本
  │   └──run_eval_gpu.sh               # 用于GPU评估的shell脚本
  ├── src
  │   ├──models                    # MNasNet架构
  │   │   └──mnasnet.py
  │   ├──config.py                 # 参数配置
  │   ├──dataset.py                # 创建数据集
  │   ├──loss.py                   # 损失函数
  │   ├──lr_generator.py           # 配置学习率
  │   └──Monitor.py                # 监控网络损失和其他数据
  ├── eval.py                      # 评估脚本
  ├── eval_onnx.py                 # ONNX模型评估脚本
  ├── export.py                    # 模型格式转换脚本
  └── train.py                     # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'class_num': 1000,                        # 数据集类别数
'batch_size': 256,                        # 数据批次大小
'loss_scale': 1024,                       # loss scale
'momentum': 0.9,                          # 动量参数
'weight_decay': 1e-5,                     # 权重衰减率
'epoch_size': 250,                        # 模型迭代次数: Ascend:250, GPU:300
'save_checkpoint': True,                  # 是否保存ckpt文件
'save_checkpoint_epochs': 1,              # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 5,                 # 保存ckpt文件的最大数量
'save_checkpoint_path': "./checkpoint",   # 保存ckpt文件的路径
'opt': 'rmsprop',                         # 优化器
'opt_eps': 0.001,                         # 改善数值稳定性的优化器参数
'warmup_epochs': 5,                       # warmup epoch数量
'lr_decay_mode': 'other',                 # 学习率下降方式
'use_label_smooth': True,                 # 是否使用label smooth
'label_smooth_factor': 0.1,               # 标签平滑因子
'lr_init': 0.0001,                        # 初始学习率
'lr_max': 0.2,                            # 最大学习率
'lr_end': 0.00001,                        # 最终学习率
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
  python:
      Ascend单卡训练示例：python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]

      GPU单卡训练示例：python train.py --dataset_path [DATA_DIR] --device_target="GPU"

  shell:
      Ascend单卡训练示例: bash ./scripts/run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
      Ascend八卡并行训练:
          cd ./scripts/
          bash ./run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]

      GPU单卡训练示例: bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
      GPU八卡并行训练:
          cd ./scripts/
          bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)
```

### 结果

ckpt文件将存储在 `./checkpoint` 路径下，训练日志将被记录到 `log.txt` 中。训练日志部分示例如下：

```shell
epoch: [  0/250], step:[  624/  625], loss:[5.583/5.583], time:[744951.372], lr:[0.040]
epoch time: 765959.118, per step time: 1225.535, avg loss: 5.583
epoch: [  1/250], step:[  624/  625], loss:[4.800/4.800], time:[294060.216], lr:[0.080]
epoch time: 295243.615, per step time: 472.390, avg loss: 4.800
epoch: [  2/250], step:[  624/  625], loss:[4.312/4.312], time:[292620.547], lr:[0.120]
epoch time: 293728.792, per step time: 469.966, avg loss: 4.312
epoch: [  3/250], step:[  624/  625], loss:[3.988/3.988], time:[297809.817], lr:[0.160]
epoch time: 299011.203, per step time: 478.418, avg loss: 3.988
epoch: [  4/250], step:[  624/  625], loss:[3.975/3.975], time:[297641.585], lr:[0.200]
epoch time: 298663.563, per step time: 477.862, avg loss: 3.975
epoch: [  5/250], step:[  624/  625], loss:[3.799/3.799], time:[299268.819], lr:[0.199]
epoch time: 300428.474, per step time: 480.686, avg loss: 3.799
epoch: [  6/250], step:[  624/  625], loss:[3.689/3.689], time:[300148.012], lr:[0.198]
epoch time: 301278.408, per step time: 482.045, avg loss: 3.689
```

## ckpt模型评估过程

### 启动

您可以使用python或shell脚本进行评估。

```shell
# 评估示例
  python:
      Ascend评估：python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]
      GPU评估：python ./eval.py --checkpoint_path [PATH_CHECKPOINT] --dataset_path [DATA_DIR] --device_target="GPU"

  shell:
      Ascend评估：bash ./scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
      GPU评估：bash scripts/run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `eval.log` 查看评估结果。

```shell
result: {'Loss': 2.0364865480325163, 'Top_1_Acc': 0.7412459935897436, 'Top_5_Acc': 0.9159655448717948} ckpt= /disk2/mnas3/model_0/Mnasnet-rank0-250_625.ckpt
```

## 导出ONNX模型

```shell
python export.py --file_format onnx --checkpoint_path [PATH_CHECKPOINT] --file_name [FILE_NAME] --device_target GPU
```

## onnx模型评估过程

### 启动

您可以使用python或shell脚本进行评估。

```shell
# 评估示例
  python:
      Ascend评估：python eval_onnx.py --dataset_path [DATA_DIR] --onnx_url [ONNX_PATH] --device_target="Ascend"
      GPU评估：python ./eval_onnx.py --dataset_path [DATA_DIR] --onnx_url [ONNX_PATH]
  shell:
      Ascend评估：bash ./scripts/run_eval_onnx.sh [DATA_DIR] [ONNX_PATH]
      GPU评估：bash scripts/run_eval_gpu_onnx.sh [DATA_DIR] [ONNX_PATH]

```

### 结果

可以在 `eval.log` 查看评估结果。

```shell
Top1-Acc: 0.7398
Top5-Acc: 0.9169

```

# 模型说明

## 训练性能

| 参数                        | Ascend                                |       GPU    |
| -------------------------- | ------------------------------------- | ------------------------------------ |
| 模型名称                    | MNasNet                          |   MNasNet                                |
| 运行环境                    | Ascend 910；CPU 2.60GHz，192核；内存：755G | Tesla V100-PCIE 32G; CPU 2.30GHz 56核；内存 256GB  |
| 上传时间                    | 2021-6-11                             |    2021-7-30                        |
| MindSpore 版本             | 1.2.0                                 |     1.3.0                            |
| 数据集                      | imagenet                              |  imagenet                           |
| 训练参数                    | src/config.py                         |  src/config.py, 其中epoch_size为300 |
| 优化器                      | RMSProp                              |   RMSProp                 |
| 损失函数                    | CrossEntropySmooth                   |   CrossEntropySmooth      |
| 最终损失                    | 2.099                                |   2.023                   |
| 精确度 (8p)                 | Top1[74.1%], Top5[91.6%]               |  Top1[74.3%], Top5[91.8%]  |
| 训练总时间 (8p)             | 20.8h                                    |  48h  |
| 评估总时间                  | 1min                                    |  1min   |
| 参数量 (M)                 | 61M                                   |     61M
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/r2.0/research/cv/mnasnet) | [链接](https://gitee.com/mindspore/models/tree/r2.0/research/cv/mnasnet) |

# 随机情况的描述

我们在 `dataset.py` 和 `train.py` 脚本中设置了随机种子。

# ModelZoo

  │   └──run_eval.sh                   # 用于Ascend评估的shell脚本
  │   └──run_eval_gpu.sh               # 用于GPU评估的shell脚本
请核对官方 [主页](https://gitee.com/mindspore/models)。
