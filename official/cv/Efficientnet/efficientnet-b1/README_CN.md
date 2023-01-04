# 目录

- [目录](#目录)
- [EfficientNet-B1描述](#efficientnet-b1描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [推理过程](#推理过程)
    - [导出MINDIR](#导出mindir)
    - [执行推理](#执行推理)
    - [导出ONNX](#导出onnx)
    - [在GPU执行ONNX推理](#在gpu执行onnx推理)
    - [结果](#结果-2)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo](#modelzoo)

<!-- /TOC -->

# EfficientNet-B1描述

EfficientNet是一种卷积神经网络架构和缩放方法，它使用复合系数统一缩放深度/宽度/分辨率的所有维度。与任意缩放这些因素的常规做法不同，EfficientNet缩放方法使用一组固定的缩放系数来均匀缩放网络宽度，深度和分辨率。（2019年）

[论文](https://arxiv.org/abs/1905.11946)：Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 2019.

# 模型架构

EfficientNet总体网络架构如下：

[链接](https://arxiv.org/abs/1905.11946)

# 数据集

使用的数据集：[imagenet](http://www.image-net.org/)

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

## 脚本和样例代码

```text
├─ EfficientNet-B1
│   ├─ README_CN.md                     # EfficientNet-B1相关描述
│   ├─ ascend310_infer                  # 310推理脚本
│   │   ├─ inc
│   │   │   └─ utils.h
│   │   └─ src
│   │       ├─ build.sh
│   │       ├─ CMakeLists.txt
│   │       ├─ main.cc
│   │       └─ utils.cc
│   ├─ scripts
│   │   ├─ run_infer_310.sh             # 用于310推理的shell脚本
│   │   ├─ run_standalone_train.sh      # 用于单卡训练的shell脚本
│   │   ├─ run_distribute_train.sh      # 用于八卡训练的shell脚本
│   │   └─ run_eval.sh                  # 用于评估的shell脚本
│   │   └─ run_infer_onnx.sh            # 用于ONNX推理的shell脚本
│   ├─ src
│   │   ├─ model_utils                  # modelarts训练适应脚本
│   │   │   └─ moxing_adapter.py
│   │   ├─ models                       # EfficientNet架构
│   │   │   ├─ effnet.py
│   │   │   └─ layers.py
│   │   ├─ callback.py                  # 参数配置
│   │   ├─ config.py                    # 配置参数
│   │   ├─ dataset.py                   # 创建数据集
│   │   ├─ loss.py                      # 损失函数
│   │   └─ utils.py                     # 工具函数脚本
│   ├─ create_imagenet2012_label.py     # 创建ImageNet2012标签
│   ├─ eval.py                          # 评估脚本
│   ├─ infer_onnx.py                    # ONNX评估
│   ├─ export.py                        # 模型格式转换脚本
│   ├─ postprocess.py                   # 310推理后处理脚本
│   └─ train.py                         # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置：

```text
'train_url': None,                      # 训练输出路径（桶）
'train_path': None,                     # 训练输出路径
'data_url': None,                       # 训练数据集路径（桶）
'data_path': None,                      # 训练数据集路径
'checkpoint_url': None,                 # checkpoint路径（桶）
'checkpoint_path': None,                # checkpoint路径
'eval_data_url': None,                  # 推理数据集路径（桶）
'eval_data_path': None,                 # 推理数据集路径
'eval_interval': 10,                    # 训练时推理的时间间隔
'modelarts': False,                     # 是否使用modelarts
'run_distribute': False,                # 是否多卡训练
'device_target': 'Ascend',              # 训练平台
'begin_epoch': 0,                       # 开始训练周期
'end_epoch': 350,                       # 结束训练周期
'total_epoch': 350,                     # 总训练周期数
'dataset': 'imagenet',                  # 数据集名称
'num_classes': 1000,                    # 数据集类别数
'batchsize': 128                        # 输入批次大小
'input_size': 240,                      # 输入尺寸大小
'lr_scheme': 'linear',                  # 学习率衰减方案
'lr': 0.15,                             # 最大学习率
'lr_init': 0.0001,                      # 初始学习率
'lr_end': 5e-5   ,                      # 最终学习率
'warmup_epochs': 2,                     # 热身周期数
'use_label_smooth': True,               # 是否使用label smooth
'label_smooth_factor': 0.1,             # 标签平滑因子
'conv_init': 'TruncatedNormal',         # 卷积层初始化方式
'dense_init': 'RandomNormal',           # 全连接层初始化方式
'optimizer': 'rmsprop',                 # 优化器
'loss_scale': 1024,                     # loss scale
'opt_momentum': 0.9,                    # 动量参数
'wd': 1e-5,                             # 权重衰减率
'eps': 0.001                            # epsilon
'save_ckpt': True,                      # 是否保存ckpt文件
'save_checkpoint_epochs': 1,            # 每迭代相应次数保存一个ckpt文件
'keep_checkpoint_max': 10,              # 保存ckpt文件的最大数量
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```bash
# 训练示例
  python:
      Ascend单卡训练示例：
          python train.py --data_path [DATA_DIR] --train_path [TRAIN_OUTPUT_PATH] --model efficientnet-b1 --run_distribute False

  shell:
      Ascend单卡训练示例: bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]
      Ascend八卡并行训练:
          bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

### 结果

多卡训练ckpt文件将存储在 `./checkpoint` 路径下，而单卡训练存放于指定目录。训练日志将被记录到 `log` 中。训练日志部分示例如下：

```text
epoch: [ 1/350], epoch time: 2709470.652, steps: 625, per step time: 4335.153, avg loss: 5.401, lr:[0.050]
epoch: [ 2/350], epoch time: 236883.599, steps: 625, per step time: 379.014, avg loss: 4.142, lr:[0.100]
epoch: [ 3/350], epoch time: 236615.708, steps: 625, per step time: 378.585, avg loss: 3.724, lr:[0.100]
epoch: [ 4/350], epoch time: 236606.486, steps: 625, per step time: 378.570, avg loss: 3.133, lr:[0.099]
epoch: [ 5/350], epoch time: 236639.009, steps: 625, per step time: 378.622, avg loss: 3.225, lr:[0.099]
```

## 评估过程

### 启动

您可以使用python或shell脚本进行评估。

```bash
# 评估示例
  python:
      python eval.py --data_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

  shell:
      bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

> 训练过程中可以生成ckpt文件。

### 结果

可以在 `eval_log` 查看评估结果。

```bash
{'Loss': 1.8175019884720827, 'Top_1_Acc': 0.7914495192307693, 'Top_5_Acc': 0.9445458333333333}
```

# 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

## 导出MINDIR

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 240 --height 240 --file_format MINDIR
```

## 执行推理

在执行推理前，mindir文件必须通过 `export.py` 脚本导出。以下展示了使用mindir模型执行推理的示例。

```bash
bash scripts/run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

## 导出ONNX

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 240 --height 240 --file_format ONNX
```

## 在GPU执行ONNX推理

在执行推理前，ONNX文件必须通过 `export.py` 脚本导出。以下展示了使用ONNX模型执行推理的示例。

```bash
# ONNX inference
bash scripts/run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [DEVICE_TARGET]
```

## 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到Ascend 310精度计算结果,在infer_onnx.log中看到ONNX推理精度计算结果。

# 模型说明

## 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | EfficientNet                          |
| 模型版本                    | B1                           |
| 运行环境                    | HUAWEI CLOUD Modelarts                     |
| 上传时间                    | 2021-12-06                             |
| 数据集                      | imagenet                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | RMSProp                              |
| 损失函数                    | CrossEntropySmooth         |
| 最终损失                    | 1.82                                 |
| 精确度 (8p)                 | Top1[79.1%], Top5[94.4%]               |
| 训练总时间 (8p)             | 25.1h                                    |
| 评估总时间                  | 84s                                    |
| 参数量 (M)                 | 30M                                   |
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1) |

# 随机情况的描述

我们在 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models) 。
