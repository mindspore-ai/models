目录

- [lresnet100e_ir描述](#lresnet100e_ir描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [分布式训练](#分布式训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
        - [导出mindir模型](#导出mindir模型)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# lresnet100e_ir描述

使用深度卷积神经网络进行大规模人脸识别的特征学习中的主要挑战之一是设计适当的损失函数以增强判别能力。继SoftmaxLoss、Center Loss、A-Softmax Loss、Cosine Margin Loss之后，Arcface在人脸识别中具有更加良好的表现。Arcface是传统softmax的改进， 将类之间的距离映射到超球面的间距，论文给出了对此的清晰几何解释。我们还研究了一种更先进的残差单元设置，提出网络LResNet100E-IR用于人脸识别模型的训练。

[论文](https://arxiv.org/pdf/1801.07698v1.pdf): Deng J, Guo J, Xue N, et al. Arcface: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4690-4699.

# 数据集

使用的训练数据集：[MS1MV2](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

验证数据集：lfw，cfp-fp，agedb_30

训练集：5,822,653张图片，85742个类

# 环境要求

- 硬件：GPU
    - 使用GPU处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 分布式训练运行示例
bash scripts/run_distribute_train.sh [DATA_PATH] [DEVICE_NUM]

# 单机训练运行示例
bash scripts/run_standalone_train.sh [DATA_PATH]

# 运行评估示例
bash scripts/run_eval.sh [EVAL_PATH] [CKPT_PATH]
```

## 脚本说明

## 脚本和样例代码

```path
└── lresnet100e_ir  
 ├── README.md                         // LResNet100E-IR相关描述
 ├── scripts
  ├── run_distribute_train.sh          // 用于分布式训练的shell脚本
  ├── run_standalone_train.sh          // 用于单机训练的shell脚本
  └── run_eval.sh                      // 用于评估的shell脚本
 ├──src
  ├── config.py                        //参数配置
  ├── dataset.py                       // 创建数据集
  ├── iresnet.py                       // ResNet架构
 ├── eval.py                           // 测试脚本
 ├── train.py                          // 训练脚本
 ├── export.py
 ├── requirements.txt
```

## 脚本参数

模型训练和评估过程中使用的参数可以在lresnet100e_ir_config_gpu.yaml中设置:

```python
    "num_classes": 85742,
    "image_size": 112,
    "batch_size": 128,
    "epoch_size": 25,
    "schedule": [10, 16, 21],
    "gamma": 0.1,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4
```

## 训练过程

### 分布式训练

```shell
bash scripts/run_distribute_train.sh [DATA_PATH] [DEVICE_NUM]
```

上述shell脚本将在后台运行分布训练。可以通过`train.log`文件查看结果。
采用以下方式达到损失值：

```log
epoch: 7 step: 5686, loss is 4.730966
epoch time: 6156469.326 ms, per step time: 1082.742 ms
epoch: 8 step: 5686, loss is 4.426299
epoch time: 6156968.563 ms, per step time: 1082.830 ms
...
epoch: 24 step: 5686, loss is 3.6390548
epoch time: 6154867.063 ms, per step time: 1082.460 ms
epoch: 25 step: 5686, loss is 3.3598282
epoch time: 6153828.222 ms, per step time: 1082.277 ms
```

## 评估过程

### 评估

- 在GPU环境运行时评估lfw、cfp_fp、agedb_30数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/LResNet100E-IR/LResNet100EIR_5-25_5686.ckpt”。

  ```bash
  bash scripts/run_eval.sh [EVAL_PATH] [CKPT_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
    [lfw]Accuracy: 0.99700+-0.00277
    [lfw]Accuracy-Flip: 0.99750+-0.00214
    [cfp_fp]Accuracy: 0.97414+-0.00668
    [cfp_fp]Accuracy-Flip: 0.97757+-0.00414
    [agedb_30]Accuracy: 0.96833+-0.01195
    [agedb_30]Accuracy-Flip: 0.96983+-0.01045
  ```

## 导出mindir模型

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

# 模型描述

## 性能

### 训练性能

| 参数          | LResNet100E-IR                                               |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | LResNet100E-IR                                               |
| 资源          | GeForce RTX 3090           |  |                                                              |
| 上传日期      | 2022-04-16                                                   |
| MindSpore版本 | 1.5.0                                |
| 数据集        | MS1MV2                                                       |
| 训练参数      | lr=0.1; gamma=0.1                                            |
| 优化器        | SGD                                                          |
| 损失函数      | SoftmaxCrossEntropyWithLogits                                |
| 输出          | 概率                                                         |
| 损失          | 3.3598282                                                    |
| 总时间(8p)    | 41h                                                 |      |                                                              |
| 脚本          | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/cv/lresnet100e_ir) |

### 评估性能

| 参数          | LResNet100E-IR            |
| ------------- | ------------------------ |
| 模型版本      | LResNet100E-IR           |
| 资源          | GeForce RTX 3090           |
| 上传日期      | 2022/04/16               |
| MindSpore版本 | 1.5.0                    |
| 数据集        | lfw，cfp-fp，agedb_30    |
| 输出          | 概率                     |
| 准确性        | lfw:0.997   cfp_fp:0.974   agedb_30:0.968 |

# 随机情况说明

网络的初始参数均为随即初始化。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。