目录

- [目录](#目录)
- [STPM概述](#STPM概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [预训练模型](#预训练模型)
        - [训练过程](#训练过程)
            - [训练](#训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
        - [导出mindir模型](#导出mindir模型)
        - [推理过程](#推理过程)
            - [用法](#用法)
            - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# STPM概述

该模型总体分为两个网络，教师与学生网络。教师网络是个在图片分类任务中预训练的网络，学生网络则具有与之相同的架构。如果测试图像或像素在两个网络中的特征有显著差异，则其具有较高的异常分数。两个网络之间具有分层特征对齐的功能，使其能够通过一次正向检测出不同大小的异常。

[论文](https://arxiv.org/pdf/2103.04257v2.pdf)： Wang G ,  Han S ,  Ding E , et al. Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection[J].  2021.

# 数据集

使用的训练数据集：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

- 数据集大小：4.9G，共15个类、5354张图片(尺寸在700x700~1024x1024之间)
    - 训练集：共3629张
    - 测试集：共1725张

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 单机训练运行示例
bash scripts/run_standalone_train.sh

# 运行评估示例
bash scripts/run_eval.sh
```

## 脚本说明

## 脚本和样例代码

```text
└── STPM
 ├── README.md                          // STPM相关描述
 ├── ascend_310_infer                   // 310推理相关文件
  ├── inc
   └── utils.h
  ├── src
   ├── main.cc
   └── utils.cc
  ├── build.sh
  ├── Cmakelists.txt
 ├── scripts
  ├── run_310_infer.sh                  // 用于310推理的脚本
  ├── run_standalone_train.sh           // 用于单机训练的shell脚本
  └── run_eval.sh                       // 用于评估的shell脚本
 ├──src
  ├── loss.py                           //损失函数
  ├── dataset.py                        // 创建数据集
  ├── callbacks.py                      // 回调
  ├── resnet.py                         // ResNet架构
  ├── EvalOneStep.py
  ├── STPM.py
  ├── pth2ckpt.py                       // 该脚本可将torch的pth文件转为ckpt
  ├── utils.py
 ├── test.py                            // 测试脚本
 ├── train.py                           // 训练脚本
 ├── preprocess.py                      // 310推理前处理脚本
 ├── postprocess.py                     // 310推理后处理脚本
 ├── requirements.txt                   //requirements
```

## 脚本参数

```text
train.py和test.py中主要参数如下：

-- modelarts：是否使用modelarts平台训练。可选值为True、False。默认为False。
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- train_url：checkpoint的输出路径。
-- data_url：训练集路径。
-- ckpt_url：checkpoint路径。
-- eval_url：验证集路径。

```

## 预训练模型

本模型的预训练模型为ResNet18。从modelzoo中下载ResNet18的预训练模型，将其中的每一个参数字段前加上"model_t"，转化为本模型所需的ckpt文件。使用该预训练模型加载到教师网络中，而学生网络则不使用预训练模型。

[点此下载](https://download.mindspore.cn/model_zoo/r1.3/resnet18_ascend_v130_imagenet2012_official_cv_bs256_acc70.64/)

## 训练过程

### 训练

```shell
bash scripts/run_standalone_train.sh [DATASET_PATH] [CHECKPOINT_PATH] [CATEGORY]
```

上述shell脚本将在后台运行训练。可以通过`train.log`文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估

  ```bash
  bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CATEGORY]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

|  Category  | pixel-level | image-level |
| :--------: | :---------: | :---------: |
|   carpet   |    0.949    |    0.919    |
|    grid    |    0.988    |    0.987    |
|  leather   |    0.986    |    0.965    |
|    tile    |    0.95     |    0.967    |
|    wood    |    0.864    |    0.897    |
|   bottle   |    0.982    |    0.998    |
|   cable    |    0.951    |    0.943    |
|  capsule   |    0.961    |    0.934    |
|  hazelnut  |    0.979    |    0.986    |
| metal nut  |    0.969    |    0.997    |
|    pill    |    0.972    |    0.914    |
|   screw    |    0.985    |    0.847    |
| toothbrush |    0.98     |    0.875    |
| transistor |    0.894    |    0.973    |
|   zipper   |    0.983    |    0.986    |
|    mean    |    0.960    |    0.946    |

## 导出mindir模型

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

# 推理过程

## 用法

在执行推理之前，需要通过export.py导出mindir文件。

```bash
# Ascend310 推理
bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CATEGORY]
```

`DEVICE_TARGET` 可选值范围为：['GPU', 'CPU', 'Ascend']，`NEED_PREPROCESS` 表示数据是否需要预处理，可选值范围为：'y' 或者 'n'，这里直接选择‘y’，`DEVICE_ID` 为设备id，`CATEGORY` 为类别。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。推理结果示例如下。

```text
category:  zipper
Total pixel-level auc-roc score :  0.9817775778360928
Total image-level auc-roc score : 0.9826680672268908
```

# 模型描述

## 性能

### 训练性能

| 参数          | STPM                                            |
| ------------- | ----------------------------------------------- |
| 模型版本      | STPM                                            |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期      | 2021-12-25                                      |
| MindSpore版本 | 1.5.0                                           |
| 数据集        | MVTec AD                                        |
| 训练参数      | lr=0.4                                          |
| 优化器        | SGD                                             |
| 损失函数      | MSELoss                                         |
| 输出          | 概率                                            |
| 损失          | 2.6                                             |
| 总时间        | 1卡：0.6小时                                    |

# 随机情况说明

网络的初始参数均为随即初始化。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。

