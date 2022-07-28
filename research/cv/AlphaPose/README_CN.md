# 目录

<!-- TOC -->

- [Alphapose描述](#AlphaPose描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [310推理过程](#310推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# AlphaPose描述

## 概述

AlphaPose是由上海交通大学卢策吾团队提出，作者提出了区域姿态估计（Regional Multi-Person Pose Estimation，RMPE）框架。主要包括symmetric spatial transformer network (SSTN)、Parametric Pose Non- Maximum-Suppression (NMS), 和Pose-Guided Proposals Generator (PGPG)。并且使用symmetric spatial transformer network (SSTN)、deep proposals generator (DPG) 、parametric pose nonmaximum suppression (p-NMS) 三个技术来解决复杂场景下多人姿态估计问题。

AlphaPose模型网络具体细节可参考[论文1](https://arxiv.org/pdf/1612.00137.pdf)，AlphaPose模型网络Mindspore实现基于上海交通大学卢策吾团队发布的Pytorch版本实现，具体可参考(<https://github.com/MVIG-SJTU/AlphaPose>)。

## 论文

1. [论文](https://arxiv.org/pdf/1804.06208.pdf)：Fang H S , Xie S , Tai Y W , et al. RMPE: Regional Multi-person Pose Estimation

# 模型架构

AlphaPose的总体网络架构如下：
[链接](https://arxiv.org/abs/1612.00137)

# 数据集

使用的数据集：[COCO2017]

- 数据集大小：
    - 训练集：19.56G, 118,287个图像
    - 测试集：825MB, 5,000个图像
- 数据格式：JPG文件
    - 注：数据在src/dataset.py中处理

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 预训练模型

  当开始训练之前需要获取mindspore图像网络预训练模型，可通过在[official model zoo](https://gitee.com/mindspore/models/tree/master/official/cv/resnet)中运行Resnet训练脚本来获取模型权重文件，预训练文件名称为resnet50.ckpt。

- 数据集准备

  Alphapose网络模型使用COCO2017数据集用于训练和推理，数据集可通过[official website](https://cocodataset.org/)官方网站下载使用。

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh --is_model_arts False --run_distribute True

# 单机训练
用法：bash run_standalone_train.sh --device_id 0

# 运行评估示例
用法：bash scripts/run_eval.sh checkpoint_path device_id

# 运行demo
用法：bash run_demo.sh
```

# 脚本说明

## 脚本及样例代码

```shell

└──AlphaPose
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_demo.sh                        # 启动demo（单卡）
    ├── run_eval.sh                        # 启动Ascend评估
    └── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
  ├── src
    ├── utils
        ├── coco.py                        # COCO数据集评估结果
        ├── fn.py                          # 根据关键点绘制人体姿势
        ├── inference.py                   # 热图关键点预测
        ├── nms.py                         # nms
        └── transforms.py                  # 图像处理转换
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── DUC.py                             # 网络部分结构DUC
    ├── FastPose.py                        # 主干网络定义
    ├── network_with_loss.py               # 损失函数定义
    ├── SE_module.py                       # 网络部分结构SE
    └── SE_module.py                       # 网络部分结构ResNet50
  ├── demo.py                              # demo
  ├── data_to_bin.py                       # 将数据集中图片转为二进制
  ├── export.py                            # 将ckpt模型文件转为mindir
  ├── postprocess.py                       # 后处理求精度
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
```

## 脚本参数

在src/config.py中配置相关参数。

- 配置模型相关参数：

```python
config.MODEL.INIT_WEIGHTS = True                                 # 初始化模型权重
config.MODEL.PRETRAINED = 'resnet50.ckpt'                        # 预训练模型
config.MODEL.NUM_JOINTS = 17                                     # 关键点数量
config.MODEL.IMAGE_SIZE = [192, 256]                             # 图像大小
```

- 配置网络相关参数：

```python
config.NETWORK.NUM_LAYERS = 50                                   # resnet主干网络层数
config.NETWORK.DECONV_WITH_BIAS = False                          # 网络反卷积偏差
config.NETWORK.NUM_DECONV_LAYERS = 3                             # 网络反卷积层数
config.NETWORK.NUM_DECONV_FILTERS = [256, 256, 256]              # 反卷积层过滤器尺寸
config.NETWORK.NUM_DECONV_KERNELS = [4, 4, 4]                    # 反卷积层内核大小
config.NETWORK.FINAL_CONV_KERNEL = 1                             # 最终卷积层内核大小
config.NETWORK.HEATMAP_SIZE = [48, 64]                           # 热图尺寸
```

- 配置训练相关参数：

```python
config.TRAIN.SHUFFLE = True                                      # 训练数据随机排序
config.TRAIN.BATCH_SIZE = 64                                     # 训练批次大小
config.TRAIN.BEGIN_EPOCH = 0                                     # 测试数据集文件名
config.DATASET.FLIP = True                                       # 数据集随机翻转
config.DATASET.SCALE_FACTOR = 0.3                                # 数据集随机规模因数
config.DATASET.ROT_FACTOR = 40                                   # 数据集随机旋转因数
config.TRAIN.BEGIN_EPOCH = 0                                     # 初始周期数
config.TRAIN.END_EPOCH = 270                                     # 最终周期数
config.TRAIN.LR = 0.001                                          # 初始学习率
config.TRAIN.LR_FACTOR = 0.1                                     # 学习率降低因子
```

- 配置验证相关参数：

```python
config.TEST.BATCH_SIZE = 32                                      # 验证批次大小
config.TEST.FLIP_TEST = True                                     # 翻转验证
config.TEST.USE_GT_BBOX = False                                  # 使用标注框
```

- 配置nms相关参数：

```python
config.TEST.OKS_THRE = 0.9                                       # OKS阈值
config.TEST.IN_VIS_THRE = 0.2                                    # 可视化阈值
config.TEST.BBOX_THRE = 1.0                                      # 候选框阈值
config.TEST.IMAGE_THRE = 0.0                                     # 图像阈值
config.TEST.NMS_THRE = 1.0                                       # nms阈值
```

- 配置demo相关参数：

```python
config.detect_image = "images/1.jpg"                             # 检测图片
config.yolo_image_size = [416, 416]                              # yolo网络输入图像大小
config.yolo_ckpt = "yolo/yolo.ckpt"                              # yolo网络权重
config.fast_pose_ckpt = "fastpose.ckpt"                          # fastpose网络权重
config.yolo_threshold = 0.1                                      # bbox阈值
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh --is_model_arts False --run_distribute True

# 单机训练
用法：bash run_standalone_train.sh --device_id 0

# 运行评估示例
用法：bash scripts/run_eval.sh checkpoint_path device_id
```

### 结果

- 使用COCO2017数据集训练Alphapose

```text
分布式训练结果（8P）
epoch:1 step:292, loss is 0.001391
epoch:2 step:292, loss is 0.001326
epoch:3 step:292, loss is 0.001001
epoch:4 step:292, loss is 0.0007763
epoch:5 step:292, loss is 0.0006757
...
epoch:288 step:292, loss is 0.0002837
epoch:269 step:292, loss is 0.0002367
epoch:270 step:292, loss is 0.0002532
```

## 评估过程

### 用法

#### Ascend处理器环境运行

可通过改变config.py文件中的"config.TEST.MODEL_FILE"文件进行相应模型推理。

```bash
# 评估
bash scripts/run_eval.sh checkpoint_path device_id
```

### 结果

使用COCO2017数据集文件夹中val2017进行评估Alphapose,如下所示：

```text
coco eval results saved to /cache/train_output/multi_train_poseresnet_v5_2-140_2340/keypoints_results.pkl
AP: 0.723
```

## 310推理过程

### 用法

#### 导出模型

```python
# 导出模型
python export.py [ckpt_url] [device_target] [device_id] [file_name] [file_format]
```

#### Ascend310处理器环境运行

```bash
# 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

#### 获取精度

```bash
# 获取精度
more acc.log
```

### 结果

```text
AP: 0.723
```

# 模型描述

## 性能

### 评估性能

#### coco2017上的性能参数

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | ResNet50                                                |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2020-12-16                                |
| MindSpore版本          | 1.3                                                 |
| 数据集                    | coco2017                                                    |
| 训练参数        | epoch=270, steps=2336, batch_size = 64, lr=0.001              |
| 优化器                  | Adam                                                    |
| 损失函数              | Mean Squared Error                                      |
| 输出                    | heatmap                                                 |
| 损失                       | 0.00025                                                      |
| 速度                      | 单卡：138.9毫秒/步;  8卡：147.28毫秒/步                          |
| 总时长                 | 单卡：24h22m36s;  8卡：3h13m31s                          |
| 参数(M)             | 13.0                                                        |
| 微调检查点 | 389.64M (.ckpt文件)                                         |
| 推理模型        | 57.26M (.om文件),  112.76M(.MINDIR文件)                     |

### 推理性能

#### coco2017上的性能参数

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | ResNet50               |
| 资源            | Ascend 910                  |
| 上传日期       | 2020-12-16 |
| MindSpore 版本   | 1.3                 |
| 数据集             | coco2017     |
| batch_size          | 32                         |
| 输出             | heatmap                 |
| 准确性            | 单卡: 72.3%;  8卡：72.5%   |
| 推理模型 | 389.64M (.ckpt文件)         |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时在model.py中使用了初始化网络权重。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
