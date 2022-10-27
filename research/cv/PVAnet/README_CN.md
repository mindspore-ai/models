# 目录

- [目录](#目录)
- [PVANet描述](#PVANet描述)
- [模型各模块简介](#模型各模块简介)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PVANet描述

PVANET基于Faster-RCNN进行改进，其含义为：Performance Vs Accuracy，意为加速模型性能，同时不丢失精度。该模型主要的工作在于使用自己设计的高效基础网络，该网络使用了C.ReLU、Inception、HyperNet多尺度思想以及Residual模块等技巧。

# 模型各模块简介

**C.ReLU**：抓住低层卷积层中滤波器核存在着负相关程度很高的特性，通过使用一半的输入通道数，进行简单的取反连接使其通道数不变，这可获得2倍的速度提升而精度损失很小。PVANet中又增加了scaling和shifting，串联之后，使得每个通道的权重和激活阈值均有不同。

**Inception**：Inception结构通过不同分支堆叠3x3与1x1的卷积核来获得不同大小的感受野，PVANet中同时引入了残差连接，稳定网络框架的后半部分，加速网络训练

**HyperNet**：论文中的HyperNet是用于融合多尺度特征，分别通过升采样与降采样将不同分辨率的特征图融合

# 数据集

使用的数据集：VOC2007训练集(5011幅)，测试集(4952幅)，共计9963幅图像，共包含20个种类，数据格式为COCO。数据目录如下：

```shell
.
└─cocodataset
  ├─annotations
    ├─instance_train2017.json
    └─instance_val2017.json
  ├─val2017
  └─train2017
```

数据集下载：

```shell
mkdir /path/to/VOC2007
cd /path/to/VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
mkdir VOCtrainval/ VOCtest/
tar xvf VOCtrainval_06-Nov-2007.tar -C VOCtrainval/
tar xvf VOCtest_06-Nov-2007.tar -C VOCtest/
```

数据集格式转换：

```shell
cd PVANet
python voc2007_2_coco.py --voc_root=/path/to/VOC2007
实例：
python voc2007_2_coco.py --voc_root=/data2/VOC2007
```

# 环境要求

- 硬件（GPU）

- 获取基础镜像

- 安装[MindSpore](https://www.mindspore.cn/install)

- 环境要求：

  ```shell
  Python==3.7.5
  Mindspore==1.6.0
  Numpy==1.21.6
  Mmcv==0.2.14
  Seaborn==0.11.2
  ```

- 下载数据集VOC2007，并转换coco格式。本示例默认使用VOC2007作为训练数据集，您也可以使用自己的数据集。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：第一次运行生成MindRecord文件，耗时较长。

## 在GPU上运行

```shell
# 设置：更改default_configs.yaml下的对应内容
mindrecord_dir: "/path/to/VOC2007/mindrecords" # 数据集mindrecord路径
coco_root: "/path/to/VOC2007/coco" # 数据集根目录
anno_path: "/path/to/VOC2007/coco/annotations/instances_val2017.json" # 验证集标签路径
pre_trained: "/path/to/pretrained_pvanet.ckpt" # 预训练权重路径
checkpoint_path: "./../../ckpt/pvanet-20_2505.ckpt" # gpu推理的权重路径
ckpt_file: "./../../ckpt/pvanet-20_2505.ckpt" # 转换mindir时的权重路径

# 训练：GPU-1p:
bash run_standalone_train_gpu.sh [CHECKPOINT_PATH] PVAnet [COCO_ROOT]

# 训练：GPU-8p：
bash run_distribute_train_gpu.sh 8 [DEVICE_NUMBER] [CHECKPOINT_PATH] PVAnet [COCO_ROOT]

# 验证：GPU
bash run_eval_gpu.sh [VALIDATION_JSON_PATH] [CKPT_PATH] PVAnet [COCO_ROOT]
实例
bash run_eval_gpu.sh  /data2/Travis/PVAnet_test/PVAnet/data/annotations/instances_val2017.json   /data2/Travis/PVAnet_test/PVAnet/ckpt/ckpt_0/pvanet_2-20_313.ckpt  PVAnet   /data2/Travis/PVAnet_test/PVAnet/data
# device_id为GPU设备id，默认为0

```

# 脚本说明

## 脚本及样例代码

```bash
├── scripts
│   ├── run_eval_gpu.sh                # GPU评估shell脚本
│   ├── run_standalone_train_gpu.sh    # GPU-1P推理shell脚本
│   ├── run_distribute_train_gpu.sh    # GPU-8P推理shell脚本
│   ├── run_export.sh                  # 导出mindir shell脚本
├── src
│   ├── dataset.py          # 数据处理相关函数
│   ├── detecteval.py       # 推理验证相关函数
│   ├── eval_callback.py    # 推理验证相关函数
│   ├── eval_utils.py       # 推理验证相关函数
│   ├── lr_schedule.py      # 学习率生成器
│   ├── model_utils
│   │   ├── config.py       # 获取ymal配置参数并设置config
│   │   ├── device_adapter.py   # 获取云上id
│   │   ├── local_adapter.py    # 获取本地id
│   │   ├── moxing_adapter.py   # 云上数据准备
│   ├── PVANet
│   │   ├── anchor_generator.py          # 锚点生成器
│   │   ├── backbone.py                  # 主干网络
│   │   ├── bbox_assign_sample.py        # 第一阶段采样器
│   │   ├── bbox_assign_sample_stage2.py # 第二阶段采样器
│   │   ├── fpn_neck.py                  # fpn特征金字塔
│   │   ├── proposal_generator.py        # 候选生成器
│   │   ├── pva_faster_rcnn.py           # PVANet网络
│   │   ├── rcnn.py                      # rcnn网络
│   │   ├── roi_align.py                 # roi对齐网络
│   │   └── rpn.py                       # 候选区域网络
│   └── util.py
├── default_config.yaml     # 配置文件
├── eval.py                 # 验证脚本
├── export.py               # 导出mindir脚本
├── train.py                # 训练脚本
└── voc_2007_2_coco.py      # voc转coco脚本
```

## 训练过程

### 用法

#### 在Gpu上运行

```shell
# 训练：GPU-1p:
bash run_standalone_train_gpu.sh [CHECKPOINT_PATH] PVAnet [COCO_ROOT]

# 训练：GPU-8p：
bash run_distribute_train_gpu.sh 8 [DEVICE_NUMBER] [CHECKPOINT_PATH] PVAnet [COCO_ROOT]
```

Notes:

1. 在default_config.yaml配置文件中修改数据集与预训练权重路径。

### 结果

```log
# train log:
Train epoch time: 854377.301 ms, per step time: 341.069 ms
Train epoch time: 259295.051 ms, per step time: 103.511 ms
Train epoch time: 261208.460 ms, per step time: 104.275 ms
Train epoch time: 262092.130 ms, per step time: 104.628 ms
Train epoch time: 262372.078 ms, per step time: 104.739 ms
Train epoch time: 262133.320 ms, per step time: 104.644 ms
Train epoch time: 262380.045 ms, per step time: 104.743 ms
Train epoch time: 262497.524 ms, per step time: 104.789 ms
Train epoch time: 262673.653 ms, per step time: 104.860 ms
Train epoch time: 262838.965 ms, per step time: 104.926 ms
Train epoch time: 262996.807 ms, per step time: 104.989 ms
Train epoch time: 262795.771 ms, per step time: 104.908 ms
Train epoch time: 262959.979 ms, per step time: 104.974 ms
Train epoch time: 263390.558 ms, per step time: 105.146 ms
Train epoch time: 262634.204 ms, per step time: 104.844 ms
Train epoch time: 262506.849 ms, per step time: 104.793 ms
Train epoch time: 262455.230 ms, per step time: 104.773 ms
Train epoch time: 262665.083 ms, per step time: 104.856 ms
Train epoch time: 262747.014 ms, per step time: 104.889 ms
Train epoch time: 262486.264 ms, per step time: 104.785 ms
```

```log
# train loss:
845 s | epoch: 1 step: 2505 total_loss: 0.04977
1112 s | epoch: 2 step: 2505 total_loss: 0.27578
1373 s | epoch: 3 step: 2505 total_loss: 0.20070
1635 s | epoch: 4 step: 2505 total_loss: 0.21217
1897 s | epoch: 5 step: 2505 total_loss: 0.25250
2160 s | epoch: 6 step: 2505 total_loss: 0.14541
2422 s | epoch: 7 step: 2505 total_loss: 0.06361
2684 s | epoch: 8 step: 2505 total_loss: 0.13346
2947 s | epoch: 9 step: 2505 total_loss: 0.03807
3210 s | epoch: 10 step: 2505 total_loss: 0.35591
3473 s | epoch: 11 step: 2505 total_loss: 0.22954
3736 s | epoch: 12 step: 2505 total_loss: 0.04614
3999 s | epoch: 13 step: 2505 total_loss: 0.29424
4262 s | epoch: 14 step: 2505 total_loss: 0.22942
4525 s | epoch: 15 step: 2505 total_loss: 0.07449
4787 s | epoch: 16 step: 2505 total_loss: 0.31901
5050 s | epoch: 17 step: 2505 total_loss: 0.20565
5312 s | epoch: 18 step: 2505 total_loss: 0.14162
5575 s | epoch: 19 step: 2505 total_loss: 0.22775
5838 s | epoch: 20 step: 2505 total_loss: 0.24014
```

## 评估过程

### 用法

#### 在GPU上运行

```shell
# shell脚本
bash run_eval_gpu.sh [ANNOTATION_PATH] [CKPT_PATH] PVAnet [COCO_ROOT]
```

Notes:

- 需在`default_config.yaml`配置文件中设置`anno_path`，验证集标签路径，或者在执行`Python`脚本时指定`checkpoint_path`。

### 结果

评估结果将保存在`eval.log`中。您可以在日志中找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.563
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
```

## 模型导出

```shell
bash scripts/run_export.sh    # 保存路径为 ./weights/pvanet.mindir
```

### 结果

推理的结果保存在当前目录下，在``acc.log`日志文件中可以找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
```

# 模型描述

## 性能

### 训练性能

| 参数           | GPU |
| ------------- | ------------------------------------------ |
| 模型版本      |V1                                         |
| 资源          |RTX 3090; CPU 2.90GHz, 16核; 内存 24G  |
| 上传日期      |2022/9/27                                  |
| MindSpore版本 |1.6.0                            |
| 数据集        |VOC2007                                    |
| 训练参数      |epoch=20, batch_size=2                     |
| 优化器        |SGD                                        |
| 损失函数      |Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度          |1卡：173毫秒/步；8卡：262毫秒/步|
| 总时间        |1卡：142分钟；8卡：27分钟|
| 参数(M)       |230|

### 评估性能

| 参数          |GPU            |
| ------------- |----------------- |
| 模型版本      |V1                |
| 资源          |RTX 3090        |
| 上传日期      |2022/9/27         |
| MindSpore版本 |1.6.0    |
| 数据集        |VOC2007           |
| batch_size    |2                 |
| 输出          |mAP               |
| 准确率        |IoU=0.50：56.5%   |
| 推理模型      |241M（.ckpt文件） |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
