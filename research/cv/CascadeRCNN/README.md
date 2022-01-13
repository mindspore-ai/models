# 目录

- [目录](#目录)
- [Cascade R-CNN描述](#Cascade-r-cnn描述)
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

# Cascade R-CNN描述

Cascade R-CNN算法是CVPR2018的文章，通过级联几个检测网络达到不断优化预测结果的目的，与普通级联不同的是，cascade R-CNN的几个检测网络是基于不同IOU阈值确定的正负样本上训练得到的，这是该算法的一大亮点。cascade R-CNN的实验大部分是在COCO数据集做的，而且效果非常出彩。

[论文](https://arxiv.org/abs/1712.00726)：  Cai Z, Vasconcelos N. Cascade r-cnn: Delving into high quality object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6154-6162.

# 数据集

使用的数据集：[COCO 2017](<https://cocodataset.org/>)

- 数据集大小：19G
    - 训练集：18G，118,000个图像  
    - 验证集：1G，5000个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

# 环境要求

- 硬件（Ascend/GPU）

    - 使用Ascend处理器来搭建硬件环境。

- 下载数据集COCO 2017。

    1. 若使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练模型是在ImageNet2012上训练的[Resnet101](https://download.mindspore.cn/model_zoo/r1.2/) 检查点。

## 在Ascend上运行

```shell

# 单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# 分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# 评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]

```

# 脚本说明

## 脚本及样例代码

```shell
.
└─Cascade R-CNN
  ├─README.md                        // Cascade R-CNN相关说明
  ├─scripts
    ├─run_standalone_train_ascend.sh // Ascend单机shell脚本
    ├─run_distribute_train_ascend.sh // Ascend分布式shell脚本
    └─run_eval_ascend.sh             // Ascend评估shell脚本
  ├─src
    ├─CascadeRCNN
      ├─__init__.py                  // init文件
      ├─anchor_generator.py          // 锚点生成器
      ├─bbox_assign_sample.py        // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py // 第二阶段级联一阶段采样器
      ├─bbox_assign_sample_stage2_1_1.py // 第二阶段级联二阶段采样器
      ├─bbox_assign_sample_stage2_2_1.py // 第二阶段级联三阶段采样器
      ├─cascade_rcnn_r101.py         // 以Resnet101作为backbone的Cascade R-CNN网络
      ├─fpn_neck.py                  // 特征金字塔网络
      ├─proposal_generator.py        // 候选生成器
      ├─rcnn.py                      // R-CNN网络
      ├─resnet.py                    // 骨干网络
      ├─roi_align.py                 // ROI对齐网络
      └─rpn.py                       // 区域候选网络
    ├─dataset.py                     // 创建并处理数据集
    ├─lr_schedule.py                 // 学习率生成器
    ├─network_define.py              // Cascade R-CNN网络定义
    ├─util.py                        // 例行操作
    ├─config.py                      // 模型参数
  ├─export.py                        // 导出 AIR,MINDIR模型的脚本
  ├─eval.py                          // 评估脚本
  └─train.py                         // 训练脚本
```

## 参数说明

```shell
'dataset':'coco'          # 数据集名称
'pre_trained':200         # 预训练模型
'run_distribute':0        # 分布式训练
'device_target':'Ascend'  # 运行设备目标
'device_id':4             # 使用卡的编号
'device_num':1            # 设备数量
'rank_id':0               # Rank id
```

## 训练过程

### 用法

#### 在Ascend上运行

```shell
# Ascend单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL]

# Ascend分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# Ascend评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647
```

# 模型描述

## 性能

### 训练性能

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/11/02 |
| MindSpore版本 | 1.3.0 |
| 数据集 | COCO 2017 |
| 训练参数 | epoch=40, batch_size=2 |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 1卡：190毫秒/步；8卡：200毫秒/步 |
| 总时间 | 1卡：182.72小时；8卡：24.89小时 |
| 参数(M) | 1013 |

### 评估性能

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/11/02 |
| MindSpore版本 | 1.3.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：55.5%  |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
