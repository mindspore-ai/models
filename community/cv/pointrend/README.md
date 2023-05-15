# 目录

<!-- TOC -->

- [目录](#目录)
- [pointrend概述](#pointrend概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [推理过程](#推理过程)
        - [注意事项](#注意事项)
        - [推理](#推理)
        - [推理结果](#推理结果)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [modelzoo主页](#modelzoo主页)

<!-- /TOC -->

# PointRend概述

PointRend 把语义分割以及实例分割问题（统称图像分割问题）当做一个渲染问题来解决。但本质上这篇论文其实是一个新型上采样方法，针对物体边缘的图像分割进行优化，使其在难以分割的物体边缘部分有更好的表现。
论文： Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick. PointRend: Image Segmentation as Rendering. Arxiv preprint, arXiv:1912.08193v2, 2020.
官网实现代码，参考：https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend Pytorch实现版本，可参考：https://gitee.com/zhangliyuan97/PointRend-PyTorch.git

# 模型架构

Pointrend是基于MaskRCNN实现的。在MaskRCNN基础上增加了对粗粒度特征提取和细粒度特征提取，并最后通过一个简单的多层感知机进行分类预测，达到精细化物体边缘的效果。

[论文](https://ieeexplore.ieee.org/document/9156402)："PointRend"

# 数据集

- [COCO2017](https://cocodataset.org/)是一个广泛应用的数据集，带有边框和像素级背景注释。这些注释可用于场景理解任务，如语义分割，目标检测和图像字幕制作。训练和评估的图像大小为118K和5K。

- 数据集大小：19G
    - 训练：18G，118,000个图像
    - 评估：1G，5000个图像
    - 注释：241M；包括实例、字幕、人物关键点等

- 数据格式：图像及JSON文件
    - 注：数据在`dataset.py`中处理。

# 环境要求

- 硬件（GPU）
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- mindspore版本
    - 1.8
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

- 第三方库

```bash
pip install opencv-python
pip install Cython
pip install pycocotools
pip install mmcv=0.2.14
pip install pandas
```

# 快速入门

1. 下载COCO2017数据集。

2. 在`default_config.yaml`中修改COCO_ROOT及设置其他参数。参考目录结构如下：

    ```text
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017
    ```

3. 执行训练脚本。

    数据集准备完成后，按照如下步骤开始训练：

    ```bash
    V1版本代码：
    # 分布式训练
    cd scripts
    bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

    # 单机训练
    cd scripts
    bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

    V2版本代码（当前不支持）：
    # 分布式训练
    cd scripts
    bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]

    # 单机训练
    cd scripts
    bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]

    ```

    注：
    1. 为加快数据预处理速度，MindSpore提供了MindRecord数据格式。因此，训练前首先需要生成基于COCO2017数据集的MindRecord文件。COCO2017原始数据集转换为MindRecord格式大概需要4小时。
    2. 当进行单卡训练或者推理时，会自动根据coco2017数据集生成对应的MindRecord文件。
    3. PRETRAINED_CKPT是一个MaskRCNN检查点。你可以在 [MaskRCNN](https://download.mindspore.cn/model_zoo/r1.2/) 下载maskrcnn_ascend_v120_coco2017_official_cv_bs2_bboxacc37_segmacc32.ckpt。
    4. DATA_PATH为coco数据集路径根目录，MINDRECORD_PATH为将要生成的mindrecord文件的路径（需要大约300G的空间），PRETRAINED_PATH为预训练模型路径。
    5. pointrend实现预留了v2版本，pointrend在实现中使用mask的方式规避了tensor的bool索引，因此预留出V2版本代码，v2版本代码与pytorch一致，是使用了tensor的bool索引方式实现的，因此当前暂不支持运行。需要使用时，将NOT_MASK参数设为True即可。

4. 执行评估脚本。
   训练结束后，按照如下步骤启动评估：

   ```bash
   # 评估
   v1版本
   cd scripts
   bash run_eval.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [checkpoint_path]

   v2版本
   cd scripts
   bash run_eval.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [checkpoint_path] [NOT_MASK]
   ```

# 脚本说明

## 脚本和样例代码

```text
└─pointrend
  ├─maskrcnn
    ├─maskrcnn_mobilenetv1
      ├─__init__.py
      ├─anchor_generator.py               # 生成基础边框锚点
      ├─bbox_assign_sample.py             # 过滤第一阶段学习中的正负边框
      ├─bbox_assign_sample.py             # 过滤第二阶段学习中的正负边框
      ├─mask_rcnn_r50.py                  # MaskRCNN主要网络架构
      ├─fpn_neck.py                       # FPN网络
      ├─proposal_generator.py             # 基于特征图生成候选区域
      ├─rcnn_cls.py                       # RCNN边框回归分支
      ├─rcnn_mask.py                      # RCNN掩码分支
      ├─resnet50.py                       # 骨干网
      ├─roi_align.py                      # 兴趣点对齐网络
      └─rpn.py                            # 区域候选网络
    ├─model_utils
      ├─__init__.py
      ├─config.py                         # 训练配置
      ├─device_adapter.py                 # 获取云上id
      ├─local_adapter.py                  # 获取本地id
      └─moxing_adapter.py                 # 参数处理
    ├─__init__.py
    ├─mask_rcnn_mobilenetv1.py            #Maskrcnn主网络
  ├─maskrcnn_pointrend
    ├─src
      ├─pointrend
        ├─__init__.py
        ├─coarse_mask_head.py             # 粗粒度特征提取网路
        ├─point_head.py                   # Pointrend多层感知机分类器
        ├─sampling_points.py              # pointrend 采点策略
      ├─dataset.py                        # 数据集工具
      ├─lr_schedule.py                    # 学习率生成器
      ├─maskrcnnPointRend_r50.py          # Pointrend主网络
      ├─network_define.py                 # Pointrend的网络定义
      ├─util.py                           # 例行操作
  ├─scripts                               # shell脚本
    ├─run_standalone_train_gpu.sh         # 单机模式训练（单卡）
    ├─run_distribute_train_gpu.sh         # 并行模式训练（8卡）
    └─run_eval.sh                         # 评估
  ├─default_config.yaml                   # 训练参数配置文件
  ├─eval.py                               # 评估脚本
  ├─postprogress.py                       # 推理后处理脚本
  ├─README.md
  └─train.py                              # 训练脚本
```

## 脚本参数

### 训练脚本参数

```bash
V1版本
# 分布式训练
用法:
cd scripts
bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

# 单机训练
用法:
cd scripts
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

V2版本
# 分布式训练
用法:
cd scripts
bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]

# 单机训练
用法:
cd scripts
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]
```

### 参数配置

```bash
img_width: 1280
img_height: 768
keep_ratio: True
flip_ratio: 0.5
expand_ratio: 1.0

max_instance_count: 128
mask_shape: (28, 28)

# anchor
feature_shapes: [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)]
anchor_scales: [8]
anchor_ratios: [0.5, 1.0, 2.0]
anchor_strides: [4, 8, 16, 32, 64]
num_anchors: 3

# resnet
resnet_block: [3, 4, 6, 3]
resnet_in_channels: [64, 256, 512, 1024]
resnet_out_channels: [256, 512, 1024, 2048]

# fpn
fpn_in_channels: [256, 512, 1024, 2048]
fpn_out_channels: 256
fpn_num_outs: 5

# rpn
rpn_in_channels: 256
rpn_feat_channels: 256
rpn_loss_cls_weight: 1
rpn_loss_reg_weight: 1
rpn_cls_out_channels: 1
rpn_target_means: [0., 0., 0., 0.]
rpn_target_stds: [1.0, 1.0, 1.0, 1.0]

# bbox_assign_sampler
neg_iou_thr: 0.3
pos_iou_thr: 0.7
min_pos_iou: 0.3
num_bboxes: 245520
num_gts: 128
num_expected_neg: 256
num_expected_pos: 128

# proposal
activate_num_classes: 2
use_sigmoid_cls: True

# roi_align
roi_layer: dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2)
roi_align_out_channels: 256
roi_align_featmap_strides: [4, 8, 16, 32]
roi_align_finest_scale: 56
roi_sample_num: 640

# bbox_assign_sampler_stage2
neg_iou_thr_stage2: 0.5
pos_iou_thr_stage2: 0.5
min_pos_iou_stage2: 0.5
num_bboxes_stage2: 2000
num_expected_pos_stage2: 128
num_expected_neg_stage2: 512
num_expected_total_stage2: 512

# rcnn
rcnn_num_layers: 2
rcnn_in_channels: 256
rcnn_fc_out_channels: 1024
rcnn_mask_out_channels: 256
rcnn_loss_cls_weight: 1
rcnn_loss_reg_weight: 1
rcnn_loss_mask_fb_weight: 1
rcnn_loss_mask_coarse_weight: 1
rcnn_loss_mask_point_weight: 1
rcnn_target_means: [0., 0., 0., 0.]
rcnn_target_stds: [0.1, 0.1, 0.2, 0.2]

# train proposal
rpn_proposal_nms_across_levels: False
rpn_proposal_nms_pre: 2000
rpn_proposal_nms_post: 2000
rpn_proposal_max_num: 2000
rpn_proposal_nms_thr: 0.7
rpn_proposal_min_bbox_size: 0

# test proposal
rpn_nms_across_levels: False
rpn_nms_pre: 1000
rpn_nms_post: 1000
rpn_max_num: 1000
rpn_nms_thr: 0.7
rpn_min_bbox_min_size: 0
test_score_thr: 0.05
test_iou_thr: 0.5
test_max_per_img: 100
test_batch_size: 1

rpn_head_use_sigmoid: True
rpn_head_weight: 1.0
mask_thr_binary: 0.5

# LR
base_lr: 0.02
base_step: 29316
total_epoch: 13
warmup_step: 10000
warmup_ratio: 1/3.0
sgd_momentum: 0.9

# train
not_mask: False
batch_size: 4
loss_scale: 1
momentum: 0.91
weight_decay: 0.0001
pretrain_epoch_size: 0
epoch_size: 12
save_checkpoint: True
save_checkpoint_epochs: 0.5
keep_checkpoint_max: 12
save_checkpoint_path: "./"
dataset_sink_mode_flag: False

mindrecord_dir: ""
coco_root: ""
train_data_type: "train2017"
val_data_type: "val2017"
instance_set: "annotations/instances_{}.json"
coco_classes: ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush')
num_classes: 81
```

## 训练过程

- 在`default_config.yaml`中设置配置项，包括loss_scale、学习率和网络超参。

### 单机训练

- 运行`run_standalone_train_gpu.sh`开始Pointrend模型的非分布式训练。

```bash
v1版本
cd scripts
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

v2版本
cd scripts
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]
```

### 分布式训练

- 运行`run_distribute_train_gpu.sh`开始Pointrend模型的分布式训练。

```bash
v1版本
cd scripts
bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH]

v2版本
cd scripts
bash run_distribute_train_gpu.sh [DATA_PATH] [MINDRECORD_PATH] [PRETRAINED_PATH] [NOT_MASK]

```

- Notes

1. PRETRAINED_PATH应该是训练好的MaskRCNN检查点的路径。

## 推理过程

### 注意事项

```text
由于推理部分使用到了numpy.nonzero等操作，而mindspore静态图下不支持这些，因此推理部分只能在pynative模式下运行

```

### 推理

- 运行`run_eval.sh`进行推理。

```bash
# 推理
v1版本
cd scripts
bash run_eval.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [checkpoint_path]

v2版本
cd scripts
bash run_eval.sh [DEVICE_ID] [DATA_PATH] [MINDRECORD_PATH] [checkpoint_path] [NOT_MASK]

```

> 关于COCO2017数据集，VALIDATION_ANN_FILE_JSON参考数据集目录下的annotations/instances_val2017.json文件。
> 检查点可在训练过程中生成并保存，其文件夹名称以“ckpt”开头。
> DATA_PATH为coco数据集路径根目录，MINDRECORD_PATH为将要生成的mindrecord文件的路径（需要大约300G的空间）。
> 数据集中图片的数量要和VALIDATION_ANN_FILE_JSON文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 推理结果

推理结果将保存在文件夹名为“eval_log”下的eval.log中。您可在该文件夹的日志中找到如下类似结果。

```text

Evaluate annotation type *bbox*
DONE (t=32.45s).
Accumulating evaluation results...
DONE (t=5.24s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
Loading and preparing results...
DONE (t=3.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=37.03s).
Accumulating evaluation results...
DONE (t=5.26s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609

```

## 性能

### 训练性能

| 参数                  | PointRend                                                 |
| -------------------   | --------------------------------------------------------- |
| 模型版本              | V1                                                        |
| 资源                  | V100                                                      |
| 上传日期              | 2022-08-04                                                |
| MindSpore版本         | 1.8.0                                                     |
| 数据集                | COCO2017                                                  |
| 训练参数              | epoch=12，batch_size=4                                    |
| 优化器                | Momentum                                                  |
| 损失函数              | BCE交叉熵                                                 |
| 速度                  | 单卡：1000毫秒/步；                                       |
| 精度                  | 34%(8p)                                                   |
| 参数（M）             | 313                                                       |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用`train.py`中的随机种子进行权重初始化。

# modelzoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
