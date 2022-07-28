# 目录

<!-- TOC -->

- [目录](#目录)
- [TextFuseNet概述](#textfusenet概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
        - [训练结果](#训练结果)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [评估结果](#评估结果)
    - [模型导出](#模型导出)
    - [推理过程](#推理过程)
        - [使用方法](#使用方法)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# TextFuseNet概述

自然场景中任意形状文本检测是一项极具挑战性的任务，与现有的仅基于有限特征表示感知文本的文本检测方法不同，本文提出了一种新的框架，即 TextFuseNet ，以利用融合的更丰富的特征进行文本检测。该算法用三个层次的特征来表示文本，字符、单词和全局级别，然后引入一种新的文本融合技术融合这些特征，来帮助实现鲁棒的任意文本检测。另外提出了一个弱监督学习机制，可以生成字符级别的标注，在缺乏字符级注释的数据集情况下也可以进行训练。

[论文](https://www.ijcai.org/Proceedings/2020/72)："TextFuseNet: Scene Text Detection with Richer Fused Features"

# 数据集

- [TotalText](https://github.com/cs-chan/Total-Text-Dataset/) 是一个综合的任意形状的文本数据集,用于场景文本识别。 Total-Text 包含 1255 个训练图像和 300 个测试图像。 所有图像都用词级的多边形标注。

# 环境要求

- 硬件（昇腾处理器）
    - 采用昇腾处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 获取基础镜像
    - [Ascend Hub](ascend.huawei.com/ascendhub/#/home)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

- 第三方库

```bash
mindspore==1.5.0
mindspore_ascend==1.3.0
mmcv==0.2.14
numpy==1.21.0rc1
opencv_python==4.5.1.48
Pillow==8.4.0
pycocotools==2.0.0
PyYAML==6.0
Shapely==1.5.9
```

# 快速入门

1. 下载TotalText数据集。

2. 在`config.py`中修改COCO_ROOT及设置其他参数。参考目录结构如下：

    ```text
    .
    └─data
      ├─annotations
        ├─totaltext_train.json
        └─totaltext_test.json
      ├─train
      └─test
    ```

3. 执行训练脚本。
    数据集准备完成后，按照如下步骤开始训练：

    ```text
    # 分布式训练
    sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_CKPT]

    # 单机训练
    sh run_standalone_train.sh [PRETRAINED_CKPT]
    ```

    注：
    1. 为加快数据预处理速度，MindSpore提供了MindRecord数据格式。因此，训练前首先需要生成基于TotalText数据集的MindRecord文件。
    2. 进行分布式训练前，需要提前创建JSON格式的[hccl配置文件](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。
    3. PRETRAINED_CKPT是一个ResNet101检查点，通过ImageNet2012训练。你可以使用ModelZoo中 [resnet101](https://download.mindspore.cn/model_zoo/r1.3/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34/) 脚本来训练, 然后使用src/convert_checkpoint.py把训练好的resnet101的权重文件转换为可加载的权重文件。
    4.字符级别的标注，首先使用Synthtext数据集训练MaskRCNN得到模型M,然后使用M对TotalText字符级别标注进行补全。将数据集标注转化为 [COCO](https://cocodataset.org/) 格式

4. 执行评估脚本。
   训练结束后，按照如下步骤启动评估：

   ```bash
   # 评估
   sh run_eval.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
   ```

   注：
   1. VALIDATION_JSON_FILE是用于评估的标签JSON文件。

5. 执行推理脚本。
   训练结束后，按照如下步骤启动推理：

   ```bash
   # 评估
   sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
   ```

   注：
   1. MINDIR_PATH是在910上使用export脚本导出的模型。
   2. ANN_FILE_PATH是推理使用的标注文件。

# 脚本说明

## 脚本和样例代码

```shell
.
└─TextFuseNet
  ├─README.md                             # README
  ├─ascend310_infer                       #实现310推理源代码
  ├─scripts                               # shell脚本
    ├─run_standalone_train.sh             # 单机模式训练（单卡）
    ├─run_distribute_train.sh             # 并行模式训练（8卡）
    ├─run_infer_310.sh                    # Ascend推理shell脚本
    └─run_eval.sh                         # 评估
  ├─src
    ├─textfusenet
      ├─__init__.py
      ├─anchor_generator.py               # 生成基础边框锚点
      ├─bbox_assign_sample.py             # 过滤第一阶段学习中的正负边框
      ├─bbox_assign_sample.py             # 过滤第二阶段学习中的正负边框
      ├─text_fuse_net_r101.py             # TextFUseNet主要网络架构
      ├─fpn_neck.py                       # FPN网络
      ├─proposal_generator.py             # 基于特征图生成候选区域
      ├─rcnn_cls.py                       # RCNN边框回归分支
      ├─rcnn_mask.py                      # RCNN掩码分支
      ├─rcnn_seg.py                       # 全局分割分支
      ├─mutil_fuse_path.py                # 多路特征融合模块
      ├─resnet101.py                      # 骨干网
      ├─roi_align.py                      # 兴趣点对齐网络
      └─rpn.py                            # 区域候选网络
    ├─convert_checkpoint.py               # 转换预训练checkpoint文件
    ├─dataset.py                          # 数据集工具
    ├─lr_schedule.py                      # 学习率生成器
    ├─network_define.py                   # TextFuseNet的网络定义
    ├─util.py                             # 例行操作
    └─model_utils
      ├─config.py                         # 训练配置
      ├─device_adapter.py                 # 获取云上id
      ├─local_adapter.py                  # 获取本地id
      └─moxing_adapter.py                 # 参数处理
  ├─default_config.yaml                   # 训练参数配置文件
  ├─mindspore_hub_conf.py                 # MindSpore hub接口
  ├─export.py                             #导出 AIR,MINDIR,ONNX模型的脚本
  ├─eval.py                               # 评估脚本
  ├─postprogress.py                       #310推理后处理脚本
  └─train.py                              # 训练脚本
```

## 脚本参数

### 训练脚本参数

```bash
# 分布式训练
用法：sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]

# 单机训练
用法：sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### 参数配置

```bash
"img_width":1280,          # 输入图像宽度
"img_height":768,          # 输入图像高度

# 数据增强随机阈值
"keep_ratio": True,
"flip_ratio":0.5,
"photo_ratio":0.5,
"expand_ratio":1.0,

"max_instance_count":128, # 各图像的边框最大值
"mask_shape": (28, 28),   # rcnn_mask中掩码的形状

# 锚点
"feature_shapes": [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)], # FPN特征图的形状
"anchor_scales": [8],                                                    # 基础锚点区域
"anchor_ratios": [0.5, 1.0, 2.0],                                        # 基础锚点高宽比
"anchor_strides": [4, 8, 16, 32, 64],                                    # 各特征图层的步长大小
"num_anchors": 3,                                                        # 各像素的锚点数

# ResNet
"resnet_block": [3, 4, 6, 3],                                            # 各层区块数
"resnet_in_channels": [64, 256, 512, 1024],                              # 各层输入通道大小
"resnet_out_channels": [256, 512, 1024, 2048],                           # 各层输出通道大小

# FPN
"fpn_in_channels":[256, 512, 1024, 2048],                               # 各层输入通道大小
"fpn_out_channels": 256,                                                # 各层输出通道大小
"fpn_num_outs":5,                                                       # 输出特征图大小

# RPN
"rpn_in_channels": 256,                                                 # 输入通道大小
"rpn_feat_channels":256,                                                # 特征输出通道大小
"rpn_loss_cls_weight":1.0,                                              # 边框分类在RPN损失中的权重
"rpn_loss_reg_weight":1.0,                                              # 边框回归在RPN损失中的权重
"rpn_cls_out_channels":1,                                               # 分类输出通道大小
"rpn_target_means":[0., 0., 0., 0.],                                    # 边框编解码方式
"rpn_target_stds":[1.0, 1.0, 1.0, 1.0],                                 # 边框编解码标准

# bbox_assign_sampler
"neg_iou_thr":0.3,                                                      # 交并后负样本阈值
"pos_iou_thr":0.7,                                                      # 交并后正样本阈值
"min_pos_iou":0.3,                                                      # 交并后最小正样本阈值
"num_bboxes":245520,                                                    # 边框总数
"num_gts": 128,                                                         # 地面真值总数
"num_expected_neg":256,                                                 # 负样本数
"num_expected_pos":128,                                                 # 正样本数

# 候选区域
"activate_num_classes":2,                                               # RPN分类中的类数
"use_sigmoid_cls":True,                                                 # 在RPN分类中是否使用sigmoid作为损失函数

# roi_alignj
"roi_layer": dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2), # ROIAlign参数
"roi_align_out_channels": 256,                                                  # ROIAlign输出通道大小
"roi_align_featmap_strides":[4, 8, 16, 32],                                     # ROIAling特征图不同层级的步长大小
"roi_align_finest_scale": 56,                                                   # ROIAlign最佳比例
"roi_sample_num": 640,                                                          # ROIAling层中的样本数

# bbox_assign_sampler_stage2                                                    # 第二阶段边框赋值样本，参数含义类似于bbox_assign_sampler
"neg_iou_thr_stage2":0.5,
"pos_iou_thr_stage2":0.5,
"min_pos_iou_stage2":0.5,
"num_bboxes_stage2":2000,
"num_expected_pos_stage2":128,
"num_expected_neg_stage2":512,
"num_expected_total_stage2":512,

# rcnn                                                                          # 第二阶段的RCNN参数，参数含义类似于FPN
"rcnn_num_layers":2,  
"rcnn_in_channels":256,
"rcnn_fc_out_channels":1024,
"rcnn_mask_out_channels":256,
"rcnn_loss_cls_weight":1,
"rcnn_loss_reg_weight":1,
"rcnn_loss_mask_fb_weight":1,
"rcnn_target_means":[0., 0., 0., 0.],
"rcnn_target_stds":[0.1, 0.1, 0.2, 0.2],
"textfusenet_channels": 256,                                                     # 多路特征融合通道数目

# 训练候选区域
"rpn_proposal_nms_across_levels":False,
"rpn_proposal_nms_pre":2000,                                                  # RPN中NMS前的候选区域数
"rpn_proposal_nms_post":2000,                                                 # RPN中NMS后的候选区域数
"rpn_proposal_max_num":2000,                                                  # RPN中最大候选区域数
"rpn_proposal_nms_thr":0.7,                                                   # RPN中NMS的阈值
"rpn_proposal_min_bbox_size":0,                                               # RPN中边框的最小尺寸

# 测试候选区域                                                                # 部分参数与训练候选区域类似
"rpn_nms_across_levels":False,
"rpn_nms_pre":1000,
"rpn_nms_post":1000,
"rpn_max_num":1000,
"rpn_nms_thr":0.7,
"rpn_min_bbox_min_size":0,
"test_roi_number": 100,
"test_score_thr":0.05,                                                        # 打分阈值
"test_iou_thr":0.5,                                                           # 交并比阈值
"test_max_per_img":100,                                                       # 最大实例数
"test_batch_size":2,                                                          # 批次大小

"rpn_head_loss_type":"CrossEntropyLoss",                                      # RPN中的损失类型
"rpn_head_use_sigmoid":True,                                                  # 是否在RPN中使用sigmoid
"rpn_head_weight":1.0,                                                        # RPN头的损失重量
"mask_thr_binary":0.5,                                                        # 输入RCNN的掩码阈值

# 逻辑回归
"base_lr":0.001,                                                               # 基础学习率
"base_step":1206,                                                            # 逻辑回归发生器中的基础步骤
"total_epoch":201,                                                             # 逻辑回归发生器总轮次
"warmup_step":500,                                                            # 逻辑回归发生器热身步骤
"warmup_mode":"linear",                                                       # 热身模式
"warmup_ratio":1/3.0,                                                         # 热身比
0.9,                                                           # 优化器中的动量

# 训练
"batch_size":1,
"loss_scale":1,
"momentum":0.91,
"weight_decay":1e-4,
"pretrain_epoch_size":0,                                                      # 预训练的轮次
"epoch_size":200,                                                              # 总轮次
"save_checkpoint":True,                                                       # 是否保存检查点
"save_checkpoint_epochs":50,                                                   # 检查点保存间隔
"keep_checkpoint_max":200,                                                     # 检查点最大保存数
"save_checkpoint_path":"./checkpoint",                                        # 检查点所在路径

"mindrecord_dir":"/home/textfusenet/MindRecord_ToTalText_Train",                  # MindRecord文件路径
"coco_root":"/home/textfusenet/",                                                # TotalText根数据集的路径
"train_data_type":"train",                                                # 训练数据集名称
"val_data_type":"test",                                                    # 评估数据集名称
"instance_set":"annotations/instances_{}.json",                               # 注释名称
"coco_classes":('background', 'text', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R','S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v','w', 'x', 'y', 'z'),
"num_classes":64
```

## 训练过程

- 在`config.py`中设置配置项，包括loss_scale、学习率和网络超参。单击[此处](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/dataset_sample.html)获取更多数据集相关信息.

### 训练

- 运行`run_standalone_train.sh`开始TextFuseNet模型的非分布式训练。

```bash
# 单机训练
sh run_standalone_train.sh [PRETRAINED_MODEL]
```

### 分布式训练

- 运行`run_distribute_train.sh`开始TextFuseNet模型的分布式训练。

```bash
sh run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

- Notes

1. 运行分布式任务时要用到由RANK_TABLE_FILE指定的hccl.json文件。您可使用[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成该文件。
2. PRETRAINED_MODEL应该是训练好的ResNet101检查点。如果此参数未设置，网络将从头开始训练。如果想要加载训练好的TextFuseNet检查点，需要对train.py作如下修改：

```python
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if item in ("global_step", "learning_rate") or "rcnn.cls" in item or "rcnn.mask" in item:
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)
        load_param_into_net(opt, param_dict)
```

3. 本操作涉及处理器内核绑定，需要设置`device_num`及处理器总数。若无需此操作，请删除`scripts/run_distribute_train.sh`中的`taskset`

### 训练结果

训练结果将保存在示例路径，文件夹名称以“train”或“train_parallel”开头。您可以在loss_rankid.log中找到检查点文件及如下类似结果。

```bash
# 分布式训练结果（8P）
3893 epoch: 1 step: 151 total_loss: 2.65625
3923 epoch: 2 step: 151 total_loss: 2.40820
3953 epoch: 3 step: 151 total_loss: 2.44922
3983 epoch: 4 step: 151 total_loss: 2.48828
4013 epoch: 5 step: 151 total_loss: 1.35156
4043 epoch: 6 step: 151 total_loss: 1.79297
4073 epoch: 7 step: 151 total_loss: 2.24414
4102 epoch: 8 step: 151 total_loss: 1.33496
4132 epoch: 9 step: 151 total_loss: 0.67822
4162 epoch: 10 step: 151 total_loss: 1.76172
4192 epoch: 11 step: 151 total_loss: 0.90430
4222 epoch: 12 step: 151 total_loss: 1.92773
4252 epoch: 13 step: 151 total_loss: 1.85840
4282 epoch: 14 step: 151 total_loss: 1.33984
4312 epoch: 15 step: 151 total_loss: 1.61719
4342 epoch: 16 step: 151 total_loss: 1.52441
4372 epoch: 17 step: 151 total_loss: 1.34863

```

## 评估过程

### 评估

- 运行`run_eval.sh`进行评估。
- 下载评估代码 [TIOU-metric-python3](https://github.com/PkuDavidGuan/TIoU-metric-python3.git)

1.将代码重命名为eval_code,并放置在scripts目录下  
2.将代码中的total-text-gt.zip放置在scripts目录下  
3.将eval_code/curved_tiou/rrc_evaluation_funcs.py中232和233行注释  
4.执行以下命令

```bash
sh run_eval.sh [VALIDATION_ANN_FILE_JSON] [CHECKPOINT_PATH]
```

> 关于TotalText数据集，VALIDATION_ANN_FILE_JSON参考数据集目录下的annotations/instances_test.json文件。  
> 检查点可在训练过程中生成并保存，其文件夹名称以“train/checkpoint”或“train_parallel*/checkpoint”开头。
> 数据集中图片的数量要和VALIDATION_ANN_FILE_JSON文件中标记数量一致，否则精度结果展示格式可能出现异常。<br>
> [TotalText](https://github.com/cs-chan/Total-Text-Dataset/)

### 评估结果

推理结果将保存在示例路径，文件夹名为“eval”。您可在该文件夹的日志中找到如下类似结果。

```text
num_gt, num_det:  2214 2430
Origin:
recall:  0.8071 precision:  0.8258 hmean:  0.8164

```

## 模型导出

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 选项 ["AIR", "MINDIR"]

## 推理过程

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。目前推理只支持batch_size=1。
在推理前需要下载评估代码 [TIOU-metric-python3](https://github.com/PkuDavidGuan/TIoU-metric-python3.git)  
1.将代码重命名为eval_code,并放置在scripts目录下  
2.将代码中的total-text-gt.zip放置在scripts目录下  
3.将eval_code/curved_tiou/rrc_evaluation_funcs.py中232和233行注释  

```shell
# Ascend310 推理
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```bash
num_gt, num_det:  2214 2422
Origin:
recall:  0.8035 precision:  0.8282 hmean:  0.8157
```

# 模型说明

## 性能

### 训练性能

| 参数                  | TextFuseNet                                                 |
| -------------------   | --------------------------------------------------------- |
| 模型版本              | V1                                                        |
| 资源                  | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8              |
| 上传日期              | 2021-07-05                                                |
| MindSpore版本         | 1.3.0                                                     |
| 数据集                | Totaltext                                                  |
| 训练参数              | epoch=200，batch_size=1                                    |
| 优化器                | SGD                                                       |
| 损失函数              | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss                |
| 速度                  | 单卡：333毫秒/步；8卡: 300毫秒/步                          |
| 总时长                | 单卡：22.4小时；8卡：2.1小时                                |

### 评估性能

| 参数                  | TextFuseNet                      |
| --------------------- | ----------------------------- |
| 模型版本              | V1                            |
| 资源                  | Ascend 910；系统 Euler2.8                    |
| 上传日期              | 2021-10-28                    |
| MindSpore版本         | 1.3.0                         |
| 数据集                | TotalText                      |
| 批次大小              | 1                             |
| 输出                  | mAP                           |
| 精确度                | 交并比（IoU）=0.50 Mask 81.64% |
| 推理模型              | 563M（.ckpt文件）             |

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用`train.py`中的随机种子进行权重初始化。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
