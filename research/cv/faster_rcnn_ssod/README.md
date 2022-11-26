# FasterRcnn-ssod

## 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [论文](#论文)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度（Ascend）)
    - [半监督](#半监督)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [GPU处理器环境运行](#gpu处理器环境运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
        - [结果](#结果-1)
- [性能](#性能)
    - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## 概述

faster_rcnn：Faster R-CNN提出，基于区域检测器（如Fast R-CNN）的卷积特征映射也可以用于生成区域候选。在这些卷积特征的顶部构建区域候选网络（RPN）需要添加一些额外的卷积层（与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选），同时输出每个位置的区域边界和客观性得分。因此，RPN是一个全卷积网络，可以端到端训练，生成高质量的区域候选，然后送入Fast R-CNN检测。

faster_rcnn_ssod：通过半监督+主动学习的方式，进行fasterrcnn检测网络的训练，在有限的标注数据上，充分利用大量无标注数据，进行伪标签的学习，提升模型能力。用于解决实际用户在标注预算有限的情况下，利用主动学习，挑选最具价值数据进行标注，并且补充大量无/少成本的无标注数据，进行半监督学习，达到利用少量标注数据+大量无标注数据，得到和全量标注一样的模型性能。

## 论文

1. [论文](<https://arxiv.org/abs/1506.01497.pdf>)：Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks."IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6)."
2. [论文](<https://openreview.net/pdf?id=MJIve1zgR_>)：Liu YC, Ma YC, He ZJ, , et al. UNBIASED TEACHER FOR SEMI-SUPERVISED OBJECT DETECTION

## 特性

### 混合精度（Ascend）

采用[混合精度](<https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6>)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。 以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子

### 半监督

本实验模型以[mindspore fasterrcnn](<https://gitee.com/mindspore/models/tree/master/research/cv/faster_rcnn_ssod>)模型为基础，在有限的标注数据上，充分利用大量无标注数据，进行伪标签的学习，提升模型能力。

## 数据集

使用的数据集：[FaceMaskDetection](<https://www.kaggle.com/datasets/andrewmvd/face-mask-detection>)

- 数据集大小：417M
    - 训练集：415M，853个图像
    - 标注集：1.6M，实例，字幕，person_keypoints等

- 数据格式：图像和json文件
    - 需要将xml格式数据转换为coco格式数据

```text
1. 数据划分,splitdata.py，将下载的原始数据解压放在/data目录下，包含images图像目录和annotations标注目录（格式为XML格式），在执行split_data.py之后，会对原始数据进行划分，得到训练集/data/train/和验证集/data/val/
此时/data目录结构如下
└─data
 ├─train
   ├─images
   └─annotations
 ├─val
   ├─images
   └─annotations
 ├─images
 └─annotations
2. xml转coco,xml2coco.py,第1步对数据划分后，标注格式是XML格式，需要将XML转成COCO格式
2.0) 在/data目录下新建face_detction目录，在facedetection新建annotations目录
2.1) 生成COCO格式训练集，python xml2coco.py --data_path /data/train/ --save_path /data/face_detection/annotations/instances_train2017.json
2.2) 生存COCO格式验证集, python xml2coco.py --data_path /data/val/ --save_path /data/face_detection/annotations/instances_val2017.json
2.3) 将/data/train/images 复制到/data/face_detection/下，并重命名为train2017；将/data/val/images 复制到/data/face_detection/下，并重命名为val2017
3. 随机筛选25%的数据作为有标数据，剩余数据清楚标签，作为无标数据，生成新的训练json
3.0) python random_select.py --ann_file /data/face_detection/annotations/instances_train2017.json --label_ratio 25 --output_ann_file /data/face_detection/annotations/instances_train2017_25.json
最终数据及目录结构如下,在训练和推理中主要涉及face_detction目录
└─data
 ├─train
   ├─images
       └─*.png
   └─annotations
       └─*.xml
 ├─val
   ├─images
       └─*.png
   └─annotations
       └─*.xml
 ├─images
 ├─annotations
 └─face_detection
   ├─train2017
       └─*.png
   ├─val2017
       └─*.png
   └─annotations
       ├─instances_train2017.json
       ├─instances_train2017_25.json
       └─instances_val2017.json
```

## 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

- 10%随机数据生成

```text
python random_select.py --ann_file [ORI_ANNO_JSON]
                        --label_ratio [RATIO]
                        --output_ann_file [RATIO_ANNO_JSON]
```

- Ascend处理器环境运行

```text
# 使用8卡训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_ROOT] [TRAIN_ANN_FILE] [ORI_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)

# 使用单卡训练
bash run_standalone_train_ascend.sh [DEVICE_ID] [DATA_ROOT] [TRAIN_ANN_FILE] [ORI_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)
```

- GPU处理器环境运行

```text
# 使用8卡训练
bash run_distribute_train_gpu.sh [DATA_ROOT] [TRAIN_ANN_FILE] [ORI_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)

# 使用单卡训练
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_ROOT] [ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)
```

## 脚本说明

### 脚本及样例代码

```shell
├── eval.py                                         # 评估模型
├── export_fasterrcnn.py                            # 推理模型转换
├── infer.py                                        # 模型在线推理
├── pick_select.py                                  # 挑选价值数据
├── random_select.py                                # 随机挑选数据，全标签数据集构造部分无标数据集
├── README.md
├── scripts
│   ├── run_distribute_train_ascend.sh              # Ascend多卡一键式训练脚本
│   ├── run_distribute_train_gpu.sh                 # GPU多卡一键式训练脚本
│   ├── run_distribute_train_model_ascend.sh        # Ascend多卡训练模型脚本
│   ├── run_distribute_train_model_gpu.sh           # GPU多卡训练模型脚本
│   ├── run_eval_ascend.sh                          # Ascend评估模型
│   ├── run_eval_gpu.sh                             # GPU评估模型
│   ├── run_standalone_train_ascend.sh              # Ascend单卡一键式训练脚本
│   ├── run_standalone_train_gpu.sh                 # GPU单卡一键式训练脚本
│   ├── run_standalone_train_model.sh               # 模型训练脚本
│   └── select_valuable_sample.sh
├── sorted_values.py                                # 无标数据价值排序
├── src
│   ├── config.py                                   # 参数配置
│   ├── dataset.py                                  # 数据集
│   ├── eval_utils.py                               # 评估常用工具
│   ├── FasterRcnn                                  # fasterrcnn训练模型
│   │   ├── anchor_generator.py
│   │   ├── bbox_assign_sample.py
│   │   ├── bbox_assign_sample_stage2.py
│   │   ├── faster_rcnn_r50.py
│   │   ├── faster_rcnn_r50_with_scores.py
│   │   ├── fpn_neck.py
│   │   ├── __init__.py
│   │   ├── proposal_generator.py
│   │   ├── rcnn.py
│   │   ├── resnet50.py
│   │   ├── roi_align.py
│   │   └── rpn.py
│   ├── FasterRcnnInfer                              # fasterrcnn推理模型
│   │   ├── faster_rcnn_r50.py
│   │   ├── __init__.py
│   │   ├── rcnn.py
│   │   └── rpn.py
│   ├── __init__.py
│   ├── lr_schedule.py                               # 学习率
│   ├── network_define.py                            # 半监督网络定义
│   ├── split_data.py                                # 训练、评估数据集分割
│   ├── utils.py                                     # 训练常用工具
│   └── xml2coco.py                                  # xml格式标签转换为coco标签
└── train.py                                         # 模型训练

```

### 脚本参数

```python
device_target = "Ascend"                              # 设备类型

# ==============================================================================
# config
img_width = 1280    # 1280                            # 模型输入宽
img_height = 768   # 768                              # 模型输出高
keep_ratio = True                                     # 数据预处理缩放是否等比
flip_ratio = 0.5                                      # 图片反转比例

# anchor                                              # 模型结构参数
feature_shapes = [
    [img_height // 4, img_width // 4],
    [img_height // 8, img_width // 8],
    [img_height // 16, img_width // 16],
    [img_height // 32, img_width // 32],
    [img_height // 64, img_width // 64],
]
anchor_scales = [8]
anchor_ratios = [0.5, 1.0, 2.0]
anchor_strides = [4, 8, 16, 32, 64]
num_anchors = 3

# resnet
resnet_block = [3, 4, 6, 3]
resnet_in_channels = [64, 256, 512, 1024]
resnet_out_channels = [256, 512, 1024, 2048]

# fpn
fpn_in_channels = [256, 512, 1024, 2048]
fpn_out_channels = 256
fpn_num_outs = 5

# rpn
rpn_in_channels = 256
rpn_feat_channels = 256
rpn_loss_cls_weight = 1.0
rpn_loss_reg_weight = 1.0
rpn_cls_out_channels = 1
rpn_target_means = [0., 0., 0., 0.]
rpn_target_stds = [1.0, 1.0, 1.0, 1.0]

# bbox_assign_sampler
neg_iou_thr = 0.3
pos_iou_thr = 0.7
min_pos_iou = 0.3
num_bboxes = num_anchors * sum([lst[0] * lst[1] for lst in feature_shapes])
num_gts = 128
num_expected_neg = 256
num_expected_pos = 128

# proposal
activate_num_classes = 2
use_sigmoid_cls = True

# roi_align
class RoiLayer:
    type = 'RoIAlign'
    out_size = 7
    sample_num = 2

roi_layer = RoiLayer()
roi_align_out_channels = 256
roi_align_featmap_strides = [4, 8, 16, 32]
roi_align_finest_scale = 56
roi_sample_num = 640

# bbox_assign_sampler_stage2
neg_iou_thr_stage2 = 0.5
pos_iou_thr_stage2 = 0.5
min_pos_iou_stage2 = 0.5
num_bboxes_stage2 = 2000
num_expected_pos_stage2 = 128
num_expected_neg_stage2 = 512
num_expected_total_stage2 = 512

# rcnn
rcnn_num_layers = 2
rcnn_in_channels = 256
rcnn_fc_out_channels = 1024
rcnn_loss_cls_weight = 1
rcnn_loss_reg_weight = 1
rcnn_target_means = [0., 0., 0., 0.]
rcnn_target_stds = [0.1, 0.1, 0.2, 0.2]

# train proposal
rpn_proposal_nms_across_levels = False
rpn_proposal_nms_pre = 2000
rpn_proposal_nms_post = 2000
rpn_proposal_max_num = 2000
rpn_proposal_nms_thr = 0.7
rpn_proposal_min_bbox_size = 0

# test proposal
rpn_nms_across_levels = False
rpn_nms_pre = 1000
rpn_nms_post = 1000
rpn_max_num = 1000
rpn_nms_thr = 0.7
rpn_min_bbox_min_size = 0
test_score_thr = 0.05
test_iou_thr = 0.5
test_max_per_img = 100
test_batch_size = 2

rpn_head_use_sigmoid = True
rpn_head_weight = 1.0

# LR
lr_schedule = "step"                                  # 学习率类型
milestones = [19990, 19995]                           # 突变位置
base_lr = 0.005                                       # 基础学习率
gamma = 0.1                                           # 学习率缩放比例
warmup_ratio = 0.0625                                 # 预热比例
warmup_step = 500                                     # 预热步数

# train
global_seed = 10                                      # 随机数种子
resume = False                                        # 断点续训
run_distribute = False                                # 分布式
batch_size = test_batch_size                          # batch size
loss_scale = 256                                      # loss缩放
momentum = 0.9                                        # 动量优化器
weight_decay = 0.0001                                 # 衰减权重
save_checkpoint = True                                # 保存模型
save_checkpoint_path = "./outputs/"                   # 权重保存路径
save_checkpoint_interval = 1000                       # 权重保存间隔

# semi-train
ema_keep_rate = 0.9996                                # ema更新比例
unsup_loss_weight = 4.0                               # ？
bbox_threshold = 0.7                                  # bbox置信度
max_iter = 30000                                      # 最大训练步数
start_iter = 0                                        # 起始训练步数
burn_up_iter = 2000                                   # burn_up阶段训练步数
teacher_update_iter = 1                               # teacher模型更新频率
print_interval_iter = 100                             # 打印间隔

# dataset
num_parallel_workers = 4                              # 数据加载时的并行工程数
num_classes = 4                                       # 类别数，需要比数据集的实际类别数+1
train_img_dir = "/home/datasets/FaceMaskDetectionDataset/train/images"
                                                      # 训练集图片路径
train_ann_file = "/home/datasets/FaceMaskDetectionDataset/train/annotations/instances_train2017_25.json"
                                                      # 训练集标签路径
eval_img_dir = "/home/datasets/FaceMaskDetectionDataset/val/images"
                                                      # 评估集图片路径
eval_ann_file = "/home/datasets/FaceMaskDetectionDataset/val/annotations/instances_val2017.json"
                                                      # 评估集标签路径

# train.py FasterRcnn training
pre_trained = None                                    # 预训练权重
filter_prefix = ["fpn_ncek", "rpn_with_loss", "rcnn"] # 模型不加载的预训练权重
pre_trained_teacher = None                            # teacher预训练权重
device_id = 0                                         # 训练device id

# eval.py FasterRcnn evaluation
checkpoint_path = None                                # 评估权重
eval_output_dir = "./outputs/"                        # 评估输出目录
eval_device_id = 0                                    # 评估device id
```

## 训练过程

### 用法

#### Ascend处理器环境运行

`run_distribute_train_ascend.sh`执行脚本中的主要内容如下：

```text
# 第一轮训练，25%有标数据，训练出模型model_0.ckpt
echo "=============================================================================================================="
echo "train in first stage"
bash run_distribute_train_model_ascend.sh $RANK_TABLE_FILE $DATA_ROOT $ANN_FILE $OUTPUT_DIR $PRE_TRAINED

# 使用第一轮训练的权重model_0.ckpt对无标数据进行推理，将推理结果进行价值排序，筛选出前35%的数据作为价值数据，进行标注
# 本实验中通过将完整数据集变为无标数据进行训练，重新标注时将原先有标数据标注写回，模拟重新标注操作
# 一键式脚本中由于无法获取一轮训练的精度信息，默认采用最后一个权重进行推理，实际应用过程中推荐使用最优模型进行推理以获取更优的精度结果
echo "=============================================================================================================="
echo "select top 35% valuable unlabel data"
NEW_JSON=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $OUTPUT_DIR/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_JSON $DEVICE_TARGET 0

# 第二轮训练，基础25%有标数据+35%价值数据，训练出最终的输出模型
echo "=============================================================================================================="
echo "train in second stage"
bash run_distribute_train_model_ascend.sh $RANK_TABLE_FILE $DATA_ROOT $NEW_JSON $OUTPUT_DIR $PRE_TRAINED
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

`select_valuable_sample.sh`执行脚本中的主要内容如下：

```text
# 模型推理，将无标数据进行推理，生成结果json用于后续价值排序
python infer_combine.py --checkpoint_path=$pretrained \
                        --device_target=$device_target \
                        --eval_output_dir="./"

# 将每个bbox的score按图片为单位进行整合，并进行价值排序
infer_json="./infer_results.json"
python sorted_values.py --infer_json=${infer_json}

top_value_json="./top_value_data.json"

# 筛选前35%的价值数据，从初始json中获取该部分标注，与数据准备时生成的25%数据量的json合并，生成60%数据量的新json
python pick_select.py --ann_file=$ori_train_ann_file \
                      --combine_ann_file=$selected_ann_file \
                      --pick_ratio=60 \
                      --top_value_file=$top_value_json \
                      --output_ann_file=$valuable_ann_file
```

`run_standalone_train_ascend.sh`执行脚本中的内容下：

```text
# 第一轮训练，25%有标数据，训练出模型model_0.ckpt
echo "=============================================================================================================="
echo "train in first stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $ANN_FILE $FIRST_STAGE_OUTPUT $PRE_TRAINED

# 使用第一轮训练的权重model_0.ckpt对无标数据进行推理，将推理结果进行价值排序，筛选出前35%的数据作为价值数据，进行标注
# 本实验中通过将完整数据集变为无标数据进行训练，重新标注时将原先有标数据标注写回，模拟重新标注操作
# 一键式脚本中由于无法获取一轮训练的精度信息，默认采用最后一个权重进行推理，实际应用过程中推荐使用最优模型进行推理以获取更优的精度结果
echo "=============================================================================================================="
echo "select top 25% valuable unlabel data"
NEW_ANNO=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $FIRST_STAGE_OUTPUT/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_ANNO $DEVICE_TARGET $DEVICE_ID

# 第二轮训练，基础25%有标数据+35%价值数据，训练出最终的输出模型
echo "=============================================================================================================="
echo "train in second stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $NEW_ANNO $SECOND_STAGE_OUTPUT $PRE_TRAINED
```

#### GPU处理器环境运行

`run_distribute_train_gpu.sh`执行脚本中的主要内容如下：

```text
# 第一轮训练，25%有标数据，训练出模型model_0.ckpt
echo "=============================================================================================================="
echo "train in first stage"
bash run_distribute_train_model_ascend.sh $DATA_ROOT $ANN_FILE $OUTPUT_DIR $PRE_TRAINED

# 使用第一轮训练的权重model_0.ckpt对无标数据进行推理，将推理结果进行价值排序，筛选出前35%的数据作为价值数据，进行标注
# 本实验中通过将完整数据集变为无标数据进行训练，重新标注时将原先有标数据标注写回，模拟重新标注操作
echo "=============================================================================================================="
echo "select top 35% valuable unlabel data"
NEW_JSON=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $OUTPUT_DIR/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_JSON $DEVICE_TARGET 0

# 第二轮训练，基础25%有标数据+35%价值数据，训练出最终的输出模型
echo "=============================================================================================================="
echo "train in second stage"
bash run_distribute_train_model_ascend.sh $DATA_ROOT $NEW_JSON $OUTPUT_DIR $PRE_TRAINED
```

`run_standalone_train_gpu.sh`执行脚本中的内容下：

```text
# 第一轮训练，25%有标数据，训练出模型model_0.ckpt
echo "=============================================================================================================="
echo "train in first stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $ANN_FILE $FIRST_STAGE_OUTPUT $PRE_TRAINED

# 使用第一轮训练的权重model_0.ckpt对无标数据进行推理，将推理结果进行价值排序，筛选出前35%的数据作为价值数据，进行标注
# 本实验中通过将完整数据集变为无标数据进行训练，重新标注时将原先有标数据标注写回，模拟重新标注操作
# 一键式脚本中由于无法获取一轮训练的精度信息，默认采用最后一个权重进行推理，实际应用过程中推荐使用最优模型进行推理以获取更优的精度结果
echo "=============================================================================================================="
echo "select top 25% valuable unlabel data"
NEW_ANNO=$FIRST_STAGE_OUTPUT/new_ann_file.json
bash select_valuable_sample.sh $FIRST_STAGE_OUTPUT/model_0.ckpt $ORI_ANN_FILE $ANN_FILE $NEW_ANNO $DEVICE_TARGET $DEVICE_ID

# 第二轮训练，基础25%有标数据+35%价值数据，训练出最终的输出模型
echo "=============================================================================================================="
echo "train in second stage"
bash run_standalone_train_model.sh $DEVICE_TARGET $DEVICE_ID $DATA_ROOT $NEW_ANNO $SECOND_STAGE_OUTPUT $PRE_TRAINED
```

#### 结果

训练checkpoint存在`EXP_DIR/first_stage`和`EXP_DIR/second_stage`中，你可以从`scripts/train*/device0/`中找到log.txt日志，里面有如下结果：

```shell
2022-11-10 11:00:57,009.009 INFO rank_id:0, step: 0, avg loss: 4.33355, rpn_cls_loss: 1.41007, rpn_reg_loss: 0.01075, rcnn_cls_loss: 2.81218, rcnn_reg_loss: 0.10054, iter_time: 438.814, data_time: 2.134, overflow: 1.000000, scaling_sens: 8388608.000000, lr: 0.000313
2022-11-10 11:01:27,701.701 INFO rank_id:0, step: 100, avg loss: 1.59982, rpn_cls_loss: 0.69614, rpn_reg_loss: 0.07560, rcnn_cls_loss: 0.74525, rcnn_reg_loss: 0.08283, iter_time: 0.305, data_time: 0.088, overflow: 0.110000, scaling_sens: 100024.320000, lr: 0.001250
2022-11-10 11:02:27,929.929 INFO rank_id:0, step: 200, avg loss: 0.99250, rpn_cls_loss: 0.23399, rpn_reg_loss: 0.07190, rcnn_cls_loss: 0.39947, rcnn_reg_loss: 0.28715, iter_time: 0.600, data_time: 0.375, overflow: 0.050000, scaling_sens: 1066.240000, lr: 0.002187
2022-11-10 11:03:35,032.032 INFO rank_id:0, step: 300, avg loss: 0.87582, rpn_cls_loss: 0.12702, rpn_reg_loss: 0.05632, rcnn_cls_loss: 0.39438, rcnn_reg_loss: 0.29810, iter_time: 0.667, data_time: 0.454, overflow: 0.050000, scaling_sens: 51.360000, lr: 0.003125
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
bash run_eval_ascned.sh [DEVICE_ID] [CKPT_PATH] [EVAL_ROOT] [EVAL_ANN_FILE]
```

#### GPU处理器环境运行

```bash
bash run_eval_gpu.sh [DEVICE_ID] [CKPT_PATH] [EVAL_ROOT] [EVAL_ANN_FILE]
```

### 结果

评估结果保存在`scripts/eval_1p_*/device*/`路径下，结果存储在`log.txt`中。评估完成后，可在日志中找到如下结果：

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.878
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.650
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.780
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.253
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.808
```

## 性能

### 评估性能

| 参数           | Ascend                                  | GPU                                   |
|---------------|------------------------------------------|---------------------------------------|
| 模型版本       | Fasterrcnn                               | Fasterrcnn                            |
| 资源           | Ascend910                                | GPU                                   |
| 上传日期       | 2022/11/08                               | 2022/11/08                            |
| Mindspore版本  | 1.7.0                                    | 1.7.0                                 |
| 数据集         | FaceMaskDetection                        | FaceMaskDetection                     |
| 训练参数       | max_iter=2000, lr=0.005, batch_size=2    | max_iter=2000, lr=0.005, batch_size=2 |
| 优化器         | SGD                                      | SGD                                   |
| 损失函数       | 交叉熵                                    | 交叉熵                                |
| 速度           | 2.375s/step(1卡)                         | 1.521s/step(1卡)                      |
| 总时长         | 12h\*2                                   | 8h\*2                                 |
| 参数(M)        | 158                                      | 158                                   |
| 配置文件       | [链接](https://gitee.com/mindspore/models/blob/master/research/cv/faster_rcnn_ssod/src/config.py) | [链接](https://gitee.com/mindspore/models/blob/master/research/cv/faster_rcnn_ssod/src/config.py) |

## 随机情况说明

在`train.py`中，使用了mindspore的set_seed接口所使用的随机种子。

## ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
