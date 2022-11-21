# 目录

- [目录](#目录)
- [RFCN描述](#RFCN描述)
- [模型架构](#模型架构)
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

# RFCN描述

RFCN（Region-based fully convolutional network）是2016年提出的一种目标检测网络架构，它在Faster RCNN的基础上舍弃了全连接层使其适用于不同尺寸的图像，并提出了位置敏感得分图（position sensitive score map）概念，将目标的位置信息融合到roI pooing中，在保持整个网络全卷积的基础上实现了“平移不变性”，并将roi pooling放在卷积层之后，使得网络权重完全共享。

RFCN通过采用全卷积的形式加快了网络的计算速度，并引入位置敏感得分图保证了全卷积网络对物体位置的敏感性，提升了目标检测的效果。

[论文](https://arxiv.org/pdf/1605.06409.pdf)：Dai J ,  Li Y ,  He K , et al. R-FCN: Object Detection via Region-based Fully Convolutional Networks[C]// Advances in Neural Information Processing Systems. Curran Associates Inc.  2016.

# 模型架构

RFCN与Faster-RCNN类似，是一个两阶段的目标检测网络。RFCN以ResNet101为backbone，C4层的输出作为特征训练RPN网络获取roi；然后利用ps roi pooling操作对图像的特征和边界框进行池化得到最后的结果。

# 数据集

使用的数据集：[COCO 2014](<https://pjreddie.com/projects/coco-mirror/>)

- 数据集大小：19G
    - 训练集：13G，82783个图像  
    - 验证集：6G，40504个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

# 环境要求

- 硬件（GPU）

    - 使用GPU处理器来搭建硬件环境。

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO 2014。

- 本示例默认使用COCO 2014作为训练数据集，您也可以使用自己的数据集。

    1. 若使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool，opencv-python。

        ```python
        pip install Cython

        pip install pycocotools

        pip install opencv-python
        ```

        根据模型运行需要，对应地在`default_config.yaml`中更改COCO_ROOT和其他需要的设置。目录结构如下：

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2014.json
            └─instance_val2014.json
          ├─val2014
          └─train2014
        ```

    2. 若使用自己的数据集，**执行脚本时选择数据集为other。将数据集信息整理成TXT文件，每行内容如下：

       ```txt
       train2014/0000001.jpg 0,259,401,459,7,0 35,28,324,201,2,0 0,30,59,80,2,0
       ```

       每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class,is_crowd]格式的框,类和是否是一群物体的信息。从`image_dir`（数据集目录）图像路径以及`anno_path`（TXT文件路径）的相对路径中读取图像。`image_dir`和`anno_path`可在`default_config.yaml`中设置。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练模型是在ImageNet2012上训练的ResNet-101检查点。你可以使用ModelZoo中 [resnet101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) 脚本来训练, 然后使用src/convert_checkpoint.py把训练好的resnet101的权重文件转换为可加载的权重文件。也可以使用mindspore官方提供的[resnet101预训练模型](https://download.mindspore.cn/model_zoo/r1.3/resnet101_ascend_v130_imagenet2012_official_cv_bs32_acc78.58/)进行训练
3. BACKBONE_MODEL是通过modelzoo中的[resnet101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)脚本训练的。PRETRAINED_MODEL是经过转换后的权重文件。VALIDATION_JSON_FILE为标签文件。CHECKPOINT_PATH是训练后的检查点文件。

## 在GPU上运行

```shell
# 权重文件转换
python -m src.convert_checkpoint --ckpt_file=[BACKBONE_MODEL]

# 单机训练
bash run_standalone_train_gpu.sh [DEVICE_ID] [PRETRAINED_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)

# 分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)

# 评估
bash run_eval_gpu.sh [DEVICE_ID] [ANNO_PATH] [CHECKPOINT_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)
```

# 脚本说明

## 脚本及样例代码

```shell
.
└─rfcn
  ├─README_CN.md                     // RFCN相关说明
  ├─scripts
    ├─run_standalone_train_gpu.sh    // GPU单机shell脚本
    ├─run_distribute_train_gpu.sh    // GPU分布式shell脚本
    └─run_eval_gpu.sh                // GPU评估shell脚本
  ├─src
    ├─rfcn
      ├─__init__.py                  // init文件
      ├─anchor_generator.py          // 锚点生成器
      ├─bbox_assign_sample.py        // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py // 第二阶段采样器
      ├─rfcn_resnet.py               // RFCN网络
      ├─proposal_generator.py        // 候选生成器
      ├─rfcn_loss.py                 // RFCN的loss网络
      ├─resnet.py                    // 骨干网络
      └─rpn.py                       // 区域候选网络
    ├─dataset.py                     // 创建并处理数据集
    ├─lr_schedule.py                 // 学习率生成器
    ├─network_define.py              // RFCN网络定义
    ├─util.py                        // 例行操作
    ├─eval_util.py                   // 计算精度用到的方法
    ├─detecteval.py                  // 评估时用到的方法
    └─model_utils
      ├─config.py                    // 获取.yaml配置参数
      ├─device_adapter.py            // 获取云上id
      ├─local_adapter.py             // 获取本地id
      └─moxing_adapter.py            // 云上数据准备
  ├─default_config.yaml              // 默认训练配置
  ├─config_standalone_gpu.yaml       // 单卡训练配置
  ├─config_distribute_gpu.yaml       // 八卡训练配置
  ├─eval.py                          // 评估脚本
  └─train.py                         // 训练脚本
```

## 训练过程

### 用法

#### 在GPU上运行

```shell
# GPU单机训练
bash run_standalone_train_gpu.sh [DEVICE_ID] [PRETRAINED_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)

# GPU分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)
```

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“run_distribute_train”开头。您可以在log查看训练信息，如下所示。

```log
# 分布式训练结果（8P）
epoch time: 2176560.423 ms, per step time: 420.673 ms
epoch time: 2176562.112 ms, per step time: 420.673 ms
epoch time: 2176555.964 ms, per step time: 420.672 ms
epoch time: 2176560.564 ms, per step time: 420.673 ms
epoch time: 2176562.216 ms, per step time: 420.673 ms
epoch time: 2176560.212 ms, per step time: 420.673 ms
epoch time: 2176561.430 ms, per step time: 420.673 ms
epoch time: 2176530.907 ms, per step time: 420.667 ms
```

在对应的loss_0.log中查看实时的损失值，如下所示。

```log
56943 epoch: 26 step: 5168 total_loss: 0.36969
56943 epoch: 26 step: 5169 total_loss: 0.47171
56944 epoch: 26 step: 5170 total_loss: 0.44770
56944 epoch: 26 step: 5171 total_loss: 0.51082
56944 epoch: 26 step: 5172 total_loss: 0.64440
56945 epoch: 26 step: 5173 total_loss: 0.61452
56945 epoch: 26 step: 5174 total_loss: 0.24274
```

## 评估过程

### 用法

#### 在GPU上运行

```shell
# GPU评估
bash run_eval_gpu.sh [DEVICE_ID] [ANNO_PATH] [CHECKPOINT_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)
```

> 第一次评估时会先生成Mindrecord文件，需耐心等待
>
> 在训练过程中生成检查点。
>
> 数据集中图片的数量要和VALIDATION_JSON_FILE文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志文件log中找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.515
```

# 模型描述

## 性能

### 训练性能

| 参数 |GPU |
| -------------------------- |----------------------------------------------------------- |
| 模型版本 |V1 |
| 资源 |V100-PCIE 16G            |
| 上传日期 | 2022/4/11 |
| MindSpore版本 |1.6.0 |
| 数据集 |COCO 2014 |
| 训练参数 |epoch=26, batch_size=2 |
| 优化器 |SGD |
| 损失函数 |Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 1卡：420毫秒/步；8卡：420毫秒/步 |
| 总时间 |1卡：130小时；8卡：15.71小时 |
| 参数(M) |670M |
| 脚本 | [RFCN脚本](https://gitee.com/mindspore/models/tree/master/research/cv/rfcn) |

### 评估性能

| 参数 |GPU |
| ------------------- | ------------------- |
| 模型版本 |V1 |
| 资源 |V100-PCIE 16G  |
| 上传日期 |2022/4/11 |
| MindSpore版本 |1.6.0 |
| 数据集 |COCO2014 |
| batch_size | 2 |
| 输出 |mAP |
| 准确率 | 交并比（IoU）=0.50:0.95 27.3% |
| 推理模型 |670M（.ckpt文件） |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
