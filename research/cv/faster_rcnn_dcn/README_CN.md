# 目录

- [目录](#目录)
- [Faster R-CNN-DCN描述](#faster-r-cnn-dcn描述)
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

# Faster R-CNN描述

在Faster R-CNN之前，目标检测网络依靠区域候选算法来假设目标的位置，如SPPNet、Fast R-CNN等。研究结果表明，这些检测网络的运行时间缩短了，但区域方案的计算仍是瓶颈。

Faster R-CNN提出，基于区域检测器（如Fast R-CNN）的卷积特征映射也可以用于生成区域候选。在这些卷积特征的顶部构建区域候选网络（RPN）需要添加一些额外的卷积层（与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选），同时输出每个位置的区域边界和客观性得分。因此，RPN是一个全卷积网络，可以端到端训练，生成高质量的区域候选，然后送入Fast R-CNN检测。

[论文](https://arxiv.org/abs/1506.01497)：   Ren S , He K , Girshick R , et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 39(6).

# Deformable Convolution描述

近年来，卷积神经网络在计算机视觉领域取得了飞速的发展，在图像识别、语义分割、目标检测领域都有很好的应用，然而由于卷积神经网络固定的几何结构，导致对几何形变的建模受到了限制，所以提出了Deformable Convolution，可变形卷积。

可变形卷积在卷积层中对卷积核增加额外偏移量的空间采样位置，并且从目标任务中学习到偏移量且不需要额外的监督。由于可变形卷积使卷积核的形态不仅仅是矩形框，更贴近特征提取目标，所以可以更准确地提取到我们想要的特征。

在本网络中使用了可变形卷积V2版本。

[论文](https://arxiv.org/pdf/1811.11168)：   Zhu X, Hu H, Lin S, et al. Deformable convnets v2: More deformable, better results[C].Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 9308-9316.

# 模型架构

Faster R-CNN-DCN是一个两阶段目标检测网络，该网络采用RPN，可以与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选计算。整个网络通过共享卷积特征，进一步将RPN和Fast R-CNN合并为一个网络。

通过加入可变形卷积网络，将resnet的3-5阶段中的卷积层更换为可变形卷积层，使卷积核的形态更贴近特征物，可以更准确的提取到想要的特征。

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

- 获取基础镜像

    - [Ascend Hub](https://ascend.huawei.com/ascendhub/#/home)

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO 2017。

- 本示例默认使用COCO 2017作为训练数据集，您也可以使用自己的数据集。

    1. 若使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool。

        ```python
        pip install Cython

        pip install pycocotools
        ```

        根据模型运行需要，对应地在`edfault_config.yaml`中更改COCO_ROOT和其他需要的设置。目录结构如下：

        ```path
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. 若使用自己的数据集，**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行内容如下：

        ```txt
        train2017/0000001.jpg 0,259,401,459,7,0 35,28,324,201,2,0 0,30,59,80,2,0
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class,is_crowd]格式的框,类和是否是一群物体的信息。从`image_dir`（数据集目录）图像路径以及`anno_path`（TXT文件路径）的相对路径中读取图像。`image_dir`和`anno_path`可在`config_50.yaml、config_101.yaml或config_152.yaml`中设置。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练模型是在ImageNet2012上训练的ResNet-50检查点。你可以使用ModelZoo中 [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) 脚本来训练。
3. BACKBONE_MODEL是通过modelzoo中的ResNet-50 [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) 脚本训练的。
4. PRETRAINED_MODEL是经过转换后的权重文件。VALIDATION_JSON_FILE为标签文件。CHECKPOINT_PATH是训练后的检查点文件。

## 在Ascend上运行

```shell

# 单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# 分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

# 评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](option)

#推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "epoch_size: 20"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "epoch_size: 20"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (6) 在网页上设置启动文件为 "eval.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

```shell
.
└─faster_rcnn
  ├─README.md                        // Faster R-CNN相关说明
  ├─ascend310_infer                  // 实现310推理源代码
  ├─scripts
    ├─run_standalone_train_ascend.sh // Ascend单机shell脚本
    ├─run_distribute_train_ascend.sh // Ascend分布式shell脚本
    ├─run_infer_310.sh               // Ascend推理shell脚本
    └─run_eval_ascend.sh             // Ascend评估shell脚本
  ├─src
    ├─FasterRcnn
      ├─__init__.py                  // init文件
      ├─anchor_generator.py          // 锚点生成器
      ├─bbox_assign_sample.py        // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py // 第二阶段采样器
      ├─dcn_v2.py                    // 可变性卷积V2网络
      ├─faster_rcnn_resnet50.py      // 以Resnet50作为backbone的Faster R-CNN网络
      ├─fpn_neck.py                  // 特征金字塔网络
      ├─proposal_generator.py        // 候选生成器
      ├─rcnn.py                      // R-CNN网络
      ├─resnet.py                    // 骨干网络
      ├─roi_align.py                 // ROI对齐网络
      └─rpn.py                       // 区域候选网络
    ├─dataset.py                     // 创建并处理数据集
    ├─lr_schedule.py                 // 学习率生成器
    ├─network_define.py              // Faster R-CNN网络定义
    ├─util.py                        // 评估相关操作
    └─model_utils
      ├─config.py                    // 获取.yaml配置参数
      ├─device_adapter.py            // 获取云上id
      ├─local_adapter.py             // 获取本地id
      └─moxing_adapter.py            // 云上数据准备
  ├─default_config.yaml              // Resnet50相关配置
  ├─export.py                        // 导出 AIR,MINDIR模型的脚本
  ├─eval.py                          // 评估脚本
  ├─postprogress.py                  // 310推理后处理脚本
  └─train.py                         // 训练脚本
```

## 训练过程

### 用法

#### 在Ascend上运行

```shell
# Ascend单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)

# Ascend分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)
```

#### 在GPU上运行

```shell
# GPU单机训练
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)

# GPU分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [COCO_ROOT] [MINDRECORD_DIR](option)
```

Notes:

1. 运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json。您可以使用[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成该文件。
2. PRETRAINED_MODEL应该是训练好的ResNet-50检查点。如果需要加载训练好的FasterRcnn的检查点，需要对train.py作如下修改:

```python
# 注释掉如下代码
#   load_path = args_opt.pre_trained
#    if load_path != "":
#        param_dict = load_checkpoint(load_path)
#        for item in list(param_dict.keys()):
#            if not item.startswith('backbone'):
#                param_dict.pop(item)
#        load_param_into_net(net, param_dict)

# 加载训练好的FasterRcnn检查点时需加载网络参数和优化器到模型，因此可以在定义优化器后添加如下代码：
    lr = Tensor(dynamic_lr(config, rank_size=device_num), mstype.float32)
    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    if load_path != "":
        param_dict = load_checkpoint(load_path)
        for item in list(param_dict.keys()):
            if item in ("global_step", "learning_rate") or "rcnn.reg_scores" in item or "rcnn.cls_scores" in item:
                param_dict.pop(item)
        load_param_into_net(opt, param_dict)
        load_param_into_net(net, param_dict)
```

3. defaule_config.yaml中包含原数据集路径，可以选择“coco_root”或“image_dir”。

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可以在loss_rankid.log中找到检查点文件以及结果，如下所示。

```log
# 分布式训练结果（8P）
339 epoch: 1 step: 1 total_loss: 5.00443
340 epoch: 1 step: 2 total_loss: 1.09367
340 epoch: 1 step: 3 total_loss: 0.90158
...
346 epoch: 1 step: 15 total_loss: 0.31314
347 epoch: 1 step: 16 total_loss: 0.84451
347 epoch: 1 step: 17 total_loss: 0.63137
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# Ascend评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [COCO_ROOT] [MINDRECORD_DIR](option)
```

> 在训练过程中生成检查点。
>
> 数据集中图片的数量要和VALIDATION_JSON_FILE文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.624
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.541
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.384
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
```

## 模型导出

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

## 推理过程

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。以下示例仅支持batch_size=1的mindir推理。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```log
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.434
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.523
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.536
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667

```

# 模型描述

## 性能

### 训练性能

| 参数 |Ascend |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |
| 上传日期 | 2021/11/5 |
| MindSpore版本 | 1.3.0 |
| 数据集 | COCO 2017 |
| 训练参数 | epoch=70, batch_size=2 |
| 优化器 | SGD |
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：448毫秒/步 |
| 总时间 | 8卡：66.2小时 |
| 参数(M) | 486 |

### 评估性能

| 参数 | Ascend |
| ------------------- | --------------------------- |
| 模型版本 | V1 |
| 资源 | Ascend 910 |
| 上传日期 | 2021/11/5 |
| MindSpore版本 | 1.3.0 |
| 数据集 | COCO2017 |
| batch_size | 2 |
| 输出 | mAP |
| 准确率 | IoU=0.50：62.0%  |
| 推理模型 | 486M（.ckpt文件） |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
