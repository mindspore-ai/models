# 目录

- [目录](#目录)
- [Faster R-CNN描述](#faster-r-cnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [在Ascend上运行](#在ascend上运行)
    - [在GPU上运行](#在gpu上运行)
    - [在CPU上运行](#在cpu上运行)
    - [在docker上运行](#在docker上运行)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [在Ascend上运行](#在ascend上运行-1)
            - [在GPU上运行](#在gpu上运行-1)
            - [在CPU上运行](#在cpu上运行-1)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [在Ascend上运行](#在ascend上运行-2)
            - [在GPU上运行](#在gpu上运行-2)
            - [在CPU上运行](#在cpu上运行-2)
        - [结果](#结果-1)
    - [模型导出](#模型导出)
    - [推理过程](#推理过程)
        - [使用方法](#使用方法)
        - [结果](#结果-2)
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

# 模型架构

Faster R-CNN是一个两阶段目标检测网络，该网络采用RPN，可以与检测网络共享整个图像的卷积特征，可以几乎无代价地进行区域候选计算。整个网络通过共享卷积特征，进一步将RPN和Fast R-CNN合并为一个网络。

# 数据集

使用的数据集：[COCO 2017](<https://cocodataset.org/>)

- 数据集大小：19G
    - 训练集：18G，118,000个图像  
    - 验证集：1G，5000个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

使用的数据集：[FaceMaskDetection](<https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/>)

- 数据集大小：417M
    - 训练集：415M，853个图像  
    - 标注集：1.6M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 需要将XML格式数据转化为COCO格式数据

 ```text
1.数据划分,splitdata.py，将下载的原始数据解压放在/data目录下，包含images图像目录和annotations标注目录（格式为XML格式），在执行split_data.py之后，会对原始数据进行划分，得到训练集/data/train/和验证集/data/val/
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
2.xml转coco,xml2coco.py,第1步对数据划分后，标注格式是XML格式，需要将XML转成COCO格式
2.0 在/data目录下新建face_detction目录，在facedetection新建annotations目录
2.1 生成COCO格式训练集，python xml2coco.py --data_path /data/train/ --save_path /data/face_detection/annotations/instances_train2017.json
2.2 生存COCO格式验证集, python xml2coco.py --data_path /data/val/ --save_path /data/face_detection/annotations/instances_val2017.json
2.3 将/data/train/images 复制到/data/face_detection/下，并重命名为train2017；将/data/val/images 复制到/data/face_detection/下，并重命名为val2017
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
        └─instances_val2017.json
```

# 环境要求

- 硬件（Ascend/GPU/CPU）

    - 使用Ascend处理器来搭建硬件环境。

- 获取基础镜像

    - [Ascend Hub](https://ascend.huawei.com/ascendhub/#/home)

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO 2017。

- 本示例默认使用COCO 2017作为训练数据集，您也可以使用自己的数据集。

    1. 若使用COCO数据集，**执行脚本时选择数据集COCO。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        pip install mmcv==0.2.14
        ```

        根据模型运行需要，对应地在`default_config.yaml、default_config_101.yaml、default_config_152.yaml或default_config_InceptionResnetV2.yaml`中更改COCO_ROOT和其他需要的设置。目录结构如下：

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
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。从`IMAGE_DIR`（数据集目录）图像路径以及`ANNO_PATH`（TXT文件路径）的相对路径中读取图像。`IMAGE_DIR`和`ANNO_PATH`可在`default_config.yaml、default_config_101.yaml、default_config_152.yaml或default_config_InceptionResnetV2.yaml`中设置。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

注意：

1. 第一次运行生成MindRecord文件，耗时较长。
2. 预训练模型是在ImageNet2012上训练的ResNet-50检查点。你可以使用ModelZoo中 [resnet50](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet) 脚本来训练, 然后使用src/convert_checkpoint.py把训练好的resnet50的权重文件转换为可加载的权重文件。
3. BACKBONE_MODEL是通过modelzoo中的[resnet50](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)脚本训练的。PRETRAINED_MODEL是经过转换后的权重文件。VALIDATION_JSON_FILE为标签文件。CHECKPOINT_PATH是训练后的检查点文件。

## 在Ascend上运行

```shell

# 权重文件转换
python -m src.convert_checkpoint --ckpt_file=[BACKBONE_MODEL]

# 单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# 分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)

# 评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

#推理(IMAGE_WIDTH和IMAGE_HEIGHT必须同时设置或者同时使用默认值。)
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

## 在GPU上运行

```shell

# 权重文件转换
python -m src.convert_checkpoint --ckpt_file=[BACKBONE_MODEL]

# 单机训练
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# 分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)

# 评估
python eval.py --anno_path=[ANN_FILE] --checkpoint_path=[CHECKPOINT_PATH] --coco_root=[FACE_DETECTION_PATH] --config_path=[CONFIG_PATH]
```

## 在CPU上运行

```shell


# 单机训练
python train.py --config_path=[CONFIG_PATH] --pre_trained=[PRE_TRAINED] --coco_root=[FACE_DETECTION_PATH]
# 评估
python eval.py --anno_path=[ANN_FILE] --checkpoint_path=[CHECKPOINT_PATH] --coco_root=[FACE_DETECTION_PATH] --config_path=[CONFIG_PATH]
```

## 在docker上运行

1. 编译镜像

```shell
# 编译镜像
docker build -t fasterrcnn:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
```

2. 启动容器实例

```shell
# 启动容器实例
bash scripts/docker_start.sh fasterrcnn:20.1.0 [DATA_DIR] [MODEL_DIR]
```

3. 训练

```shell
# 单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# 分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

4. 评估

```shell
# 评估
python eval.py --anno_path=[ANN_FILE] --checkpoint_path=[CHECKPOINT_PATH] --coco_root=[FACE_DETECTION_PATH] --config_path=[CONFIG_PATH]
```

5. 推理

```shell
# 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [DEVICE_ID](optional)
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "data_path=/cache/data"
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
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 20"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_path='/cache/data'"
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
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./faster_rcnn/faster_rcnn_trained.ckpt'"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='faster_rcnn'"
    #          在 base_config.yaml 文件中设置 "file_format='MINDIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='faster_rcnn'"
    #          在网页上设置 "file_format='MINDIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
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
    ├─run_standalone_train_gpu.sh    // GPU单机shell脚本
    ├─run_distribute_train_ascend.sh // Ascend分布式shell脚本
    ├─run_distribute_train_gpu.sh    // GPU分布式shell脚本
    ├─run_infer_310.sh               // Ascend推理shell脚本
    └─run_eval_ascend.sh             // Ascend评估shell脚本
    └─run_eval_gpu.sh                // GPU评估shell脚本
  ├─src
    ├─FasterRcnn
      ├─__init__.py                  // init文件
      ├─anchor_generator.py          // 锚点生成器
      ├─bbox_assign_sample.py        // 第一阶段采样器
      ├─bbox_assign_sample_stage2.py // 第二阶段采样器
      ├─faster_rcnn.py               // Faster R-CNN网络
      ├─fpn_neck.py                  // 特征金字塔网络
      ├─proposal_generator.py        // 候选生成器
      ├─rcnn.py                      // R-CNN网络
      ├─resnet.py                    // 骨干网络
      ├─resnet50v1.py                // Resnet50v1.0骨干网络
      ├─inceptionresnetv2.py         // inception resnet v2骨干网络
      ├─roi_align.py                 // ROI对齐网络
      └─rpn.py                       // 区域候选网络
    ├─dataset.py                     // 创建并处理数据集
    ├─lr_schedule.py                 // 学习率生成器
    ├─network_define.py              // Faster R-CNN网络定义
    ├─util.py                        // 例行操作
    └─model_utils
      ├─config.py                    // 获取.yaml配置参数
      ├─device_adapter.py            // 获取云上id
      ├─local_adapter.py             // 获取本地id
      └─moxing_adapter.py            // 云上数据准备
  ├─default_config.yaml              // Resnet50相关配置,COCO数据集
  ├─fasterrcnn_facemask_config_cpu.yaml              // Resnet50相关配置，FaceMaskDetection数据集
  ├─default_config_101.yaml          // Resnet101相关配置
  ├─default_config_152.yaml          // Resnet152相关配置
  ├─default_config_InceptionResnetV2.yaml   // inception resnet v2相关配置
  ├─export.py                        // 导出 AIR,MINDIR模型的脚本
  ├─eval.py                          // 评估脚本
  ├─postprogress.py                  // 310推理后处理脚本
  └─train.py                         // 训练脚本
```

```bash
`BACKBONE` should be in ["resnet_v1.5_50", "resnet_v1_101", "resnet_v1_152", "resnet_v1_50", "inception_resnet_v2"]

if backbone in ("resnet_v1.5_50", "resnet_v1_101", "resnet_v1_152", "inception_resnet_v2"):
    from src.FasterRcnn.faster_rcnn_resnet import Faster_Rcnn_Resnet
    "resnet_v1.5_50" -> "./default_config.yaml"
    "resnet_v1_101"  -> "./default_config_101.yaml"
    "resnet_v1_152"  -> "./default_config_152.yaml"
    "inception_resnet_v2"  -> "./default_config_InceptionResnetV2.yaml"

elif backbone == "resnet_v1_50":
    from src.FasterRcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet
    "resnet_v1_50" -> "./default_config.yaml"
```

## 训练过程

### 用法

#### 在Ascend上运行

```shell
# Ascend单机训练
bash run_standalone_train_ascend.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# Ascend分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

#### 在GPU上运行

```shell
# GPU单机训练
bash run_standalone_train_gpu.sh [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)

# GPU分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [PRETRAINED_MODEL] [BACKBONE] [COCO_ROOT] [MINDRECORD_DIR](optional)
```

#### 在CPU上运行

```shell
# CPU单机训练
python train.py --config_path=[CONFIG_PATH] --pre_trained=[PRE_TRAINED] --coco_root=[FACE_DETECTION_PATH]
```

Notes:

1. 运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json。您可以使用[hccl_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)生成该文件。
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

3. default_config.yaml、default_config_101.yaml、default_config_152.yaml或default_config_InceptionResnetV2.yaml中包含原数据集路径，可以选择“coco_root”或“image_dir”。

### 结果

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可以在loss_rankid.log中找到检查点文件以及结果，如下所示。

```log
# 分布式训练结果（8P）
epoch: 1 step: 7393, rpn_loss: 0.12054, rcnn_loss: 0.40601, rpn_cls_loss: 0.04025, rpn_reg_loss: 0.08032, rcnn_cls_loss: 0.25854, rcnn_reg_loss: 0.14746, total_loss: 0.52655
epoch: 2 step: 7393, rpn_loss: 0.06561, rcnn_loss: 0.50293, rpn_cls_loss: 0.02587, rpn_reg_loss: 0.03967, rcnn_cls_loss: 0.35669, rcnn_reg_loss: 0.14624, total_loss: 0.56854
epoch: 3 step: 7393, rpn_loss: 0.06940, rcnn_loss: 0.49658, rpn_cls_loss: 0.03769, rpn_reg_loss: 0.03165, rcnn_cls_loss: 0.36353, rcnn_reg_loss: 0.13318, total_loss: 0.56598
...
epoch: 10 step: 7393, rpn_loss: 0.03555, rcnn_loss: 0.32666, rpn_cls_loss: 0.00697, rpn_reg_loss: 0.02859, rcnn_cls_loss: 0.16125, rcnn_reg_loss: 0.16541, total_loss: 0.36221
epoch: 11 step: 7393, rpn_loss: 0.19849, rcnn_loss: 0.47827, rpn_cls_loss: 0.11639, rpn_reg_loss: 0.08209, rcnn_cls_loss: 0.29712, rcnn_reg_loss: 0.18115, total_loss: 0.67676
epoch: 12 step: 7393, rpn_loss: 0.00691, rcnn_loss: 0.10168, rpn_cls_loss: 0.00529, rpn_reg_loss: 0.00162, rcnn_cls_loss: 0.05426, rcnn_reg_loss: 0.04745, total_loss: 0.10859
```

## 评估过程

### 用法

#### 在Ascend上运行

```shell
# Ascend评估
bash run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

#### 在GPU上运行

```shell
# GPU评估
bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](optional)
```

#### 在CPU上运行

```shell
# CPU评估
python eval.py --anno_path=[ANN_FILE] --checkpoint_path=[CHECKPOINT_PATH] --coco_root=[FACE_DETECTION_PATH] --config_path=[CONFIG_PATH]
```

> 在训练过程中生成检查点。
>
> 数据集中图片的数量要和VALIDATION_JSON_FILE文件中标记数量一致，否则精度结果展示格式可能出现异常。

### 结果

评估结果将保存在示例路径中，文件夹名为“eval”。在此文件夹下，您可以在日志中找到类似以下的结果。

```log
COCO2017数据集结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```

```log
FaceMaskDetction结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.595
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.906
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.722
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.618
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.857
```

## 模型导出

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT] --backbone [BACKBONE]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]
`BACKBONE` 可选 ["resnet_v1.5_50", "resnet_v1_101", "resnet_v1_152", "resnet_v1_50", "inception_resnet_v2"]

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。以下示例仅支持batch_size=1的mindir推理。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_TYPE] [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional) [KEEP_RATIO](optional) [DEVICE_ID](optional)
```

- `IMAGE_WIDTH` 可选，默认值为1024。
- `IMAGE_HEIGHT` 可选，默认值为768。
- `KEEP_RATIO` 可选，默认值为true。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.622
```

# 模型描述

## 性能

### 训练性能

| 参数 |Ascend |GPU |CPU|
| -------------------------- | ----------------------------------------------------------- |----------------------------------------------------------- |------|
| 模型版本 | V1 |V1 |V1|
| 资源 | Ascend 910；CPU 2.60GHz，192核；内存：755G |V100-PCIE 32G            |V100-PCIE 32G|
| 上传日期 | 2020/8/31 | 2021/2/10 |2022/8/10|
| MindSpore版本 | 1.0.0 |1.2.0 |1.7.0|
| 数据集 | COCO 2017 |COCO 2017 |FaceMaskDetection|
| 训练参数 | epoch=12, batch_size=2 |epoch=12, batch_size=2 |epoch=20,batch_size=2|
| 优化器 | SGD |SGD |SGD|
| 损失函数 | Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss |Softmax交叉熵，Sigmoid交叉熵，SmoothL1Loss|
| 速度 | 1卡：190毫秒/步；8卡：200毫秒/步 | 1卡：320毫秒/步；8卡：335毫秒/步 |1卡：7328毫秒/步|
| 总时间 | 1卡：37.17小时；8卡：4.89小时 |1卡：63.09小时；8卡：8.25小时 |1卡：13.88小时|
| 参数(M) | 250 |250 |495|
| 脚本 | [Faster R-CNN脚本](https://gitee.com/mindspore/models/tree/r2.0/official/cv/FasterRCNN) | [Faster R-CNN脚本](https://gitee.com/mindspore/models/tree/r2.0/official/cv/FasterRCNN) |[Faster R-CNN脚本](https://gitee.com/mindspore/models/tree/r2.0/official/cv/FasterRCNN) |

### 评估性能

| 参数 | Ascend |GPU |CPU|
| ------------------- | --------------------------- | --------------------------- |-----|
| 模型版本 | V1 |V1 |V1|
| 资源 | Ascend 910 |V100-PCIE 32G  |V100-PCIE 32G|
| 上传日期 | 2020/8/31 |2021/2/10 |2022/8/10|
| MindSpore版本 | 1.0.0 |1.2.0 |1.7.0|
| 数据集 | COCO2017 |COCO2017 |FaceMaskDetection|
| batch_size | 2 | 2 |2|
| 输出 | mAP |mAP |mAP|
| 准确率 | IoU=0.50：58.6%  |IoU=0.50：59.1%  |IoU=0.5: 90.6%|
| 推理模型 | 250M（.ckpt文件） |250M（.ckpt文件） |495M（.ckpt文件）|

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
