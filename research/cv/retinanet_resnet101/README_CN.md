# 内容

<!-- TOC -->

[View English](./README.md)

- [Retinanet描述](#retinanet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用-法)
        - [运行](#运-行)
        - [结果](#结-果)
    - [推理过程](#推理过程)
        - [用法](#用途)
        - [运行](#运行方式)
        - [结果](#运行结果)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

## [Retinanet描述](#content)

RetinaNet算法源自2018年Facebook AI Research的论文 Focal Loss for Dense Object Detection。该论文最大的贡献在于提出了Focal Loss用于解决类别不均衡问题，从而创造了RetinaNet（One Stage目标检测算法）这个精度超越经典Two Stage的Faster-RCNN的目标检测网络。

[论文](https://arxiv.org/pdf/1708.02002.pdf)
Lin T Y , Goyal P , Girshick R , et al. Focal Loss for Dense Object Detection[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 2017:2999-3007.

## [模型架构](#content)

Retinanet的整体网络架构如下所示：

[链接](https://arxiv.org/pdf/1708.02002.pdf)

## [数据集](#content)

数据集可参考文献.

[COCO2017](https://cocodataset.org/#download)

- 数据集大小: 19.3G, 123287张80类彩色图像

    - 训练:19.3G, 118287张图片

    - 测试:1814.3M, 5000张图片

- 数据格式:RGB图像.

> 注意：数据将在src/dataset.py 中被处理。

## [环境要求](#content)

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。
- 架构
    - [MindSpore](https://www.mindspore.cn/install)
- 想要获取更多信息，请检查以下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "coco_root='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size=500"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "coco_root=/cache/data"
    #          在网页上设置 "epoch_size=500"
    #          (可选)在网页上设置 "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          在网页上设置其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/retinanet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "coco_root='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size=500"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "coco_root='/cache/data'"
    #          在网页上设置 "epoch_size=500"
    #          (可选)在网页上设置 "checkpoint_path='s3://dir_to_your_pretrained/'"
    #          在网页上设置其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/retinanet"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "checkpoint_path='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "mindrecord_dir='./MindRecord_COCO'"
    #          在 default_config.yaml 文件中设置 "coco_root='/cache/data'"
    #          在 default_config.yaml 文件中设置其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "checkpoint_path='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "mindrecord_dir='./MindRecord_COCO'"
    #          在网页上设置 "coco_root='/cache/data'"
    #          在网页上设置其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/retinanet"
    # (6) 在网页上设置启动文件为 "eval.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "file_name='retinanet'"
    #          在 default_config.yaml 文件中设置 "file_format='MINDIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_path='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='retinanet'"
    #          在网页上设置 "file_format='MINDIR'"
    #          在网页上设置 "checkpoint_path='/The path of checkpoint in S3/'"
    #          在网页上设置其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/retinanet"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

## [脚本说明](#content)

### [脚本和示例代码](#content)

```shell
.
└─Retinanet_resnet101
  ├─README.md
  ├─ascend310_infer                           # 实现310推理源代码
  ├─scripts
    ├─run_single_train.sh                     # 使用Ascend环境单卡训练
    ├─run_single_train_gpu.sh
    ├─run_distribute_train.sh                 # 使用Ascend环境八卡并行训练
    ├─run_distribute_train_gpu.sh
    ├─run_eval.sh                             # 使用Ascend环境运行推理脚本
    ├─run_eval_gpu.sh
  ├─src
    ├─backbone.py                             # 网络模型定义
    ├─bottleneck.py                           # 网络颈部定义
    ├─config.py                               # 参数配置
    ├─dataset.py                              # 数据预处理
    ├─retinahead.py                           # 网络预测头部定义
    ├─init_params.py                          # 参数初始化
    ├─lr_generator.py                         # 学习率生成函数
    ├─coco_eval                               # coco数据集评估
    ├─box_utils.py                            # 先验框设置
    ├─_init_.py                               # 初始化
    └──model_utils
       ├──config.py                           # 训练配置
       ├──device_adapter.py                   # 获取云上id
       ├──local_adapter.py                    # 获取本地id
       └──moxing_adapter.py                   # 参数处理
  ├─default_config.yaml                       # 参数配置
  ├─train.py                                  # 网络训练脚本
  └─eval.py                                   # 网络推理脚本

```

### [脚本参数](#content)

```text
在train.py和default_config.yaml脚本中使用到的主要参数是:
"img_shape": [600, 600],                                                                        # 图像尺寸
"num_retinanet_boxes": 67995,                                                                   # 设置的先验框总数
"match_thershold": 0.5,                                                                         # 匹配阈值
"softnms_sigma": 0.5,                                                                           # softnms算法σ值
"nms_thershold": 0.6,                                                                           # 非极大抑制阈值
"min_score": 0.1,                                                                               # 最低得分
"max_boxes": 100,                                                                               # 检测框最大数量
"global_step": 0,                                                                               # 全局步数
"lr_init": 1e-6,                                                                                # 初始学习率
"lr_end_rate": 5e-3,                                                                            # 最终学习率与最大学习率的比值
"warmup_epochs1": 2,                                                                            # 第一阶段warmup的周期数
"warmup_epochs2": 5,                                                                           # 第二阶段warmup的周期数
"warmup_epochs3": 23,                                                                           # 第三阶段warmup的周期数
"warmup_epochs4": 60,                                                                          # 第四阶段warmup的周期数
"warmup_epochs5": 160,                                                                          # 第五阶段warmup的周期数
"momentum": 0.9,                                                                                # momentum
"weight_decay": 1.5e-4,                                                                         # 权重衰减率
"num_default": [9, 9, 9, 9, 9],                                                                 # 单个网格中先验框的个数
"extras_out_channels": [256, 256, 256, 256, 256],                                               # 特征层输出通道数
"feature_size": [75, 38, 19, 10, 5],                                                            # 特征层尺寸
"aspect_ratios": [(0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0), (0.5,1.0,2.0)],   # 先验框大小变化比值
"steps": ( 8, 16, 32, 64, 128),                                                                 # 先验框设置步长
"anchor_size":(32, 64, 128, 256, 512),                                                          # 先验框尺寸
"prior_scaling": (0.1, 0.2),                                                                    # 用于调节回归与回归在loss中占的比值
"gamma": 2.0,                                                                                   # focal loss中的参数
"alpha": 0.75,                                                                                 # focal loss中的参数
"mindrecord_dir": "/opr/root/data/MindRecord_COCO",                                             # mindrecord文件路径
"coco_root": "/opr/root/data/",                                                                 # coco数据集路径
"train_data_type": "train2017",                                                                 # train图像的文件夹名
"val_data_type": "val2017",                                                                     # val图像的文件夹名
"instances_set": "annotations_trainval2017/annotations/instances_{}.json",                     # 标签文件路径
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',     # coco数据集的种类
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
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes": 81,                                                                              # 数据集类别数
"voc_root": "",                                                                                 # voc数据集路径
"voc_dir": "",
"image_dir": "",                                                                                # 图像路径
"anno_path": "",                                                                                # 标签文件路径
"save_checkpoint": True,                                                                        # 保存checkpoint
"save_checkpoint_epochs": 1,                                                                    # 保存checkpoint epoch数
"keep_checkpoint_max":1,                                                                        # 保存checkpoint的最大数量
"save_checkpoint_path": "./model",                                                              # 保存checkpoint的路径
"finish_epoch":0,                                                                               # 已经运行完成的 epoch 数
"checkpoint_path":"/opr/root/reretina/retinanet2/LOG0/model/retinanet-400_458.ckpt"             # 用于验证的checkpoint路径
```

### [训练过程](#content)

#### 用法

您可以使用python或shell脚本进行训练。shell脚本的用法如下:

- Ascend:

```shell
# data和存储mindrecord文件的路径在default_config.yaml里设置

# 训练以前， 请运行：
python train.py --only_create_dataset=True --run_platform="Ascend"

# 八卡并行训练示例：
# 创建RANK_TABLE_FILE
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)

# 单卡训练示例：
bash run_single_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

> 注意: RANK_TABLE_FILE相关参考资料见[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html), 获取device_ip方法详见[链接](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools).

- GPU

```shell
# data和存储mindrecord文件的路径在default_config.yaml里设置

# 训练以前， 请运行：
python train.py --only_create_dataset=True --run_platform="GPU"

# 八卡并行训练示例：
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)

# 单卡训练示例：
bash run_single_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

#### 运行

- Ascend

```shell
# 八卡并行训练示例(在scripts目录下运行)：
bash run_distribute_train.sh 8 500 0.1 coco scripts/rank_table_8pcs.json /dataset/retinanet-322_458.ckpt 322

# 单卡训练示例(在scripts目录下运行)：
bash run_single_train.sh 0 500 0.1 coco /dataset/retinanet-322_458.ckpt 322
```

- GPU

```shell
# 八卡并行训练示例(在scripts目录下运行)：
bash run_distribute_train_gpu.sh 8 400 0.025 coco /dataset/retinanet-322_1221.ckpt 322

# 单卡训练示例(在scripts目录下运行)：
bash run_single_train_gpu.sh 0 400 0.025 coco /dataset/retinanet-322_1221.ckpt 322
```

#### 结果

路径在default_config.yaml里设置。checkpoint将存储在 `./model` 路径下，
训练日志将被记录到 `./train.log` 中，训练日志部分示例如下：

```text
epoch: 397 step: 458, loss is 0.6153226
lr:[0.000598]
epoch time: 313364.642 ms, per step time: 684.202 ms
epoch: 398 step: 458, loss is 0.5491791
lr:[0.000544]
epoch time: 313486.094 ms, per step time: 684.467 ms
epoch: 399 step: 458, loss is 0.51681435
lr:[0.000511]
epoch time: 313514.348 ms, per step time: 684.529 ms
epoch: 400 step: 458, loss is 0.4305706
lr:[0.000500]
epoch time: 314138.455 ms, per step time: 685.892 ms
```

### [评估过程](#content)

#### 用法

您可以使用python或shell脚本进行训练。shell脚本的用法如下:

- Ascend

```shell
bash run_eval.sh [DATASET] [DEVICE_ID]
```

- GPU

```shell
bash run_eval_gpu.sh [DATASET] [DEVICE_ID] [CHECKPOINT_PATH]
```

#### 运行

- Ascend:

```shell
bash run_eval.sh coco 0
```

- GPU

```shell
bash run_eval_gpu.sh coco 0 LOG/model/retinanet-500_610.ckpt
```

> checkpoint可以在训练过程中产生.

#### 结果

计算结果将存储在示例路径中，您可以在 `eval.log` 查看.

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680

========================================

mAP: 0.3710347196613514
```

### [模型导出](#content)

#### 具体用法

导出模型前要修改default_config.yaml文件中的checkpoint_path配置项，值为checkpoint的路径。

```shell
python export.py --file_name [device_target] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

#### 运行命令

```shell
python export.py
```

### [推理过程](#content)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

#### 用途

在推理之前需要在昇腾910环境上完成模型的导出。推理时要将iscrowd为true的图片排除掉。在ascend310_infer目录下保存了去排除后的图片id。
还需要修改default_config.yaml文件中的coco_root、val_data_type、instances_set配置项，值分别取coco数据集的目录，推理所用数据集的目录名称，推理完成后计算精度用的annotation文件，instances_set是用val_data_type拼接起来的，要保证文件正确并且存在。

```shell
# Ascend310 inference
sh run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

#### 运行方式

```shell
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

#### 运行结果

推理的结果保存在当前目录下，在`acc.log`日志文件中可以找到类似以下的结果。

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
mAP: 0.36858371862143824

```

## [模型说明](#content)

### [性能](#content)

#### 训练性能

| 参数                        | Ascend                           | GPU(1pcs)            | GPU(8pcs)            |
| -------------------------- | -------------------------------- | -------------------- | -------------------- |
| 模型名称                    | Retinanet                         | Retinanet-resnet-101 | Retinanet-resnet-101 |
| 运行环境                    | 华为云 Modelarts                   | Ubuntu 18.04.6, 1pcs Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB | Ubuntu 18.04.6, 8pcs Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB |
| 上传时间                    | 10/03/2021                        | 27/12/2021           | 27/12/2021           |
| MindSpore 版本             | 1.0.1                             | 1.6.0                | 1.6.0                |
| 数据集                      | 118287 张图片                      | 118287 张图片         | 118287 张图片        |
| 训练参数                    | batch_size=32                     | batch_size=16, epochs=400, lr=0.025 | batch_size=12, epochs=400, lr=0.025 |
| 其他训练参数                 | src/config.py                     | default_config.yaml  | default_config.yaml        |
| 优化器                      | Momentum                          | Momentum             | Momentum             |
| 损失函数                    | Focal loss                        | Focal loss           | Focal loss           |
| 最终损失                    | 0.43                              | 0.49                 | 0.49 |
| 速度                       |                                   | 696 毫秒/步           | 881 毫秒/步 |
| 训练总时间 (8p)             | 34h50m20s                         | 708h                 | 120h |
| 脚本                       | [Retianet script](https://gitee.com/mindspore/models/tree/r2.0/research/cv/retinanet_resnet101) |

#### 评估性能

| 参数                | Ascend        | GPU       |
| ------------------- |:--------------| -------- |
| 模型名称             | Retinanet     | Retinanet-resnet-101  |
| 运行环境             | 华为云 Modelarts | Ubuntu 18.04.6, Tesla V100-PCIE 32G, CPU 2.90GHz, 64cores, RAM 252GB |
| 上传时间             | 10/03/2021    | 27/12/2021 |
| MindSpore 版本       | 1.0.1         | 1.6.0 |
| 数据集               | 5k 张图片        | 5k images |
| Batch_size          | 1             | 1         |
| 精确度              | mAP[0.3710]   | mAP[0.3687] |
| 总时间              | 10m 50s       | 10 min |

# [随机情况的描述](#内容)

在 `train.py` 脚本中设置了随机种子.

# [ModelZoo 主页](#内容)

请核对官方 [主页](https://gitee.com/mindspore/models).

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

- **Q: 使用PYNATIVE_MODE发生内存溢出怎么办？**
- **A**：内存溢出通常是因为PYNATIVE_MODE需要更多的内存， 将batch size设置为16降低内存消耗，可进行网络训练。
