# 目录

<!-- TOC -->

- [目录](#目录)
- [模型说明](#RefineDet说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [Ascend上训练](#ascend上训练)
        - [GPU训练](#gpu训练)
    - [评估过程](#评估过程)
        - [Ascend处理器环境评估](#ascend处理器环境评估)
        - [GPU处理器环境评估](#gpu处理器环境评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)

<!-- /TOC -->

# RefineDet说明

RefineDet是CVPR 2018中提出的一种目标检测模型。它融合one-stage方法和two-stage方法的优点（前者更快，后者更准）并克服了它们的缺点。通过使用ARM对随机生成的检测框进行先一步的回归，再使用TCB模块融合多尺度的特征，最后使用类似SSD的回归和分类结构大大提高了目标检测的速度和精度。

[论文](https://arxiv.org/pdf/1711.06897.pdf)：   S. Zhang, L. Wen, X. Bian, Z. Lei and S. Z. Li, "Single-shot refinement neural network for object detection", Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 4203-4212, Jun. 2018.

# 模型架构

RefineDet的结构由三部分组成————负责预回归检测框的ARM，检测目标的ODM和连接两者的TCB
![refinedet_structure](https://github.com/sfzhang15/RefineDet/raw/master/refinedet_structure.jpg)

RefineDet的结构，图片来自原论文

特征提取部分使用VGG-16作为backbone

# 数据集

使用的数据集： [COCO2017](<http://images.cocodataset.org/>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图像
    - 验证集：1 GB，5000张图像
    - 标注：241 MB，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理

# 环境要求

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO2017。

- 本示例默认使用COCO2017作为训练数据集，您也可以使用自己的数据集。

    1. 如果使用coco数据集。**执行脚本时选择数据集coco。**
        安装Cython、pycocotool和opencv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        pip install opencv-python
        ```

        并在模型对应的config文件中更改coco_root,mindrecord_dir和其他您需要的设置。目录结构如下：

        ```text
        .
        └─ cocodataset
          ├─ annotations
          │ ├─ instance_train2017.json
          │ └─ instance_val2017.json
          ├─ val2017
          └─ train2017

        ```

    2. 如果使用自己的数据集。**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行如下：

        ```text
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。我们从`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。在`config.py`中设置`IMAGE_DIR`和`ANNO_PATH`。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```shell script
# Ascend单卡直接训练示例
python train.py --device_id=0 --epoch_size=500 --dataset=coco
# 或者
bash run_standardalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET]
# 示例
bash run_standardalone_train.sh 0 500 0.05 coco
```

```shell script
# Ascend分布式训练
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE]
# 示例
bash run_distribute_train.sh 8 500 0.05 coco ./hccl_rank_tabel_8p.json
```

在modelarts上训练请增加参数run_online，并设置data_url与train_url，其它与Ascend平台上运行的参数相同

```shell script
# Ascend ModelArts训练示例
python train.py --run_online=True --data_url=obs://xxx/coco2017 --train_url=obs://xxx/train_output --distribute=True --epoch_size=500 --dataset=coco
```

```shell script
# Ascend处理器环境进行评估
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
# 示例
bash run_eval.sh coco  ./ckpt/refinedet.ckpt 0
# 或直接运行eval.py，示例如下
python eval.py --dataset=coco --device_id=0 --checkpoint_path=./ckpt/refinedet.ckpt
```

- GPU处理器环境运行

```shell script
# GPU单卡训练
bash run_standardalone_train_gpu.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET]
# 示例
bash ./run_standardalone_train_gpu.sh 0 5 0.02 coco
```

```shell script
# GPU分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET]
# 示例
bash ./run_distribute_train_gpu.sh 8 500 0.05 coco
```

```shell script
# GPU处理器环境运行eval
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
# 示例
bash run_eval_gpu.sh coco ../ckpt/ckpt_0/refinedet-500_458.ckpt 0
```

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ RefineDet
    ├─ README.md                         ## RefineDet相关说明
    ├─ scripts
    │ ├─ run_infer_310.sh                ## 310推理shell脚本
    │ ├─ run_standardalone_train.sh      ## Ascend单卡shell脚本
    │ ├─ run_standardalone_train_gpu.sh  ## GPU单卡shell脚本
    │ ├─ run_distribute_train.sh         ## Ascend分布式shell脚本
    │ ├─ run_distribute_train_gpu.sh     ## GPU分布式shell脚本
    │ ├─ run_eval.sh                     ## Ascend评估shell脚本
    │ └─ run_eval_gpu.sh                 ## GPU评估shell脚本
    ├─ src
    │ ├─ anchor_generator.py             ## 生成初始的随机检测框的脚本
    │ ├─ box_utils.py                    ## bbox处理脚本
    │ ├─ config.py                       ## 总的config文件
    │ ├─ dataset.py                      ## 处理并生成数据集的脚本
    │ ├─ eval_utils.py                   ## 评估函数的脚本
    │ ├─ init_params.py                  ## 初始化网络参数的脚本
    │ ├─ __init__.py
    │ ├─ l2norm.py                       ## 实现L2 Normalization的脚本
    │ ├─ lr_schedule.py                  ## 实现动态学习率的脚本
    │ ├─ multibox.py                     ## 实现多检测框回归的脚本
    │ ├─ refinedet_loss_cell.py          ## 实现loss函数的脚本
    │ ├─ refinedet.py                    ## 定义了整个网络框架的脚本
    │ ├─ resnet101_for_refinedet.py      ## 实现了resnet101作为backbone
    │ └─ vgg16_for_refinedet.py          ## 实现了vgg16作为backbone
    ├─ eval.py                           ## 评估脚本
    ├─ export.py                         ## 导出模型脚本
    ├─ train.py                          ## 训练脚本
    └─ postprocess.py                    ## 用于310推理的后处理脚本
```

## 脚本参数

  ```text
  train.py和config.py中主要参数如下：

    "device_num": 1                            # 使用设备数量
    "lr": 0.05                                 # 学习率初始值
    "dataset": coco                            # 数据集名称
    "epoch_size": 500                          # 轮次大小
    "batch_size": 32                           # 输入张量的批次大小
    "pre_trained": None                        # 预训练检查点文件路径
    "pre_trained_epoch_size": 0                # 预训练轮次大小
    "save_checkpoint_epochs": 10               # 两个检查点之间的轮次间隔。默认情况下，每10个轮次都会保存检查点。
    "loss_scale": 1024                         # 损失放大

    "class_num": 81                            # 数据集类数
    "image_shape": [320, 320]                  # 作为模型输入的图像高和宽
    "mindrecord_dir": "/data/MindRecord"       # MindRecord路径
    "coco_root": "/data/coco2017"              # COCO2017数据集路径
    "voc_root": ""                             # VOC原始数据集路径
    "image_dir": ""                            # 其他数据集图片路径，如果使用coco或voc，此参数无效。
    "anno_path": ""                            # 其他数据集标注路径，如果使用coco或voc，此参数无效。

  ```

## 训练过程

运行`train.py`训练模型。如果`mindrecord_dir`为空，则会通过`coco_root`（coco数据集）或`image_dir`和`anno_path`（自己的数据集）生成MindRecord文件。**注意，如果mindrecord_dir不为空，将使用mindrecord_dir代替原始图像。**

### Ascend上训练

- 分布式

注意训练前请确保对应模型的config中的数据集路径（例如coco数据集对应coco_root）以及mindrecord_dir路径设置正确，请使用绝对路径。

```shell script
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `RANK_TABLE_FILE`：hccl配置文件的路径，最好使用绝对路径。具体操作见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)
- `PRE_TRAINED`：预训练检查点文件的路径。最好使用绝对路径。
- `PRE_TRAINED_EPOCH_SIZE`：预训练的轮次数。

    训练结果保存在当前路径中，文件夹名称以"LOG"开头。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 458, loss is 3.1681802
epoch time: 228752.4654865265, per step time: 499.4595316299705
epoch: 2 step: 458, loss is 2.8847265
epoch time: 38912.93382644653, per step time: 84.96273761232868
epoch: 3 step: 458, loss is 2.8398118
epoch time: 38769.184827804565, per step time: 84.64887516987896
...

epoch: 498 step: 458, loss is 0.70908034
epoch time: 38771.079778671265, per step time: 84.65301261718616
epoch: 499 step: 458, loss is 0.7974688
epoch time: 38787.413120269775, per step time: 84.68867493508685
epoch: 500 step: 458, loss is 0.5548882
epoch time: 39064.8467540741, per step time: 85.29442522723602
```

### GPU训练

- 分布式

注意训练前请确保对应模型的config中的数据集路径（例如coco数据集对应coco_root）以及mindrecord_dir路径设置正确，请使用绝对路径。

```shell script
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `PRE_TRAINED`：预训练检查点文件的路径。最好使用绝对路径。
- `PRE_TRAINED_EPOCH_SIZE`：预训练的轮次数。

    训练结果保存在当前路径中，文件夹名称以"LOG"开头。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 3664, loss is 6.7545223
epoch time: 1163814.209 ms, per step time: 317.635 ms
epoch: 2 step: 3664, loss is 6.201771
epoch time: 1147062.749 ms, per step time: 313.063 ms
epoch: 3 step: 3664, loss is 5.6326284
epoch time: 1145688.651 ms, per step time: 312.688 ms
epoch: 4 step: 3664, loss is 5.1610036
epoch time: 1145988.975 ms, per step time: 312.770 ms
epoch: 5 step: 3664, loss is 4.332919
epoch time: 1146872.690 ms, per step time: 313.011 ms
...
```

## 评估过程

### Ascend处理器环境评估

```shell script
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

此脚本需要三个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702

========================================

mAP:0.2885878918173237
```

### GPU处理器环境评估

```shell script
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID]
```

此脚本需要三个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.696

========================================

mAP: 0.288743125771368
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。精度计算过程需要70G+的内存，否则进程将会因为超出内存被系统终止。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本所在目录的父目录下，你可以在acc.log中看到以下精度计算结果。

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702
 mAP:0.2885878918173237
```

# 模型描述

## 性能

### 评估性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | -------------------------------------------------------------| -------------------------------------------------------------|
| 模型版本 | RefineDet_vgg16_320 | RefineDet_vgg16_320 |
| 资源 | Ascend 910；CPU： 2.60GHz，192核；内存：755 GB | NV RTX 3090-24G |
| 上传日期 | 2021-11-11  | 2022-05-03 |
| MindSpore版本          | 1.3.0                                                  | 1.5.0                                                        |
| 数据集                   | COCO2017                                                     | COCO2017                                                     |
| 训练参数    | epoch = 500,  batch_size = 32                                | epoch = 500,  batch_size = 32                                |
| 优化器               | Momentum                                                     | Momentum                                                     |
| 损失函数 | Sigmoid交叉熵，SmoothL1Loss | Sigmoid交叉熵，SmoothL1Loss |
| 速度 | 8卡：769毫秒/步 | 8卡：423毫秒/步 |
| 总时长 | 8卡：49.5小时 | 8卡：26.9小时 |
| 参数(M) | 272 | 272 |

### 推理性能

| 参数          | Ascend                      | GPU                         |
| ------------------- | ----------------------------| ----------------------------|
| 模型版本       | RefineDet_vgg16_320                    | RefineDet_vgg16_320                     |
| 资源           | Ascend 910                  | GPU                         |
| 上传日期  | 2021-06-01  | 2021-09-24 |
| MindSpore版本    | 1.3.0                 | 1.5.0                       |
| 数据集         | COCO2017                    | COCO2017                    |
| batch_size          | 1                           | 1                           |
| 准确率 | IoU=0.50: 28.9%             | IoU=0.50: 28.9%             |
| 推理模型   | 272M（.ckpt文件）            | 272M（.ckpt文件）            |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。