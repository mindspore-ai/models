# 目录

<!-- TOC -->

- [目录](#目录)
- [SSD说明](#ssd说明)
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
        - [ONNX处理器环境评估](#onnx处理器环境评估)
    - [推理过程](#推理过程)
        - [Ascend310执行推理](#ascend310执行推理)
            - [导出MindIR](#导出mindir)
        - [执行推理](#执行推理)
            - [结果](#结果)
        - [ONNX推理](#onnx推理)
            - [导出ONNX](#导出onnx)
            - [执行推理](#执行推理-1)
            - [结果](#结果-1)
- [模型描述](#模型描述)
    - [性能](#性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SSD说明

SSD将边界框的输出空间离散成一组默认框，每个特征映射位置具有不同的纵横比和尺度。在预测时，网络对每个默认框中存在的对象类别进行评分，并对框进行调整以更好地匹配对象形状。此外，网络将多个不同分辨率的特征映射的预测组合在一起，自然处理各种大小的对象。

[论文](https://arxiv.org/abs/1512.02325)：   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

# 模型架构

SSD方法基于前向卷积网络，该网络产生固定大小的边界框集合，并针对这些框内存在的对象类实例进行评分，然后通过非极大值抑制步骤进行最终检测。早期的网络层基于高质量图像分类的标准体系结构，被称为基础网络。后来通过向网络添加辅助结构进行检测。

我们提供了4种不同的基础架构:

- **ssd300**， 参考论文实现。 使用mobilenet-v2作为骨干网络， 并使用和论文相同的bbox预测器。
- ***ssd-mobilenet-v1-fpn**， 使用mobilenet-v1和FPN作为特征提取器， 并使用权重共享box预测器。
- ***ssd-resnet50-fpn**， 使用resnet50和FPN作为特征提取器， 并使用权重共享box预测器。
- **ssd-vgg16**， 参考论文实现。 使用vgg16作为骨干网络， 并使用和论文相同的bbox预测器。

# 数据集

使用的数据集： [COCO2017](<https://cocodataset.org/#download>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图像
    - 验证集：1 GB，5000张图像
    - 标注：241 MB，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理

[helmet](<https://osf.io/4pwj8>)

- 数据集大小：526 MB
    - 训练集：325 MB，1500张图像
    - 验证集：201 MB，1000张图像

# 环境要求

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO2017。

    `scripts`中提供了自动下载COCO2017的脚本`run_download_dataset.sh`，可使用此脚本自动下载并解压数据集。
    目前仅支持COCO2017数据集自动下载。

```shell script
bash run_download_dataset.sh [DATASET] [CONFIG_FILE]
# 例如：
bash run_download_dataset.sh coco config/ssd_resnet50_fpn_config.yaml
```

- 本示例默认使用COCO2017作为训练数据集，您也可以使用自己的数据集。

    1. 如果使用coco数据集。**执行脚本时选择数据集coco。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install Cython

        pip install pycocotools

        ```

        并在yaml文件中更改COCO_ROOT和其他您需要的设置。目录结构如下：

        ```text
        .
        └─cocodataset
          ├─annotations
            ├─instance_train2017.json
            └─instance_val2017.json
          ├─val2017
          └─train2017

        ```

    2. 如果使用自己的数据集。**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行如下：

        ```text
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。我们从`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。在`*yaml`中设置`IMAGE_DIR`和`ANNO_PATH`。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```shell script
# Ascend分布式训练
bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [CONFIG_PATH]
```

```shell script
# Ascend单卡训练
bash run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH]
```

```shell script
# Ascend处理器环境运行eval
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

- GPU处理器环境运行

```shell script
# GPU分布式训练
bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH]
```

```shell script
# GPU处理器环境运行eval
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2)设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (3) 在modelarts的界面上设置代码的路径 "/path/ssd"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/ssd"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ ssd
    ├─ README.md                      ## SSD相关说明
    ├─ ascend310_infer                ## 实现310推理源代码
    ├─ scripts
      ├─ docker_start.sh              ## 容器启动脚本
      ├─ run_distribute_train.sh      ## Ascend分布式shell脚本
      ├─ run_distribute_train_gpu.sh  ## GPU分布式shell脚本
      ├─ run_download_dataset.sh      ## 下载数据集shell脚本
      ├─ run_eval.sh                  ## Ascend评估shell脚本
      ├─ run_eval_gpu.sh              ## GPU评估shell脚本
      ├─ run_export.sh                ## MINDIR、AIR、ONNX导出脚本
      └─ run_infer_cpp.sh             ## C++推理脚本
    ├─ src
      ├─ __init__.py                      ## 初始化文件
      ├─ anchor_generator.py              ## 锚点生成器
      ├─ box_util.py                      ## bbox工具
      ├─ dataset.py                       ## 创建并处理数据集
      ├─ eval_callback.py                 ## eval回调方法定义
      ├─ eval_utils.py                    ## eval工具
      ├─ fpn.py                           ## 特征金字塔网络
      ├─ init_params.py                   ## 参数工具
      ├─ lr_schedule.py                   ## 学习率生成器
      ├─ mobilenet_v1.py                  ## mobilenet-v1网络定义
      ├─ resnet.py                        ## resnet网络定义
      ├─ ssd.py                           ## SSD架构
      └─ vgg16.py                         ## vgg16网络定义
      ├── model_utils
      │   ├──config.py                    ## 参数配置
      │   ├──device_adapter.py            ## 设备配置
      │   ├──local_adapter.py             ## 本地设备配置
      │   ├──moxing_adapter.py            ## modelarts设备配置
    ├─ config
        ├─ ssd_mobilenet_v1_300_config_gpu.yaml ## 参数配置
        ├─ ssd_mobilenet_v1_fpn_config.yaml ## 参数配置
        ├─ ssd_resnet50_fpn_config.yaml ## 参数配置
        ├─ ssd_vgg16_config.yaml ## 参数配置
        ├─ ssd300_config.yaml ## 参数配置
        ├─ ssd_mobilenet_v1_fpn_config_gpu.yaml ## GPU参数配置
        ├─ ssd_resnet50_fpn_config_gpu.yaml ## GPU参数配置
        ├─ ssd_vgg16_config_gpu.yaml ## GPU参数配置
        ├─ ssd300_config_gpu.yaml ## GPU参数配置
        └─ ssd_mobilenet_v1_fpn_ONNX_config.yaml ## 参数配置
    ├─ Dockerfile                         ## docker文件
    ├─ download_dataset.py                ## 下载数据集脚本
    ├─ eval.py                            ## 评估脚本
    ├─ export.py                          ## 导出 AIR,MINDIR模型的脚本
    ├─ postprocess.py                     ## 310推理后处理脚本
    ├─ train.py                           ## 训练脚本
    └─ mindspore_hub_conf.py              ## MindSpore Hub接口
```

## 脚本参数

  ```text
  yaml文件中中主要参数如下：

    "device_num": 1                            # 使用设备数量
    "lr": 0.05                                 # 学习率初始值
    "dataset": coco                            # 数据集名称
    "epoch_size": 500                          # 轮次大小
    "batch_size": 32                           # 输入张量的批次大小
    "pre_trained": None                        # 预训练检查点文件路径
    "pre_trained_epoch_size": 0                # 预训练轮次大小
    "save_checkpoint_epochs": 10               # 两个检查点之间的轮次间隔。默认情况下，每10个轮次都会保存检查点
    "loss_scale": 1024                         # 损失放大
    "data_path": "your_path/data"              # your_path是你自己的路径，一定要是绝对路径
    "output_path": "your_path/data/train"      # your_path是你自己的路径，一定要是绝对路径
    "load_path": "your_path/data/checkpoint"   # your_path是你自己的路径，一定要是绝对路径

    "class_num": 81                            # 数据集类数
    "img_shape": [300, 300]                    # 作为模型输入的图像高和宽
    "mindrecord_dir": "/data/MindRecord_COCO"  # MindRecord路径
    "coco_root": "your_path/cocodataset"       # COCO2017数据集路径
    "voc_root": ""                             # VOC原始数据集路径
    "image_dir": ""                            # 其他数据集图片路径，如果使用coco或voc，此参数无效。
    "anno_path": ""                            # 其他数据集标注路径，如果使用coco或voc，此参数无效。

  ```

## 训练过程

运行`train.py`训练模型。如果`mindrecord_dir`为空，则会通过`coco_root`（coco数据集）或`image_dir`和`anno_path`（自己的数据集）生成[MindRecord](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset/record.html)文件。**注意，如果mindrecord_dir不为空，将使用mindrecord_dir代替原始图像。**

### Ascend上训练

- 分布式

```shell script
    bash run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [RANK_TABLE_FILE] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `RANK_TABLE_FILE`：[rank_table.json](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)的路径。最好使用绝对路径。
- `CONFIG_PATH`: 参数配置。
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

```shell script
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [EPOCH_SIZE] [LR] [DATASET] [CONFIG_PATH] [PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)
```

此脚本需要五或七个参数。

- `DEVICE_NUM`：分布式训练的设备数。
- `EPOCH_NUM`：分布式训练的轮次数。
- `LR`：分布式训练的学习率初始值。
- `DATASET`：分布式训练的数据集模式。
- `CONFIG_PATH`: 参数配置。
- `PRE_TRAINED`：预训练检查点文件的路径。最好使用绝对路径。
- `PRE_TRAINED_EPOCH_SIZE`：预训练的轮次数。

    训练结果保存在当前路径中，文件夹名称以"LOG"开头。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 1, loss is 420.11783
epoch: 1 step: 2, loss is 434.11032
epoch: 1 step: 3, loss is 476.802
...
epoch: 1 step: 458, loss is 3.1283689
epoch time: 150753.701, per step time: 329.157
...

```

## 评估过程

### Ascend处理器环境评估

```shell script
bash run_eval.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

此脚本需要四个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。
- `CONFIG_PATH`: 参数配置。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.238
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.400
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.240
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.250
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.389
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.424
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697

========================================

mAP: 0.23808886505483504
```

### GPU处理器环境评估

```shell script
bash run_eval_gpu.sh [DATASET] [CHECKPOINT_PATH] [DEVICE_ID] [CONFIG_PATH]
```

此脚本需要四个参数。

- `DATASET`：评估数据集的模式。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `DEVICE_ID`: 评估的设备ID。
- `CONFIG_PATH`: 参数配置。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.224
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.375
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.228
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.189
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.407
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.243
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.382
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.417
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.425
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686

========================================

mAP: 0.2244936111705981
```

### ONNX处理器环境评估

```shell script
bash run_eval_onnx.sh <DATA_PATH> <COCO_ROOT> <ONNX_MODEL_PATH> [<INSTANCES_SET>] [<DEVICE_TARGET>] [<CONFIG_PATH>]
```

此脚本需要三个参数。

- `DATA_PATH`：mindspore评估生成的data路径。
- `COCO_ROOT`：COCO2017数据集路径。
- `ONNX_MODEL_PATH`: onnx模型的路径。

推理结果保存在示例路径中您可以在eval.log日志中找到类似以下的结果。

```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.242
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.688
  mAP: 0.23850595066045968
```

## 推理过程

### Ascend310执行推理

#### [导出MindIR](#contents)

本地导出mindir

```shell
python export.py --checkpoint_file_path /path/to/ssd.ckpt --file_name /path/to/ssd.onnx --file_format ONNX --config_path config/ssd300_config_gpu.yaml --batch_size 1
```

参数ckpt_file为必填项，
`FILE_FORMAT` 必须在 ["AIR", "MINDIR", "ONNX"]中选择。

也可使用`scripts`中的shell脚本进行导出，仅需给出配置脚本路径和导出类型即可。导出类型可选`AIR`/`MINDIR`/`ONNX`。

```shell
bash run_export.sh [CONFIG_FILE_PATH] [FILE_FORMAT]
```

ModelArts导出mindir

```python
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./ssd'"参数在yaml文件。
#          设置 "file_format: 'MINDIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./ssd'"参数在modearts的界面上。
#          设置 "file_format: 'MINDIR'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/ssd"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

### 执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。精度计算过程需要70G+的内存，否则进程将会因为超出内存被系统终止。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [CONFIG_PATH] [DEVICE_TYPE] [DEVICE_ID]

```

- `DVPP` 为必填项，需要在["DVPP", "CPU"]选择，大小写均可。需要注意的是ssd_vgg16执行推理的图片尺寸为[300, 300]，由于DVPP硬件限制宽为16整除，高为2整除，因此，这个网络需要通过CPU算子对图像进行前处理。
- `DEVICE_ID` 可选，默认值为0。

#### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.339
Average Precision (AP) @[ IoU=0.50      | area= all   | maxDets=100 ] = 0.521
Average Precision (AP) @[ IoU=0.75      | area= all   | maxDets=100 ] = 0.370
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=  1 ] = 0.310
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets= 10 ] = 0.481
Average Recall    (AR) @[ IoU=0.50:0.95 | area= all   | maxDets=100 ] = 0.515
Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
mAP: 0.33880018942412393
```

### ONNX推理

#### 导出ONNX

```shell
python export.py --checkpoint_file_path /path/to/ssd.ckpt --file_name /path/to/ssd.onnx --file_format ONNX --config_path config/ssd300_config_gpu.yaml --batch_size 1
```

参数ckpt_file为必填项，
`FILE_FORMAT` 选择ONNX。

#### 执行推理

目前仅支持batch_Size为1的推理，batch_Size在推理脚本中已经给出。

```shell
/bin/bash ./infer_ssd_mobilenet_v1_fpn_onnx.sh <DATA_PATH> <COCO_ROOT> <ONNX_MODEL_PATH> [<INSTANCES_SET>] [<DEVICE_TARGET>] [<CONFIG_PATH>]

# 示例
/bin/bash ./infer_ssd_mobilenet_v1_fpn_onnx.sh ../cocodataset/ val2017 ../SSDMOBILE.onnx /home/workspace/ssd/cocodataset/annotations/instances_{}.json GPU ../config/ssd_mobilenet_v1_fpn_ONNX_config.yaml

```

#### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702

========================================

mAP: 0.3510553464069668
```

# 模型描述

## 性能

| 参数         | Ascend                                                       | GPU                                                    | CPU                                                               |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ |
| 网络         | SSD MobileNetV2                                              | SSD MobileNetV2                                        | SSD MobileNetV2                                        |
| 资源         | Ascend 910; OS Euler2.8                                      | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G | CPU                                                          |
| 上传日期     | 07/05/2020 (month/day/year)                                  | 09/24/2020 (month/day/year)                            | 11/05/2020 (month/day/year)                                  |
| 昇思版本     | 1.3.0                                                        | 1.3.0                                                  | 1.3.0                                                        |
| 数据集       | COCO2017                                                     | COCO2017                                               | helmet                                                     |
| 训练参数     | epoch = 500,  batch_size = 32                                | epoch = 800,  batch_size = 24(8ps)/32(1ps)             | epoch = 40,  batch_size = 16                               |
| 优化器       | Momentum                                                     | Momentum                                               | Momentum                                               |
| 损失函数     | Sigmoid Cross Entropy,SmoothL1Loss                           | Sigmoid Cross Entropy,SmoothL1Loss                     | Sigmoid Cross Entropy,SmoothL1Loss                     |
| 性能         | 8pcs: 90ms/step                                              | 8pcs: 121ms/step                                       | 1230ms/step                                                      |
| 训练耗时     | 8pcs: 4.81hours                                              | 8pcs: 12.31hours                                       | 1.2h                                                              |
| 推理输出     | mAP                                                          | mAP                                                    | mAP                                                         |
| 评价指标     | IoU=0.50: 22%                                                | IoU=0.50: 22%                                          | IoU=0.50: 49.0%                                                |
| 推理模型大小 | 34M(.ckpt file)                                              | 34M(.ckpt file)                                        | 34M(.ckpt file)                                              |
| 参数文件     | ssd300_config.yaml                                           | ssd300_config_gpu.yaml                                 | ssd300_config.yaml                                           |
| 脚本链接     | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |                                                        |                                                        |

| 参数         | Ascend                                                       | GPU                                                    |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| 网络         | SSD-MobileNetV1-FPN                                          | SSD-MobileNetV1-FPN                                    |
| 资源         | Ascend 910; OS Euler2.8                                      | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G |
| 上传日期     | 11/14/2020 (month/day/year)                                  | 07/23/2021 (month/day/year)                            |
| 昇思版本     | 1.3.0                                                        | 1.3.0                                                  |
| 数据集       | COCO2017                                                     | COCO2017                                               |
| 训练参数     | epoch = 60,  batch_size = 32                                 | epoch = 60,  batch_size = 16                           |
| 优化器       | Momentum                                                     | Momentum                                               |
| 损失函数     | Sigmoid Cross Entropy,SmoothL1Loss                           | Sigmoid Cross Entropy,SmoothL1Loss                     |
| 性能         | 8pcs: 408 ms/step                                            | 8pcs: 640 ms/step                                      |
| 训练耗时     | 8pcs: 4.5 hours                                              | 8pcs: 9.7 hours                                        |
| 推理输出     | mAP                                                          | mAP                                                    |
| 评价指标     | IoU=0.50: 29.1 %                                             | IoU=0.50: 29.1 %                                       |
| 推理模型大小 | 96M(.ckpt file)                                              | 96M(.ckpt file)                                        |
| 参数文件     | ssd_mobilenet_v1_fpn_config.yaml                             | ssd_mobilenet_v1_fpn_config_gpu.yaml                   |
| 脚本链接     | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |                                                        |

| 参数         | Ascend                                                       | GPU                                                    |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| 网络         | SSD-Resnet50-FPN                                             | SSD-Resnet50-FPN                                       |
| 资源         | Ascend 910; OS Euler2.8                                      | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G |
| 上传日期     | 03/10/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                            |
| 昇思版本     | 1.3.0                                                        | 1.3.0                                                  |
| 数据集       | COCO2017                                                     | COCO2017                                               |
| 训练参数     | epoch = 60,  batch_size = 32                                 | epoch = 60,  batch_size = 16                           |
| 优化器       | Momentum                                                     | Momentum                                               |
| 损失函数     | Sigmoid Cross Entropy,SmoothL1Loss                           | Sigmoid Cross Entropy,SmoothL1Loss                     |
| 性能         | 8pcs: 345 ms/step                                            | 8pcs: 877 ms/step                                      |
| 训练耗时     | 8pcs: 4.1 hours                                              | 8pcs: 12 hours                                         |
| 推理输出     | mAP                                                          | mAP                                                    |
| 评价指标     | IoU=0.50: 34.3%                                              | IoU=0.50: 34.3 %                                       |
| 推理模型大小 | 255M(.ckpt file)                                             | 255M(.ckpt file)                                       |
| 参数文件     | ssd_resnet50_fpn_config.yaml                                 | ssd_resnet50_fpn_config_gpu.yaml                       |
| 脚本链接     | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |                                                        |

| 参数         | Ascend                                                       | GPU                                                    |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| 网络         | SSD VGG16                                                    | SSD VGG16                                              |
| 资源         | Ascend 910; OS Euler2.8                                      | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G |
| 上传日期     | 03/27/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                            |
| 昇思版本     | 1.3.0                                                        | 1.3.0                                                  |
| 数据集       | COCO2017                                                     | COCO2017                                               |
| 训练参数     | epoch = 150,  batch_size = 32                                | epoch = 150,  batch_size = 32                          |
| 优化器       | Momentum                                                     | Momentum                                               |
| 损失函数     | Sigmoid Cross Entropy,SmoothL1Loss                           | Sigmoid Cross Entropy,SmoothL1Loss                     |
| 性能         | 8pcs: 117 ms/step                                            | 8pcs: 403 ms/step                                      |
| 训练耗时     | 8pcs: 4.81hours                                              | 8pcs: 16.8 hours                                       |
| 推理输出     | mAP                                                          | mAP                                                    |
| 评价指标     | IoU=0.50: 23.2%                                              | IoU=0.50: 23.2%                                        |
| 推理模型大小 | 186M(.ckpt file)                                             | 186M(.ckpt file)                                       |
| 参数文件     | ssd_vgg16_config.yaml                                        | ssd_vgg16_config_gpu.yaml                              |
| 脚本链接     | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |                                                        |

| 参数         | GPU                                                          |
| ------------ | ------------------------------------------------------------ |
| 网络         | SSD MobileNetV1                                              |
| 资源         | GPU(Tesla V100 PCIE)，CPU 2.1GHz 64 cores，Memory 128G       |
| 上传日期     | 03/03/2022 (month/day/year)                                  |
| 昇思版本     | 1.5.0                                                        |
| 数据集       | COCO2017                                                     |
| 训练参数     | epoch = 500,  batch_size = 32                                |
| 优化器       | Momentum                                                     |
| 损失函数     | Sigmoid Cross Entropy,SmoothL1Loss                           |
| 性能         | 8pcs: 108 ms/step                                            |
| 训练耗时     | 8pcs: 6.87hours                                              |
| 推理输出     | mAP                                                          |
| 评价指标     | IoU=0.50: 21.5%                                              |
| 推理模型大小 | 88M(.ckpt file)                                              |
| 参数文件     | ssd_mobilenet_v1_300_config_gpu.yaml                         |
| 脚本链接     | <https://gitee.com/mindspore/models/tree/master/official/cv/SSD> |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。