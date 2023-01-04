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
    - [评估过程](#评估过程)
        - [Ascend处理器环境评估](#ascend处理器环境评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
        - [性能](#性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SSD说明

SSD将边界框的输出空间离散成一组默认框，每个特征映射位置具有不同的纵横比和尺度。在预测时，网络对每个默认框中存在的对象类别进行评分，并对框进行调整以更好地匹配对象形状。此外，网络将多个不同分辨率的特征映射的预测组合在一起，自然处理各种大小的对象。

[论文](https://arxiv.org/abs/1512.02325)：   Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.European Conference on Computer Vision (ECCV), 2016 (In press).

# 模型架构

SSD方法基于前向卷积网络，该网络产生固定大小的边界框集合，并针对这些框内存在的对象类实例进行评分，然后通过非极大值抑制步骤进行最终检测。早期的网络层基于高质量图像分类的标准体系结构，被称为基础网络。后来通过向网络添加辅助结构进行检测。

# 数据集

使用的数据集： COCO 14 mini。验证集所有ID号为[COCO 14 minival](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt),共8059张图像，其中996张选自[COCO2017](<http://images.cocodataset.org/>)的验证集，剩余7063张选自[COCO2017](<http://images.cocodataset.org/>)的训练集。

生成数据集做法：根据COCO2017生成目标数据集的方法为运行`train.py`，其中参数`--data_complete`设为`False`，或直接运行`create_dataset.sh`。

另外需要从[这里](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt)下载coco14mini的验证数据ID，并保存为`number_id_val.txt`。

- 数据集大小：18.7GB
    - 训练集：17.5G 115228张图像  
    - 验证集：1.22G 8059张图像
    - 标注：526 MB，图像信息，框等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理

# 环境要求

- 安装[MindSpore](https://www.mindspore.cn/install)。

- 下载数据集COCO 14 mini。

- 本示例默认使用COCO 14 mini作为训练数据集，您也可以使用自己的数据集。

    1. 如果使用coco数据集。**执行脚本时选择数据集coco。**
        安装Cython和pycocotool，也可以安装mmcv进行数据处理。

        ```python
        pip install pycocotools

        ```

        并在`config.py`中更改COCO_ROOT和其他您需要的设置。目录结构如下：

        ```text
        .
        └─coco
          ├─annotations
            ├─instance_train.json
            └─instance_val.json
          ├─val
          └─train

        ```

    2. 如果使用自己的数据集。**执行脚本时选择数据集为other。**
        将数据集信息整理成TXT文件，每行如下：

        ```text
        train/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2

        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。我们从`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。在`config.py`中设置`IMAGE_DIR`和`ANNO_PATH`。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```shell script
# Ascend分布式训练
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional)
```

```shell script
# 单卡训练
bash run_standalone_train.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH]
```

```shell script
# Ascend处理器环境运行eval
bash run_eval.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]
```

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ ssd_inceptionv2
    ├─ README.md                      # SSD相关说明
    ├─ ascend310_infer                # 实现310推理源代码
    ├─ scripts
      ├─ run_distribute_train.sh      # Ascend分布式shell脚本
      ├─ run_infer_310.sh             # Ascend推理shell脚本
      ├─ create_dataset.sh            # 生成数据集shell脚本
      └─ run_eval.sh                  # Ascend评估shell脚本
    ├─ src
      ├─ __init__.py                  # 初始化文件
      ├─ box_util.py                  # bbox工具
      ├─ callback.py                  # 边训练边验证工具
      ├─ config.py                    # 配置文件入口
      ├─ config_ssd_inception_v2.py   # 网络总配置
      ├─ dataset.py                   # 创建并处理数据集
      ├─ eval_utils.py                # 验证工具
      ├─ feature_map_generators.py    # 生成网络特征层
      ├─ inception_v2.py              # 骨干网网络结构
      ├─ init_params.py               # 参数工具
      ├─ lr_schedule.py               # 学习率生成器
      ├─ ssd.py                       # 损失函数等定义
      ├─ number_id_val.txt            # 验证数据集的id号
      └─ ssd_inception_v2.py          # ssd_inceptionv2架构
    ├─ eval.py                        # 评估脚本
    └─ train.py                       # 训练脚本
```

## 脚本参数

  ```text
  train.py和config.py中主要参数如下：

    "device_num": 1                            # 使用设备数量
    "lr": 0.09                                 # 学习率初始值
    "dataset": coco                            # 数据集名称
    "epoch_size": 1000                         # 轮次大小
    "batch_size": 32                           # 输入张量的批次大小
    "pre_trained": ""                          # 预训练检查点文件路径
    "backbone_pre_trained": ""                 # 骨干网预训练检查点文件路径
    "pre_trained_epoch_size": 0                # 预训练轮次大小
    "save_checkpoint_epochs": 1                # 两个检查点之间的轮次间隔。默认情况下，每1个轮次都会保存检查点。
    "loss_scale": 1024                         # 损失放大

    "class_num": 81                            # 数据集类数
    "image_shape": [300, 300]                  # 作为模型输入的图像高和宽
    "mindrecord_dir": "/data/mindrecord"       # MindRecord路径
    "coco_root": "/data/coco"                  # COCO14 mini数据集路径
    "coco_root_raw": "/data/coco"              # COCO2017 数据集路径
    "voc_root": ""                             # VOC原始数据集路径
    "image_dir": ""                            # 其他数据集图片路径，如果使用coco或voc，此参数无效。
    "anno_path": ""                            # 其他数据集标注路径，如果使用coco或voc，此参数无效。

  ```

## 训练过程

运行`train.py`训练模型。如果`mindrecord_dir`为空，则会通过`coco_root`（coco数据集）或`image_dir`和`anno_path`（自己的数据集）生成[MindRecord](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset/record.html)文件。**注意，如果mindrecord_dir不为空，将使用mindrecord_dir代替原始图像。**

### Ascend上训练

- 分布式

```shell script
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [DATASET_PATH] [MINDRECORD_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional)
```

此脚本需要五或六个参数。

- `RANK_TABLE_FILE`：[rank_table.json](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)的路径。最好使用绝对路径。
- `DATASET`：分布式训练的数据集模式。
- `DATASET_PATH`：分布式训练的原始数据集路径。
- `MINDRECORD_PATH`：分布式训练的mindrecord类型数据集路径。
- `TRAIN_OUTPUT_PATH`：输出检查点的文件路径。
- `PRE_TRAINED_PATH`：预训练检查点文件的路径。最好使用绝对路径。

    训练结果保存在当前路径下新生成的以`train_parallel`开头的文件中，文件夹名称为"log"。  您可在此文件夹中找到检查点文件以及结果，如下所示。

```text
epoch: 1 step: 446, loss is 3.4383242
epoch time: 1152301.622 ms, per step time: 2583.636 ms
epoch: 2 step: 446, loss is 2.8700542
epoch time: 24801.569 ms, per step time: 55.609 ms
epoch: 3 step: 446, loss is 2.60494
epoch time: 24826.825 ms, per step time: 55.666 ms
...

epoch: 998 step: 446, loss is 1.2264478
epoch time: 25357.279 ms, per step time: 56.855 ms
epoch: 999 step: 446, loss is 1.4033155
epoch time: 25396.187 ms, per step time: 56.942 ms
epoch: 1000 step: 446, loss is 1.3579155
epoch time: 24013.690 ms, per step time: 53.842 ms
```

## 评估过程

### Ascend处理器环境评估

```shell script
bash run_eval.sh [DEVICE_ID] [DATASET] [DATASET_PATH] [CHECKPOINT_PATH] [MINDRECORD_PATH]
```

此脚本需要五个参数。

- `DEVICE_ID`: 评估的设备ID。
- `DATASET`：评估数据集的模式。
- `DATASET_PATH`：原始数据集路径。
- `CHECKPOINT_PATH`：检查点文件的绝对路径。
- `MINDRECORD_PATH`：mindrecord格式数据集路径。

> 在训练过程中可以生成检查点。

推理结果保存在示例路径中，文件夹名称以“eval”开头。您可以在日志中找到类似以下的结果。

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551

========================================

mAP: 0.24446736384477105

```

## 导出过程

### 导出

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT`可选 ["AIR", "MINDIR"]

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [ANNO_FILE] [DEVICE_TYPE] [DEVICE_ID]
```

推理结果被保存到了当前目录，可以在acc.log中获得类似下面的结果。

```shell
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
 mAP:0.24448441137252833
```

### 性能

| 参数          | Ascend                      |
| ------------------- | --------------------- |
| 模型版本       | SSD inceptionv2                    |
| 资源           | Ascend 910                 |
| 上传日期  | 2021-10-19  |
| MindSpore版本    | 1.3.0                    |
| 数据集         | COCO14mini                    |
| mAP | IoU=0.50:0.95: 24.5%              |
| 单step时间 |   55ms            |
| 总训练时间 |    7h27m06s   (八卡)        |
| 收敛的loss值 |     1.03         |
| 模型大小   | 107M（.ckpt文件）            |

参数ckpt_file为必填项，
`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
