# 目录

<!-- TOC -->

- [目录](#目录)
- [EfficientDet d0描述](#EfficientDet b0描述)
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
        - [Ascend评估](#ascend评估)
    - [导出mindir模型](#导出mindir模型)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# EfficientDet d0描述

EfficientDet d0是EfficientDet的最轻量级版本，使用EfficientDet d0 进行检测时，运行时间更短，准确性更低。

[论文](https://arxiv.org/abs/1911.09070):  Mingxing Tan, Ruoming Pang, Quoc V. Le
[代码](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

# 模型架构

EfficientDet整体网络架构如下：

EfficientDet d0是EfficientDet的一个轻量级版本，参数量最小，性能最差。输入图大小为[3, 512, 512]。图片由Efficientnet b0提取出抽象特征图，再送入Bifpn进行特征融合，最后交由regressor和classifier给出预测。

# 数据集

使用的数据集：[COCO 2017](<http://images.cocodataset.org/>)

- 数据集大小：19 GB
    - 训练集：18 GB，118000张图片  
    - 验证集：1GB，5000张图片
    - 标注：241 MB，包含实例，字幕，person_keypoints等
- 数据格式：图片和json文件
    - 标注：数据在dataset.py中处理。

- 数据集

    1. 目录结构如下：

        ```text
        .
        ├── annotations  # 标注jsons
        ├── train2017    # 训练数据集
        └── val2017      # 推理数据集
        ```

    2. 将数据集信息整理成TXT文件，每行如下：

        ```text
        train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
        ```

        每行是按空间分割的图像标注，第一列是图像的相对路径，其余为[xmin,ymin,xmax,ymax,class]格式的框和类信息。`dataset.py`是解析脚本，我们从`image_dir`（数据集目录）和`anno_path`（TXT文件路径）的相对路径连接起来的图像路径中读取图像。`image_dir`和`anno_path`为外部输入。

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

- 运行前的配置：
     自定义参数配置均在src/config.py中进行设置，如学习率, batch_size, epoch_num等。其中src/config.py中定义了一些关于coco数据集路径的参数，如：coco_root，mindrecord_dir。在线下环境中，训练脚本默认从config.py中读取coco_root参数指定的路径中存放的coco数据集，并生成mindrecord文件存入mindrecord_dir参数指定的路径，之后使用mindrecord文件进行模型训练。如mindrecord_dir中已有处理好的数据集，则不再重复进行处理。也可以单独调用src/create_data.py将coco数据集处理为mindrecord形式。Modelarts环境中由于coco数据集过于分散不便传输，采取直接读入mindrecord文件进行训练的方式。故运行前需要调用提供的create_data.py脚本进行数据集创建。

- Minrecord文件创建：

    ```bash
    # Ascend线下环境：
    python src/create_data.py --is_modelarts=False
    # ModelArts:
    python src/create_data.py --is_modelarts=True --data_url=[coco_root] --train_url=[OBS_PATH]
    ```

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

    ```shell
    # 运行单机训练示例
    bash run_standalone_train.sh [DEVICE_ID]
    # 运行分布式训练示例
    bash run_distribute_train.sh [RANK_TABLE_FILE]
    # 运行评估示例
    bash run_eval.sh [CKPT_PATH] [DEVICE_ID]
    ```

- 在 [ModelArts](https://support.huaweicloud.com/modelarts/) 上训练

  ```bash
  # 在modelarts上进行8卡训练（Ascend）
  # (1) 上传原始coco数据集至obs桶
  # (2) 调用上面提到的create_data.py脚本进行数据集创建
  # (3) 调用训练脚本train.py
  #     在网页上设置 "is_modelarts=True"
  #     在网页上设置 "train_url=[存放ckpt的OBS路径]"
  #     在网页上设置 "data_url=[mindrecord数据集OBS路径]"
  # (4) 在网页上设置你的代码路径为 "/path/effdetd0"
  # (5) 创建任务
  #
  # 在modelarts上进行验证（Ascend）
  # (1) 上传你的coco验证集到obs桶上
  # (2) 在网页上设置你的代码路径为 "/path/effdetd0"
  # (3) 在网页上设置启动文件为 "eval.py"
  # (4) 设置参数
  #     在网页上设置"is_modelarts=True"
  #     在网页上设置"data_url"为coco验证集在obs上的路径
  #     在网页上设置"checkpoint_path"为checkpoint文件在obs上的路径
  # (5) 创建任务
  ```

# 脚本说明

## 脚本及样例代码

```bash
└── EfficientDet_d0
    ├── README_CN.md                    // EfficientDet_d0相关说明中文版
    ├── scripts
        ├── run_distribute_train.sh     // Ascend上分布式shell脚本
        ├── run_standalone_train.sh     // Ascend上分布式shell脚本
        └── run_eval.sh                 // Ascend上评估的shell脚本
    ├── src
        ├── efficientnet
            ├── model.py                    // efficientnet定义
            └── utils.py                    // efficientnet用到的工具函数
        ├── efficientdet
            ├── model.py                    // efficientdet各个组成模块的定义
            ├── loss.py                     // 网络的loss定义
            └── utils.py                    // efficientdet用到的工具函数
        ├── config.py                       // 参数配置项
        ├── backbone.py                     // 网络的整体结构
        ├── dataset.py                      // 数据集相关模块
        ├── utils.py                        // 一些工具
        ├── nms.py                          // nms函数定义
        ├── lr_scheduler.py                 // 学习率生成器
        └── create_data.py                  // 用于生成mindrecord数据
    ├── requirements.txt                    // 第三方依赖包
    ├── eval.py                             // 验证脚本
    ├── export.py                           // 导出脚本
    └── train.py                            // 训练脚本
```

## 脚本参数

  ```bash
  --run_platform         实现代码的设备：“Ascend“
  --data_url             云上训练数据集路径(线下训练时读config.py中的路径参数)
  --train_url            云上训练时ckpt的obs输出路径
  --checkpoint_path      继续训练的ckpt路径
  --is_modelarts         是否在云上训练
  --override             eval时是否覆盖之前的输出结果（json文件）
  --nms_threshold        eval时的nms阈值，默认0.5
  ```

## 训练过程

### Ascend上训练

- 单机模式

```shell script
bash run_standalone_train.sh
```

```bash
epoch: 1 step: 7392, loss is 10.422406
epoch time: 4526599.740 ms, per step time: 612.365 ms
epoch: 2 step: 7392, loss is 42.83396
epoch time: 3833618.996 ms, per step time: 518.617 ms
epoch: 3 step: 7392, loss is 10.459145
epoch time: 3833070.669 ms, per step time: 518.543 ms
epoch: 4 step: 7392, loss is 8.264204
epoch time: 3833533.955 ms, per step time: 518.606 ms
epoch: 5 step: 7392, loss is 8.744354
epoch time: 3834358.051 ms, per step time: 518.717 ms
epoch: 6 step: 7392, loss is 5.3504305
epoch time: 3831983.720 ms, per step time: 518.396 ms
epoch: 7 step: 7392, loss is 40.863556
epoch time: 3831745.351 ms, per step time: 518.364 ms
epoch: 8 step: 7392, loss is 6.06872
epoch time: 3832552.226 ms, per step time: 518.473 ms
```

- 分布式模式

```shell
bash run_distribute_train.sh rank_table_8p.json
```

可通过指令grep "loss:" train_parallel0/log.txt查看每步的损失值和时间，如下。其中每个epoch会输出八次，因为是八卡同时进行的。

  ```bash
    ...
  epoch: 2 step: 462, loss is 54.36381
  epoch time: 244711.103 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 12.47888
  epoch time: 244711.051 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 15.332263
  epoch time: 244711.092 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 11.096275
  epoch time: 244710.987 ms, per step time: 529.677 ms
  epoch: 2 step: 462, loss is 13.366732
  epoch time: 244711.051 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 12.979013
  epoch time: 244711.051 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 12.622669
  epoch time: 244711.166 ms, per step time: 529.678 ms
  epoch: 2 step: 462, loss is 8.949266
  epoch time: 244188.903 ms, per step time: 528.547 ms
    ...
  ```

## 评估过程

### Ascend评估

  ```shell
  bash run_eval.sh checkpoint/effdet_d0.ckpt 0
  ```

输入变量为数据集目录路径、模型路径。

您将获得mAP：

  ```bash
 # log.txt
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.223
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.354
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.235
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.246
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.354
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.222
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.338
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.361
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.105
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.552
  ```

## 导出mindir模型

```python
python export.py --pre_trained [CKPT_PATH] --file_format [FILE_FORMAT]
```

参数pre_trained是必需的，FILE_FORMAT 选择 "MINDIR"。

# 模型描述

## 性能

### 评估性能

EfficientDet d0应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

| 参数          | EfficientDet d0                                              |
| ------------- | ------------------------------------------------------------ |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8     |
| 上传日期      | 2021-12-2                                                    |
| MindSpore版本 | 1.3.0                                                        |
| 数据集        | COCO2017                                                     |
| 训练参数      | epoch=500, batch_size=16, lr=0.012, momentum=0.9             |
| 优化器        | Momentum                                                     |
| 损失函数      | Sigmoid交叉熵、Focal Loss                                    |
| 输出          | 预测的回归框位置信息及类别                                   |
| 速度          | 单卡：30imgs/s;  8卡：240imgs/s                              |
| 准确性        | mAP=22.3%(shape=512)                                         |
| 总时长        | 8卡: 70小时                                                  |
| 参数(M)       | 3.9                                                          |
| 脚本          | [EfficientDet d0脚本](https://gitee.com/mindspore/models/tree/master/research/cv/EfficientDet_d0) |

# 随机情况说明

设置了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
