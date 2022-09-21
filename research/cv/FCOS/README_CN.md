# 目录

<!-- TOC -->

- [目录](#目录)
- [FCOS描述](#fcos描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
            - [相关说明](#相关说明)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- TOC -->

# FCOS描述

**FCOS**是  anchor-free 模型，通过在卷积阶段加入FPN、在计算损失阶段加入centerness这一分支，实现了更高的检测精度；通过调整模型的backbone，可以达到44.7%的AP。\
[论文](https://arxiv.org/pdf/1904.01355.pdf): ```FCOS: Fully Convolutional One-Stage Object Detection.```

[官方代码](https://github.com/tianzhi0549/FCOS): <https://github.com/tianzhi0549/FCOS>

# 模型架构

FCOS的网络架构模型Backbone采用resnet50，backbone的C3、C4、C5特征层作为FPN的输入，FPN生成P3,P4,P5,P6,P7特征图，送入后续的检测头Head。每个Head包含3个分支： classification分支：预测类别，图中的C表示类别数，相当于C个二分类 regression分支：回归位置，图中的4表示：l,t,r,b，预测锚点到检测框上下左右四条边界的距离 center-ness：中心度，一个锚点对应一个中心度，用于锚点相对于检测框中心性的判断在检测子网络Head中，分类分支和回归分支都先经过了4个卷积层进行了特征强化。

# 数据集

使用的数据集:[COCO 2017](https://cocodataset.org/#download)

支持的数据集: COCO2017 或者与 MS COCO 格式相同的数据集

支持的标注: COCO2017 或者与 MS COCO 相同格式的标注

- 目录结构如下，由用户定义目录和文件的名称

    ```ext

            ├── dataset
                ├── coco2017
                    ├── annotations
                    │   ├─ train.json
                    │   └─ val.json
                    ├─ train
                    │   ├─picture1.jpg
                    │   ├─ ...
                    │   └─picturen.jpg
                    └─ val
                        ├─picture1.jpg
                        ├─ ...
                        └─picturen.jpg

    ```

- 如果用户需要自定义数据集，则需要将数据集格式转化为coco数据格式，并且，json文件中的数据要和图片数据对应好。

# 环境要求

- 硬件（GPU）
    - 准备GPU处理器搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

- 通过官方网站安装Mindspore后，您可以按照如下步骤进行训练和评估
- 在本地进行训练

    ```shell
  # 通过shell脚本进行8卡训练
  sh run_distribute_train_gpu.sh [TRAIN_DATA_PATH] [ANNO_DATA_PATH] [PRETRAIN_PATH] [CKPT_SAVE_PATH]
    ```

  ```shell
  # 通过shell脚本进行单卡训练
  sh run_standalone_train_gpu.sh [TRAIN_DATA_PATH] [ANNO_DATA_PATH] [PRETRAIN_PATH] [CKPT_SAVE_PATH] [DEVICE_ID] [DEVICE_NUM]
  ```

- 在本地进行评估

    ```shell
    sh run_standalone_eval_gpu.sh [EVAL_DATA_PATH] [ANNO_DATA_PATH] [CKPT_PATH] [DEVICE_ID]
    ```

# 脚本说明

## 脚本及样例代码

```text
├── cv
    ├── FCOS
        ├── README.md                                 // FCOS相关说明
        ├── scripts
        │   ├──run_distribute_train_gpu.sh            // GPU多卡训练脚本
        │   ├──run_standalone_eval_gpu.sh             // GPU单卡推理脚本
        │   ├──run_standalone_train_gpu.sh            // GPU单卡训练脚本
        ├── src
        │   ├──COCO_dataset.py                        // 创建数据集
        │   ├──augment.py                             // 数据增强
        │   ├──resnet.py                              // 骨干网络
        │   ├──eval_utils.py                          // 评估工具
        │   ├──fcos.py                                // FCOS模型网络
        │   ├──fpn_neck.py                            // FPN处理
        │   ├──head.py                                // 模型head
        │   ├──network_define.py                      // 网络定义
        │   ├──config.py                              // 参数配置
        ├── train.py                                  // 训练脚本
        ├── eval.py                                   // 评估脚本
```

## 脚本参数

train.py中主要的参数如下:

```text

--device_num                 使用设备的数量,默认为8
--device_id                  使用设备的卡号，默认值为0
--pretrain_ckpt_path         预训练resnet50权重文件
--ckpt_save_path             训练后保存的检查点文件的绝对完整路径
--train_path                 train2017保存路径
--anno_path                  instances_train2017.json保存路径

```

## 训练过程

### 分布式训练

在GPU设备上，使用shell脚本执行分布式训练示例(8卡)

- 第一步

  ```shell

  # 通过shell脚本进行8卡训练
  sh run_distribute_train_gpu.sh /coco2017/train2017 /coco2017/annotations/instances_train2017.json  resnet50.ckpt  /checkpoint

  ```

  上述shell脚本将在后台运行分布式训练。 得到如下损失值：

  ```log

  epoch: 1 step: 1, loss is 0.7623271
  epoch: 1 step: 1, loss is 0.7853986
  epoch: 1 step: 1, loss is 0.8126975
  epoch: 1 step: 1, loss is 0.63795793
  epoch: 1 step: 1, loss is 0.6717266
  epoch: 1 step: 1, loss is 0.5369471
  epoch: 1 step: 1, loss is 0.50559396
  epoch: 1 step: 1, loss is 0.6490997
  epoch: 1 step: 2, loss is 0.7356057
  epoch: 1 step: 2, loss is 0.7328874
  epoch: 1 step: 2, loss is 0.79695445
  epoch: 1 step: 2, loss is 0.8426137
  epoch: 1 step: 2, loss is 0.87362385
  epoch: 1 step: 2, loss is 0.7765503
  epoch: 1 step: 2, loss is 0.67726403
  epoch: 1 step: 2, loss is 0.48694384

  ```

## 评估过程

### 评估

```shell
sh run_standalone_eval_gpu.sh  /coco2017/val2017 instances_val2017.json  ms8p_24epoch.ckpt 0
```

测试数据集的mAP如下：

```log

   ===============================coco eval result===============================
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.410
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.497
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.509
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.603
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.688

```

# 模型描述

## 性能

### 评估性能

FCOS应用于coco2017

|参数| FCOS-resnet50 |
| -------------------------- | ----------------------------------------------------------- |
|资源| Tesla V100*8；内存：755G；系统：EulerOS 2.8；|
|上传日期|2022年8月16日|
| MindSpore版本|1.5.0-alpha|
|数据集|118000张图像|
|训练参数|epoch=25, batch_size=16, lr=0.001,momentum=0.9|
| 优化器                  | SGD                                                    |
|输出|框和标签|
|速度| 530ms/step|
|总时长|48小时|

# 随机情况说明

代码中对数据集进行了随机augmentation操作，其中含有对图片进行旋转、裁剪操作。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
