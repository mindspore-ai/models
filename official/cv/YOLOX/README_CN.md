# 目录

<!-- TOC -->

- [目录](#目录)
- [YOLOX描述](#yolox描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [单卡训练](#单卡训练)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单卡训练](#单卡训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
            - [python命令启动](#python命令启动)
            - [shell脚本启动](#shell脚本启动)
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

# YOLOX描述

**YOLOX**是 YOLO(You Only Look Once) 系列的 anchor-free 版本，和经典的 YOLOv3~5 版本相比，它的网络设计更加的简单但是却拥有更加优秀的性能！YOLOX
致力于在学术研究和工业界之间架起一座桥梁。了解更多的网络细节，请参考Arxiv论文。\
[论文](https://arxiv.org/pdf/2107.08430.pdf): ```YOLOX: Exceeding YOLO Series in 2021```

[官方代码](https://github.com/Megvii-BaseDetection/YOLOX): <https://github.com/Megvii-BaseDetection/YOLOX>

# 模型架构

作为2021年 YOLO 系列的后起之秀，也为了更加公平的对比，YOLOX 的模型主干网络参考了YOLOv3 的**DarkNet-53**以及YOLOv4~5中的 CSP、Focus 模块、SPP(spatial pyramid
pooling)模块、PANet path-aggregation neck等。DarkNet53以及其它的网络模块具体可以参考YOLOv3、YOLOv4、YOLOv5的设计。为了解决目标检测中分类和回归的冲突问题，YOLOX 将
head中的回归分支和分类分支进行了解耦(Decoupled head)，并且将obj分支加在了回归分支中。

# 数据集

使用的数据集：[COCO 2017](https://cocodataset.org/#download)

支持的数据集：COCO2017 或者与 MS COCO 格式相同的数据集

支持的标注：COCO2017 或者与 MS COCO 相同格式的标注

- 目录结构如下，由用户定义目录和文件的名称

    ```ext

            ├── dataset
                ├── coco2017
                    ├── annotations
                    │   ├─ instances_train2017.json
                    │   └─ instances_val2017.json
                    ├─ train2017
                    │   ├─picture1.jpg
                    │   ├─ ...
                    │   └─picturen.jpg
                    └─ val2017
                        ├─picture1.jpg
                        ├─ ...
                        └─picturen.jpg

    ```

- 如果用户需要自定义数据集，则需要将数据集格式转化为coco数据格式，并且，json文件中的数据要和图片数据对应好。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

- 通过官方网站安装Mindspore后，您可以按照如下步骤进行训练和评估
- 在运行网络之前，准备hccl_8p.json文件，生成hccl_8p.json文件，运行[hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py)

```command
python hccl_tools.py
```

- 训练前推荐安装快速计算mAP计算库，可以显著的加快mAP计算速度，按照方法如下：
- ```cd third_party&&bash build.sh```
- 选择backbone：训练支持 yolox_darknet53 以及 yolox_x, 在训练之前需要指定yaml路径, shell脚本训练通过backbone名称寻找对应的yaml文件
- ```python train.py --config_path=yolox_darknet53.yaml --data_dir=your data dir```
- 训练分为前285轮和后15轮，区别主要在于后15轮的训练关闭了数据增强以及使用了L1 loss。

```shell
# 单卡训练
python train.py --config_path=yolox_darknet53.yaml --is_distributed=0 --data_dir=your data dir
```

```shell
# 通过shell脚本进行8卡训练
bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53
```

- 在本地进行评估

```shell
python eval.py --config_path=yolox_darknet53.yaml --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --per_batch_size=8

# 多卡评估
bash run_distribute_eval.sh xx/dataset/ your_val_ckpt_file_path yolox_darknet53 8 rank_table_8pcs.json
```

# 脚本说明

## 脚本及样例代码

```text
    |----README_CN.md
    |----ascend310_infer
    |    |----build.sh
    |    |----CMakeLists.txt
    |    |----inc
    |    |    |----utils.h
    |    |----src
    |    |    |----main.cc
    |    |    |----utils.cc
    |----model_utils
    |    |----__init__.py
    |    |----config.py
    |    |----device_adapter.py
    |    |----hccl_tools.py
    |    |----local_adapter.py
    |    |----moxing_adapter.py
    |----scripts
    |    |----run_distribute_train.sh
    |    |----run_distribute_eval.sh
    |    |----run_infer_310.sh
    |    |----run_eval.sh
    |    |----run_standalone_train.sh
    |----serving
    |    |----yolox
    |    |    |----1
    |    |    |----paraser.py
    |    |    |----servable_config.py
    |    |----serving_client.py
    |    |----serving_server.py
    |----src
    |    |----__init__.py
    |    |----boxes.py
    |    |----darknet.py
    |    |----initializer.py
    |    |----logger.py
    |    |----network_blocks.py
    |    |----transform.py
    |    |----util.py
    |    |----yolox.py
    |    |----yolox_dataset.py
    |    |----yolo_fpn.py
    |    |----yolo_pafpn.py
    |----third_party
    │    |----__init__.py
    │    |----build.sh
    │    |----cocoeval
    │    │    |----cocoeval.cpp
    │    │    |----cocoeval.h
    │    |----fast_coco_eval_api.py
    │    |----setup.py
    |----train.py
    |----eval.py
    |----predict.py
    |----export.py
    |----postprocess.py
    |----preprocess.py
    |----yolox_darknet53_config.yaml
    |----yolox_x_config.yaml
```

## 脚本参数

train.py中主要的参数如下：

```text

--backbone                  训练的主干网络，默认为yolox_darknet53,你也可以设置为yolox_x
--device_target             实现代码的设备，默认为'Ascend'
--save_graphs               是否保存图文件，默认为False
--aug_epochs                开启数据增强的训练轮次，默认为285
--no_aug_epochs             关闭数据增强的训练轮次，默认为15
--data_dir                  数据集的目录
--need_profiler             是否使用profiler。0表示否，1表示是。默认值：0
--per_batch_size            训练的批处理大小。默认值：8
--max_gt                    图片中gt的最大数量，默认值：70
--num_classes               数据集中类别的个数，默认值：80
--input_size                输入网络的尺度大小，默认值：640
--fpn_strides               fpn缩放的步幅，默认：[8, 16, 32]
--use_l1                    是否使用L1 loss，只有在关闭数据增强的训练轮次中才为True，默认为False
--use_syc_bn                是否开启同步BN，默认True
--n_candidate_k             动态k中候选iou的个数，默认为10
--lr                        学习率，默认为0.01
--min_lr_ratio              学习率衰减比率，默认为0.05
--warmup_epochs             warm up 轮次，默认为5
--weight_decay              权重衰减，默认为0.0005
--momentum                  动量
--log_interval              日志记录间隔步数
--ckpt_interval             保存checkpoint间隔。默认值：-1
--is_save_on_master         在master或all rank上保存ckpt，1代表master，0代表all ranks。 默认值：1
--is_distributed            是否分发训练，1代表是，0代表否。默认值：1
--rank                      分布式本地进程序号。默认值：0
--group_size                设备进程总数。默认值：1
--run_eval                  是否开启边训练边推理。默认为False
--eval_parallel             是否开启并行推理。默认为 True。仅在 run_eval 为 True，并且 is_distributed 为 1 时有效
```

## 训练过程

由于 YOLOX 使用了强大的数据增强，在ImageNet上的预训练模型参数不再重要，因此所有的训练都将从头开始训练。训练分为两步：第一步是从头训练并开启数据增强，第二步是使用第一步训练好的检查点文件作为预训练模型并关闭数据增强训练。

### 单卡训练

在Ascend设备上，使用python脚本直接开始训练(单卡)

- 第一步\
    python命令启动

    ```shell
    # 单卡训练
    python train.py --config_path=yolox_darknet53.yaml --data_dir=~/coco2017 --is_distributed=0
    ```

    shell脚本启动

    ```shell
    bash run_standalone_train.sh  [DATASET_PATH] [BACKBONE]
    ```

### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例(8卡)

- 第一步

```shell
# 通过shell脚本进行8卡训练
bash run_distribute_train.sh xxx/dataset/  rank_table_8pcs.json  yolox_darknet53
```

```log

  上述shell脚本将在后台运行分布式训练。您可以通过train_parallel0/log.txt文件查看结果。得到如下损失值：

    ```log

    ...
    2022-10-10 11:43:14,405:INFO:epoch: [1/300] step: [150/1848], loss: 15.9977, overflow: False, scale: 65536, lr: 0.000003, avg step time: 332.07ms
    2022-10-10 11:43:37,711:INFO:epoch: [1/300] step: [160/1848], loss: 14.6404, overflow: False, scale: 65536, lr: 0.000003, avg step time: 330.58ms
    2022-10-10 11:44:41,012:INFO:epoch: [1/300] step: [170/1848], loss: 16.2315, overflow: False, scale: 65536, lr: 0.000004, avg step time: 330.08ms
    2022-10-10 11:43:44,326:INFO:epoch: [1/300] step: [180/1848], loss: 16.9418, overflow: False, scale: 65536, lr: 0.000004, avg step time: 331.37ms
    2022-10-10 11:43:47,646:INFO:epoch: [1/300] step: [190/1848], loss: 17.1101, overflow: False, scale: 65536, lr: 0.000005, avg step time: 331.87ms
    2022-10-10 11:43:50,943:INFO:epoch: [1/300] step: [200/1848], loss: 16.7288, overflow: False, scale: 65536, lr: 0.000005, avg step time: 329.74ms
    ...

```

## 评估过程

### 评估

#### python命令启动

```shell

python eval.py --data_dir=./dataset/xxx --val_ckpt=your_val_ckpt_file_path --per_batch_size=8

```

backbone参数指定为yolox_darknet53或者yolox_x，上述python命令将在后台运行。您可以通过```%Y-%m-%d_time_%H_%M_%S.log```文件查看结果。

#### shell脚本启动

```shell

bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [BACKBONE] [BATCH_SIZE]

```

```text

   ===============================coco eval result===============================
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.671
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.521
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.522
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.588
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.454
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772

```

## 单图片推理可视化示例

```shell

python predict.py --config_path=yolox_darknet53.yaml --device_target=CPU/Ascend --val_ckpt=your ckpt path --img_path=demo/demo.jpg --conf_thre=0.5

```

## 可视化效果

![image](demo/predict-demo.jpg)

## 导出mindir模型

```shell

python export.py --config_path=yolox_darknet53.yaml --val_ckpt [CKPT_PATH] --file_format [MINDIR/AIR]

```

参数```backbone```用于指定主干网络，你可以选择 yolox_darknet53 或者是 yolox_x ，```val_ckpt```用于指定模型的检查点文件

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 用法

#### 相关说明

- 首先要通过执行export.py导出mindir文件，同理可在配置文件中制定默认backbone的类型
- 通过preprocess.py将数据集转为二进制文件
- 执行postprocess.py将根据mindir网络输出结果进行推理，并保存评估指标等结果

执行完整的推理脚本如下：

```shell

bash run_infer_cpp.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_TYPE] [DEVICE_ID]

```

### 结果

推理结果保存在当前路径，通过cat acc.log中看到最终精度结果。

```text

                            yolox-darknet53
=============================coco eval result==================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.671
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772
                                    yolox-x
=============================coco eval result==================================
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.700
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818

```

# 启动服务

```shell

mindir文件复制到指定目录
cp yolofpn.mindir serving/yolox/1

# 启动服务
python serving_server.py --ip=0.0.0.0 --port=8080

# 客户端发起请求
python serving_client --infer_img=demo/demo.jpg --nms_thre=0.65 --conf_thre=0.5

```

# 模型描述

## 性能

### 评估性能

YOLOX应用于118000张图像上（标注和数据格式必须与COCO 2017相同）

|参数| YOLOX_darknet53                                                    |
| -------------------------- |--------------------------------------------------------------------|
|资源| Ascend 910；CPU 2.60GHz, 192核；内存：755G；系统：EulerOS 2.8；               |
|上传日期| 2022年10月21日                                                        |
| MindSpore版本| 1.8.1-alpha                                                        |
|数据集| coco2017                                                           |
|训练参数| epoch=300, batch_size=8, lr=0.01,momentum=0.9                      |
| 优化器                  | SGD                                                                |
|损失函数| Sigmoid Cross Entropy, Iou Loss, L1 Loss                           |
|输出| 框和标签                                                               |
|速度| 1卡：25FPS；8卡：190FPS (shape=640)                                     |
|总时长| 52小时                                                               |
|微调检查点| 约1000M（.ckpt文件）                                                    |
|脚本| <https://gitee.com/mindspore/models/tree/r2.0/official/cv/YOLOX> |

|参数| YOLOX_x                                                            |
| -------------------------- |--------------------------------------------------------------------|
|资源| Ascend 910；CPU 2.60GHz，192核；内存：755G；系统：EulerOS 2.8；                |
|上传日期| 2022年3月11日                                                         |
| MindSpore版本| 1.3.0-alpha                                                        |
|数据集| 118000张图像                                                          |
|训练参数| epoch=300, batch_size=8, lr=0.04,momentum=0.9                      |
| 优化器                  | Momentum                                                           |
|损失函数| Sigmoid Cross Entropy, Iou Loss, L1 Loss                           |
|输出| 框和标签                                                               |
|损失| 50                                                                 |
|速度| 1卡：12FPS；8卡：93FPS (shape=640)                                      |
|总时长| 106小时                                                              |
|微调检查点| 约1100M（.ckpt文件）                                                    |
|脚本| <https://gitee.com/mindspore/models/tree/r2.0/official/cv/YOLOX> |

### 推理性能

YOLOX应用于118000张图像上（标注和数据格式必须与COCO test 2017相同）

|参数| YOLOX_darknet53                     |
| -------------------------- |-------------------------------------|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G |
|上传日期| 2022年10月21日                         |
| MindSpore版本| 1.3.0-alpha                         |
|数据集| 118000张图像                           |
|批处理大小| 1                                   |
|输出| 边框位置和分数，以及概率                        |
|精度| map = 47.8%(shape=640)              |
|推理模型| 约1000M（.ckpt文件）                     |

|参数| YOLOX_x                             |
| -------------------------- |-------------------------------------|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存：755G |
|上传日期| 2020年10月16日                         |
| MindSpore版本| 1.3.0-alpha                         |
|数据集| 118000张图像                           |
|批处理大小| 1                                   |
|输出| 边框位置和分数，以及概率                        |
|精度| map =50.2%(shape=640)               |
|推理模型| 约1100M（.ckpt文件）                     |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子。在var_init.py中，我们设置了权重初始化的种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
