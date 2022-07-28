
<!-- TOC -->

- <span id="content">[NAS-FPN 描述](#NAS-FPN-描述)</span>
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
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
     - [模型导出](#模型导出)
        - [用法](#usage)
        - [运行](#running)
    - [推理过程](#推理过程)
        - [用法](#usage)
        - [运行](#running)
        - [结果](#outcome)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [推理性能](#推理性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

<!-- /TOC -->

## [NAS-FPN 描述](#content)

NAS-FPN算法源自2019年Google Brain的论文 NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection。该论文NAS技术在目标检测领域的开山之作，主要集中在对FPN架构的搜索,最大创新点在于搜索空间的设置.

[论文](https://arxiv.org/abs/1904.07392)
Ghiasi G, Lin T Y, Le Q V. Nas-fpn: Learning scalable feature pyramid architecture for object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 7036-7045.

## [模型架构](#content)

NAS-FPN的整体网络架构如下所示：

[链接](https://arxiv.org/pdf/1904.07392.pdf)

## [数据集](#content)

数据集可参考文献.

MSCOCO2017

- 数据集大小: 19.3G, 123287张80类彩色图像

    - 训练:19.3G, 118287张图片

    - 测试:1814.3M, 5000张图片

- 数据格式:RGB图像.

    - 注意：数据将在src/dataset.py 中被处理

## [环境要求](#content)

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。
- 架构
    - [MindSpore](https://www.mindspore.cn/install)
- 想要获取更多信息，请检查以下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [脚本说明](#content)

### [脚本和示例代码](#content)

```NAS_FPN
.
└─NAS_FPN
  ├─README.md
  ├──ascend310_infer                          # Ascend310推理
  ├─scripts
    ├─run_single_train.sh                     # 使用Ascend环境单卡训练
    ├─run_distribute_train.sh                 # 使用Ascend环境八卡并行训练
    ├─run_eval.sh                             # 使用Ascend环境运行推理脚本
    ├─run_infer_310.sh                        # Ascend推理shell脚本
  ├─src
    ├─dataset.py                              # 数据预处理
    ├─retinanet_nasfpn.py                     # 整个网络模型定义
    ├─init_params.py                          # 参数初始化
    ├─lr_generator.py                         # 学习率生成函数
    ├─loss.py                                 # 网络损失
    ├─merge_cells.py                          # cell合并
    ├─nasfpn.py                               # nasfpn
    ├─resnet.py                               # resnet主干网络
    ├─retinahead.py                           # 检测头
    ├─box_utils.py                            # 先验框设置
    ├─coco_eval.py                            # coco数据集评估
    ├──model_utils
      ├──config.py                            # 参数生成
      ├──device_adapter.py                    # 设备相关信息
      ├──local_adapter.py                     # 设备相关信息
      ├──moxing_adapter.py                    # 装饰器(主要用于ModelArts数据拷贝)
  ├─train.py                                  # 网络训练脚本
  ├─export.py                                 # 导出 AIR,MINDIR模型的脚本
  ├─postprogress.py                           # 310推理后处理脚本
  └─eval.py                                   # 网络推理脚本
  └─create_data.py                            # 构建Mindrecord数据集脚本
  └─default_config.yaml                       # 参数配置

```

### [脚本参数](#content)

```default_config.yaml
在脚本中使用到的主要参数是:
"img_shape": [640, 640],                                                                        # 图像尺寸
"num_retinanet_boxes": 76725,                                                                   # 设置的先验框总数
"match_thershold": 0.5,                                                                         # 匹配阈值
"nms_thershold": 0.6,                                                                           # 非极大抑制阈值
"min_score": 0.1,                                                                               # 最低得分
"max_boxes": 100,                                                                               # 检测框最大数量
"global_step": 0,                                                                               # 全局步数
"lr_init": 1e-6,                                                                                # 初始学习率
"lr_end_rate": 5e-3,                                                                            # 最终学习率与最大学习率的比值
"warmup_epochs1": 2,                                                                            # 第一阶段warmup的周期数
"warmup_epochs2": 5,                                                                            # 第二阶段warmup的周期数
"warmup_epochs3": 23,                                                                           # 第三阶段warmup的周期数
"warmup_epochs4": 60,                                                                           # 第四阶段warmup的周期数
"warmup_epochs5": 160,                                                                          # 第五阶段warmup的周期数
"momentum": 0.9,                                                                                # momentum
"gamma": 2.0,                                                                                   # focal loss中的参数
"alpha": 0.75,                                                                                  # focal loss中的参数
"nasfpn_input_channels":[512, 1024, 2048],                                                      # 主干网络输出通道数量
"nasfpn_out_channel": 256,                                                                      # nasfpn输出通道数量
"nasfpn_num_outs": 5,                                                                           # nasfpn输出尺度数量
"nasfpn_stack_times": 7,                                                                        # nasfpn叠加数量
"nasfpn_start_level": 0,                                                                        # nasfpn输入开始的索引
"nasfpn_end_level": -1,                                                                         # nasfpn输入结束的索引
"epoch_size": 300,                                                                              # epoch
"batch_size": 16                                                                                # batch size
```

### [训练过程](#content)

#### 用法

使用shell脚本进行训练。shell脚本的用法如下:

```训练
# 八卡并行训练示例：

创建 RANK_TABLE_FILE
bash scripts/run_distribute_train.sh DEVICE_NUM RANK_TABLE_FILE MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

# 单卡训练示例：

bash scripts/run_single_train.sh DEVICE_ID MINDRECORD_DIR PRE_TRAINED(optional) PRE_TRAINED_EPOCH_SIZE(optional)

```

> 注意:
RANK_TABLE_FILE相关参考资料见[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html), 获取device_ip方法详见[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

#### 运行

```cocodataset
数据集结构
└─cocodataset
  ├─train2017
  ├─val2017
  ├─test2017
  ├─annotations
```

```default_config.yaml
训练前，先创建MindRecord文件，以COCO数据集为例，yaml文件配置好coco数据集路径和mindrecord存储路径
# your cocodataset dir
coco_root: /home/DataSet/cocodataset/
# mindrecord dataset dir
mindrecord_dr: /home/DataSet/MindRecord_COCO
```

```MindRecord
# 生成训练数据集
python create_data.py --create_dataset coco --prefix retinanet.mindrecord --is_training True

# 生成测试数据集
python create_data.py --create_dataset coco --prefix retinanet_eval.mindrecord --is_training False
```

```bash
Ascend:
# 八卡并行训练示例：
bash scripts/run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [MINDRECORD_DIR] [PRE_TRAINED(optional)] [PRE_TRAINED_EPOCH_SIZE(optional)]
# example: bash scripts/run_distribute_train.sh 8 ~/rank_table_8pcs.json /home/DataSet/MindRecord_COCO/

# 单卡训练示例：
bash scripts/run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR]
# example: bash scripts/run_single_train.sh 0 /home/DataSet/MindRecord_COCO/
```

#### 结果

训练结果将存储在示例路径中。checkpoint将存储在 `./ckpt` 路径下，训练日志将被记录到 `./log.txt` 中，训练日志部分示例如下：

```训练日志
epoch: 2 step: 916, loss is 32.88424
lr:[0.000003]
epoch time: 406803.542 ms, per step time: 444.109 ms
epoch: 3 step: 916, loss is 12.249367
lr:[0.000028]
epoch time: 405872.029 ms, per step time: 443.092 ms
epoch: 4 step: 916, loss is 9.737508
lr:[0.000046]
epoch time: 405853.168 ms, per step time: 443.071 ms
epoch: 5 step: 916, loss is 10.690951
lr:[0.000064]
epoch time: 405932.714 ms, per step time: 443.158 ms
epoch: 6 step: 916, loss is 3.598938
lr:[0.000082]
epoch time: 405846.706 ms, per step time: 443.064 ms
```

- 如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── MindRecord_COCO                                              # dir
#    ├── annotations                                                # annotations dir
#       ├── instances_val2017.json                                  # annotations file
#    ├── checkpoint                                                 # checkpoint dir
#    ├── pred_train                                                 # predtrained dir
#    ├── MindRecord_COCO.zip                                        # train mindrecord file and eval mindrecord file

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "distribute=True"
#          设置 "keep_checkpoint_max=5"
#          设置 "save_checkpoint_path=/cache/train/checkpoint"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "epoch_size=300"
#          设置 "modelarts_dataset_unzip_name=MindRecord_COCO"
#          设置 "pre_trained=/cache/data/train/train_predtrained/pred file name" 如果没有预训练权重 pre_trained=""

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/NAS_FPN"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#        a.设置 "enable_modelarts=True"
#          设置 "mindrecord_dir=/cache/data/MindRecord_COCO"
#          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"
#          设置 "instance_set=/cache/data/MindRecord_COCO/annotations/instances_{}.json"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/NAS_FPN"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

### [评估过程](#content)

#### <span id="usage">用法</span>

使用shell脚本进行评估。shell脚本的用法如下:

```eval
bash scripts/run_eval.sh [DEVICE_ID] [DATASET] [MINDRECORD_DIR] [CHECKPOINT_PATH] [ANN_FILE PATH]
# example: bash scripts/run_eval.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/nasfpn/ckpt/retinanet_nasfpn_300-916.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json
```

#### <span id="running">运行</span>

```eval运行
bash scripts/run_eval.sh 0 coco /home/DataSet/MindRecord_COCO/ /home/model/nasfpn/ckpt/retinanet_nasfpn_300-916.ckpt /home/DataSet/cocodataset/annotations/instances_{}.json
```

> checkpoint 可以在训练过程中产生.

#### <span id="outcome">结果</span>

计算结果将存储在示例路径中，您可以在 `eval.log` 查看.

```mAP

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.607
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754

========================================

mAP: 0.4170593194142041
```

### [模型导出](#content)

#### <span id="usage">用法</span>

导出模型前要修改config.py文件中的checkpoint_path配置项，值为checkpoint的路径。

```shell
python export.py --file_name [RUN_PLATFORM] --file_format[EXPORT_FORMAT] --checkpoint_path [CHECKPOINT PATH]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

#### <span id="running">运行</span>

```运行
python export.py  --file_name retinanet_nasfpn --file_format MINDIR --checkpoint_path /cache/checkpoint/retinanet_nasfpn-300-916.ckpt
```

- 在modelarts上导出MindIR

```Modelarts
在ModelArts上导出MindIR示例
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=retinanet_nasfpn"
#          设置 "file_format=MINDIR"
#          设置 "checkpoint_path=/cache/data/checkpoint/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/NAS_FPN"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../MindRecord_COCO"(选择MindRecord_COCO文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

### [推理过程](#content)

#### <span id="usage">用法</span>

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID]
```

#### <span id="running">运行</span>

```运行
bash run_infer_310.sh ../retinanet_nasfpn.mindir /home/datasets/COCO2017/val2017 /home/datasets/COCO2017/annotations/instances_val2017.json 0
```

#### <span id="outcome">结果</span>

```mAP
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.607
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.457
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.754
mAP: 0.41668179380388803
```

## [模型说明](#content)

### [性能](#content)

#### 训练性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | NAS_FPN                             |
| 运行环境                    | Ascend 910；CPU 2.6GHz，192cores；Memory 766G；系统 Euler2.8  |
| 上传时间                    | 2021/11/25                                     |
| MindSpore 版本             | 1.3.0                                 |
| 数据集                      | 123287 张图片                          |
| Batch_size                 | 16                                   |
| 训练参数                    | src/config.py                         |
| 优化器                      | Momentum                              |
| 损失函数                    | Focal loss                            |
| 最终损失                    | 0.64                                  |
| 精确度 (8p)                 | mAP[0.4170]                          |
| 训练速度 (ms/step)                 | 443.071 ms                          |
| 训练总时间 (8p)             | 34h18m58s                              |
| 脚本                       |  [链接](https://gitee.com/mindspore/models/tree/master/research/cv/nas-fpn)|

#### 推理性能

#### 推理性能

| 参数                 | Ascend                      |
| ------------------- | --------------------------- |
| 模型名称             | NAS_FPN                |
| 运行环境             | Ascend 310；CPU 2.6GHz，192cores；Memory 755G；系统 Euler2.8|
| 上传时间             | 12/17/2021                  |
| MindSpore 版本      | 1.5.0                        |
| 数据集              | 5000 张图片                   |
| Batch_size          | 1                          |
| 精确度              | mAP[0.4166]                  |
| 推理速度 (ms/img)   | 213.508 ms             |

## [随机情况的描述](#content)

在 `dataset.py` 脚本中, 我们在 `create_dataset` 函数中设置了随机种子. 我们在 `train.py` 脚本中也设置了随机种子.

## [ModelZoo 主页](#content)

请核对官方 [主页](https://gitee.com/mindspore/models).
