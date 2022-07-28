# DLinkNet

<!-- TOC -->

- [DLinkNet](#DLinkNet)
    - [D-LinkNet说明](#d-linknet说明)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [推理性能](#推理性能)
        - [用法](#用法)
            - [推理](#推理)
                - [Ascend 310环境运行](#ascend-310环境运行)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## D-LinkNet说明

D-LinkNet模型基于LinkNet架构构建。实现方式见论文[D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html)
在2018年的DeepGlobe道路提取挑战赛中，这一模型表现最好。该网络采用编码器-解码器结构、空洞卷积和预训练编码器进行道路提取任务。

[D-LinkNet 论文](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf): chen Zhou, Chuang Zhang, Ming Wu; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018, pp. 182-186

你可以在[此百度网盘链接](https://pan.baidu.com/s/1KAPPfkSbe5T4wdwLngcIhw?pwd=hkju) 获取已经训练好、符合精度的dlinknet34模型的.ckpt格式的权重文件。

注意：在执行python命令或者bash命令前，请记得在dlinknet_config.yaml文件补全对应部分所需的参数。

## 模型架构

在“DeepGlobe道路提取挑战”中，提供的image和mask的原始大小为1024×1024，并且大多数图像中的道路跨越整个图像。尽管如此，道路仍具有一些自然属性，例如连通性、复杂性等。考虑到这些属性，D-LinkNet旨在接收1024×1024图像作为输入并保留详细的空间信息。D-LinkNet可分为A，B，C三个部分，分别称为编码器，中央部分和解码器。

D-LinkNet使用在ImageNet数据集上预训练的ResNet34作为其编码器。ResNet34最初是为256×256尺寸的中分辨率图像分类而设计的，但在这一挑战中，任务是从1024×1024的高分辨率卫星图像中分割道路。考虑到狭窄性、连通性、复杂性和道路跨度长等方面，重要的是增加网络中心部分的特征的感受范围，并保留详细信息。使用池化层可以成倍增加特征的感受范围，但可能会降低中心特征图的分辨率并降低空间信息。空洞卷积层可能是池化层的理想替代方案。D-LinkNet使用几个空洞卷积层，中间部分带有skip-connection。

本项目中使用的MindSpore框架下的ResNet34模型代码来自于[该网址](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo/official/cv/resnet), 而对应的预训练权重可以在[这里找到](https://download.mindspore.cn/model_zoo/r1.3/) 。

空洞卷积可以级联模式堆叠。如果堆叠的空洞卷积层的膨胀系数分别为1、2、4、8、16，则每层的接受场将为3、7、15、31、63。编码器部分（ResNet34）具有5个下采样层，如果大小为1024×1024的图像通过编码器部分，则输出特征图的大小将为32×32。在这种情况下，D-LinkNet在中心部分使用膨胀系数为1、2、4、8的空洞卷积层，因此最后一个中心层上的特征点将在第一个中心特征图上看到31×31点，覆盖第一中心特征图的主要部分。尽管如此，D-LinkNet还是利用了多分辨率功能，D-LinkNet的中心部分可以看作是并行模式。

D-LinkNet的解码器与原始LinkNet相同，这在计算上是有效的。解码器部分使用转置卷积层进行上采样，将特征图的分辨率从32×32恢复到1024×1024。

## 数据集

使用的数据集： [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset)

- 说明：该数据集由6226个训练图像、1243个验证图像和1101个测试图像组成。每个图像的分辨率为1024×1024。数据集被表述为二分类分割问题，其中道路被标记为前景，而其他对象被标记为背景。
- 数据集大小：3.83 GB

    - 训练集：2.79 GB，6226张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - 验证集：552 MB，1243张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - 测试集：511 MB，1101张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。

- 注意：由于该数据集为比赛用数据集，验证集与测试集的标签图像不会公开，本人在采用了将训练集划出十分之一作为验证集验证模型训练精度的方法。
- 上面给出的数据集链接为上传到Kaggle社区中的，可以直接下载。

- 如果你不想自己划分训练集，你可以只下载 [这个百度网盘链接](https://pan.baidu.com/s/1DofqL6P13PEDGUvNMPo-1Q?pwd=5rp1) ，其中包含了三个文件夹：

    - train：用于训练脚本的文件，5604张图像，包含对应的标签图像，原图像以`xxx_sat.jpg`命名，对应的标签图像则以`xxx_mask.png`命名。
    - valid：用于测试脚本的文件，622张图像，不包含对应的标签图像，原图像以`xxx_sat.jpg`命名。
    - valid_mask：用于评估脚本的文件，622张图像，是valid中图像对应的标签图像，以`xxx_mask.png`命名。

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 选择模型或调整参数

    如果使用其他的参数，也可以参考`dlinknet/`下的yaml文件，通过设置`'model_name'` 为 `'dinknet34'` 或者 `'dinknet50'` 来选择使用什么网络结构。

    注意，在线下或云端使用不同的网络结构时，需要对应在yaml文件中修改对应网络的预训练权重模型的地址路径。

- Ascend处理器环境运行

    注意，在线下机器运行前，请确认`dlinknet_config.yaml`文件中的`enable_modelarts`参数被设置为`False`。

    另外，在运行训练和评估推理脚本前，请确认在[这里](https://download.mindspore.cn/model_zoo/r1.3/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61/) 下载了resnet34的预训练权重文件，并将`dlinknet_config.yaml`文件中的`pretrained_ckpt`参数设置为其绝对路径。

  ```shell
  # 训练示例
  python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]

  # 分布式训练示例
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]

  # 评估示例
  python eval.py --data_path=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config_path=$CONFIG_PATH > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH]

  # 模型导出
  python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[model_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
  ```

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和评估，具体操作如下：

```text
# 在modelarts上使用分布式训练的示例：
# (1) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2)设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (3) 在modelarts的界面上设置代码的路径 "/path/dlinknet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型评估的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "trained_ckpt='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "trained_ckpt='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/dlinknet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的评估。
```

## 脚本说明

### 脚本及样例代码

```text
├── model_zoo
    ├── README.md                           // 模型描述
    ├── dlinknet
        ├── README.md                       // DLinknet描述
        ├── README_CN.md                    // DLinknet中文描述
        ├── ascend310_infer                 // Ascend 310 推理代码
        ├── scripts
        │   ├──run_disribute_train.sh       // Ascend 上分布式训练脚本
        │   ├──run_standalone_train.sh      // Ascend 上单卡训练脚本
        │   ├──run_standalone_eval.sh       // Ascend 上推理评估脚本
        │   ├──run_infer_310.sh             // Ascend 310 推理脚本
        ├── src
        │   ├──__init__.py
        │   ├──callback.py                  // 自定义Callback
        │   ├──data.py                      // 数据处理
        │   ├──loss.py                      // 损失函数
        │   ├──resnet.py                    // resnet网络结构（引用自站内modelzoo）
        │   ├──dinknet.py                   // dlinknet网络结构
        │   ├──model_utils
                ├──__init__.py
                ├──config.py                // 参数配置
                ├──device_adapter.py        // 设备配置
                ├──local_adapter.py         // 本地设备配置
                └──moxing_adapter.py        // modelarts设备配置
        ├── dlinknet_config.yaml            // 配置文件
        ├── train.py                        // 训练脚本
        ├── eval.py                         // 推理脚本
        ├── export.py                       // 导出脚本
        ├── postprocess.py                  // 310 推理后处理脚本
        └── requirements.txt                // 需要的三方库.
```

### 脚本参数

在*.yaml中可以同时配置训练参数和评估参数。

- D-LinkNet配置，DeepGlobe Road Extraction 数据集

  ```yaml
  enable_modelarts: True              # 是否在云端训练
  data_url: ""                        # 在云端训练或评估的数据路径，无需填写
  train_url: ""                       # 在云端训练或评估的输出路径，无需填写
  data_path: "/cache/data"            # 在本地训练的数据路径
  output_path: "/cache/train"         # 在本地训练的输出路径
  device_target: "Ascend"             # 目标设备的类型
  epoch_num: 300                      # 运行1p时的总训练轮次
  run_distribute: "False"             # 是否分布式训练
  distribute_epoch_num: 1200          # 运行8p时的总训练轮次
  pretrained_ckpt: '~/resnet34.ckpt'  # 预训练模型路径
  log_name: "weight01_dink34"         # 模型权重的保存名称
  batch_size: 4                       # 训练批次大小
  learning_rate: 0.0002               # 学习率
  model_name: "dlinknet34"             # 选择的模型名称
  scale_factor: 2                     # loss scale 因子
  scale_window: 1000                  # loss scale 窗口
  init_loss_scale: 16777216           # loss scale 初始值
  trained_ckpt: '~/dinknet34.ckpt'    # 用于评估推理、导出的模型权重路径
  label_path: './'                    # 用于验证的标准标签路径
  predict_path: './'                  # 用于验证的预测标签路径
  num_channels: 3                     # 用于导出的图片通道数
  width: 1024                         # 用于导出的图片宽度
  height: 1024                        # 用于导出的图片长度
  file_name: "dinknet34"              # 用于导出的文件名
  file_format: "MINDIR"               # 用于导出的文件格式
  ```

## 训练过程

- 注意，在线下机器运行前，请确认`dlinknet_config.yaml`文件中的`enable_modelarts`参数被设置为`False`。

- 另外，在运行训练和评估推理脚本前，请确认在[这里](https://download.mindspore.cn/model_zoo/r1.3/resnet34_ascend_v130_imagenet2012_official_cv_bs256_top1acc73.83__top5acc91.61/) 下载了resnet34的预训练权重文件，并将`dlinknet_config.yaml`文件中的`pretrained_ckpt`参数设置为其绝对路径。

### 单机训练

- Ascend处理器环境运行

  ```shell
  python train.py --data_path=/path/to/data/ --config_path=/path/to/yaml > train.log 2>&1 &
  OR
  bash scripts/run_standalone_train.sh [DATASET] [CONFIG_PATH]
  ```

  `[DATASET]`参数对应的路径是数据集解压后的train文件，请记得从中划出十分之一用于接下来验证iou的过程。
  如果你下载了划分后的数据集，将`[DATASET]`设置为其中`train`文件的绝对路径即可。

  上述python命令在后台运行，可通过`train.log`文件查看结果。
  模型检查点和日志储存在`'./output'`这一路径中。

### 分布式训练

- Ascend处理器环境运行

```shell
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET] [CONFIG_PATH]
```

  `[DATASET]`参数对应的路径是数据集解压后的train文件，请记得**从中划出十分之一用于接下来验证iou的过程**。
  如果你下载了划分后的数据集，将`[DATASET]`设置为其中`train`文件的绝对路径即可。

  上述shell脚本在后台运行分布式训练。可通过`LOG[X]/log.log`文件查看结果。
  模型检查点和日志储存在`'LOG[X]/output'`这一路径中。

## 评估过程

### 评估

- Ascend处理器环境运行评估

  ```shell
  python eval.py --data_path=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config_path=$CONFIG_PATH > eval.log 2>&1 &
  OR
  bash scripts/run_standalone_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH] [DEVICE_ID](option, default is 0)
  ```

  `[DATASET]`参数对应的路径是我们之前划出的十分之一的train文件中的图像部分所在的路径。
  如果你下载了划分后的数据集，将`[DATASET]`设置为其中`valid`文件的绝对路径即可。

  `[LABEL_PATH]`参数对应的路径是我们之前划出的十分之一的train文件中的标签部分所在的路径。
  如果你下载了划分后的数据集，将`[LABEL_PATH]`设置为其中`valid_mask`文件的绝对路径即可。

  `[CHECKPOINT]`参数对应的路径是训练好的模型检查点路径。

  `[PREDICT_PATH]`参数对应的路径是通过模型预测验证集的标签的输出路径，如果已该路径存在则会将其删除重新创建。

  上述python命令在后台运行。可通过"eval.log"文件查看结果。

# 模型描述

## 性能

### 训练性能

| 参数                 | Ascend     |
| -------------------------- | ------------------------------------------------------------ |
| 模型版本 | D-LinkNet(DinkNet34) |
| 资源 | Ascend 910；CPU：2.60GHz，192核；内存：755 GB；系统 Euler2.8  |
| 上传日期 | 2022-1-22 |
| MindSpore版本 | 1.5.0 |
| 数据集             | DeepGlobe Road Extraction Dataset|
| 训练参数   | 1pc: epoch=300, total steps=1401, batch_size = 4, lr=0.0002  |
| 优化器 | ADAM |
| 损失函数              | Dice Bce Loss|
| 输出 | 概率 |
| 损失 | 0.249542944|
| 速度 | 1卡：407 ms/step；8卡：430 ms/step |
| 训练总时长 | 1卡：25.30h；8卡：6.27h |
| 精度 | IOU 98% |
| 参数(M)  | 31M|
| 微调检查点 | 118.70M (.ckpt文件)|
| 配置文件 | dlinknet_config.yaml |
| 脚本| [D-LinkNet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/dlinknet) |

### 推理性能

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本             | D-LinkNet(DinkNet34)                |
| 资源                 | Ascend 310；OS Euler2.8                  |
| 上传日期       | 2022-02-11 |
| MindSpore 版本   | 1.5.0                 |
| 数据集             | DeepGlobe Road Extraction Dataset    |
| batch_size          | 1                         |
| 准确率            | acc: 98.13% <br>  acc_cls: 87.19% <br>  iou: 0.9807  |
| 推理模型 | 118M (.mindir 文件)         |

### 用法

#### 推理

如果您需要使用训练好的模型在Ascend 910、Ascend 310等多个硬件平台上进行推理上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。下面是一个简单的操作步骤示例：

##### Ascend 310环境运行

导出mindir模型

在执行导出前需要修改配置文件中的trained_ckpt和batch_size参数。trained_ckpt为ckpt文件路径，batch_size设置为1。
您只需要在(1)配置文件修改以上两个参数，或者(2)在执行python语句时带有这两个参数，在这两个选项中至少采用一个即可。
导出的文件会直接生成在export.py同级的路径下。

本地导出mindir

```shell
python export.py --config_path=[CONFIG_PATH] --trained_ckpt=[trained_ckpt_path] --file_name=[model_name] --file_format=MINDIR --batch_size=1
```

ModelArts导出mindir

```text
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选择a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "trained_ckpt='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "file_name='./dlinknet'"参数在yaml文件。
#          设置 "file_format='MINDIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "trained_ckpt='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          设置 "file_name='./dlinknet'"参数在modearts的界面上。
#          设置 "file_format='MINDIR'" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"
# (4) 在modelarts的界面上设置代码的路径 "/path/dlinknet"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始导出mindir。
```

在执行推理前，MINDIR文件必须在910上通过export.py文件导出。
310推理目前仅可处理batch_Size为1。

```shell
# Ascend310 推理
bash run_infer_310.sh [DATA_PATH] [LABEL_PATH] [MINDIR_PATH] [DEVICE_ID]
```

`DATA_PATH` 为图片数据文件夹路径。
`LABEL_PATH`为标签图片文件夹路径。
`MINDIR_PATH`为导出的mindir模型路径。
`DEVICE_ID` 可选，默认值为 0。

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```text
acc:   0.9813138557017042
acc cls:   0.8719874771543723
iou:   0.980713602209491
```

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
