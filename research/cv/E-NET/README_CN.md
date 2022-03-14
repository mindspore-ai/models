# E-NET

## 目录

<!-- TOC -->

- [E-NET](#E-NET)
    - [目录](#目录)
    - [E-NET描述](#E-NET描述)
    - [概述](#概述)
        - [论文](#论文)
        - [精度](#精度)
    - [环境](#环境)
    - [数据集](#数据集)
    - [脚本说明](#脚本说明)
    - [训练与验证](#训练与验证)
        - [单卡训练](#单卡训练)
        - [多卡训练](#多卡训练)
        - [验证单个ckpt](#验证单个ckpt)
    - [模型描述](#模型描述)
    - [310推理](#310推理)

<!-- /TOC -->

## E-NET描述

## 概述

E-NET主要用于图像分割领域，是一种端到端的分割方法。语义分割的落地（应用于嵌入式设备如手机、可穿戴设备等低功耗移动设备）是一个很重要的问题。基于VGG架构的语义分割模型需要大量的浮点运算，导致运行时间长，从而降低了时效性。 相比之下，ENet网络推理速度快，浮点计算量少，参数少，且有相似的精度。

使用mindpsore复现E-NET[[论文]](https://arxiv.org/abs/1606.02147)。
这个项目迁移于ENet的Pytorch实现[[HERE]](https://github.com/davidtvs/PyTorch-ENet)。

### 论文

[论文地址](https://arxiv.org/abs/1606.02147)：A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello."ENet: A deep neural network architecture for real-time semantic segmentation."

### 精度

| (Val IOU)      | enet_pytorch | enet_mindspore |
| -------------- | ------------ | -------------- |
| **512 x 1024** | **59.5**     | **62.1**       |

其中各个类的IOU的具体计算方法来自[erfnet的pytorch实现](https://github.com/Eromera/erfnet_pytorch)

## 环境

Ascend，GPU

## 数据集

[**The Cityscapes dataset**](https://www.cityscapes-dataset.com/):

在官网直接下载的标签文件, 像素被分为30多类, 在训练时我们需要将其归纳到20类, 所以对其需要进行处理. 为了方便可以直接下载已经处理好的数据.

链接：[[HERE]](https://pan.baidu.com/s/1jH9GUDX4grcEoDNLsWPKGw). 提取码：aChQ.

下载后可以得到以下目录:

```sh
└── cityscapes
    ├── gtFine .................................. ground truth
    └── leftImg8bit ............................. 训练集&测试集&验证集
```

键入

```bash
python src/build_mrdata.py \
--dataset_path /path/to/cityscapes/ \
--subset train \
--output_name train.mindrecord
```

脚本会在/path/to/cityscapes/数据集根目录下，找到训练集，在output_name指出的路径下生成mindrecord文件，然后在项目根目录下新建data文件夹，
再将生成的mindrecord文件移动到项目根目录下的data文件夹下，来让脚本中的相对路径能够定位

## 脚本说明

```bash
|
├── ascend310_infer
│   ├── inc
│   │   └── utils.h                               // utils头文件
│   └── src
│       ├── CMakeLists.txt                        // cmakelist
│       ├── main.cc                               // 推理代码
│       ├── build.sh                              // 运行脚本
│       └── utils.cc                              // utils实现
├── scripts
│   ├── run_distribute_train.sh                   // 多卡训练脚本
│   ├── run_standalone_train.sh                   // 单卡训练脚本
│   ├── run_standalone_train_gpu.sh               // 单卡训练脚本（GPU)
│   └── run_distribute_train_gpu.sh               // 单卡训练脚本（GPU)
├── src
│   ├── build_mrdata.py                           // 生成mindrecord数据集
│   ├── config.py                                 // 配置参数脚本
│   ├── dataset.py                                // 数据集脚本
│   ├── iou_eval.py                               // metric计算脚本
│   ├── criterion.py                              // 损失函数脚本
│   ├── model.py                                  // 模型脚本
│   └── util.py                                   // 工具函数脚本
├── README_CN.md                                  // 描述文件
├── eval.py                                       // 测试脚本
├── export.py                                     // MINDIR模型导出脚本
└── train.py                                      // 训练脚本
```

## 训练与验证

训练之前需要生成mindrecord数据文件并放到项目根目录的data文件夹下，然后启动脚本。

### 单卡训练

如果你要使用单卡进行训练，进入项目根目录，键入

#### Ascend单卡

```bash
nohup bash scripts/run_standalone_train.sh /home/name/cityscapes 0 &
```

其中/home/name/cityscapes指数据集的位置，其后的0指定device_id.

#### GPU单卡

```bash
nohup `bash scripts/run_standalone_train_gpu.sh  0 /home/name/cityscapes` &
```

其中0指定device_id，其后的/home/name/cityscapes指数据集的位置

运行该脚本会完成对模型的训练和评估两个阶段。

其中训练阶段分三步，前两步用于训练Enet模型的编码器部分，第三步会训练完整的Enet网络。
训练过程中在项目根目录下会生成log_single_device文件夹，其中log_stage*.txt即为程序log文件，键入

```bash
tail -f log_single_device/log_stage*.txt
```

显示训练状态。

评估阶段会在验证集上计算log_single_device文件夹下所有权重的精度，并同位置生成后缀metrics.txt文件，显示结果

### 多卡训练

例如，你要使用4卡进行训练，进入项目根目录，键入

#### Ascend多卡

```bash
nohup bash scripts/run_distribute_train.sh /home/name/cityscapes 4 0,1,2,3 /home/name/rank_table_4pcs.json &
```

其中/home/name/cityscapes指数据集的位置，其后的4指rank_size, 再后的0,1,2,3制定了设备的编号, /home/name/rank_table_4pcs.json指并行训练配置文件的位置。其他数目的设备并行训练也类似。

在项目根目录下会生成log_multi_device文件夹，./log_multi_device/log0/log*.txt即为多卡日志文件，键入

```bash
tail -f log_multi_device/log0/log*.txt
```

显示训练状态。

#### GPU多卡

```bash
nohup `bash scripts/run_distribute_train_gpu.sh  4 0,1,2,3 /home/name/cityscapes` &
```

其中4指rank_size, 再后的0,1,2,3制定了设备的编号, /home/name/cityscapes指数据集的位置， 在项目根目录下会生成log_distribute_device文件夹，./log_distribute_device/log_output*/1/rank.*/stdout即为多卡日志文件，
键入

```bash
tail -f log_distribute_device/log_output*/1/rank.*/stdout
```

显示训练状态。

### 验证单个ckpt

键入

```bash
bash scripts/run_eval_gpu.sh 0 /home/name/cityscapes /checkpoint/E-NET.ckpt

```

其中0制定了设备的编号, /home/name/cityscapes指数据集的位置，/checkpoint/E-NET.ckpt指ckpt文件的位置键入
验证完毕后，会在ckpt文件同目录下后缀metrics.txt文件记录结果。

```txt
model path ./ENet-100_496.ckpt
mean_iou 0.6219186616013426
mean_loss 0.3161865407142856
iou_class [0.96626199 0.75290523 0.87924483 0.43634233 0.44190292 0.50485979
 0.50586298 0.60316052 0.89555818 0.56628902 0.92109006 0.66907491
 0.4730712  0.89284724 0.45698707 0.62259347 0.32161359 0.29706163
 0.6097276 ]

```

## 模型描述

### 性能

#### 训练性能

##### Cityscapes上训练E-Net

| 参数                 | Ascend                                                |  GPU|
| --------------------| ----------------------------------------------------- |--------|
| 模型版本             | E-Net                                                 |  E-Net   |
| 资源                | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8  | RTX3090；CPU 2.90GHz，64核；内存 252G；系统 Ubuntu20.04|
| 上传日期             | 2021-10-09                                            | 2022-3-23 |
| MindSpore版本       | 1.2.0                                                 | 1.6.1    |
| 数据集              | Cityscapes                                            | Cityscapes      |
| 训练参数            | epoch=250, steps=496, batch_size = 6, lr=5e-4         | epoch=250, steps=495, batch_size = 6, lr=5e-4     |
| 优化器              | Adam                                                  | Adam  |
| 损失函数            | 带权重的Softmax交叉熵                                    | 带权重的Softmax交叉熵  |
| 输出               | 语义分割图                                              |  语义分割图 |
| 损失               | 0.17356214                                            |  0.20114072        |
| 速度               | 单卡：882毫秒/步;                                       |  单卡：571毫秒/步;      |
| 总时长             | 单卡：30h;                                             |   单卡：25h;   |
| 参数(M)           | 0.34                                                  |   0.34      |
| 微调检查点 | 4.40M (.ckpt文件)                                              |   4.60M    |
| 推理模型        | 9.97M(.air文件)                                          |   9.97M(.air文件)     |

#### 评估性能

##### Cityscapes上评估E-Net

| 参数          | Ascend                           | GPU                       |
| ------------------- | --------------------------|---------------------------|
| 模型版本       |       E-Net                      |   E-Net                  |
| 资源            |  Ascend 910；系统 Euler2.8      | RTX3090；系统 Ubuntu20.04 |
| 上传日期       | 2021-10-09                      | 2022-3-23                |
| MindSpore 版本   | 1.2.0                        | 1.6.1                    |
| 数据集             | Cityscapes, 500张图像       | Cityscapes, 500张图像      |
| batch_size          | 6                       |  6                       |
| 输出             | 语义分割图                   |  语义分割图                 |
| 准确性            | 单卡: 62.19%;              | 单卡: 62.22%;            |

## 310推理

需要导出训练好的ckpt文件, 得到能在310上直接推理的mindir模型文件:

```sh
python export.py --model_path /path/to/net.ckpt
```

会在当前目录下得到enet.mindir文件。

```sh
bash scripts/run_infer_310.sh /path/to/enet.mindir /path/to/images /path/to/result  /path/to/label 0
```

其中/path/to/images指验证集的图片, 由于原始数据集的路径cityscapes/leftImg8bit/val/的图片根据拍摄的城市进行了分类, 需要先将其归到一个文件夹下才能供推理。
例如

```sh
cp /path/to/cityscapes/leftImg8bit/val/frankfurt/* /path/to/images/
cp /path/to/cityscapes/leftImg8bit/val/lindau/* /path/to/images/
cp /path/to/cityscapes/leftImg8bit/val/munster/* /path/to/images/
```

验证集的ground truth, 同理也要归到/path/to/labels/下. 其余的参数/path/to/enet.mindir指mindir文件的路径, /path/to/result推理结果的输出路径(也需要提前生成该文件夹), 0指的是device_id

最终推理结果会输出在/res/result/文件夹下, 当前目录下会生成metric.txt, 其中包含精度.
