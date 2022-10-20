# 目录

<!-- TOC -->

<!-- /TOC -->

## CMR描述

CMR是2019年提出的一种基于卷积网格回归的对单图片体型的重构模型，它解决了单图片的3D人物姿势和体型估计的问题。想比传统的方法与方法，CMR无需回归模型参数，而是直接在图片网格点上回归3D坐标点。

[论文](https://arxiv.org/pdf/1905.03244.pdf): Kolotouros, Nikos, Georgios Pavlakos, and Kostas Daniilidis. "Convolutional mesh regression for single-image human shape reconstruction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

## 模型架构

CMR总共有两个子模型组成，从浅到深分别是Graph CNN和SMPLParamRegressor，前者用来回归网格点的3D坐标，后者将无参数预测和含参数模型如SMPL连结起来。

## 数据集

使用的数据集一共有四个：

1. [UP-3D](http://files.is.tuebingen.mpg.de/classner/up/)
2. [LSP](http://sam.johnson.io/research/lsp.html)
3. [MPII](http://human-pose.mpi-inf.mpg.de/)
4. [COCO2014](http://cocodataset.org/#home)

详细地，

### UP-3D

- UP-3D数据集用来训练模型和验证模型。需要下载[UP-3D.zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip)(提供训练和验证需要的图片和3D形状)和[UPi-S1h.zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/upi-s1h.zip)(用来对 LSP 数据集进行轮廓评估)。

- 数据集压缩包大小
    - up-3d.zip: 4.6G
    - upi-s1h.zip: 45G

- 解压后，将`src/config.py`中的`UP_3D_ROOT`改为数据集路径

### LSP

- LSP dataset original数据集用来训练模型。需要下载[LSP-dataset-origin](http://sam.johnson.io/research/lsp_dataset_original.zip)。
- LSP dataset数据集用来验证模型。需要下载[LSP-datset](http://sam.johnson.io/research/lsp_dataset.zip)
- 数据集压缩包大小
    - lsp_dataset_originall.zip: 252M
    - lsp_dataset.zip: 34M
- 解压后，将`src/config.py`中的`LSP_ORIGINAL_ROOT`改为LSP dataset original数据集路径，`LSP_ROOT`改为LSP dataset数据集路径

### MPII

- MPII数据集用来训练模型。需要下载[MPII](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)。

- 解压后，将`src/config.py`中的`MPII_ROOT`改为数据集路径

### COCO2014

- COCO2014数据集用来训练模型。需要下载[images](http://images.cocodataset.org/zips/train2014.zip)和[annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)。

- 解压后，将`src/config.py`中的`COCO_ROOT`改为数据集路径

- COCO2014数据集目录结构如下：

```bash
${COCO root}
|-- train2014
|-- annotations
```

---------
综上，一共有四个训练集，分别是up-3d, lsp_dataset_origin, mpii和coco2014；两个测试集，分别是up-3d和lsp_dataset。

总的数据集目录结构如下：

```bash
${DATASET root}
├── coco2014
│   ├── annotations
│   └── train2014
├── LSP
│   ├── lsp_dataset
│   │   ├── images
│   │   └── visualized
│   └── lsp_dataset_original
│       └── images
├── MPII
│   └── images
└── UP_3D
    ├── up-3d
    └── upi-s1h
        └── data
            ├── lsp
            ├── lsp_extended
            └── mpii
                ├── images
                └── visualizations

```

## 环境要求

- 硬件(Ascend/GPU/CPU)
    - 使用GPU搭建硬件环境
- 框架
    [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 下载数据集并更改`src/config`中相应的数据集路径
- PyTorch的Resnet50预训练模型[点此下载](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- 从[百度云](https://pan.baidu.com/s/1jMBIDCGicfFROLBCE9YP7w?pwd=bwq0)下载data.zip文件并解压到cmr主目录下
- 从[公共服务器](https://download.mindspore.cn/thirdparty/cmr/)下载smpl.ckpt和mesh.ckpt文件于data目录下

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 生成预处理后的数据集

# 仅生成训练集
bash scripts/generate_dataset.sh train

# 仅生成测试集
bash scripts/generate_dataset.sh test

# 生成训练集和测试集
bash scripts/generate_dataset.sh train test
```

```bash
# 将PyTorch的resnet50预训练模型转化为Mindspore的预训练模型
# 同时请将src/config.py中的PRETRAINED_RESNET_50改成PTH_PATH
bash scripts/convert_resnet.sh [PTH_PATH] [CKPT_PATH]
# example: bash scripts/convert_resnet.sh resnet50-19c8e357.pth pretrained_resnet50.ckpt
```

```bash
# 单卡训练
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOAD_CHECKPOINT_PATH](optional)
# example: bash scripts/run_standalone_train_gpu.sh 0 /path/load_ckpt
# if no ckpt: bash scripts/run_standalone_train_gpu.sh 0
```

```bash
# 多卡训练
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [LOAD_CHECKPOINT_PATH](optional)
# example: bash scripts/run_distributed_train_gpu.sh 8 /path/load_ckpt
# if no ckpt: bash scripts/run_distributed_train_gpu.sh 8
```

```bash
# 评估模型
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_NAME] [LOAD_CHECKPOINT_PATH]
# example: bash scripts/run_eval_gpu.sh 0 up-3d /path/ckpt
```

## 脚本说明

### 脚本及样例代码

```bash
CMR
├── data
│   ├── mesh.ckpt                               // meshCell模型的参数，不可训练
│   ├── namesUPlsp.txt                          // UP-3D数据集的评估文件
│   ├── resnet50_dict.json                      // resnet50模型的PyTorch和Mindspore的参数名对应表
│   ├── smpl.ckpt                               // SMPL模型的参数，不可训练
│   ├── train.h5                                // MPII人体姿势数据集的标注文件
├── eval.py                                     // 评估脚本
├── preprocess.py                               // 数据集预处理脚本
├── README_CN.md                                // CMR中文描述文档
├── scripts
│   ├── convert_resnet.sh                       // PyTorch的Resnet50预训练模型转化为Mindspore的预训练模型shell脚本
│   ├── generate_dataset.sh                     // 生成训练数据集和评估数据集shell脚本
│   ├── run_distributed_train_gpu.sh            // GPU多卡训练shell脚本
│   ├── run_eval_gpu.sh                         // GPU模型评估shell脚本
│   └── run_standalone_train_gpu.sh             // GPU单卡训练shell脚本
├── src
│   ├── config.py                               // 模型配置文件
│   ├── dataset
│   │   ├── base_dataset.py                     // 基本数据集处理脚本
│   │   ├── datasets.py                         // 创建Mindspore数据集
│   │   ├── preprocess
│   │   │   ├── coco.py                         // coco数据集预处理脚本
│   │   │   ├── lsp_dataset_original.py         // lsp-dataset-original数据集预处理脚本
│   │   │   ├── lsp_dataset.py                  // lsp-dataset数据集预处理脚本
│   │   │   ├── mpii.py                         // mpii数据集预处理脚本
│   │   │   └── up_3d.py                        // up-3d数据集预处理脚本
│   ├── loss
│   │   ├── loss.py                             // 损失函数
│   ├── models
│   │   ├── cmr.py                              // CMR模型架构
│   │   ├── geometric_layers.py                 // 几何操作
│   │   ├── graph_cnn.py                        // Graph CNN模型架构
│   │   ├── graph_layers.py                     // Graph CNN模型的层次
│   │   ├── layers.py                           // SMPLParamRegressor模型中用于搭建模块的网络层
│   │   ├── resnet.py                           // Resnet模型架构
│   │   ├── smpl_param_regressor.py             // SMPLParamRegressor模型架构
│   │   └── smpl.py                             // SMPL模型架构
│   ├── netCell
│   │   ├── evalNet.py                          // CMR模型的评估网络
│   │   ├── netWithLoss.py                      // 包含损失函数的Cell
│   └── utils
│       ├── imutils.py                          // 图像处理工具
│       ├── mesh.py                             // meshCell定义
│       ├── options.py                          // 模型训练和评估的选项
│       ├── pth2ckpt.py                         // PyTorch模型转换到Mindspore模型
└── train.py                                    // 训练模型脚本

```

### 脚本参数

#### Resnet50预训练模型转化

```bash
usage: pth2ckpt.py [--pth-path PTH_PATH]
                   [--ckpt-path CKPT_PATH]
                   [--dict-file DICT_FILE]

optional arguments:
  --pth-path PTH_PATH   pth文件
  --ckpt-path CKPT_PATH
                        保存的ckpt文件目标路径
  --dict-file DICT_FILE
                        模型参数名映射文件
```

#### 生成预处理数据集

```bash
usage: preprocess.py [--train_files] [--eval_files]

optional arguments:
  --train_files  解析训练数据集
  --eval_files   解析评估数据集
```

#### 训练

```bash
usage: train.py [-h] [--save_checkpoint_dir SAVE_CHECKPOINT_DIR]
                [--keep_checkpoint_max KEEP_CHECKPOINT_MAX]
                [--pretrained_checkpoint PRETRAINED_CHECKPOINT]
                [--device_target {GPU,Ascend}] [--num_channels NUM_CHANNELS]
                [--num_layers NUM_LAYERS] [--img_res IMG_RES]
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--checkpoint_steps CHECKPOINT_STEPS]
                [--rot_factor ROT_FACTOR] [--noise_factor NOISE_FACTOR]
                [--scale_factor SCALE_FACTOR] [--do_shuffle] [--distribute]
                [--num_workers NUM_WORKERS] [--adam_beta1 ADAM_BETA1]
                [--lr LR] [--wd WD]

optional arguments:
    --save_checkpoint_dir: 保存checkpoint的路径
    --keep_checkpoint_max: checkpoint的最大保存数量
    --pretrained_checkpoint: 预训练模型路径
    --device_target: 模型训练平台，如GPU或Ascend
    --num_channels: Graph Residual layers的通道数
    --num_layers: Graph CNN中的残差块数
    --img_res: 输入图片宽度和高度
    --num_epochs: 模型训练轮数
    --batch_size: 一批样本的数量
    --checkpoint_steps: checkpoint保存频率
    --distribute: 多卡训练
    --num_workers: 抓取数据进程数
    --lr: 学习率
    --wd: 权重衰减
```

#### 评估

```bash
usage: eval.py [--num_channels NUM_CHANNELS] [--num_layers NUM_LAYERS]
               [--img_res IMG_RES] --checkpoint CHECKPOINT
               [--dataset {up-3d}] [--batch_size BATCH_SIZE] [--shuffle]
               [--num_workers NUM_WORKERS] [--log_freq LOG_FREQ]
               [--device_target DEVICE_TARGET]

Options to eval the model

optional arguments:
  --num_channels: Graph Residual layers的通道数
  --num_layers: Graph CNN中的残差块数
  --img_res: 输入图片宽度和高度
  --checkpoint: 评估的模型
  --dataset: 评估数据集
  --batch_size: 评估数据批大小
  --num_workers: 抓取数据进程数
  --log_freq: 打印评估结果频率
  --device_target: 模型评估平台，如GPU或Ascend
```

#### 配置参数

```bash
config.py:
    LSP_ROOT: 解压后lsp_dataset文件
    LSP_ORIGINAL_ROOT: 解压后lsp_dataset_original文件
    UPI_S1H_ROOT: 解压后upi-s1h文件
    MPII_ROOT: 解压后mpii文件
    COCO_ROOT: 解压后coco2014文件
    UP_3D_ROOT: 解压后up-3d文件

    PRETRAINED_RESNET_50: Resnet50预训练模型
    DATASET_NPZ_PATH: 保存预训练后数据集文件夹

    SMPL_CKPT_FILE: SMPL模型参数文件
    MESH_CKPT_FILE: MeshCell参数文件

    PARENT: SMPL模型中的parent参数
    JOINTS_IDX: SMPL模型中的joints_idx参数

    IMG_NORM_MEAN: 图像归一化均值
    IMG_NORM_STD: 图像归一化标准差
```

## 训练过程

在第一次训练之前，需要对原始数据集进行预处理，并生成相应的npz文件。请确保正确下载原始数据集，且`src/config.py`中的数据集文件路径正确配置。从[百度云](https://pan.baidu.com/s/1jMBIDCGicfFROLBCE9YP7w?pwd=bwq0)下载data.zip文件并解压到主目录下，用于数据预处理。之后，需要从[公共服务器](https://download.mindspore.cn/thirdparty/cmr/)下载mesh.ckpt和smpl.ckpt两个文件于data文件夹下，用于初始化untrainable模型参数。

```bash
# 仅生成训练集
bash scripts/generate_dataset.sh train

# 仅生成测试集
bash scripts/generate_dataset.sh test

# 生成训练集和测试集
bash scripts/generate_dataset.sh train test
```

根据需要，选择生成数据集，生成过程将在系统后台运行。

### 单卡训练

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOAD_CHECKPOINT_PATH](optional)
```

训练过程会在后台运行，训练模型将保存在`checkpoints/ckpt`文件夹中，可以通过`logs/training_gpu.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 1, loss is 0.7637087106704712
epoch: 1 step: 2, loss is 0.5255427360534668
epoch: 1 step: 3, loss is 0.797945499420166
epoch: 1 step: 4, loss is 0.7719042897224426
epoch: 1 step: 5, loss is 0.47390806674957275
···
```

### 多卡训练

```bash
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [LOAD_CHECKPOINT_PATH](optional)
```

训练过程会在后台运行，只保存第一张卡的训练模型，训练模型将保存在`checkpoints/ckpt_rank0`文件夹中，可以通过`logs/distribute)training_gpu.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 1, loss is 1.5541653633117676
epoch: 1 step: 1, loss is 1.433321475982666
epoch: 1 step: 1, loss is 1.4187467098236084
epoch: 1 step: 1, loss is 1.4918560981750488
epoch: 1 step: 1, loss is 1.3962852954864502
epoch: 1 step: 1, loss is 1.467834711074829
epoch: 1 step: 1, loss is 1.5032799243927002
epoch: 1 step: 1, loss is 1.4833412170410156
epoch: 1 step: 2, loss is 1.3063676357269287
epoch: 1 step: 2, loss is 1.3526051044464111
epoch: 1 step: 2, loss is 1.372483253479004
epoch: 1 step: 2, loss is 1.3633804321289062
epoch: 1 step: 2, loss is 1.450531005859375
epoch: 1 step: 2, loss is 1.2272545099258423
epoch: 1 step: 2, loss is 1.2886061668395996
epoch: 1 step: 2, loss is 1.3142821788787842
...
```

## 评估过程

```bash
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_NAME] [LOAD_CHECKPOINT_PATH]
```

其中DATASET_NAME为up-3d。评估过程会在后台进行，评估结果可以通过`logs/eval_gpu.log`文件查看，输出结果如下所示：

```bash
Start eval
Shape Error (NonParam): 148.25130109157828
Shape Error (Param): 159.54252814925792

*** Final Results ***

Shape Error (NonParam): 149.73837524456414
Shape Error (Param): 161.51834581365907

```

## 模型描述

## 性能

| 参数                 | CMR                                                    |
| -------------------- | ------------------------------------------------------- |
| 资源                 | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G|
| 上传日期             | 2022-07-05                                              |
| MindSpore版本        | 1.7.0                                                   |
| 数据集               | coco2014, LSP, UP-3D, MPII                                                   |
| 训练参数             | epoch=80, steps=1875, batch_size = 16, lr=0.004          |
| 优化器               | Adam                                                |
| 损失函数             | L1Loss, MSELoss                                           |
| 输出                 | 坐标                                                    |
| 损失                 | 0.01                                                   |
| 速度                 | 3000.0毫秒/步                                             |
| 总时长               | 17时                                                |
| 微调检查点 | 774M (.ckpt文件)       |

