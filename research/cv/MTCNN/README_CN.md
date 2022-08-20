# 目录

<!-- TOC -->

- [目录](#目录)
- [MTCNN描述](#mtcnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [WIDER Face](#wider-face)
    - [Dataset of  Deep Convolutional Network Cascade for Facial Point Detection](#dataset-of-deep-convolutional-network-cascade-for-facial-point-detection)
    - [FDDB](#fddb)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [wider_face_train_bbx_gt.txt预处理](#widerfacetrainbbxgttxt预处理)
        - [训练模型](#训练模型)
        - [评估模型](#评估模型)
        - [配置参数](#配置参数)
    - [训练过程](#训练过程)
        - [1.训练PNet](#1-训练pnet)
            - [单卡训练PNet](#单卡训练pnet)
            - [多卡训练PNet](#多卡训练pnet)
        - [2.训练RNet](#2-训练rnet)
            - [单卡训练RNet](#单卡训练rnet)
            - [多卡训练RNet](#多卡训练rnet)
        - [3.训练ONet](#3-训练onet)
            - [单卡训练ONet](#单卡训练onet)
            - [多卡训练ONet](#多卡训练onet)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)

<!-- /TOC -->

# MTCNN描述

MTCNN(Multi-task Cascaded Convolutional Networks)是一种多任务级联卷积神经网络，用以同时处理人脸检测和人脸关键点定位问题。作者认为人脸检测和人脸关键点检测两个任务之间往往存在着潜在的联系，然而大多数方法都未将两个任务有效的结合起来，MTCNN充分利用两任务之间潜在的联系，将人脸检测和人脸关键点检测同时进行，可以实现人脸检测和5个特征点的标定。

[论文](https://kpzhang93.github.io/MTCNN_face_detection_alignment/):  Zhang K , Zhang Z , Li Z , et al. Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks[J]. IEEE Signal Processing Letters, 2016, 23(10):1499-1503.

# 模型架构

MTCNN为了解决人脸识别的两阶段问题，提出三个级联的多任务卷积神经网络（Proposal Network (P-Net)、Refine Network (R-Net)、Output Network (O-Net)，每个多任务卷积神经网络均有三个学习任务，分别是人脸分类、边框回归和关键点定位。每一级的输出作为下一级的输入。

# 数据集

使用的数据集一共有三个：

1. [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
1. [Dataset of Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
1. [FDDB](http://vis-www.cs.umass.edu/fddb/index.html)

详细地，

## WIDER Face

- WIDER Face数据集用于训练模型，下载训练数据WIDER  Face Training Images，解压下载的WIDER_train数据集于项目dataset文件夹下。
- 下载WIDER Face的[标注文件](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)，解压并将wider_face_train_bbx_gt.txt文件保存在dataset文件夹下。
- 数据集大小：包含32,203张图片，393,703个标注人脸。
    - WIDER_train: 1.4G
- 检查WIDER_train文件夹在dataset文件夹下，并检查dataset文件下含有wider_face_train_bbx_gt.txt文件，包含人脸标注信息。

## Dataset of  Deep Convolutional Network Cascade for Facial Point Detection

- 该数据集用于训练模型，下载数据集Training set并解压，将其中的lfw_5590和net_7876文件夹以及trainImageList.txt文件放置在datatset文件夹下。
- 数据集大小：包含5,590张LFW图片和7,876张其他图片。
    - lfw_5590：58M
    - net_7876：100M
- 检查trainImageList.txt文件、lfw_5590文件夹和net_7876文件夹在dataset文件夹下。

## FDDB

- FDDB数据集用来评估模型，下载[originalPics.tar.gz](http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz)压缩包和[FDDB-folds.tgz](http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz)压缩包，originalPics.tar.gz压缩包包含未标注的图片，FDDB-folds.tgz包含标注信息。

- 数据集大小：包含2,845张图片和5,171个人脸标注。

    - originalPics.tar.gz：553M
    - FDDB-folds.tgz：1M

- 在dataset文件夹下新建文件夹FDDB。

- 解压originalPics.tar.gz至FDDB，包含两个文件夹2002和2003：

  ````bash
  ├── 2002
  │ ├── 07
  │ ├── 08
  │ ├── 09
  │ ├── 10
  │ ├── 11
  │ └── 12
  ├── 2003
  │ ├── 01
  │ ├── 02
  │ ├── 03
  │ ├── 04
  │ ├── 05
  │ ├── 06
  │ ├── 07
  │ ├── 08
  │ └── 09
  ````

- 解压FDDB-folds.tgz至FDDB，包含20个txt文件：

  ```bash
  FDDB-folds
  │ ├── FDDB-fold-01-ellipseList.txt
  │ ├── FDDB-fold-01.txt
  │ ├── FDDB-fold-02-ellipseList.txt
  │ ├── FDDB-fold-02.txt
  │ ├── FDDB-fold-03-ellipseList.txt
  │ ├── FDDB-fold-03.txt
  │ ├── FDDB-fold-04-ellipseList.txt
  │ ├── FDDB-fold-04.txt
  │ ├── FDDB-fold-05-ellipseList.txt
  │ ├── FDDB-fold-05.txt
  │ ├── FDDB-fold-06-ellipseList.txt
  │ ├── FDDB-fold-06.txt
  │ ├── FDDB-fold-07-ellipseList.txt
  │ ├── FDDB-fold-07.txt
  │ ├── FDDB-fold-08-ellipseList.txt
  │ ├── FDDB-fold-08.txt
  │ ├── FDDB-fold-09-ellipseList.txt
  │ ├── FDDB-fold-09.txt
  │ ├── FDDB-fold-10-ellipseList.txt
  │ ├── FDDB-fold-10.txt
  ```

- 检查2002，2003，FDDB-folds三个文件夹在FDDB文件夹下，且FDDB文件夹在dataset文件夹下。

---------
综上，一共有两个训练集，分别是WIDER Face和Dataset of  Deep Convolutional Network Cascade for Facial Point Detection；一个测试集FDDB。

训练之前，请修改`config.py`文件中的`DATASET_DIR`字段为dataset文件夹路径。

总的数据集目录结构如下：

```bash
dataset
├── FDDB
    ├── 2002
    ├── 2003
    └── FDDB-folds
├── lfw_5590
├── net_7876
├── trainImageList.txt
├── wider_face_train_bbx_gt.txt
└── WIDER_train
    └── images
```

# 环境要求

- 硬件(Ascend/GPU/CPU)
    - 使用GPU搭建硬件环境
- 框架
    [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

数据集准备完成后，请修改`config.py`文件中的`DATASET_DIR`字段为dataset文件夹路径。

在开始训练前，需要对wider_face_train_bbx_gt.txt文件进行预处理以生成wider_face_train.txt文件。

````bash
# 预处理wider_face_train_bbx_gt.txt文件

# 切换到项目主目录
python preprocess.py
# 成功执行后dataset文件夹下出现wider_face_train.txt
````

因为MTCNN由PNet, RNet, ONet三个子模型组成，因此训练过程总体分为三大步骤：

1. 训练PNet

```bash
# 1-1. 生成用于训练PNet模型的mindrecord文件，默认保存在mindrecords文件夹中
bash scripts/generate_train_mindrecord_pnet.sh

```

``` bash
# 1-2. 待mindrecord文件生成完毕，开始训练PNet模型

# 单卡训练
bash scripts/run_standalone_train_gpu.sh pnet DEVICE_ID MINDRECORD_FILE
# example: bash scripts/run_standalone_train_gpu.sh pent 0 mindrecords/PNet_train.mindrecord

# 多卡训练
bash scripts/run_distribute_train.sh pnet DEVICE_NUM MINDRECORD_FILE
# example: bash scripts/run_distribute_train.sh pnet 8 mindrecords/PNET_train.mindrecord
```

2. 训练RNet

```bash
# 2-1. 生成用于训练RNet模型的mindrecord文件，默认保存在mindrecords文件夹中
bash scripts/generate_train_mindrecord_rnet.sh PNET_CKPT
# example: scripts/generate_train_mindrecord_rnet.sh checkpoints/pnet.ckpt
```

```bash
# 2-2. 待mindrecord文件生成完毕，开始训练RNet模型

# 单卡训练
bash scripts/run_standalone_train_gpu.sh rnet DEVICE_ID MINDRECORD_FILE
# example: bash scripts/run_standalone_train_gpu.sh rnet 0 mindrecords/RNET_train.mindrecord

# 多卡训练
bash scripts/run_distribute_train_gpu.sh rnet DEVICE_NUM MINDRECORD_FILE
# example: bash scripts/rum_distribute_train_gpu.sh rnet 8 mindrecords/RNET_train.mindrecord
```

3. 训练ONet

```bash
# 3-1. 生成用于训练ONet模型的mindrecord文件，默认保存在mindrecords文件夹中
bash scripts/generate_train_mindrecord_onet.sh PNET_CKPT RNET_CKPT
# example: bash scripts/generate_train_mindrecord_onet.sh checkpoints/pnet.ckpt checkpoints/rnet.ckpt
```

```bash
# 3-2. 待mindrecord文件生成完毕，开始训练ONet模型

# 单卡训练
bash scripts/run_standalone_train_gpu.sh onet DEVICE_ID MINDRECORD_FILE
# example: bash scripts/run_standalone_train_gpu.sh onet 0 mindrecords/ONET_train.mindrecord

# 多卡训练
bash scripts/run_distribute_train_gpu.sh onet DEVICE_NUM MINDRECORD_FILE
# example: bash scripts/rum_distribute_train_gpu.sh onet 8 mindrecords/ONET_train.mindrecord
```

训练完毕后，开始评估MTCNN模型。

``` bash
# 评估模型
bash scripts/run_eval_gpu.sh PNET_CKPT RNET_CKPT ONET_CKPT
# example: bash scripts/run_eval_gpu.sh checkpoints/pnet.ckpt checkpoints/rnet.ckpt checkpoints/onet.ckpt
```

# 脚本说明

## 脚本及样例代码

```bash
MTCNN
├── dataset                                     // 保存原始数据集和标注文件（需要自行创建该文件夹）
├── eval.py                                     // 评估脚本
├── preprocess.py                               // wider_face_train_bbx_gt.txt文件预处理脚本
├── README_CN.md                                // MTCNN中文描述文档
├── config.py                                   // 配置文件
├── scripts
│   ├── generate_train_mindrecord_pnet.sh       // 生成用于训练PNet的mindrecord文件shell脚本
│   ├── generate_train_mindrecord_rnet.sh       // 生成用于训练RNet的mindrecord文件shell脚本
│   ├── generate_train_mindrecord_onet.sh       // 生成用于训练ONet的mindrecord文件shell脚本
│   ├── run_distributed_train_gpu.sh            // GPU多卡训练shell脚本
│   ├── run_eval_gpu.sh                         // GPU模型评估shell脚本
│   └── run_standalone_train_gpu.sh             // GPU单卡训练shell脚本
├── src
│   ├── acc_callback.py                         // 自定义训练回调函数脚本
│   ├── dataset.py                              // 创建Mindspore数据集脚本
│   ├── evaluate.py                             // 模型评估脚本
│   ├── loss.py                                 // 损失函数
│   ├── utils.py                                // 工具函数
│   ├── models
│   │   ├── mtcnn.py                            // MTCNN模型
│   │   ├── mtcnn_detector.py                   // MTCNN检测器
│   │   └── predict_nets.py                     // 模型推理函数
│   ├── prepare_data
│   │   ├── generate_PNet_data.py               // 生成PNet的mindrecord文件
│   │   ├── generate_RNet_data.py               // 生成RNet的mindrecord文件
│   │   └── generate_ONet_data.py               // 生成ONet的mindrecord文件
│   └── train_models
│       ├── train_p_net.py                      // 训练PNet脚本
│       ├── train_r_net.py                      // 训练RNet脚本
│       └── train_o_net.py                      // 训练ONet脚本
└── train.py                                    // 训练模型脚本

```

## 脚本参数

### wider_face_train_bbx_gt.txt预处理

```bash
usage: preprocess.py [-f F]

Preprocess WIDER Face Annotation file

optional arguments:
  -f F        Original wider face train annotation file
```

### 训练模型

```bash
usage: train.py --model {pnet,rnet,onet} --mindrecord_file
                MINDRECORD_FILE [--ckpt_path CKPT_PATH]
                [--save_ckpt_steps SAVE_CKPT_STEPS] [--max_ckpt MAX_CKPT]
                [--end_epoch END_EPOCH] [--lr LR] [--batch_size BATCH_SIZE]
                [--device_target {GPU,Ascend}] [--distribute]
                [--num_workers NUM_WORKERS]

Train PNet/RNet/ONet

optional arguments:
  --model {pnet,rnet,onet}
                        Choose model to train
  --mindrecord_file MINDRECORD_FILE
                        mindrecord file for training
  --ckpt_path CKPT_PATH
                        save checkpoint directory
  --save_ckpt_steps SAVE_CKPT_STEPS
                        steps to save checkpoint
  --max_ckpt MAX_CKPT   maximum number of ckpt
  --end_epoch END_EPOCH
                        end epoch of training
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        train batch size
  --device_target {GPU,Ascend}
                        device for training
  --distribute
  --num_workers NUM_WORKERS
```

### 评估模型

```bash
usage: eval.py --pnet_ckpt PNET_CKPT --rnet_ckpt RNET_CKPT --onet_ckpt ONET_CKPT

Evaluate MTCNN on FDDB dataset

optional arguments:
  --pnet_ckpt PNET_CKPT, -p PNET_CKPT checkpoint of PNet
  --rnet_ckpt RNET_CKPT, -r RNET_CKPT checkpoint of RNet
  --onet_ckpt ONET_CKPT, -o ONET_CKPT checkpoint of ONet
```

### 配置参数

```bash
config.py:
    DATASET_DIR: 原始数据集文件夹
    FDDB_DIR: 验证数据集FDDB文件夹
    TRAIN_DATA_DIR: 训练数据集文件夹，保存用于生成mindrecord的临时数据文件
    MINDRECORD_DIR: mindrecord文件夹
    CKPT_DIR: checkpoint文件夹
    LOG_DIR: logs文件夹

    RADIO_CLS_LOSS：classification loss比例
    RADIO_BOX_LOSS：box loss比例
    RADIO_LANDMARK_LOSS: landmark loss比例

    TRAIN_BATCH_SIZE: 训练batch size大小
    TRAIN_LR: 默认学习率
    END_EPOCH: 训练轮数
    MIN_FACE_SIZE: 脸最小尺寸
    SCALE_FACTOR: 缩放比例
    P_THRESH: PNet阈值
    R_THRESH: RNet阈值
    O_THRESH: ONet阈值
```

## 训练过程

在开始训练之前，需要先在主目录下创建dataset文件夹，按照数据集部分的步骤下载并保存原始数据集文件在dataset文件夹下。

dataset文件夹准备完毕后，即可开始数据预处理、训练集生成以及模型训练。

因为MTCNN由PNet, RNet和ONet三个子模型串联而成，因此整个训练过程分为三大步骤：

### 1. 训练PNet

```bash
# 预处理wider_face_train_bbx_gt.txt文件
python preprocess.py

# 生成用于训练PNet的mindrecord文件
bash scripts/generate_train_mindrecord_pnet.sh
```

运行后，将产生`generate_pnet_mindrecord.log`日志文件，保存于`logs`文件夹下。

运行完成后，生成`PNet_train.mindrecord`文件，默认保存在`mindrecords`文件夹下。

#### 单卡训练PNet

```bash
bash scripts/run_standalone_train_gpu.sh pnet [DEVICE_ID] [MINDRECORD_FILE]
# example: bash scripts/run_standalone_train_gpu.sh pnet 0 mindrecords/PNet_train.mindrecord
```

训练过程会在后台运行，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/training_gpu_pnet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 2 step: 456, loss is 0.3661264
epoch: 2 step: 457, loss is 0.32284224
epoch: 2 step: 458, loss is 0.29254544
epoch: 2 step: 459, loss is 0.32631972
epoch: 2 step: 460, loss is 0.3065704
epoch: 2 step: 461, loss is 0.3995605
epoch: 2 step: 462, loss is 0.2614449
epoch: 2 step: 463, loss is 0.50305885
epoch: 2 step: 464, loss is 0.30908597
···
```

#### 多卡训练PNet

```bash
bash scripts/run_distributed_train_gpu.sh pnet [DEVICE_NUM] [MINDRECORD_FILE]
# example: bash scripts/run_distributed_train_gpu.sh pnet 8 mindrecord/PNet_train.mindrecord
```

训练过程会在后台运行，只保存第一张卡的训练模型，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/distribute_training_gpu_pnet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 2 step: 456, loss is 0.3661264
epoch: 2 step: 457, loss is 0.32284224
epoch: 2 step: 458, loss is 0.29254544
epoch: 2 step: 459, loss is 0.32631972
epoch: 2 step: 460, loss is 0.3065704
epoch: 2 step: 461, loss is 0.3995605
epoch: 2 step: 462, loss is 0.2614449
epoch: 2 step: 463, loss is 0.50305885
epoch: 2 step: 464, loss is 0.30908597
...
```

### 2. 训练RNet

``` bash
# 生成用于训练RNet的mindrecord文件
bash scripts/generate_train_mindrecord_rnet.sh [PNET_CKPT]
# example: bash scripts/generate_train_mindrecord_rnet.sh checkpoints/pnet.ckpt
```

将产生`generate_rnet_mindrecord.log`日志文件，保存于`logs`文件夹下。

运行完成后，生成`RNet_train.mindrecord`文件，默认保存在`mindrecords`文件夹下。

#### 单卡训练RNet

```bash
bash scripts/run_standalone_train_gpu.sh rnet [DEVICE_ID] [MINDRECORD_FILE]
# example: bash scripts/run_standalone_train_gpu.sh rnet 0 mindrecords/RNet_train.mindrecord
```

训练过程会在后台运行，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/training_gpu_rnet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 1189, loss is 0.4912308
epoch: 1 step: 1190, loss is 0.52638006
epoch: 1 step: 1191, loss is 0.44296187
epoch: 1 step: 1192, loss is 0.522378
epoch: 1 step: 1193, loss is 0.5238542
epoch: 1 step: 1194, loss is 0.49850246
epoch: 1 step: 1195, loss is 0.47963354
epoch: 1 step: 1196, loss is 0.49311465
epoch: 1 step: 1197, loss is 0.45008135
···
```

#### 多卡训练RNet

```bash
bash scripts/run_distributed_train_gpu.sh rnet [DEVICE_NUM] [MINDRECORD_FILE]
# example: bash scripts/run_distributed_train_gpu.sh rnet 8 mindrecord/RNet_train.mindrecord
```

训练过程会在后台运行，只保存第一张卡的训练模型，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/distribute_training_gpu_rnet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 1189, loss is 0.4912308
epoch: 1 step: 1190, loss is 0.52638006
epoch: 1 step: 1191, loss is 0.44296187
epoch: 1 step: 1192, loss is 0.522378
epoch: 1 step: 1193, loss is 0.5238542
epoch: 1 step: 1194, loss is 0.49850246
epoch: 1 step: 1195, loss is 0.47963354
epoch: 1 step: 1196, loss is 0.49311465
epoch: 1 step: 1197, loss is 0.45008135
...
```

### 3. 训练ONet

``` bash
# 生成用于训练ONet的mindrecord文件
bash scripts/generate_train_mindrecord_onet.sh [PNET_CKPT] [RNET_CKPT]
# example: bash scripts/generate_train_mindrecord_rnet.sh checkpoints/pnet.ckpt checkpoints/rnet.ckpt
```

将产生`generate_onet_mindrecord.log`日志文件，保存于`logs`文件夹下。

运行完成后，生成`ONet_train.mindrecord`文件，默认保存在`mindrecords`文件夹下。

#### 单卡训练ONet

```bash
bash scripts/run_standalone_train_gpu.sh onet [DEVICE_ID] [MINDRECORD_FILE]
# example: bash scripts/run_standalone_train_gpu.sh onet 0 mindrecords/ONet_train.mindrecord
```

训练过程会在后台运行，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/training_gpu_onet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 561, loss is 0.1627587
epoch: 1 step: 562, loss is 0.20395292
epoch: 1 step: 563, loss is 0.24887425
epoch: 1 step: 564, loss is 0.31067476
epoch: 1 step: 565, loss is 0.20113933
epoch: 1 step: 566, loss is 0.2834522
epoch: 1 step: 567, loss is 0.18775874
epoch: 1 step: 568, loss is 0.2714229
epoch: 1 step: 569, loss is 0.22088407
epoch: 1 step: 570, loss is 0.22690454
···
```

#### 多卡训练ONet

```bash
bash scripts/run_distributed_train_gpu.sh onet [DEVICE_NUM] [MINDRECORD_FILE]
# example: bash scripts/run_distributed_train_gpu.sh onet 8 mindrecord/ONet_train.mindrecord
```

训练过程会在后台运行，只保存第一张卡的训练模型，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/distribute_training_gpu_onet.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 step: 561, loss is 0.1627587
epoch: 1 step: 562, loss is 0.20395292
epoch: 1 step: 563, loss is 0.24887425
epoch: 1 step: 564, loss is 0.31067476
epoch: 1 step: 565, loss is 0.20113933
epoch: 1 step: 566, loss is 0.2834522
epoch: 1 step: 567, loss is 0.18775874
epoch: 1 step: 568, loss is 0.2714229
epoch: 1 step: 569, loss is 0.22088407
epoch: 1 step: 570, loss is 0.22690454
...
```

## 评估过程

```bash
bash scripts/run_eval_gpu.sh [PNET_CKPT] [RNET_CKPT] [ONET_CKPT]
# example: bash scripts/run_eval_gpu.sh checkpoints/pnet.ckpt checkpoints/rnet.ckpt checkpoints/onet.ckpt
```

评估过程会在后台进行，评估结果可以通过`logs/eval_gpu.log`文件查看，输出结果如下所示：

```bash
==================== Results ====================
FDDB-fold-1 Val AP: 0.846041313059397
FDDB-fold-2 Val AP: 0.8452332863014286
FDDB-fold-3 Val AP: 0.854312327697665
FDDB-fold-4 Val AP: 0.8449615417375469
FDDB-fold-5 Val AP: 0.868903617729559
FDDB-fold-6 Val AP: 0.8857753502792894
FDDB-fold-7 Val AP: 0.8200462708769559
FDDB-fold-8 Val AP: 0.8390865359172448
FDDB-fold-9 Val AP: 0.8584513847530266
FDDB-fold-10 Val AP: 0.8363366158400566
FDDB Dataset Average AP: 0.8499148244192171
=================================================
```

# 模型描述

## 性能

| 参数                 | MTCNN                                               |
| -------------------- | ------------------------------------------------------- |
| 资源                 | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G|
| 上传日期             | 2022-08-05                                             |
| MindSpore版本        | 1.8.0                                                  |
| 数据集               | WIDER Face, Dataset of  Deep Convolutional Network Cascade for Facial Point Detection, FDDB |
| 训练参数             | PNet: epoch=30,batch_size=384, lr=0.001; RNet: epoch=22, batch_size=384, lr=0.001; ONet: epoch=22, batch_size=384, lr=0.001 |
| 优化器               | Adam                                                |
| 损失函数             | SoftmaxCrossEntropyWithLogits, MSELoss                       |
| 输出                 | 类别，坐标                                               |
| 损失                 | PNet: 0.20 RNet: 0.15 ONet: 0.04                |
| 速度                 | PNet: 6毫秒/步 RNet: 8毫秒/步 ONet: 18毫秒/步                                            |
| 总时长               | 8时40分(单卡)；1时22分(八卡)                                                |
| 微调检查点 | PNet: 1M (.ckpt文件) RNet: 2M (.ckpt文件) ONet: 6M (.ckpt文件) |

