
# 目录

<!-- TOC -->

- [目录](#目录)
- [PyramidBox描述](#pyramidbox描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [WIDER Face](#wider-face)
    - [FDDB](#fddb)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练模型](#训练模型)
        - [评估模型](#评估模型)
        - [配置参数](#配置参数)
    - [训练过程](#训练过程)
        - [单卡训练](#单卡训练)
        - [多卡训练](#多卡训练)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)

<!-- /TOC -->

# PyramidBox描述

[PyramidBox](https://arxiv.org/pdf/1803.07737.pdf) 是一种基于SSD的单阶段人脸检测器，它利用上下文信息解决困难人脸的检测问题。如下图所示，PyramidBox在六个尺度的特征图上进行不同层级的预测。该工作主要包括以下模块：LFPN、Pyramid Anchors、CPM、Data-anchor-sampling。

[论文](https://arxiv.org/pdf/1803.07737.pdf):  Tang, Xu, et al. "Pyramidbox: A context-assisted single shot face detector." Proceedings of the European conference on computer vision (ECCV). 2018.

# 模型架构

**LFPN**: LFPN全称Low-level Feature Pyramid Networks, 在检测任务中，LFPN可以充分结合高层次的包含更多上下文的特征和低层次的包含更多纹理的特征。高层级特征被用于检测尺寸较大的人脸，而低层级特征被用于检测尺寸较小的人脸。为了将高层级特征整合到高分辨率的低层级特征上，我们从中间层开始做自上而下的融合，构建Low-level FPN。

**Pyramid Anchors**: 该算法使用半监督解决方案来生成与人脸检测相关的具有语义的近似标签，提出基于anchor的语境辅助方法，它引入有监督的信息来学习较小的、模糊的和部分遮挡的人脸的语境特征。使用者可以根据标注的人脸标签，按照一定的比例扩充，得到头部的标签（上下左右各扩充1/2）和人体的标签（可自定义扩充比例）。

**CPM**: CPM全称Context-sensitive Predict Module, 本方法设计了一种上下文敏感结构(CPM)来提高预测网络的表达能力。

**Data-anchor-sampling**: 设计了一种新的采样方法，称作Data-anchor-sampling，该方法可以增加训练样本在不同尺度上的多样性。该方法改变训练样本的分布，重点关注较小的人脸。

# 数据集

使用的数据集一共有两个：

1. [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
1. [FDDB](http://vis-www.cs.umass.edu/fddb/index.html)

详细地，

## WIDER Face

- WIDER Face数据集用于训练模型和验证模型，下载训练数据WIDER Face Training Images，解压下载的WIDER_train数据集；下载验证数据集WIDER Face Validation Images，解压下载的WIDER_val数据集。
- 下载WIDER Face的[标注文件](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)，解压成文件夹wider_face_split。
- 在dataset文件夹下新建目录WIDERFACE，将WIDER_train，WIDER_val和wider_face_split文件夹放在目录WIDERFACE下。
- 数据集大小：包含32,203张图片，393,703个标注人脸。
    - WIDER_train: 1.4G
    - WIDER_val：355M
- 检查WIDER_train，WIDER_val和wider_face_split文件夹在WIDERFACE目录下。

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
综上，一共有一个训练集文件，是WIDER_train；两个验证集文件，WIDER_val和FDDB。

编辑`src/config.py`文件，将`_C.HOME`字段改成dataset数据集路径。

总的数据集目录结构如下：

```bash
dataset
├── FDDB
│   ├── 2002
│   ├── 2003
│   └── FDDB-folds
└──  WIDERFACE
    ├── wider_face_split
    │   ├── readme.txt
    │   ├── wider_face_test_filelist.txt
    │   ├── wider_face_test.mat
    │   ├── wider_face_train_bbx_gt.txt
    │   ├── wider_face_train.mat
    │   ├── wider_face_val_bbx_gt.txt
    │   └── wider_face_val.mat
    ├── WIDER_train
    │   └── images
    └── WIDER_val
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

在开始训练前，需要进行以下准备工作：

1. 检查`src/config.py`文件的`_C.HOME`字段为dataset文件夹路径。
2. 对wider_face_train_bbx_gt.txt和wider_face_val_bbx_gt.txt文件进行预处理以生成face_train.txt和face_val.txt文件。

````bash
# 预处理wider_face_train_bbx_gt.txt和wider_face_val_bbx_gt.txt文件

# 进入项目主目录
python preprocess.py
# 成功执行后data文件夹下出现face_train.txt和face_val.txt
````

3. 生成face_val.txt的mindrecord文件，用于训练过程中验证每一轮模型精度，找出最佳训练模型。

```bash
bash scripts/generate_mindrecord.sh
# 成功执行后data文件夹下出现val.mindrecord和val.mindrecord.db文件
```

4. 下载预训练完成的[vgg16.ckpt](https://pan.baidu.com/s/1e5qSW4e1QVZRnbyGRWi91Q?pwd=dryt)文件，该预训练模型转自PyTorch。

完成以上步骤后，开始训练模型。

1. 单卡训练

```bash
bash scripts/run_standalone_train_gpu.sh DEVICE_ID VGG16_CKPT VAL_MINDRECORD_FILE
# example: bash scripts/run_standalone_train_gpu.sh 0 vgg16.ckpt val.mindrecord
```

2. 多卡训练

```bash
bash scripts/run_distribute_train_gpu.sh DEVICE_NUM VGG16_CKPT VAL_MINDRECORD_FILE
# example: bash scripts/run_distribute_train_gpu.sh 8 vgg16.ckpt val.mindrecord
```

训练完毕后，开始验证PyramidBox模型。

3. 评估模型

``` bash
# 用FDDB数据集评估
bash scripts/run_eval_gpu.sh PYRAMIDBOX_CKPT
# example: bash scripts/run_eval_gpu.sh checkpoints/pyramidbox.ckpt
```

# 脚本说明

## 脚本及样例代码

```bash
PyramidBox
├── data                                        // 保存预处理后数据集文件和mindrecord文件
├── eval.py                                     // 评估模型脚本
├── preprocess.py                               // 数据集标注文件预处理脚本
├── generate_mindrecord.py                      // 创建mindrecord文件脚本
├── README_CN.md                                // PyramidBox中文描述文档
├── scripts
│   ├── generate_mindrecord_onet.sh             // 生成用于验证的mindrecord文件shell脚本
│   ├── run_distributed_train_gpu.sh            // GPU多卡训练shell脚本
│   ├── run_eval_gpu.sh                         // GPU模型评估shell脚本
│   └── run_standalone_train_gpu.sh             // GPU单卡训练shell脚本
├── src
│   ├── augmentations.py                        // 数据增强脚本
│   ├── dataset.py                              // 数据集脚本
│   ├── evaluate.py                             // 模型评估脚本
│   ├── loss.py                                 // 损失函数
│   ├── config.py                               // 配置文件
│   ├── bbox_utils.py                           // box处理函数
│   ├── detection.py                            // decode模型预测点和置信度
│   ├── prior_box.py                            // 默认候选框生成脚本
│   └── pyramidbox.py                           // PyramidBox模型
└── train.py                                    // 训练模型脚本

```

## 脚本参数

### 训练模型

```bash
usage: train.py [-h] [--basenet BASENET] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--device_target {GPU,Ascend}]
                [--lr LR] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                [--gamma GAMMA] [--distribute DISTRIBUTE]
                [--save_folder SAVE_FOLDER] [--epoches EPOCHES]
                [--val_mindrecord VAL_MINDRECORD]

Pyramidbox face Detector Training With MindSpore

optional arguments:
  -h, --help            show this help message and exit
  --basenet BASENET     Pretrained base model
  --batch_size BATCH_SIZE
                        Batch size for training
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --device_target {GPU,Ascend}
                        device for training
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum MOMENTUM   Momentum value for optim
  --weight_decay WEIGHT_DECAY
                        Weight decay for SGD
  --gamma GAMMA         Gamma update for SGD
  --distribute DISTRIBUTE
                        Use mutil Gpu training
  --save_folder SAVE_FOLDER
                        Directory for saving checkpoint models
  --epoches EPOCHES     Epoches to train model
  --val_mindrecord VAL_MINDRECORD
                        Path of val mindrecord file
```

### 评估模型

```bash
usage: eval.py [-h] [--model MODEL] [--thresh THRESH]

PyramidBox Evaluatuon on Fddb

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    trained model
  --thresh THRESH  Final confidence threshold
```

### 配置参数

```bash
config.py:
    LR_STEPS: 单卡训练学习率衰减步数
    DIS_LR_STEPS: 多卡训练学习率衰减步数
    FEATURE_MAPS: 训练集数据特征形状列表
    INPUT_SIZE: 输入数据大小
    STEPS: 生成默认候选框步数
    ANCHOR_SIZES: 默认候选框尺寸
    NUM_CLASSES: 分类类别数
    OVERLAP_THRESH: 重合度阈值
    NEG_POS_RATIOS: 负样本与正样本比例
    NMS_THRESH: nms阈值
    TOP_K: top k数量
    KEEP_TOP_K: 保留的top k数量
    CONF_THRESH: 置信度阈值
    HOME: 数据集主目录
    FACE.FILE_DIR: data文件夹路径
    FACE.TRIN_FILE: face_train.txt文件
    FACE.VAL_FILE: face_val.txt文件
    FACE.FDDB_DIR: FDDB文件夹
    FACE.WIDER_DIR: WIDER face文件夹
```

## 训练过程

在开始训练之前，请确保已完成准备工作，即：

1. `src/config.py`文件的`_C.HOME`字段为dataset文件夹路径
2. 对wider_face_train_bbx_gt.txt和wider_face_val_bbx_gt.txt文件进行预处理以生成face_train.txt和face_val.txt文件。
3. 生成face_val.txt的mindrecord文件。
4. 下载预训练完成的[vgg16.ckpt](https://pan.baidu.com/s/1e5qSW4e1QVZRnbyGRWi91Q?pwd=dryt)文件。

准备工作完成后方可训练。

### 单卡训练

```bash
bash scripts/run_standalone_train_gpu.sh DEVICE_ID VGG16_CKPT VAL_MINDRECORD_FILE
# example: bash scripts/run_standalone_train_gpu.sh 0 vgg16.ckpt val.mindrecord
```

训练过程会在后台运行，训练模型将保存在`checkpoints`文件夹中，可以通过`logs/training_gpu.log`文件查看训练输出，输出结果如下所示：

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

### 多卡训练

```bash
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VGG16_CKPT] [VAL_MINDRECORD_FILE]
# example: bash scripts/run_distribute_train_gpu.sh 8 vgg16.ckpt val.mindrecord
```

训练过程会在后台运行，只保存第一张卡的训练模型，训练模型将保存在`checkpoints/distribute_0/`文件夹中，可以通过`logs/distribute_training_gpu.log`文件查看训练输出，输出结果如下所示：

```bash
epoch: 1 total step: 2, step: 2, loss is 25.479286
epoch: 1 total step: 2, step: 2, loss is 30.297405
epoch: 1 total step: 2, step: 2, loss is 28.816475
epoch: 1 total step: 2, step: 2, loss is 25.439453
epoch: 1 total step: 2, step: 2, loss is 28.585438
epoch: 1 total step: 2, step: 2, loss is 31.117134
epoch: 1 total step: 2, step: 2, loss is 25.770748
epoch: 1 total step: 2, step: 2, loss is 27.557945
epoch: 1 total step: 3, step: 3, loss is 28.352016
epoch: 1 total step: 3, step: 3, loss is 31.99873
epoch: 1 total step: 3, step: 3, loss is 31.426039
epoch: 1 total step: 3, step: 3, loss is 24.02226
epoch: 1 total step: 3, step: 3, loss is 30.12824
epoch: 1 total step: 3, step: 3, loss is 29.977898
epoch: 1 total step: 3, step: 3, loss is 24.06476
epoch: 1 total step: 3, step: 3, loss is 28.573633
epoch: 1 total step: 4, step: 4, loss is 28.599226
epoch: 1 total step: 4, step: 4, loss is 34.262005
epoch: 1 total step: 4, step: 4, loss is 30.732353
epoch: 1 total step: 4, step: 4, loss is 28.62697
epoch: 1 total step: 4, step: 4, loss is 39.44549
epoch: 1 total step: 4, step: 4, loss is 27.754185
epoch: 1 total step: 4, step: 4, loss is 26.15754
...
```

## 评估过程

```bash
bash scripts/run_eval_gpu.sh [PYRAMIDBOX_CKPT]
# example: bash scripts/run_eval_gpu.sh checkpoints/pyramidbox.ckpt
```

注：模型名称为`pyramidbox_best_{epoch}.ckpt`，epoch表示该检查点保存时训练的轮数，epoch越大，WIDER val的loss值越小，模型精度相对越高，因此在评估最佳模型时，优先评估epoch最大的模型，按照epoch从大到小的顺序评估。

评估过程会在后台进行，评估结果可以通过`logs/eval_gpu.log`文件查看，输出结果如下所示：

```bash
==================== Results ====================
FDDB-fold-1 Val AP: 0.9614604685893
FDDB-fold-2 Val AP: 0.9615593696135745
FDDB-fold-3 Val AP: 0.9607889632039851
FDDB-fold-4 Val AP: 0.972454404596466
FDDB-fold-5 Val AP: 0.9734522365236052
FDDB-fold-6 Val AP: 0.952158002966933
FDDB-fold-7 Val AP: 0.9618735923917133
FDDB-fold-8 Val AP: 0.9501671313630741
FDDB-fold-9 Val AP: 0.9539008001056393
FDDB-fold-10 Val AP: 0.9664355605240443
FDDB Dataset Average AP: 0.9614250529878333
=================================================
```

# 模型描述

## 性能

| 参数                 | PyramidBox                                               |
| -------------------- | ------------------------------------------------------- |
| 资源                 | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G|
| 上传日期             | 2022-09-17                                             |
| MindSpore版本        | 1.8.1                                                  |
| 数据集               | WIDER Face, FDDB |
| 训练参数             | epoch=100,batch_size=4, lr=5e-4 |
| 优化器               | SGD                                                |
| 损失函数             | SoftmaxCrossEntropyWithLogits, SmoothL1Loss                       |
| 输出                 | 坐标，置信度                                               |
| 损失                 | 2-6                |
| 速度                 | 570毫秒/步(单卡)  650毫秒/步(八卡)                            |
| 总时长               | 50时58分(单卡)；7时12分(八卡)                                                |
| 微调检查点 | 655M (.ckpt文件)       |
