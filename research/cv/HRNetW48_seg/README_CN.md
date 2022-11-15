# 目录

<!-- TOC -->

- [目录](#目录)
- [HRNet描述](#hrnet描述)
    - [概述](#概述)
    - [论文](#论文)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [训练时推理](#训练时推理)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
        - [结果](#结果-1)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
    - [MindX推理](#mindx推理)
        - [导出AIR](#导出air)
        - [导出OM](#导出om)
        - [MxBase推理](#mxbase推理)
        - [SDK推理](#sdk推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [Cityscapes上HRNetV2-W48的性能](#cityscapes上HRNetV2-w48的性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# HRNet描述

## 概述

HRNet是一个全能型的计算机视觉骨干网络，可用于图像分类、语义分割、面部识别等多种计算机视觉任务的特征提取。该网络分为四个阶段，每个阶段产生一个分辨率更低的分支，通过重复连接不同分辨率的特征图的方式增强特征表示。本脚本以HRNet为骨干网络，实现语义分割任务。

## 论文

[High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514)

# 数据集

1. 数据集 [Cityscapes](https://www.cityscapes-dataset.com/)

Cityscapes数据集包含5000幅高质量像素级别精细注释的街城市道场景图像。图像按2975/500/1525的分割方式分为三组，分别用于训练、验证和测试。数据集中共包含30类实体，其中19类用于验证。

2. 数据集 [LIP](https://lip.sysuhcp.com/overview.php)

Look into person (LIP)数据集用于训练模型进行人类身体部位分割。该数据集包含50000张图像，带有详细的像素注释，其中包含19个语义人体部位标签和16个关键点的二维人体姿势。

数据集下载后的结构模式如下：

```text
$DATASET
├─ cityscapes                           # cityscapes数据集根目录
│   ├─ gtFine                           # 标签文件
│   │   ├─ train                        # 训练标签文件
│   │   │   └─ [city folders]
│   │   │       └─ [label images]
│   │   └─ val                          # 推理标签文件
│   │       └─ [city folders]
│   │           └─ [label images]
│   ├─ leftImg8bit                      # 图像文件
│   │   ├─ train                        # 训练图像文件
│   │   │   └─ [city folders]
│   │   │       └─ [images]
│   │   └─ val                          # 推理图像文件
│   │       └─ [city folders]
│   │           └─ [images]
│   ├─ train.lst                        # 训练样本列表
│   └─ val.lst                          # 推理样本列表
├─ lip                                  # lip数据集根目录
│   ├─ TrainVal_images                  # 图像文件
│   │   ├─ train_images                 # 训练图像文件
│   │   │   └─ [images]
│   │   └─ val_images                   # 推理图像文件
│   │       └─ [images]
│   ├─ TrainVal_parsing_annotations     # 标签文件
│   │   ├─ train_segmentations          # 训练标签文件
│   │   │   └─ [label images]
│   │   └─ val_segmentations            # 推理标签文件
│   │       └─ [label images]
│   ├─ train_id.txt                     # 训练样本id
│   ├─ val_id.txt                       # 推理样本id
│   ├─ train.lst                        # 训练样本列表
│   └─ val.lst                          # 推理样本列表
```

生成 `train.lst` 和 `val.lst` 文件：

```bash
# 首先将lst文件之外的其他文件按上述目录结构存放
# 运行./src/dataset/目录下对应的maker脚本生成lst文件
# 生成cityscapes数据集的lst文件
python cityscapes_lst_maker.py --root [DATASET_ROOT]
# 生成lip数据集的lst文件
python lip_lst_maker.py --root [DATASET_ROOT]
# Example
python cityscapes_lst_maker.py --root ../data/cityscapes/
```

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境
- 框架
    - [Mindspore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 分布式训练，从指定周期开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定周期开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

如果要在ModelArts上进行模型的训练，可以参考ModelArts的 [官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```text
# 训练模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    dataset             [cityscapes/lip]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    run_distribute      [True/False]
    begin_epoch         [起始周期]
    end_epoch           [结束周期]
    eval                [True/False]
    eval_start          [推理起始周期]
    interval            [推理周期间隔]
3. 选择相应数量的处理器
4. 开始运行

# 评估模型
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    dataset             [cityscapes/lip]
3. 选择单个处理器
4. 开始运行
```

# 脚本说明

## 脚本及样例代码

```text
├─ seg-hrnet
│   ├─ ascend310_infer                      # 310推理相关脚本
│   │   ├─ inc
│   │   │   └─ utils.h
│   │   └─ src
│   │       ├─ build.sh
│   │       ├─ CMakeLists.txt
│   │       ├─ main.cc
│   │       └─ utils.cc
│   ├─ infer                                # MindX推理相关脚本
│   │   ├─ convert                          # om模型转换相关脚本
│   │   │   ├─ convert.sh                   # om模型转换执行脚本
│   │   │   └─ hrnetw48seg_aipp.cfg         # om模型转换配置信息
│   │   ├─ data                             # 推理过程所需数据信息
│   │   │   └─ config
│   │   │       ├─ hrnetw48seg.cfg
│   │   │       ├─ hrnetw48seg.names
│   │   │       └─ hrnetw48seg.pipeline
│   │   ├─ mxbase                           # MxBase推理相关脚本
│   │   │   ├─ src
│   │   │   │   ├─ hrnetw48seg.cpp
│   │   │   │   ├─ hrnetw48seg.h
│   │   │   │   └─ main.cpp
│   │   │   ├─ build.sh                     # MxBase推理执行脚本
│   │   │   └─ CMakeLists.txt
│   │   ├─ sdk                              # SDK推理相关脚本
│   │   │   ├─ cityscapes.py
│   │   │   ├─ do_infer.sh                  # SDK推理执行脚本
│   │   │   └─ main.py
│   │   └─ docker_start.sh                  # 镜像启动
│   ├─ modelarts                            # ModelArts训练相关脚本
│   │   └─ start.py                         # ModelArts训练启动脚本
│   ├─ scripts                              # Ascend执行脚本
│   │   ├─ docker_start.sh                  # MindX推理及MxBase推理docker启动
│   │   ├─ ascend310_inference.sh           # 启动Ascend310推理（单卡）
│   │   ├─ run_standalone_train.sh          # 启动Ascend910单机训练（单卡）
│   │   ├─ run_distribute_train.sh          # 启动Ascend910分布式训练（8卡）
│   │   └─ run_eval.sh                      # 启动Ascend910推理（单卡）
│   ├─ src
│   │   ├─ dataset                          # 数据集预处理相关脚本
│   │   │   ├─ basedataset.py               # 数据集生成器基类
│   │   │   ├─ cityscapes.py                # Cityscapes生成器
│   │   │   ├─ cityscapes_lst_maker.py      # Cityscapes list生成器
│   │   │   ├─ lip.py                       # LIP生成器
│   │   │   ├─ lip_lst_maker.py             # LIP list生成器
│   │   │   └─ dataset_generator.py         # 数据集生成脚本
│   │   ├─ callback.py                      # 自定义回调函数
│   │   ├─ config.py                        # 参数配置
│   │   ├─ loss.py                          # 损失函数定义
│   │   └─ seg_hrnet.py                     # HRNetW48语义分割网络定义
│   ├─ eval.py                              # 910推理脚本
│   ├─ export.py                            # 模型转换脚本
│   ├─ postprocess.py                       # 310推理后处理脚本，计算mIoU
│   ├─ preprocess.py                        # 310推理前处理脚本，数据预处理
│   ├─ torch2mindspore.py                   # 转换torch模型为mindspore格式，用于预训练模型获取
│   └─ train.py                             # 910训练脚本
```

## 脚本参数

在配置文件中可以同时配置训练参数和评估参数。

```text
# 训练超参数
"lr": 0.01,                     # 初始学习率
"lr_min": 1e-7,                 # 最终学习率
"lr_power": 0.9,                # poly学习率衰减策略系数
"lr_scheme": "cos",             # 学习率衰减策略
"optimizer": "adam",            # 优化器
"save_checkpoint_epochs": 5,    # checkpoint保存频率
"keep_checkpoint_max": 10,      # checkpoint保存数量
"total_epoch": 600,             # 训练周期数
"batchsize": 4,                 # 批量大小
"loss_scale": 1024,             # loss scale
"opt_momentum": 0.99,           # 优化器动量
"wd": 1e-5,                     # 优化器weight decay
"eps": 1e-3                     # 优化器epsilon

# HRNetW48模型参数
"FINAL_CONV_KERNEL": 1,         # 最后一层卷积的kernel size
"STAGE1":                       # Stage1 结构
    "NUM_MODULES": 1,           # HRModule数量
    "NUM_BRANCHES": 1,          # 分支数量
    "BLOCK": "BOTTLENECK",      # 块类型
    "NUM_BLOCKS": [4],          # 各个分支块的数量
    "NUM_CHANNELS": [64],       # 各个分支的通道数量（宽度W）
    "FUSE_METHOD": "SUM"        # 融合方式
"STAGE2":                       # Stage2 结构
    "NUM_MODULES": 1,
    "NUM_BRANCHES": 2,
    "BLOCK": "BASIC",
    "NUM_BLOCKS": [4, 4],
    "NUM_CHANNELS": [48, 96],
    "FUSE_METHOD": "SUM"
"STAGE3":                       # Stage3 结构
    "NUM_MODULES": 4,
    "NUM_BRANCHES": 3,
    "BLOCK": "BASIC",
    "NUM_BLOCKS": [4, 4, 4],
    "NUM_CHANNELS": [48, 96, 192],
    "FUSE_METHOD": "SUM"
"STAGE4":                       # Stage4 结构
    "NUM_MODULES": 3,
    "NUM_BRANCHES": 4,
    "BLOCK": "BASIC",
    "NUM_BLOCKS": [4, 4, 4, 4],
    "NUM_CHANNELS": [48, 96, 192, 384],
    "FUSE_METHOD": "SUM"
```

## 训练过程

### 获取预训练模型

训练HRNet实现语义分割任务需要以HRNet基于ImageNet2012数据集实现图像分类任务的训练结果作为预训练模型，以下提供几种获取预训练模型的方法：

1. 转换HRNet官方提供模型

   [HRNet官方网站](https://github.com/HRNet/HRNet-Image-Classification) 提供预训练模型下载链接，下载文件经 `torch2mindspore.py` 转换产出训练所需预训练模型。

   ```bash
   python torch2mindspore.py --pth_path [TORCH_MODEL_PATH] --ckpt_path [OUTPUT_MODEL_PATH]
   ```

### 用法

#### Ascend处理器环境运行

```bash
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 分布式训练，从指定周期开始恢复训练
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# 单机训练，从指定周期开始恢复训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见 [hccn_tools]([MindSpore](https://www.mindspore.cn/tutorials/zh-CN/r1.5/intermediate/distributed_training/distributed_training_ascend.html)) 中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

运行单卡用例时如果想更换运行卡号，可以通过设置环境变量 `export DEVICE_ID=x`。

#### 训练时推理

如果需要训练时推理，在执行shell脚本时为 `EVAL_CALLBACK` 参数传入 `True` 即可，其默认值为 `False` 。

### 结果

使用Cityscapes数据集训练HRNetV2-W48

```text
# 分布式训练结果（8p）
epoch: [  1/600], epoch time: 520409.557, steps:   743, per step time: 700.417, avg loss: 0.343, lr:[0.000100]
epoch: [  2/600], epoch time: 191926.929, steps:   743, per step time: 258.313, avg loss: 0.227, lr:[0.000100]
epoch: [  3/600], epoch time: 191954.042, steps:   743, per step time: 258.350, avg loss: 0.658, lr:[0.000100]
epoch: [  4/600], epoch time: 191917.645, steps:   743, per step time: 258.301, avg loss: 0.291, lr:[0.000100]
epoch: [  5/600], epoch time: 195191.187, steps:   743, per step time: 262.707, avg loss: 0.343, lr:[0.000100]
epoch: [  6/600], epoch time: 191863.135, steps:   743, per step time: 258.228, avg loss: 0.144, lr:[0.000100]
···
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。你可在此路径下的日志文件中找到如下结果：

```text
Number of samples:  500, total time: 400.28s, average time: 0.80s
============= 910 Inference =============
miou: 0.7921274714446462
iou array:
 [0.98434563 0.86852768 0.93243294 0.51712274 0.61742523 0.71696446
 0.7468067  0.83170647 0.92999597 0.62933721 0.95104717 0.84875123
 0.6644514  0.95406839 0.66889207 0.8800112  0.81180921 0.69305802
 0.80366822]
=========================================
```

## 推理过程

### 导出MindIR

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CKPT_PATH] --file_name [FILE_NAME] --file_format MINDIR --device_target Ascend --dataset [DATASET]
```

### 在Ascend310执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理之前，必须先通过 `export.py` 脚本到本mindir文件。以下展示了使用mindir模型执行推理的示例。目前只支持Cityscapes数据集batchsize为1的推理。

```bash
bash scripts/ascend_310_infer.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件的存储路径
- `DATA_PATH` Cityscapes原始数据集的存储路径
- `DEVICE_ID` 卡号

脚本内部分为三步：

1. `preprocess.py` 对原始数据集进行预处理，并将处理后的数据集以二进制的形式存储在`./preprocess_Result/`路径下；
2. `ascend310_infer/src/main.cc` 执行推理过程，并将预测结果以二进制的形式存储在`./result_Files/`路径下，推理日志可在`infer.log`中查看；
3. `postprocess.py` 利用预测结果与相应标签计算mIoU，计算结果可在 `acc.log` 中查看。

### 结果

```text
Total number of images:  500
=========== 310 Inference Result ===========
miou: 0.7920974410146139
iou array:
 [0.98435467 0.86849459 0.93240479 0.5170266  0.61721129 0.71663695
 0.74685659 0.83161668 0.92993842 0.62925042 0.95103679 0.84861963
 0.66363543 0.95415057 0.67026404 0.88005176 0.81205242 0.6925974
 0.80365232]
============================================
```

## MindX推理

### 导出AIR

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CKPT_PATH] --file_name [FILE_NAME] --file_format AIR --device_target Ascend --dataset [DATASET]
```

### 导出OM

```bash
cd ./infer/convert/
bash convert.sh [AIR_MODEL_PATH] [AIPP_CONFIG_PATH] [OM_MODEL_NAME]

# Example
bash convert.sh ./hrnetw48seg.air ./hrnetw48seg_aipp.cfg ../data/model/hrnetw48seg
```

其中， `[OM_MODEL_NAME]` 为om模型输出路径，默认路径为 `../data/model/hrnetw48seg` 。用户亦可自定义，但需要在 `./infer/data/config/hrnetw48seg.pipeline` 和 `./infer/mxbase/src/main.cpp` 脚本中进行文件名统一。

### MxBase推理

```bash
# 执行MxBase推理
cd ./infer/mxbase/
bash build.sh
build/hrnetw48seg [TEST_IMAGE_PATH]

# Example
build/hrnetw48seg ./test.png
```

推理结果为输入图片的语义分割效果图，输出位置与输入图片相同，命名以 `_infer` 结尾。

### SDK推理

```bash
# 执行SDK推理
cd ./infer/sdk/
bash do_infer.sh [DATA_PATH] [DATA_LST_PATH]

# Example
bash do_infer.sh ../data/input/cityscapes ../data/input/cityscapes/val.lst
```

推理得到测试集全部图片的语义分割效果图，结果存储于 `./inferResults/` 目录。精度结果在推理结束后打印在执行窗口，与910推理的精度差异可控制在0.5%以内。

# 模型描述

## 性能

### 评估性能

#### Cityscapes上HRNetV2-W48的性能

|参数|Ascend 910|
|------------------------------|------------------------------|
|模型版本|HRNetV2-W48|
|资源|Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8|
|上传日期|2021-12-12|
|MindSpore版本|1.1.3|
|数据集|Cityscapes|
|训练参数|epoch=600, steps per epoch=93, batch_size = 4|
|优化器|Adam|
|损失函数|SoftmaxCrossEntropyWithLogits|
|输出|mIoU|
|损失|0.0507812|
|速度|256毫秒/步（8卡）|
|总时长| 5小时                                                    |

# 随机情况说明

`train.py`中使用了随机种子。

# ModelZoo主页

 请浏览官网 [主页](https://gitee.com/mindspore/models) 。
