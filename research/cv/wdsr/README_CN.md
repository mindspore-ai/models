目录

<!-- TOC -->

- [目录](#目录)
- [WDSR描述](#WDSR描述)
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
- [模型导出](#模型导出)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# WDSR描述

WDSR于2018年提出的 WDSR用于提高深度超分辨率网络的精度，它在 NTIRE 2018 单幅图像超分辨率挑战赛中获得了所有三个真实赛道的第一名。
[论文1](https://arxiv.org/abs/1707.02921)：Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** *2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**.[论文2](https://arxiv.org/abs/1808.08718): Jiahui Yu, Yuchen Fan, Jianchao Yang, Ning Xu, Zhaowen Wang, Xinchao Wang, Thomas Huang, **"Wide Activation for Efficient and Accurate Image Super-Resolution"**, arXiv preprint arXiv:1808.08718.

# 模型架构

WDSR网络主要由几个基本模块（包括卷积层和池化层）组成。通过更广泛的激活和线性低秩卷积，并引入权重归一化实现更高精度的单幅图像超分辨率。这里的基本模块主要包括以下基本操作： **1 × 1 卷积**和**3 × 3 卷积**。
经过1次卷积层,再串联32个残差模块,再经过1次卷积层,最后上采样并卷积。

# 数据集

使用的数据集：[DIV2K](<http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf>)

- 数据集大小：7.11G，

    - 训练集：共900张图像
    - 测试集：共100张图像

- 数据格式：png文件

  - 注：数据将在src/data/DIV2K.py中处理。

  ```shell
  DIV2K
  ├── DIV2K_test_LR_bicubic
  │   ├── X2
  │   │   ├── 0901x2.png
  │   │   ├─ ...
  │   │   └── 1000x2.png
  │   ├── X3
  │   │   ├── 0901x3.png
  │   │   ├─ ...
  │   │   └── 1000x3.png
  │   └── X4
  │       ├── 0901x4.png
  │        ├─ ...
  │       └── 1000x4.png
  ├── DIV2K_test_LR_unknown
  │   ├── X2
  │   │   ├── 0901x2.png
  │   │   ├─ ...
  │   │   └── 1000x2.png
  │   ├── X3
  │   │   ├── 0901x3.png
  │   │   ├─ ...
  │   │   └── 1000x3.png
  │   └── X4
  │       ├── 0901x4.png
  │       ├─ ...
  │       └── 1000x4.png
  ├── DIV2K_train_HR
  │   ├── 0001.png
  │   ├─ ...
  │   └── 0900.png
  ├── DIV2K_train_LR_bicubic
  │   ├── X2
  │   │   ├── 0001x2.png
  │   │   ├─ ...
  │   │   └── 0900x2.png
  │   ├── X3
  │   │   ├── 0001x3.png
  │   │   ├─ ...
  │   │   └── 0900x3.png
  │   └── X4
  │       ├── 0001x4.png
  │       ├─ ...
  │       └── 0900x4.png
  └── DIV2K_train_LR_unknown
      ├── X2
      │   ├── 0001x2.png
      │   ├─ ...
      │   └── 0900x2.png
      ├── X3
      │   ├── 0001x3.png
      │   ├─ ...
      │   └── 0900x3.png
      └── X4
          ├── 0001x4.png
          ├─ ...
          └── 0900x4.png
  ```

# 环境要求

- 硬件（Ascend/GPU）
    - 使用ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```shell
# 单卡训练
# Ascend
sh ./script/run_ascend_standalone.sh [TRAIN_DATA_DIR]
# GPU
bash ./script/run_gpu_standalone.sh [TRAIN_DATA_DIR]
```

```shell
# 分布式训练
# Ascend
sh ./script/run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
# GPU
bash ./script/run_gpu_distribute.sh [TRAIN_DATA_DIR] [DEVICE_NUM]
```

```python
#评估
bash ./script/run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] DIV2K
```

- TEST_DATA_DIR = ".../wdsr/"
- CHECKPOINT_PATH ckpt文件目录。

# 脚本说明

## 脚本及样例代码

```bash
WDSR
   ├── README_CN.md                           //自述文件
   ├── script
   │        ├── run_ascend_distribute.sh      //Ascend分布式训练shell脚本
   │        ├── run_ascend_standalone.sh      //Ascend单卡训练shell脚本
   │        ├── run_gpu_distribute.sh         //GPU分布式训练shell脚本
   │        ├── run_gpu_standalone.sh         //GPU单卡训练shell脚本
   │        └── run_eval.sh                   //eval验证shell脚本
   │        └── run_eval_onnx_gpu.sh          //onnx验证shell脚本
   ├── src
   │     ├── args.py                          //超参数
   │     ├── common.py                        //公共网络模块
   │     ├── data
   │     │      ├── common.py                 //公共数据集
   │     │      ├── div2k.py                  //div2k数据集
   │     │      └── srdata.py                 //所有数据集
   │     ├── metrics.py                       //PSNR和SSIM计算器
   │     ├── model.py                         //WDSR网络
   │     └── utils.py                         //辅助函数
   ├── train.py                               //训练脚本
   ├── eval.py                                //评估脚本
   ├── eval_onnx.py                           //onnx评估脚本
   └── export.py
```

## 脚本参数

主要参数如下:

```python
  -h, --help                  show this help message and exit
  --dir_data DIR_DATA         dataset directory
  --data_train DATA_TRAIN     train dataset name
  --data_test DATA_TEST       test dataset name
  --data_range DATA_RANGE     train/test data range
  --ext EXT                   dataset file extension
  --scale SCALE               super-resolution scale
  --patch_size PATCH_SIZE     output patch size
  --rgb_range RGB_RANGE       maximum value of RGB
  --n_colors N_COLORS         number of color channels to use
  --no_augment                do not use data augmentation
  --model MODEL               model name
  --n_resblocks N_RESBLOCKS   number of residual blocks
  --n_feats N_FEATS           number of feature maps
  --res_scale RES_SCALE       residual scaling
  --test_every TEST_EVERY     do test per every N batches
  --epochs EPOCHS             number of epochs to train
  --batch_size BATCH_SIZE     input batch size for training
  --test_only                 set this option to test the model
  --lr LR                     learning rate
  --ckpt_path CKPT_PATH       path of saved ckpt
  --ckpt_save_path CKPT_SAVE_PATH              path to save ckpt
  --ckpt_save_interval CKPT_SAVE_INTERVAL      save ckpt frequency, unit is epoch
  --ckpt_save_max CKPT_SAVE_MAX                max number of saved ckpt
  --task_id TASK_ID

```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  sh ./script/run_ascend_standalone.sh [TRAIN_DATA_DIR]
  ```

- GPU环境运行

  ```bash
  sh ./script/run_gpu_standalone.sh [TRAIN_DATA_DIR]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  sh ./script/run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]
  ```

- GPU环境运行

  ```bash
  sh ./script/run_gpu_distribute.sh [TRAIN_DATA_DIR] [DEVICE_NUM]
  ```

TRAIN_DATA_DIR = ".../wdsr/"

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```bash
bash ./script/run_eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] DIV2K
```

- TEST_DATA_DIR = ".../wdsr/"
- CHECKPOINT_PATH ckpt文件目录。

您可以通过eval.log文件查看结果。

### Ascend310评估

- 评估过程如下，需要指定数据集类型为“DIV2K”。

```bash
sh ./script/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATASET_TYPE] [SCALE] [DEVICE_ID]
```

- MINDIR_PATH mindir模型文件路径
- DATA_PATH 数据集路径
- DATASET_TYPE 数据集名称(DIV2K)
- SCALE 超分辨率比例(2, 3, 4)
- DEVICE_ID 设备ID， 默认为：0
- 上述python命令在后台运行，可通过`run_infer.log`文件查看结果。

# 模型导出

```bash
python export.py --ckpt_path [CKPT_PATH] --file_format [FILE_FORMAT] --device_target [DEVICE_TARGET] --dir_data [DIR_DATA] --test_only --ext "img"  --data_test DIV2K --data_range "801-900"
```

- CKPT_PATH ckpt文件目录。
- FILE_FORMAT 可选 ['MINDIR', 'AIR', 'ONNX'], 默认['MINDIR']。
- DEVICE_TARGET 可选 ['Ascend', 'GPU'], 默认['Ascend']。
- DIR_DATA = ".../wdsr/" 数据集所在文件夹目录

## ONNX模型评估

```bash
bash ./script/run_eval_onnx_gpu.sh [DIR_DATA] [ONNX_PATH] DIV2K
```

- DIR_DATA = ".../wdsr/"
- ONNX_PATH = ".../wdsr/wdsr"  不指定特定的onnx模型，对于不同大小的图像采用不同的onnx文件

## 310推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

# 模型描述

## 性能

### 训练性能

| 参数          | Ascend                                                       | GPU|
| ------------- | ------------------------------------------------------------ |----|
| 资源          | Ascend 910                                                   |NVIDIA GeForce RTX 3090|
| 上传日期      | 2021-7-4                                                     |2021-11-22|
| MindSpore版本 | 1.2.0                                                        |1.5.0|
| 数据集        | DIV2K                                                        |DIV2K|
| 训练参数      | epoch=1000, steps=100, batch_size =16, lr=0.0001            |epoch=300, batch_size=16, lr=0.0005|
| 优化器        | Adam                                                         |Adam|
| 损失函数      | L1                                                           |L1|
| 输出          | 超分辨率图片                                                 |超分辨率图片|
| 损失          | 3.5                                                          |3.3|
| 速度          | 1卡：约115毫秒/步                                            |8卡：约140毫秒/步|
|              | 8卡：约130毫秒/步                                            |               |
| 总时长        | 1卡：4小时                                                     |8卡：1.5小时|
|              | 8卡：0.5小时                                                   |          |
| 微调检查点    | 35 MB(.ckpt文件)                                        |14 MB(.ckpt文件)|
| 脚本          | [WDSR](https://gitee.com/mindspore/models/tree/r2.0/research/cv/wdsr) |[WDSR](https://gitee.com/mindspore/models/tree/r2.0/research/cv/wdsr)|

### 评估性能

| 参数          | Ascend                                                      |GPU                    |
| ------------- | ----------------------------------------------------------- |----------------------|
| 资源          | Ascend 910                                                  |NVIDIA GeForce RTX 3090|
| 上传日期      | 2021-7-4                                                    |2021-11-22              |
| MindSpore版本 | 1.2.0                                                       |1.5.0                  |
| 数据集        | DIV2K                                                       |DIV2K                   |
| batch_size    | 1                                                           |1                      |
| 输出          | 超分辨率图片                                                  |超分辨率图片              |
| PSNR          | 1p DIV2K 34.77                                            |8p DIV2K 35.97            |
|               | 8p DIV2K 33.59                                           |                          |

### 310评估性能

| 参数          | Ascend                                                      |
| ------------- | ----------------------------------------------------------- |
| 资源          | Ascend 310                                                  |
| 上传日期      | 2021-10-4                                                    |
| MindSpore版本 | 1.3.0                                                       |
| 数据集        | DIV2K                                                       |
| batch_size    | 1                                                           |
| 输出          | 超分辨率图片                                                |
| PSNR          | DIV2K 33.5745                                               |

### ONNX评估性能

| 参数          | GPU           |
| ------------- |---------------|
| 资源          | RTX 3090      |
| 上传日期      | 2022-10-19    |
| MindSpore版本 | 1.8.0         |
| 数据集        | DIV2K         |
| batch_size    | 1             |
| 输出          | 超分辨率图片        |
| PSNR          | DIV2K 35.3531 |

# 随机情况说明

在train.py中，我们设置了“train_net”函数内的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
