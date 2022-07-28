# 目录

[View English](./README.md)

<!--TOC-->

- [3D_DenseNet](#3D_DenseNet)

- [简介](#简介)

- [数据集](#数据集)

- [环境要求](#环境要求)

  -[Python Package](#Python-Package)

- [快速入门](#快速入门)

  -[安装步骤](#安装步骤)

  -[在modelarts上训练](#在modelarts上训练)

- [脚本说明](#脚本说明)

  -[脚本和样例代码](#脚本和样例代码)

  -[脚本参数](#脚本参数)

- [训练过程](#训练过程)

  -[单P训练](#单P训练)

  -[分布式训练](#分布式训练)

- [评估过程](#评估过程)

  -[评估](#评估)

- [导出过程](#导出)

  -[导出](#导出)

- [性能](#性能)

  -[训练性能](#训练性能)

# 3D_DenseNet

## 简介

3D-SkipDenseSeg —— Skip-connected 3D DenseNet for volumetric infant brain MRI segmentation By Toan Duc Bui, Jitae Shin, Taesup Moon
6个月大的婴儿大脑分割数据集将大脑分为:（1） 白质（White matter）, （2） 灰质 （Gray matter）, 和（3）脑脊液（Cerebrospinal fluid）。因为组织之间的大范围重合，图片数据的低对比度，这是一个非常困难的任务。 论文采用了非常深的3D卷积神经网络来解决这个问题，最终的结果在6种评价指标下处于前列。

## 数据集

数据集是MICCAI Grand Challenge 的 [6-month infant brain MRI segmentation-in conjunction with MICCAI 2017](http://iseg2017.web.unc.edu)。
人类出生后的第一年是人类大脑发育最活跃的阶段，伴随着组织的快速生长和广泛的认知和运动功能的发展。这一早期阶段对于许多神经发育和神经精神疾病的诊断，都是至关重要的。越来越多的人开始关注这一关键时期。在这一关键时期，对婴儿脑MR图像进行白质(WM)、灰质(GM)和脑脊液(CSF)的准确分割，对于研究正常和异常的早期大脑发育都具有重要的基础意义。

下载完成之后将得到两个压缩包，iSeg-2017-Training.zip 和iSeg-2017-Testing.zip，使用代码仓提供的prepare_hdf5_cutedge.py进行处理iSeg-2017-Training文件夹下的数据。整理target_path输出的结果。具体地，是将第九个hdf5文件作为验证集。 进行eval和test的文件夹是解压后的原生数据。标号1-10用来进行训练和验证，后面11-23是提供的测试数据。

```python
└─data_train_no_cut
  ├── train_iseg_nocut_1.h5  // 训练集目录结构
  ├── train_iseg_nocut_2.h5
  ├── train_iseg_nocut_3.h5
  ├── train_iseg_nocut_4.h5
  ├── train_iseg_nocut_5.h5
  ├── train_iseg_nocut_6.h5
  ├── train_iseg_nocut_7.h5
  ├── train_iseg_nocut_8.h5
  ├── train_iseg_nocut_10.h5
```

```python
└─data_val_no_cut
  ├── train_iseg_nocut_9.h5  //验证集目录结构
```

```python
└─data_val
  ├──subject-9-label.hdr  // eval目录结构
  ├──subject-9-label.img
  ├──subject-9-T1.hdr
  ├──subject-9-T1.img
  ├──subject-9-T2.hdr
  ├──subject-9-T2.img
```

```python
└─data_test
  ├──subject-11-T1.hdr  // test目录结构
  ├──subject-11-T1
  ├──subject-11-T2.hdr
  ├──subject-11-T2
  ······
  ├──subject-23-T2.hdr
  ├──subject-23-T2
```

## 环境要求

-- 硬件（Ascend 或 GPU）
  -准备Ascend或GPU的硬件环境

-框架

  -[MindSpore](https://www.mindspore.cn/install/en)

-为了获得更多的信息请参考下列信息：

  -[MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)

  -[MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

### Python Package

```python
opencv_python==4.5.2.52
numpy==1.19.4
MedPy==0.4.0
h5py==3.4.0
mindspore==1.3.0
PyYAML==5.4.1
SimpleITK==2.1.1
```

## 快速入门

```python
# 在 defalut_config.yaml下面，添加数据集路径
train_dir: "/cache/data/data_train_nocut"
val_dir: "/cache/data/data_val_nocut"
```

```python
# Ascend/GPU/CPU处理器环境运行
# 为了在不同的处理器环境运行，请对配置文件default_config.yaml中的device_target 进行修改

# 运行训练示例
bash run_standalone_train.sh [TRAIN_DIR][VAL_DIR]
# example: bash  run_standalone_train.sh data/data_train_no_cut data/data_val_no_cut

# 运行分布式训练示例
bash run_distribute_train.sh [RANK_TABLE_FILE] [TRAIN_DIR] [VAL_DIR]
# example: bash run_distribute.sh  ranktable.json data/train_no_cut data/val_no_cut

# 运行评估示例
bash run_eval.sh  [CHECKPOINT_FILE_PATH] [TEST_DIR]
# example: bash run_eval.sh 3D-DenseSeg-20000_36.ckpt data/data_val

# 运行测试示例
bash run_test.sh [CHECKPOINT_FILE_PATH] [TEST_DIR]
# example: bash run_test.sh data/data_test
```

### 安装步骤

- Step 1: 下载源代码

```python
cd 3D_DenseNet
```

- Step 2: 下载数据集 链接在 `http://iseg2017.web.unc.edu/download/` 在`prepare_hdf5_cutedge.py`中配置输入的数据路径，和输出准备好的数据。

```python
data_path = '/path/to/your/dataset/'
target_path = '/path/to/your/save/hdf5 folder/'
```

- Step 3:产生 hdf5格式的数据集

```python
python prepare_hdf5_cutedge.py
```

- Step 4:1p训练

```python
run_standalone_train.sh data/data_train_no_cut data/data_val_no_cut
```

如果输出一下信息，说明训练成功。

```python
============== Starting Training ==============
epoch: 1 step: 36, loss is 0.29248548
valid_dice: 0.5158226623757749
epoch time: 119787.393 ms, per step time: 3327.428 ms
epoch: 2 step: 36, loss is 0.4542764
valid_dice: 0.577169897796093
epoch time: 3151.715 ms, per step time: 87.548 ms
epoch: 3 step: 36, loss is 0.45287344
valid_dice: 0.6642792932561518
epoch time: 3145.802 ms, per step time: 87.383 ms
epoch: 4 step: 36, loss is 0.36013693
valid_dice: 0.6175640794605014
epoch time: 3161.118 ms, per step time: 87.809 ms
epoch: 5 step: 36, loss is 0.38933912
valid_dice: 0.6884333695452182
```

执行评价脚本

```python
bash run_eval.sh 3D-DenseSeg-20000_36.ckpt data/data_val
```

你应该得到如下结果
第9个h5py文件的DICE系数（在训练中有9个文件参与训练，1个文件参与验证）
|                   |  CSF       | GM             | WM   | Average
|-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|
|3D-SkipDenseSeg  | 93.66| 90.80 | 90.65 | 91.70 |

Notes: 分布式训练需要一个RANK_TABLE_FILE，文件的删除方式可以参考该链接[Link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) ,device_ip的设置参考该链接 [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) 对于像InceptionV4这样的大模型来说, 最好导出一个外部环境变量，export HCCL_CONNECT_TIMEOUT=600，以将hccl连接检查时间从默认的120秒延长到600秒。否则，连接可能会超时，因为编译时间会随着模型大小的增加而增加。在1.3.0版本下，3D算子可能存在一些问题，您可能需要更改context.set_auto_parallel_context的部分代码:

in train.py：

```python
context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                  device_num=rank_size,
                                  gradients_mean=False)### Conv3d 需要将这一部分的gradients_mean设置为false
```

```python
bash run_distribute_train.sh ranktable.json  data/data_train_no_cut data/data_val_no_cut
```

分布式训练执行之后，可以在相关的scripts下面得到相关结果：

```python
.
└─scripts
  ├── train_parallel0
  │   ├── log.txt                      // 训练日志
  │   ├── env.txt                      // 环境日志
  │   ├── XXX.yaml                     // 训练过程中的配置
  │   ├──.
  │   ├──.
  │   ├──.
  ├── train_parallel1
  ├── .
  ├── .
  ├── train_parallel7
```

### 在modelarts上训练

如果您想要在modelarts上进行训练，您可以查阅相关model_arts的文档，按照下面所示的步骤进行训练和评价。

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2)设置网络配置文件的路径 "config_path=/The path of config in S3/"。
# (3) 在modelarts的界面上设置代码的路径 "/path/3D_DenseNet"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,模型的输出路径"Output file path" 和模型的日志路径 "Job log path"。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"。
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 设置网络配置文件的路径 "config_path=/The path of config in S3/"。
# (4) 在modelarts的界面上设置代码的路径 "/path/3D_DenseNet"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
# 注意：在多卡训练的时候，配置路径时不要有冲突的路径，云上可能会出现找不到路径的情况，一个解决办法是需要读取数据的路径加上相应前缀，例如 将配置文件中相关路径加上'/cache'前缀data_path: "/cache/data"  train_dir: "/cache/data/data_train_nocut" val_dir: "/cache/data/data_val_nocut"
```

如果在云上发现包配置不一致，可以尝试将下面代码写入到相应的.py文件中:

```python
import os
os.system('pip install --upgrade pip -y')
os.system('pip uninstall numpy -y')
os.system('pip install numpy -y'）
```

## 脚本说明

### 脚本和样例代码

```python
└─3D_DenseNet
  ├── README.md                      // 网络相关描述信息
  ├── scripts
  │   ├──run_distribute_train.sh     // Ascend分布式训练脚本
  │   ├──run_standalone_train.sh     // Ascend单卡训练脚本
  ├── src
  │   ├──common.py                   // 数据预处理函数
  │   ├──dataloader.py               // MindSpore的dataloader
  │   ├──eval_call_back.py           // MindSpore自定义call back 函数
  │   ├──loss.py                     // 损失函数
  │   ├──lr_schedule.py              // 学习率函数
  │   ├──metrics.py                  // 训练和验证函数
  │   ├──model.py                    // 模型定义
  │   ├──prepare_hdf5_cutedge.py     // 准备数据集函数
  │   ├──var_init.py                 // 网络初始化
          ├── model_utils
          │   ├──config.py                    // 参数配置.py
          │   ├──device_adapter.py            // device adapter
          │   ├──local_adapter.py             // local adapter
          │   ├──moxing_adapter.py            // moxing adapter
  ├── default_config.yaml             // 参数yaml文件
  ├── train.py                        // 训练py文件
  ├── eval.py                         // 评价py文件
  ├── test.py                         // 测试集py文件
  ├──export.py                        // 将checkpoint文件导出为 air/mindir格式 相关310脚本在开发中
```

### 脚本参数

```Python
enable_modelarts: False                                         # 当使用model_arts云上环境，将其设置为True
# Url for modelarts
data_url: ""
train_url: ""
checkpoint_url: ""
# Path for local
run_distribute: False                                           # 当你要进行分布式计算的时候，设置为True
enable_profiling: False                                         # 是否profiling
data_path: "./data"                                             # 本地数据路径
val_dir: "/data/data_val_nocut"                                 # 训练时的验证集路径
eval_dir : "/data/data_val"                                     # 评价数据集路径
test_dir : "/data/iseg-testing"                                 # 测试数据集路径
output_path: "./saved"                                          # 本地输出路径
load_path: "./checkpoint_path/"                                 # 本地加载路径
device_target: "Ascend"
checkpoint_path: "./checkpoint/"                                # checkpoint文件保存路径
checkpoint_file_path: "3D-DenseSeg-20000_36.ckpt"    # 本地checkpoint文件路径

# Training options
lr: 0.0002                                                      # 学习率
batch_size: 1                                                   # batch_size
epoch_size: 20000                                               # epoch_size
num_classes: 4                                                  # 要进行分类的类别
num_init_features: 32                                           # 模型初始化参数
save_checkpoint_steps : 5000                                    # 每隔 5000 步 保存
keep_checkpoint_max: 16                                         # 保存最大的check_point文件数量
loss_scale: 256.0                                               # 损失尺度
drop_rate: 0.2                                                  # drop out rate
```

## 训练过程

### 单P训练

```python
 bash  run_standalone_train.sh data/data_train_no_cut data/data_val_no_cut
```

上述命令执行后将在后台运行，结果都在scripts/train文件夹下，您可以通过train.log文件查看结果。训练结束后，您可在默认脚本文件夹下找到检查点文件。

```python
# grep "loss is " train.log
epoch: 1 step: 36, loss is 0.29248548
valid_dice: 0.5158226623757749
epoch: 2 step: 36, loss is 0.4542764
valid_dice: 0.577169897796093
```

### 分布式训练

```python
bash run_distribute.sh  ranktable.json data/train_no_cut data/val_no_cut
```

shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

```python
# grep "result:" train_parallel*/log
train_parallel0/log:epoch: 1 step: 4, loss is 0.8156291
train_parallel0/log:epoch: 2 step: 4, loss is 0.47224823
train_parallel1/log:epoch: 1 step: 4, loss is 0.7149776
train_parallel1/log:epoch: 2 step: 4, loss is 0.47474277
```

## 评估过程

### 评估

在Ascend环境运行时评估测试数据集
在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径

```python
bash run_eval.sh username/3D_DenseNet/3D-DenseSeg-20000_36.ckpt username/3D_DenseNet/data/data_val
```

上述python命令将在后台运行，您可以通过eval.log文件查看结果。几分钟之后，结果会以如下的表格方式呈现：
|                   |  CSF       | GM             | WM   | Average
|-------------------|:-------------------:|:---------------------:|:-----:|:--------------:|
|3D-SkipDenseSeg  | 93.66| 90.80 | 90.65 | 91.70 |

## 导出过程

### 导出

执行相应的export函数即可

```python
python export.py
```

默认会在当前代码目录下生成mindir格式文件

## 性能

### 训练性能

|           | Ascend                                         |
| ------------------- | --------------------------------------------------------- |
|模型       | 3D_DenseNet                                         |
| MindSpore| 1.3.0                                       |
| 数据集             | i-seg2017                                        |
| 训练参数 | epoch = 20000,  batch_size = 1                              |
| 优化器           | SGD                                                |
| 损失函数       | SoftmaxCrossEntropyWithLogits                         |
| 参数数量          | 466320                                             |
| Dice 系数   |91.70                                               |

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读 [昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md) 和 [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## ModelZoo 主页

请浏览官方 [主页](https://gitee.com/mindspore/models)
