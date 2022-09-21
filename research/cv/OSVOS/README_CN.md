# 目录

<!-- TOC -->

- [目录](#目录)
- [OSVOS描述](#osvos描述)
- [描述](#描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练结果](#训练结果)
        - [GPU处理器环境运行](#GPU处理器环境运行)
    - [评估结果](#评估结果)
        - [训练准确率](#训练准确率)
- [模型描述](#模型描述)
  - [性能](#性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# OSVOS描述

## 描述

OSVOS网络主要处理半监督视频物体分割问题，即给定视频第一帧中物体的掩膜，在后续视频帧中将物体从视频背景中分离出来。OSVOS使用VGG16网络作为骨干网络并有效利用了其预训练模型，整个训练过程分为两个阶段，首先在DAVIS数据集上训练parent 网络，然后针对每一个视频序列进行在线训练。

有关网络详细信息，请参阅[论文](https://arxiv.org/pdf/1611.05198.pdf)`Caelles S, Maninis K K, Pont-Tuset J, et al. One-shot video object segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.`

# 模型架构

One-shot视频物体分割网络，解决半监督视频物体分割任务：

- 网络结构使用VGG16作为骨干网络，使用其在ImageNet数据集上的预训练模型。
- 分为两阶段训练，首先在DAVIS上训练parent网络，然后对视频序列在线训练。
- 损失函数使用BCE损失函数。
- 实验证明OSVOS在数据集DAVIS上达到SOTA的分割精度。

# 数据集

数据集使用视频物体分割数据集（[DAVIS 2016](https://davischallenge.org/)）。DAVIS 2016数据集包含30个训练集，20个验证集。

- 下载数据集。
- 数据集结构

```text
.
└──DAVIS
    ├── Annotations               # 标签文件夹
    │   ├── 480p
    │   │   ├── bear
    │   │   ├── blackswan
    │   │   ...
    │   └── 1080p
    │       ├── bear
    │       ├── blackswan
    │       ...
    ├── ImageSets                 # 训练以及测试序列
    │   ├── 480p
    │   │   ├── train.txt
    │   │   ├── val.txt
    │   │   ...
    │   └── 1080p
    │       ├── train.txt
    │       ├── val.txt
    │       ...
    └── JPEGImages                # 图片文件夹
        ├── 480p
        │   ├── bear
        │   ├── blackswan
        │   ...
        └── 1080p
            ├── bear
            ├── blackswan
            ...
```

# 环境要求

- 硬件（GPU）
    - 准备GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 运行前准备

修改配置文件`src/config.py`。

```text
from easydict import EasyDict as edict
osvos_cfg = edict(
    {
        'task': 'OSVOS',
        'dirResult': './results',          # 结果输出地址

        # train_parent setting
        'tp_batch_size': 12,
        'tp_lr': 5e-5,
        'tp_wd': 0.0002,
        'tp_epoch_size': 240,

        # train_online setting
        'to_batch_size': 1,
        'to_epoch_size': 10000,
        'to_lr': 0.000005,
        'to_wd': 0.0002,
    }
)

```

获取训练和验证序列名文件`train.txt`和`val.txt`。

```bash
# 进入根目录
cd osvos/

# OUT_PATH: 序列名文件保存地址，保存在数据集文件夹下
bash scripts/create_seq_txt.sh OUT_PATH
# 示例：bash scripts/create_seq_txt.sh ./DAVIS
```

下载[VGG16](https://download.mindspore.cn/model_zoo/r1.3/vgg16_gpu_v130_imagenet2012_official_cv_bs32_acc73.48/vgg16_gpu_v130_imagenet2012_official_cv_bs32_acc73.48.ckpt)预训练模型，获取OSVOS骨干网络预训练参数。

```bash
# 进入根目录
cd osvos/

# VGG_MODEL: 下载的VGG16模型地址
# OUT_FILE: 输出的文件名
bash scripts/convert_checkpoint.sh VGG_MODEL OUT_FILE
# 示例：bash scripts/convert_checkpoint.sh ./models/vgg16_gpu.ckpt vgg16_features.ckpt
```

- 阶段1：训练parent网络

运行以下训练脚本配置单卡训练参数：

```bash
# 进入根目录
cd osvos/

# 运行单卡训练
# DEVICE_ID: GPU处理器的id，需用户指定
# DATA_PATH: DAVIS2016数据集路径，包含Annotations、ImageSets和JPEGImages三个文件夹
# VGG_CKPT_PATH: 剪裁参数后的VGG16模型，在上一步得到
bash scripts/run_parent_standalone_train_gpu.sh DEVICE_ID DATA_PATH VGG_CKPT_PATH
# 示例：bash scripts/run_parent_standalone_train_gpu.sh 0 ./DAVIS ./vgg16_features.ckpt
```

运行一下训练脚本配置多卡训练参数：

```bash
# 进入根目录
cd osvos/

# 运行8卡训练
# DEVICE_NUM: GPU处理器数量
# VISIABLE_DEVICE: 设置可见GPU处理器id，和DEVICE_NUM对应
# DATA_PATH: DAVIS2016数据集路径，包含Annotations、ImageSets和JPEGImages三个文件夹
# VGG_CKPT_PATH: 剪裁参数后的VGG16模型
bash scripts/run_parent_distributed_train_gpu.sh DEVICE_NUM VISIABLE_DEVICE DATA_PATH VGG_CKPT_PATH
# 示例：bash scripts/run_parent_distributed_train_gpu.sh 8 0,1,2,3,4,5,6,7 ./DAVIS ./vgg16_features.ckpt
```

- 阶段2：在线训练网络

```bash
# 进入根目录
cd osvos/

# DEVICE_NUM: GPU处理器数量，要求大于4张
# SEQ_TXT: 在训练前准备中得到的val.txt
# DATA_PATH: DAVIS2016数据集路径，包含Annotations、ImageSets和JPEGImages三个文件夹
# PARENT_CKPT_PATH: parent预训练模型，在上一步得到
bash scripts/run_online_train_gpu.sh DEVICE_NUM DATA_PATH SEQ_TXT PARENT_CKPT_PATH
# 示例：bash scripts/run_online_train_gpu.sh 8 ./DAVIS ./DAVIS/val.txt \
# ./results/parent/1/checkpoint_parent.ckpt
```

- 评估网络训练性能

```bash
# 进入根目录
cd osvos/

# DEVICE_ID: GPU处理器id
# SEQ_TXT: 在训练前准备中得到的val.txt
# DATA_PATH：DAVIS2016数据集路径，包含Annotations、ImageSets和JPEGImages三个文件夹
# CKPT_PATH: 在线训练模型文件路径，包含多个序列文件夹
# PREDICTION_PATH: 模型预测结果路径
bash scripts/run_eval_gpu.sh [DEVICE_ID] [SEQ_TXT] [DATA_PATH] [CKPT_PATH] [PREDICTION_PATH]
# 示例: bash scripts/run_eval_gpu.sh 0 ./DAVIS/val.txt ./DAVIS ./results/online ./results/images
```

# 脚本说明

## 脚本及样例代码

```text
.
├── scripts
│   ├── convert_checkpoint.sh                    # 剪裁VGG16预训练模型脚本
│   ├── create_seq_txt.sh                        # 获取训练验证序列名文件脚本
│   ├── run_eval_gpu.sh                          # 测试脚本
│   ├── run_online_train_gpu.sh                  # 在线训练脚本
│   ├── run_parent_standalone_train_gpu.sh       # 单卡训练脚本
│   └── run_parent_distributed_train_gpu.sh      # 多卡训练脚本
├── src
│   ├── config.py                                # 训练参数配置文件
│   ├── convert_checkpoint.py                    # 剪裁VGG16预训练模型
│   ├── create_seq_txt.py                        # 获取训练验证序列名文件
│   ├── dataset.py                               # 加载数据
│   ├── utils.py                                 # 模型功能函数
│   └── vgg_osvos.py                             # OSVOS网络模型
│
├── eval.py                                      # 测试代码
├── evaluation_davis.py                          # DAVIS验证代码
├── train.py                                     # 训练代码
├── requirements.txt
└── README_CN.md
```

## 脚本参数

默认训练配置

```text
'tp_batch_size': 12,                              # parent网络batch size
'tp_epoch_size': 240,                             # parent网络epoch size
'tp_lr': 5e-5,                                    # parent网络学习率
'tp_wd': 0.0002,                                  # parent网络权重衰减
'to_batch_size': 1,                               # online网络batch size
'to_epoch_size': 10000,                           # online网络epoch size
'to_lr': 0.000005,                                # online网络学习率
'to_wd': 0.0002,                                  # online网络权重衰减
```

## 训练结果

### GPU处理器环境运行

- 阶段1：训练parent网络

```text
# 单卡训练结果
epoch: 1 step: 173, loss is 0.11066732
epoch time: 120480.930 ms, per step time: 696.422 ms
epoch: 2 step: 173, loss is 0.08072116
epoch time: 120437.981 ms, per step time: 696.173 ms
epoch: 3 step: 173, loss is 0.095244825
epoch time: 120493.963 ms, per step time: 696.497 ms
epoch: 4 step: 173, loss is 0.09305578
epoch time: 120482.885 ms, per step time: 696.433 ms
epoch: 5 step: 173, loss is 0.07420572
epoch time: 120375.160 ms, per step time: 695.810 ms
epoch: 6 step: 173, loss is 0.07288652
epoch time: 120419.507 ms, per step time: 696.067 ms
...
```

```text
# 多卡训练结果
epoch: 1 step: 21, loss is 0.42234254
epoch time: 15167.245 ms, per step time: 722.250 ms
epoch: 2 step: 21, loss is 0.20202039
epoch time: 15167.284 ms, per step time: 722.252 ms
epoch: 3 step: 21, loss is 0.22595137
epoch time: 15166.857 ms, per step time: 722.231 ms
epoch: 4 step: 21, loss is 0.29434553
epoch time: 15167.685 ms, per step time: 722.223 ms
epoch: 5 step: 21, loss is 0.17858279
epoch time: 15205.949 ms, per step time: 724.093 ms
epoch: 6 step: 21, loss is 0.14984499
epoch time: 15206.557 ms, per step time: 724.122 ms
...
```

- 阶段2：在线训练网络

```text
# 单卡训练结果，示例序列为blackswan
epoch: 1 step: 1, loss is 2.1926537
epoch time: 530.483 ms, per step time: 530.483 ms
epoch: 2 step: 1, loss is 2.116636
epoch time: 239.443 ms, per step time: 239.443 ms
epoch: 3 step: 1, loss is 2.1444561
epoch time: 205.402 ms, per step time: 205.402 ms
epoch: 4 step: 1, loss is 1.9663961
epoch time: 199.764 ms, per step time: 199.764 ms
epoch: 5 step: 1, loss is 2.0743327
epoch time: 244.921 ms, per step time: 244.921 ms
epoch: 6 step: 1, loss is 1.3092588
epoch time: 240.559 ms, per step time: 240.559 ms
...
```

## 评估结果

### 训练准确率

> 注：该部分展示的是GPU8卡训练结果。

- 在DAVIS 2016上的评估结果

| **网络** | Avg. jaccard | Avg. f1 |
| :----------: | :-----: | :----: |
| OSVOS(MindSpore_GPU版本) | 80.30% | 84.71% |

## 性能

### 评估性能

| 参数 | GPU |
| -------------------------- | -------------------------------------- |
| 模型版本 | OSVOS |
| 资源 | Tesla V100-PCIE , cpu 2.60GHz 52cores, RAM 754G |
| 上传日期 | 2022-5-7 |
| MindSpore版本 | 1.6.0.20211118 |
| 数据集 | DAVIS 2016 |
| 训练参数 | tp_batch_size=12, tp_lr=5e-5, tp_epoch_size=240, to_batch_size=1, to_epoch_size=10000 |
| 优化器 | Adam |
| 损失函数 | BCE损失函数 |
| 输出 | 视频序列分割掩膜 |
| 性能 | 695ms/step（单卡）;722ms/step（八卡） |
| 总时长 | 8.1h（单卡）;1h（八卡） |
| 脚本 | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/OSVOS) |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。



