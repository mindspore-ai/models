# 目录

<!-- TOC -->

- [目录](#目录)
- [VNet描述](#vnet描述)

    - [描述](#描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [GPU处理器环境运行](#GPU处理器环境运行)
        - [结果](#结果)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
            - [GPU处理器环境运行](#GPU处理器环境运行)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-2)
            - [GPU处理器环境运行](#GPU处理器环境运行)
        - [结果](#结果-1)
            - [训练准确率](#训练准确率)
    - [导出过程](#导出过程)
        - [导出](#[导出](#))
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
  - [性能](#性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Vnet描述

## 描述

VNet适用于医学图像分割，使用3D卷积，能够处理3D MR图像数据，能够端到端地分割目标。设计了独特的V型结构，借用了UNet从压缩路径叠加特征图，从而补充损失信息。损失函数使用Dice损失函数，可以平衡前景体素和背景体素之间的不平衡。

有关网络详细信息，请参阅[论文][1]`F Milletari, Navab N, Ahmadi S A. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation[C]// 2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016.`

# 模型架构

3D卷积神经网络，可以端到端地分割MRI体积：

- 整体网络结构借鉴UNet的U型结构，从压缩路径叠加特征图来补充损失信息。
- 使用3D卷积算子，并且采用ResNet的短路连接方式构建3D ResBlock。
- 损失函数使用Dice损失函数。
- 实验证明该网络可以在数据集PROMISE 2012上达到不错的分割效果。

# 数据集

数据集使用前列腺MRI分割数据集（[PROMISE 2012][2]）。PROMISE 2012数据集共包含80张MR图像，其中50张带有GT，其余30张没有GT。

- 下载数据集。
- 数据集结构

```shell
.
└──data
    ├── TrainData               # 训练数据集
    │   ├── gt
    │   │   ├── Case00_segmentation.mhd
    │   │   ├── Case00_segmentation.raw
    │   │   ...
    │   └── img
    │       ├── Case00.mhd
    │       ├── Case00.raw
    │       ...
    └── TestData                # 测试数据集
        ├── Case00.mhd
        ├── Case00.raw
        ...
```

由于测试数据集没有GT，因此在训练数据集上划分出训练子集和测试子集。训练子集：随机选取40张带有GT的MR图像；测试子集：其余10张带有GT的MR图像。

# 特性

## 混合精度

采用[混合精度][6]的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 准备Ascend/GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 生成config json文件用于多卡训练。
    - [简易教程](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)
    - 详细配置方法请参照[官网教程](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#配置分布式环境变量)。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 运行前准备

修改配置文件`src/config.py`。

```python
from easydict import EasyDict as edict
vnet_cfg = edict(
    {
        'task': 'promise12',                            #任务
        'fold': 0,                                      #第0次验证
        # data setting
        'dirResult': 'results/infer',                   #模型参数保存地址
        'dirPredictionImage': 'results/prediction',     #预测图像保存地址
        'normDir': False,
        'dstRes': [1, 1, 1.5],
        'VolSize': [128, 128, 64],
        # training setting
        'batch_size': 4,
        'epochs': 500,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.99,
        'warmup_step': 120,
        'warmup_ratio': 0.3,
    }
)

```

获取数据集划分文件，得到train.csv和val.csv

```bash
# 进入根目录
cd vnet/

# 划分数据集
# OUT_PATH: 数据集划分文件保存地址
# RANDOM: 是否随机采样，0--False，1--True
bash scripts/create_csv.sh OUT_PATH RANDOM
# 示例：bash scripts/create_csv.sh ./ 0
```

- Ascend处理器环境运行

在VNet原始论文的基础上，我们在PROMISE 2012数据集上训练以及测试，其中40张用于训练，10张用于测试。注意多卡训练过程中默认只有rank_id=0的卡保存模型。

运行以下训练脚本配置单卡训练参数：

```bash
# 进入根目录
cd vnet/

# 运行单卡训练
# DEVICE_ID: Ascend处理器的id，需用户指定
# DATA_PATH: PROMISE12训练数据集路径，包含img和gt两个文件夹
# TRAIN_SPLIT_FILE_PATH：train.csv文件路径
bash scripts/train_standalone.sh DEVICE_ID DATA_PATH TRAIN_SPLIT_FILE_PATH
```

运行一下训练脚本配置多卡训练参数：

```bash
cd vnet/scripts

# 运行8卡训练
# hccl.json: Ascend配置信息，需用户自行配置，详见官网教程
# DEVICE_NUM应与train_distribute.sh中修改device_ids的长度相同
# DATA_PATH: PROMISE12训练数据集路径，包含img和gt两个文件夹
# TRAIN_SPLIT_FILE_PATH：train.csv文件路径
bash train_distribute.sh hccl.json DEVICE_NUM DATA_PATH TRAIN_SPLIT_FILE_PATH
```

hccl.json用[此方法](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成并放在script文件夹下

评估步骤如下：

```bash
# 进入根目录
cd vnet/

# 评估VNet在PROMISE 2012数据集上的表现
# DEVICE_ID: Ascend处理器id
# CKPT_PATH: 预训练模型文件路径
# DATA_PATH：PROMISE12训练数据集路径，包含img和gt两个文件夹
# EVAL_SPLIT_FILE_PATH: val.csv文件路径
bash scripts/eval.sh DEVICE_ID CKPT_PATH DATA_PATH EVAL_SPLIT_FILE_PATH
```

- GPU处理器环境运行

```bash
# 进入根目录
cd vnet/

# 运行单卡训练
bash scripts/train_standalone_gpu.sh DEVICE_ID DATA_PATH TRAIN_SPLIT_FILE_PATH

# 运行8卡训练
# DEVICE_NUM显卡数量
# DEVICE_LIST: GPU处理器的列表，需用户指定，例如“0,1,2,3,4,5,6,7”
bash scripts/train_distribute_gpu.sh DEVICE_NUM DEVICE_ LIST DATA_PATH TRAIN_SPLIT_FILE_PATH

# 评估VNet在PROMISE 2012数据集上的表现
bash scripts/eval_gpu.sh DEVICE_ID CKPT_PATH DATA_PATH EVAL_SPLIT_FILE_PATH
```

# 脚本说明

## 脚本及样例代码

```shell
.
├── scripts
│   ├── eval.sh                                     # Ascend测试脚本
│   ├── train_distribute.sh                         # Ascend多卡并行训练脚本
│   ├── train_standalone.sh                         # Ascend单卡训练脚本
│   ├── eval_gpu.sh                                 # GPU测试脚本
│   ├── train_distribut_gpu.sh                      # GPU多卡并行训练脚本
│   └── train_standalone_gpu.sh                     # GPU单卡训练脚本
├── src
│   ├── config.py                                   # 训练参数配置文件
│   ├── dataset.py                                  # 加载训练数据集
│   ├── data_manager.py                             # 加载MR图像
│   ├── vnet.py                                     # VNet模型文件
│   └── utils.py                                    # 模型功能函数
│
├── eval.py                                         # PROMISE 2012数据集测试脚本
├── train.py                                        # 训练脚本
└── README_CN.md
```

## 脚本参数

默认训练配置

```bash
'dstRes': [1, 1, 1.5],                              # 体素间距
'VolSize': [128, 128, 64],                          # 输入图像体积
'batch_size': 4,                                    # batch size
'epochs': 500,                                      # 总训练epoch数
'lr': 0.001,                                        # 训练学习率（GPU：0.0005）
'weight_decay': 1e-4,                               # 权重衰减
'momentum': 0.99,                                   # 动量
'warmup_step': 120,                                 # warm up步数
'warmup_ratio': 0.3,                                # warm up学习率占比
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# trian_standalone.sh
python3 train.py \
          --device_target Ascend \
          --device_id "$1" \
          --data_path $2 \
          --train_split_file_path $3 > train_standalone.log 2>&1 &
```

```bash
# train_distribute.sh

# distributed devices id
device_ids=(0 1 2 3 4 5 6 7)

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=${device_ids[i]}
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp *.sh ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  python3 train.py \
    --device_target Ascend \
    --device_id $1 \
    --run_distribute 1 \
    --device_num $2 \
    --data_path $3 \
    --train_split_file_path $4 > train_distribute.log 2>&1 &
  cd ..
done
```

#### GPU处理器环境运行

```bash
python3 train.py \
          --device_target GPU \
          --device_id "$1" \
          --data_path $2 \
          --train_split_file_path $3 > train_standalone_gpu.log 2>&1 &
```

```bash
# train_distribute_gpu.sh
mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python3 train.py --device_target GPU --run_distribute 1 --device_num $1 \
--data_path $3 --train_split_file_path $4  > train_distribute_gpu.log 2>&1 &
```

### 结果

#### Ascend处理器环境运行

```bash
# 单卡训练结果
epoch: 1 step: 10, loss is 0.8907114
epoch time: 195045.276 ms, per step time: 19504.528 ms
epoch: 2 step: 10, loss is 0.8099437
epoch time: 24754.049 ms, per step time: 2475.405 ms
epoch: 3 step: 10, loss is 0.8360894
epoch time: 24741.982 ms, per step time: 2474.198 ms
epoch: 4 step: 10, loss is 0.8259954
epoch time: 23744.493 ms, per step time: 2374.449 ms
epoch: 5 step: 10, loss is 0.8108934
epoch time: 24054.200 ms, per step time: 2405.420 ms
epoch: 6 step: 10, loss is 0.7308113
epoch time: 23846.928 ms, per step time: 2384.693 ms
...
```

```bash
# 分布式训练结果（2P）
epoch: 1 step: 5, loss is 0.9019523
epoch time: 237280.151 ms, per step time: 47456.030 ms
epoch: 2 step: 5, loss is 0.87858737
epoch time: 15955.368 ms, per step time: 3191.074 ms
epoch: 3 step: 5, loss is 0.6631024
epoch time: 16681.054 ms, per step time: 3336.211 ms
epoch: 4 step: 5, loss is 0.8407575
epoch time: 15848.884 ms, per step time: 3169.777 ms
epoch: 5 step: 5, loss is 0.63558793
epoch time: 15644.324 ms, per step time: 3128.865 ms
epoch: 6 step: 5, loss is 0.81737673
epoch time: 15364.686 ms, per step time: 3072.937 ms
...
```

#### GPU处理器环境运行

```bash
# 单卡训练结果
epoch: 1 step: 10, loss is 0.89366055
epoch time: 44482.961 ms, per step time: 4448.296 ms
epoch: 2 step: 10, loss is 0.8281902
epoch time: 29566.847 ms, per step time: 2956.685 ms
epoch: 3 step: 10, loss is 0.85853046
epoch time: 29729.315 ms, per step time: 2972.932 ms
epoch: 4 step: 10, loss is 0.85585916
epoch time: 29955.715 ms, per step time: 2995.572 ms
epoch: 5 step: 10, loss is 0.8431523
epoch time: 30154.752 ms, per step time: 3015.475 ms
epoch: 6 step: 10, loss is 0.7514138
epoch time: 29824.363 ms, per step time: 2982.436 ms
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 进入根目录
cd vnet/

# 评估VNet在PROMISE 2012数据集上的表现
bash scripts/eval.sh DEVICE_ID CKPT_PATH DATA_PATH EVAL_SPLIT_FILE_PATH
```

测试脚本示例如下：

```bash
# eval.sh
# ${DEVICE_ID}: Ascend处理器id
# eval.log：保存的测试结果
python3 eval.py \
          --device_target Ascend \
          --dev_id "${DEVICE_ID}" \
          --ckpt_path $2\
          --data_path $3\
          --eval_split_file_path $4 > eval.log 2>&1 &
```

#### GPU处理器环境运行

```bash
# 进入根目录
cd vnet/

# 评估VNet在PROMISE 2012数据集上的表现
bash scripts/eval_gpu.sh DEVICE_ID CKPT_PATH DATA_PATH EVAL_SPLIT_FILE_PATH
```

测试脚本示例如下：

```bash
# eval_gpu.sh
# ${DEVICE_ID}: GPU处理器id
# eval.log：保存的测试结果
python3 eval.py \
          --device_target GPU \
          --dev_id "${DEVICE_ID}" \
          --ckpt_path $2\
          --data_path $3\
          --eval_split_file_path $4 > eval_gpu.log 2>&1 &
```

### 结果

运行适用的训练脚本获取结果。要获得相同的结果，请按照快速入门中的步骤操作。

#### 训练准确率

> 注：该部分展示的是Ascend/GPU单卡训练结果。

- 在PROMISE 2012上的评估结果

| **网络** | Avg. Dice | Avg. Hausdorff distance |
| :----------: | :-----: | :----: |
| Vnet(MindSpore_Ascend版本) | 85.60% | 9.61 |
| Vnet(MindSpore_GPU版本) | 85.74% | 9.46 |

## 导出过程

### 导出

将保存的网络模型导出为MINDIR模型

```bash
# 进入根目录
cd vnet/

# 修改CKPT_PATH路径
python export.py --ckpt_file CKPT_PATH --file_format MINDIR
```

## 推理过程

### 推理

在执行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，MINDIR可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用PROMISE 2012数据集进行推理

```bash
# 进入根目录
cd vnet/scripts

# MINDIR_PATH：已经导出的MINDIR模型文件路径
# DATA_PATH：PROMISE12训练数据集路径，包含img和gt两个文件夹
# SPLIT_FILE_PATH: val.csv文件路径
# DEVICE_ID: 310处理器ID，可选
bash run_infer_310.sh MINDIR_PATH DATA_PATH SPLIT_FILE_PATH DEVICE_ID
```

# 模型描述

## 性能

### 评估性能

| 参数 | Ascend 910| GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| 模型版本 | VNet | VNet |
| 资源 | Ascend 910 | Tesla V100-PCIE , cpu 2.60GHz 52cores, RAM 754G |
| 上传日期 | 2021-09-22 | 2021-10-25 |
| MindSpore版本 | 1.2.0 | 1.6.0.20211118 |
| 数据集 | PROMISE 2012 | PROMISE 2012 |
| 训练参数 | epoch = 500, batch_size = 4, lr = 0.001 | epoch = 500, batch_size = 4, lr = 0.0005 |
| 优化器 | Adam | Adam |
| 损失函数 | Dice损失函数 | Dice损失函数 |
| 输出 | 预测体积 | 预测体积 |
| 损失 | 0.036 | 0.036 |
| 性能 | 1080ms/step（单卡）;1017ms/step（八卡） | 1870ms/step（单卡）;2170ms/step（八卡） |
| 总时长 | 1.5h（单卡）;15m（八卡） | 2.6h（单卡）;22m（八卡） |
| 脚本 | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/vnet) | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/vnet) |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

[1]: https://arxiv.org/abs/1606.04797
[2]: https://promise12.grand-challenge.org/Download/

