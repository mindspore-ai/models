# 目录

<!-- TOC -->

- [目录](#目录)
- [DGCNet描述](#dgcnet描述)
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
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [cityscapes上的DGCNet](#cityscapes上的dgcnet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DGCNet描述

DGCNet是2019年提出的语义分割网络，其骨干网络使用了ResNet模型，提高了语义分割的健壮性和运行速率。

[论文](https://arxiv.org/abs/1909.06121v3)：Zhang, L. , Li, X. , Arnab, A. , Yang, K. , Tong, Y. , and Torr, P. "*Dual graph convolutional network for semantic segmentation.*", Proceedings of the 30th British Machine Vision Conference (BMVC), 2019.

# 模型架构

以ResNet-101为骨干，利用双图卷积网络同时对空间域和通道域建模以获取全局上下文信息。

# 数据集

使用前请先下载数据并按照要求存放。

下载数据集：[cityscapes](https://www.cityscapes-dataset.com)

下载数据集文件列表：[cityscapes_datalist](https://github.com/lxtGH/GALD-DGCNet/tree/master/data)

- 数据集大小：11G，共19个目标类、5000张2048*1024彩色图像
    - 训练集：共2975张图像
    - 验证集：共500张图像
    - 测试集：共1525张图像
- 数据格式：png文件
    - 注：数据将在src/cityscapes中处理。

数据集按照如下目录格式放置

```bash
├── datasets
    ├── cityscapes
        ├── gtFine
            ├── test
            ├── train
            ├── val
        ├── leftImg8bit
            ├── test
            ├── train
            ├── val
```

数据集文件列表按照如下目录格式放置

```bash
├── datalist
    ├── cityscapes
        ├── test.txt
        ├── train++.txt
        ├── train+.txt
        ├── train.txt
        ├── trainval.txt
        ├── val.txt
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 安装requirements.txt中的python包。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

    - DGCNet 以 ResNet - 101 为骨干网络， 需要加载 ResNet - 101 预训练权重并在此基础上进行训练。我们提供了源码预训练权重转换的脚本，请修改脚本中的参数并按照如下命令开始模型转换。
    pth预训练模型文件获取路径如下：[预训练模型pth](https://drive.google.com/file/d/1JlERBWT8fHvf-uD36k5-LRZ5taqUbraj/view)

    ```bash
    # 运行模型转换脚本
    bash ./scripts/run_pth2ckpt.sh /path/pth /path/containing/ckpt
    ```

    - 在GPU处理器环境运行，请修改脚本中的参数并按照如下命令开始训练及评估。

    ```python
    # 运行单卡训练示例
    bash ./scripts/run_standalone_gpu_train.sh /path/dataset /path/datalist /path/ckpt

    # 运行分布式2卡训练示例
    bash ./scripts/run_distributed_gpu_train2p.sh /path/dataset /path/datalist /path/ckpt

    # 运行分布式8卡训练示例
    bash ./scripts/run_distributed_gpu_train8p.sh /path/dataset /path/datalist /path/ckpt

    # 运行评估示例
    bash ./scripts/run_standalone_gpu_eval.sh /path/dataset /path/datalist /path/ckpt
    ```

# 脚本说明

## 脚本及样例代码

```bash
├── model
    ├── README.md                                 // 所有模型相关说明
    ├── dgcnet_res101
        ├── README.md                             // dgcnet相关说明
        ├── scripts
            ├──run_distributed_gpu_train2p.sh       // GPU处理器分布式训练的shell脚本
            ├──run_distributed_gpu_train8p.sh       // GPU处理器分布式训练的shell脚本
            ├──run_pth2ckpt.sh                     // pth模型转换为ckpt模型的shell脚本
            ├──run_standalone_gpu_eval.sh         // GPU处理器评估的shell脚本
            ├──run_standalone_gpu_train.sh        // GPU处理器单卡训练的shell脚本
        ├── src
            ├──cityscapes.py                      // 数据预处理
            ├──DualGCNNet.py                      // dgcnet网络架构
            ├──loss.py                            // dgcnet定义的损失函数
            ├──pth2ckpt.py                        // 预训练模型转换
        ├── train.py                              // 训练脚本
        ├── eval.py                               // 评估脚本
        ├── requirements.txt                      // 安装依赖列表
```

## 脚本参数

脚本包括模型转换脚本、训练脚本以及评估脚本，请仔细阅读以下脚本参数介绍并正确配置。

- 配置模型转换脚本。

  ```python
  'load_path': "PATH/TO/PTH" \                    # pth模型的绝对全路径
  'save_path': "PATH/CONTAINING/CKPT/" \          # 转换后的ckpt模型的存放目录
  ```

- 配置训练脚本。

  ```python
  'dataset': cityscapes \                         # 数据集名称
  'data_dir': "PATH/TO/Cityscapes" \              # 数据集的绝对全路径
  'data_list': "PATH/TO/FileList" \               # 文件列表的绝对全路径
  'restore_from': "PATH/TO/CKPT" \                # 转换后预训练权重的绝对全路径
  'input_size': 832 \                             # 输入图片大小
  'batch_size': 1 \                               # 训练批次大小
  'rgb': 1 \                                      # 是否使用RGB
  'learning_rate': 0.01 \                         # 初始学习率
  'num_steps': 60000 \                            # 训练步数
  'save_dir': "./dgc_832" \                       # 输出存储路径
  'run_distribute': 1 \                           # 是否使用分布式训练
  'ohem_thres': 0.7 \                             # 损失的边界参数
  'ohem_keep': 100000 \                           # 损失的保持值参数
  ```

- 配置评估脚本。

  ```python
  'dataset': cityscapes \                         # 数据集名称
  'data_dir': "PATH/TO/Cityscapes" \              # 数据集的绝对全路径
  'data_list': "PATH/TO/FileList" \               # 文件列表的绝对全路径
  'restore_from': "PATH/TO/CKPT" \                # 待评估模型的绝对全路径
  'output_dir': "./eval_dual_seg_r101_832" \      # 评估结果的输出路径
  ```

## 预训练模型

可以使用模型转换脚本将预训练的pth文件转换为ckpt文件。
pth预训练模型文件获取路径如下：[预训练模型pth](https://drive.google.com/file/d/1JlERBWT8fHvf-uD36k5-LRZ5taqUbraj/view)
已转换的ckpt预训练模型文件获取路径如下：[预训练模型ckpt](https://drive.google.com/file/d/1WSWtgUKp_pCGczrNbxSES97Hj84Qgg1X/view?usp=sharing)

```bash
# 运行模型转换脚本
bash ./scripts/run_pth2ckpt.sh /path/pth /path/containing/ckpt
```

## 训练过程

### 训练

- GPU处理器环境运行

  ```bash
  bash ./scripts/run_standalone_gpu_train.sh /path/dataset /path/datalist /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过train_1p.log文件查看训练日志。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train_1p.log
  iter = 0 of 62475 completed,loss = 4.1163725150403155, lr=0.01
  iter = 1 of 62475 completed,loss = 4.073230450549691, lr=0.009999855
  iter = 2 of 62475 completed,loss = 3.9937671950353637, lr=0.009999855
  iter = 3 of 62475 completed,loss = 3.8110699114060145, lr=0.009999712
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- GPU处理器环境运行

  ```bash
  bash ./scripts/run_distributed_gpu_train8p.sh /path/dataset /path/datalist /path/ckpt
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_8p.log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train_8p.log
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.162871224016667, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.1193725150403155, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.074177762226125, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.134421219104891, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.164086284604172, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.117778927463441, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.095111136931157, lr=0.01
  [train.py, 215] iter = 0 of 62475 completed,loss = 4.167529094804941, lr=0.01
  [train.py, 215] iter = 1 of 62475 completed,loss = 4.063230450549691, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 4.030281865043261, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 4.079729922532756, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 4.03640633604002, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 3.9799675946740436, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 3.9790950554291715, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 4.0361444627218, lr=0.009999855
  [train.py, 215] iter = 1 of 62475 completed,loss = 3.9837671950353637, lr=0.009999855
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.796128534819948, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.759911706461043, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.7537016285931655, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.7251990701242006, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.8199783570984582, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.8110699114060145, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.889293317068125, lr=0.009999712
  [train.py, 215] iter = 2 of 62475 completed,loss = 3.8515068722479695, lr=0.009999712
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.5477280859383775, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.5086564002228204, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.4115621999152874, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.74630097046128, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.559092365252866, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.8657682306644174, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.319122442720146, lr=0.009999568
  [train.py, 215] iter = 3 of 62475 completed,loss = 3.2635988000052016, lr=0.009999568
  ...
  ...
  ```

## 评估过程

### 评估

- 在GPU环境运行时评估Cityscapes数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径。

  ```bash
  bash ./scripts/run_standalone_gpu_eval.sh /path/dataset /path/datalist /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过输出目录下的result.txt文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "mIOU:" ./eval_dual_seg_r101_832/result/result.txt
  {"meanIU": 0.8029964223555024, "IU_array": [0.9830370983083244, 0.8603258979926158, 0.9325856768475076,
  0.6019455834787047, 0.6303812416242722, 0.680111004469171, 0.74290688232998, 0.819073882691201,
  0.9271561802311508, 0.6502855618654508, 0.9491181287222692, 0.8402684074305187, 0.6666261949798068,
  0.9574500239711363, 0.8325262043865758, 0.8887043536308898, 0.8061762616951351, 0.6935363732230393,
  0.7947170668767967]}
  ```

## 推理过程

### 推理

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径。

  ```bash
  bash ./scripts/run_standalone_gpu_eval.sh /path/dataset /path/datalist /path/ckpt
  ```

  上述python命令运行结束后，您可以在输出文件查看结果。

# 模型描述

## 性能

### 评估性能

#### Cityscapes上的DGCNet

| 参数                       | GPU                   |
| ------------------------- | ---------------------- |
| 模型版本                   | DGCNet ResNet101           |
| 资源                       | Tesla V100-PICE-32G       |
| 上传日期                   | 2022-03-03  |
| MindSpore版本              | 1.5.0     |
| 数据集                    | Cityscapes     |
| 训练参数                  | steps=60000, batch_size = 1, lr=0.01      |
| 优化器                    | SGD                  |
| 损失函数                  | OhemSoftmax交叉熵        |
| 输出                      | 概率                                   |
| 损失                       | 0.49625                              |
| 速度                      | 单卡：1023毫秒/步；8卡：940毫秒/步    |
| 总时长                    | 单卡：130.5小时；8卡：17.6小时         |
| 参数                      | 617                                     |
| 微调检查点                 | 830.29 MB [(.ckpt文件)](https://drive.google.com/file/d/1DFvUZWbkVxjd9eb4DeFMb2GhcvJiUcon/view?usp=sharing)                                |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
