# 目录

<!-- TOC -->

- [目录](#目录)
- [GAN描述](#GAN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [MNIST上的GAN](#MNIST上的GAN)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GAN描述

GAN是2014年提出的，是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。模型通过框架中两个模块：生成模型（Generative Model）和判别模型（Discriminative Model）的互相博弈学习产生相当好的输出。

[论文](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)：Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[J]. Advances in neural information processing systems, 2014, 27.

# 模型架构

一个生成器和一个判别器。

# 数据集

使用的数据集：[MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images  
    - Test：10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py

```text

└─data
  └─MNIST_Data
    └─t10k-images.idx3-ubyte
    └─t10k-labels.idx1-ubyte
    └─train-images.idx3-ubyte
    └─train-labels.idx1-ubyte
```

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 添加数据集路径,以训练MNIST为例
  train_data_path: data/MNIST_Data/
  test_data_path: data/MNIST_Data/

  ```

```python
# 单卡训练
bash ./scripts/run_standalone_train.sh [DEVICE_ID]

# Ascend多卡训练
bash ./scripts/run_distributed_train.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START]
```

示例：

  ```python
  # 单卡训练
  bash ./scripts/run_standalone_train.sh 0

  # Ascend多卡训练（8P）
  bash ./scripts/run_distributed_train.sh ./rank_table_8pcs.json 8 0
  ```

- 评估：

```python
# 评估
bash ./scripts/run_eval.sh [DEVICE_ID]
```

示例：

  ```python
  # 评估
  bash ./scripts/run_eval.sh 0
  ```

# 脚本说明

## 脚本及样例代码

```bash
├──gan
    ├── README_CN.md           # README
    ├── requirements.txt       # required modules
    ├─scripts                  # shell script
        ├─run_standalone_train.sh            # training in standalone mode
        ├─run_distributed_train.sh    # training in parallel mode
        ├─export.sh            # export checkpoints into mindir model
        └─run_eval.sh          # evaluation
    ├─ src
        ├─loss.py              # loss function
        ├─gan.py               # define the construction of gan
        ├─param_parse.py       # parameter parser
        └─dataset.py           # dataset create
    ├── train.py               # train model
    ├── export.py              # export checkpoints into mindir model
    ├── eval.py                # test model
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  模型检查点保存在当前目录下。

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估MNIST数据集

  ```bash
  python eval.py > eval.log 2>&1 &

  ```

# 模型描述

## 性能

### 评估性能

#### MNIST上的GAN

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型名称              | GAN                                                         |
| 资源                  | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8      |
| 上传日期              | 2021-09-28                                                  |
| MindSpore版本         | 1.3.0                                                       |
| 数据集                | MNIST                                                       |
| 训练参数              | epoch=200, batch_size = 64, lr=0.001                         |
| 优化器                | Adam                                                        |
| 损失函数              | 自定义损失函数                                                |
| 检查点                | 8914KB, .ckpt文件                                            |
| 脚本 | [gan脚本](https://gitee.com/mindspore/models/tree/master/research/cv/gan)     |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
