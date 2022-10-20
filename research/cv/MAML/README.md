# 目录

<!-- TOC -->

- [目录](#目录)
- [MAML描述](#MAML描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MAML描述

[论文](https://arxiv.org/pdf/1703.03400.pdf) ： 《Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks》

# 数据集

使用的数据集：

- [Omniglot](<https://github.com/brendenlake/omniglot>) 包含1623个类，每个类有20个训练数据。每个图像的大小是105x105像素。通常用于小样本学习任务。

数据集获取及预处理过程：
    1.新建文件夹omniglot,在omniglot下新建两个子文件夹raw和processed
    2.点击链接[link1](<https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip>)下载images_background.zip, 点击链接[link2](<https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip>)下载images_evaluation.zip, 将这两个文件放到raw文件夹下
    3.将步骤2的两个文件夹解压到processed文件夹下
    4.运行src下的data_preprocess.py文件(python data_preprocess.py --data_path='your/path/omniglot/')
    5.运行完成可以看到omniglot文件夹下生成了omniglot.npy文件，结束。

omniglot数据集的文件目录结构如下所示：

```text
├── omniglot
    ├── processed
        ├─ images_background
            ├─ Alphabet_of_the_Magi
                ├─ character01
                    ├─ 0709_01.png
                    ├─ ...
                    └─ 0709_20.png
                ├─ ...
                └─ character20
            ├─ ...
            └─ Tifinagh
                ├─ character01
                    ├─ 0910_01.png
                    ├─ ...
                    └─ 0910_20.png
                ├─ ...
                └─ character55
        ├─ images_evaluation
            ├─ Angelic
                ├─ character01
                    ├─ 0965_01.png
                    ├─ ...
                    └─ 0965_20.png
                ├─ ...
                └─ character20
            ├─ ...
            └─ ULOG
                ├─ character01
                    ├─ 1598_01.png
                    ├─ ...
                    └─ 1598_20.png
                ├─ ...
                └─ character26
    ├──raw
        ├── images_background.zip
        └─ images_evaluation.zip
    └─ omniglot.npy
```

# 代码目录结构

```text
├── MAML
    ├── script
        ├── run_eval_ascend.sh
        ├── run_eval_gpu.sh
        ├── run_standalone_train_ascend.sh
        └─ run_standalone_train_gpu.sh
    ├── src
        ├── crossentropy.py
        ├── data_preprocess.py
        ├── KaimingNormal.py
        ├── learner.py
        ├── meta.py
        ├── Omniglot.py
        └─ Omniglotlter.py
    ├── eval.py
    ├── omniglot_train.py
    └─ README.md
```

# 环境要求

- 硬件（GPU）
    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

  ```bash
  # 单机训练
  用法：bash run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH] [DATA_PATH]

  # 运行评估示例
  用法：bash run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
  ```

- Ascend处理器环境运行

  ```bash
  # 单机训练
  用法：bash run_standalone_train_ascend.sh [DEVICE_ID] [EPOCH] [DATA_PATH]

  # 运行评估示例
  用法：bash run_eval_ascend.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
  ```

# 脚本说明

## 脚本参数

在omiglot_train.py中配置训练参数

  ```python
  'device_id': 0                                 # GPU ID
  'device_target': "GPU"                         # 运行设备
  'mode': "graph"                                # 运行模式（graph or pynativate）
  'epoch': 20000                                  # 训练轮数
  'n_way': 5                                     # n way
  'k_spt': 1                                     # k shot for support set
  'k_qry': 5                                     # k shot for query set
  'imgsz': 28                                    # imgsz
  'imgc': 1                                      # imgc
  'task_num': 32                                 # meta batch size, namely task num
  'meta-lr': 32                                  # meta-level outer learning rate
  'update-lr': 1e-3                              # task-level inner update learning rate
  'update_step': 5                               # task-level inner update steps
  'update_step_test': 10                         # update steps for finetunning
  'lr_scheduler_gamma': 0.5                      # lr_scheduler_gamma
  'output_dir': './ckpt_outputs'                 # checkpoint文件保存的绝对全路径
  'ckpt': ''                                     # 模型参数路径(默认为'')
  'data_path': "your/path/omniglot"              # 数据集路径
  ```

在test.py中配置训练参数

  ```python
  'device_id': 0                                 # GPU ID
  'mode': "graph"                                # 运行模式（graph or pynativate）
  'epoch': 1                                     # 轮数
  'n_way': 5                                     # n way
  'k_spt': 1                                     # k shot for support set
  'k_qry': 5                                     # k shot for query set
  'imgsz': 28                                    # imgsz
  'imgc': 1                                      # imgc
  'task_num': 32                                 # meta batch size, namely task num
  'meta-lr': 32                                  # meta-level outer learning rate
  'update-lr': 1e-3                              # task-level inner update learning rate
  'update_step': 5                               # task-level inner update steps
  'update_step_test': 10                         # update steps for finetunning
  'lr_scheduler_gamma': 0.5                      # lr_scheduler_gamma
  'ckpt': '/your/path/maml.ckpt'                 # 模型参数路径
  'data_path': "your/path/omniglot"              # 数据集路径
  ```

## 训练过程

### 训练

- GPU处理器环境运行

    ```bash
    # 单机训练
    用法：bash run_standalone_train_gpu.sh [DEVICE_ID] [EPOCH] [DATA_PATH]

    # 运行评估示例
    用法：bash run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
    ```

- Ascend处理器环境运行

    ```bash
    # 单机训练
    用法：bash run_standalone_train_ascend.sh [DEVICE_ID] [EPOCH] [DATA_PATH]

    # 运行评估示例
    用法：bash run_eval_ascend.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
    ```

### 结果

```text
# 单卡训练结果（1P）
epoch: 1 step: 1, loss is 1.2703509330749512
epoch time: 1833839.449 ms, per step time: 1833839.449 ms
Tensor(shape=[1], dtype=Float32, value=[ 5.77499986e-01])
epoch: 2 step: 1, loss is 1.2066479921340942
epoch time: 499.590 ms, per step time: 499.590 ms
Tensor(shape=[1], dtype=Float32, value=[ 6.31250024e-01])
epoch: 3 step: 1, loss is 1.0877124071121216
epoch time: 500.080 ms, per step time: 500.080 ms
Tensor(shape=[1], dtype=Float32, value=[ 6.49999976e-01])
```

## 评估过程

### 评估

- GPU处理器环境运行

```bash
# 评估
用法：bash run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
```

- Ascend处理器环境运行

```bash
# 评估
用法：bash run_eval_ascend.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
```

### 结果

- GPU处理器环境运行结果

```bash
[0.9608333]
[0.95458335]
[0.94416666]
[0.945]
[0.93875]
[0.9483333]
[0.95166665]
[0.95125]
[0.9433333]
[0.94458336]
[0.95]
[0.9475]
[0.935]
[0.93875]
[0.9583333]
[0.9533333]
[0.94208336]
[0.9483333]
[0.94083333]
[0.96375]
[0.95458335]
[0.94708335]
[0.94125]
[0.95208335]
[0.965]
[0.94458336]
[0.94416666]
[0.9525]
[0.96125]
[0.9479167]
[0.94902784]
```

# 模型描述

## 性能

### GPU训练性能

|           | GPU                                         |
| ------------------- | --------------------------------------------------------- |
|模型       | MAML                                         |
| MindSpore| 1.7.0                                       |
| 数据集             | omniglot                                        |
| 训练参数 | epoch = 20000,batch_size=32                              |
| 优化器           | Adam                                                |
| 损失函数       | SoftmaxCrossEntropyWithLogits                         |
| 训练速度          | 530ms/step                                             |

### Ascend训练性能

|           | Ascend                                         |
| ------------------- | --------------------------------------------------------- |
|模型       | MAML                                         |
| MindSpore| 1.7.0                                       |
| 数据集             | omniglot                                        |
| 训练参数 | epoch = 20000,batch_size=32                              |
| 优化器           | Adam                                                |
| 损失函数       | SoftmaxCrossEntropyWithLogits                         |
| 训练速度          | 155ms/step                                             |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
