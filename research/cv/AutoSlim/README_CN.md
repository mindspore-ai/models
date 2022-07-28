# 目录

<!-- TOC -->

- [目录](#目录)
- [Autoslim描述](#autoslim描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [310推理过程](#310推理过程)
        - [310推理](#310推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Autoslim描述

Autoslim基于文章《Universally Slimmable Networks and Improved Training Techniques》将universally slimmable networks 进一步推广到了神经网络架构搜索（NAS）领域，通过搜索代理挑选出在性能评估标准下的最优模型。

[论文](https://arxiv.org/abs/1903.11728v1) ：Jiahui Yu, Thomas Huang."AutoSlim: Towards One-Shot Architecture Search for Channel Numbers".2019.

# 数据集

使用的数据集：Imagenet2012, [下载地址](https://image-net.org/download.php)

数据集大小：共1000个类、224*224彩色图像

训练集：共1,281,167张图像

测试集：共50,000张图像

数据格式：JPEG

```bash
    - 注：数据在dataset.py中加载。
 └─dataset
   ├─train                 # 训练数据集
   └─val                   # 评估数据集
```

# 环境要求

- 硬件(Ascend)
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 安装依赖包
pip install -r requirements.txt

# 进入脚本目录，在Ascend机器上单卡训练Autoslim
cd scripts
sh run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# 进入脚本目录，在Ascend机器上分布式训练Autoslim
cd scripts
sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] /path/to/imagenet-1k

# 进入脚本目录，在Ascend机器上验证Autoslim
cd scripts
sh run_standalone_eval_ascend.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]

# 进入脚本目录，在GPU机器上单卡训练Autoslim
cd scripts
sh run_standalone_train_gpu.sh 0 /path/to/imagenet-1k

# 进入脚本目录，在GPU机器上分布式训练Autoslim
cd scripts
sh run_distribute_train_gpu.sh /path/to/imagenet-1k

# 进入脚本目录，在GPU机器上验证Autoslim
cd scripts
sh run_standalone_eval_gpu.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]
```

# 脚本说明

## 脚本及样例代码

```text
├── cv
    ├── AutoSlim
        ├── README.md                    // AutoSlim描述
        ├── README_CN.md                 // AutoSlim中文描述
        ├── requirements.txt             // 需要的包
        ├── ascend310_infer              // 310推理 c++源码
        ├── preprocess.py                // 310推理 前处理
        ├── postprocess.py               // 310推理 后处理
        ├── scripts
        │   ├──run_distribute_train_ascend.sh   // ascend分布式训练
        │   ├──run_distribute_train_gpu.sh  // gpu分布式训练
        │   ├──run_eval_ascend.sh           // ascend推理
        │   ├──run_eval_gpu.sh              // gpu推理
        │   ├──run_infer_310.sh             // 310推理
        │   ├──run_standalone_train_ascend.sh   // ascend单卡训练
        │   └──run_standalone_train_gpu.sh  // gpu单卡训练
        ├── src
        │   ├──dataset.py                   // 数据集加载
        │   ├──lr_generator.py              // 学习率生成
        │   ├──autoslim_resnet.py           // 训练主干
        │   ├──autoslim_resnet_for_val.py   // 验证主干
        │   ├──slimmable_ops.py             // 函数定义
        │   ├──config.py                    // 参数配置
        │   └──autoslim_cfg.py              // 基本参数
        ├── train.py                     // 训练脚本
        ├── eval.py                      // 精度验证脚本
        └── export.py                    // 推理模型导出脚本
```

## 脚本参数

```bash

# train.py和set_parser.py中主要参数如下:

--device_target:运行代码的设备, 默认为"Ascend"
--device_id:运行代码设备的编号
--device_num:训练设备数量
--dataset_path:数据集所在路径
--batch_size:训练批次大小
--epoch_size:训练轮数
--save_checkpoint_path:模型保存路径
--file_format:模型转换格式

```

## 训练过程

### 训练

```bash
python train.py --dataset_path=/path/to/imagenet-1k

# 或进入./script目录, 运行脚本, 在Ascend机器上单卡训练
cd scripts
bash run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# 或进入./script目录, 运行脚本, 在Ascend机器上分布式训练
cd scripts
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] /path/to/imagenet-1k

# 或进入./script目录, 运行脚本, 在GPU机器上单卡训练
cd scripts
bash run_standalone_train_gpu.sh 0 /path/to/imagenet-1k

# 或进入./script目录, 运行脚本, 在GPU机器上分布式训练
cd scripts
bash run_distribute_train_gpu.sh /path/to/imagenet-1k
```

训练结束，损失值如下：

```bash
============== Starting Training ==============
epoch: 1 step: 100, loss is 6.9037023
epoch: 1 step: 200, loss is 6.9010477
epoch: 1 step: 300, loss is 6.895539

...

epoch: 94 step: 535, loss is 1.0588518
epoch: 94 step: 635, loss is 0.923507

...

```

模型检查点保存在已指定的目录下。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

```bash
python eval.py --dataset_path=/path/to/imagenet-1k --pretained_checkpoint_path=[PRETAINED_CHECKPOINT_PATH]

# 或进入./script目录, 运行脚本, 在Ascend机器上验证
cd scripts
sh run_eval_ascend.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]

# 或进入./script目录, 运行脚本, 在GPU机器上验证
cd scripts
sh run_eval_gpu.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]
```

测试数据集的准确度如下：

```bash
Start loading model.
Start evaluating.
Accuracy = 0.685
```

## 310推理过程

### 310推理

在310推理前需要输出模型的mindir文件

```bash
python export.py --pretained_checkpoint_path=[PRETAINED_CHECKPOINT_PATH] --export_model_name=[EXPORT_MODEL_NAME --file_format=[FILE_FORMAT]
```

310推理时，按如下方式运行脚本：

```bash
cd scripts
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

# 模型描述

## 性能

### 评估性能

| Parameters            | AutoSlim  |
| ------------------ | -------------------|
| Resource          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 CentOS 8.2；GPU V100；   |
| uploaded Date    | 2022-01-03       |
| MindSpore Version    | 1.3.0           |
| Dataset      |  Imagenet-1k   |
| Training Parameters   | epoch = 100, batch_size = 256, lr_max=0.1  momentum=0.9  weight_decay=1e-4  |
| Optimizer     | SGD  |
| Loss Function   |  交叉熵 |
| outputs      | 概率   |
| Speed      | 142毫秒/步   |
| Total time    |  20小时       |
| Checkpoint for Fine tuning |  86.94MB (.ckpt文件)   |
| Accuracy calculation method   |  分类准确率    |

# 随机情况说明

在train.py中，我们使用了dataset.Generator(shuffle=True)进行随机处理。

# ModelZoo主页

请浏览官网[主页](<https://gitee.com/mindspore/models>)。  

## FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models/blob/master/README_CN.md#faq)来查找一些常见的公共问题。

- **Q**：使用PYNATIVE_MODE发生内存溢出。

  **A**：内存溢出通常是因为PYNATIVE_MODE需要更多的内存， 将batch size设置为16降低内存消耗，可进行网络训练。