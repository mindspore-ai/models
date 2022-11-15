# 目录

<!-- TOC -->

- [目录](#目录)
- [PGAN模型简介](#模型简介)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CelebA上的PGAN](#CelebA上的PGAN)
- [ModelZoo主页](#modelzoo主页)

# 模型简介

PGAN指的是Progressive Growing of GANs for Improved Quality, Stability, and Variation, 这个网络的特点是渐进地生成人脸图像

[论文](https://arxiv.org/abs/1710.10196)：Progressive Growing of GANs for Improved Quality, Stability, and Variation//2018 ICLR

[参考的github地址](https://github.com/facebookresearch/pytorch_GAN_zoo)

# 模型架构

整个网络结构由一个生成器和一个判别器构成。该网络的核心idea是从低精度开始生成网络，随着训练的进程添加新的图层并且逐渐开始学习到网络中细节的部分。这样做加快了训练速度，也稳定了训练状态。除此之外，本代码实现了论文中的equalized learning rate，exponential running average ，残差结构，WGANGPGradientPenalty等核心trick。

# 数据集

使用的数据集: [CelebA](<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)

CelebFaces Attributes Dataset (CelebA) 是一个大规模的人脸属性数据集，拥有超过 200K 的名人图像，每个图像有 40 个属性注释。 CelebA 多样性大、数量多、注释丰富，包括

- 10,177 number of identities,
- 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.

该数据集可用作以下计算机视觉任务的训练和测试集：人脸属性识别、人脸检测以及人脸编辑和合成。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train.py --train_data_path /path/data/image --config_path ./910_config.yaml
  OR
  bash run_standalone_train.sh  /path/data/image device_id ./910_config.yaml

  # 运行分布式训练示例
  bash run_distributed_train.sh /path/data/image  /path/hccl_config_file ./910_config.yaml
  # 运行评估示例
  export DEVICE_ID=0
  export RANK_SIZE=1
  python eval.py --checkpoint_g=/path/checkpoint  --device_id=0
  OR
  bash run_eval.sh /path/checkpoint 0
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。该配置文件的绝对路径作为运行分布式脚本的第二个参数。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

  对于评估脚本，checkpoint文件被训练脚本默认放置在
  `/output/{scale}/checkpoint`目录下，执行脚本时需要将检查点文件（Generator）的名称作为参数传入。

# 脚本说明

## 脚本及样例代码

```text
.
└─ cv
  └─ PGAN
    ├── scripts
      ├──run_distributed_train_ascend.sh          # 分布式训练的shell脚本
      ├──run_standalone_train.sh                  # 单卡训练的shell脚本
      ├──run_eval_ascend.sh                       # 评估脚本
    ├─ src
      ├─ customer_layer.py                 # 基础cell
      ├─ dataset.py                        # 数据加载
      ├─ image_transform.py                # 处理图像函数
      ├─ network_D.py                      # 判别网络
      ├─ network_G.py                      # 生成网络
      ├─ optimizer.py                      # loss计算
    ├─ eval.py                             # 测试脚本
    ├─ export.py                           # MINDIR模型导出脚本
    ├─ train.py                            # 训练脚本
    └─ README_CN.md                        # PGAN的文件描述
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  export DEVICE_ID=0
  export RANK_SIZE=1
  python train.py --train_data_path /path/data/image --config_path ./910_config.yaml
  OR
  bash run_standalone_train.sh  /path/data/image device_id ./910_config.yaml
  ```

  训练结束后，当前目录下会生成output目录，在该目录下会根据你设置的ckpt_dir参数生成相应的子目录，训练时保存各个scale的参数

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_distributed_train.sh /path/data/image  /path/hccl_config_file ./910_config.yaml
  ```

  上述shell脚本将在后台运行分布式训练。该脚本将在脚本目录下生成相应的LOG{RANK_ID}目录，每个进程的输出记录在相应LOG{RANK_ID}目录下的log_distribute文件中。checkpoint文件保存在output/rank{RANK_ID}下。

## 评估过程

### 评估

- 在Ascend环境下生成图片
  用户生成64张人脸图片

  评估时选择已经生成好的检查点文件，作为参数传入测试脚本，对应参数为`checkpoint_g`(保存了生成器的checkpoint)。请指定“AvG**.ckpt”进行推理。

  ```bash
  bash run_eval.sh /path/checkpoint 0
  ```

  测试脚本执行完成后，用户进入当前目录下的`img_eval/`下查看生成的人脸图片。
  默认推理配置为scale=128，如推理scale=64规格的图片，请将eval.py中的设置更改为

  ```bash
  scales = [4, 8, 16, 32, 64]
  depth = [512, 512, 512, 512, 256]
  ```

  如推理scale=32规格的图片，请将eval.py中的设置更改为

  ```bash
  scales = [4, 8, 16, 32]
  depth = [512, 512, 512, 512]
  ```

  如推理scale=16规格的图片，请将eval.py中的设置更改为

  ```bash
  scales = [4, 8, 16]
  depth = [512, 512, 512]
  ```

  如推理scale=8规格的图片，请将eval.py中的设置更改为

  ```bash
  scales = [4, 8]
  depth = [512, 512]
  ```

  如推理scale=4规格的图片，请将eval.py中的设置更改为

  ```bash
  scales = [4]
  depth = [512]
  ```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出MindIR

```shell
python export.py --checkpoint_g [GENERATOR_CKPT_NAME] --device_id [DEVICE_ID]
```

默认使用scale=128生成的ckpt文件进行导出，脚本会在当前目录下生成对应的MINDIR文件。

### 在Ascend310执行推理

在执行推理前，必须通过export脚本导出MINDIR模型。以下命令展示了如何通过命令在Ascend310上对图片进行属性编辑：

```bash
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [NIMAGES] [DEVICE_ID]
```

- `MINDIR_PATH` MINDIR文件的路径
- `NEED_PREPROCESS` 表示属性编辑文件是否需要预处理，可以在y或者n中选择，如果选择y，表示进行预处理（在第一次执行推理时需要设置为y）
- `NIMAGES` 表示生成图片的数量.
- `DEVICE_ID` 可选，默认值为0.

### 结果

推理结果保存在脚本执行的目录下，属性编辑后的图片保存在`result_Files/`目录下，推理的时间统计结果保存在`time_Result/`目录下。编辑后的图片以`generated_{NUMBER}.png`的格式保存.

# 模型描述

## 性能

### 评估性能

#### CelebA上的PGAN

| 参数                       | Ascend 910                                                  |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本                   | PGAN                                                      |
| 资源                       | Ascend                                                      |
| 上传日期                   | 09/31/2021 (month/day/year)                                 |
| MindSpore版本              | 1.3.0                                                       |
| 数据集                     | CelebA                                                      |
| 训练参数                   | batch_size=128, lr=0.001                                   |
| 优化器                     | Adam                                                        |
| 生成器输出                 | image                                                       |
| 速度                       |8p:9h26m54S; 1p:76h23m39s; 1.1s/step                                |
| 收敛loss                    |G:[-232.61 to 273.87] loss D:[-27.736 to 2.601]                             |
| 脚本                       | [PGAN script](https://gitee.com/mindspore/models/tree/master/research/cv/PGAN) |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)