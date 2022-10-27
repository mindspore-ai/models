# 目录

<!-- TOC -->

- [目录](#目录)
- [StyTr^2描述](#StyTr^2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [使用的数据集](#使用的数据集)
    - [数据结构](#数据结构)
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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# StyTr^2描述

StyTr^2指的是SyTr^2 : Image Style Transfer with Transformers，提出了一种基于变压器模型的无偏图像样式传输方法。

[论文](https://arxiv.org/abs/2105.14576)

[作者主页]*作者: [Yingying Deng](https://diyiiyiii.github.io/), Fan Tang, XingjiaPan, Weiming Dong, Chongyang Ma, Changsheng Xu*

# 模型架构

StyTr^2网络结构，将内容图像和样式图像分割成小块，并使用线性投影得到图像序列，然后将使用CAPE添加的内容序列输入到内容转换器编码器中，而样式序列则输入到样式转换器编码器中。在两个变压器编码器之后，采用多层变压器解码器根据样式序列对内容序列进行风格化。最后，使用渐进式上采样解码器来获得具有高分辨率的程式化图像。

# 数据集

## 使用的数据集

- 内容数据集：COCO2014，包含80个类、80000张训练图像和40000张验证图像
    - [下载地址](https://pjreddie.com/projects/coco-mirror/) 训练集13GB，80K张图像
- 风格数据集：风格数据集是从[WikiArt](https://www.wikiart.org)上收集的，可自行收集，也可以在提供的下载连接中下载
    - 此处是ArtGAN作者收集的数据集[下载地址](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md) 25.4GB
    - 注意:WikiArt数据集只能用于非商业研究目，WikiArt数据集中的图像是从 WikiArt.org 获得的,作者既不对这些图像的内容或含义负责，通过使用维基艺术数据集，您同意遵守 WikiArt.org 的条款和条件。
- 数据格式：RGB图像

## 数据结构

将数据集dataset解压到任意路径，文件夹结构如下：

使用utils/split_dataset.py脚本，将wikiart数据集划分成train和test，并删除原始的数据

```bash
├── dataset
│   ├── COCO2014
│   │   ├── train2014
│   │   └── val2014
│   ├── wikiart
│   │   └── train
|   │   │   └── ...
│   │   └── test
|   │   │   └── ...
```

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.6/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 运行

  ```bash
  # 运行训练示例
  bash bash scripts/run_train.sh [BATCH_SIZE] [DEVICE_ID] [CONTENT_PATH] [STYLE_PATH] [AUXILIARY_PATH] [SAVE_PATH] [IMG_SAVE_PATH]
  # 例如：
  bash scripts/run_train.sh 4 0 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture
  # 运行分布式训练示例GPU
  bash scripts/run_train_distribute.sh [BATCH_SIZE] [CONTENT_PATH] [STYLE_PATH] [AUXILIARY_PATH] [SAVE_PATH] [IMG_SAVE_PATH]
  # 例如：
  bash scripts/run_train_distribute.sh 4 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture

  # 运行评估示例
  bash scripts/run_eval.sh [CONTENT_PATH] [STYLE_PATH] [DECODER_PATH] [EMBEDDING_PATH] [TRANS_PATH] [OUTPUT_PATH]
  # 例如：
  bash scripts/run_eval.sh ./dataset/COCO2014/val2014 ./dataset/wikiart/test decoder_160000.ckpt embedding_160000.ckpt transformer_160000.ckpt ./output
  ```

# 脚本说明

## 脚本及样例代码

```bash
.
├── StyTr^2
    ├─ README_CN.md                        # 模型相关说明
    ├─ eval.py                             # 评估脚本
    ├─ train.py                            # 训练脚本
    ├─ requirements.txt                    # 环境要求
    ├─ auxiliary                           #准备文件的文件夹
    ├─ dataset
    │  ├─ COCO2014                             # 内容图像
    │  │  ├─ train2014                         # 训练图像
    │  │  └─ val2014                          # 评估图像
    │  ├─ wikiart                         # 风格图像
    │  │  └─ train                       # 训练图像
    │  │  └─ test                       # 评估图像
    ├─ scripts
    │  ├─ run_eval.sh                      # 启动评估
    │  ├─ run_distribute_train.sh          # 启动多卡训练
    │  └─ run_train.sh                     # 启动单卡训练
    └─ src
       ├─models
       │  ├─ StyTR.py                # StyTr模型
       │  ├─ transformer.py             # transformer模型
       │  ├─ ViT_helper.py             # ViT模型
       │  └─ WithLossCellD.py          # 损失函数
       └─tils
          ├─ split_dataset.py             # 划分数据集
          └─ function.py                   # 辅助计算方法

```

auxiliary文件可以在此下载([链接](https://pan.baidu.com/s/1Shg5Faftyk9NjsJ4fWbWsA)) 提取码:4ga0。

## 脚本参数

  ```python
  'lr'=5e-4          # 学习速率
  'epoch'=16           # 轮次数
  'batch_size'=4            #批大小
  'lr_decay'=1e-5       # 学习率衰减值
  'style_weight'=10.0     # 风格损失权重值
  'content_weight'=7.0  # 内容损失权重值
  'save_model_interval'=10000 # 开始保存模型的间隔step数
  ```

## 训练过程

### 训练

- GPU处理器环境运行

  ```bash
  bash scripts/run_train.sh 4 0 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture
  ```

  或

  ```bash
  python train.py --batch_size=4 --device_id=0 --content_dir=./dataset/COCO2014/train2014 --style_dir=./dataset/wikiart/train --auxiliary_dir=./auxiliary --save_dir=./save_model --save_picture=./picture
  ```

  ```bash
  用法：train.py [--batch_size BATCH_SIZE] [--device_id DEVICE_ID]
                [--content_dir CONTENT_PATH]
                [--auxiliary_dir AUXILIARY_PATH]
                [--style_dir STYLE_PATH][--save_dir SAVE_PATH]
                [--save_picture IMG_SAVE_PATH]

  选项：
    --content_dir                    内容图片数据路径
    --batch_size                      批大小
    --device_id                       device_id
    --style_dir                       风格图片数据路径
    --auxiliary_dir                   预处理模型路径
    --save_dir                        模型保存路径
    --save_picture                    图片保存路径
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认`./save_model/`脚本文件夹下找到检查点文件。

### 分布式训练

- 8卡GPU处理器环境运行

    ```bash
    bash scripts/run_train_distribute.sh 4 ./dataset/COCO2014/train2014 ./dataset/wikiart/train ./auxiliary ./save_model ./picture
    ```

## 评估过程

### 评估

- 评估dataset数据集
  运行评估脚本，对用户指定的测试数据集进行测试，生成测试集内风格的内容图像。在运行以下命令前，请检查各数据源的路径是否正确。

  ```bash
  bash scripts/run_eval.sh ./dataset/COCO2014/val2014 ./dataset/wikiart/test decoder_160000.ckpt embedding_160000.ckpt transformer_160000.ckpt ./output
  ```

  或者，

  ```bash
  python eval.py    --content_dir=./dataset/COCO2014/val2014
                    --style_dir=./dataset/wikiart/test
                    --decoder_path=decoder_160000.ckpt
                    --embedding_path=embedding_160000.ckpt
                    --trans_path=transformer_160000.ckpt
                    --output=./output > eval.log 2>&1 &
  ```

  如果只是想用单张图片试一下我们模型，可以使用如下命令，正确输入单张图片路径即可，

  ```bash
  python eval.py    --content=./dataset/COCO2014/val2014/COCO_val2014_000000000136.jpg
                    --style=./dataset/wikiart/test/Impressionism/abdullah-suriosubroto_indonesian-landscape-1.jpg
                    --decoder_path=decoder_160000.ckpt
                    --embedding_path=embedding_160000.ckpt
                    --trans_path=transformer_160000.ckpt
                    --output=./output > eval.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看评估过程。测试数据生成的图片保存在指定的结果目录中。我们训练好的模型可以在此下载([链接](https://pan.baidu.com/s/1Shg5Faftyk9NjsJ4fWbWsA)) 提取码:4ga0

# 模型描述

## 性能

### 训练性能

| 参数                      |GPU               |
| --------------------------| ---------------------- |
| 模型版本                  | StyTr^2           |
| 资源                      |GPU(NVIDIA-A100-PCIE-40G)|
| 上传日期                  | 2022-09-01      |
| MindSpore版本             |1.7.0              |
| 数据集                    | COCO2014/train2014，wikiart/train     |
| 训练参数                  | iters=320000, lr=5e-4   |
| 优化器                    | Adam                           |
| 损失函数                  | MSE loss                    |
| 输出                      | 图片                   |
| 损失                      |loss_c, loss_s, l_identity1, l_identity2|
| 速度                      | 单卡: 792毫秒/步|
| 总时长                    | 单卡：1天8小时21分41秒;  |
| 微调检查点                | 121.28MB (transformer.ckpt文件),13.37 MB(decoder.ckpt),386 KB(embedding.ckpt)|
| 脚本                      | scripts/run_train.sh |

### 评估性能

| 参数          |GPU       |
| ------------------- |--------------------------- |
| 模型版本       | StyTr^2  |
| 资源            |  GPU(NVIDIA-A100-PCIE-40G) |
| 上传日期       | 2022-09-01      |
| MindSpore 版本   | 1.7.0|
| 数据集             | COCO02014/val2014,wikiart/test |
| batch_size          | 1   |
| 输出             | 图片        |

# 随机情况说明

使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)