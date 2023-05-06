# 目录

- [FaceNet描述](#facenet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练用法](#训练用法)
        - [训练样例](#训练样例)
    - [推理过程](#推理过程)
        - [推理用法](#推理用法)
        - [推理结果](#推理结果)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# FaceNet描述

FaceNet是一个通用人脸识别系统：采用深度卷积神经网络（CNN）学习将图像映射到欧式空间。空间距离直接和图片相似度相关：同一个人的不同图像在空间距离很小，不同人的图像在空间中有较大的距离，可以用于人脸验证、识别和聚类。

[论文](https://arxiv.org/abs/1503.03832)：Florian Schroff, Dmitry Kalenichenko, James Philbin. 2015.

# 模型架构

FaceNet总体网络架构如下：

[链接](https://arxiv.org/abs/1503.03832)

# 数据集

VGGFace2训练集：[VGGFACE2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2)

LFW测试集和已生成的三元组：[LFW](https://pan.baidu.com/s/1BqiMaK-jp0Wi0Ez1Exdg8g?pwd=vfka)

- 训练集大小: 145G, 3.31 million images, 9131种ID
- 数据格式：RGB
    - 注：数据在src/dataloader.py中处理。
- 下载数据集，目录结构如下：

```text
└─dataset
   ├─n000002     # id1
   └─n000003     # id2
```

resnet50 checkpoint文件，放在./src目录下：[resnet50](https://www.mindspore.cn/resources/hub/details?MindSpore/1.7/resnet50_imagenet2012)

本项目以及提前生成三元组序列csv文件[triplets.csv](https://pan.baidu.com/s/1BqiMaK-jp0Wi0Ez1Exdg8g?pwd=vfka)，建议使用data_loader.py加载此文件

# 环境要求

- 硬件（GPU）
    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```python
├── FaceNet
  ├── README_CN.md                      # 模型相关描述
  ├── scripts
   ├──run_distribute_train.sh           # 用于Ascend分布式训练脚本
   ├──run_distribute_train_gpu.sh       # 用于GPU分布式训练脚本
   ├──run_standalone_train.sh           # 用于Ascend单卡训练脚本
   ├──run_standalone_train_gpu.sh       # 用于GPU单卡训练脚本
   ├──run_eval.sh                       # 用于测试脚本
  ├── src
   ├──config.py                         # 参数配置
   ├──data_loader.py                    # 训练集dataloader(读取生成好的triplets)
   ├──data_loader_gen_online.py         # 训练集dataloader(在线生成triplets)
   ├──eval_callback.py                  # 自定义的训练回调函数
   ├──eval_metrics.py                   # 评估标准
   ├──LFWDataset.py                     # 测试集dataloader
   ├──loss.py                           # tripletloss损失函数
   ├──models                            # FaceNet模型
   ├──resnet.py                         # ResNet骨干网络
   ├──write_csv_for_making_triplets.py  # 通过原始数据生成训练用的数据读取序列csv
   ├──generating_training_triplets.py   # 通过读取序列csv生成训练用的三元组csv
  ├── eval.py                           # 验证脚本
  ├── export.py                         # 导出模型
  ├── train.py                          # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'rank': 0,                                              # 默认rank
"num_epochs":240(Ascend) or 600(GPU),                                      # 训练epoch数
"num_train_triplets":3000000,                           # 生成的训练三元组数
"num_valid_triplets":10000,                             # 生成的评估三元组数
"batch_size":64,                                        # 数据批次大小
"num_workers":4,                                        # 数据并行处理数
"learning_rate":0.004,                                  # 初始学习率
"margin":0.5,                                           # Triplet Loss约束
"step_size":50,                                         # 学习率调整epoch间隔
"keep_checkpoint_max":10,                               # 保存ckpt文件的最大数量
"lr_epochs": '30,60,90,120,150,180',                    # 学习率衰减轮数
"lr_gamma": 0.1,                                        # 学习率调整gamma值
"T_max": 200,                                           # 学习率调整最大轮数
"warmup_epochs": 0                                      # warmup epoch数量
```

## 训练过程

### 训练用法

生成训练用triplets

```python

# 生成数据读取序列csv

python write_csv_for_making_triplets.py --data_root [root to your dataset] --csv_name [dir/csv_name.csv]

# 生成训练用三元组序列csv(使用data_loader_generate_triplets_online.py作为dataloader时可跳过这一步）

python generating_training_triplets.py --data_url [root to your dataset] --csv_dir [root to your data sequence csv file (generated from previous step)] \
--output_dir [outputdir/] --triplet_num [number of triplets]

```

VGGFACE2下载后解压VGGFACE2.zip到项目文件夹中,确保解压文件夹中有triplets.csv,vggface2.csv,更改train.py中data_url默认参数为该文件夹路径，pretrain_ckpt_path默认参数为./src/resnet50.ckpt

使用GPU作为训练平台

```shell
# 单卡训练
cd FaceNet/scripts
bash run_standalone_train_gpu.sh
# 8卡训练
cd FaceNet/scripts
bash run_distribute_train_gpu.sh 8
```

使用Ascend作为训练平台

```shell
# 单卡训练
cd FaceNet/scripts
bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]
# 8卡训练
cd FaceNet/scripts
bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE]
```

### 训练样例

ckpt文件将存储在 `./result/` 路径下，启动训练后loss值和inference结果如下：

```shell
epoch: 1 step: 19, loss is 0.518976628780365
epoch: 1 step: 19, loss is 0.5445463061332703
epoch: 1 step: 19, loss is 0.6057891249656677
epoch: 1 step: 19, loss is 0.48626136779785156
epoch: 1 step: 19, loss is 0.4919613003730774
epoch: 1 step: 19, loss is 0.4735066592693329
epoch: 1 step: 19, loss is 0.39775145053863525
epoch: 1 step: 19, loss is 0.5671045184135437
epoch time: 12290.290 ms, per step time: 646.857 ms
epoch time: 12773.835 ms, per step time: 672.307 ms
epoch time: 12755.992 ms, per step time: 671.368 ms
epoch time: 12048.930 ms, per step time: 634.154 ms
epoch time: 12819.222 ms, per step time: 674.696 ms
epoch time: 11697.613 ms, per step time: 615.664 ms
epoch time: 12122.336 ms, per step time: 638.018 ms
epoch time: 11700.574 ms, per step time: 615.820 ms
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
Accuracy on LFW: 0.7186+-0.0117
```

## 推理过程

### 推理用法

LFW下载后解压lfw.tar.gz到项目文件夹中
请在eval.py中更改"--eval_root_dir"参数为lfw.tar.gz解压后文件路径，"--eval_pairs_path"参数为LFW_pairs.txt路径

```shell
# 推理示例
bash run_eval.sh [CKPT_PATH]
```

### 推理结果

```shell
"Validating on LFW! ..."
Accuracy on LFW: 0.9401+-0.0118
```

# 模型说明

## 训练性能

| 参数 | Ascend | GPU | Ascend-8P |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 资源 | Ascend 910；ARM CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | Tesla V100-PCIE-32GB; X86_64 CPU Xeon 8180 2.50GHz; 112核 |  Ascend 910 * 8；ARM CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期 | 2022-06-29 | 2022-4-11 | 2022-06-29 |
| MindSpore版本 | 1.7.0 | 1.6.0 | 1.7.0 |
| 数据集 | VGGFACE2 | VGGFACE2 | VGGFACE2 |
| 训练参数 | epoch=240, step=156, batch_size=64, lr=0.004 | epoch=600, step=156, batch_size=64, lr=0.004 | epoch=240, step=20, batch_size=64*8, lr=0.004 |
| 优化器 | Adam | Adam | Adam |
| 损失函数 | TripletLoss | TripletLoss | TripletLoss |
| 输出 | 概率 | 概率 | 概率 |
| 损失 | 0.32 | 0.36 | 0.32 |
| 速度 | 315毫秒/步 |   | 2194毫秒/步 |
| 总时间 | 2h46m43s|  | 3h16m48s |
| 微调检查点 | 402M （.ckpt文件） | 143M （.ckpt文件） | 402M （.ckpt文件）|
| 脚本 | [Facenet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/FaceNet) | [Facenet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/FaceNet) | [Facenet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/FaceNet) |
                                              |

## 推理性能

| 参数 | Ascend | GPU | Ascend-8P |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 资源 | Ascend 910；ARM CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 | Tesla V100-PCIE-32GB; X86_64 CPU Xeon 8180 2.50GHz; 112核 |  Ascend 910 * 8；ARM CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期 | 2022-06-29 | 2022-4-11 | 2022-06-29 |
| MindSpore版本 | 1.7.0 | 1.6.0 | 1.7.0 |
| 数据集 | LFW | LFW | LFW |
| 训练参数 | batch_size=64| batch_size=64 |batch_size=64 |
| 识别准确率 | 90.5% | 94.42% |92.60%|

# ModelZoo主页

请核对官方 [主页](https://gitee.com/mindspore/models)。