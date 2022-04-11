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

VGGFace2训练集：[VGGFACE2](https://112.95.163.82:443/nanhang/shenao/data/dataset.zip?AWSAccessKeyId=ZQMSJWHQSZAKGA1LD2WR&Expires=1671081389&Signature=0w2hd7GtIHkZkzEpDIqTfjjWi3Q%3D)

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

# 环境要求

- 硬件（GPU）
    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

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
   ├──write_csv_for_making_triplets.py  # 生成csv
  ├── eval.py                           # 验证脚本
  ├── export.py                         # 导出模型
  ├──train.py                           # 训练脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
'rank': 0,                                              # 默认rank
"num_epochs":600,                                       # 训练epoch数
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

使用GPU作为训练平台

```shell
# 单卡训练
cd FaceNet/scripts
bash run_standalone_train_gpu.sh
# 8卡训练
cd FaceNet/scripts
bash run_distribute_train_gpu.sh 8
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

```shell
# 推理示例
python eval.py --ckpt [CKPT_PATH]
example:python eval.py --ckpt "/data1/face/FaceNet_mindspore/result/330/facenet-rank0-300_56.ckpt"
```

### 推理结果

```shell
"Validating on LFW! ..."
Accuracy on LFW: 0.9401+-0.0118
```

# 模型说明

## 训练性能

| 参数                        | GPU                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | FaceNet                          |
| 运行环境                    | GeForce RTX 3090 ；CPU 2.90GHz，16cores；内存，252G |
| 上传时间                    | 2022-4-11                            |
| MindSpore版本 | 1.6 |
| 数据集                      | VGGFace2                              |
| 训练参数                    | src/config.py                         |
| 优化器                      | Adam                              |
| 损失函数                    | TripletLoss         |
| 损失 | 0.36 |
| 调优检查点    | 143M                                                |

## 推理性能

| 参数          | GPU                                                 |
| ------------- | --------------------------------------------------- |
| 模型名称      | FaceNet                                             |
| 运行环境      | GeForce RTX 3090 ；CPU 2.90GHz，16cores；内存，252G |
| 上传时间      | 2022-4-11                                           |
| MindSpore版本 | 1.6                                                 |
| 数据集        | LFW                                                 |
| batchsize     | 64                                                  |
| 识别准确率    | 94.42%                                              |

# ModelZoo主页

请核对官方 [主页](https://gitee.com/mindspore/models)。