# 目录

<!-- TOC -->

- [SPPNet描述](#spatial_pyramid_pooling描述)
- [模型架构](#模型架构)
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
- [推理过程](#推理过程)
  - [导出MindIR](#导出MindIR)
  - [在Ascend310执行推理](#在Ascend310执行推理)
  - [结果](#结果)
- [模型描述](#模型描述)
  - [性能](#性能)
  - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SPPNET描述

SPPNET是何凯明等人2015年提出。该网络在最后一层卷积后加入了空间金字塔池化层(Spatial Pyramid Pooling layer)替换原来的池化层(Pooling layer),使网络接受不同的尺寸的feature maps并输出相同大小的feature maps，从而解决了Resize导致图片型变的问题。

[论文](https://arxiv.org/pdf/1406.4729.pdf)： K. He, X. Zhang, S. Ren and J. Sun, "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1904-1916, 1 Sept. 2015, doi: 10.1109/TPAMI.2015.2389824.

# 模型架构

SPPNET基于ZFNET，ZFNET由5个卷积层和3个全连接层组成，SPPNET在原来的ZFNET的conv5之后加入了Spatial Pyramid Pooling layer。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像

- 数据格式：JPEG
    - 注：数据在dataset.py中处理。

- 下载数据集。目录结构如下：

```text
└─dataset
    ├─ilsvrc                # 训练数据集
    └─validation_preprocess # 评估数据集
```

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

  ```bash
  # 进入脚本目录，训练SPPNET实例
  bash run_standalone_train_ascend.sh [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [DEVICE_ID] [TRAIN_MODEL]

  # 进入脚本目录，评估SPPNET实例
  bash run_standalone_eval_ascend.sh [TEST_DATA_PATH] [CKPT_PATH] [DEVICE_ID] [TEST_MODEL]

  # 运行分布式训练实例
  bash run_distribution_ascend.sh [RANK_TABLE_FILE] [TRAIN_DATA_PATH] [EVAL_DATA_PATH] [TRAIN_MODEL]
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── cv
    ├── sppnet
        ├── README.md                                 // SPPNet相关说明
        ├── scripts
        │   ├──run_distribution_ascend.sh             // Ascend多卡训练+推理的shell脚本
        │   ├──run_standalone_eval_ascend.sh          // Ascend单卡推理的shell脚本
        │   ├──run_standalone_train_ascend.sh         // Ascend单卡训练+推理的shell脚本
        ├── src
        │   ├──dataset.py                             // 创建数据集
        │   ├──sppnet.py                              // sppnet/zfnet架构
        │   ├──spatial_pyramid_pooling.py             // 金字塔池化层架构
        │   ├──generator_lr.py                        // 生成每个步骤的学习率
        │   ├──eval_callback.py                       // 训练时进行推理的脚本
        │   ├──config.py                              // 参数配置
        ├── train.py                                  // 训练脚本
        ├── eval.py                                   // 评估脚本
        ├── export.py                                 // 模型转换将checkpoint文件导出到air/mindir
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数：

  ```bash
  # zfnet配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 150,              # epoch大小
  'batch_size': 256,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'warmup_epochs' : 5,            # 热身周期数
  'iteration_max': 150,           # 余弦退火最大迭代次数
  'lr_init': 0.035,               # 初始学习率
  'lr_min': 0.0,                  # 最小学习率
  'weight_decay': 0.0001,         # 权重衰减
  'loss_scale': 1024,             # 损失等级
  'is_dynamic_loss_scale': 0,     # 是否动态调节损失

  # sppnet(single train)配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 160,              # epoch大小
  'batch_size': 256,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'warmup_epochs' : 0,            # 热身周期数
  'iteration_max': 150,           # 余弦退火最大迭代次数
  'lr_init': 0.01,                # 初始学习率
  'lr_min': 0.0,                  # 最小学习率
  'weight_decay': 0.0001,         # 权重衰减
  'loss_scale': 1024,             # 损失等级
  'is_dynamic_loss_scale': 0,     # 是否动态调节损失

  # sppnet(mult train)配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 160,              # epoch大小
  'batch_size': 128,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'warmup_epochs' : 0,            # 热身周期数
  'iteration_max': 150,           # 余弦退火最大迭代次数
  'lr_init': 0.01,                # 初始学习率
  'lr_min': 0.0,                  # 最小学习率
  'weight_decay': 0.0001,         # 权重衰减
  'loss_scale': 1024,             # 损失等级
  'is_dynamic_loss_scale': 0,     # 是否动态调节损失
  ```

train.py中主要参数如下：

  ```bash
  --train_model: 训练的模型，可选值为"zfnet"、"sppnet_single"、"sppnet_mult"，默认值为"sppnet_single"
  --train_path: 到训练数据集的绝对完整路径，默认值为"./imagenet_original/train"
  --eval_path: 到评估数据集的绝对完整路径，默认值为"./imagenet_original/val"
  --device_target: 实现代码的设备，默认值为"Ascend"
  --ckpt_path: 训练后保存的检查点文件的绝对完整路径，默认值为"./ckpt"
  --dataset_sink_mode: 是否进行数据下沉，默认值为True
  --device_id: 使用设备的卡号，默认值为0
  --device_num: 使用设备的数量，默认值为1
  ```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  # 单卡训练zfnet
  python train.py --train_path ./imagenet/train --eval_path ./imagenet/val --device_id 0 --train_model zfnet > log 2>&1 &

  # 或进入脚本目录，执行脚本
  bash run_standalone_train_ascend.sh ./imagenet_original/train ./imagenet_original/val 0 zfnet

  # 分布式训练zfnet，进入脚本目录，执行脚本
  bash run_distribution_ascend.sh ./hccl.json ./imagenet_original/train ./imagenet_original/val zfnet

  # 单卡训练sppnet(single train)
  python train.py --train_path ./imagenet/train --eval_path ./imagenet/val --device_id 0 > log 2>&1 &

  # 或进入脚本目录，执行脚本
  bash run_standalone_train_ascend.sh ./imagenet_original/train ./imagenet_original/val 0 sppnet_single

  # 分布式训练sppnet(single train)，进入脚本目录，执行脚本
  bash run_distribution_ascend.sh ./hccl.json ./imagenet_original/train ./imagenet_original/val sppnet_single

  # 单卡训练sppnet(mult train)
  python train.py --train_path ./imagenet/train --eval_path ./imagenet/val --device_id 0 --train_model sppnet_mult > log 2>&1 &

  # 或进入脚本目录，执行脚本
  bash run_standalone_train_ascend.sh ./imagenet_original/train ./imagenet_original/val 0 sppnet_mult

  # 分布式训练sppnet(mult train)，进入脚本目录，执行脚本
  bash run_distribution_ascend.sh ./hccl.json ./imagenet_original/train ./imagenet_original/val sppnet_mult
  ```

- 使用ImageNet2012数据集单卡进行训练zfnet

  经过训练后，损失值如下：

  ```bash
  ============== Starting Training ==============
  epoch: 1 step: 5004, loss is 6.906126
  epoch time: 571750.162 ms, per step time: 114.259 ms
  epoch: 1, {'top_5_accuracy', 'top_1_accuracy'}: {'top_5_accuracy': 0.005809294871794872, 'top_1_accuracy': 0.0010216346153846154}, eval_cost:19.47
  epoch: 2 step: 5004, loss is 5.69701
  epoch time: 531087.048 ms, per step time: 106.133 ms
  epoch: 2, {'top_5_accuracy', 'top_1_accuracy'}: {'top_5_accuracy': 0.1386017628205128, 'top_1_accuracy': 0.04453125}, eval_cost:14.53
  epoch: 3 step: 5004, loss is 4.6244116
  epoch time: 530828.240 ms, per step time: 106.081 ms
  epoch: 3, {'top_5_accuracy', 'top_1_accuracy'}: {'top_5_accuracy': 0.36738782051282054, 'top_1_accuracy': 0.1619591346153846}, eval_cost:13.73

  ...

  epoch: 149 step: 5004, loss is 1.448152
  epoch time: 531029.101 ms, per step time: 106.121 ms
  epoch: 149, {'top_5_accuracy', 'top_1_accuracy'}: {'top_5_accuracy': 0.8547876602564103, 'top_1_accuracy': 0.6478966346153846}, eval_cost:14.25
  update best result: {'top_5_accuracy': 0.8547876602564103, 'top_1_accuracy': 0.6478966346153846}
  update best checkpoint at: ./ckpt/best.ckpt
  epoch: 150 step: 5004, loss is 1.5808313
  epoch time: 530946.874 ms, per step time: 106.104 ms
  epoch: 150, {'top_5_accuracy', 'top_1_accuracy'}: {'top_5_accuracy': 0.8547876602564103, 'top_1_accuracy': 0.6483173076923077}, eval_cost:15.02
  update best result: {'top_5_accuracy': 0.8547876602564103, 'top_1_accuracy': 0.6483173076923077}
  update best checkpoint at: ./ckpt/best.ckpt
  End training, the best {'top_5_accuracy', 'top_1_accuracy'} is: {'top_1_accuracy': 0.6483173076923077, 'top_5_accuracy': 0.8547876602564103}, the best {'top_5_accuracy', 'top_1_accuracy'} epoch is 150
  ```

  模型检查点保存在当前目录ckpt中。

- 使用ImageNet2012数据集单卡进行单尺度训练sppnet(single train)

  经过训练后，损失值如下：

  ```bash
  ============== Starting Training ==============
  epoch: 1 step: 5004, loss is 6.754609
  epoch time: 1065948.526 ms, per step time: 213.019 ms
  epoch: 1, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.002864583333333333, 'top_5_accuracy': 0.014082532051282052}, eval_cost:20.34
  epoch: 2 step: 5004, loss is 5.5111685
  epoch time: 1021084.963 ms, per step time: 204.054 ms
  epoch: 2, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.0616386217948718, 'top_5_accuracy': 0.1776642628205128}, eval_cost:13.30
  epoch: 3 step: 5004, loss is 4.6289835
  epoch time: 1020991.373 ms, per step time: 204.035 ms
  epoch: 3, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.15853365384615384, 'top_5_accuracy':   0.35985576923076923}, eval_cost:13.60

  ...

  epoch: 159, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.6475560897435897, 'top_5_accuracy': 0.8568309294871795}, eval_cost:13.35
  epoch: 160 step: 5004, loss is 1.7843108
  epoch time: 1020822.415 ms, per step time: 204.001 ms
  epoch: 160, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.64765625, 'top_5_accuracy': 0.8556891025641026}, eval_cost:13.28
  End training, the best {'top_1_accuracy', 'top_5_accuracy'} is: {'top_1_accuracy': 0.6489783653846154, 'top_5_accuracy': 0.8572516025641026}, the best {'top_1_accuracy', 'top_5_accuracy'} epoch is 146
  ```

  模型检查点保存在当前目录ckpt中。

- 使用ImageNet2012数据集单卡多尺度训练SPPNET(mult train)

  经过训练后，损失值如下：

  ```bash
  ============== Starting Training ==============
  epoch: 1 step: 10009, loss is 6.8730383
  epoch time: 1529142.058 ms, per step time: 152.777 ms
  epoch: 1, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.0015825320512820513, 'top_5_accuracy': 0.009094551282051283}, cost:21.83
  update best result: {'top_1_accuracy': 0.0015825320512820513, 'top_5_accuracy': 0.009094551282051283}
  update best checkpoint at: ./ckpt/best.ckpt
  =================================================
  ================ Epoch:2 ==================
  epoch: 1 step: 10009, loss is 5.8023987
  epoch time: 2104207.357 ms, per step time: 210.232 ms
  ================ Epoch:3 ==================
  epoch: 1 step: 10009, loss is 4.779583
  epoch time: 1506529.824 ms, per step time: 150.518 ms
  epoch: 1, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.1718349358974359, 'top_5_accuracy': 0.3845753205128205}, cost:21.55
  update best result: {'top_1_accuracy': 0.1718349358974359, 'top_5_accuracy': 0.3845753205128205}
  update best checkpoint at: ./ckpt/best.ckpt
  =================================================

  ...

  ================ Epoch:148 ==================
  epoch: 1 step: 10009, loss is 1.8939599
  epoch time: 2134993.076 ms, per step time: 213.307 ms
  ================ Epoch:149 ==================
  epoch: 1 step: 10009, loss is 1.6252799
  epoch time: 1552970.284 ms, per step time: 155.157 ms
  epoch: 1, {'top_1_accuracy', 'top_5_accuracy'}: {'top_1_accuracy': 0.6448918269230769, 'top_5_accuracy': 0.8544270833333333}, cost:21.66
  =================================================

  ...

  ```

  模型检查点保存在当前目录ckpt中。

## 评估过程

### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行

  ```bash

  python eval.py --data_path ./imagenet_original/val --ckpt_path ./ckpt/best.ckpt --device_id 0 --train_model sppnet_single > eval_log.txt 2>&1 &

  # 或进入脚本目录，执行脚本

  bash run_standalone_eval_ascend.sh ./imagenet_original/val ./ckpt/best.ckpt 0 sppnet_single

  ```

  可通过"eval_log”文件查看结果。测试数据集的准确率如下：

  ```bash
  ============== Starting Testing ==============
  load checkpoint from [./ckpt/best.ckpt].
  result : {'top_5_accuracy': 0.8577724358974359, 'top_1_accuracy': 0.6503605769230769}
  ```

# 推理过程

## 导出MindIR

  ```bash
  python export.py --ckpt_file [CKPT_PATH] --export_model [EXPORT_MODEL] --device_id [DEVICE_ID]
  ```

参数ckpt_file为必填项， export_model必须在["zfnet", "sppnet_single", "sppnet_mult"]中选择。

## 在Ascend310执行推理

在执行推理前，mindir文件必须通过export.py脚本导出。以下展示了使用mindir模型执行推理的示例。 目前imagenet2012数据集仅支持batch_Size为1的推理。

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

## 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

  ```bash
  # zfnet.mindir 310推理的acc.log计算结果如下
  Total data: 50000, top1 accuracy: 0.6546, top5 accuracy: 0.85934.
  ```

# 模型描述

## 性能

### 评估性能

#### Imagenet2012上的zfnet

| 参数 | Ascend |
| -------------------------- | ------------------------------------------------------------|
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G |
| 上传日期 | 2021-09-21 |
| MindSpore版本 | 1.2.0-beta |
| 数据集 | ImageNet2012 |
| 训练参数 | epoch=150, step_per_epoch=5004, batch_size=256, lr=0.0035 |
| 优化器 | 动量 |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 | 概率 |
| 损失 | 1.58 |
| 速度 | 106毫秒/步 |
| 总时间 | 22小时 |
| 微调检查点 | 594M （.ckpt文件） |
| 脚本 | <https://gitee.com/mindspore/models/tree/master/research/cv/SPPNet> |

#### Imagenet2012上的sppnet(single train)

| 参数 | Ascend |
| -------------------------- | ------------------------------------------------------------|
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G |
| 上传日期 | 2021-09-21 |
| MindSpore版本 | 1.2.0-beta |
| 数据集 | ImageNet2012 |
| 训练参数 | epoch=160, step_per_epoch=5004, batch_size=256, lr=0.001 |
| 优化器 | 动量 |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 | 概率 |
| 损失 | 1.55 |
| 速度 | 203毫秒/步 |
| 总时间 | 200小时 |
| 微调检查点 | 594M （.ckpt文件） |
| 脚本 | <https://gitee.com/mindspore/models/tree/master/research/cv/SPPNet> |

#### Imagenet2012上的sppnet(single mult)

| 参数 | Ascend |
| -------------------------- | ------------------------------------------------------------|
| 资源 | Ascend 910；CPU 2.60GHz, 192核；内存：755G |
| 上传日期 | 2021-09-21 |
| MindSpore版本 | 1.2.0-beta |
| 数据集 | ImageNet2012 |
| 训练参数 | epoch=160, step_per_epoch=10009, batch_size=128, lr=0.001 |
| 优化器 | 动量 |
| 损失函数 | Softmax交叉熵 |
| 输出 | 概率 | 概率 |
| 损失 | 1.78 |
| 速度 | 180毫秒/步 |
| 总时间 | 200小时 |
| 微调检查点 | 601M （.ckpt文件） |
| 脚本 | <https://gitee.com/mindspore/models/tree/master/research/cv/SPPNet> |

# 随机情况说明

dataset.py中设置了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
