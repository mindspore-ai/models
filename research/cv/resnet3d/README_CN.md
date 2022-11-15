# 目录

<!-- TOC -->

- [resnet3d描述](#resnet3d描述)
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
  - [评估过程](#评估过程)
  - [ONNX评估](#ONNX评估)
  - [导出过程](#导出过程)
  - [导出](#导出)
  - [推理过程](#推理过程)
  - [推理](#推理)
- [模型描述](#模型描述)
  - [性能](#性能)
  - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# resnet3d描述

## 概述

通常做图像分类使用的ResNet网络的卷积核一般只是在2D图像上做滑动窗口，计算特征图，卷积核的形状一般为[out_channel, in_channel, W, H]。而在视频分类任务中一般对网络输入的是视频中的一段序列，比如16帧或32帧，这样在原有WH维度上又增加了一个时间T的维度，卷积核的形状为 [out_channel, in_channel, T, W, H]。这时，卷积核不止在2D平面上滑动，还需要在第三个维度T上移动，提取帧之间的关联特征。这样就需要对2D的ResNet进行改造，将其改造为3D的ResNet网络。ResNet3D保持原有的ResNet的整体架构不变，替换每个block中的basicblock或bottleneckblock中的卷积核为Conv3D,同时池化层也需要替换为3D池化。

如下为MindSpore使用ucf101数据集和hmdb51数据集对resnet3d进行训练的示例。resnet3d可参考[论文1](https://arxiv.org/abs/2004.04968)

## 论文

1. [论文](https://arxiv.org/abs/2004.04968)：Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, Yutaka Satoh."Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs?"

# 模型架构

resnet3d的总体网络架构如下：
[链接](https://arxiv.org/abs/2004.04968)

# 数据集

使用的数据集：

- [Kinetics-400](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)
    - 具有400个类别的大型视频行为识别数据集, 用于预训练
- [Kinetics-700](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)
  - 具有700个类别的大型视频行为识别数据集, 用于预训练
- [MIT](http://moments.csail.mit.edu/)
  - MIT-IBM Watson AI Lab 推出的一个全新的百万规模视频理解数据集Moments in Time,共有100,0000 个视频, 用于预训练
- [hmdb51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
  - 一个小型的视频行为识别数据集，包含51类动作，共有6849个视频，每个动作至少包含51个视频, 用于Fine-tune，此处使用Stabilized HMDB51
  - labels地址(http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
  - 从YouTube收集的具有101个动作类别的真实动作视频的动作识别数据集, 共计13320个视频, 用于Fine-tune
  - labels地址(https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)

预训练模型获取地址：
[链接](https://github.com/kenshohara/3D-ResNets-PyTorch)
使用以下命令进行转换

```text
python pth_to_ckpt.py --pth_path=./pretrained.pth --ckpt_path=./pretrained.ckpt
```

特别说明：

按照下面格式创建目录，将下载好的hmdb51_sta.rar解压，把解压出来的文件夹放到videos目录中。将数据集对应的labels解压，把解压出的txt文件移动到labels目录中。

```text
.
└──hmdb51
  ├──videos
  ├──labels
```

需要将其组织为：

```text
└──hmdb51
  ├──videos
  ├──labels
  ├──jpg
  └──json
```

使用src/generate_video_jpgs.py将avi格式的视频文件转换为jpg格式的图片文件

```text
cd ~/src
python3 generate_video_jpgs.py --video_path ~/dataset/hmdb51/videos/ --target_path ~/dataset/hmdb51/jpg/
```

使用src/generate_hmdb51_json.py生成json格式的标注文件

```text
cd ~/src
python3 generate_hmdb51_json.py --dir_path ~/dataset/hmdb51/labels/ --video_path ~/dataset/hmdb51/jpg/ --dst_dir_path ~/dataset/hmdb51/json
```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [DEVIDE_NUM] [RANK_TABLE_FILE] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]

# 单机训练
用法：run_standalone_train.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]

# 运行评估示例
用法：bash run_eval.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [INFERENCE_CKPT_PATH]
```

# 脚本说明

## 脚本及样例代码

```shell
.
└──resnet3d
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├──run_eval_onnx.sh                    # ONNX评估的shell脚本
  ├── src
    ├── __init__.py
    ├── config.py                          # yaml文件解析
    ├── dataset.py                         # 数据预处理
    ├── generate_hmdb51_json.py            # 生成hmdb51数据集的json格式标注文件
    ├── generate_ucf101_json.py            # 生成ucf101数据集的json格式标注文件
    ├── generate_video_jpgs.py             # 将avi格式的视频文件转换为jpg格式的图片文件
    ├── inference.py                       # 推理逻辑定义
    ├── loader.py                          # 通过PIL加载图片
    ├── loss.py                            # ImageNet2012数据集的损失定义
    ├── lr.py                              # 生成每个步骤的学习率
    ├── pil_transforms.py                  # 通过PIL对视频片段进行预处理
    ├── ResNet3D.py                        # ResNet3D网络定义
    ├── save_callback.py                   # 定义边训练边推理
    ├── temporal_transforms.py             # 定义视频片段裁切方式
    ├── videodataset.py                    # 自定义数据集加载方式
    └──  videodataset_multiclips.py        # 自定义数据集加载方式
  ├── pth_to_ckpt.py                       # 将预训练模型从pth格式转换为ckpt格式
  ├── eval.py                              # 评估网络
  ├── eval_onnx.py                         # ONNX评估脚本
  ├── train.py                             # 训练网络
  ├── hmdb51_config.yaml                   # 参数配置
  └── ucf101_config.yaml                   # 参数配置  
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

```text
    'video_path': Path('~your_path/dataset/ucf101/jpg/'),
    'annotation_path': Path('~your_path/dataset/ucf101/json/ucf101_01.json'),      # 标注文件路径
    'result_path': './results/ucf101',                                             # 训练、推理结果路径
    'pretrain_path': '~/your_path/pretrained.ckpt',                                # 预训练模型文件路径
    'inference_ckpt_path': "~/your_path/results/ucf101/result.ckpt",               # 用于推理的模型文件路径
    'onnx_path': "~/your_path/results/result-3d.onnx",               # 用于推理的模型文件路径
    'n_classes': 101,                                                              # 数据集类别数
    'sample_size': 112,                                                            # 图片分辨率
    'sample_duration': 16,                                                         # 视频片段长度，单位：帧
    'sample_t_stride': 1,                                                          # 裁剪视频帧距离
    'train_crop': 'center',                                                        # 视频片段裁剪方式
    'h_flip': False,  # Use random_horizontal_flip, default:False                  # 是否使用随机水平翻转
    'colorjitter': False,                                                          # 是否使用色彩增强
    'train_crop_min_scale': 0.25,                                                  # 随机裁剪比例
    'train_crop_min_ratio': 0.75,                                                  # 随机裁剪率
    'train_t_crop': 'random',
    'inference_stride': 16,                                                        # 推理时视频片段长度
    'ignore': True,                                                                # 是否忽略标注中没有的类别
    'start_ft': 'layer4',  # choices = [conv1, layer1, layer2, layer3, layer4, fc] # 模型微调时的起始模块
    'loss_scale': 1024,                                                            # 损失等级
    'momentum': 0.9,                                                               # 动量优化器
    'weight_decay': 0.001,                                                         # 权重衰减
    'batch_size': 128,                                                             # 输入张量的批次大小
    'n_epochs': 200,                                                               # 训练轮次
    'save_checkpoint_epochs': 1,                                                   # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
    'keep_checkpoint_max': 20,                                                     # 只保存最后一个keep_checkpoint_max检查点
    'lr_decay_mode': 'steps',  # choices = [poly, linear, cosine, steps]           # 用于生成学习率的衰减模式
    'warmup_epochs': 15,                                                           # 热身周期数
    'lr_init': 0,                                                                  # 初始学习率
    'lr_max': 0.03,                                                                # 最大学习率
    'lr_end': 0,                                                                   # 最小学习率
    'eval_in_training': False                                                      # 是否边训练边推理
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [DEVIDE_NUM] [RANK_TABLE_FILE] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]

# 单机训练
用法：bash run_standalone_train.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [PRETRAIN_PATH]

```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

### 结果

- 使用hmdb51数据集训练resnet3d

```text
# 分布式训练结果（8P）
epoch: 1 step: 1, loss is 3.9611845
epoch: 1 step: 2, loss is 3.9543657
epoch: 1 step: 3, loss is 3.9457293
epoch time: 143772.346 ms, per step time: 47924.115 ms
epoch: 2 step: 1, loss is 3.9611716
epoch: 2 step: 2, loss is 3.9526463
epoch: 2 step: 3, loss is 3.9503071
epoch time: 10481.911 ms, per step time: 3493.970 ms
epoch: 3 step: 1, loss is 3.8921037
epoch: 3 step: 2, loss is 3.8538368
epoch: 3 step: 3, loss is 3.8057396
epoch time: 13893.360 ms, per step time: 4631.120 ms
epoch: 4 step: 1, loss is 3.747679
...
```

- 使用ucf101数据集训练resnet3d

```text
# 分布式训练结果（8P）
epoch: 23 step: 6, loss is 0.88092417
epoch: 23 step: 7, loss is 0.89366925
epoch: 23 step: 8, loss is 0.8902888
epoch: 23 step: 9, loss is 0.9039843
epoch time: 33055.406 ms, per step time: 3672.823 ms
epoch: 24 step: 1, loss is 0.87126017
epoch: 24 step: 2, loss is 0.900363
epoch: 24 step: 3, loss is 0.91156125
epoch: 24 step: 4, loss is 0.89792097
epoch: 24 step: 5, loss is 0.92281365

...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 评估
用法：bash run_eval.sh [DEVICE_ID] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [RESULT_PATH] [INFERENCE_CKPT_PATH]
```

```bash
# 评估示例
bash run_eval.sh 0 ucf101 \
~/dataset/ucf101/jpg/ \
~/dataset/ucf101/json/ucf101_01.json \
~/results/ \
~/results/ucf101
```

### 结果

评估结果保存在示例路径中，文件名为“~/datasetname_eval.log”。您可在此路径下的日志找到如下结果：

- 使用hmdb51数据集评估resnet3d(8P)

```text
clip: 66.5% top-1: 69.7%  top-5: 93.8%
```

- 使用ucf101数据集评估resnet3d(8P)

```text
clip: 88.8% top-1: 92.7%  top-5: 99.3%
```

## ONNX评估

### 导出onnx模型

```bash
python export.py --ckpt_file=/path/best.ckpt --file_format=ONNX --n_classes=51 --batch_size=1 --device_target=GPU
 ```

- `ckpt_file` ckpt文件路径
- `file_format` 导出模型格式，此处为ONNX
- `n_classes` 使用数据集类别数，hmdb51数据集此参数为51，ucf101数据集此参数为101
- `batch_size` 批次数，固定为1
- `device_target` 目前仅支持GPU或CPU

### 运行ONNX模型评估

```bash
用法：bash run_eval_onnx.sh [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [ONNX_PATH]
实例：bash run_eval_onnx.sh ucf101 /path/ucf101/jpg /path/ucf101/json/ucf101_01.json /path/resnet-3d.onnx
 ```

- `[ucf101|hmdb51]` 选择所使用的数据集
- `[VIDEO_PATH]` 视频路径
- `[ANNOTATION_PATH]` 标签路径
- `[ONNX_PATH]` onnx模型的路径

### 结果

评估结果保存在示例路径中，文件名为“~/eval_onnx.log”。您可在此路径下的日志找到如下结果：

- 使用hmdb51数据集评估resnet3d

```text
clip: 66.5% top-1: 69.7%  top-5: 93.8%
```

- 使用ucf101数据集评估resnet3d

```text
clip: 88.8% top-1: 92.7%  top-5: 99.3%
```

## 导出过程

### 导出

在导出时，hmdb51数据集,参数n_classes设置为51,ucf101数据集,参数n_classes设置为101, 参数batch_size只能设置为1.

```shell
python export.py --ckpt_file=./saved_model/best.ckpt --file_format=MINDIR --n_classes=51 --batch_size=1
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用hmdb51数据集或者ucf101数据集进行推理

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  sh run_infer_310.sh [MINDIR_PATH] [ucf101|hmdb51] [VIDEO_PATH] [ANNOTATION_PATH] [NEED_PREPROCESS] [DEVICE_ID]
  clip: 66.5% top-1: 69.7%  top-5: 93.8%
  ```

# 模型描述

## 性能

### 评估性能

#### UCF101上的resnet3d

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | resnet3d
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2021-09-23 |
| MindSpore版本          | r1.3 |
| 数据集                    | UCF101 |
| 训练参数        | epoch=200, steps per epoch=9, batch_size = 128 |
| 优化器                  | SGD |
| 损失函数              | Softmax交叉熵 |
| 输出                    | 概率 |
| 损失                       | 1.0250647 |
| 速度                      | 3730.840毫秒/步（8卡）|
| 总时长                 | 1.7小时 |
| 参数(M)             | 210M |
| 微调检查点| 401M（.ckpt文件）|
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/resnet3d) |

#### hmdb51上的resnet3d

| 参数                 | Ascend 910
| -------------------------- | -------------------------------------- |
| 模型版本              | resnet3d
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755G |
| 上传日期              | 2021-09-23 |
| MindSpore版本          | r1.3 |
| 数据集                    | hmdb51 |
| 训练参数        | epoch=200, steps per epoch=3, batch_size = 128 |
| 优化器                  | SGD |
| 损失函数              | Softmax交叉熵 |
| 输出                    | 概率 |
| 损失                       | 0.823 |
| 速度                      | 3730.933毫秒/步（8卡）|
| 总时长                 | 1.1小时 |
| 参数(M)             | 210M |
| 微调检查点| 401M（.ckpt文件）|
| 脚本                    | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/resnet3d) |

# 随机情况说明

使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models/)。
