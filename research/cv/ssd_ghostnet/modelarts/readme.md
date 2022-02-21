# SSD_GhostNet-Ascend (目标检测/MindSpore)

## 1.概述

SSD GhostNet 将边界框的输出空间离散为一组默认框，每个特征地图位置的纵横比和比例不同。在预测时，网络为每个默认框中每个对象类别 生成分数，并对该框进行调整，以更好地匹配对象形状。此外，该网络结合了来自不同分辨率的多个特征图的预测，从而自然地处理不同尺寸的物体。

## 2.训练

### 2.1.算法基本信息

- 任务类型: 目标检测
- 支持的框架引擎: Ascend-Powered-Engine- Mindspore-1.1.1-python3.7-aarch64
- 算法输入:
    - obs数据集路径，下面存放使用coco2017数据集。数据集的格式见训练手册说明。
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明

名称|默认值|类型|是否必填|描述
---|---|---|---|---
lr|0.05|float|True|初始学习率
dataset|coco|string|True|数据集格式，可选值coco、voc、other
epoch_size|500|int|True|训练轮数
batch_size|32|int|True|一次训练所抓取的数据样本数量
save_checkpoint_epochs|10|int|False|保存checkpoint的轮数。
num_classes|81|string|True|数据集类别数+1。
voc_json|-|string|False|dataset为voc时，用于指定数据集标注文件，填相对于data_url的路径。
anno_path|-|string|False|dataset为other时，用于指定数据集标注文件，填相对于data_url的路径。
pre_trained|-|string|False|迁移学习时，预训练模型路径，模型放在data_url下，填相对于data_url的路径。
loss_scale|1024|int|False|Loss scale.
filter_weight|False|Boolean|False|Filter head weight parameters，迁移学习时需要设置为True。

### 2.3. 训练输出文件

训练完成后的输出文件如下

```json
训练输出目录 V000X
├── ckpt_0
│   ├── ssd-2_4.ckpt
│   ├── ssd-3_4.ckpt
│   └── ssd-graph.meta
├── kernel_meta
│   ├── ApplyMomentum_13796921261177776697_0.info
│   ├── AddN_4688903218960634315_0.json
│   ├── ...
```

## 3.迁移学习指导

### 3.1. 数据集准备：

参考训练手册：`迁移学习指导`->`数据集准备`

### 3.2. 上传预训练模型ckpt文件到obs数据目录pretrain_model中，示例如下：

```json
MicrocontrollerDetection              # obs数据目录
  |- train                            # 训练图片数据集目录
        |- IMG_20181228_102033.jpg
        |- IMG_20181228_102041.jpg
        |- ..。
  |- train_labels.txt                 # 训练图片数据标注
  |- pretrain_model
        |- ssd-3-61.ckpt              # 预训练模型 ckpt文件
```

classes_label_path参数对应的train_labels.txt的内容如下所示：

```json
background
Arduino Nano
ESP8266
Raspberry Pi 3
Heltec ESP32 Lora
```

### 3.3. 修改调优参数

目前迁移学习支持修改数据集类别，订阅算法创建训练任务，创建训练作业时需要修改如下调优参数：

1. dataset改为other。
2. num_classes改为迁移学习数据集的类别数+1。
3. anno_path指定迁移学习数据集的标注文件路径。
4. filter_weight改为True。
5. pre_trained指定预训练模型路径。

以上参数的说明见`训练参数说明`。

### 3.4. 创建训练作业

指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。