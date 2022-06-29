# 目录

<!-- TOC -->

- [SRFlow描述](#SRFlow描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [训练](#训练)
    - [训练过程](#训练过程)
    - [训练参数](#训练参数)
- [推理](#推理)
    - [推理过程](#推理过程)
    - [推理结果](#推理结果)
- [性能](#性能)
    - [推理性能](#推理性能)

<TOC>

# SRFlow描述

> SRFlow是一个基于Glow模型所开发的用于超分辨率的模型，SRFlow是流模型的一种变体。其先通过ResNet对图像进行处理，再用流模型进行训练。模型通过将放缩的图像和原图像作为有监督学习的输入对模型进行训练，最后在推理的时候输入低分辨率图像并生成高分辨率图像。
本项目的主要目标是使用MindSpore框架复现SRFlow网络，并在GPU环境下训练以达到与论文相近的性能表现。
本项目的参考论文为SRFlow: Learning the Super-Resolution Space with Normalizing Flow，原版PyTorch开源实现为https://github.com/andreas128/SRFlow。

# 模型架构

> 网络主要由Glow和ResNet这两部分模型组成。

# 数据集

> 模型使用的数据集是DIV2K和Fliker2K混合的数据集，其中包括160*160的原像素图片和40*40的放缩后的图片。

数据集下载地址： 链接：https://pan.baidu.com/s/138ZkF38fhrjg3ZBqR8C3eQ 提取码：pjt5

下载的数据集需要存放在SRFlow/src路径下

模型的参数由包含用于训练的预训练参数和已经训练好用于推理的参数

下载的参数需要存放在SRFlow路径下

参数下载地址： 链接：https://pan.baidu.com/s/1jmGuhIwSB6m0Eap4q2nm7A 提取码：8sty

链接：https://pan.baidu.com/s/1EqvBc9UlvZAVYbCE4RcAsg 提取码：jmas

下载好之后，进行解压缩：

```shell
cd /src/dataset/DF2K_4X/train/
unzip DIV2K_X4_train.zip
```

# 环境要求

> 本实验的官方库为mindspore=1.7.0

本实验的依赖库详见requirements.txt

# 快速入门

> 安装以上依赖后，可通过如下方式快速启动

```shell
训练：
bash ./scripts/run_standalone_train_gpu.sh SRFlow_DF2K_4X.yml 0
推理：
bash ./scripts/run_standalone_eval_gpu.sh SRFlow_DF2K_4X.yml 0
```

## 脚本说明

```text
.
└─ SRFlow
  ├── scripts                                            // 脚本
  │   ├── run_standalone_eval_gpu.sh                     // 推理脚本
  │   ├── run_standalone_train_gpu.sh                    // 训练脚本
  │    ...
  ├── src
  │   ├── dataloader                                     // 数据集加载
  |   |   ├── init.py
  │   ├── dataset                                        // 数据集
  │   |   ├── DF2K-4X
  │   |   |  ├── example                                  // 样例图片
  │   |   |  |  ├── 0801_lr_X4.png
  │   |   |  |  ├── 0801_sr.png
  │   |   |  ├── train                                    // 训练集（mindrecord格式）
  │   |   |  |  ├── DIV2K_X4_train.mindrecord
  │   |   |  |  ├── DIV2K_X4_train.mindrecord.db
  │   |   |  ├── valid                                    // 验证集（mindrecord格式）
  │   |   |  |  ├── DIV2K_X4_valid.mindrecord
  │   |   |  |  ├── DIV2K_X4_valid.mindrecord.db
  │   |   |  |   ...
  │   ├── model                                           // 模型主体
  │   |   ├── Flow.py
  │   |   ├── FlowActNorms.py
  │   |   ├── FlowAffineCouplingsAblation.py
  │   |   ├── FlowStep.py
  │   |   ├── FlowUpsamplerNet.py
  │   |   ├── InvertibleConv1x1.py
  │   |   ├── RRDBNet.py
  │   |   ├── Split.py
  │   |   ├── SRFlow.py
  │   |   ├── SRFlowNet.py
  │   ├── model_utils                                      // 工具库
  │   |   ├── __init__.py
  │   |   ├── config.py
  │   |   ├── device_adapter.py
  │   |   ├── local_adapter.py
  │   |   ├── moxing_adapter.py
  │   |   ├── options.py
  │   |   ├── util.py
  │   ├── scheduler                                         // 学习率调整
  │   |   ├── scheduler.py
  ├── DF2K_4X_test.ckpt                                     // 测试参数
  ├── DF2K_4X_train.ckpt                                    // 训练参数
  ├── eval.py                                               // 推理
  ├── SRFlow_DF2K_4X.yml                                    // 训练配置
  ├── train.py                                              // 训练
  ├── train2test.py                                         // 将训练完的参数转换成用于推理的参数
```

# 训练

本实验训练部分缺少用于计算loss的算子，导致训练出的结果异常，推理的结果良好。

## 训练过程

因为缺少slogdet算子，目前训练暂时不提供

缺少slogdet算子的路径和代码片段：

```shell
cd ./src/model/
vim InvertibleConv1x1.py
```

```python
def construct(self, x, logdet=None):
    conv2d = ops.Conv2D(out_channel=self.num_channels, kernel_size=(1, 1), stride=1)
    reshape = ops.Reshape()
    weight = reshape(self.weight, (self.num_channels, self.num_channels, 1, 1))
    z = conv2d(x, weight)
    pixels = x[2] * x[3]
    dlogdet = torch.slogdet(self.weight)    # 需要添加slogdet的操作
    logdet += dlogdet[1] * pixels  
    return z, logdet
```

待算子补齐后，可以在单GPU上运行下面的命令：

```shell
bash ./scripts/run_standalone_train_gpu.sh [config_path] [device_id]
```

## 训练参数

mindspore训练参数由pytorch的初始化参数转换过来，所有参数都需要进行训练。

# 推理

## 推理过程

在单GPU上运行下面的命令

推理参数下载地址： 链接：https://pan.baidu.com/s/1jmGuhIwSB6m0Eap4q2nm7A 提取码：8sty

```shell
bash ./scripts/run_standalone_eval_gpu.sh [config_path] [device_id]
```

### 推理结果

```shell
Get the number 4691 data psnrloss: [34.00094], ssim_loss: [0.9170928]
Get the number 4692 data psnrloss: [29.821362], ssim_loss: [0.6185117]
Get the number 4693 data psnrloss: [35.297928], ssim_loss: [0.8019606]
Get the number 4694 data psnrloss: [30.294598], ssim_loss: [0.87265956]
Get the number 4695 data psnrloss: [21.80051], ssim_loss: [0.6229686]
Get the number 4696 data psnrloss: [29.517988], ssim_loss: [0.8870666]
Get the number 4697 data psnrloss: [15.776173], ssim_loss: [0.68317175]
Get the number 4698 data psnrloss: [25.705627], ssim_loss: [0.88152087]
Get the number 4699 data psnrloss: [31.076763], ssim_loss: [0.91741335]
Get the number 4700 data psnrloss: [32.79779], ssim_loss: [0.9525596]
The mean of psnr is: [28.495592]
The mean of ssim is: [0.76350427]
The mean of psnr is: [28.495592]
The mean of ssim is: [0.76350427]
```

# 性能

## 推理性能

你可以参照如下模板

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | SRFlow                      |
| Resource            | RTX 3090                    |
| Uploaded Date       | 06/24/2021 (month/day/year) |
| MindSpore Version   | 1.7.0                       |
| Dataset             | DF2K-4X                     |
| batch_size          | 1                           |
| outputs             | probability                 |
| Accuracy            | 28.49                       |
