# 目录

- [目录](#目录)
- [Cross-Scale-Non-Local-Attention描述](#Cross-Scale-Non-Local-Attention描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [DIV2K上的CSNL](#DIV2K上的CSNL)

# Cross-Scale-Non-Local-Attention描述

CSNLN是2020年提出的图像超分辨率神经网络，基于深度学习的图像超分取得了前所未有的进展，但这些方法往往受益于网络的更深、感受野的更宽。图像块的非局部相似性是图像的一种基本先验信息，而这却很少在深度学习方面得到探索与应用。尽管已有相关方法尝试采用非局部注意力机制进行图像超分，但跨尺度相似性却并未受到关注。
在该文中，作者将跨尺度特征相关性先验信息纳入到深度学习中并嵌入到递归神经网络中。它通过递归单元组合所提跨尺度非局部注意力机制与单尺度非局部注意力。通过组合上述先验信息，所提图像超分在多个公开数据集上取得了SOTA性能。

[论文](https://arxiv.org/abs/2006.01424) “Image Super-Resolution With Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining.”

## 模型架构

CSNLN由多个样本挖掘模块（Self-Examplars-Mining）串联组成。模块中同时使用了Cross-Scale、In-Scale和local branch三个尺度的信息，并在出口处将不同特征进行融合。

## 数据集

使用的数据集：[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- 数据集大小：
    - 训练集：800张图像
  - 测试集：100张图像
  - 验证集：100张图像
- 数据格式：RGB
  - 注：数据将在src/dataset/dataset.py中处理。

使用的其他数据集：

- Set5
- Set14
- B100
- Urban100
- 数据格式：RGB
- 测试集下载链接：[benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)
    - 注：数据将在src/dataset/dataset.py中处理。

数据集可以存放在任意文件夹，只需依照规定结构组织即可。数据集路径通过`default_config.yaml`文件中的`data_path`选项配置，使用的训练集和测试集通过`data_train`选项和`data_test`选项指定.

数据集组织示例如下，`DIV2K`为模型使用的训练集，`benchmark`中存放所使用的测试集

```text
|--dataset
  |--benchmark
    |--Set14
    |--B100
    |--Set5
      |--HR
      |--LR_bicubic
        |--X2
        |--X3
  |--DIV2K
    |--DIV2K_train_HR
      |--0001.png
      |--0002.png
      ...
    |--DIV2k_train_LR_bicubic
      |--X2
        |--0001x2.png
        |--0002x2.png
        ...
      |--X3
      |--X4
```

其中，训练集和测试集中lr图片命名方式为`对应hr图片名+x+下采样比例`，如`0001x2.png`对应hr图片中`0001.png`。

配置`default_config.yaml`文件时，`data_path`选项为示例中dataset文件夹的绝对路径，`data_train`和`data_test`选项为所使用数据集的名称。

以上图为例，`default_config.yaml`文件配置如下

```yaml
data_path: "/absolute/path/dataset"
data_train: "DIV2K"
data_test: "Set5"
```

## 特性

### 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

```bash
# Train demo
python train.py > train.log 2>&1 &
OR
bash scripts/run_train_gpu.sh 1 [device_number]
# Distributed Train demo
bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

# Eval demo
python eval.py --ckpt_file=[CHECKPOINT_PATH] > eval.log 2>&1 &
OR
bash script/run_eval_gpu.sh [CHECKPOINT_PATH] [DATASET]
```

## 脚本说明

### 脚本及样例代码

```text
├── model_zoo
    ├── README.md                       // 所有模型相关说明
    ├── CSNL
        ├── README.md                   // CSNL相关说明
        ├── scripts
        │   ├──run_train_gpu.sh         // 分布式到GPU处理器的shell脚本
        │   ├──run_eval_gpu.sh          // GPU处理器评估的shell脚本
        ├── src
        │   ├──dataset                  // 创建数据集
        │   │   ├──benchmark.py         // 测试集数据处理
        │   │   ├──common.py            // 通用图像处理函数
        │   │   ├──dataset.py           // 构建训练集
        │   │   ├──div2k.py             // div2k数据集处理
        │   │   ├──srdata.py            // 主要数据处理脚本
        │   ├──CSNLN.py                 //  CSNLN架构
        │   ├──config.py                // 参数配置
        │   ├──public.py                // 复用的网络层结构
        │   ├──utils.py                 // 其他辅助功能
        ├── train.py                    // 训练脚本
        ├── eval.py                     // 评估脚本
        ├── default_config.yaml         // 模型参数配置
```

## 模型描述

### 性能

#### 评估性能

##### DIV2K上的CSNL

| 参数          | GPU                                     |
| ------------- | --------------------------------------- |
| 上传日期      | 2022-08-03                              |
| Mindspore版本 | 1.5.0                                   |
| 数据集        | DIV2K                                   |
| 训练参数      | epoch=500，patch_size=48，batch_size=16 |
| 优化器        | Adam                                    |
| 损失函数      | L1 Loss                                 |
| 输出          | loss                                    |
| 损失          | 0.013844                                |
| 速度          | 1500ms/step(8p)                         |
| 总时长        | 1562 分钟                               |
| 参数          | 3.06M                                   |