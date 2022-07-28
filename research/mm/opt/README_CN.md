# OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation

> 该项目是OPT模型的实现，一个图-文-音三模态预训练模型。

## OPT描述

OPT是一个基于编码器和解码器构建的模型，包括三个模态的编码器来生成每个模态的编码，一个跨模态编码器来编码各个 模态之间的关系，和两个跨模态的解码器来生成文本和图像。

## 模型整体结构

<img src="./8159685ccda2be63fd92cb1109fe7f8.png" alt="image-20211117104252504" />

## 环境要求

- 硬件（Ascend/GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 预训练数据集

- [CC3M](https://github.com/google-research-datasets/conceptual-captions) 提供了大约300万的图像-文本对，在此项目中我们将英文翻译成中文。
- [COCO Captions](https://cocodataset.org/#home) 提供了大约 415K 的图像-文本对，在此项目中我们将英文翻译成中文。
- AIC提供了大约100万的中文图像-文本对。

## 预训练

```shell
bash run_distributed_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

## 下游任务finetune

### 跨模态检索

- 数据集: [COCO Captions](https://cocodataset.org/#home) 提供了大约 415K 的图像-文本对，在此项目中我们将英文翻译成中文。

- 训练:

```shell
bash scripts/run_standalone_ascend_train_retrieval.sh
```

- 推理:

```shell
bash scripts/run_standalone_ascend_inf_retrieval.sh
```

### 图像到文本生成

- 数据集: [COCO Captions](https://cocodataset.org/#home) 提供了大约 415K 的图像-文本对，在此项目中我们将英文翻译成中文。

- 训练:

```shell
bash scripts/train_caption.sh
```

- 推理:

```shell
bash scripts/test_caption.sh
```

### 结果