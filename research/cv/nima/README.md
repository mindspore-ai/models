# 目录

<!-- TOC -->

- [目录](#目录)
- [ResNet描述](#resnet描述)
    - [概述](#概述)
    - [论文](#论文)
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
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [运行参数服务器模式训练](#运行参数服务器模式训练)
            - [训练时推理](#训练时推理)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
        - [结果](#结果-1)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [AVA_Dataset上的ResNet50](#cifar-10上的resnet50)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ResNet描述

## 概述

残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。

如下为MindSpore使用AVA_Dataset数据集对ResNet50进行训练的示例。ResNet50可参考[论文1](https://arxiv.org/pdf/1512.03385.pdf)。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf)：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"

2. [论文](https://arxiv.org/abs/1709.05424)：H. Talebi and P. Milanfar, "NIMA: Neural Image Assessment"

# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

预训练模型：
[链接](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44/)

# 数据集

## 下载数据集, 并划分训练集与测试集

使用的数据集：[AVA_Dataset](<https://github.com/mtobeiyf/ava_downloader/tree/master/AVA_dataset>)

使用label：[AVA.txt](https://github.com/mtobeiyf/ava_downloader/blob/master/AVA_dataset/AVA.txt)

准备好数据，执行下面python命令划分数据集

```text
python ./src/dividing_label.py --config_path=~/config.yaml
#更改配置文件：data_path、label_path、val_label_path、train_label_path
```

- 数据集大小：255,502张彩色图像
    - 训练集：229,905张图像
    - 测试集：25,597张图像
- 数据格式：JEPG图像

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```text
# 运行训练示例
python train.py --config_path=./config.yaml >train.log 2>&1 &

# 分布式训练
bash ./scripts/run_train_ascend.sh ~/hccl_8p.json

# 运行评估示例
python eval.py --config_path ./config.yaml >eval.log 2>&1 &

```

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 在 config.yaml 文件中设置 "enable_modelarts=True","is_distributed=True"，并设置其他参数，
#     如：data_path、output_path、train_data_path、val_data_path、checkpoint_path等。
# (2) 在modelarts的界面上设置代码目录"~/NIMA/"。
# (3) 在modelarts的界面上设置模型的启动文件 "~/NIMA/train.py" 。
# (4) 在modelarts的界面上添加运行参数 config_path = "~/NIMA/config.yaml"
# (5) 在modelarts的界面上设置模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 在 config.yaml 文件中设置 "enable_modelarts=True"，并设置如下参数：
#     data_path、val_data_path、ckpt_file
# (3) 在modelarts的界面上设置代码目录"~/NIMA/"。
# (4) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (5) 在modelarts的界面上添加运行参数 config_path = "~/config.yaml"
# (6) 在modelarts的界面上设置模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

# 脚本说明

## 脚本及样例代码

```shell
.
├──NIMA
  ├── README.md                 #相关说明
  ├──ascend310_infer            #实现310推理源代码
  ├──model                      #预训练模型
    ├──ascend.ckpt
  ├──scripts
    ├──run_eval.sh              #910评估shell脚本
    ├──run_infer_310.sh         #310推理shell脚本
    ├──run_train_ascend.sh      #910训练shell脚本
  ├──src
    ├──resnet.py                #主干网络架构
    ├──test_data.py             #生成310推理数据集
    ├──config.py                #参数配置
    ├──device_adapter.py        #设备适配
    ├──dividing_label.py        #划分数据集
    ├──callback.py              #回调
    ├──dataset.py               #数据处理
    ├──metric.py                #损失及指标
  ├──eval.py                    #评估脚本
  ├──export.py                  #将checkpoint文件导出到mindir
  ├──postprocess.py             #310推理后处理脚本
  ├──train.py                   #训练脚本
  ├──AVA_train.txt              #训练集label
  ├──AVA_test.txt               #测试集label

```

## 脚本参数

```python
"device_target": "Ascend"               #运行设备
"batch_size": 256                       #训练批次大小
"epoch_size": 50                        #总计训练epoch数
"num_parallel_workers": 16              #进程数
"learning_rate": 0.001                  #学习率
"momentum": 0.95                        #动量
"weight_decay": 0.001                   #权值衰减值
"bf_crop_size": 256                     #裁剪前图片大小
"image_size": 224                       #实际送入网络图片大小
"train_label_path": "AVA_train.txt"     #训练集绝对路径
"val_label_path": "AVA_test.txt"        #测试集绝对路径
"keep_checkpoint_max": 10               #保存 checkpoint 的最大数量
"checkpoint_path": "./resnet50.ckpt"    #预训练模型的绝对路径
"ckpt_save_dir": "./ckpt/"              #模型保存路径
"is_distributed": False                 #是否分布式训练，默认False
"enable_modelarts": False               #是否使用modelarts训练，默认False
"output_path": "./"                     #modelarts训练时，将ckpt_save_dir文件复制到桶

```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 单机训练
python train.py --config_path=./config.yaml >train.log
```

可指定`config.yaml`中的`device_id`

运行上述python命令后，您可以通过`train.log`文件查看结果

```text
# 分布式训练
Usage：bash scripts/run_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
#example: bash ./scripts/run_train_ascend.sh ~/hccl_8p.json ~/config.yaml
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果。

运行单卡用例时如果想更换运行卡号，可以通过配置环境中设置`device_id=x`或者在context中设置 `device_id=x`指定相应的卡号。

### 结果

```text
# 分布式训练结果（8P）
epoch: 1 step: 898, loss is 0.08514725
epoch: 2 step: 898, loss is 0.072653964
epoch: 3 step: 898, loss is 0.06939027
epoch: 4 step: 898, loss is 0.087793864
epoch: 5 step: 898, loss is 0.08969345
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```text
# 运行评估示例
Usage：bash run_eval.sh [CONFIG_PATH]
#example: bash scripts/run_eval.sh config.yaml >export.log
```

更改配置文件`config.yaml`中`data_path`、`val_data_path`、`ckpt_file`即可

### 结果

评估结果保存在示例文件`eval.log`中。您可在此文件中找到的日志找到如下结果：

```bash
SRCC: 0.657146300995645
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#contents)

数据准备

```shell
python ./src/test_data.py --config_path=config.yaml
```

确保`data_path`、`val_data_path`路径正确
执行该命令后会按照 `val_data_path` 生成310推理的数据集

导出mindir模型

```shell
python export.py --config_path=config.yaml >export.log
```

更改`ckpt_file`、`file_name`即可

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```shell
# Ascend310 inference
bash ./scripts/run_infer_310.sh [MODEL_PATH] [VAL_DATA_PATH] [DEVICE_ID]
# example: bash ./scripts/run_infer_310.sh  ~/model/NIMA.mindir ~/test_data/ 0
```

- `DEVICE_ID` 可选，默认值为0。

### 结果

```shell
python ./postprocess.py --config_path=config.yaml &> acc.log
```

推理结果保存在脚本执行的当前路径，你可以在`acc.log`中看到以下精度计算结果。

```shell
cat acc.log

SRCC: 0.6571463000995645.
```

# 模型描述

## 性能

### 评估性能

#### AVA_Dataset上的ResNet50

| 参数                   | Ascend 910                             |
| ---------------------- | -------------------------------------- |
| 模型版本               | ResNet50                               |
| 资源                   |  Ascend 910；CPU 2.60GHz，192核；内存 720G；系统 Euler2.8 |
| 上传日期               | 2021-11-19  ;                          |
| MindSpore版本          | 1.3.0                                  |
| 数据集                 | AVA_Dataset                            |
| 训练参数               | epoch=50, steps per epoch=898, batch_size = 256|
| 优化器                 | SGD                                    |
| 损失函数               | EmdLoss(推土机距离)                    |
| 输出                   | 概率分布                               |
| 损失                   | 0.05819133                             |
| 速度                   | 356毫秒/步（8卡）                      |
| 总时长                 | 174分钟                                |
| 参数(M)                | 25.57                                  |
| 微调检查点             | 195M（.ckpt文件）                      |
| 配置文件               | [链接](https://gitee.com/mindspore/models/blob/master/research/cv/nima/config.yaml) |

# 随机情况说明

`dividing_label.py`中设置了random.seed(10)，`train.py`中同样设置了set_seed(10)。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)
