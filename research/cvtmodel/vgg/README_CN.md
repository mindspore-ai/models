# 目录

<!-- TOC -->

- [目录](#目录)
- [VGG描述](#vgg描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RegNet描述

VGG是一个基于卷积的神经网络，用于图像分类。有关该模型的描述，可查阅[此论文](https://arxiv.org/abs/1409.1556)。
本仓库中是基于提供的模型文件，使用MindConverter工具转化出Mindspore框架内的ckpt文件，进行全量推理以验证模型文件精度。
模型参照网址[VGG](https://pytorch.org/hub/pytorch_vision_vgg)。

# 模型架构

VGG模型支持模式包括：vgg11，vgg11_bn，vgg13，vgg13_bn，vgg16_bn，vgg19_bn。

# 数据集

VGG使用的数据集： ImageNet

数据集的默认配置如下：

- 测试数据集预处理：
    - 图像的输入尺寸：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 根据平均值和标准偏差对输入图像进行归一化

# 环境要求

- 硬件（Ascend/GPU）
- 准备Ascend或GPU处理器搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```shell
├── model_zoo
    ├── README.md                                // 所有模型的说明
    ├── vgg
        ├── README_CN.md                         // regnet说明文件
        ├── ascend310_infer                      // 实现310推理源代码
        |   ├── CMakeLists.txt
        |   ├── build.sh
        |   ├── inc
        |   |   ├── utils.h
        |   ├── src
        |       ├── main.cc
        |       ├── utils.cc
        ├── export.py                             // 模型导出脚本
        ├── imagenet2012_config.yaml              // 配置项文件
        ├── model_utils                           // 配置项处理脚本文件夹
        |   ├── __init__.py
        |   ├── config.py
        |   ├── device_adapter.py
        |   ├── local_adapter.py
        |   ├── moxing_adapter.py
        ├── postprocess.py                        // 推理结果精度计算脚本
        ├── preprocess.py                         // 数据预处理脚本
        ├── scripts
        |   ├── run_infer_310.sh                  // Ascend 310 推理shell脚本
        ├── src
            ├── __init__.py
            ├── vgg11.py                    // vgg11模型文件
            ├── vgg11_bn.py                 // vgg11_bn模型文件
            ├── vgg13.py                    // vgg13模型文件
            ├── vgg13_bn.py                 // vgg13_bn模型文件
            ├── vgg16_bn.py                 // vgg16_bn模型文件
            ├── vgg19_bn.py                 // vgg19_bn模型文件
```

## 导出过程

### 导出

```shell
python export.py --backbone [NET_NAME] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT] --file_name [FILE_NAME]
```

- `backbone` 可选 ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
- `DEVICE_TARGET` 可选["Ascend", "GPU"]
- `EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

## 推理过程

### 推理

在推理之前需要先通过`export.py`导出MINDIR模型。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` 通过`export.py`导出MINDIR模型。
- `DATASET_PATH` 需要模型进行推理的数据集路径，本示例代码使用ImageNet数据集。
- `NEED_PREPROCESS` 根据ImageNet数据集的文件夹分类生成label标签。
- `DEVICE_ID` Ascend310执行推理的芯片卡号，默认值为0。

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
`vgg11`推理结果如下:

  ```log
Top1 acc:  0.6877
Top5 acc:  0.885
  ```

`vgg11_bn`推理结果如下:

  ```log
Top1 acc: 0.70102
Top5 acc:  0.8963
  ```

`vgg13`推理结果如下:

  ```log
Top1 acc:  0.69576
Top5 acc:  0.89132
  ```

`vgg13_bn`推理结果如下:

  ```log
Top1 acc:  0.71176
Top5 acc:  0.90152
  ```

`vgg16_bn`推理结果如下:

  ```log
Top1 acc:  0.73042
Top5 acc:  0.91358
  ```

`vgg19_bn`推理结果如下:

  ```log
Top1 acc:  0.74026
Top5 acc:  0.9174
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。  
