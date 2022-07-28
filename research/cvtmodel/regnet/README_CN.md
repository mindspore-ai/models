# 目录

<!-- TOC -->

- [目录](#目录)
- [RegNet描述](#regnet描述)
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

RegNet是一个基于卷积的神经网络，用于图像分类。有关该模型的描述，可查阅[此论文](https://arxiv.org/abs/2003.13678)。
本仓库中是基于提供的模型文件，使用MindConverter工具转化出Mindspore框架内的ckpt文件，进行全量推理以验证模型文件精度。
模型参照网址[RegNet](https://pytorch.org/vision/stable/_modules/torchvision/models/regnet.html)。

# 模型架构

RegNet模型支持模式包括：regnet_x_16gf，regnet_x_1_6gf，regnet_x_32gf，regnet_x_3_2gf，regnet_x_400mf，regnet_x_800mf，regnet_x_8gf，regnet_y_16gf，regnet_y_1_6gf，regnet_y_32gf，regnet_y_3_2gf，regnet_y_400mf，regnet_y_800mf，regnet_y_8gf。

# 数据集

RegNet使用的数据集： ImageNet

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
    ├── regnet
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
            ├── regnet_x_16gf.py                  // regnet_x_16gf模型文件
            ├── regnet_x_1_6gf.py                 // regnet_x_1_6gf模型文件
            ├── regnet_x_32gf.py                  // regnet_x_32gf模型文件
            ├── regnet_x_3_2gf.py                 // regnet_x_3_2gf模型文件
            ├── regnet_x_400mf.py                 // regnet_x_400mf模型文件
            ├── regnet_x_800mf.py                 // regnet_x_800mf模型文件
            ├── regnet_x_8gf.py                   // regnet_x_8gf模型文件
            ├── regnet_y_16gf.py                  // regnet_y_16gf模型文件
            ├── regnet_y_1_6gf.py                 // regnet_y_1_6gf模型文件
            ├── regnet_y_32gf.py                  // regnet_y_32gf模型文件
            ├── regnet_y_3_2gf.py                 // regnet_y_3_2gf模型文件
            ├── regnet_y_400mf.py                 // regnet_y_400mf模型文件
            ├── regnet_y_800mf.py                 // regnet_y_800mf模型文件
            ├── regnet_y_8gf.py                   // regnet_y_8gf模型文件
```

## 导出过程

### 导出

```shell
python export.py --backbone [NET_NAME] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT] --file_name [FILE_NAME]
```

- `backbone` 可选 ["regnet_x_16gf", "regnet_x_1_6gf", "regnet_x_32gf", "regnet_x_3_2gf", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_8gf", "regnet_y_16gf", "regnet_y_1_6gf", "regnet_y_32gf", "regnet_y_3_2gf", "regnet_y_400mf", "regnet_y_800mf", "regnet_y_8gf"]
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
`regnet_y_400mf`推理结果如下:

  ```log
Top1 acc:  0.73812
Top5 acc:  0.91694
  ```

`regnet_y_800mf`推理结果如下:

  ```log
Top1 acc: 0.7617
Top5 acc:  0.93072
  ```

`regnet_y_1_6gf`推理结果如下:

  ```log
Top1 acc:  0.77764
Top5 acc:  0.93956
  ```

`regnet_y_3_2gf`推理结果如下:

  ```log
Top1 acc:  0.78752
Top5 acc:  0.94458
  ```

`regnet_y_8gf`推理结果如下:

  ```log
Top1 acc:  0.79804
Top5 acc:  0.9497
  ```

`regnet_y_16gf`推理结果如下:

  ```log
Top1 acc:  0.80326
Top5 acc:  0.9522
  ```

`regnet_y_32gf`推理结果如下:

  ```log
Top1 acc:  0.80678
Top5 acc:  0.95278
  ```

`regnet_x_400mf`推理结果如下:

  ```log
Top1 acc:  0.7258
Top5 acc:  0.90928
  ```

`regnet_x_800mf`推理结果如下:

  ```log
Top1 acc:  0.74852
Top5 acc:  0.92254
  ```

`regnet_x_1_6gf`推理结果如下:

  ```log
Top1 acc:  0.76736
Top5 acc:  0.93284
  ```

`regnet_x_3_2gf`推理结果如下:

  ```log
Top1 acc:  0.78162
Top5 acc:  0.93946
  ```

`regnet_x_8gf`推理结果如下:

  ```log
Top1 acc:  0.79174
Top5 acc:  0.94548
  ```

`regnet_x_16gf`推理结果如下:

  ```log
Top1 acc:  0.79884
Top5 acc:  0.94824
  ```

`regnet_x_32gf`推理结果如下:

  ```log
Top1 acc:  0.80374
Top5 acc:  0.95158
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。  
