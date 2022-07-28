# 目录

<!-- TOC -->

- [目录](#目录)
- [Hrnet描述](#Hrnet描述)
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

# Hrnet系列描述

Hrnet系列是一系列基于Hrnet扩展的网络模型，用于图像分类。有关该模型的描述，可查阅(http://rwightman.github.io/pytorch-image-models/models/)。
本仓库中是基于torch提供的模型文件，使用MindConverter工具转化出Mindspore来ckpt文件，进行全量推理以验证模型文件精度。

# 模型架构

Hesnet模型支持五种模式：Hrnet_w18_small, Hrnet_w30, Hrnet_w40, Hrnet_w48, Hrnet_w64。

# 数据集

Hesnet使用的数据集： ImageNet

数据集的默认配置如下：

- 测试数据集预处理：
    - 图像的输入尺寸(Hrnet_w18_small)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Hrnet_w30)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Hrnet_w40)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Hrnet_w48)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Hrnet_w64)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
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
    ├── README_CN.md                          // 所有模型的说明
    ├── Hrnet
        ├── README_CN.md                 // Hrnet系列相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├── run_infer_310.sh                    // Ascend 310 推理shell脚本
        ├── src
        │   ├── hrnet_w18_small.py             // hrnet_w18_small模型文件
        │   ├── hrnet_w30.py             // hrnet_w30模型文件
        │   ├── hrnet_w40.py             // hrnet_w40模型文件
        │   ├── hrnet_w48.py             // hrnet_w48模型文件
        │   ├── hrnet_w64.py             // hrnet_w64模型文件
        ├── export.py                   // 导出脚本
        ├── preprocess.py                   // 数据预处理脚本
        ├── postprocess.py                   // 310 推理后处理脚本
```

## 导出过程

### 导出

```shell
python export.py --ckpt_path [CKPT_PATH] --device_target [DEVICE_TARGET] --device_id 0 --file_format [EXPORT_FORMAT] --file_name [FILE_NAME]
```

`EXPORT_FORMAT` 设定为 ["MINDIR"]

## 推理过程

### 推理

在推理之前需要先导出模型，MINDIR可以在任意环境上导出。

```shell
# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

-注: Hrnet系列网络使用ImageNet数据集。

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
Hrnet_w18_small网络使用ImageNet推理得到的结果如下:

  ```log
  Eval: top1_correct=35778, tot=50000, acc=71.56%
  ```

Hrnet_w30网络使用ImageNet推理得到的结果如下:

  ```log
  Eval: top1_correct=38987, tot=50000, acc=77.97%
  ```

Hrnet_w40网络使用ImageNet推理得到的结果如下:

  ```log
  Eval: top1_correct=39251, tot=50000, acc=78.5%
  ```

Hrnet_w48网络使用ImageNet推理得到的结果如下:

  ```log
  Eval: top1_correct=39447, tot=50000, acc=78.89%
  ```

Hrnet_w64网络使用ImageNet推理得到的结果如下:

  ```log
  Eval: top1_correct=39547, tot=50000, acc=79.09%
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
