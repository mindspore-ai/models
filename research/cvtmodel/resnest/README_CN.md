# 目录

<!-- TOC -->

- [目录](#目录)
- [ResNeSt描述](#resnest描述)
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

# ResNeSt描述

ResNeSt是一个基于卷积的神经网络，用于图像分类。有关该模型的描述，可查阅(https://pytorch.org/hub/pytorch_vision_resnest/)。
本仓库中是基于torch提供的模型文件，使用MindConverter工具转化出Mindspore来ckpt文件，进行全量推理以验证模型文件精度。

# 模型架构

ResNeSt模型支持四种模式：ResNeSt-50, ResNeSt-101，ResNeSt-201, ResNeSt-269。

# 数据集

ResNeSt使用的数据集： ImageNet

数据集的默认配置如下：

- 测试数据集预处理：
    - 图像的输入尺寸(ResNeSt-50)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(ResNeSt-101)：256\*256（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(ResNeSt-200)：320\*320（将图像缩放到320\*320，然后在中央区域裁剪图像）
    - 图像的输入尺寸(ResNeSt-269)：416\*416（将图像缩放到416\*416，然后在中央区域裁剪图像）
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
    ├── README.md                          // 所有模型的说明
    ├── ResNeSt
        ├── README_CN.md                 // ResNeSt相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├── run_infer_310.sh                    // Ascend 310 推理shell脚本
        ├── src
        │   ├── resnest50.py             // resnest-50模型文件
        │   ├── resnest101.py             // resnest-101模型文件
        │   ├── resnest200.py             // resnest-200模型文件
        │   ├── resnest269.py             // resnest-269模型文件
        ├── export.py                   // 导出脚本
        ├── preprocess.py                   // 数据预处理脚本
        ├── postprocess.py                   // 310 推理后处理脚本
```

## 导出过程

### 导出

```shell
python export.py --backbone [NET_NAME] --ckpt_path [CKPT_PATH] --device_target [DEVICE_TARGET] --device_id 0 --file_format [EXPORT_FORMAT] --file_name [FILE_NAME]
```

`backbone` 可选 ["resnest50", "resnest101", "resnest200", "resnest269"]
`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

## 推理过程

### 推理

在推理之前需要先导出模型，AIR模型只能在昇腾910环境上导出，MINDIR可以在任意环境上导出。

```shell
# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [BACKBONE] [DATASET] [DATA_PATH] [DEVICE_ID]
```

-注: ResNeSt系列网络使用ImageNet数据集。

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
ResNeSt-50网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=40560, tot=50000, acc=81.12%
  after allreduce eval: top5_correct=47792, tot=50000, acc=95.58%
  ```

ResNeSt-101网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=41330, tot=50000, acc=82.66%
  after allreduce eval: top5_correct=48151, tot=50000, acc=96.30%
  ```

ResNeSt-200网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=42082, tot=50000, acc=84.16%
  after allreduce eval: top5_correct=48469, tot=50000, acc=96.94%
  ```  

ResNeSt-269网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=42288, tot=50000, acc=84.58%
  after allreduce eval: top5_correct=48562, tot=50000, acc=97.12%
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
