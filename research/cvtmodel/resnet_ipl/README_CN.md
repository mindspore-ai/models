# 目录

<!-- TOC -->

- [目录](#目录)
- [Resnet描述](#Resnet描述)
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

# Resnet_ipl描述

Resnet_ipl是一系列基于resnet扩展的网络模型，用于图像分类。有关该模型的描述，可查阅(http://rwightman.github.io/pytorch-image-models/models/)。
本仓库中是基于torch提供的模型文件，使用MindConverter工具转化出Mindspore来ckpt文件，进行全量推理以验证模型文件精度。

# 模型架构

Resnet模型支持四种模式：Resnet26t, Resnet51q，Resnet101d, Resnetrs50, Resnetrs200, Seresnet152d, Resnet101e, Gernet_l。

# 数据集

Resnet使用的数据集： ImageNet

数据集的默认配置如下：

- 测试数据集预处理：
    - 图像的输入尺寸(Resnet26t)：272\*272（将图像缩放到272\*272，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Resnet51q)：288\*288（将图像缩放到288\*288，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Resnet101d)：256\*256（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Resnetrs50)：224\*224（将图像缩放到256\*256，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Resnetrs200)：320\*320（将图像缩放到320\*320，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Seresnet152d)：320\*320（将图像缩放到320\*320，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Resnet101e)：320\*320（将图像缩放到320\*320，然后在中央区域裁剪图像）
    - 图像的输入尺寸(Gernet_l)：292\*292（将图像缩放到292\*292，然后在中央区域裁剪图像）
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
    ├── Resnet
        ├── README_CN.md                 // Resnet相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├── run_infer_310.sh                    // Ascend 310 推理shell脚本
        ├── src
        │   ├── resnet26t.py             // resnet26t模型文件
        │   ├── resnet51q.py             // resnet51q模型文件
        │   ├── resnet101d.py             // resnet101d模型文件
        │   ├── resnetrs50.py             // resnetrs50模型文件
        │   ├── resnetrs200.py             // resnetrs200模型文件
        │   ├── seresnet152d.py            // seresnet152d模型文件
        │   ├── Resnet101e.py            // Resnet101e模型文件
        │   ├── gernet_l.py             // gernet_l模型文件
        ├── export.py                   // 导出脚本
        ├── preprocess.py                   // 数据预处理脚本
        ├── postprocess.py                   // 310 推理后处理脚本
```

## 导出过程

### 导出

```shell
python export.py --backbone [NET_NAME] --ckpt_path [CKPT_PATH] --device_target [DEVICE_TARGET] --device_id 0 --file_format [EXPORT_FORMAT] --file_name [FILE_NAME]
```

`backbone` 可选 ["resnet26t", "resnet51q"，"resnet101d", "Resnetrs50", "Resnetrs200", "Seresnet152d", "Resnet101e"]
`EXPORT_FORMAT` 设定为 ["MINDIR"]

## 推理过程

### 推理

在推理之前需要先导出模型，MINDIR可以在任意环境上导出。

```shell
# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [BACKBONE] [DATASET] [DATA_PATH] [DEVICE_ID]
```

-注: Resnet系列网络使用ImageNet数据集。

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
Resnet26t网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=40560, tot=50000, acc=81.12%
  after allreduce eval: top5_correct=47792, tot=50000, acc=95.58%
  ```

Resnet51q网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=40816, tot=50000, acc=81.63%
  after allreduce eval: top5_correct=47901, tot=50000, acc=95.80%
  ```

Resnet101d网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=41240, tot=50000, acc=82.48%
  after allreduce eval: top5_correct=48055, tot=50000, acc=96.11%
  ```  

Resnetrs50网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=40106, tot=50000, acc=80.21%
  after allreduce eval: top5_correct=47648, tot=50000, acc=95.30%
  ```

Resnetrs200网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=41613, tot=50000, acc=83.23%
  after allreduce eval: top5_correct=48288, tot=50000, acc=96.58%
  ```

Seresnet152d网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=41813, tot=50000, acc=83.63%
  after allreduce eval: top5_correct=48396, tot=50000, acc=96.79%
  ```

Resnet101e网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=41334, tot=50000, acc=82.67%
  after allreduce eval: top5_correct=48153, tot=50000, acc=96.31%
  ```

Gernet_l网络使用ImageNet推理得到的结果如下:

  ```log
  after allreduce eval: top1_correct=40700, tot=50000, acc=81.40%
  after allreduce eval: top5_correct=47806, tot=50000, acc=95.61%
  ```

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
