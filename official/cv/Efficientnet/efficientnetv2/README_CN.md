# 目录

<!-- TOC -->

- [目录](#目录)
- [EfficientNetV2描述](#EfficientNetV2描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的EfficientNetV2](#imagenet-1k上的EfficientNetV2)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的EfficientNetV2](#imagenet-1k上的EfficientNetV2-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [EfficientNetV2描述](#目录)

本文是谷歌的MingxingTan与Quov V.Le对EfficientNet的一次升级，旨在保持参数量高效利用的同时尽可能提升训练速度。在EfficientNet的基础上，引入了Fused-MBConv到搜索空间中；同时为渐进式学习引入了自适应正则强度调整机制。两种改进的组合得到了本文的EfficientNetV2，它在多个基准数据集上取得了SOTA性能，且训练速度更快。比如EfficientNetV2取得了87.3%的top1精度且训练速度快5-11倍。

# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```bash
├── EfficientNetV2
  ├── README_CN.md                        // EfficientNetV2相关说明
  ├── ascend310_infer                     // Ascend310推理需要的文件
  ├── scripts
      ├──run_standalone_train_ascend.sh   // 单卡Ascend910训练脚本
      ├──run_distribute_train_ascend.sh   // 多卡Ascend910训练脚本
      ├──run_eval_ascend.sh               // 测试脚本
      ├──run_infer_onnx.sh                // onnx推理脚本
      ├──run_infer_310.sh                 // 310推理脚本
  ├── src
      ├──configs                          // EfficientNetV2的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet_finetune.py         // imagenet配置文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // EfficientNetV2定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
  │   ├──weight_conver                    // 将TensorFlow的权重转化为MindSpore的ckpt
  │   ├──tools                            // 工具文件夹
          ├──callback.py                  // 自定义回调函数，训练结束测试
          ├──cell.py                      // 一些关于cell的通用工具函数
          ├──criterion.py                 // 关于损失函数的工具函数
          ├──get_misc.py                  // 一些其他的工具函数
          ├──optimizer.py                 // 关于优化器和参数的函数
          ┕──schedulers.py                // 学习率衰减的工具函数
  ├── train.py                            // 训练文件
  ├── eval.py                             // 评估文件
  ├── export.py                           // 导出模型文件
  ├── infer_onnx.py                       // ONNX推理文件
  ├── postprocess.py                      // 推理计算精度文件
  ├── preprocess.py                       // 推理预处理图片文件

```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置EfficientNetV2和ImageNet-1k数据集。

  ```python
    # Architecture
    arch: effnetv2_s                             # 模型结构
    # ===== Dataset ===== #
    data_url: ./imagenet                         # 数据集地址
    set: ImageNetFinetune                        # 数据集类别
    num_classes: 1000                            # 数据集种类数
    interpolation: bilinear                      # 插值方法
    # ===== Learning Rate Policy ======== #
    eps: 0.001                                   # 最小参数
    optimizer: rmsprop                           # 优化器类别
    base_lr: 0.0005                              # 基础学习率
    warmup_lr: 0.                                # 热身学习率
    min_lr: 0.                                   # 学习率最小值
    lr_scheduler: constant_lr                    # 学习率策略
    warmup_length: 1                             # 学习率热身轮数
    # ===== Network training config ===== #
    amp_level: O0                                # 混合精度类别
    clip_global_norm: True                       # 是否全局梯度裁剪
    clip_global_norm_value: 5                    # 全局梯度裁剪范数
    is_dynamic_loss_scale: True                  # 是否是动态损失缩放
    epochs: 15                                   # 训练轮数
    label_smoothing: 0.1                         # 标签平滑系数
    weight_decay: 0.00001                        # L2权重衰减系数
    decay: 0.9                                   # RMSProp衰减系数
    momentum: 0.9                                # 动量系数
    batch_size: 32                               # 批次大小
    # ===== Hardware setup ===== #
    num_parallel_workers: 16                     # 数据预处理线程数
    device_target: Ascend                        # 设备类别
    # ===== Model config ===== #
    drop_path_rate: 0.2                          # drop_path的概率
    drop_out_rate: 0.000001                      # drop_out概率
    image_size: 384                              # 图像大小
    pretrained: ./efficientnets_imagenet22k.ckpt # 预训练权重地址
  ```

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- 运行准备

  ```bash
  # 下载ImageNet22k预训练权重[网站](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s-21k.tgz)

  # 解压后运行
  python weight_convert.py --pretrained ./efficientnetv2-s-21k/model

  # 转换权重转换完成后, 在efficientv2_s_finetune的pretrained中配置权重地址
  ```

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/effnetv2_s_finetune.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/effnetv2_s_finetune.yaml \
  --pretrained ./ckpt_0/effnetv2_s.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件

  请遵循以下链接中的说明

[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 导出过程

### 导出

  ```shell
  python export.py --pretrained [CKPT_FILE] --config [CONFIG_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
  ```

`EXPORT_FORMAT`可选["AIR", "MINDIR", "ONNX"]
导出的模型会以模型的结构名字命名并且保存在当前目录下

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出，onnx可以在CPU/GPU环境下导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Top1 acc:  0.838
  Top5 acc:  0.96956
  ```

- 在GPU/CPU上使用ImageNet-1k数据集进行推理

  推理的结果保存在主目录下，在infer_onnx.log日志文件中可以找到推理结果。

  ```shell
  bash run_infer_onnx.sh [ONNX_PATH] [CONFIG] [DEVICE_TARGET]
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的EfficientNetV2

| 参数                 | Ascend                           |
| -------------------------- | ----------------------- |
|模型|EfficientNetV2|
| 模型版本              | EfficientNetV2-S     |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-12-19              |
| MindSpore版本          | 1.3.0     |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像        |
| 训练参数        | epoch=15, batch_size=32(16卡)   |
| 优化器                  | RMSProp         |
| 损失函数              | CrossEntropySmooth   |
| 损失|0.687|
| 输出                    | 概率                |
| 分类准确率             | 十六卡：top1:83.778% top5:96.956%                   |
| 速度                      | 十六卡：582.105毫秒/步                        |
| 训练耗时          |7h25min15s（run on ModelArts）|

### 推理性能

#### ImageNet-1k上的EfficientNetV2

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型                 |EfficientNetV2|
| 模型版本              | EfficientNetV2-S|                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-12-19                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Val，共50,000张图像                                                 |
| 分类准确率             | top1:83.8%,top5:96.956%                      |
| 速度                      | 平均耗时11.4918ms每张|
| 推理耗时| 约22min|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)