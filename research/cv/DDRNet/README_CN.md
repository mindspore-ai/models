# 目录

<!-- TOC -->

- [目录](#目录)
- [DDRNet描述](#DDRNet描述)
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
            - [ImageNet-1k上的DDRNet](#imagenet-1k上的DDRNet)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的DDRNet](#imagenet-1k上的DDRNet-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [DDRNet描述](#目录)

语义分割是自主车辆理解周围场景的关键技术。对于实际的自主车辆，不希望花费大量的推理时间来获得高精度的分割结果。使用轻量级架构(编码器解码器或双通道)或对低分辨率图像进行推理，最近的方法实现了非常快速的场景解析，甚至可以在单个1080Ti GPU上以100 FPS以上的速度运行。然而，在这些实时方法和基于膨胀主干的模型之间仍然存在明显的性能差距。

为了解决这个问题，受HRNet的启发，作者提出了一种具有深度高分辨率表示能力的深度双分辨率网络，用于高分辨率图像的实时语义分割，特别是道路行驶图像。作者提出了一种新的深度双分辨率网络用于道路场景的实时语义分割。 DDRNet从一个主干开始，然后被分成两个具有不同分辨率的平行深分支。一个深分支生成相对高分辨率的特征映射，另一个通过多次下采样操作提取丰富的上下文信息。为了有效的信息融合，在两个分支之间桥接多个双边连接。此外，我们还提出了一个新的模块DAPPM，它大大增加了接受域，比普通的PPM更充分地提取了上下文信息。

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

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)
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
├── DDRNet
  ├── README_CN.md                        // DDRNet相关说明
  ├── ascend310_infer                     // Ascend310推理需要的文件
  ├── scripts
      ├──run_standalone_train_ascend.sh   // 单卡Ascend910训练脚本
      ├──run_distribute_train_ascend.sh   // 多卡Ascend910训练脚本
      ├──run_eval_ascend.sh               // 测试脚本
      ├──run_infer_310.sh                 // 310推理脚本
  ├── src
      ├──configs                          // DDRNet的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // DDRNet定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
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
  ├── postprocess.py                      // 推理计算精度文件
  ├── preprocess.py                       // 推理预处理图片文件

```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置DDRNet和ImageNet-1k数据集。

  ```python
    # Architecture Top1-75.9%
    arch: DDRNet23                              # 模型结构
    # ===== Dataset ===== #
    data_url: ./data/imagenet                   # 数据集地址
    set: ImageNet                               # 数据集类别
    num_classes: 1000                           # 数据集种类数
    mix_up: 0.8                                 # MixUp数据增强参数
    cutmix: 1.0                                 # CutMix数据增强参数  
    color_jitter: 0.4                           # color参数
    auto_augment: rand-m9-mstd0.5-inc1          # auto_augment策略
    interpolation: bicubic                      # 图像缩放插值方法
    re_mode: pixel                              # 数据增强参数
    re_count: 1                                 # 数据增强参数
    mixup_prob: 1.                              # 数据增强参数
    switch_prob: 0.5                            # 数据增强参数
    mixup_mode: batch                           # 数据增强参数
    mixup_off_epoch: 0.                         # 使用多少轮mixup, 0为一直使用
    image_size: 224                             # 图像大小
    crop_pct: 0.875                             # 图像缩放比
    # ===== Learning Rate Policy ======== #
    optimizer: momentum                         # 优化器类别
    use_nesterov: True                          # 是否使用牛顿法收敛
    base_lr: 0.1                                # 基础学习率
    warmup_lr: 0.000001                         # 学习率热身初始学习率
    min_lr: 0.00001                             # 最小学习率
    lr_scheduler: cosine_lr                     # 学习率衰减策略
    warmup_length: 10                           # 学习率热身轮数
    lr_adjust: 30 # for multistep lr            # 多步学习率的衰减轮数
    # ===== Network training config ===== #
    amp_level: O2                               # 混合精度策略
    keep_bn_fp32: True                          # 保持bn为fp32
    beta: [ 0.9, 0.999 ]                        # 优化器的beta参数
    clip_global_norm_value: 5.                  # 全局梯度范数裁剪阈值
    clip_global_norm: True                      # 是否使用全局梯度裁剪
    is_dynamic_loss_scale: True                 # 是否使用动态损失缩放
    epochs: 300                                 # 训练轮数
    label_smoothing: 0.1                        # 标签平滑参数
    loss_scale: 1024                            # 损失缩放
    weight_decay: 0.0001                        # 权重衰减参数
    decay: 0.9 # for rmsprop                    # rmsprop的decay系数
    momentum: 0.9                               # 优化器动量
    batch_size: 512                             # 批次
    # ===== Hardware setup ===== #
    num_parallel_workers: 16                    # 数据预处理线程数
    device_target: Ascend                       # GPU或者Ascend
    # ===== Model config ===== #
    drop_path_rate: 0.1                         # drop_path的概率
  ```

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --ddr_config ./src/configs/ddrnet23_imagenet.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --ddr_config ./src/configs/ddrnet23_imagenet.yaml \
  --pretrained ./ckpt_0/DDRNet23.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 导出过程

### 导出

  ```shell
  python export.py --pretrained [CKPT_FILE] --ddr_config [CONFIG_PATH] --device_target [DEVICE_TARGET]
  ```

导出的模型会以模型的结构名字命名并且保存在当前目录下

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Top1 acc: 0.76578
  Top5 acc: 0.9331
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的DDRNet

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型|DDRNet|
| 模型版本              | DDRNet23                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-12-04                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像                                              |
| 训练参数        | epoch=300, batch_size=512            |
| 优化器                  | Momentum                                                    |
| 损失函数              | SoftTargetCrossEntropy                                       |
| 损失| 1.313|
| 输出                    | 概率                                                 |
| 分类准确率             | 八卡：top1:76.598% top5:93.312%                   |
| 速度                      | 八卡：940.911 ms毫秒/步                        |
| 训练耗时          |34h50min07s（run on ModelArts）|

### 推理性能

#### ImageNet-1k上的DDRNet

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型                 |DDRNet|
| 模型版本              | DDRNet23|                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-12-04                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Val，共50,000张图像                                                 |
| 分类准确率             | top1:76.578%,top5:93.31%                      |
| 速度                      | 平均耗时3.29687 ms每张|
| 推理耗时| 约38min|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)