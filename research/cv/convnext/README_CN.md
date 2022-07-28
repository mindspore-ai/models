# 目录

<!-- TOC -->

- [目录](#目录)
- [ConvNeXt描述](#ConvNeXt描述)
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
            - [ImageNet-1k上的ConvNeXt](#imagenet-1k上的ConvNeXt)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的ConvNeXt](#imagenet-1k上的ConvNeXt-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [ConvNeXt描述](#目录)

自从ViT提出之后，在过去的一年里（2021年），基于transformer的模型在计算机视觉各个领域全面超越CNN模型。然而，这很大程度上都归功于Local Vision Transformer模型，Swin Transformer是其中重要代表。原生的ViT模型其计算量与图像大小的平方成正比，而Local Vision Transformer模型由于采用local attention（eg. window attention），其计算量大幅度降低，除此之外，Local Vision Transformer模型往往也采用金字塔结构，这使得它更容易应用到密集任务如检测和分割中，因为密集任务往往输入图像分辨率较高，而且也需要多尺度特征（eg. FPN）。虽然Local Vision Transformer模型超越了CNN模型，但是它却越来越像CNN了：首先locality是卷积所具有的特性，其次金字塔结构也是主流CNN模型所采用的设计。

近日，MetaAI在论文A ConvNet for the 2020s中从ResNet出发并借鉴Swin Transformer提出了一种新的CNN模型：ConvNeXt，其效果无论在图像分类还是检测分割任务上均能超过Swin Transformer，而且ConvNeXt和vision transformer一样具有类似的scalability（随着数据量和模型大小增加，性能同比提升）。

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

- 硬件
    - 使用Ascend来搭建硬件环境。
    - 使用GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```bash
├── convnext
  ├── README_CN.md                        // ConvNeXt相关说明
  ├── scripts
      ├──run_standalone_train_ascend.sh   // 单卡Ascend910训练脚本
      ├──run_distribute_train_ascend.sh   // 多卡Ascend910训练脚本
      ├──run_eval_ascend.sh               // 测试脚本
      ├──run_standalone_train_gpu.sh      // 单卡GPU训练脚本
      ├──run_distribute_train_gpu.sh      // 多卡GPU训练脚本
      ┕──run_eval_gpu.sh                  // 测试脚本
  ├── src
      ├──configs                          // ConvNeXt的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ┕──ConvNeXt                          // ConvNeXt定义文件
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
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ConvNeXt和ImageNet-1k数据集。

  ```python
    # Architecture
    arch: convnext_tiny                 # ConvNeXt结构选择
    # ===== Dataset ===== #
    data_url: ./data/imagenet           # 数据集地址
    set: ImageNet                       # 数据集名字
    num_classes: 1000                   # 数据集分类数目
    mix_up: 0.8                         # MixUp数据增强参数
    cutmix: 1.0                         # CutMix数据增强参数
    auto_augment: rand-m9-mstd0.5-inc1  # AutoAugment参数
    interpolation: bicubic              # 图像缩放插值方法
    re_prob: 0                          # RandomErasing概率
    re_mode: pixel                      # RandomErasing模式
    re_count: 1                         # RandomErasing重复次数
    mixup_prob: 1.                      # MixUp概率
    switch_prob: 0.5                    # MixUp和CutMix切换概率
    mixup_mode: batch                   # MixUp模式
    # ===== Learning Rate Policy ======== #
    optimizer: adamw                    # 优化器类别
    base_lr: 0.0005                     # 基础学习率
    warmup_lr: 0.00000007               # 学习率热身初始学习率
    min_lr: 0.000006                    # 最小学习率
    lr_scheduler: cosine_lr             # 学习率衰减策略
    warmup_length: 20                   # 学习率热身轮数
    nonlinearity: GELU                  # 激活函数类别
    image_size: 224                     # 图像大小
    # ===== Network training config ===== #
    amp_level: O1                       # 混合精度策略
    beta: [ 0.9, 0.999 ]                # adamw参数
    clip_global_norm_value: 5.          # 全局梯度范数裁剪阈值
    is_dynamic_loss_scale: True         # 是否使用动态缩放
    epochs: 300                         # 训练轮数
    label_smoothing: 0.1                # 标签平滑参数
    weight_decay: 0.05                  # 权重衰减参数
    momentum: 0.9                       # 优化器动量
    batch_size: 128                     # 批次
    # ===== EMA ===== #
    with_ema: False                     # 是否使用ema更新
    ema_decay: 0.9999                   # ema移动系数
    # ===== Hardware setup ===== #
    num_parallel_workers: 16            # 数据预处理线程数
    device_target: Ascend               # GPU或者Ascend
  ```

更多配置细节请参考脚本`config.py`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/convnext_tiny.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/convnext_tiny.yaml \
  --pretrained ./ckpt_0/convnext_tiny.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  [hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

- GPU环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target GPU --config ./src/configs/convnext_tiny_gpu.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_gpu.sh [CUDA_VISIBLE_DEVICES] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target GPU --config ./src/configs/convnext_tiny.yaml \
  --pretrained ./ckpt_0/convnext_tiny.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_run.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```

## 导出过程

### 导出

  ```shell
  python export.py --pretrained [CKPT_FILE] --config [CONFIG_PATH] --device_target [DEVICE_TARGET]
  ```

导出的模型会以模型的结构名字命名并且保存在当前目录下

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的ConvNeXt

| 参数          | Ascend                               | GPU                                  |
| ------------- | ------------------------------------ | ------------------------------------ |
| 模型          | ConvNeXt                             | ConvNeXt                             |
| 模型版本      | convnext_tiny                        | convnext_tiny                        |
| 资源          | Ascend 910                           | NVIDIA GeForce RTX 3090              |
| 上传日期      | 2022-01-23                           | 2022-05-23                           |
| MindSpore版本 | 1.3.0                                | 1.7.0                                |
| 数据集        | ImageNet-1k Train，共1,281,167张图像 | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数      | epoch=300, batch_size=256            | epoch=300, batch_size=224            |
| 优化器        | AdamWeightDecay                      | AdamWeightDecay                      |
| 损失函数      | SoftTargetCrossEntropy               | SoftTargetCrossEntropy               |
| 损失          | 0.8114                               | 0.7939                               |
| 输出          | 概率                                 | 概率                                 |
| 分类准确率    | 八卡：top1:82.072% top5:95.694%      | 8卡：top1:81.504% top5:95.630%       |
| 速度          | 16卡：1106.875毫秒/步                | 8卡：6103.531毫秒/步                 |
| 训练耗时      | 31h36min04s（run on ModelArts）      | 415h30min43s                         |

GPU训练经过多次调参，目前最好能达到81.504%，但是由于需要的算力太多，没有再做过多的尝试，用户在使用过程中可以基于使用的数据集进行调参

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)