# 目录

<!-- TOC -->

- [目录](#目录)
- [TNT描述](#TNT描述)
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
            - [ImageNet-1k上的TNT](#imagenet-1k上的TNT)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的TNT](#imagenet-1k上的TNT-1)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [TNT描述](#目录)

Transformer是一种最初用于NLP任务的基于自注意力的神经网络。最近，提出了基于纯Transformer的模型来解决计算机视觉问题。这些视觉Transformer通常将图像视为一系列patches，而忽略每个patch内部的固有结构信息。

在本文中，作者提出了一种新颖的Transformer-iN-Transformer（TNT）模型，用于对patch级和像素级表示进行建模。在每个TNT块中，外部transformer块用于处理patch嵌入，而内部transformer块则从像素嵌入中提取局部特征。通过线性变换层将像素级特征投影到patch嵌入的空间，然后将其添加到patch中。通过堆叠TNT块，作者建立了用于图像识别的TNT模型。

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
├── TNT
  ├── README_CN.md                        // TNT相关说明
  ├── ascend310_infer                     // Ascend310推理需要的文件
  ├── scripts
      ├──run_standalone_train_ascend.sh   // 单卡Ascend910训练脚本
      ├──run_distribute_train_ascend.sh   // 多卡Ascend910训练脚本
      ├──run_eval_ascend.sh               // 测试脚本
      ├──run_infer_310.sh                 // 310推理脚本
  ├── src
      ├──configs                          // TNT的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ┕──TNT                          // TNT定义文件
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

- 配置TNT和ImageNet-1k数据集。

  ```python
    # Architecture
    arch: tnt_s_patch16_224             # TNT结构选择
    # ===== Dataset ===== #
    data_url: ./data/imagenet           # 数据集地址
    set: ImageNet                       # 数据集名字
    num_classes: 1000                   # 数据集分类数目
    mix_up: 0.8                         # MixUp数据增强参数
    cutmix: 1.0                         # CutMix数据增强参数
    auto_augment: rand-m9-mstd0.5-inc1  # AutoAugment参数
    interpolation: bicubic              # 图像缩放插值方法
    re_prob: 0.25                       # 数据增强参数
    re_mode: pixel                      # 数据增强参数
    re_count: 1                         # 数据增强参数
    mixup_prob: 1.                      # 数据增强参数
    switch_prob: 0.5                    # 数据增强参数
    mixup_mode: batch                   # 数据增强参数
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
    amp_level: O2                       # 混合精度策略
    beta: [ 0.9, 0.999 ]                # adamw参数
    clip_global_norm_value: 5.          # 全局梯度范数裁剪阈值
    is_dynamic_loss_scale: True         # 是否使用动态缩放
    epochs: 300                         # 训练轮数
    label_smoothing: 0.1                # 标签平滑参数
    weight_decay: 0.05                  # 权重衰减参数
    momentum: 0.9                       # 优化器动量
    batch_size: 128                     # 批次
    # ===== Hardware setup ===== #
    num_parallel_workers: 16            # 数据预处理线程数
    device_target: Ascend               # GPU或者Ascend
  ```

更多配置细节请参考脚本`config.py`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --tnt_config ./src/configs/tnt_s_patch16_224.yaml \
  > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]

  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --tnt_config ./src/configs/tnt_s_patch16_224.yaml \
  --pretrained ./ckpt_0/tnt_s_patch16_224.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

[hccl工具](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)

## 导出过程

### 导出

  ```shell
  python export.py --pretrained [CKPT_FILE] --tnt_config [CONFIG_PATH] --device_target [DEVICE_TARGET]
  ```

导出的模型会以模型的结构名字命名并且保存在当前目录下

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
  Top1 acc:  0.81376
  Top5 acc:  0.9547
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的TNT

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型|TNT|
| 模型版本              | tnt_s_patch16_224                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-11-06                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像                                              |
| 训练参数        | epoch=300, batch_size=128            |
| 优化器                  | AdamWeightDecay                                                    |
| 损失函数              | SoftTargetCrossEntropy                                       |
| 损失| 0.8123|
| 输出                    | 概率                                                 |
| 分类准确率             | 八卡：top1:81.47% top5:95.42%                   |
| 速度                      | 八卡：1747.865毫秒/步                        |
| 训练耗时          |209h45min24s（run on ModelArts）|

### 推理性能

#### ImageNet-1k上的TNT

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型                 |TNT|
| 模型版本              | tnt_s_patch16_224|                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-10-25                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Val，共50,000张图像                                                 |
| 分类准确率             | top1:81.38%,top5:95.47%                      |
| 速度                      | 平均耗时178.592 ms每张|
| 推理耗时| 约149min|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)