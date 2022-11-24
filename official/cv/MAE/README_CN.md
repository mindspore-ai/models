# 目录

<!-- TOC -->

- [目录](#目录)
- [MAE描述](#MAE描述)
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
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [120万张图像上的GoogleNet](#120万张图像上的vit)
        - [推理性能](#推理性能)
            - [120万张图像上的GoogleNet](#120万张图像上的vit)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MAE描述

Masked Autoencoders: A MindSpore Implementation，由何凯明团队提出MAE模型，将NLP领域大获成功的自监督预训练模式用在了计算机视觉任务上，效果拔群，在NLP和CV两大领域间架起了一座更简便的桥梁。MAE 是一种简单的自编码方法，可以在给定部分观察的情况下重建原始信号。由编码器将观察到的信号映射到潜在表示，再由解码器从潜在表示重建原始信号。

This is a MindSpore/NPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):

# 模型架构

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

在预训练期间，大比例的随机的图像块子集（如 75%）被屏蔽掉。编码器用于可见patch的小子集。在编码器之后引入掩码标记，并且完整的编码块和掩码标记集由一个小型解码器处理，该解码器以像素为单位重建原始图像。预训练后，解码器被丢弃，编码器应用于未损坏的图像以生成识别任务的表示。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：预训练阶段数据将在src/datasets/imagenet.py中处理，finetune阶段数据将在src/datasets/dataset.py中处理。

 ```bash
└─dataset
    ├─train                # 训练集，云上训练得是 .tar压缩文件格式
    └─val                  # 评估数据集
 ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```shell
  # 运行训练示例 CONFIG_PATH配置文件请参考'./config'路径下相关文件
  # pretrain
  python pretrain.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  # finetune
  python finetune.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &

  # 运行分布式训练示例
  cd scripts;
  # pretrain
  sh pretrain_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  # finetune
  sh finetune_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 运行评估示例
  cd scripts;
  bash eval_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]

  # 运行推理示例
  暂无
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件，参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用多机训练 ImageNet 数据集

      ```text
      # (1) 在网页上设置 "config_path='/path_to_code/config/vit-base-p16.yaml'"
      # (2) 执行a或者b
      #       a. 在 .yaml 文件中设置 "enable_modelarts=True"
      #          在 .yaml 文件中设置 "output_path"
      #          在 .yaml 文件中设置 "data_path='/cache/data/ImageNet/'"
      #          在 .yaml 文件中设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/path/mae"
      # (5) 在网页上设置启动文件为 "pretrain.py or finetune.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                            // 所有模型相关说明
    ├── mae
        ├── README.md                        // mae模型相关说明
        ├── scripts
        │   ├──pretrain_dist.sh              // 分布式到Ascend的shell脚本
        │   ├──finetune_dist.sh              // 单卡到Ascend的shell脚本
        │   ├──eval_dist.sh                  // Ascend评估的shell脚本
        ├── src
        │   ├──datasets
        │       ├──auto_augment.py           // 数据自动增强策略
        │       ├──dataset.py                // finetune数据集创建
        │       ├──image_policy.py           // 数据自动增强策略2
        │       ├──imagenet.py               // pretrain数据集创建
        │       ├──mixup.py                  // 自定义数据强增策略
        │   ├──loss
        │       ├──loss.py                   // soft-target loss 函数
        │   ├──lr
        │       ├──lr_decay.py               // layer-wise lr schedual函数
        │       ├──lr_generator.py           // lr策略
        │   ├──models
        │       ├──eval_engine.py            // 评估策略
        │       ├──metric.py                 // 评估结果计算方式
        │       ├──modules.py                // 模型基础模块
        │       ├──mae_vit.py                // mae 模型结构定义
        │       ├──vit.py                    // vit 模型结构定义
        │   ├──monitors
        │       ├──monitor.py                // 模型状态监测函数
        │   ├──trainer
        │       ├──ema.py                    // ema 方法
        │       ├──trainer.py                // 训练过程定义
        │   ├──model_utils                   // 云上训练依赖
        ├── config
        │   ├──eval.yml                      // 评估配置
        │   ├──vit-base-p16.yaml             // 64p训练参数配置
        │   ├──mae-vit-base-p16.yaml         // 64p训练参数配置
        │   ├──finetune-vit-base-16.yaml     // 32p训练参数配置
        ├── pretrain.py                      // 预训练脚本
        ├── finetune.py                      // 微调训练脚本
        ├── eval.py                          // 评估脚本
        ├── requirements.txt                 // 依赖python包
```

## 脚本参数

在./config/.yaml中可以同时配置训练参数和评估参数。

- mae和ImageNet数据集配置。

  ```yaml
  # mae模型配置
  encoder_layers: 12
  encoder_num_heads: 12
  encoder_dim: 768
  decoder_layers: 8
  decoder_num_heads: 16
  decoder_dim: 512
  mlp_ratio: 4
  masking_ratio: 0.75
  norm_pixel_loss: True

  # Ascend 训练环境初始化
  seed: 2022
  context:
      mode: "GRAPH_MODE" #0--Graph Mode; 1--Pynative Mode
      device_target: "Ascend"
      max_call_depth: 10000
      save_graphs: False
      device_id: 0
  use_parallel: True
  parallel:
      parallel_mode: "DATA_PARALLEL"
      gradients_mean: True

  # 训练数据集
  data_path: "/mnt/vision/ImageNet1K/CLS-LOC"
  img_ids: "tot_ids.json" # ImageNet index of data path
  num_workers: 8
  image_size: 224

  # 训练配置
  epoch: 800
  batch_size: 64
  patch_size: 16
  sink_mode: True
  per_step_size: 0
  use_ckpt: ""

  # loss scale 管理
  use_dynamic_loss_scale: True # default use FixLossScaleUpdateCell

  # 优化器配置
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.05

  # 学习率配置
  base_lr: 0.00015
  start_learning_rate: 0.
  end_learning_rate: 0.
  warmup_epochs: 40

  # EMA配置
  use_ema: False
  ema_decay: 0.9999

  # 梯度裁剪配置
  use_global_norm: False
  clip_gn_value: 1.0

  # callback配置
  cb_size: 1
  save_ckpt_epochs: 1
  prefix: "MaeFintuneViT-B-P16"

  # 保存目录配置
  save_dir: "./output/"
  ```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```shell
  # pretrain
  python pretrain.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  # finetune
  python finetune.py --config_path=[CONFIG_PATH] --use_parallel False > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。
  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```shell
  # vim pretrain log
  待补充
  ...
  # vim finetune log
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```shell
  # 运行分布式训练示例
  cd scripts;
  # pretrain
  sh pretrain_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  # finetune
  sh finetune_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```shell
  # vim train_parallel0/log
  待补充
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/vit/vit_base_patch32.ckpt”。

  ```bash
  # 运行评估示例
  cd scripts;
  bash eval_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.81
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为用户保存的检查点文件，如“username/mae/train_parallel0/outputs/finetune-vit-base-p16-300_312.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy=" eval0/log
  accuracy=0.81
  ```

# 模型描述

## 性能

### 评估性能

#### imagenet 120万张图像上的MAE-Vit-B-P16

| 参数                       | Ascend                                                      |
| -------------------------- | -----------------------------------------------------------|
| 模型版本                   | MAE-Vit-Base-P16                                            |
| 资源                       | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8      |
| 上传日期                   | 03/30/2022                                                |
| MindSpore版本              | 1.6.0                                                       |
| 数据集                     | 120万张图像                                                  |
| 训练参数                   | epoch=800, steps=349*800, batch_size=64, base_lr=0.00015 |
| 优化器                     | Adamw                                                       |
| 损失函数                   | nMSE                              |
| 输出                       | 概率                                                        |
| 损失                       | 0.19                                                      |
| 速度                       | 64卡：481ms/step（ModelArts训练数据） |
| 总时长                     | 64卡：38h（ModelArts训练数据）                                 |
| 微调检查点                 | 1.34G (.ckpt文件)                                         |
| 脚本                    | [mae脚本](https://gitee.com/mindspore/models/blob/master/official/cv/MAE/pretrain.py)                                             |

#### imagenet 120万张图像上的Finetune-Vit-B-P16

| 参数          | Ascend                                                  |
| ------------- | ------------------------------------------------------- |
| 模型版本      | Finetune-Vit-Base-P16                                   |
| 资源          | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8 |
| 上传日期      | 03/30/2022                                              |
| MindSpore版本 | 1.6.0                                                   |
| 数据集        | 120万张图像                                             |
| 训练参数      | epoch=100, steps=312*100, batch_size=32, base_lr=0.001  |
| 优化器        | Adamw                                                   |
| 损失函数      | SoftTargetCrossEntropy                                  |
| 输出          | 概率                                                    |
| 损失          | 2.5                                                     |
| 速度          | 32卡：332ms/step（ModelArts训练数据）                   |
| 总时长        | 32卡：6h（ModelArts训练数据）                           |
| 准确率Top1    | 0.807                                                   |
| 微调检查点    | 1009M (.ckpt文件)                                       |
| 脚本          | [finetune脚本](https://gitee.com/mindspore/models/blob/master/research/cv/squeezenet/finetune.py)                                        |

#### imagenet 120万张图像上的Vit-B-P16

| 参数          | Ascend                                                  |
| ------------- | ------------------------------------------------------- |
| 模型版本      | Vit-Base-P16                                            |
| 资源          | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8 |
| 上传日期      | 03/30/2022                                              |
| MindSpore版本 | 1.6.0                                                   |
| 数据集        | 120万张图像                                             |
| 训练参数      | epoch=300, steps=312*300, batch_size=64, base_lr=0.001  |
| 优化器        | Adamw                                                   |
| 损失函数      | SoftTargetCrossEntropy                                  |
| 输出          | 概率                                                    |
| 损失          | 2.5                                                     |
| 速度          | 64卡：332ms/step（ModelArts训练数据）                   |
| 总时长        | 64卡：16h（ModelArts训练数据）                          |
| 准确率Top1    | 0.799                                                  |
| 微调检查点    | 1009M (.ckpt文件)                                       |
| 脚本          | [vit脚本](https://gitee.com/mindspore/models/blob/master/official/cv/MAE/eval.py)                                             |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 配置文件读取+通过配置文件生成模型训练需要的参数
  args.loss_scale = ...
  lrs = ...
  ...
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=args.device_target)
  context.set_context(device_id=args.device_id)

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(args.data_path, 1, False)

  # 定义模型
  net = FinetuneViT(args.vit_config)
  opt = AdamW(filter(lambda x: x.requires_grad, net.get_parameters()), lrs, args.beta1, args.beta2, loss_scale=args.loss_scale, weight_decay=cfg.weight_decay)
  loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor, num_classes=args.class_num)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 加载预训练模型
  param_dict = load_checkpoint(args.pretrained)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # 执行评估
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。