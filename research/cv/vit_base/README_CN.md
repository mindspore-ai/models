# 目录

<!-- TOC -->

- [目录](#目录)
- [vit_base描述](#vit_base描述)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CIFAR-10上的vit_base](#cifar-10上的vit_base)
        - [推理性能](#推理性能)
            - [CIFAR-10上的vit_base](#cifar-10上的vit_base)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# vit_base描述

Transformer架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

[论文](https://arxiv.org/abs/2010.11929) ：Dosovitskiy, A. , Beyer, L. , Kolesnikov, A. , Weissenborn, D. , & Houlsby, N.. (2020). An image is worth 16x16 words: transformers for image recognition at scale.

# 模型架构

vit_base的总体网络架构如下： [链接](https://arxiv.org/abs/2010.11929)

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、6万张彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/enable_mixed_precision.html) 的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# 环境要求

- 硬件（Ascend or GPU）
    - 使用Ascend or GPU 来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估，特别地，进行训练前需要先下载官方基于[ImageNet21k](https://console.cloud.google.com/storage/vit_models/)的预训练模型[ViT-B_16](http://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) ，并将其转换为MindSpore支持的ckpt格式模型，命名为"cifar10_pre_checkpoint_based_imagenet21k.ckpt"，和训练集测试集数据放于同一级目录下：

 ```text
└─dataset
    ├─cifar10
        ├─cifar-10-batches-bin
        └─cifar-10-verify-bin
    └─cifar10_pre_checkpoint_based_imagenet21k.ckpt
```

注：你可以使用 .npz 格式的文件，但是仅适用于ViT-base-16。为此，请将`config.py`文件的`checkpoint_path`参数改为 .npz 文件的路径。

- Ascend or GPU 处理器环境运行

  ```python
  # 运行训练示例
  # Ascend
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATASET_NAME]
  # GPU
  bash ./scripts/run_standalone_train_gpu.sh [DATASET_NAME] [DEVICE_ID] [LR_INIT] [LOGS_CKPT_DIR]

  # 运行分布式训练示例
  # Ascend
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE] [DEVICE_NUM] [RANK_SIZE] [DATASET_NAME]
  # GPU
  bash ./scripts/run_distribute_train_gpu.sh [DATASET_NAME] [DEVICE_NUM] [LR_INIT] [LOGS_CKPT_DIR]

  # 运行评估示例
  # Ascend
  bash ./scripts/run_standalone_eval_ascend.sh [CKPT_PATH]
  # GPU
  bash ./scripts/run_standalone_eval_gpu.sh [DATASET_NAME] [DEVICE_ID] [CKPT_PATH]

  # 运行推理示例
  bash run_infer_310.sh ../vit_base.mindir Cifar10 /home/dataset/cifar-10-verify-bin/ 0
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用多卡训练 cifar10 数据集

      ```python
      # (1) 在网页上设置AI引擎为MindSpore
      # (2) 在网页上设置 "ckpt_url=obs://path/pre_ckpt/"（预训练模型命名为"cifar10_pre_checkpoint_based_imagenet21k.ckpt"）
      #     在网页上设置 "modelarts=True"
      #     在网页上设置 其他参数
      # (3) 上传你的数据集到 S3 桶上
      # (4) 在网页上设置你的代码路径为 "/path/vit_base"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集（如/dataset/cifar10/cifar-10-batches-bin/）"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
├── vit_base
    ├── README.md                              // vit_base相关说明 (ENG)
    ├── README_CN.md                           // vit_base相关说明 (CN)
    ├── ascend310_infer                        // 实现310推理源代码
    ├── scripts
    │   ├──run_distribute_train_ascend.sh    // 分布式到Ascend的shell脚本
    │   ├──run_distribute_train_gpu.sh       // 分布式到GPU的shell脚本
    │   ├──run_infer_310.sh                    // Ascend推理的shell脚本
    │   ├──run_standalone_eval_ascend.sh       // Ascend评估的shell脚本
    │   ├──run_standalone_eval_gpu.sh          // GPU评估的shell脚本
    │   ├──run_standalone_train_ascend.sh      // Ascend单卡训练的shell脚本
    │   └──run_standalone_train_gpu.sh         // GPU单卡训练的shell脚本
    ├── src
    │   ├──config.py                           // 参数配置
    │   ├──dataset.py                          // 创建数据集
    │   ├──modeling_ms.py                      // vit_base架构
    │   └──net_config.py                       // 结构参数配置
    │   └──npz_converter.py                    // 将 .npz 检查点转换为 .ckpt
    ├── eval.py                                // 评估脚本
    ├── export.py                              // 将checkpoint文件导出到air/mindir
    ├── postprocess.py                         // 310推理后处理脚本
    ├── preprocess.py                          // 310推理前处理脚本
    └── train.py                               // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置vit_base和CIFAR-10数据集。

  ```python
  'name':'cifar10'         # 数据集
  'pre_trained':True       # 是否基于预训练模型训练
  'num_classes':10         # 数据集类数
  'lr_init':0.013          # 初始学习率，双卡并行训练
  'batch_size':32          # 训练批次大小
  'epoch_size':60          # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'/dataset/cifar10/cifar-10-batches-bin/'     # 训练数据集的绝对全路径
  'val_data_path':'/dataset/cifar10/cifar-10-verify-bin/'  # 评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':0            # 用于训练或评估数据集的设备ID，进行分布式训练时可以忽略
  'keep_checkpoint_max':2  # 最多保存2个ckpt模型文件
  'checkpoint_path':'/dataset/cifar10_pre_checkpoint_based_imagenet21k.ckpt'  # 保存预训练模型的绝对全路径
  # optimizer and lr related
  'lr_scheduler':'cosine_annealing'
  'T_max':50
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend or GPU 处理器环境运行

  ```bash
  # Ascend
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATASET_NAME]
  # GPU
  bash ./scripts/run_standalone_train_gpu.sh [DATASET_NAME] [DEVICE_ID] [LR_INIT] [LOGS_CKPT_DIR]
  ```

  上述python命令将在后台运行，可以通过生成的train.log文件查看结果。

  训练结束后，可以在默认脚本文件夹下得到损失值：

  ```bash
  Load pre_trained ckpt: ./cifar10_pre_checkpoint_based_imagenet21k.ckpt
  epoch: 1 step: 1562, loss is 0.12886986
  epoch time: 289458.121 ms, per step time: 185.312 ms
  epoch: 2 step: 1562, loss is 0.15596801
  epoch time: 245404.168 ms, per step time: 157.109 ms
  {'acc': 0.9240785256410257}
  epoch: 3 step: 1562, loss is 0.06133139
  epoch time: 244538.410 ms, per step time: 156.555 ms
  epoch: 4 step: 1562, loss is 0.28615832
  epoch time: 245382.652 ms, per step time: 157.095 ms
  {'acc': 0.9597355769230769}
  ```

  注意：如果您想在训练期间验证模型，请在 `train.py` 中设置标志 `--do_val=True`
### 分布式训练

- Ascend or GPU 处理器环境运行

  ```bash
  # Ascend
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE] [DEVICE_NUM] [RANK_SIZE] [DATASET_NAME]
  # GPU
  bash ./scripts/run_distribute_train_gpu.sh [DATASET_NAME] [DEVICE_NUM] [LR_INIT] [LOGS_CKPT_DIR]
  ```

  上述shell脚本将在后台运行分布训练。

  训练结束后，可以得到损失值：

  ```bash
  Load pre_trained ckpt: ./cifar10_pre_checkpoint_based_imagenet21k.ckpt
  epoch: 1 step: 781, loss is 0.015172593
  epoch time: 195952.289 ms, per step time: 250.899 ms
  epoch: 2 step: 781, loss is 0.06709316
  epoch time: 135894.327 ms, per step time: 174.000 ms
  {'acc': 0.9853766025641025}
  epoch: 3 step: 781, loss is 0.050968178
  epoch time: 135056.020 ms, per step time: 172.927 ms
  epoch: 4 step: 781, loss is 0.01949552
  epoch time: 136084.816 ms, per step time: 174.244 ms
  {'acc': 0.9854767628205128}
  ```

## 评估过程

### 评估

- 在Ascend or GPU 环境运行时评估CIFAR-10数据集

  ```bash
  # Ascend
  bash ./scripts/run_standalone_eval_ascend.sh [CKPT_PATH]
  # GPU
  bash ./scripts/run_standalone_eval_gpu.sh [DATASET_NAME] [DEVICE_ID] [CKPT_PATH]
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。

  ```shell
  python export.py --ckpt_file [CKPT_FILE]
  ```

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用CIFAR-10数据集进行推理

  执行推理的命令如下所示，其中'MINDIR_PATH'是mindir文件路径；'DATASET'是使用的推理数据集名称，为'Cifar10'；'DATA_PATH'是推理数据集路径；'DEVICE_ID'可选，默认值为0。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [DEVICE_ID]
  ```

  推理的精度结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的分类准确率结果。推理的性能结果保存在scripts/time_Result目录下，在test_perform_static.txt文件中可以找到类似以下的性能结果。

  ```shell
  after allreduce eval: top1_correct=9854, tot=10000, acc=98.54%
  NN inference cost average time: 52.2274 ms of infer_count 10000
  ```

# 模型描述

## 性能

### 评估性能

#### CIFAR-10上的vit_base

| 参数                 | Ascend                                                      | GPU |
| -------------------------- | ------------------------|----------------------------------- |
| 模型版本              | vit_base                                                |vit_base   |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Red Hat 8.3.1-5         | 1 Nvidia Tesla V100-PCIE, CPU 3.40GHz; 8 Nvidia RTX 3090, CPU 2.90GHz |
| 上传日期              | 2021-10-26                                 | 2021-11-29 |
| MindSpore版本          | 1.3.0                                                 | 1.5.0 |
| 数据集                    | CIFAR-10                                                | CIFAR-10 |
| 训练参数        | epoch=60, batch_size=32, lr_init=0.013（双卡并行训练时）             | epoch=60, batch_size=32, lr_init=0.0065 or 0.052（1 or 8 GPUs）|
| 优化器                  | Momentum                                                    | Momentum |
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵 |
| 输出                    | 概率                                                 | 概率 |
| 分类准确率             | 双卡：98.99%               | 单卡: 98.84%; 八卡: 98.83%
| 速度                      | 单卡：157毫秒/步；八卡：174毫秒/步                        | 单卡：467毫秒/步；八卡：469毫秒/步 |
| 总时长                 | 双卡：2.48小时/60轮                                             | 单卡: 12.1小时/60轮; 八卡: 1.52小时/60轮 |

### 推理性能

#### CIFAR-10上的vit_base

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | vit_base                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-10-26                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | CIFAR-10                                                |
| 分类准确率             | 98.54%                       |
| 速度                      | NN inference cost average time: 52.2274 ms of infer_count 10000           |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models) 。
