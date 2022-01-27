# 目录

<!-- TOC -->

- [目录](#目录)
- [InceptionV2描述](#inceptionv2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度ascend)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的Inceptionv2](#imagenet-1k上的Inceptionv2)
        - [推理性能](#推理性能)
            - [ImageNet-1k上的Inceptionv2](#imagenet-1k上的Inceptionv2)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# InceptionV2描述

Google的InceptionV2是深度学习卷积架构系列的第2个版本。InceptionV2主要通过修改以前的Inception架构来减少计算资源的消耗。这个想法是在2015年出版的Rethinking the Inception Architecture for Computer Vision, published in 2015一文中提出的。

[论文](https://arxiv.org/pdf/1512.00567.pdf)： Min Sun, Ali Farhadi, Steve Seitz.Ranking Domain-Specific Highlights by Analyzing Edited Videos[J].2014.

# 模型架构

InceptionV2的总体网络架构如下：

[链接](https://arxiv.org/pdf/1512.00567.pdf)

# 数据集

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

# 特性

## 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
- 使用Ascend来搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```shell
.
└─Inception-v2
  ├─README_CN.md
  ├─ascend310_infer                           # 实现310推理源代码
  ├─scripts
    ├─run_standalone_train.sh                 # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh                 # 启动Ascend分布式训练（8卡）
    ├─run_infer_310.sh                        # Ascend推理shell脚本
    └─run_eval.sh                             # 启动Ascend评估
  ├─src
    ├─dataset.py                      # 数据预处理
    ├─inception_v2.py                 # 网络定义
    ├─loss.py                         # 自定义交叉熵损失函数
    ├─lr_generator.py                 # 学习率生成器
    └─config.py                       # 获取gpu、Ascend、cpu配置参数

  ├─eval.py                           # 评估网络
  ├─export.py                         # 导出 AIR,MINDIR模型的脚本
  ├─postprogress.py                   # 310推理后处理脚本
  └─train.py                          # 训练网络
```

## 脚本参数

```python
#train.py和config.py中主要参数如下：
'platform'                   # 运行平台
'random_seed'                # 修复随机种子
'rank'                       # 分布式的本地序号
'group_size'                 # 分布式进程总数
'work_nums'                  # 读取数据的worker个数
'decay_method'               # 学习率调度器模式
"loss_scale"                 # 损失等级
'batch_size'                 # 输入张量的批次大小
'epoch_size'                 # 总轮次数
'num_classes'                # 数据集类数
'ds_type'                    # 数据集类型，如：imagenet, cifar10
'ds_sink_mode'               # 使能数据下沉
'smooth_factor'              # 标签平滑因子
'lr_init'                    # 初始学习率
'lr_max'                     # 最大学习率
'lr_end'                     # 最小学习率
'warmup_epochs'              # 热身轮次数
'weight_decay'               # 权重衰减
'momentum'                   # 动量
'opt_eps'                    # epsilon
'keep_checkpoint_max'        # 保存检查点的最大数量
'ckpt_path'                  # 保存检查点路径
'is_save_on_master'          # 保存Rank0的检查点，分布式参数
'dropout_keep_prob'          # 保持率，介于0和1之间，例如keep_prob = 0.9，表示放弃10%的输入单元
'has_bias'                   # 层是否使用偏置向量
'amp_level'                  # `mindspore.amp.build_train_network`中参数`level`的选项，level表示混合
                             # 精准训练支持[O0, O2, O3]

```

更多配置细节请参考脚本`config.py`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_num 1 --platform Ascend  --data_url ./Imagenet --train_url ./Train_out > train.log 2>&1 &

  # 使用脚本启动单卡训练
  bash scripts/run_standalone_train.sh  [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional)

  # 使用脚本启动多卡训练
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [PRE_TRAINED_PATH](optional)

  # 使用python启动单卡运行评估示例
  python eval.py --data_url ./Imagenet --platform Ascend --checkpoint ./ckpt_0/inceptionv2-rank0-250_1251.ckpt > ./eval.log 2>&1 &

  # 使用脚本启动单卡运行评估示例
  bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME(imagenet2012)] [DATASET_PATH] [DEVICE_ID(optional)]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 导出过程

### 导出

  ```shell
  python export.py --ckpt_file [CKPT_FILE]  --device_target [DEVICE_TARGET] --file_format [FILE FORMAT]
  ```

导出的模型会以模型的结构名字命名并且保存在当前目录下

## 推理过程

### 推理

在进行推理之前我们需要先导出模型。mindir可以在任意环境上导出，air模型只能在昇腾910环境上导出。以下展示了使用mindir模型执行推理的示例。

- 在昇腾310上使用ImageNet-1k数据集进行推理

  推理的结果保存在scripts目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 76.22%
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的Inceptionv2

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型|Inceptionv2|
| 资源                   | Ascend 910               |
| 上传日期              | 2021-12-25                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像                                              |
| 训练参数        | epoch=250, batch_size=128            |
| 优化器                  | Momentum                                                    |
| 损失函数              | CrossEntropy                                      |
| 损失| 0.8279|
| 输出                    | 概率                                                 |
| 分类准确率             | 八卡：top1:76.22% top5:93.04%                   |
| 速度                      | 八卡：91毫秒/步                        |
| 训练耗时          |08h54min58s（8pcs, run on ModelArts）|

### 推理性能

#### ImageNet-1k上的Inceptionv2

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
|模型                 |Inceptionv2|                                                |
| 资源                   | Ascend 310               |
| 上传日期              | 2021-12-25                                 |
| MindSpore版本          | 1.3.0                                                 |
| 数据集                    | ImageNet-1k Val，共50,000张图像                                                 |
| 分类准确率             | top1:76.22% top5:93.04%                     |
| 速度                      | 平均耗时2.58ms每张|
| 推理耗时| 约22min|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)