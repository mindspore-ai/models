# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [TSM描述](#TSM描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
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
        - [训练性能](#训练性能)
            - [SomethingSometing-v2上训练TSM](#SomethingSometing-v2上训练TSM)
        - [评估性能](#评估性能)
            - [SomethingSometing-v2上评估TSM](#SomethingSometing-v2上评估TSM)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# TSM描述

TSM(Temporal Shift Module)网络在线视频流的爆炸式增长对有效提取时空信息进行视频理解提出了挑战。传统的二维神经网络计算成本低，但不能捕捉长期的时间关系;基于3D CNN的方法可以获得较好的性能，但计算量大，因此部署成本较高。在TSM中，我们提出了一种通用而有效的时间转移模块(TSM)，它具有高效率和高性能。具体来说，它可以达到3D CNN的性能，但保持2d复杂度。TSM的核心思想是将部分信道沿时间维进行移位，便于相邻帧之间的信息交换。在以时间建模为核心的SomethingSomething-v1数据集上，更少的FLOPs (每秒浮点运算次数)，获得了比i3d系列和eco系列更好的结果。在P100 GPU上测量，与I3D相比，在单一模型在8×较低的延迟和12×较高的吞吐量下，提高了1.8%精度。在本论文提交时，TSM网络在V1和V2排行榜上都排名第一。

[论文](https://arxiv.org/abs/1811.08383v1)：Lin, Ji, Chuang Gan and Song  Han. “TSM: Temporal Shift Module for Efficient Video Understanding.” 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 7082-7092.

# 模型架构

TSM应用了一种通用而有效的时间转移模块。  时间转移模块将多张连续图像卷积后的通道互相交换，以达到交换视频时域信息的目的。

# 数据集

使用的数据集：[SomethingSometing-v2](https://download.mindspore.cn/dataset/somethingv2.tar)

- 数据集大小：281G，共174个类、220,847段视频
    - 训练集：共168,913段视频
    - 测试集：共24,777段视频

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend/GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 添加数据集路径,以训练SomethingSometing-v2为例
  data_path:"/home/DataSet/"

  # 推理前添加checkpoint路径参数
  checkpoint_path:"./checkpoint/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e50_None/ckpt_0/"
  test_filename:"tsm-50_2639.ckpt"
  ```

  ```python
  # 运行训练示例
  python train.py --config_path=../config/tsm_sthv2_config_ascend.yaml > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_train_ascend.sh
  # example: bash scripts/run_train.sh

  # 运行评估示例
  bash scripts/run_eval.sh [checkpoint_path] [checkpoint_name]
  # example: python eval.py

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

  ```python
  # 运行训练示例
  bash scripts/run_train_gpu.sh 1 0 tsm_sthv2_config_gpu.yaml

  # 运行分布式训练示例
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7 tsm_sthv2_config_gpu.yaml

  # 运行评估示例
  bash scripts/run_eval.sh [checkpoint_path] [checkpoint_name]
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 SomethingSometing-v2 数据集

      ```python
      # (1) 在网页上设置 "config_path='../config/tsm_sthv2_config_ascend.yaml'"
      # (2) 执行a或者b
      #       a. 在 tsm_sthv2_config_ascend.yaml 文件中设置 "train_url=/tsm/cloud_unzip.py/sth_v2/V00xx/"
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 "data_url=/tsm/dataset1/"
      #          在 tsm_sthv3_config_ascend.yaml 文件中设置 "enable_modelarts=True"
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "train_url=/tsm/cloud_unzip.py/sth_v2/V00xx/"
      #          在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "data_url=/tsm/dataset1/"
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 obs 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/tsm/tsm/"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 SomethingSometing-v2 数据集

      ```python
      # (1) 在网页上设置 "config_path='../config/tsm_sthv2_config_ascend.yaml'"
      # (2) 执行a或者b
      #       a. 在 tsm_sthv2_config_ascend.yaml 文件中设置 "checkpoint_path: "/tsm/dataset1/""
      #          在 tsm_sthv3_config_ascend.yaml 文件中设置 "enable_modelarts=True"
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 "test_filename: "tsm-50.ckpt""
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 "data_url: "/tsm/dataset1/""
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "checkpoint_path=/tsm/dataset1/"
      #          在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "test_filename=tsm-50.ckpt"
      #          在网页上设置 "data_url=/tsm/dataset1/"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 obs 桶上
      # (4) 上传你的压缩数据集到 obs 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (5) 在网页上设置你的代码路径为 "/tsm/tsm/"
      # (6) 在网页上设置启动文件为 "eval.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡导出 SomethingSometing-v2 数据集

      ```python
      # (1) 在网页上设置 "config_path='../config/tsm_sthv2_config_ascend.yaml'"
      # (2) 执行a或者b
      #       a. 在 tsm_sthv2_config_ascend.yaml 文件中设置 "checkpoint_path: "/tsm/dataset1/""
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 "test_filename: "tsm-50.ckpt""
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 "train_url=/tsm/cloud_unzip.py/sth_v2/V00xx/"
      #          在 tsm_sthv3_config_ascend.yaml 文件中设置 "enable_modelarts=True"
      #          在 tsm_sthv2_config_ascend.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "checkpoint_path=/tsm/dataset1/"
      #          在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "test_filename=tsm-50.ckpt"
      #          在网页上设置 "train_url=/tsm/cloud_unzip.py/sth_v2/V00xx/"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (5) 在网页上设置你的代码路径为 "/tsm/tsm/"
      # (6) 在网页上设置启动文件为 "export.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash

├── tsm
    ├── eval.py   // 模型评估
    ├── preprocess.py   // 310推理数据预处理
    ├── postprocess.py   // 310推理结果后处理
    ├── export.py   //将checkpoint文件导出到air/mindir
    ├── README_CN.md    //TSM相关说明
    ├── scripts
    │   ├── run_train_ascend.sh   //ascend训练脚本
    │   ├── run_eval.sh   //评估脚本
    │   ├── run_export.sh   //导出脚本
    │   ├── run_infer_310.sh   //310推理脚本
    │   └── run_train_gpu.sh    //GPU训练脚本
    ├── src
    │   ├── config
    │   │   ├── tsm_sthv2_config_gpu.yaml   //GPU训练配置文件
    │   │   ├── tsm_sthv2_config_2bin.yaml   //数据集转bin
    │   │   └── tsm_sthv2_config_ascend.yaml   //ascend训练配置文件
    │   ├── model
    │   │   ├── cross_entropy_smooth.py   //交叉熵
    │   │   ├── net.py    //网络结构
    │   │   └── resnet.py   //基本网络
    │   ├── model_utils
    │   │   ├── config.py   //获取配置信息
    │   │   ├── device_adapter.py   //获取硬件信息
    │   │   ├── local_adapter.py    //线下硬件信息
    │   │   └── moxing_adapter.py   //云上硬件信息
    │   ├── tools
    │   │   ├── gen_label_sthv2.py    //生成sthv2数据集标签
    │   │   └── vid2img_sthv2.py    //sthv2数据集解码
    │   ├── ascend310_infer
    │   └── utils
    │       ├── consensus.py    //图像矩阵拼接
    │       ├── dataset_config.py   //数据集配置
    │       ├── dataset.py    //数据集加载
    │       ├── distributed_sampler.py    //数据集分配
    │       ├── hccl_tools.py   //多卡训练协同
    │       ├── lr_generator.py   //学习率生成
    │       ├── non_local.py    //网络结构生成
    │       ├── temporal_shift.py   //通道交换模块
    │       └── transforms.py   //图像预处理
    └── train.py    //训练模型
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --config_path=../config/tsm_sthv2_config_ascend.yaml > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过log.txt文件查看结果。
  同时将tsm_sthv2_config_ascend.yaml中data_path改为数据集所在位置，load_path改为预训练参数所在位置。
  训练结束后，您可在`./checkpoint`脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # vim ./log.txt
  epoch: 50 step: 2269, loss is 1.3481619
  epoch: 50 step: 2289, loss is 1.4777875
  ...
  ```

  模型检查点保存在当前生成的checkpoint目录下。

- GPU处理器环境运行

  ```bash
  bash scripts/run_train_gpu.sh 1 0 tsm_sthv2_config_gpu.yaml
  ```

  同时将tsm_sthv2_config_gpu.yaml中data_path改为数据集所在位置，load_path改为预训练参数所在位置。
  上述python命令将在后台运行，您可以通过log.txt文件查看结果。

  训练结束后，您可在默认`./checkpoint`脚本文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train_ascend.sh
  ```

  同时将tsm_sthv2_config_gpu.yaml中data_path改为数据集所在位置，load_path改为预训练参数所在位置。
  上述shell脚本将在后台运行分布训练。您可以通过./scripts/train_parallel/train_parallel[X]/log.txt文件查看结果。采用以下方式达到损失值：

  ```bash
  # vim ./scripts/train_parallel2/log.txt
  train_parallel0/log:epoch: 50 step: 2269, loss is 1.3481619
  train_parallel0/log:epoch: 50 step: 2289, loss is 1.4777875
  ...
  train_parallel1/log:epoch: 50 step: 2269, loss is 1.3481619
  train_parallel1/log:epoch: 50 step: 2289, loss is 1.4777875
  ...
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7 tsm_sthv2_config_gpu.yaml
  ```

  同时将tsm_sthv2_config_gpu.yaml中data_path改为数据集所在位置，load_path改为预训练参数所在位置。
  上述shell脚本将在后台运行分布训练。您可以通过train/log.txt文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估

  ```bash
  bash scripts/run_eval.sh [checkpoint_path] [checkpoint_name]
  ```

  测试数据集的准确性如下：

  ```bash
  result: {'top_5_accuracy': 0.8517441860465116, 'top_1_accuracy': 0.5833333333333334}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/tsm/checkpoint/tsm-19_2639.ckpt”。测试数据集的准确性如下：

  ```bash
  result: {'top_5_accuracy': 0.8517441860465116, 'top_1_accuracy': 0.5833333333333334}
  ```

- 在GPU处理器环境运行时评估

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/tsm/checkpoint/tsm-19_2639.ckpt”。

  ```bash
  bash scripts/run_eval.sh [checkpoint_path] [checkpoint_name]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  result: {'top_5_accuracy': 0.8517441860465116, 'top_1_accuracy': 0.5833333333333334}
  ```

## 导出过程

### 导出

```shell
bash scripts/run_export.sh [checkpoint_path] [checkpoint_name]
```

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用SomethingSometing-v2数据集进行推理

  在执行下面的命令之前，我们需要先修生成推理的bin文件

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  python preprocess.py --config_path=../config/tsm_sthv2_config_2bin.yaml
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  after allreduce eval: top1_correct=14431, tot=24777, acc=58.24%
  ```

  MINDIR_PATH：模型参数路径
  DATA_PATH: 数据集路径
  LABEL_FILE: 标签路径
  DEVICE_ID: 硬件路径
  tsm_sthv2_config_2bin.yaml: data_path:"真实路径"

# 模型描述

## 性能

### 训练性能

#### SomethingSometing-v2上训练TSM

| 参数                 | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | TSM                                                | TSM           |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | NV RTX3090       |
| 上传日期              | 2021-12-01                                 | 2021-12-01 |
| MindSpore版本          | 1.3.0                                                       | 1.5.0                  |
| 数据集                    | SomethingSometing-v2                                     | SomethingSometing-v2              |
| 训练参数        | epoch=50, steps=2639, batch_size = 64, lr=0.1              | epoch=50, steps=2639, batch_size = 64, lr=0.1    |
| 优化器                  | SGD                                                    | SGD               |
| 损失函数              | CrossEntropySmooth                                       | CrossEntropySmooth  |
| 输出                    | 概率                                                 | 概率            |
| 损失                       | 0.9808                                                    | 0.9703                 |
| 速度                      | 单卡：963.812毫秒/步;  8卡：832.684毫秒/步                          | 单卡：841.264毫秒/步;  8卡：809.137毫秒/步      |
| 总时长                 | 单卡：/;  8卡：31h18min4s                        | 单卡：/;  8卡：35h43min16s      |
| 微调检查点 | 275.85M (.ckpt文件)                                         | 275.85M (.ckpt文件)    |
| 推理模型        |  96.99M(.air文件)                     |      |
| 脚本                    | [TSM脚本](https://gitee.com/li-jianwei123/models) | [TSM脚本](https://gitee.com/li-jianwei123/models) |

### 评估性能

#### SomethingSometing-v2上评估TSM

| 参数          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | TSM                | TSM                |
| 资源            |  Ascend 910；系统 Euler2.8                  | GPU                         |
| 上传日期       | 2021-12-01 | 2021-12-01 |
| MindSpore 版本   | 1.3.0                       | 1.5.0                       |
| 数据集             | SomethingSometing-v2, 24777段视频     | SomethingSometing-v2, 24777段视频     |
| batch_size          | 64                         | 64                         |
| 输出             | 概率                 | 概率                 |
| 准确性            | 单卡: /;  8卡：58.65%   | 单卡：/; 8卡：58.63%      |
| 推理模型 | 161.76M (.om文件)         |  |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
