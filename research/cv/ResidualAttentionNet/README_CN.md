# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [ResidualAttentionNet描述](#ResidualAttentionNet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
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
        - [训练性能](#训练性能)
            - [CIFAR-10上训练ResidualAttentionNet](#cifar-10上训练ResidualAttentionNet)
            - [ImageNet2012上训练ResidualAttentionNet](#imagenet2012上训练ResidualAttentionNet)
        - [评估性能](#评估性能)
            - [CIFAR-10上评估ResidualAttentionNet](#cifar-10上评估ResidualAttentionNet)
            - [ImageNet2012上评估ResidualAttentionNet](#imagenet2012上评估ResidualAttentionNet)
    - [使用流程](#使用流程)
        - [推理](#推理-1)
        - [继续训练预训练模型](#继续训练预训练模型)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ResidualAttentionNet描述

ResidualAttentionNet这项工作中，提出了残差注意网络，使用注意机制的卷积神经网络，它可以在端到端训练方式中与先进的前馈网络架构相结合。残差注意网络是由产生注意感知特征的注意模块叠加而成。不同模块的注意感知特征随着层次的加深而自适应地变化。在每个注意模块内部，采用自底向上自顶向下的前馈结构，将前馈和反馈的注意过程展开为单个前馈过程。重要的是，使用注意残差来训练非常深的残差注意网络可以很容易地扩展到数百层。

[论文](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf)：Residual Attention Network for Image Classification (CVPR-2017 Spotlight) By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang

# 模型架构

模型由多个残差模块和注意模块堆叠。对于ResidualAttentionModel_92由AttentionModule_stage1x1，AttentionModule_stage2x2，AttentionModule_stage3x3和多个残差结构堆叠。对于ResidualAttentionModel_56由AttentionModule_stage1x1，AttentionModule_stage2x1，AttentionModule_stage3x1和多个残差结构堆叠。

# 数据集

使用的数据集：[CIFAR-10](https://gitee.com/link?target=http%3A%2F%2Fwww.cs.toronto.edu%2F~kriz%2Fcifar.html)

- 数据集大小：175M，共10个类、6万张32*32彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理，数据路径data_path需要在config中指定。

使用的数据集：[ImageNet2012](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org%2F)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据将在src/dataset.py中处理，数据路径data_path需要在config中指定。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall%2Fen)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
    - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fapi%2Fzh-CN%2Fmaster%2Findex.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 添加数据集路径,以训练cifar10为例
  data_path:/home/DataSet/cifar10/
  # 添加配置文件路径
  config_path:/home/config.yaml
  # 推理前添加checkpoint路径参数
  checkpoint_file_path:path_to_ckpt/cifar10-300.ckpt
  ```

  ```bash
  # 运行训练示例
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > train.log &
  或
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml
  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh rank_table.json /data/cifar10/ ../config/cifar10_Ascend_8p_config.yaml
  # 运行评估示例
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > eval.log &
  # example: python  eval.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > eval.log &
  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh cifar10-300.mindir cifar10 /data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml 0
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.

- CPU处理器环境运行

  为了在CPU处理器环境运行，请将配置文件[dataset]_config.yaml中的`device_target`从`Ascend`改为`CPU`

  ```bash
  # 运行训练示例
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > train.log &
  或
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_CPU_config.yaml
  # 运行评估示例
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > eval.log &
  # example: python  eval.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > eval.log &
  ```

默认使用CIFAR-10数据集。您也可以将`$dataset`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://gitee.com/link?target=https%3A%2F%2Fsupport.huaweicloud.com%2Fmodelarts%2F))

    - 在 ModelArts 云脑上使用8卡训练 ImageNet 数据集

      ```bash
      # (1) 在网页上设置启动文件为train.py
      # (2) 在网页上设置数据集为imagenet.zip
      # (3) 在网页上设置"config_path='/path_to_code/imagenet2012_Modelart_config.yaml'"
      # (4) 在网页上设置"lr_init"、"epoch_size"、"lr_decay_mode"等
      # (5) 在网页上设置使用8卡
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 ImageNet 数据集

      ```bash
      # (1) 在网页上设置启动文件为eval.py
      # (2) 在网页上设置数据集为imagenet.zip
      # (3) 在网页上设置"config_path='/path_to_code/imagenet2012_Modelart_config.yaml'"
      # (4) 在网页上设置"chcekpoint_file_path='path_to_ckpt/imagenet2012.ckpt'"
      # (5) 创建训练作业
      ```

    - 在 ModelArts 上使用8卡训练 cifar10 数据集

      ```bash
      # (1) 在网页上设置启动文件为train.py
      # (2) 在网页上设置数据集为cifar10-bin.zip
      # (3) 在网页上设置"config_path='/path_to_code/cifar10_Modelart_8p_config.yaml'"
      # (4) 在网页上设置"lr"
      # (5) 在网页上设置使用8卡
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 cifar10 数据集

      ```bash
      # (1) 在网页上设置启动文件为eval.py
      # (2) 在网页上设置数据集为cifar10-bin.zip
      # (3) 在网页上设置"config_path='/path_to_code/cifar10_Modelart_1p_config.yaml'"
      # (4) 在网页上设置"chcekpoint_file_path='path_to_ckpt/cifar10.ckpt'"
      # (5) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡导出 cifar10 数据集

      ```bash
      # (1) 在网页上设置启动文件为export.py
      # (2) 在网页上设置数据集为cifar10-bin.zip
      # (3) 在网页上设置"config_path='/path_to_code/cifar10_Modelart_1p_config.yaml'"
      # (4) 在网页上设置"ckpt_file='path_to_ckpt/cifar10.ckpt'"
      # (5) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── residual_attention_net
        ├── README.md                    // residual_attention_net相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── config
        │   ├──cifar10_Ascend_1p_config.yaml      // cifar10数据集在Ascend上单卡的配置文件
        │   ├──cifar10_Ascend_8p_config.yaml      // cifar10数据集在Ascend上八卡的配置文件
        │   ├──cifar10_Modelart_1p_config.yaml    // cifar10数据集在Modelart上单卡的配置文件
        │   ├──cifar10_Modelart_8p_config.yaml    // cifar10数据集在Modelart上八卡的配置文件
        │   ├──cifar10_CPU_config.yaml            // cifar10数据集在CPU上的配置文件
        │   ├──imagenet2012_Ascend_config.yaml    // imagenet2012数据集在Ascend上的配置文件
        │   ├──imagenet2012_Modelart_config.yaml  // imagenet2012数据集在Modelart上的配置文件
        ├── model                                 // 模型框架
        │   ├──attention_module.py
        │   ├──basic_layers.py
        │   ├──residual_attention_network.py
        ├── scripts
        │   ├──run_distribute_train.sh        // 分布式8p到Ascend的shell脚本
        │   ├──run_standalone_train.sh         // 单卡到Ascend的shell脚本
        │   ├──run_infer_310.sh                // Ascend310推理的shell脚本
        ├── src
        │   ├──model_utils                    // 相关配置
        │   ├──conv2d_ops.py                  // 卷积算子转换
        │   ├──cross_entropy_loss_ops.py      // 损失函数
        │   ├──dataset.py                     // 数据处理
        │   ├──eval_callback.py               // eval_callback设置
        │   ├──local_adapter.py               // 本地设置
        │   ├──lr_generator.py                // 学习率设置
        │   ├──moxing_adapter.py              // moxing设置
        ├── train.py               // 训练脚本
        ├── eval.py                // 评估脚本
        ├── postprogress.py        // 310推理后处理脚本
        ├── export.py              // 将checkpoint文件导出到air/mindir
        ├── preprocess.py          // 310推理前处理脚本
        ├── create_imagenet2012_label.py    // 310推理imagenet2012数据的标签处理
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 单卡配置ResidualAttentionNet和CIFAR-10数据集。

  ```bash
  'lr':0.1                 # 初始学习率
  'batch_size':64          # 训练批次大小
  'epoch_size':220         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':32        # 输入到模型的图像高度
  'image_width':32         # 输入到模型的图像宽度
  'data_path':'./data'     # 训练和评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':4            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':10 # 最多保存checkpoint文件的数量
  'checkpoint_file_path':path_to_ckpt/cifar10-300.ckpt  # 推理前添加checkpoint路径参数
  ```

- 八卡配置ResidualAttentionNet和CIFAR-10数据集。

  ```bash
  'lr':0.24                # 初始学习率
  'batch_size':64          # 训练批次大小
  'epoch_size':220         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':32        # 输入到模型的图像高度
  'image_width':32         # 输入到模型的图像宽度
  'data_path':'./data'     # 训练和评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':4            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':10 # 最多保存checkpoint文件的数量
  'checkpoint_file_path':path_to_ckpt/cifar10-300.ckpt  # 推理前添加checkpoint路径参数
  ```

- 八卡配置ResidualAttentionNet和ImageNet2012数据集。

  ```bash
  'lr_init': 0.24           # 初始学习率
  'batch_size': 32          # 训练批次大小
  'epoch_size': 60          # 总计训练epoch数
  'momentum': 0.9           # 动量
  'weight_decay': 1e-4      # 权重衰减值
  'image_height': 224       # 输入到模型的图像高度
  'image_width': 224        # 输入到模型的图像宽度
  'data_path':'./data'     # 训练和评估数据集的绝对全路径
  'device_target': 'Ascend' # 运行程序的目标设备
  'device_id': 0            # 训练或者评估使用的设备卡号。 如果是分布式训练，忽略该参数。
  'keep_checkpoint_max': 10 # 最多保存checkpoint文件的数量
  'checkpoint_file_path':path_to_ckpt/imagenet2012-300.ckpt  # 推理前添加checkpoint路径参数
  'lr_scheduler': 'exponential'     # 学习率调度器
  'warmup_epochs': 0         # 学习率预热epoch数
  'loss_scale': 1024         # loss scale
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ > train.log &
  或
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```text
  # grep "loss is " train.log
  epoch:1 step:97, loss is 1.4842823
  epcoh:2 step:97, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

- CPU处理器环境运行

  ```bash
  # 运行训练示例
  python train.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] > train.log &
  # example: python train.py --config_path config/cifar10_CPU_config.yaml --data_path /data/cifar10/ > train.log &
  或
  bash scripts/run_standalone_train.sh [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh /data/cifar10/ ../config/cifar10_CPU_config.yaml
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在yaml文件中配置的文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CONFIG_PATH]
  # example: bash scripts/run_standalone_train.sh rank_table.json /data/cifar10/ ../config/cifar10_Ascend_8p_config.yaml
  ```

  上述shell脚本将在后台运行分布训练。您可以通过log文件查看结果。采用以下方式达到损失值：

  ```text
  log:epoch:1 step:48, loss is 1.4302931
  log:epcoh:2 step:48, loss is 1.4023874
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径checkpoint_file_path，例如“username/RedidualAttentionNet/mindspore_cifar10.ckpt”。

  ```bash
  python eval.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] --checkpoint_file_path [CHECKPOINT_FILE_PATH] > eval.log &
  # example: python eval.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ --checkpoint_file_path cifar10-1p.ckpt > eval.log &
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  grep "accuracy" eval.log
  # accuracy：{'top_1_accuracy':0.9952 'top_5_accuracy:0.9978'}
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件，Cifar10的配置文件为config/cifar10_Ascend_config.yaml，imagenet的配置文件为config/imagenet2012_Ascend_config.yaml. 需要修改配置项原始文件名 ckpt_file，导出文件名file_name。

```bash
python export.py --config_path [CONFIG_PATH] --data_path [DATA_PATH] --ckpt_file [CKPT_FILE] --file_name [FILE_NAME]
# python export.py --config_path config/cifar10_Ascend_1p_config.yaml --data_path /data/cifar10/ --ckpt_file cifar10-1p.ckpt --file_name ResidualAttentionNet92-cifar10_1
```

## 推理过程

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用CIFAR-10数据集进行推理

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh cifar10-300.mindir cifar10 /data/cifar10/ ../config/cifar10_Ascend_1p_config.yaml 0
  ```

- 推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```bash
  Total data:10000, top1 accuracy:0.9514, top5 accuracy:0.9978.
  ```

# 模型描述

## 性能

### 训练性能

#### CIFAR-10上训练ResidualAttentionNet

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | V1                                                           |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8     |
| 上传日期      | 2022-05-29                                                   |
| MindSpore版本 | 1.5.1                                                        |
| 数据集        | CIFAR-10                                                     |
| 训练参数      | epoch=220, steps=97, batch_size = 64, lr=0.24(8p)            |
| 优化器        | Momentum                                                     |
| 损失函数      | Softmax交叉熵                                                |
| 输出          | 概率                                                         |
| 损失          | 0.0003                                                       |
| 速度          | 8卡：72毫秒/步                                               |
| 总时长        | 8卡：46分钟                                                  |
| 参数(M)       | 51.3                                                         |
| 微调检查点    | 153M (.ckpt文件)                                             |
| 脚本          | [residual_attention_net脚本](https://gitee.com/fuguidan/models/tree/master/research/cv/ResidualAttentionNet) |

#### ImageNet2012上训练ResidualAttentionNet

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | V1                                                           |
| 资源          | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8      |
| 上传日期      | 2022-05-29                                                   |
| MindSpore版本 | 1.5.0                                                        |
| 数据集        | ImageNet2012                                                 |
| 训练参数      | epoch=60, steps=5004, batch_size=32, lr=0.24(8p)             |
| 优化器        | Momentum                                                     |
| 损失函数      | Softmax交叉熵                                                |
| 输出          | 概率                                                         |
| 损失          | 0.5                                                          |
| 速度          | 8卡：109毫秒/步                                              |
| 总时长        | 8卡：10.16小时                                               |
| 参数(M)       | 31.9                                                         |
| 微调检查点    | 657M (.ckpt文件)                                             |
| 脚本          | [residual_attention_net脚本](https://gitee.com/fuguidan/models/tree/master/research/cv/ResidualAttentionNet) |

### 评估性能

#### CIFAR-10上评估ResidualAttentionNet

| 参数           | Ascend                    |
| -------------- | ------------------------- |
| 模型版本       | V1                        |
| 资源           | Ascend 910；系统 Euler2.8 |
| 上传日期       | 2022-05-29                |
| MindSpore 版本 | 1.5.1                     |
| 数据集         | CIFAR-10, 1万张图像       |
| batch_size     | 64                        |
| 输出           | 概率                      |
| 准确性         | 单卡: 95.4%; 8卡：95.2%   |

#### ImageNet2012上评估ResidualAttentionNet

| 参数          | Ascend                    |
| ------------- | ------------------------- |
| 模型版本      | V1                        |
| 资源          | Ascend 910；系统 Euler2.8 |
| 上传日期      | 2022-05-29                |
| MindSpore版本 | 1.5.1                     |
| 数据集        | ImageNet2012              |
| batch_size    | 32                        |
| 输出          | 概率                      |
| 准确性        | 8卡: 77.5%                |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/docs/zh-CN/r1.7/migration_guide/inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=config.device_target)
  context.set_context(device_id=config.device_id)
  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(config.data_path)
  # 定义模型
  net = ResidualAttentionNet()
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 config.momentum, weight_decay=config.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean',
                                          is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})
  # 加载预训练模型
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)
  # 对未知数据集进行预测
  acc = model.eval(dataset)
  print("accuracy:", acc)
  ```

### 继续训练预训练模型

- Ascend处理器环境运行

  ```python
  # 加载数据集
  dataset = create_dataset(config.data_path)
  batch_num = dataset.get_dataset_size()
  # 定义模型
  net = ResidualAttentionNet()
  # 若pre_trained为True，继续训练
  if config.pre_trained:
      param_dict = load_checkpoint(config.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=config.lr_init, total_epochs=config.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), config.momentum, weight_decay=config.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)
  # 设置回调
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()
  # 开始训练
  model.train(config.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

# 随机情况说明

# 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。