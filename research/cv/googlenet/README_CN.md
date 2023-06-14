# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [GoogleNet描述](#googlenet描述)
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
        - [训练性能](#训练性能)
            - [CIFAR-10上训练GoogleNet](#cifar-10上训练googlenet)
            - [ImageNet2012上训练GoogleNet](#imagenet2012上训练googlenet)
        - [评估性能](#评估性能)
            - [CIFAR-10上评估GoogleNet](#cifar-10上评估googlenet)
            - [ImageNet2012上评估GoogleNet](#imagenet2012上评估googlenet)
    - [使用流程](#使用流程)
        - [推理](#推理-1)
        - [继续训练预训练模型](#继续训练预训练模型)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GoogleNet描述

GoogleNet是2014年提出的22层深度网络，在2014年ImageNet大型视觉识别挑战赛（ILSVRC14）中获得第一名。  GoogleNet，也称Inception v1，比ZFNet（2013年获奖者）和AlexNet（2012年获奖者）改进明显，与VGGNet相比，错误率相对较低。  深度学习网络包含的参数更多，更容易过拟合。网络规模变大也会增加使用计算资源。为了解决这些问题，GoogleNet采用1*1卷积核来降维，从而进一步减少计算量。在网络末端使用全局平均池化，而不是使用全连接的层。  inception模块为相同的输入设置不同大小的卷积，并堆叠所有输出。

[论文](https://arxiv.org/abs/1409.4842)：Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich."Going deeper with convolutions."*Proceedings of the IEEE conference on computer vision and pattern recognition*.2015.

# 模型架构

GoogleNet由多个inception模块串联起来，可以更加深入。  降维的inception模块一般包括**1×1卷积**、**3×3卷积**、**5×5卷积**和**3×3最大池化**，同时完成前一次的输入，并在输出处再次堆叠在一起。

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、6万张32*32彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理。

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据将在src/dataset.py中处理。

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

  ```yaml
  # 添加数据集路径,以训练cifar10为例
  train_data_path:/home/DataSet/cifar10/
  val_data_path:/home/DataSet/cifar10/

  # 推理前添加checkpoint路径参数
  chcekpoint_path:/home/model/googlenet/ckpt/train_googlenet_cifar10-125_390.ckpt
  ```

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_train.sh [RANK_TABLE_FILE] [DATASET_NAME]
  # example: bash scripts/run_train.sh ~/hccl_8p.json cifar10

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh [DATASET_NAME]
  # example: bash run_eval.sh cifar10

  # 运行推理示例
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools.>

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件[dataset]_config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```python
  # 运行训练示例
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # 运行评估示例
  python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &
  OR
  bash run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

- CPU处理器环境运行

  ```python
  # 运行训练示例
  bash run_train_cpu.sh
  OR
  nohup python train.py --config_path=cifar10_config_cpu.yaml --dataset_name=cifar10  > train.log 2>&1 &

  # 运行评估示例
  bash run_eval_cpu.sh
  OR
  nohup python eval.py --checkpoint_path=[CHECKPOINT_PATH] --config_path=cifar10_config_cpu.yaml > eval.log 2>&1 &
  ```

默认使用CIFAR-10数据集。您也可以将`$dataset_type`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 ImageNet 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/imagenet_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 imagenet_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 imagenet_config.yaml 文件中设置 "dataset_name='imagenet'"
      #          在 imagenet_config.yaml 文件中设置 "train_data_path='/cache/data/ImageNet/train/'"
      #          在 imagenet_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "dataset_name=imagenet"
      #          在网页上设置 "train_data_path=/cache/data/ImageNet/train/"
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/path/googlenet"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 ImageNet 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/imagenet_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 imagenet_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 imagenet_config.yaml 文件中设置 "dataset_name='imagenet'"
      #          在 imagenet_config.yaml 文件中设置 "val_data_path='/cache/data/ImageNet/val/'"
      #          在 imagenet_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在 imagenet_config.yaml 文件中设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在 imagenet_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "dataset_name=imagenet"
      #          在网页上设置 "val_data_path=/cache/data/ImageNet/val/"
      #          在网页上设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在网页上设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (5) 在网页上设置你的代码路径为 "/path/googlenet"
      # (6) 在网页上设置启动文件为 "eval.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

    - 在 ModelArts 上使用8卡训练 cifar10 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/cifar10_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 cifar10_config.yaml 文件中设置 "dataset_name='cifar10'"
      #          在 cifar10_config.yaml 文件中设置 "train_data_path='/cache/data/'"
      #          在 cifar10_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "dataset_name=cifar10"
      #          在网页上设置 "train_data_path=/cache/data/"
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/path/googlenet"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 cifar10 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/cifar10_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 cifar10_config.yaml 文件中设置 "dataset_name='cifar10'"
      #          在 cifar10_config.yaml 文件中设置 "val_data_path='/cache/data/'"
      #          在 cifar10_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在 cifar10_config.yaml 文件中设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在 cifar10_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "dataset_name=cifar10"
      #          在网页上设置 "val_data_path=/cache/data/"
      #          在网页上设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在网页上设置 "checkpoint_path='/cache/checkpoint_path/model.ckpt'"
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (4) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (5) 在网页上设置你的代码路径为 "/path/googlenet"
      # (6) 在网页上设置启动文件为 "eval.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡导出 cifar10 数据集

      ```python
      # (1) 在网页上设置 "config_path='/path_to_code/cifar10_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 cifar10_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 cifar10_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_trained_ckpt/'"
      #          在 cifar10_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
      #          在 cifar10_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "checkpoint_url=s3://dir_to_trained_ckpt/" on the website UI interface.
      #          在网页上设置 "ckpt_file=/cache/checkpoint_path/model.ckpt" on the website UI interface.
      #          在网页上设置 其他参数
      # (3) 上传你的预训练模型到 S3 桶上
      # (5) 在网页上设置你的代码路径为 "/path/googlenet"
      # (6) 在网页上设置启动文件为 "export.py"
      # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (8) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── googlenet
        ├── README.md                    // googlenet相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_train_gpu.sh         // 分布式到GPU处理器的shell脚本
        │   ├──run_train_cpu.sh         // CPU处理器训练的shell脚本
        │   ├──run_eval.sh              // Ascend评估的shell脚本
        │   ├──run_infer_310.sh         // Ascend推理shell脚本
        │   ├──run_eval_gpu.sh          // GPU处理器评估的shell脚本
        │   ├──run_eval_cpu.sh          // CPU处理器评估的shell脚本
        ├── src
        │   ├──dataset.py             // 创建数据集
        │   ├──googlenet.py          //  googlenet架构
        │   ├──config.py            // 参数配置
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── postprogress.py       // 310推理后处理脚本
        ├── export.py            // 将checkpoint文件导出到air/mindir
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置GoogleNet和CIFAR-10数据集。

  ```python
  'pre_trained':'False'    # 是否基于预训练模型训练
  'nump_classes':10        # 数据集类数
  'lr_init':0.1            # 初始学习率
  'batch_size':128         # 训练批次大小
  'epoch_size':125         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':5e-4      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'./cifar10'  # 训练和评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':4            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':10 # 最多保存checkpoint文件的数量
  'checkpoint_path':'./train_googlenet_cifar10-125_390.ckpt'  # checkpoint文件保存的绝对全路径
  'onnx_filename':'googlenet.onnx' # export.py中使用的onnx模型文件名
  'geir_filename':'googlenet.geir' # export.py中使用的geir模型文件名
  ```

- 配置GoogleNet和ImageNet2012数据集。

  ```python
  'pre_trained': 'False'    # 是否基于预训练模型训练
  'num_classes': 1000       # 数据集类数
  'lr_init': 0.1            # 初始学习率
  'batch_size': 256         # 训练批次大小
  'epoch_size': 300         # 总计训练epoch数
  'momentum': 0.9           # 动量
  'weight_decay': 1e-4      # 权重衰减值
  'image_height': 224       # 输入到模型的图像高度
  'image_width': 224        # 输入到模型的图像宽度
  'data_path': './ImageNet_Original/train/'  # 训练数据集的绝对全路径
  'val_data_path': './ImageNet_Original/val/'  # 评估数据集的绝对全路径
  'device_target': 'Ascend' # 运行程序的目标设备
  'device_id': 0            # 训练或者评估使用的设备卡号。 如果是分布式训练，忽略该参数。
  'keep_checkpoint_max': 10 # 最多保存checkpoint文件的数量
  'checkpoint_path': './train_googlenet_cifar10-125_390.ckpt'  # 保存checkpoint文件的绝对全路径
  'onnx_filename': 'googlenet.onnx' # export.py使用的onnx模型的文件名
  'air_filename': 'googlenet.air'   # export.py使用的air模型的文件名
  'lr_scheduler': 'exponential'     # 学习率调度器
  'lr_epochs': [70, 140, 210, 280]  # lr改变的epoch数
  'lr_gamma': 0.3            # 指数衰减学习率调度器学习率衰减因子
  'eta_min': 0.0             # 余弦退火学习率调度器的最小学习率
  'T_max': 150               # 余弦退火学习率调度器cos周期的1/2
  'warmup_epochs': 0         # 学习率预热epoch数
  'is_dynamic_loss_scale': 0 # 是否是动态loss scale
  'loss_scale': 1024         # loss scale
  'label_smooth_factor': 0.1 # 标签平滑因子
  'use_label_smooth': True   # 标签平滑
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:390, loss is 1.4842823
  epcoh:2 step:390, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认`./ckpt_0/`脚本文件夹下找到检查点文件。

- CPU处理器环境运行

  ```bash
  nohup python train.py --config_path=cifar10_config_cpu.yaml --dataset_name=cifar10 > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在yaml文件中配置的文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train.sh ~/hccl_8p.json cifar10
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "result:" train_parallel*/log
  train_parallel0/log:epoch:1 step:48, loss is 1.4302931
  train_parallel0/log:epcoh:2 step:48, loss is 1.4023874
  ...
  train_parallel1/log:epoch:1 step:48, loss is 1.3458025
  train_parallel1/log:epcoh:2 step:48, loss is 1.3729336
  ...
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash scripts/run_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train/train.log文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/googlenet/train_googlenet_cifar10-125_390.ckpt”。

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh cifar10
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.934}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/googlenet/train_parallel0/train_googlenet_cifar10-125_48.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" dist.eval.log
  accuracy:{'acc':0.9217}
  ```

- 在GPU处理器环境运行时评估CIFAR-10数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/googlenet/train/ckpt_0/train_googlenet_cifar10-125_390.ckpt”。

  ```bash
  python eval.py --checkpoint_path=[CHECKPOINT_PATH] > eval.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.930}
  ```

  或者，

  ```bash
  bash run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval/eval.log
  accuracy:{'acc':0.930}
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件，Cifar10的配置文件为cifar10_config.yaml，imagenet的配置文件为imagenet_config.yaml.
需要修改的配置项为 batch_size 和 ckpt_file.

```shell
python export.py --config_path [CONFIG_PATH]
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用CIFAR-10数据集进行推理

  在执行下面的命令之前，我们需要先修改cifar10的配置文件。修改的项包括batch_size和val_data_path。LABEL_FILE参数只对ImageNet数据集有用，可以传任意值。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  after allreduce eval: top1_correct=9252, tot=10000, acc=92.52%
  ```

# 模型描述

## 性能

### 训练性能

#### CIFAR-10上训练GoogleNet

| 参数                 | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | Inception V1                                                | Inception V1           |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | NV SMX2 V100-32G       |
| 上传日期              | 2021-07-05                                 | 2021-07-05 |
| MindSpore版本          | 1.3.0                                                       | 1.3.0                  |
| 数据集                    | CIFAR-10                                                    | CIFAR-10               |
| 训练参数        | epoch=125, steps=390, batch_size = 128, lr=0.1              | epoch=125, steps=390, batch_size=128, lr=0.1    |
| 优化器                  | Momentum                                                    | Momentum               |
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵  |
| 输出                    | 概率                                                 | 概率            |
| 损失                       | 0.0016                                                      | 0.0016                 |
| 速度                      | 单卡：79毫秒/步;  8卡：82毫秒/步                          | 单卡：150毫秒/步;  8卡：164毫秒/步      |
| 总时长                 | 单卡：63.85分钟;  8卡：11.28分钟                          | 单卡：126.87分钟;  8卡：21.65分钟      |
| 参数(M)             | 13.0                                                        | 13.0                   |
| 微调检查点 | 43.07M (.ckpt文件)                                         | 43.07M (.ckpt文件)    |
| 推理模型        | 21.50M (.onnx文件),  21.60M(.air文件)                     |      |
| 脚本                    | [googlenet脚本](https://gitee.com/mindspore/models/tree/r2.0/research/cv/googlenet) | [googlenet 脚本](https://gitee.com/mindspore/models/tree/r2.0/research/cv/googlenet) |

#### ImageNet2012上训练GoogleNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | Inception V1                                                |
| 资源                   | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8               |
| 上传日期              | 2021-07-05                                 |
| MindSpore版本          | 1.3.0                                                       |
| 数据集                    | ImageNet2012                                                |
| 训练参数        | epoch=300, steps=5000, batch_size=256, lr=0.1               |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |
| 损失                       | 2.0                                                         |
| 速度                      | 单卡：152毫秒/步;  8卡：171毫秒/步                        |
| 总时长                 | 8卡：8.8小时                                             |
| 参数(M)             | 13.0                                                        |
| 微调检查点 | 52M (.ckpt文件)                                            |
| 脚本                    | [googlenet脚本](https://gitee.com/mindspore/models/tree/r2.0/research/cv/googlenet) |

### 评估性能

#### CIFAR-10上评估GoogleNet

| 参数          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | Inception V1                | Inception V1                |
| 资源            |  Ascend 910；系统 Euler2.8                  | GPU                         |
| 上传日期       | 2021-07-05 | 2021-07-05 |
| MindSpore 版本   | 1.3.0                       | 1.3.0                       |
| 数据集             | CIFAR-10, 1万张图像     | CIFAR-10, 1万张图像     |
| batch_size          | 128                         | 128                         |
| 输出             | 概率                 | 概率                 |
| 准确性            | 单卡: 93.4%;  8卡：92.17%   | 单卡：93%, 8卡：92.89%      |
| 推理模型 | 21.50M (.onnx文件)         |  |

#### ImageNet2012上评估GoogleNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | Inception V1                |
| 资源            |  Ascend 910；系统 Euler2.8                 |
| 上传日期       | 2021-07-05 |
| MindSpore版本   | 1.3.0                       |
| 数据集             | ImageNet2012                |
| batch_size          | 256                         |
| 输出             | 概率                 |
| 准确性            | 8卡: 71.81%                |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=cfg.device_target)
  context.set_context(device_id=cfg.device_id)

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # 定义模型
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
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

- GPU处理器环境运行

  ```python
  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target="GPU")

  # 加载未知数据集进行推理
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # 定义模型
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean',
                                          is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # 加载预训练模型
  param_dict = load_checkpoint(args_opt.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy:", acc)

  ```

### 继续训练预训练模型

- Ascend处理器环境运行

  ```python
  # 加载数据集
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # 定义模型
  net = GoogleNet(num_classes=cfg.num_classes)
  # 若pre_trained为True，继续训练
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

  # 设置回调
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # 开始训练
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

- GPU处理器环境运行

  ```python
  # 加载数据集
  dataset = create_dataset(cfg.data_path, 1)
  batch_num = dataset.get_dataset_size()

  # 定义模型
  net = GoogleNet(num_classes=cfg.num_classes)
  # 若pre_trained为True，继续训练
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

  # 设置回调
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./ckpt_" + str(get_rank()) + "/",
                               config=config_ck)
  loss_cb = LossMonitor()

  # 开始训练
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

### 迁移学习

待补充

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
