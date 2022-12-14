# 目录

<!-- TOC -->

- [目录](#目录)
- [Cnet描述](#Cnet描述)
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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [liberty上训练Cnet](#liberty上训练Cnet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Cnet描述

## 概述

Cnet将卷积注意力模块(Convolutional Block Attention Module)引入到描述子学习过程中，使得网络可以重点关注需要的局部特征，从而抑制不需要的局部特征，并提出间隔损失函数(Margin Loss)，进一步增强描述子的泛化能力。

## 论文

[Deep Feature Correlation Learning for Multi-Modal Remote Sensing Image Registration](https://ieeexplore.ieee.org/document/9850980)

# 模型架构

Cnet中一共包含7个卷积层，3个卷积注意力模块，每个卷积层后面都跟着批归一化层(Batch Normalization)以及ReLU激活函数，最后一个卷积层不加ReLU激活函数。网络输入一个大小为32×32的图像patch图像块，输出一个经过L2范数归一化的128维特征描述子。

# 数据集

使用的数据集：[UBC PhotoTour](<http://phototour.cs.washington.edu/patches/default.htm>)

- 数据集大小：该数据集一共有三个子集分别是：Liberty、 Notredame、 Yosemite，每个子集包含160K、468K、634K独立的图像块和160K、147K、230K唯一的3D Point。每个图像块是从一个关键点周围裁剪得到的64×64图像区域。
    - 训练与测试：其中一个子集为训练，另外两个子集为测试

- liberty数据下载地址：[liberty.zip](http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip)
- notredame数据下载地址：[notredame.zip](http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip)
- yosemite数据下载地址：[yosemite.zip](http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip)

下载数据集后解压得到如下所示目录：

  ```python
  ~/data/liberty
  ~/data/notredame
  ~/data/yosemite
  ```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 添加数据集路径,以训练liberty为例
  dataroot: "/home/user/data"
  training_set: "liberty"

  # 推理前添加checkpoint路径参数
  checkpoint_path: "./ckpt/checkpoint_Cnet_10.ckpt"
  ```

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_train.sh [RANK_TABLE_FILE] [DATA_PATH] [DATASET_NAME]
  # example: bash scripts/run_train.sh ~/hccl_8p.json /home/data liberty

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh [DATA_PATH] [PATH_CHECKPOINT] [DATASET_NAME]
  # example: bash run_eval.sh /home/data ./ckpt/checkpoint_Cnet_10.ckpt liberty
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上训练 liberty 数据集

      ```python
      # (1) 执行a或者b
      #       a. 在 ubc_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 ubc_config.yaml 文件中设置 "training_set='liberty'"
      #          在 ubc_config.yaml 文件中设置 "dataroot='/cache/data/brown/'"
      #          在 ubc_config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "training_set=liberty"
      #          在ModelArts网页上设置 "train_data_path='/cache/data/brown/"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/Cnet"
      # (4) 在ModelArts网页上设置启动文件为 "train.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上推理 liberty 数据集

      ```python
      # (1) 执行a或者b
      #       a. 在 ubc_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 ubc_config.yaml 文件中设置 "eval_set='liberty'"
      #          在 ubc_config.yaml 文件中设置 "dataroot='/cache/data/brown/'"
      #          在 ubc_config.yaml 文件中设置 "checkpoint_path='/cache/ckpt/checkpoint_Cnet_1.ckpt'"
      #          在 ubc_config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "eval_set=liberty"
      #          在ModelArts网页上设置 "dataroot='/cache/data/brown/"
      #          在ModelArts网页上设置 "checkpoint_path='/cache/ckpt/checkpoint_Cnet_1.ckpt'"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/Cnet"
      # (4) 在ModelArts网页上设置启动文件为 "eval.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

      其他数据集同理，只需修改training_set和eval_set参数。

# 脚本说明

## 脚本及样例代码

```bash
├── Cnet
    ├── model_utils
    │   ├──config.py                // 参数配置
    │   ├──device_adapter.py        // device adapter
    │   ├──local_adapter.py         // local adapter
    ├── scripts
    │   ├──run_train.sh             // 分布式到Ascend的shell脚本
    │   ├──run_eval.sh              // Ascend评估的shell脚本
    ├── src
    │   ├──model
    │   │   ├──CBAM.py              // 卷积注意力模块
    │   │   ├──Cnet.py              // Cnet架构
    │   ├──dataset.py               // 数据处理
    │   ├──EvalMetrics.py           // 验证指标
    │   ├──Losses.py                // 损失函数
    │   ├──prepare_data.py          // 数据集预处理
    │   ├──Utils.py                 // 工具包
    ├── train.py                    // 训练脚本
    ├── eval.py                     // 评估脚本
    ├── export.py                   // 将checkpoint文件导出到air/mindir
    ├── README_CN.md                // 所有模型相关说明
    ├── ubc_config.yaml             // 参数配置
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

  ```python
  'modelArts_mode': False    # 当使用model_arts云上环境，将其设置为True
  'is_distributed': False    # 进行分布式计算的时候，将其设置为True
  'training_set':"liberty"   # 选择训练集
  'eval_set':"notredame"     # 选择测试集
  'lr':10                    # 初始学习率
  'batch_size':1024          # 训练批次大小
  'test_batch_size':2048     # 测试批次大小
  'epochs':10                # 总计训练epoch数
  'wd':1e-4                  # 权重衰减值
  'optimizer': 'sgd'         # 优化器
  'imageSize':32             # 输入到模型的图像块大小
  'dataroot':'./data'        # 训练和评估数据集的绝对全路径
  'device_target':'Ascend'   # 运行设备
  'device_id':0              # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'checkpoint_path':'./ckpt/checkpoint_Cnet_1.ckpt'  # 推理时加载checkpoint文件的绝对路径
  'loss': "triplet_margin"   # loss类型
  ```

更多配置细节请参考配置文件`ubc_config.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # train.log
  epoch: [  1/ 10], epoch time: 1575744.996, steps:  4883, per step time: 322.700, avg loss: 0.444, lr:[9.000166]
  epoch:1, notredame, Accuracy(FPR95): 0.01374000
  epoch:1, liberty, Accuracy(FPR95): 0.04702000
  epoch: [  2/ 10], epoch time: 1456287.147, steps:  4883, per step time: 298.236, avg loss: 0.330, lr:[8.000128]
  epoch:2, notredame, Accuracy(FPR95): 0.01244000
  epoch:2, liberty, Accuracy(FPR95): 0.03812000
  ...
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train.sh ~/hccl_8p.json /home/data liberty
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  train_parallel0/log:epoch: [  1/ 10], epoch time: 484908.890, steps:   611, per step time: 497.814, avg loss: 0.820, lr:[9.000576]
  train_parallel0/log:epoch:1, yosemite, Accuracy(FPR95): 0.12810000
  train_parallel0/log:epoch:1, notredame, Accuracy(FPR95): 0.09368000
  ...
  train_parallel1/log:epoch: [  1/ 10], epoch time: 479172.144, steps:   611, per step time: 497.814, avg loss: 0.822, lr:[9.000576]
  ...
  ...
  ```

  训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估notredame数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/Cnet/ckpt/checkpoint_Cnet_10.ckpt”。
  并修改ubc_config.yaml中‘eval_set’为‘notredame’。

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh /home/data username/Cnet/ckpt/checkpoint_Cnet_10.ckpt notredame
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  ============= 910 Inference =============
  eval dataset:notredame, Accuracy(FPR95): 0.03196
  =========================================
  ```

## 导出过程

### 导出MindIR

```shell
python export.py --ckpt_file=[CKPT_PATH] --file_format=[MINDIR, AIR]
```

# 模型描述

## 性能

### 训练性能

#### liberty上训练Cnet

|参数|Ascend 910|
|------------------------------|------------------------------|
|模型版本|Cnet|
|资源|Ascend 910；系统 ubuntu18.04|
|上传日期|2022-11-9|
|MindSpore版本|1.8.1|
|数据集|liberty|
|训练参数|epoch=10, steps per epoch=4883, batch_size = 1024|
|优化器|SGD|
|损失函数|Adaptive_Augular_Margin_Loss|
|输出|fpr95|
|损失|0.291|
|速度|298毫秒/步|
|总时长| 1p:4小时 8p:1小时

# 随机情况说明

在train.py中设置随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
