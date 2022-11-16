# 目录

<!-- TOC -->

- [目录](#目录)
- [SDNet描述](#SDNet描述)
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
            - [sen1-2上训练SDNet](#sen1-2上训练SDNet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SDNet描述

## 概述

SDNet从网络结构和网络优化两个方面提高了配准性能。首先设计了一个部分非共享特征学习网络，用于多模态图像特征学习。并提出了一种自蒸馏特征学习方法，利用更多的相似信息进行深度网络优化增强，如一系列不匹配补丁对之间的相似排序。利用丰富的相似度信息可以显著提高网络训练，提高匹配精度。构建了辅助任务重构学习来优化特征学习网络，以保留更多的判别信息。

## 论文

[Self-Distillation Feature Learning Network for Optical and SAR Image Registration](https://ieeexplore.ieee.org/abstract/document/9770793)

# 模型架构

SDNnet使用参数不共享的低层多模态特征学习模型进行特征提取，将得到的低层特征通过共享特征映射模型进行共享特征学习。同时，为了不丢失判别性，通过重构网络进行图像重构。

# 数据集

使用的数据集：[SEN1-2](<https://mediatum.ub.tum.de/1436631>)

- 数据集大小：该数据集共282384对SAR-Opt图像块，哨兵一、二号观测结果。空间上，来自全球，时间上则囊括了每一个气象季节。

- 数据处理：下载SEN1-2数据集，首先使用SIFT在每张图像中检测关键点，并围绕关键点裁取图像块，图像块的大小为64x64。共得到583180对可见光-SAR 图像块作为训练集，248274对可见光-SAR图像块作为测试集,将图像块文件存储成npz格式。

处理后得到如下所示目录：

  ```python
  ~/data/train.npz
  ~/data/test.npz
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
  # 添加数据集路径
  dataroot: "/home/user/data"

  # 推理前添加checkpoint路径参数
  checkpoint_path: "./ckpt/checkpoint_SDNet_20.ckpt"
  ```

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
  # example: bash scripts/run_train.sh ~/hccl_8p.json /home/data

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval.sh [DATA_PATH] [PATH_CHECKPOINT]
  # example: bash run_eval.sh /home/data ../ckpt/checkpoint_SDNet_20.ckpt
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上训练

      ```python
      # (1) 执行a或者b
      #       a. 在 sen1-2_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 sen1-2_config.yaml 文件中设置 "dataroot='/cache/data/'"
      #          在 sen1-2_config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "train_data_path='/cache/data/"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/SDNet"
      # (4) 在ModelArts网页上设置启动文件为 "train.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上测试

      ```python
      # (1) 执行a或者b
      #       a. 在 sen1-2_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 sen1-2_config.yaml 文件中设置 "dataroot='/cache/data/'"
      #          在 sen1-2_config.yaml 文件中设置 "checkpoint_path='/cache/ckpt/checkpoint_SDNet_20.ckpt'"
      #          在 sen1-2_config.yaml 文件中设置 其他参数
      #       b. 在ModelArts网页上设置 "modelArts_mode=True"
      #          在ModelArts网页上设置 "dataroot='/cache/data/"
      #          在ModelArts网页上设置 "checkpoint_path='/cache/ckpt/checkpoint_SDNet_20.ckpt'"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到 S3 桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/path/SDNet"
      # (4) 在ModelArts网页上设置启动文件为 "eval.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
├── SDNet
    ├── model_utils
    │   ├──config.py                // 参数配置
    │   ├──device_adapter.py        // device adapter
    │   ├──local_adapter.py         // local adapter
    ├── scripts
    │   ├──run_train.sh             // 分布式到Ascend的shell脚本
    │   ├──run_eval.sh              // Ascend评估的shell脚本
    ├── src
    │   ├──model
    │   │   ├──Decoder.py           // 共享特征映射模型
    │   │   ├──SDNet.py             // SDNet架构
    │   ├──dataset.py               // 数据处理
    │   ├──EvalMetrics.py           // 验证指标
    │   ├──Losses.py                // 损失函数
    │   ├──prepare_data.py          // 数据集预处理
    │   ├──Utils.py                 // 工具包
    ├── train.py                    // 训练脚本
    ├── eval.py                     // 评估脚本
    ├── export.py                   // 将checkpoint文件导出到air/mindir
    ├── README_CN.md                // 所有模型相关说明
    ├── sen1-2_config.yaml             // 参数配置
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

  ```python
  'modelArts_mode': False    # 当使用model_arts云上环境，将其设置为True
  'is_distributed': False    # 进行分布式计算的时候，将其设置为True
  'lr':0.004                 # 初始学习率
  'batch_size':500           # 训练批次大小
  'test_batch_size':600      # 测试批次大小
  'epochs':20                # 总计训练epoch数
  'wd':1e-4                  # 权重衰减值
  'optimizer': 'adam'        # 优化器
  'imageSize':64             # 输入到模型的图像块大小
  'dataroot':'./data'        # 训练和测试数据集的绝对全路径
  'device_target':'Ascend'   # 运行设备平台
  'device_id':0              # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'checkpoint_path':'./ckpt/checkpoint_SDNet_20.ckpt'  # 推理时加载checkpoint文件的绝对路径
  'loss': "triplet_margin"   # loss类型
  ```

更多配置细节请参考配置文件`sen1-2_config.yaml`。

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
  epoch: [  1/ 20], epoch time: 654675.890, steps:  1167, per step time: 560.990, avg loss: 1.039, lr:[0.003800]
  Accuracy(FPR95): 0.35339745  Acc:0.87125515
  epoch: [  2/ 20], epoch time: 589682.745, steps:  1167, per step time: 505.298, avg loss: 0.986, lr:[0.003600]
  Accuracy(FPR95): 0.54343869  Acc:0.82371896
  epoch: [  3/ 20], epoch time: 589538.510, steps:  1167, per step time: 505.174, avg loss: 0.975, lr:[0.003400]
  Accuracy(FPR95): 0.40809653  Acc:0.85347640
  ...
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train.sh ~/hccl_8p.json /home/data
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。采用以下方式达到损失值：

  ```bash
  train_parallel0/log:epoch: [  1/ 20], epoch time: 180954.569, steps:   146, per step time: 1239.415, avg loss: 1.247, lr:[0.009503]
  train_parallel0/log:Accuracy(FPR95): 0.92823313  Acc:0.58462022
  train_parallel0/log:epoch: [  2/ 20], epoch time: 73182.297, steps:   146, per step time: 501.249, avg loss: 1.154, lr:[0.009002]
  train_parallel0/log:Accuracy(FPR95): 0.64673143  Acc:0.77539332
  ...
  train_parallel1/log:epoch: [  1/ 20], epoch time: 177502.195, steps:   146, per step time: 1215.768, avg loss: 1.243, lr:[0.009503]
  train_parallel1/log:epoch: [  2/ 20], epoch time: 73222.867, steps:   146, per step time: 501.526, avg loss: 1.167, lr:[0.009002]
  train_parallel1/log:epoch: [  3/ 20], epoch time: 73132.261, steps:   146, per step time: 500.906, avg loss: 1.158, lr:[0.008501]
  ...
  ...
  ```

  训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果。

## 评估过程

### 评估

- 在Ascend环境运行时测试

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/SDNet/ckpt/checkpoint_SDNet_20.ckpt”。

  ```bash
  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval.sh /home/data username/SDNet/ckpt/checkpoint_SDNet_20.ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  ============= 910 Inference =============
  Accuracy(FPR95): 0.00914202  Acc:0.97957901
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

#### sen1-2上训练SDNet

|参数|Ascend 910|
|------------------------------|------------------------------|
|模型版本|SDNet|
|资源|Ascend 910；系统 ubuntu18.04|
|上传日期|2022-11-16|
|MindSpore版本|1.8.1|
|数据集|SEN1-2|
|训练参数|epoch=20, steps per epoch=1167, batch_size = 500|
|优化器|ADAM|
|损失函数|Lmatch + Lcon + Lrecon|
|输出|fpr95, Accuracy|
|损失|0.762|
|速度|506毫秒/步|
|总时长| 1p:4小时 8p:24分钟

# 随机情况说明

在train.py中设置随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
