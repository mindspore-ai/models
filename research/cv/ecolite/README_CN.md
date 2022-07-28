# 目录

<!-- TOC -->

- [目录](#目录)
- [ECO-lite描述](#ecolite描述)
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
            - [UCF101上训练ECO-lite](#UCF101上的ECO-lite)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ECO-lite描述

ECO-Lite模型，发表于ECCV 2018，ECO: Efficient Convolutional Network for Online Video Understanding。ECO网络仅采用RGB图像输入，无需光流信息，其基本思路为：对视频均匀采样得到N帧图像，然后使用共享的2D CNN网络获得对应的2D 特征图，并堆叠所得到的特征图，用3D CNN网络得到最后的分类结果。在获得相近性能的前提下，ECO网络比SOTA方法要快10-80倍，适合于实用化。论文中该模型主要是用于视频理解中的动作分类识别等。

[论文](https://arxiv.org/pdf/1804.09066.pdf): Mohammadreza Zolfaghari, Kamaljeet Singh, Thomas Brox."ECO: Efficient Convolutional Network forOnline Video Understanding"

# 模型架构

模型大致架构可参考上一小节的ECO-lite的描述，具体架构可参见论文。

# 数据集

本项目使用[UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)数据集实验。

- 数据集大小：共101 类，大小为(320, 240)

- 训练集：官方数据集[划分地址](https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip)

- 测试集：同上

注：数据集大小为6.46G，数据划分分为三种方式，可自行选择使用。（以上下载内容需要被放入该项目的data/ucf101_splits目录下）

- 数据格式：AVI

- 注：预处理时需要将UCF101中的视频保持结构不变逐帧分解为图像(参考代码：src/extract_frames.py，运行格式见本小节注解②)

- 最终处理过后的数据集目录结构如下： （注：其中每个class为每个视频的名字，另外处理好该目录下还需要执行src/gen_dataset_list.py来产生最终训练模型所使用的数据集目录，运行格式见本小节注解①）

  ```txt
   ├──ucf101dataset
       ├──class1
       │   ├──picture1.jpg
       │   ├──...
       │   ├──picturen.jpg
       ├──class2
       │   ├──picture1.jpg
       │   ├──...
       │   ├──picturen.jpg
       │   ...
       ├──classn
       │   ├──picture1.jpg
       │   ├──...
       │   ├──picturen.jpg
       ...
  ```

- 注解①
  运行格式：python ./src/gen_dataset_lists.py [数据集名称] [数据集绝对路径]
  运行实例：python ./src/gen_dataset_lists.py ucf101 /disk3/hlj/ucf101
- 注解②
  运行格式：python ./src/extract_frames.py --video_src_path [视频数据集路径数据集] --image_save_path [视频帧保存路径]
  运行实例：python ./src/extract_frames.py --video_src_path /path/to/UCF-101 --image_save_path /path/to/ucf101_images

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  注：在运行之前，需下载好预训练权重文件放到该工程目录下，[可通过网盘获取](https://pan.baidu.com/s/1jdnZ33yKkMuKIThssR1QdA )，提取码为6i9h。

  ```shell
  # Ascend分布式训练
  bash ./scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM]
  # example:  bash ./scripts/run_distribute_train.sh ./scripts/rank_table_8pcs.json 8
  # Ascend单卡训练
  bash ./scripts/run_standalone_train.sh [CHECKPOINT_PATH] [DEVICE_ID]
  # example: bash ./scripts/run_standalone_train.sh ms_model_kinetics_checkpoint0720.ckpt 0
  # Ascend处理器环境运行eval
  bash ./scripts/run_eval.sh [CHECKPOINT_PATH] [DEVICE_ID]
  # example: bash ./scripts/run_eval.sh ./ms_model_kinetics_checkpoint0720.ckpt 0
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。
  请遵循以下链接中的说明：
 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>
默认使用ucf101数据集。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))
    - 在 ModelArts 上使用八卡、单卡训练 ucf101数据集，区别仅在于是否进行分布式并行训练run_distribute的设置。

    ```bash
    # (1) 在网页上设置 "config_path='/path_to_code/imagenet_config.yaml'"
    # (2)     在网页上设置 "lr=0.001"
    #         在网页上设置 "epoch=60"
    #         在网页上设置 "run_distribute=True"
    #         在网页上设置 "enable_modelarts=True"
    #         在网页上设置 其他参数,具体参见src/opts.py文件
    # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
    # (4) 在网页上设置你的代码路径为 "/path/ECO-lite"
    # (5) 在网页上设置启动文件为 "train.py"
    # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (7) 创建训练作业
    ```

## 脚本及样例代码

```bash
├── ECO-lite
    ├── README_CN.md                // ECO-lite相关说明
    ├── model_utils                 // 云上处理工具
    ├── ascend310_infer             // 实现310推理源代码
    ├── scripts
    │   ├──run_distribute_train.sh  // 启动Ascend分布式训练的shell脚本
    │   ├──run_standalone_train.sh  // 启动Ascend单机训练的shell脚本
    │   ├──run_eval.sh              // Ascend评估的shell脚本
    │   ├──run_infer_310.sh         // Ascend推理的shell脚本
    ├── src
    │   ├──pyActionRecog        // 产生训练模型所要用的数据集文件的基本操作
    │   ├──ops                     // 同上
    │   ├──transforms.py           //数据预处理
    │   ├──gen_dataset_list.py     // 产生训练模型所要用的数据集文件
    │   ├──dataset.py              // 创建数据集
    │   ├──econet.py               // econet架构
    │   ├──utils.py                // 云上数据传输工具包
    │   ├──extract_frames.py       // 将ucf101数据集提取为关键帧
    ├── train.py                   // 训练脚本
    ├── eval.py                    // 评估脚本
    ├── get_310_eval_dataset.py    // 310推理数据获取
    ├── postprogress.py            // 310推理后处理脚本
    ├── export.py                  // 将checkpoint文件导出到mindir
    ├── default_config.yaml        // 参数配置
```

## 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

```python
"train_url":                                                 # 仅云上需要配置
"data_url":                                                  # 仅云上需要配置
"dataset": ucf101                                            # 数据集
"modality": RGB                                              # 默认利用视频的RGB特征
"train_list":./data/ucf101_rgb_train_split_1.txt             # 训练集路径
"val_list":./data/ucf101_rgb_val_split_1.txt                 # 验证集路径
"arch": ECO                                                  # backbone网络
"num_segments": 4                                            # 提取RGB帧的数量
"dropout": 0.7                                               # 丢弃比例
"epochs": 60                                                     # 训练总轮数
"batch-size": 16                                                 # 每次训练传入模型的批大小
"momentum": "0.9"                                            # 优化器所设置的momentum值大小
"lr": 0.001                                                  # 学习率
"no_partialbn": True                                         # 是否冻结bn层
"run_distribute":True                                        # 是否进行分布式并行训练
"run_online": True                                           # 是否进行云上训练
"resume": 'ms_model_kinetics_checkpoint0720.ckpt'            # 预训练参数权重路径
"rgb_prefix":False                                           # RGB视频帧文件名前缀
"device_target":Ascend                                       # 硬件平台
"device_id":0                                                # 设备ID
```

```bash
----------------run_standalone_train.sh--------------
$1:[CHECKPOINT_PATH]     # 预训练模型参数存放路径
$2:[DEVICE_ID]                    # 指定运行设备卡号
----------------run_distribute_train.sh-----------------
$1:[RANK_TABLE_FILE]      # 并行训练配置文件路径
$2:[DEVICE_NUM]               # 指定进行分布式并行训练的卡数
----------------run_eval.sh---------------------------
$1:[CHECKPOINT_PATH]     # 模型参数存放路径
$2:[DEVICE_ID]                    # 指定运行设备卡号
```

更多配置细节请参考脚本`default_config.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  # Ascend单卡训练
  bash ./scripts/run_standalone_train.sh [CHECKPOINT_PATH] [DEVICE_ID]
  ```

  上述脚本命令将在后台运行，您可以通过log.txt文件查看结果。
  训练结束后，您可在该工程目录下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " log.txt
  epoch: 1 step: 1, loss is 5.0373483
  epoch: 1 step: 2, loss is 4.7960615
  epoch: 1 step: 3, loss is 5.7136536
  epoch: 1 step: 4, loss is 4.738636
  epoch: 1 step: 5, loss is 4.5478897
  epoch: 1 step: 6, loss is 4.794219
  epoch: 1 step: 7, loss is 4.573368
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过log.txt文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "loss is " log.txt
  epoch: 1 step: 50, loss is 2.9836612
  epoch: 1 step: 50, loss is 2.0120254
  epoch: 1 step: 50, loss is 2.7175844
  epoch: 1 step: 50, loss is 2.9692264
  epoch: 1 step: 50, loss is 2.5570111
  epoch: 1 step: 50, loss is 2.3821626
  epoch: 1 step: 50, loss is 2.8024788
  epoch: 1 step: 50, loss is 2.329533
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估UCF101数据集

  在运行以下命令之前，请检查用于评估的检查点路径。

  ```bash
  # Ascend处理器环境运行eval
  bash ./scripts/run_eval.sh [CHECKPOINT_PATH] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过log.txt文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "Acc:" log.txt
  {'Loss': 0.4464788576615165, 'Top1-Acc': 0.8802966101694916, 'Top5-Acc': 0.9804025423728814}
  ```

## 导出过程

### 导出

可以使用 `export.py` 脚本进行模型导出

```shell
python export.py --dataset ucf101 --modality RGB --arch ECO --num_segments 4 --batch-size [BATCH_SIZE] --no_partialbn True --checkpoint_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]  --device_id [DEVICE_ID]
```

## 推理过程

### 推理

在执行推理之前我们需要先导出模型。

- 在昇腾310上使用UCF101数据集进行推理

  其中数据准备过程具体参考数据集章节，其大致流程为，准备将视频数据集逐个切分成帧，其次再准备好ucf101_splits目录；最后再对数据进行预处理，并保存成二进制文件(可参考该工程下的get_310_evaldataset.py)。注：用户不需要手动执行，该预处理过程已在启动脚本中被自动执行。

  ```shell
  # Ascend310 inference
  bash ./scripts/run_310_infer.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID] [BATCH_SIZE]
  # example：bash scripts/run_310_infer.sh /home/stu/hlj/ecolite.mindir /home/stu/hlj/eval_dataset3 0 16
  # example：bash scripts/run_310_infer.sh ./ecolite.mindir ./eval_dataset3 0 16
  ```

  注：以上的EVAL_DATA_DIR和MINDIR_PATH支持相对路径和绝对路径，但请用户在使用相对路径时，将路径设置在该工程目录下，如上示例所示。

  推理的结果保存在ascend310_infer目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```bash
  Total data: 3776, top1 accuracy: 0.8805614406779662, top5 accuracy: 0.9804025423728814
  ```

# 模型描述

## 性能

### 训练性能

#### UCF101上的ECO-lite

| 参数          | Ascend 910                                |
| ------------- | ----------------------------------------- |
| 模型版本      | ECOLite                                   |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G |
| 上传日期      | 2021-12-08                                |
| MindSpore版本 | 1.5.0                                     |
| 数据集        | UCF101                                    |
| 训练参数      | 详见src/opts.py                           |
| 优化器        | SGD                                       |
| 损失函数      | Softmax交叉熵                             |
| 输出          | 概率                                      |
| 损失          | 0.67934（单卡）                           |
| 速度          | 220ms/step（单卡）                        |
| 总时长        | 6h13m34s（单卡）                          |

# 随机情况说明

train.py可以设置随机种子，以避免训练过程中的随机性。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。