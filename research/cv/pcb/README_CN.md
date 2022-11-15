# 目录

<!-- TOC -->

- [目录](#目录)
- [PCB描述](#pcb描述)
    - [概述](#概述)
    - [论文](#论文)
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
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
        - [结果](#结果-1)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [Market-1501上的PCB](#market-1501上的pcb)
            - [DukeMTMC-reID上的PCB](#dukemtmc-reid上的pcb)
            - [CUHK03上的PCB](#cuhk03上的pcb)
            - [Market-1501上的PCB-RPP](#market-1501上的pcb-rpp)
            - [DukeMTMC-reID上的PCB-RPP](#dukemtmc-reid上的pcb-rpp)
            - [CUHK03上的PCB-RPP](#cuhk03上的pcb-rpp)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PCB描述

## 概述

PCB（Part-level Convolutional Baseline）模型是行人重识别任务的经典模型，它通过对输入图像进行均匀分划，构建出包含图像各部分特征的卷积描述子，用于后续行人检索。此外，原论文提出了RPP（Refined Part Pooling）方法对离群点所属区域进行重新分配，进一步提高了区域内特征的一致性，有效提升了PCB模型的精度。

如下为MindSpore使用Market-1501/DukeMTMC-reID/CUHK03数据集对PCB/PCB+RPP进行训练的示例。

## 论文

1. [论文](https://arxiv.org/pdf/1711.09349.pdf)：Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang."Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)"

# 模型架构

PCB的总体网络架构如下：
[链接](https://arxiv.org/pdf/1711.09349.pdf)

# 数据集

使用的数据集：[Market-1501](<http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html>)
[下载地址](https://pan.baidu.com/s/1qWEcLFQ?_at_=1640837580475)

- 数据集组成：
    - 训练集：包含751个行人的12936个RGB图像
    - 测试集：

        -query set: 包含750个行人的3368个RGB图像

        -gallery set：包含751个行人的15913个RGB图像

- 数据格式：PNG图像
    - 注：数据在src/datasets/market.py中处理。
- 下载数据集。目录结构如下：

```text
├─ Market-1501
 │
 ├─bounding_box_test
 │
 └─bounding_box_train
 │
 └─gt_bbx
 │
 └─gt_query
 │
 └─query
```

使用的数据集：[DukeMTMC-reID](http://vision.cs.duke.edu/DukeMTMC/)

- 数据集组成：
    - 训练集：包含702个行人的16522个RGB图像
    - 测试集：

        -query set: 包含702个行人的2228个RGB图像

        -gallery set：包含1110个行人的17661个RGB图像

- 数据格式：PNG图像
    - 注：数据在src/datasets/duke.py中处理。
- 下载数据集。目录结构如下：

```text
├─ DukeMTMC-reID
 │
 ├─bounding_box_test
 │
 └─bounding_box_train
 │
 └─query
```

使用的数据集：[CUHK03](<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>)
[下载地址](https://pan.baidu.com/s/1o8txURK)

原论文在CUHK03上使用了新的 [测试协议](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
新协议将原始的CUHK03数据集按照Market-1501数据集格式进行了划分
[新协议下载地址](https://drive.google.com/file/d/0B7TOZKXmIjU3OUhfd3BPaVRHZVE/view?resourcekey=0-hU4gyE6hFsBgizIh9DFqtA)

- 新协议下CUHK03数据集组成：
    - 训练集：包含767个行人的7365个RGB图像
    - 测试集:

        -query set: 包含700个行人的1400个RGB图像

        -gallery set：包含700个行人的5332个RGB图像

- 数据格式：PNG图像
    - 注：数据在src/datasets/cuhk03.py中处理。

- 下载数据集。

需要先分别下载原始数据集与新协议划分文件，然后将原始数据集中的cuhk03.mat文件与新协议中的cuhk03_new_protocol_config_detected.mat文件放在同一文件夹（CUHK03）下：

```text
├─ CUHK03
 │
 ├─ cuhk03.mat
 │
 └─ cuhk03_new_protocol_config_detected.mat
```

- 预训练的Resnet50权重 [下载地址](https://gitee.com/starseekerX/PCB_pretrained_checkpoint/blob/master/pretrained_resnet50.ckpt)

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
    - 准备Ascend910处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

> - <font size=2>训练、评估和推理时，请按以下方式使用数据集：如果使用Market-1501数据集, DATASET_PATH={Market-1501目录所在路径};如果使用Duke-MTMC-reID数据集, DATASET_PATH={Duke-MTMC-reID目录所在路径};如果使用CUHK03数据集, DATASET_PATH={CUHK03目录所在路径}</font>

- Ascend处理器环境运行

```bash
# 单机训练
# 用法：
bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）

# 例如：
bash run_standalone_train.sh PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 分布式训练(以PCB模型在Market上训练为例)
# 用法：
bash run_distribute_train.sh [RANK_TABLE_FILE] [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](可选)

# 例如：
bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 运行评估示例(以PCB模型在Market上评估为例)
# 用法：
bash run_eval.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]

# 例如：
bash run_eval.sh PCB market ../../Datasets/Market-1501 ../config/eval_PCB_market.yaml ./output/checkpoint/PCB/market/train/PCB-60_202.ckpt True
```

# 脚本说明

## 脚本及样例代码

```text
.
└──PCB
  ├── README.md
  ├── config                              # 参数配置
    ├── base_config.yaml
    ├── train_PCB_market.yaml
    ├── train_PCB_duke.yaml
    ├── train_PCB_cuhk03                  # PCB在CUHK03数据集上训练包含训练与微调两个阶段，因此需要两个配置文件
      ├── train_PCB.yaml
      ├── finetune_PCB.yaml
    ├── train_RPP_market                  # PCB+RPP在Market-1501数据集上训练需要先训练PCB，然后将PCB权重载入PCB+RPP续训，因此需要两个配置文件
      ├── train_PCB.yaml
      ├── train_RPP.yaml
    ├── train_RPP_duke
      ├── train_PCB.yaml
      ├── train_RPP.yaml
    ├── train_RPP_cuhk03
      ├── train_PCB.yaml
      ├── train_RPP.yaml
      ├── finetune_RPP.yaml
    ├── eval_PCB_market.yaml              # 评估PCB在Market-1501数据集上的精度
    ├── eval_PCB_duke.yaml
    ├── eval_PCB_cuhk03.yaml
    ├── eval_RPP_market.yaml              # 评估PCB+RPP在Market-1501数据集上的精度
    ├── eval_RPP_duke.yaml
    ├── eval_RPP_cuhk03.yaml
    ├── infer_310_config.yaml             # 用于模型导出成mindir、310推理的配置文件
  ├── scripts
    ├── run_standalone_train.sh           # 启动单卡训练脚本
    ├── run_distribute_eval.sh            # 启动八卡训练脚本
    ├── run_eval.sh                       # 启动评估脚本
    ├── run_infer_310.sh                  # 启动310推理
  ├── src
    ├── dataset.py                         # 数据预处理
    ├── eval_callback.py                   # 训练时推理回调函数
    ├── eval_utils.py                      # 评估mAP、CMC所需的util函数
    ├── meter.py
    ├── logging.py                         # 日志管理程序
    ├── lr_generator.py                    # 生成每个步骤的学习率
    ├── pcb.py                             # PCB模型结构、损失函数
    ├── rpp.py                             # PCB+RPP模型结构
    ├── resnet.py                          # resnet50模型结构
    ├── datasets                           # 包含处理Market-1501、DukeMTMC-reID、CUHK03数据集的程序
       ├── market.py                       # 处理Market-1501的程序
       ├── duke.py                         # 处理DukeMTMC-reID的程序
       ├── cuhk03.py                       # 处理CUHK03的程序
    ├── model_utils
       ├── config.py                       # 参数配置
       ├── device_adapter.py               # 设备配置
       ├── local_adapter.py                # 本地设备配置
       └── moxing_adapter.py               # modelarts设备配置
  ├── eval.py                              # 评估网络
  └── train.py                             # 训练网络
  └── export.py                            # 模型导出
  └── preprocess.py                        # 310推理预处理
  └── postprocess.py                       # 310推理后处理
```

## 脚本参数

在配置文件中可以配置训练、评估、模型导出、推理参数，下面以PCB在Market-1501数据集上的训练、评估，模型导出与推理为例。

- 配置PCB在Market-1501数据集训练。

```text
enable_modelarts: False                    # 是否开启modelarts云上训练作业训练
data_url: ""
train_url: ""
checkpoint_url: ""
run_distribute: False                      # 是否开启分布式训练
enable_profiling: False
dataset_path: "/cache/dataset/"            # 数据集路径
output_path: "/cache/output/"              # 结果输出路径
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
log_save_path: "./log/PCB/market/train"    # 日志保存路径
checkpoint_save_path: "./checkpoint/PCB/market/train"   # 断点保存路径
checkpoint_file_path: "/cache/load_checkpoint/pretrained_resnet50.ckpt"   # 断点加载路径

mindrecord_dir: "./MindRecord"             # MindRecord文件保存路径
dataset_name: "market"                     # 数据集简名
batch_size: 64                             # 一个数据批次大小
num_parallel_workers: 4
device_num: 1                              # 设备数量

model_name: "PCB"                          # 模型名
learning_rate: 0.1                         # 学习率
lr_mult: 0.1                               # 主干网络学习率倍数
decay_rate: 0.1                            # 学习率衰减率
momentum: 0.9                              # SGD优化器的动量
weight_decay: 5e-4                         # SGD优化器的权重衰减率
nesterov: True

mode_name: "GRAPH"                         # 程序运行模式
sink_mode: True                            # 是否开启数据下沉
seed: 37                                   # 随机种子
epoch_size: 60                             # 训练回合数
decay_epoch_size: 40                       # 学习率衰减的回合数
warmup_epoch_size: 1                       # 训练开始的warmup回合数

save_checkpoint: True                      # 是否保存断点
save_checkpoint_epochs: 60                 # 保存断点的回合数
keep_checkpoint_max: 15                    # 最多保存的断点数

run_eval: False                            # 是否在训练时进行评估
eval_interval: 15                          # 评估的回合间隔
eval_start_epoch: 60                       # 评估的开始回合
use_G_feature: True                        # 评估时是否使用G feature，若不使用则代表使用H feature
```

- 配置PCB在Market-1501数据集评估。

```text
enable_modelarts: False                    # 是否开启modelarts云上训练作业训练
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"            # 数据集路径
output_path: "/cache/output/"              # 结果输出路径
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
log_save_path: "./log/PCB/market/eval"     # 日志保存路径
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"   # 断点加载路径

mindrecord_dir: "./MindRecord"             # MindRecord文件保存路径
dataset_name: "market"                     # 数据集简名
batch_size: 64                             # 一个数据批次大小
num_parallel_workers: 4

model_name: "PCB"                          # 模型名
use_G_feature: True                        # 评估时是否使用G feature，若不使用则代表使用H feature
```

- 配置PCB模型导出与推理。

```text
enable_modelarts: False                    # 是否开启modelarts云上训练作业训练
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"            # 数据集路径
output_path: "/cache/output/"              # 结果输出路径
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"   # 断点加载路径
batch_size: 1                              # 目前仅支持batch size为1的推理
model_name: "PCB"                          # 模型名
use_G_feature: True                        # 模型导出时是否使用G feature，若不使用则代表使用H feature，G/H feature选择的差异会影响导出模型的结构

device_id: 0
image_height: 384                          # 导出模型输入的高
image_width: 128                           # 导出模型输入的宽
file_name: "export_PCB_market_G"           # 导出的模型名
file_format: "MINDIR"                      # 导出的模型格式

preprocess_result_path: "./preprocess_Result"  #310推理预处理结果路径

query_prediction_path: "./query_result_files"  #query集合10推理结果输出路径
gallery_prediction_path: "./gallery_result_files"  #gallery集合310推理结果输出路径
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# 单机训练
# 用法：
bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]（可选）
# 其中MODEL_NAME可从['PCB', 'RPP']中选择，DATASET_NAME可从['market', 'duke', 'cuhk03']中选择。

# 请注意数据集与配置文件需与训练脚本对应，请见下面的示例

# 示例：

# 1、PCB在Market-1501上训练

bash run_standalone_train.sh PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 2、PCB在DukeMTMC-reID上训练

bash run_standalone_train.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/train_PCB_duke.yaml ../../pretrained_resnet50.ckpt

# 3、PCB在CUHK03上训练（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_standalone_train.sh PCB cuhk03 ../../Datasets/CUHK03 ../config/train_PCB_cuhk03 ../../pretrained_resnet50.ckpt

# 4、PCB+RPP在Market-1501上训练（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_standalone_train.sh RPP market ../../Datasets/Market-1501 ../config/train_RPP_market ../../pretrained_resnet50.ckpt

# 5、PCB+RPP在DukeMTMC-reID上训练（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_standalone_train.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/train_RPP_duke ../../pretrained_resnet50.ckpt

# 6、PCB+RPP在CUHK03上训练（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_standalone_train.sh RPP cuhk03 ../../Datasets/CUHK03 ../config/train_RPP_cuhk03 ../../pretrained_resnet50.ckpt


# 分布式训练
# 用法：
bash run_distribute_train.sh [RANK_TABLE_FILE] [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](可选)

# 其中MODEL_NAME可从['PCB', 'RPP']中选择，DATASET_NAME可从['market', 'duke', 'cuhk03']中选择。

# 请注意数据集与配置文件需与训练脚本对应，请见下面的示例

# 示例

# 1、PCB在Market-1501上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 2、PCB在DukeMTMC-reID上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB duke ../../Datasets/DukeMTMC-reID ../config/train_PCB_duke.yaml ../../pretrained_resnet50.ckpt

# 3、PCB在CUHK03上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB cuhk03 ../../Datasets/CUHK03 ../config/train_PCB_cuhk03 ../../pretrained_resnet50.ckpt

# 4、RPP在Market-1501上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP market ../../Datasets/Market-1501 ../config/train_RPP_market ../../pretrained_resnet50.ckpt

# 5、RPP在DukeMTMC-reID上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP duke ../../Datasets/DukeMTMC-reID ../config/train_RPP_duke ../../pretrained_resnet50.ckpt

# 6、RPP在CUHK03上分布式训练（Ascend 8卡）

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP cuhk03 ../../Datasets/CUHK03 ../config/train_RPP_cuhk03 ../../pretrained_resnet50.ckpt
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果保存在脚本目录的output文件夹下，其中日志文件保存在./output/log/{MODEL_NAME}/{DATASET_NAME}/train下，断点文件保存在./output/checkpoint/{MODEL_NAME}/{DATASET_NAME}/train下，您可以在其中找到所需的信息。

#### Modelarts训练作业

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```text
# 在modelarts上进行单卡训练的示例：
# (1) 在网页上设置 "config_path='/path_to_code/config/train_PCB_market.yaml'"
# (2) 选择a或者b其中一种方式。
#       a. 在yaml文件中设置 "enable_modelarts=True" 。
#          设置 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#          如需自定义参数，可在yaml文件中设置其它网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          如需自定义参数，可在modelarts的界面上设置其它网络所需的参数。
#
# (3) 在modelarts的界面上设置代码的路径 "/path/PCB"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上进行分布式训练的示例：
# (1) 在网页上设置 "config_path='/path_to_code/config/train_PCB_market.yaml'"
# (2) 选择a或者b其中一种方式。
#       a. 在yaml文件中设置 "enable_modelarts=True" 。
#          设置 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#          设置 "run_distribute=True" 在 yaml 文件.
#          设置 "device_num = {可用的卡数}" 在 yaml 文件.
#          如需自定义参数，可在yaml文件中设置其它网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          增加 "run_distribute=True" 参数在modearts的界面上。
#          增加 "device_num = {可用的卡数}" 参数在modearts的界面上。
#          如需自定义参数，可在modelarts的界面上设置其它网络所需的参数。
# (3) 在modelarts的界面上设置代码的路径 "/path/PCB"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" , 断点路径 “checkpoint file path” ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。
```

### 结果

- 使用Market-1501数据集训练PCB

```log
# 单卡训练结果
epoch: 1 step: 202, loss is 28.947758
epoch time: 88804.387 ms, per step time: 439.626 ms
epoch: 2 step: 202, loss is 18.160383
epoch time: 35282.132 ms, per step time: 174.664 ms
epoch: 3 step: 202, loss is 14.728483
epoch time: 35331.460 ms, per step time: 174.908 ms
...
```

- 使用DukeMTMC-reID数据集训练PCB

```log
# 单卡训练结果
epoch: 1 step: 258, loss is 23.912783
epoch time: 100480.371 ms, per step time: 389.459 ms
epoch: 2 step: 258, loss is 13.815624
epoch time: 33952.824 ms, per step time: 131.600 ms
epoch: 3 step: 258, loss is 9.111069
epoch time: 33952.491 ms, per step time: 131.599 ms
...
```

- 使用CUHK03数据集训练PCB

```log
# 单卡训练结果
epoch: 1 step: 115, loss is 34.977722
epoch time: 87867.500 ms, per step time: 764.065 ms
epoch: 2 step: 115, loss is 24.710325
epoch time: 15645.867 ms, per step time: 136.051 ms
epoch: 3 step: 115, loss is 16.847214
epoch time: 15694.620 ms, per step time: 136.475 ms
...
```

- 使用Market-1501数据集训练RPP

```log
# 单卡训练结果
epoch: 1 step: 202, loss is 28.807777
epoch time: 90390.587 ms, per step time: 447.478 ms
epoch: 2 step: 202, loss is 18.29936
epoch time: 35274.752 ms, per step time: 174.627 ms
epoch: 3 step: 202, loss is 14.982595
epoch time: 35277.650 ms, per step time: 174.642 ms
...
```

- 使用DukeMTMC-reID数据集训练RPP

```log
# 单卡训练结果
epoch: 1 step: 258, loss is 23.096334
epoch time: 96244.296 ms, per step time: 373.040 ms
epoch: 2 step: 258, loss is 13.114418
epoch time: 33972.328 ms, per step time: 131.676 ms
epoch: 3 step: 258, loss is 8.97956
epoch time: 33965.507 ms, per step time: 131.649 ms
...
```

- 使用CUHK03数据集训练RPP

```log
# 单卡训练结果
epoch: 1 step: 115, loss is 37.5888
epoch time: 68445.567 ms, per step time: 595.179 ms
epoch: 2 step: 115, loss is 26.582499
epoch time: 15640.461 ms, per step time: 136.004 ms
epoch: 3 step: 115, loss is 17.900295
epoch time: 15637.023 ms, per step time: 135.974 ms
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 用法：
bash run_eval.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]

# 其中MODEL_NAME可从['PCB', 'RPP']中选择，DATASET_NAME可从['market', 'duke', 'cuhk03']中选择, USE_G_FEATURE表示评估时是否使用论文中的G feature，如果不使用，则表示使用H feature。

# 请注意数据集与配置文件需与训练脚本对应，请见下面的示例

# 示例：

# 1、PCB在Market-1501上使用G feature评估

bash run_eval.sh PCB market ../../Datasets/Market-1501 ../config/eval_PCB_market.yaml ./output/checkpoint/PCB/market/train/PCB-60_202.ckpt True

# 2、PCB在DukeMTMC-reID上使用G feature评估

bash run_eval.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/eval_PCB_duke.yaml ./output/checkpoint/PCB/duke/train/PCB-60_258.ckpt True

# 3、PCB在CUHK03上使用G feature评估（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_eval.sh PCB cuhk03 ../../Datasets/CUHK03 ../config/eval_PCB_cuhk03.yaml ./output/checkpoint/PCB/cuhk03/train/PCB_1-45_115.ckpt True

# 4、PCB+RPP在Market-1501上使用G feature评估（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_eval.sh RPP market ../../Datasets/Market-1501 ../config/eval_RPP_market.yaml ./output/checkpoint/RPP/market/train/RPP_1-10_202.ckpt True

# 5、PCB+RPP在DukeMTMC-reID上使用G feature评估（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_eval.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/eval_RPP_duke.yaml ./output/checkpoint/RPP/duke/train/RPP-40_258.ckpt True

# 6、PCB+RPP在CUHK03上使用G feature评估（由于训练涉及多个配置文件，因此在这里CONFIG_PATH传入配置文件所在目录路径即可）

bash run_eval.sh RPP cuhk03 ../../Datasets/CUHK03 ../config/eval_RPP_cuhk03.yaml ./output/checkpoint/RPP/cuhk03/train/RPP_1-10_115.ckpt True
```

评估结果保存在脚本目录的output/log/{MODEL_NAME}/{DATASET_NAME}/eval中。

#### Modelarts训练作业

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```text
# 在modelarts上进行模型评估的示例
# (1) 在网页上设置 "config_path='/path_to_code/config/eval_PCB_market.yaml'"
# (2) 把训练好的模型放到桶的对应位置。
# (3) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 在 yaml 文件.
#          设置 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#          设置 "use_G_feature = True (False)" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          增加 "use_G_feature = True (False)" 参数在modearts的界面上。
# (4) 在modelarts的界面上设置代码的路径 "/path/PCB"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的评估。
```

### 结果

- PCB在Market-1501数据集使用G feature进行评估

```log
Mean AP: 78.5%
CMC Scores      market
  top-1          93.0%
  top-5          97.4%
  top-10         98.4%
```

- PCB在Market-1501数据集使用H feature进行评估

```log
Mean AP: 77.9%
CMC Scores      market
  top-1          93.0%
  top-5          97.1%
  top-10         98.1%
```

- PCB在DukeMTMC-reID数据集使用G feature进行评估

```log
Mean AP: 69.8%
CMC Scores        duke
  top-1          84.2%
  top-5          92.4%
  top-10         94.1%
```

- PCB在DukeMTMC-reID数据集使用H feature进行评估

```log
Mean AP: 68.6%
CMC Scores        duke
  top-1          84.2%
  top-5          91.6%
  top-10         93.9%
```

- PCB在CUHK03数据集使用G feature进行评估

```log
Mean AP: 55.1%
CMC Scores      cuhk03
  top-1          61.1%
  top-5          79.5%
  top-10         86.1%
```

- PCB在CUHK03数据集使用H feature进行评估

```log
Mean AP: 55.4%
CMC Scores      cuhk03
  top-1          61.9%
  top-5          79.9%
  top-10         85.9%
```

- RPP在Market-1501数据集使用G feature进行评估

```log
Mean AP: 81.7%
CMC Scores      market
  top-1          93.8%
  top-5          97.5%
  top-10         98.6%
```

- RPP在Market-1501数据集使用H feature进行评估

```log
Mean AP: 81.0%
CMC Scores      market
  top-1          93.6%
  top-5          97.3%
  top-10         98.5%
```

- RPP在DukeMTMC-reID数据集使用G feature进行评估

```log
Mean AP: 71.4%
CMC Scores        duke
  top-1          85.0%
  top-5          92.6%
  top-10         94.4%
```

- RPP在DukeMTMC-reID数据集使用H feature进行评估

```log
Mean AP: 70.2%
CMC Scores        duke
  top-1          85.1%
  top-5          92.1%
  top-10         94.0%
```

- RPP在CUHK03数据集使用G feature进行评估

```log
Mean AP: 58.6%
CMC Scores      cuhk03
  top-1          63.9%
  top-5          81.2%
  top-10         87.1%
```

- RPP在CUHK03数据集使用H feature进行评估

```log
Mean AP: 59.1%
CMC Scores      cuhk03
  top-1          65.0%
  top-5          81.4%
  top-10         86.9%
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出MindIR

导出mindir模型

```shell
python export.py --model_name [MODEL_NAME] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --checkpoint_file_path [CKPT_PATH] --use_G_feature [USE_G_FEATURE] --config_path [CONFIG_PATH]
```

参数model_name可从["PCB","RPP"]中选择。
参数file_name为导出的模型名称。
参数file_format 仅支持 "MINDIR"。
参数checkpoint_file_path为断点路径。
参数use_G_feature表示导出模型是否使用G feature，若不使用，则使用H feature。 feature类型的不同对应导出模型的结构也会不同。
参数config_path表示infer_310_config.yaml的路径

```shell
# 示例：
# 1、导出在Market-1501上训练后使用G feature的PCB模型。
python export.py --model_name "PCB" --file_name "PCB_market_G" --file_format MINDIR --checkpoint_file_path ../PCB_market.ckpt --use_G_feature True --config_path ./config/infer_310_config.yaml
```

ModelArts导出mindir

```text
# (1) 在网页上设置 "config_path='/path_to_code/config/infer_310_config.yaml'"
# (2) 把训练好的模型地方到桶的对应位置。
# (3) 选择a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "model_name='PCB'" 在 yaml 文件。
#          设置 "file_name='PCB_market_G'"参数在yaml文件。
#          设置 "file_format='MINDIR'" 参数在yaml文件。
#          设置 "use_G_feature=True" 参数在 yaml 文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "model_name='PCB'"参数在modearts的界面上。
#          设置 "file_name='PCB_market_G'"参数在modearts的界面上。
#          设置 "file_format='MINDIR'" 参数在modearts的界面上。
#          设置 "use_G_feature=True"参数在modearts的界面上。
# (4) 在modelarts的界面上设置代码的路径 "/path/PCB"。
# (5) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (6) 在modelarts的界面上设置模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始导出mindir。
```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
目前仅支持batch_Size为1的推理。

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [USE_G_FEATURE][CONFIG_PATH] [DEVICE_ID](optional)
```

- `DATASET_NAME` 选择范围：[market, duke, cuhk03]。
- `USE_G_FEATURE` 应与模型导出时的选项一致
- `CONFIG_PATH` 表示infer_310_config.yaml的路径
- `DEVICE_ID` 可选，默认值为0。

```bash
# 示例：
# 1、PCB在Market-1501上使用G feature进行推理。
bash run_infer_310.sh  ../../mindir/PCB_market_G.mindir market ../../Datasets/Market-1501 True ../config/infer_310_config.yaml
```

### 结果

推理结果保存在脚本执行的当前路径，你可以在metrics.log中看到以下精度计算结果。

- PCB在Market-1501数据集使用G feature进行推理

```log
Mean AP: 78.5%
  top-1          93.1%
  top-5          97.4%
  top-10         98.4%
```

- PCB在Market-1501数据集使用H feature进行推理

```log
Mean AP: 77.9%
  top-1          92.9%
  top-5          97.1%
  top-10         98.1%
```

- PCB在DukeMTMC-reID数据集使用G feature进行推理

```log
Mean AP: 69.8%
  top-1          84.2%
  top-5          92.4%
  top-10         94.1%
```

- PCB在DukeMTMC-reID数据集使用H feature进行推理

```log
Mean AP: 68.6%
  top-1          84.2%
  top-5          91.5%
  top-10         93.9%
```

- PCB在CUHK03数据集使用G feature进行推理

```log
Mean AP: 55.1%
  top-1          60.9%
  top-5          79.4%
  top-10         86.1%
```

- PCB在CUHK03数据集使用H feature进行推理

```log
Mean AP: 55.3%
  top-1          61.7%
  top-5          80.1%
  top-10         86.0%
```

由于PCB+RPP模型含有AvgPool3D算子，该算子在Ascend310环境暂不支持，因此这一部分未进行推理。

# 模型描述

## 性能

### 评估性能

#### Market-1501上的PCB

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | Market-1501
| 训练参数        | epoch=60, steps per epoch=202, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.05631405
| 速度                      | 175毫秒/步（1卡）
| 总时长                 | 37分钟
| 参数(M)             | 27.2

#### DukeMTMC-reID上的PCB

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | DukeMTMC-reID
| 训练参数        | epoch=60, steps per epoch=258, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.095855206
| 速度                      | 132毫秒/步（1卡）
| 总时长                 | 36分钟
| 参数(M)             | 27.2

#### CUHK03上的PCB

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | CUHK03
| 训练参数        | epoch=85, steps per epoch=115, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.094226934
| 速度                      | 137毫秒/步（1卡）
| 总时长                 | 25分钟
| 参数(M)             | 27.2

#### Market-1501上的PCB-RPP

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB-RPP
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | Market-1501
| 训练参数        | epoch=75, steps per epoch=202, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.04336106
| 速度                      | 307毫秒/步（1卡）
| 总时长                 | 72分钟
| 参数(M)             | 27.2

#### DukeMTMC-reID上的PCB-RPP

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB-RPP
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | DukeMTMC-reID
| 训练参数        | epoch=60, steps per epoch=258, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       |  0.03547495
| 速度                      | 264毫秒/步（1卡）
| 总时长                 | 59分钟
| 参数(M)             | 27.2

#### CUHK03上的PCB-RPP

| 参数                 | Ascend 910
| -------------------------- | --------------------------------------
| 模型版本              | PCB-RPP
| 资源                   | Ascend 910；CPU 2.60GHz，24核；内存 96G；系统 Euler2.8
| MindSpore版本          | 1.3.0
| 数据集                    | CUHK03
| 训练参数        | epoch=95, steps per epoch=115, batch_size = 64
| 优化器                  | SGD
| 损失函数              | Softmax交叉熵
| 输出                    | 概率
| 损失                       | 0.083887264
| 速度                      | 268毫秒/步（1卡）
| 总时长                 | 59分钟
| 参数(M)             | 27.2

# 随机情况说明

`dataset.py`中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
