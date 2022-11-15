# 目录

<!-- TOC -->

- [目录](#目录)
- [MMoE描述](#mmoe描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [Ascend处理器或GPU环境或CPU环境运行](#Ascend处理器或GPU环境或CPU环境运行)
        - [结果](#结果)
- [评估过程](#评估过程)
    - [评估用法](#评估用法)
    - [Ascend处理器或GPU环境或CPU环境运行评估](#Ascend处理器或GPU环境或CPU环境运行评估)
    - [结果](#结果)
- [Ascend310推理过程](#推理过程)
    - [导出MindIR](#导出MindIR)
    - [在Acsend310执行推理](#在Acsend310执行推理)
    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [census-income上的MMoE](#census-income上的MMoE)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MMoE描述

## 概述

为了解决任务之间相关性降低导致模型效果下降的问题，在MoE的基础上进行改进，提出了MMoE。MMoE为每一个task设置一个gate，用这些gate控制不同任务不同专家的权重。

## 论文

1. [论文](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-): modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-

# 模型架构

MMoE的总体网络架构如下：
![Architecture](http://img.5iqiqu.com/images5/52/528211eb81718499fc2475dfcdcd690e.png)

# 数据集

使用的数据集：[Census-income](http://github.com/drawbridge/keras-mmoe)

- 数据集大小：共9.4Mb、299,285条数据
    - 训练集：共6.3Mb，199,523条数据

    - 测试集：共3.1Mb，99726条数据

    - 注：数据在data.py中处理成mindrecord格式。

      使用命令 python data.py --local_data_path  ./Census-income
- 下载原始数据集，目录结构如下：

```text
└─Census-income
  ├── Census-income.data.gz             # 训练数据集
  ├── Census-income.test.gz             # 评估数据集
```

# 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器或GPU环境运行

```Shell
# 分布式训练（Ascend）
Usage: bash run_distribution_ascend.sh [RANK_TABLE_FILE] [DATA_PATH] [CKPT_PATH] [CONFIG_FILE]
[RANK_TABLE_FILE]是多卡的具体信息。
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是要将ckpt保存的位置。
[CONFIG_FILE]是模型及运行的整体参数。

# 单机训练(Ascend)
Usage: bash run_standalone_train_ascend.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE]
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是要将ckpt保存的位置。
[DEVICE_ID]为执行train.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 单机训练(GPU)
Usage: bash run_standalone_train_gpu.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE]
[DATA_PATH]是数据集的路径(mindrecord文件所在的目录)。
[CKPT_PATH]是要将ckpt保存的位置。
[DEVICE_ID]为执行train.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 运行评估示例（Ascend）
Usage: bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 运行评估示例（GPU）
Usage: bash run_standalone_eval_gpu.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是数据集的路径(mindrecord文件所在的目录)。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。
```

# 脚本说明

## 脚本及样例代码

```text
└──mmoe
  ├── README_CN.md
  ├── ascend310_infer
    ├── inc
      ├── util.h
    ├── src
      ├── build.sh
      ├── CMakeList.txt
      ├── main.cc
      ├── utils.cc
  ├── scripts
    ├── run_distribute_ascend.sh            # 启动Ascend分布式训练（8卡）
    ├── run_standalone_eval_ascend.sh       # 启动Ascend910评估
    ├── run_standalone_eval_gpu.sh          # 启动GPU评估
    ├── run_infer_310.sh                    # 启动Ascend310评估
    ├── run_standalone_train_ascend.sh      # 启动Ascend单机训练（单卡）
    └── run_standalone_train_gpu.sh         # 启动GPU单机训练（单卡）
  ├── src
    ├── model_utils
        ├── config.py                        # 参数配置
        ├── device_adapter.py                # 适配云上或线下
        ├── local_adapter.py                 # 线下配置
        ├── moxing_adapter.py                # 云上配置
    ├── callback.py                          # 训练过程中进行评估的回调  
    ├── data.py                              # 数据预处理
    ├── load_dataset.py                      # 加载处理好的mindrecord格式数据
    ├── get_lr.py                            # 生成每个步骤的学习率
    ├── mmoe.py                              # 模型整体架构
    └── mmoe_utils.py                        # 每一层架构
  ├── eval.py                                # 910评估网络
  ├── default_config.yaml                    # 默认的参数配置
  ├── default_config_cpu.yaml                # 针对CPU环境默认的参数配置
  ├── default_config_gpu.yaml                # 针对GPU环境默认的参数配置
  ├── export.py                              # 910导出网络
  ├── fine_tune.py                           # CPU训练网络
  ├── postprocess.py                         # 310推理精度计算
  ├── preprocess.py                          # 310推理前数据处理
  └── train.py                               # 910训练网络
```

# 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

- 配置MMoE和Census-income数据集。

```Python
"num_features":499,                # 每一条数据的特征数
"num_experts":8,                   # 专家数
"units":4,                         # 每一层的unit数
"batch_size":32,                   # 输入张量的批次大小
"epoch_size":100,                  # 训练周期大小
"learning_rate":0.001,             # 初始学习率
"save_checkpoint":True,            # 是否保存检查点
"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":5,                 # 热身周期
```

- CPU环境下参数设置

```Python
"num_features":499,                # 每一条数据的特征数
"num_experts":8,                   # 专家数
"units":4,                         # 每一层的unit数
"batch_size":32,                   # 输入张量的批次大小
"epoch_size":10,                  # 训练周期大小
"learning_rate":0.0001,             # 初始学习率
"save_checkpoint":True,            # 是否保存检查点
"save_checkpoint_epochs":1,        # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,          # 只保存最后一个keep_checkpoint_max检查点
"warmup_epochs":5,                 # 热身周期
```

# 训练过程

## 用法

## Ascend处理器或GPU环境运行

```Shell
# 分布式训练（Ascend）
Usage: bash run_distribution_ascend.sh [RANK_TABLE_FILE] [DATA_PATH] [CKPT_PATH] [CONFIG_FILE]
[RANK_TABLE_FILE]是多卡的具体信息。
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是要将ckpt保存的位置。
[CONFIG_FILE]是模型及运行的整体参数。

# 单机训练(Ascend)
Usage: bash run_standalone_train_ascend.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE]
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是要将ckpt保存的位置。
[DEVICE_ID]为执行train.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 单机训练(GPU)
Usage: bash run_standalone_train_gpu.sh [DATA_PATH] [DEVICE_ID] [CKPT_PATH] [CONFIG_FILE]
[DATA_PATH]是数据集的路径(mindrecord文件所在的目录)。
[CKPT_PATH]是要将ckpt保存的位置。
[DEVICE_ID]为执行train.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 运行评估示例（Ascend）
Usage: bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。

# 运行评估示例（GPU）
Usage: bash run_standalone_eval_gpu.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是数据集的路径(mindrecord文件所在的目录)。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

训练结果保存在示例路径中，文件夹名称以“train”或“train_parallel”开头。您可在此路径下的日志中找到检查点文件以及结果，如下所示。

## CPU环境运行

### 数据处理

[根据提供的数据集链接加载数据集](http://github.com/drawbridge/keras-mmoe)在train.py文件同级目录下新建data文件夹，执行src中文件

```shell
python data.py --local_data_path  ../data
```

即可得到所需的测试集与验证集。

### 用法

您可以通过python脚本开始训练

```shell
python train.py  --config_path ./default_config_cpu.yaml
```

## 结果

- 使用census-income数据集训练MMoE

```text
# 分布式训练结果（8P Ascend）
epoch: 1 step: 780, loss is 0.5584626
epoch: 2 step: 780, loss is 0.72126234
epoch: 3 step: 780, loss is 0.28167123
epoch: 4 step: 780, loss is 0.19597104
epoch: 5 step: 780, loss is 0.28420725
epoch: 6 step: 780, loss is 0.32970852
epoch: 7 step: 780, loss is 0.26188123
epoch: 8 step: 780, loss is 0.15461507
epoch: 9 step: 780, loss is 0.37079066
epoch: 10 step: 780, loss is 0.2761521

...

# 单卡GPU训练结果
epoch: 1 step: 1558, loss is 0.7738624215126038
epoch time: 23271.168 ms, per step time: 14.937 ms
start infer...
infer data finished, start eval...
result : income_auc=0.956143804122577, marital_auc=0.8883598309142848, use time 2s
The best income_auc is 0.956143804122577,             the best marital_auc is 0.8883598309142848,             the best income_marital_auc_avg is 0.9222518175184309
epoch: 2 step: 1558, loss is 0.4517086148262024
epoch time: 17804.081 ms, per step time: 11.428 ms
start infer...
infer data finished, start eval...
result : income_auc=0.9856142129882843, marital_auc=0.9194419616798691, use time 1s
The best income_auc is 0.9856142129882843,             the best marital_auc is 0.9194419616798691,             the best income_marital_auc_avg is 0.9525280873340767
epoch: 3 step: 1558, loss is 0.41103610396385193
epoch time: 17853.932 ms, per step time: 11.460 ms
start infer...
infer data finished, start eval...
result : income_auc=0.9876599788311389, marital_auc=0.9663552616198483, use time 1s
The best income_auc is 0.9876599788311389,             the best marital_auc is 0.9663552616198483,             the best income_marital_auc_avg is 0.9770076202254936
...

# 单卡CPU训练结果
epoch: 1 step: 6235, loss is 0.8481878638267517
Train epoch time: 27365.547 ms, per step time: 4.389 ms
start infer...
infer data finished, start eval...
result : income_auc=0.9528846425952942, marital_auc=0.7993896372126021, use time 8s
The best income_auc is 0.9528846425952942,             the best marital_auc is 0.7993896372126021,             the best income_marital_auc_avg is 0.8761371399039481
epoch: 2 step: 6235, loss is 0.5404471158981323
Train epoch time: 17965.760 ms, per step time: 2.881 ms
start infer...
infer data finished, start eval...
result : income_auc=0.9833082917947681, marital_auc=0.9176945078776066, use time 5s
The best income_auc is 0.9833082917947681,             the best marital_auc is 0.9176945078776066,             the best income_marital_auc_avg is 0.9505013998361873
epoch: 3 step: 6235, loss is 0.26600515842437744
Train epoch time: 20357.339 ms, per step time: 3.265 ms
start infer...
infer data finished, start eval...
result : income_auc=0.9843190639741299, marital_auc=0.9634857856721967, use time 4s
The best income_auc is 0.9843190639741299,             the best marital_auc is 0.9634857856721967,             the best income_marital_auc_avg is 0.9739024248231634
...
```

# 评估过程

## 评估用法

### Ascend处理器或GPU环境运行评估

```Shell
# 评估
Usage: bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID] [CONFIG_FILE]
[DATA_PATH]是数据集的路径。
[CKPT_PATH]是保存ckpt的位置。
[DEVICE_ID]为执行eval.py的ID号。
[CONFIG_FILE]是模型及运行的整体参数。
```

```Shell
# 评估示例
bash  run_standalone_eval_ascend.sh  /home/mmoe/data/ /home/mmoe/MMoE_train-50_6236.ckpt 1 /home/mmoe/default_config.yaml
```

### CPU环境运行评估

您可以通过python脚本开始进行评估

```shell
python eval.py --data_path ././data/mindrecord/ --ckpt_file ./ckpt/best_marital_auc.ckpt
```

其中././data/mindrecord/验证集路径，./ckpt/ckpt/best_marital_auc.ckpt为选择的最好ckpt文件。

## 结果

评估结果保存在示例路径中，您可在此路径下的日志找到如下结果：

- 使用census-income数据集评估MMoE

```text
result: {'income_auc': 0.9969135802469136, 'marital_auc': 1.0}
```

- cpu环境下使用census-income数据集评估MMoE

```text
result : income_auc=0.9872372448355503, marital_auc=0.9820659214506045
```

# Ascend310推理过程

## 导出MindIR

```shell
python export.py
```

参数ckpt_file为必填项，
`file_format` 必须在 ["AIR", "MINDIR"]中选择,本模型导出MINDIR格式。

## 在Ascend310执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理前，mindir文件必须通过`export.py`脚本导出，同时使用的验证数据集必须在910场景下通过`data.py`将原验证数据处理成mindrecord格式，其中`data.py`中do_train=False。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATA_PATH` 推理数据集路径
- `NEED_PREPROCESS` 是否需要对数据做预处理，本次推理需要对数据做预处理，因此默认为：y。
- `DEVICE_ID` 可选，默认值为0。

## 结果

推理结果保存在脚本执行的当前路径，
你可以在当前文件夹中acc.log查看推理精度，在time_Result中查看推理时间。

# 模型描述

## 性能

### 评估性能

#### census-income上的MMoE

| 参数 | Ascend 910  | V100 GPU |
|---|---|---|
| 模型版本  | MMoE  |MMoE|
| 资源  |  Ascend 910；CPU：2.60GHz，192核；内存：755G |V100 GPU；CPU：8核；内存：64G|
| 上传日期  |2021-11-12 ;  |2022-2-19|
| MindSpore版本  | 1.3.0 |1.6.0|
| 数据集  | census-income |census-income|
| 训练参数  | epoch=100, batch_size = 32  |epoch=100, batch_size = 128|
| 优化器  | Adam  |Adam|
| 损失函数  | BCELoss |BCELoss|
| 输出  | 概率 |概率|
|  损失 | 0.20949207 |0.21848808228969574|
|速度|0.671毫秒/步 |11.399毫秒/步|
|总时长   |  17分钟 |32分钟|
|参数   | 23.55KB |23.55KB|
|精度指标   | best income_auc:0.9895    best marital_auc:0.9837 |best income_auc:0.9892    best marital_auc:0.9826|
|  微调检查点 | 2.66MB（.ckpt文件）  |893.8KB（.ckpt文件）|
| 脚本  | [链接](https://gitee.com/mindspore/models/tree/master/research/recommend/mmoe)  |[链接](https://gitee.com/mindspore/models/tree/master/research/recommend/mmoe)|

| 参数 | i5-10400 CPU  |
|---|---|
| 模型版本  | MMoE  |
| 资源  |  i5-10400 CPU 2.90GHz |
| 上传日期  |2022-9-30 ;  |
| MindSpore版本  | 1.8.0 |
| 数据集  | census-income |
| 训练参数  | epoch=10, batch_size = 32  |
| 优化器  | Adam  |
| 损失函数  | BCELoss |
| 输出  | 概率 |
|  损失 | 0.209593266248703 |
|速度|3.437毫秒/步 |
|总时长   |  17分钟 |
|参数   | 23.55KB |
|精度指标   | best income_auc:0.9872    best marital_auc:0.9820 |
|  微调检查点 | 4.92 MB （.ckpt文件）  |

# 随机情况说明

train.py中使用随机种子

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

- **Q: 使用PYNATIVE_MODE发生内存溢出怎么办？** **A**：内存溢出通常是因为PYNATIVE_MODE需要更多的内存， 将batch size设置为16降低内存消耗，可进行网络训练。
