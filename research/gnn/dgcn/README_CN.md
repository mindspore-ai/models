目录

<!-- TOC -->

- [目录](#目录)
    - [网络描述](#网络描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练、评估、测试过程](#训练评估测试过程)
            - [用法](#用法)
            - [启动](#启动)
            - [结果](#结果)
    - [推理过程](#推理过程)
       - [导出MindIR](#导出mindir)
       - [在Ascend310执行推理](#在Ascend310执行推理)
    - [模型描述](#模型描述)
    - [性能](#性能)
    - [随机情况说明](#随机情况说明)

<!-- /TOC -->

## 网络描述

双通路图卷积神经网络（DGCN）于2018年提出，它基于半监督学习的两个基本假设：（1).局部一致性：距离比较近的数据，通常有相同的标签；(2).全局一致性：处在相似的上下文中的数据，通常有相同的标签，使用了双通路分别嵌入半监督学习局部一致性与全局一致性的信息，并设计了新的loss函数将二者结合，来对图结构数据进行半监督学习，取得了很好的实验效果。

[论文](https://www.researchgate.net/publication/324514333_Dual_Graph_Convolutional_Networks_for_Graph-Based_Semi-Supervised_Classification):  Dual Graph Convolutional Networks for Graph-Based Semi-Supervised Classification[C]// the 2018 World Wide Web Conference. 2018.

## 模型架构

模型架构如图：![输入图片说明](https://images.gitee.com/uploads/images/2021/0925/213545_fe5290a2_7510699.png "捕获.PNG")

模型采用并行的两个前馈神经网络ConvA和ConvP，两个并行网络共享参数，区别为输入的图结构信息不同。ConvA为局部一致性卷积，输入为图的邻接矩阵，ConvP为全局一致性卷积， 输入为基于随机游走的策略生成的PPMI矩阵。首先，上面支路利用有标签的节点计算交叉熵损失，并对网络参数进行训练，得到一种后验分布。之后逐渐增加下面支路得到的均方差损失的权值，使得两个损失同时对模型参数产生影响。

## [数据集](https://github.com/kimiyoung/planetoid/tree/master/data)

|  数据集  | 节点  |  边   |  类  | 特征 |
| :------: | :---: | :---: | :--: | :--: |
|   Cora   | 2708  | 5429  |  7   | 1433 |
| Citeseer | 3327  | 4732  |  6   | 3703 |
|  Pubmed  | 19717 | 44338 |  3   | 500  |

## 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 安装[MindSpore](https://www.mindspore.cn/install)
- 安装相关依赖 pip install -r requirements.txt
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 下载数据集Cora，Citeseer和Pubmed，[可点此下载](https://github.com/kimiyoung/planetoid/tree/master/data)
- 将数据集放到代码目录下，文件夹应该包含如下文件（以Cora数据集为例）：

```text
.
└─dgcn
  ├─scripts
  ├─src
  ├─data
  | ├─ind.cora.allx
  | ├─ind.cora.ally
  | ├─ind.cora.graph
  | ├─ind.cora.test.index
  | ├─ind.cora.tx
  | ├─ind.cora.ty
  | ├─ind.cora.x
  | └─ind.cora.y
  ...
```

## 脚本说明

### 脚本及样例代码

```shell
.
└─dgcn
  |
  ├─scripts
  | ├─run_eval.sh          # Ascend启动评估
  | ├─run_eval_gpu.sh      # GPU启动评估
  | ├─run_train_8p.sh      # Ascend启动多卡训练
  | ├─run_train_8p_gpu.sh  # GPU启动多卡训练
  | ├─run_train.sh         # Ascend启动训练
  | └─run_train_gpu.sh     # GPU启动训练
  |
  ├─src
  | ├─config.py            # 参数配置
  | ├─data_process.py      # 数据预处理
  | ├─dgcn.py              # GCN骨干
  | ├─utilities.py         # 计算PPMI矩阵
  | └─metrics.py           # 计算损失和准确率
  |
  ├─README_CN.md
  ├─requirements.txt       # 依赖包
  ├─eval.py                # 评估
  ├─export.py             # 导出模型
  └─train.py               # 训练
```

### 脚本参数

训练参数可以在config.py中配置。

```python
"learning_rate": 0.01,            # 学习率
"epochs": 200,                    # 训练轮次
"hidden1": 36,                    # 第一图卷积层隐藏大小
"dropout": 0.5,                   # 第一图卷积层dropout率
"weight_decay": 5e-4,             # 第一图卷积层参数的权重衰减
"path_len" = 2,                   # 随机游走路径长度
"early_stopping": 120,            # 早停容限
```

### 训练、评估、测试过程

#### 训练过程

```bash
cd ./script
单卡模式下：
# 使用Cora或Citeseer或Pubmed数据集进行训练，DATASET_NAME为cora或citeseer或pubmed，DEVICE_ID为卡的物理序号
bash run_train.sh [DATASET_NAME][DEVICE_ID]
多卡模式下：
#rank_size为总卡数，device_start为起始卡的序号，distributed请设置为True
bash run_train_8p.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START] [DATASET_NAME] [DISTRIBUTED]
###
# GPU处理器运行
cd ./script
单卡模式下：
# 使用Cora或Citeseer或Pubmed数据集进行训练，DATASET_NAME为cora或citeseer或pubmed
bash run_train_gpu.sh [DATASET_NAME] [DEVICE_ID]
多卡模式下：
# DEVICE_NUM为总卡数，VISIABLE_DEVICES(0,1,2,3,4,5,6,7)为卡的物理序号列表
bash run_train_8p_gpu.sh [DATASET_NAME] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)]
```

- rank_table用[此方法](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)生成并放在script文件夹下
- 注意：代码中early_stopping机制可能会导致训练早停，但不影响实验结果

##### 启动

```bash
# Ascend处理器运行
单卡：bash run_train.sh cora 4
多卡：bash run_train_8p.sh ./hccl_8p_01234567_127.0.0.1.json 8 0 cora True
# GPU处理器运行
单卡：bash run_train_gpu.sh cora
多卡：bash run_train_8p_gpu.sh cora 8 0,1,2,3,4,5,6,7
```

##### 结果

```text
单卡结果可以在script文件夹下train子文件夹中生成的train.log文件下查看
Epoch: 0180 train_loss= 0.06309 train_acc= 0.98571 val_loss= 0.26957 val_acc= 0.75400 time= 0.01507
Test set results: accuracy= 0.81200 time= 0.00680
Epoch: 0181 train_loss= 0.06287 train_acc= 1.00000 val_loss= 0.26949 val_acc= 0.75200 time= 0.01516
Test set results: accuracy= 0.80900 time= 0.00680
Epoch: 0182 train_loss= 0.06290 train_acc= 1.00000 val_loss= 0.26940 val_acc= 0.76000 time= 0.01516
Test set results: accuracy= 0.81100 time= 0.00686
Epoch: 0183 train_loss= 0.06261 train_acc= 1.00000 val_loss= 0.26943 val_acc= 0.76200 time= 0.01511
Test set results: accuracy= 0.80900 time= 0.00685
Epoch: 0184 train_loss= 0.06300 train_acc= 1.00000 val_loss= 0.26931 val_acc= 0.76400 time= 0.01506
Test set results: accuracy= 0.80500 time= 0.00677
Epoch: 0185 train_loss= 0.06287 train_acc= 1.00000 val_loss= 0.26921 val_acc= 0.76600 time= 0.01497
Test set results: accuracy= 0.80000 time= 0.00677
Epoch: 0186 train_loss= 0.06298 train_acc= 1.00000 val_loss= 0.26921 val_acc= 0.76800 time= 0.01509
Test set results: accuracy= 0.79800 time= 0.00672
Epoch: 0187 train_loss= 0.06309 train_acc= 1.00000 val_loss= 0.26917 val_acc= 0.76200 time= 0.01518
Test set results: accuracy= 0.79700 time= 0.00679
Epoch: 0188 train_loss= 0.06285 train_acc= 1.00000 val_loss= 0.26906 val_acc= 0.76200 time= 0.01508
Test set results: accuracy= 0.80200 time= 0.00680
Epoch: 0189 train_loss= 0.06290 train_acc= 1.00000 val_loss= 0.26900 val_acc= 0.76200 time= 0.01518
Test set results: accuracy= 0.80200 time= 0.00675
Epoch: 0190 train_loss= 0.06260 train_acc= 1.00000 val_loss= 0.26905 val_acc= 0.76400 time= 0.01507
Test set results: accuracy= 0.80000 time= 0.00684
Epoch: 0191 train_loss= 0.06273 train_acc= 1.00000 val_loss= 0.26923 val_acc= 0.76800 time= 0.01508
Test set results: accuracy= 0.80500 time= 0.00687
Epoch: 0192 train_loss= 0.06264 train_acc= 1.00000 val_loss= 0.26951 val_acc= 0.76400 time= 0.01505
Test set results: accuracy= 0.79500 time= 0.00678
Epoch: 0193 train_loss= 0.06276 train_acc= 1.00000 val_loss= 0.26973 val_acc= 0.76000 time= 0.01499
Test set results: accuracy= 0.79300 time= 0.00676
Epoch: 0194 train_loss= 0.06329 train_acc= 1.00000 val_loss= 0.27001 val_acc= 0.75800 time= 0.01507
Early stopping...
Best test accuracy =  0.82800
Finished
```

```text
多卡结果可以在script文件夹下每个device子文件夹中生成的train.log文件下查看
Epoch: 0140 train_loss= 0.06275 train_acc= 0.99286 val_loss= 0.27006 val_acc= 0.77000 time= 0.03695
Test set results: accuracy= 0.79500 time= 0.01787
Epoch: 0141 train_loss= 0.06301 train_acc= 1.00000 val_loss= 0.27004 val_acc= 0.76600 time= 0.03697
Test set results: accuracy= 0.79800 time= 0.01781
Epoch: 0142 train_loss= 0.06317 train_acc= 1.00000 val_loss= 0.26997 val_acc= 0.76000 time= 0.03667
Test set results: accuracy= 0.80400 time= 0.01776
Epoch: 0143 train_loss= 0.06296 train_acc= 1.00000 val_loss= 0.27046 val_acc= 0.75600 time= 0.03642
Test set results: accuracy= 0.79900 time= 0.01749
Epoch: 0144 train_loss= 0.06290 train_acc= 1.00000 val_loss= 0.27060 val_acc= 0.75200 time= 0.03668
Test set results: accuracy= 0.79600 time= 0.01777
Epoch: 0145 train_loss= 0.06313 train_acc= 1.00000 val_loss= 0.27096 val_acc= 0.75400 time= 0.03674
Test set results: accuracy= 0.79300 time= 0.01765
Epoch: 0146 train_loss= 0.06316 train_acc= 1.00000 val_loss= 0.27020 val_acc= 0.75600 time= 0.03687
Test set results: accuracy= 0.79700 time= 0.01771
Epoch: 0147 train_loss= 0.06306 train_acc= 0.99286 val_loss= 0.26951 val_acc= 0.76000 time= 0.03751
Test set results: accuracy= 0.79500 time= 0.01792
Epoch: 0148 train_loss= 0.06333 train_acc= 1.00000 val_loss= 0.26820 val_acc= 0.77200 time= 0.03689
Test set results: accuracy= 0.79900 time= 0.01787
Epoch: 0149 train_loss= 0.06363 train_acc= 1.00000 val_loss= 0.26731 val_acc= 0.77800 time= 0.03689
Test set results: accuracy= 0.80400 time= 0.01795
Epoch: 0150 train_loss= 0.06285 train_acc= 1.00000 val_loss= 0.26687 val_acc= 0.77400 time= 0.03671
Test set results: accuracy= 0.80400 time= 0.01787
Epoch: 0151 train_loss= 0.06367 train_acc= 0.99286 val_loss= 0.26749 val_acc= 0.77400 time= 0.03684
Test set results: accuracy= 0.80400 time= 0.01783
Epoch: 0152 train_loss= 0.06303 train_acc= 1.00000 val_loss= 0.26867 val_acc= 0.76400 time= 0.03658
Test set results: accuracy= 0.80600 time= 0.01761
Epoch: 0153 train_loss= 0.06285 train_acc= 1.00000 val_loss= 0.27015 val_acc= 0.76000 time= 0.03639
Test set results: accuracy= 0.79800 time= 0.01759
Epoch: 0154 train_loss= 0.06326 train_acc= 1.00000 val_loss= 0.27158 val_acc= 0.74600 time= 0.03669
Early stopping...
Best test accuracy =  0.82200
Finished
```

#### 评估过程

```bash
# Ascend处理器运行
# CHECKPOINT为保存的模型文件绝对路径
cd ./script
bash run_eval.sh [CHECKPOINT] [DATASET]
# GPU处理器运行
cd ./script
bash run_eval_gpu.sh [CHECKPOINT] [DATASET_NAME]
```

##### 启动

```bash
# Ascend处理器运
bash run_eval.sh ./checkpoint/cora/dgcn.ckpt cora
# GPU处理器运行
bash run_eval_gpu.sh ./checkpoint/cora/dgcn.ckpt cora
```

##### 结果

```text
结果可以在script文件夹下eval子文件夹中生成的eval.log文件下查看
Feature matrix:(2708, 1433)
Label matrix:(2708, 7)
Adjacent matrix:(2708, 2708)
Do the sampling...
Calculating the PPMI...
1.0
Offset: 0.0
Sparsity: 0.9981942556547807
Convolution Layers:[(1433, 36), (36, 7)]
Eval results: loss= 0.52596 accuracy= 0.82800 time= 13.57054
```

## 推理过程

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

示例

```text
python export.py --ckpt_file ./checkpoint/cora/dgcn.ckpt
参数ckpt_file为必填项，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。
```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` 表示数据集名称，取值范围： ['cora', 'citeseer'， 'pubmed']。
- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围：'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### result

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
Test set results: accuracy= 0.82800
```

## 模型描述

### 性能

| 参数          | Ascend 910                                                   | GPU                                                          |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 资源          | Ascend 910                                                   | GPU                                                          |
| 上传日期      | 2021-09-25                                                   | 2021-10-30                                                   |
| MindSpore版本 | 1.3.0                                                        | 1.4.1                                                        |
| 数据集        | cora/citeseer/pubmed                                         | cora/citeseer/pubmed                                         |
| 训练参数      | epoch=200,dropout=0.5,learning_rate:cora和pubmed设置为0.01，citeseer设置为0.0153 | epoch=200,dropout=0.5,learning_rate: 详见代码                |
| 优化器        | Adam                                                         | Adam                                                         |
| 损失函数      | Softmax交叉熵                                                | Softmax交叉熵                                                |
| 损失          | 0.06288/0.3948/0.00199                                       | 0.06255/0.3899/0.00201                                       |
| 训练时间      | 3分钟/3分钟/8分钟                                            | 2分钟/2分钟/7分钟                                            |
| 卡数          | 单卡                                                         | 单卡                                                         |
| 准确率        | 82.8/72.2/80.3                                               | 82.8/72.2/80.2                                               |
| 脚本          | [DGCN脚本](https://gitee.com/mindspore/models/tree/master/research/gnn/dgcn) | [DGCN脚本](https://gitee.com/mindspore/models/tree/master/research/gnn/dgcn) |

## 随机情况说明

train.py和eval.py脚本中使用mindspore.set_seed()对全局随机种子进行了固定，可在对应的parser中进行修改即可，cora和pubmed默认为1024，citeseer默认为1235，GPU参数详见代码。
