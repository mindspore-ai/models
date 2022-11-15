# 目录

<!-- TOC -->

- [目录](#目录)
- [RDN描述](#RDN描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练](#训练)
        - [评估](#评估)
    - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练-1)
    - [评估过程](#评估过程)
        - [评估](#评估-1)
    - [推理过程](#推理过程)
        - [模型导出](#模型导出)
        - [执行推理](#执行推理)
        - [查看结果](#查看结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RDN描述

一个非常深的卷积神经网络 (CNN) 最近在图像超分辨率 (SR) 方面取得了巨大成功，并且还提供了分层特征。然而，大多数基于深度 CNN 的 SR 模型没有充分利用原始低分辨率 (LR) 图像的层次特征，从而实现相对较低的性能。在本文中，我们提出了一种新颖的残差密集网络 (RDN) 来解决图像 SR 中的这个问题。我们充分利用了所有卷积层的分层特征。具体来说，我们提出残差密集块（RDB）来通过密集连接的卷积层提取丰富的局部特征。 RDB 还允许从先前 RDB 的状态直接连接到当前 RDB 的所有层，从而形成连续内存 (CM) 机制。然后使用 RDB 中的局部特征融合从先前和当前局部特征中自适应地学习更有效的特征，并稳定更广泛网络的训练。在充分获得密集的局部特征后，我们使用全局特征融合以整体的方式联合自适应地学习全局层次特征。在具有不同退化模型的基准数据集上进行的实验表明，我们的 RDN 相对于最先进的方法取得了良好的性能。
![RDB](Figs/RDB.png)
Figure 1. Residual dense block (RDB) architecture.
![RDN](Figs/RDN.png)
Figure 2. The architecture of our proposed residual dense network (RDN).

# 数据集

## 使用的数据集：[Div2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- 数据集大小：约7.12GB，共900张图像
 - 训练集：前800张图像
- 基准数据集可下载如下：[Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)。
- 数据格式：png文件
 - 注：数据将在src/data/DIV2K.py中处理。

```bash
DIV2K
├── DIV2K_train_HR
│   ├── 0001.png
│   ├─ ...
│   └── 0900.png
├── DIV2K_train_LR_bicubic
│   ├── X2
│   │   ├── 0001x2.png
│   │   ├─ ...
│   │   └── 0900x2.png
│   ├── X3
│   │   ├── 0001x3.png
│   │   ├─ ...
│   │   └── 0900x3.png
│   └── X4
│       ├── 0001x4.png
│       ├─ ...
│       └── 0900x4.png
└── DIV2K_train_LR_unknown
    ├── X2
    │   ├── 0001x2.png
    │   ├─ ...
    │   └── 0900x2.png
    ├── X3
    │   ├── 0001x3.png
    │   ├─ ...
    │   └── 0900x3.png
    └── X4
        ├── 0001x4.png
        ├─ ...
        └── 0900x4.png
```

# 环境要求

- 硬件
    - Ascend: 准备Ascend处理器搭建硬件环境。
    - GPU: 准备GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                                 // 所有模型相关说明
    ├── RDN
        ├── Figs
        │   ├── RDB.png                           // RDB结构示意图
        │   ├── RDN.png                           // RDN网络结构示意图
        ├── scripts
        │   ├── run_ascend_distribute.sh          // Ascend分布式训练shell脚本
        │   ├── run_eval.sh                       // eval验证shell脚本
        │   ├── run_ascend_standalone.sh          // Ascend训练shell脚本
        ├── src
        │   ├── data
        │   │   ├──common.py                      //公共数据集
        │   │   ├──div2k.py                       //div2k数据集
        │   │   ├──srdata.py                      //所有数据集
        │   ├── model.py                          //RDN网络
        │   ├── metrics.py                        //PSNR,SSIM计算器
        │   ├── args.py                           //超参数
        ├── train.py                              //训练脚本
        ├── eval.py                               //评估脚本
        ├── export.py                             //模型导出
        ├── README.md                             //自述文件
```

## 脚本参数

### 训练

```bash
用法：python train.py [--device_target][--dir_data]
              [--ckpt_path][--test_every][--scale][--task_id]
选项：
  --device_target       训练后端类型，Ascend，默认为Ascend。
  --dir_data            数据集存储路径。
  --ckpt_path           存放检查点的路径。
  --test_every          每N批进行一次试验。
  --scale               超分规模
  --task_id             任务ID。
```

### 评估

```bash
用法：python eval.py [--device_target][--dir_data]
               [--task_id][--scale][--data_test]
               [--ckpt_save_path]

选项：
  --device_target       评估后端类型，Ascend。
  --dir_data            数据集路径。
  --task_id             任务id。
  --scale               超分倍数。
  --data_test           测试数据集名字。
  --ckpt_path           检查点路径。
```

## 参数配置

在args.py中可以同时配置训练参数和评估参数。

- RDN配置，div2k数据集

```bash
"lr": 0.00005,                       # 学习率
"epochs": 1800,                      # 训练轮次数
"batch_size": 16,                    # 输入张量的批次大小
"weight_decay": 0,                   # 权重衰减
"loss_scale": 1024,                  # 损失放大
"buffer_size": 10,                   # 混洗缓冲区大小
"init_loss_scale":65536,             # 比例因子
"betas":(0.9, 0.999),                # ADAM beta
"weight_decay":0,                    # 权重衰减
"test_every":1000,                   # 每N批进行一次试验
"patch_size":48,                     # 输出块大小
"scale":'2',                         # 超分辨率比例尺
"task_id":0,                         # 任务id
"n_colors":3,                        # 颜色通道数
"RDNkSize":3,                        # kernel size
"G0":64,                             # default number of filters
```

## 训练过程

### 训练

#### Ascend处理器环境运行RDN

- 单设备训练（1p)
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2
- 需要指定训练集路径(TRAIN_DATA_DIR)和芯片序号(DEVICE_ID)

```bash
sh scripts/run_ascend_distribute.sh [TRAIN_DATA_DIR] [DEVICE_ID]
```

- 分布式训练
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2
- 需要指定配置文件(RANK_TABLE_FILE)、训练集路径(TRAIN_DATA_DIR)和芯片数量(DEVICE_NUM)

```bash
sh scripts/run_ascend_distribute.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR] [DEVICE_NUM]
```

- 分布式训练需要提前创建JSON格式的HCCL配置文件。具体操作，参见：<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

#### GPU处理器环境运行RDN

- 单设备训练（1p)
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2
- 需要指定训练集路径(TRAIN_DATA_DIR)

```bash
bash scripts/run_gpu_standalone.sh [TRAIN_DATA_DIR]
```

- 分布式训练
- 二倍超分task_id 0
- 三倍超分task_id 1
- 四倍超分task_id 2
- 需要指定训练集路径(TRAIN_DATA_DIR)和芯片数量(DEVICE_NUM)

```bash
bash scripts/run_gpu_distribute.sh [TRAIN_DATA_DIR] [DEVICE_NUM]
```

## 评估过程

### 评估

- 评估过程如下，需要指定数据集类型(DATASET_TYPE)为“Set5”。
- 需要指定测试集路径(TEST_DATA_DIR)、ckpt文件路径(CHECKPOINT_PATH)和推理设备

```bash
bash scripts/eval.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DATASET_TYPE] [DEVICE_TARGET]
```

- 上述python命令在后台运行，可通过`eval.log`文件查看结果。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 模型导出

```bash
用法：python export.py  [--batch_size] [--ckpt_path] [--file_format]
选项：
  --batch_size      输入张量的批次大小。
  --ckpt_path       检查点路径。
  --file_format     可选 ['MINDIR', 'AIR', 'ONNX'], 默认['MINDIR']。
```

### 执行推理

```bash
在执行推理前，mindir文件必须通过export.py脚本导出。以下展示了使用minir模型执行推理的示例。
用法：bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATASET_TYPE] [SCALE] [DEVICE_ID]
选项：
  --MINDIR_PATH      mindir文件的路径。
  --DATA_PATH        数据集路径。  
  --DATASET_TYPE     数据集种类。
  --SCALE            超分辨率比例尺。
  --DEVICE_ID        芯片卡号。
```

### 查看结果

```bash
用法：vim run_infer.log
```

# 模型描述

## 性能

### 训练性能

| 参数           | RDN(Ascend)                |  RDN(GPU)                  |
| -------------------------- | --------------| ------------------- |
| 模型版本                | RDN               | RDN                  |
| 资源                   | Ascend 910；      | GeForce RTX 3090      |
| 上传日期              | 2021-09-15         | 2021-11-30            |
| MindSpore版本        | 1.2.0             |  1.5.0            |
| 数据集                |DIV2K             | DIV2K                  |
| 训练参数  |epoch=1800, batch_size = 16, lr=0.00005  | epoch=300,batch_size=16,lr=0.00005 |
| 优化器                  | Adam           | Adam                     |
| 损失函数 | L1loss |              L1Loss                         |
| 输出              | 超分辨率图片          |超分变率图像               |
| 速度 | 1卡：146毫秒/步 |          1卡:351毫秒/步 ;8卡:约445毫秒/步    |
| 总时长 | 1卡：60小时 |            1卡:约52小时 ;8卡:约4.8小时 |
| 调优检查点 |    0.25 GB（.ckpt 文件）    |  0.25 GB（.ckpt 文件）|

### 评估性能

| 参数  | RDN(Ascend)    | RDN（GPU）                     |
| ------| --------------|---------------------------- |
| 模型版本      | RDN     |RDN                  |
| 资源        | Ascend 910   |  GeForce RTX 3090             |
| 上传日期              | 2021-09-15 | 2021-11-30                   |
| MindSpore版本   | 1.2.0       |1.5.0                  |
| 数据集 | Set5 |Set5|
| batch_size          |   1     |1                   |
| 输出 | 超分辨率图片 |超分辨率图片 |
| 准确率 | 单卡：Set5: psnr:38.2302/ssim:0.9612 |8卡：Set5: psnr:38.0654/ssim:0.9610  |

### 推理性能

| 参数  | RDN(Ascend)                         |
| ------------------- | --------------------------- |
| 模型版本      | RDN                       |
| 资源        | Ascend 310                  |
| 上传日期              | 2021-10-15                    |
| Run包版本   | V100R001C78B100 Alpha                 |
| 数据集 | Set5 |
| batch_size          |   1                        |
| 输出 | 超分辨率图片 |
| 准确率 | 单卡：Set5: psnr:36.8578 |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
