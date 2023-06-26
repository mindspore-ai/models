# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
    - [OctSqueeze描述](#octsqueeze描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [训练](#训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
    - [推理过程](#推理过程)
        - [以固定的batch\_size训练网络（已完成的话，可跳过）：](#以固定的batch_size训练网络已完成的话可跳过)
        - [生成可输入网络的数据：](#生成可输入网络的数据)
        - [导出MindIR](#导出mindir)
        - [执行推理](#执行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
            - [KITTI 数据集中的点云选取000000.bin~001023.bin生成训练集](#kitti-数据集中的点云选取000000bin001023bin生成训练集)
        - [推理性能](#推理性能)
            - [KITTI 数据集中的点云选取007000.bin~007099.bin作为测试集，结果如下：](#kitti-数据集中的点云选取007000bin007099bin作为测试集结果如下)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## OctSqueeze描述

OctSqueeze是原Uber ATG于2020年提出的针对稀疏点云的压缩方法。通过在传统八叉树点云压缩的基础上引入基于网络的上下文预测，成功超越了包括MPEG G-PCC，Google Draco在内传统方法。

[论文](https://arxiv.org/abs/2005.07178)：Huang L, Wang S, Wong K, Liu J, Urtasun R. Octsqueeze: Octree-structured entropy model for lidar compression. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2020

## 模型架构

OctSqueeze网络由特征提取和特征融合两部分。由5层mlp组成的特征提取部分，负责提取各个节点的隐藏特征，由mlp组成的两两聚合的特征融合部分负责将当前节点与其祖先节点的特征融合，最后通过softmax预测出八叉划分后的概率分布。

## 数据集

使用的数据集：[KITTI](<http://www.cvlibs.net/datasets/kitti/>)

- 数据集大小：13.2G，共7481帧点云
    - 训练集和测试集可根据需要自由划分
- 数据格式：二进制文件
    - 注意：OctSqueeze作为一种hybrid的压缩算法，需要先将点云转化成八叉树的形式，再将树结构中节点的各类特征输入网络。因此需要先通过run_process_data.sh处理点云数据，得到网络可以直接处理的训练数据，再训练网络。
- 下载数据集（只需要用到velodyne文件夹下的.bin文件）。目录结构如下：

```text
├─ImageSets
|
└─object
  ├─training
    ├─calib
    ├─image_2
    ├─label_2
    └─velodyne
      ├─000000.bin
      ├─000001.bin
      ...
      └─007480.bin
```

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，生成训练数据集
bash run_process_data.sh [POIND_CLOUD_PATH] [OUTPUT_PATH] [MIN_ID] [MAX_ID] [MODE]
# example: bash run_process_data.sh ./KITTI/object/training/velodyne/ /home/ma-user/work/training_dataset/ 0 1000 train (即000000.bin ~ 001000.bin之间的1001帧会被用作生成训练集)

# 训练OctSqueeze (1P)
bash run_train_standalone.sh [TRAINING_DATASET_PATH] [DEVICE] [CHECKPOINT_SAVE_PATH] [batch_size]
# example: bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 0

# 或分布式训练OctSqueeze (8P)
bash run_train_distribute.sh [TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH] [RANK_TABLE_FILE]
# example: bash run_train_distribute.sh /home/ma-user/work/training_dataset/ ./ckpt/ /path/hccl_8p.json

# 评估OctSqueeze
bash run_eval.sh [TEST_DATASET_PATH] [COMPRESSED_DATA_PATH] [RECONSTRUCTED_DATA_PATH] [MODE] [DEVICE]
# example: bash run_eval.sh /home/ma-user/work/test_dataset/ ./com/ ./recon/ /home/ma-user/work/checkpoint/CKP-199_1023.ckpt Ascend
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

   ```bash
      # 在 ModelArts 上使用8卡训练
      # (1) 在网页上设置 "enable_modelarts=True"
      #     在网页上设置 "distribute=True"
      #     在网页上设置 "data_path='/cache/data'"
      #     在网页上设置 "ckpt_path='/cache/train'"
      #     在网页上设置 其他参数
      #
      # (2) 准备模型代码
      # (3) 上传经过run_process_data.sh脚本生成的训练数据集到 S3 桶上
      # (4) 在网页上设置你的代码路径为 "/path/octsqueeze"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      #
      # 在 ModelArts 上使用单卡训练
      # (1) 在网页上设置 "enable_modelarts=True"
      #     在网页上设置 "data_path='/cache/data'"
      #     在网页上设置 "ckpt_path='/cache/train'"
      #     在网页上设置 其他参数
      # (2) 准备模型代码
      # (3) 上传经过run_process_data.sh脚本生成的训练数据集到 S3 桶上
      # (4) 在网页上设置你的代码路径为 "/path/octsqueeze"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
   ```

## 脚本说明

### 脚本及样例代码

```python
├── cv
    ├── octsqueeze
        ├── README_CN.md                 # octsqueeze相关中文说明
        ├── README.md                    # octsqueeze相关英文说明
        ├── requirements.txt             # 所需要的包
        ├── scripts
        │   ├──run_eval.sh                    # 在Ascend上评估
        │   ├──run_proccess_data.sh           # 从KITTI的.bin文件中生成训练集
        │   ├──run_train_standalone.sh        # 在Ascend上进行单卡训练
        │   ├──run_train_distribute.sh        # 在Ascend上进行8卡训练
        ├── src
        │   ├──network.py                # octsqueeze网络架构
        │   ├──dataset.py                # 读取数据集
        |   └──tools
        |      ├──__init__.py
        |      ├──utils.py                    # 保存ply文件，计算熵，error等
        |      └──octree_base.py              # 八叉树模块
        ├── third_party
        │   └──arithmetic_coding_base.py  # 算术编码模块
        ├── ascend310_infer                  # 用于在Ascend310推理设备上进行离线推理的脚本(C++)
        ├── train.py                 # 在Ascend上进行训练的主程序
        ├── process_data.py          # 从KITTI的.bin文件中生成训练集的主程序
        ├── eval.py                  #  评估的主程序
```

### 脚本参数

```python
process_data.py中的主要参数如下：

--input_route：KITTI数据集中保存有点云（.bin）的文件夹的路径
--output_route：输出路径，生成的训练数据集的路径
--min_file：被用于生成训练集的点云数据的最小编号
--max_file：被用于生成训练集的点云数据的最小编号
--mode：模式选择可选值为"train"、"inference"，"inference"仅用于生成310推理所需的数据。生成训练集时选用"train"

train.py中的主要参数如下：

--batch_size：训练批次大小（batch_size设置为0表示使用动态batch_size，每次将一帧点云中的所有节点输入网络。推荐在训练时将batch_size设定为0。310推理需要使用固定batch_size进行训练，此时需设定大于0的batch_size。）
--train：训练数据集的路径（推荐在8卡训练时使用绝对路径），训练数据集由process_data.py和KITTI数据集生成
--max_epochs：总训练轮次
--device_target：实现代码的设备。当前仅支持 "Ascend"。
--checkpoint：训练后保存的检查点文件的路径（推荐在8卡训练时使用相对路径）

eval.py中的主要参数如下：

--test_dataset：测试集路径，测试数据是从KITTI数据集中任意挑选的点云帧（.bin）
--compression：压缩后的文件存储的路径
--recon：解压后的文件存储的路径
--model：需要读入的检查点文件的路径
--device_target：实现代码的设备。可选值为"Ascend"、"GPU"、"CPU"
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --train=[TRAINING_DATASET_PATH] --device_target=[DEVICE] --checkpoint=[CHECKPOINT_SAVE_PATH] --batch_size=[batch_size] --is_distributed=0
  # 或进入脚本目录，执行1P脚本
  bash bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 0
  # 或进入脚本目录，执行8P脚本
  bash run_train_distribute.sh /home/ma-user/work/training_dataset/ ./ckpt/ /path/hccl_8p.json
  ```

  经过训练后，损失值如下：

  ```bash
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 10 step: 1023, loss is 2.7296906
  epoch: 11 step: 1023, loss is 2.7205226
  epoch: 12 step: 1023, loss is 2.7087197
  ...
  ```

  模型检查点保存在指定位置。

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行

  ```bash
  python eval.py --test_dataset=[TEST_DATASET_PATH] --compression=[COMPRESSED_DATA_PATH] --recon=[RECONSTRUCTED_DATA_PATH] --model=[MODE] --device_target=[DEVICE]
  # 或进入脚本目录，执行脚本
  bash run_eval.sh /home/ma-user/work/test_dataset/ ./com/ ./recon/ /home/ma-user/work/checkpoint/CKP-199_1023.ckpt Ascend
  ```

  KITTI数据集007000.bin~007099.bin的评估结果如下：

  ```python
  bpip and chamfer distance at different bitrates:
  [[7.04049839 0.0191274 ]
   [4.47647693 0.03779216]
   [2.51015919 0.07440553]
   [1.2202392  0.14568059]]
  ```

## 推理过程

### 以固定的batch_size训练网络（已完成的话，可跳过）：

```shell
python train.py --train=[TRAINING_DATASET_PATH] --device_target=[DEVICE] --checkpoint=[CHECKPOINT_SAVE_PATH] --batch_size=[batch_size] --is_distributed=0
# 或进入脚本目录，执行1P脚本
bash bash run_train_standalone.sh /home/ma-user/work/training_dataset/ Ascend ./ckpt/ 98304
```

### 生成可输入网络的数据：

```shell
bash bash run_process_data.sh [POIND_CLOUD_PATH] [OUTPUT_PATH] [MIN_ID] [MAX_ID] [MODE]
# 或进入脚本目录，执行run_process_data.sh脚本
bash run_process_data.sh ./KITTI/object/training/velodyne/ /home/ma-user/work/infernece_dataset/ 7000 7099 inference
```

### 导出MindIR

```shell
python export.py ----ckpt_file=[CKPT_PATH] --batch_size=[BATCH_SIZE] --file_name=[FILE_NAME]
# Example:
python export.py --ckpt_file='/home/ma-user/work/AM_OctSqueeze/checkpoint/CKP-196_1024.ckpt' --batch_size=98304 --file_name=octsqueeze
```

参数ckpt_file为必填项

### 执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```bash
bash run_cpp_infer.sh [MINDIR_PATH] [DATA_PATH] [BATCH_SIZE] [DEVICE_TYPE] [DEVICE_ID]
```

- `MODEL_PATH` mindir文件路径
- `DATASETS_DIR` 推理数据集路径（该路径下会有复数个子文件夹像是./0.01 ./0.02等对应不同精度）
- `BATCH_SIZE` 导出MindIR设定的batch_size

### 结果

推理结果保存在./logs下，你可以在test_performance.txt中看到以下精度计算结果。

```bash
At precision 0.01: bpip =  7.03702; each frame cost 149717 ms
At precision 0.02: bpip =  4.46777; each frame cost 101945 ms
At precision 0.04: bpip =  2.50509; each frame cost 56277 ms
At precision 0.08: bpip =  1.22112; each frame cost 53100 ms
```

## 模型描述

### 性能

#### 评估性能

#### KITTI 数据集中的点云选取000000.bin~001023.bin生成训练集

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | OctSqueeze                                                   |
| 资源          | Ascend 910；CPU 10核；内存 120G；                            |
| 上传日期      | TBD                                                          |
| MindSpore版本 | 1.3.0                                                        |
| 数据集        | 7481帧点云                                                   |
| 训练参数      | epoch=100, batch_size=0（动态batch_size）, lr=0.001          |
| 优化器        | Adam                                                         |
| 损失函数      | SoftmaxCrossEntropy                                          |
| 输出          | 概率分布                                                     |
| 损失          | 1.0                                                          |
| 速度          | 单卡：58毫秒/步;  8卡：80毫秒/步                             |
| 总时长        | 单卡：3.3h；8卡：0.6h                                        |
| 参数(M)       | 0.34M                                                        |
| 微调检查点    | 3M (.ckpt文件)                                               |
| 脚本          | [octsqueeze脚本](https://gitee.com/mindspore/models/tree/master/research/cv/OctSqueeze) |

### 推理性能

#### KITTI 数据集中的点云选取007000.bin~007099.bin作为测试集，结果如下：

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | OctSqueeze                                                   |
| 资源          | Ascend 910                                                   |
| 上传日期      | TBD                                                          |
| MindSpore版本 | 1.3.0                                                        |
| 数据集        | 7481帧点云                                                   |
| batch_size    | 1                                                            |
| 输出          | 在4个不同的bitrate点下，平均的bpip和对应的chamfer distance   |
| 准确性        | [[7.04049839 0.0191274 ]<br/> [4.47647693 0.03779216]<br/> [2.51015919 0.07440553]<br/> [1.2202392  0.14568059]] |

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
