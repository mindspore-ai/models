目录

- [目录](#目录)
- [Arcface概述](#Arcface概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
            - [分布式训练](#分布式训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
        - [导出mindir模型](#导出mindir模型)
        - [推理过程](#推理过程)
            - [用法](#用法)
            - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# Arcface概述

使用深度卷积神经网络进行大规模人脸识别的特征学习中的主要挑战之一是设计适当的损失函数以增强判别能力。继SoftmaxLoss、Center Loss、A-Softmax Loss、Cosine Margin Loss之后，Arcface在人脸识别中具有更加良好的表现。Arcface是传统softmax的改进， 将类之间的距离映射到超球面的间距，论文给出了对此的清晰几何解释。同时，基于10多个人脸识别基准的实验评估，证明了Arcface优于现有的技术，并且可以轻松实现。

[论文](https://arxiv.org/pdf/1801.07698v3.pdf)： Deng J ,  Guo J ,  Zafeiriou S . ArcFace: Additive Angular Margin Loss for Deep Face Recognition[J].  2018.

# 数据集

使用的训练数据集：[MS1MV2](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

验证数据集：lfw，cfp-fp，agedb，cplfw，calfw，[IJB-B，IJB-C](https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg)

训练集：5,822,653张图片，85742个类

```python
#将rec数据格式转换成jpg
python src/rec2jpg_dataset.py --include rec/dataset/path --output output/path
```

注意：在arm环境中，需源码编译[mxnet](https://mxnet.apache.org/versions/1.9.0/get_started/build_from_source.html)才能正常运行。

# 环境要求

- 硬件：昇腾处理器（Ascend）
    - 使用Ascend处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 分布式训练运行示例
  bash scripts/run_distribute_train.sh rank_size /path/dataset

  # 单机训练运行示例
  bash scripts/run_standalone_train.sh /path/dataset device_id

  # 运行评估示例
  bash scripts/run_eval.sh /path/evalset /path/ckpt
  ```

- GPU处理器环境运行

  ```python
  # 分布式训练运行示例
  bash scripts/run_distribute_train_gpu.sh  /path/dataset rank_size

  # 单机训练运行示例
  bash scripts/run_standalone_train_gpu.sh /path/dataset

  # 运行评估示例
  bash scripts/run_eval_gpu.sh /path/evalset /path/ckpt
  ```

## 脚本说明

## 脚本和样例代码

```path
└── Arcface  
 ├── README.md                           // Arcface相关描述
 ├── ascend310_infer                     //310推理
  ├── inc
   ├── utils.h
  ├── src
   ├── main.cc
   ├── utils.cc
  ├── build.sh
  └── CMakeLists.txt
 ├── scripts
  ├── run_310_infer.sh           // 用于310推理的shell脚本
  ├── run_distribute_train.sh    // 用于分布式训练的shell脚本
  ├── run_standalone_train.sh    // 用于单机训练的shell脚本
  ├── run_eval_ijbc.sh           // 用于IJBC数据集评估的shell脚本
  └── run_eval.sh                // 用于评估的shell脚本
 ├──src
  ├── loss.py                         //损失函数
  ├── dataset.py                      // 创建数据集
  ├── iresnet.py                      // ResNet架构
  ├── rec2jpg_dataset.py                  // 将rec数据格式转换成jpg
 ├── val.py                            // 测试脚本
 ├── train.py                          // 训练脚本
 ├── export.py
 ├── requirements.txt
 ├── preprocess.py                    // 310推理数据预处理
 ├── preprocess.py                    // 310推理数据后处理

```

## 脚本参数

```python
train.py和val.py中主要参数如下：

-- modelarts：是否使用modelarts平台训练。可选值为True、False。默认为False。
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- device_num：使用tra进行分布式训练时使用的设备数。
-- train_url：checkpoint的输出路径。
-- data_url：训练集路径。
-- ckpt_url：checkpoint路径。
-- eval_url：验证集路径。

```

## 训练过程

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distribute_train.sh rank_size /path/dataset
  ```

  上述shell脚本将在后台运行分布训练。可以通过`device[X]/train.log`文件查看结果。
  采用以下方式达到损失值：

  ```log
  epoch: 2 step: 11372, loss is 12.807039
  epoch time: 1104549.619 ms, per step time: 97.129 ms
  epoch: 3 step: 11372, loss is 9.13787
  ...
  epoch: 21 step: 11372, loss is 1.5028578
  epoch time: 1104673.362 ms, per step time: 97.140 ms
  epoch: 22 step: 11372, loss is 0.8846929
  epoch time: 1104929.793 ms, per step time: 97.162 ms
  ```

- GPU处理器环境运行

  ```bash
  bash scripts/run_distribute_train_gpu.sh /path/dataset rank_size
  ```

  上述shell脚本将在后台运行分布训练。可以通过`train_parallel/train.log`文件查看结果。
  采用以下方式达到损失值：

  ```log
  epoch: 2 step: 11372, loss is 10.572094
  epoch time: 1104549.619 ms, per step time: 991.390 ms
  epoch: 3 step: 11372, loss is 7.442794
  ...
  epoch: 21 step: 11372, loss is 0.8472798
  epoch time: 1104673.362 ms, per step time: 989.479 ms
  epoch: 22 step: 11372, loss is 0.5226351
  epoch time: 1104929.793 ms, per step time: 989.548 ms
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估lfw、cfp_fp、agedb_30、calfw、cplfw数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/arcface/arcface-11372-1.ckpt”。

  ```bash
  bash scripts/run_eval.sh /path/evalset /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  [lfw]Accuracy-Flip: 0.99817+-0.00273
  [cfp_fp]Accuracy-Flip: 0.98000+-0.00586
  [agedb_30]Accuracy-Flip: 0.98100+-0.00642
  [calfw]Accuracy-Flip: 0.96150+-0.01099
  [cplfw]Accuracy-Flip: 0.92583+-0.01367
  ```

- 在Ascend环境运行时评估IJB-B、IJB-C数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/arcface/arcface-11372-1.ckpt”。

  同时，情确保传入的评估数据集路径为“IJB_release/IJBB/”或“IJB_release/IJBC/”。

  ```bash
  bash scripts/run_eval_ijbc.sh /path/evalset /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbb-IJBB | 40.01 | 87.91 | 94.36  | 96.48 | 97.72 | 98.70 |
  +-----------+-------+-------+--------+-------+-------+-------+

  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbc-IJBC | 82.08 | 93.37 | 95.87  | 97.40 | 98.40 | 99.05 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

- 在GPU环境运行时评估lfw、cfp_fp、agedb_30、calfw、cplfw数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/arcface/arcface-11372-1.ckpt”。

  ```bash
  bash scripts/run_eval_gpu.sh /path/evalset /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  [lfw]Accuracy-Flip: 0.99767+-0.00271
  [cfp_fp]Accuracy-Flip: 0.98414+-0.00659
  [agedb_30]Accuracy-Flip: 0.98033+-0.00878
  [calfw]Accuracy-Flip: 0.95983+-0.01141
  [cplfw]Accuracy-Flip: 0.92817+-0.01279
  ```

- 在GPU环境运行时评估IJB-B、IJB-C数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/arcface/arcface-11372-1.ckpt”。

  同时，情确保传入的评估数据集路径为“IJB_release/IJBB/”或“IJB_release/IJBC/”。

  ```bash
  bash scripts/run_eval_ijbc_gpu.sh /path/evalset /path/ckpt
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbb-IJBB | 42.46 | 89.76 | 94.81  | 96.58 | 97.73 | 98.78 |
  +-----------+-------+-------+--------+-------+-------+-------+

  +-----------+-------+-------+--------+-------+-------+-------+
  |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
  +-----------+-------+-------+--------+-------+-------+-------+
  | ijbc-IJBC | 86.67 | 94.35 | 96.19  | 97.55 | 98.38 | 99.10 |
  +-----------+-------+-------+--------+-------+-------+-------+
  ```

## 导出mindir模型

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

# 推理过程

## 用法

在执行推理之前，需要通过export.py导出mindir文件。

```bash
# Ascend310 推理
bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

`DEVICE_TARGET` 可选值范围为：['GPU', 'CPU', 'Ascend']，`NEED_PREPROCESS` 表示数据是否需要预处理，可选值范围为：'y' 或者 'n'，这里直接选择‘y’，`DEVICE_ID` 可选, 默认值为0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

# 模型描述

## 性能

### 训练性能

| 参数          | Arcface                                                      | GPU
| ------------- | ------------------------------------------------------------ | ------------------------------------------- |
| 模型版本      | arcface                                                        | arcface                                   |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G                   | GeForce RTX 3090； CPU： 2.90GHz，64内核；内存，755G                                           |
| 上传日期      | 2021-05-30                                                     | 2021-11-12                               |
| MindSpore版本 | 1.2.0-c77-python3.7-aarch64                                   | 1.5.0                                     |
| 数据集        | MS1MV2                                                        | MS1MV2                                       |
| 训练参数      | lr=0.08; gamma=0.1                                             | lr=0.08; gamma=0.1                           |
| 优化器        | SGD                                                           | SGD                                          |
| 损失函数      | Arcface                                                      | Arcface                                        |
| 输出          | 概率                                                         | 概率                                             |
| 损失          | 0.6                                                          | 0.7                                          |
| 速度          | 1卡：108毫秒/步；8卡：97毫秒/步                              | 8卡：990毫秒/步                                      |
| 总时间        | 1卡：65小时；8卡：8.5小时                                    | 8卡：75小时                                        |
| 参数(M)       | 85.2                                                         | 85.2                                         |
| 微调检查点    | 1249M （.ckpt file）                                         | 1249M （.ckpt file）                            |
| 脚本          | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/cv/arcface) | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/cv/arcface) |

### 评估性能

| 参数          | Ascend                  | GPU                         |
| ------------- | ------------------------ | -------------------------  |
| 模型版本      | arcface                  | arcface                   |
| 资源          | Ascend 910               | GeForce RTX 3090                 |
| 上传日期      | 2021/05/30               | 2021/11/12            |
| MindSpore版本 | 1.2.0-c77-python3.7-aarch64            |  1.5.0      |
| 数据集        | IJBC、IJBB、lfw、cfp_fp、agedb_30、calfw、cplfw | IJBC、IJBB、lfw、cfp_fp、agedb_30、calfw、cplfw |
| 输出          | 概率                     | 概率                     |
| 准确性        | lfw:0.998   cfp_fp:0.98   agedb_30:0.981   calfw:0.961   cplfw:0.926   IJB-B:0.943   IJB-C:0.958 | lfw:0.998   cfp_fp:0.984   agedb_30:0.9803   calfw:0.9598   cplfw:0.928   IJB-B:0.943   IJB-C:0.958 |

# 随机情况说明

网络的初始参数均为随即初始化。

# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。
