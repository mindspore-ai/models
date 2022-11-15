## 目录

<!-- TOC -->

- [目录](#目录)
- [DLRM 概述](#dlrm-概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
        - [训练结果](#训练结果)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [评估结果](#评估结果)
    - [推理过程](#推理过程)
        - [导出 MindIR](#导出-mindir)
        - [在 Ascend310 执行推理](#在-ascend310-执行推理)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## DLRM 概述

点击率预估模型中，输入特征通常包含大量稀疏类别特征以及一些数值特征，DLRM 使用嵌入技术处理类别特征，使用 MLP 处理数值特征，然后通过显式的点积特征交互层进行特征交互，最后通过另一个顶部的 MLP 产生 CTR 预测结果。通常推荐系统中的数据量巨大，且类别特征众多，导致推荐模型的参数量巨大，嵌入表占主要部分。DLRM 通过新颖的混合并行机制提升模型的效率，将不同特征域的嵌入表和底部 MLP 划分到各个并行的 GPU 中，进行模型并行；而顶部的交互层和 MLP 则进行数据并行。

论文：Naumov M, Mudigere D, Shi H J M, et al. Deep learning recommendation model for personalization and recommendation systems[J]. arXiv preprint arXiv:1906.00091, 2019.

## 模型架构

DLRM 可以分为底部和顶部两部分。底部包含一个用于处理数值特征的 MLP 和一个嵌入层，然后通过交互层产生特征交互。顶部是一个 MLP ，最后通过 sigmoid 激活函数产生预测结果。

由于能力和时间有限，这里仅实现了单卡和数据并行模式，后续有时间会再进行混合并行的探索。

## 数据集

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

## 环境要求

- 硬件（Ascend 或 GPU）
    - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install) (1.3 / 1.5 / 1.6)
- 第三方库

  ```bash
  pip install sklearn
  pip install pandas
  pip install pyyaml
  ```

## 快速入门

- 数据集预处理

  ```bash
  #下载数据集
  mkdir -p data/origin_data && cd data/origin_data
  wget DATA_LINK
  tar -zxvf criteo.tar.gz

  #数据集预处理脚步执行 (耗时约 3.5H)
  python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --train_line_count=45840617
  ```

- Ascend 处理器环境运行

  ```bash
  # 运行训练示例
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=Ascend \
    --do_eval=True > ms_log/output.log 2>&1 &
  OR
  bash scripts/run_standalone_train_ascend.sh DEVICE_ID/CUDA_VISIBLE_DEVICES DEVICE_TARGET DATASET_PATH

  # 运行分布式训练示例
  bash scripts/run_distribute_train_ascend.sh 8 /dataset_path /rank_table_8p.json

  # 运行评估示例
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/dlrm.ckpt' \
    --device_target=Ascend > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/dlrm.ckpt
  ```

  在分布式训练中，JSON 格式的 HCCL 配置文件需要提前创建。

  具体操作，参见：

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

## 脚本说明

### 脚本和样例代码

```dlrm
.
└─dlrm
  ├─ascend310_info                    # 310推理代码
  ├─README.md
  ├─mindspore_hub_conf.md             # mindspore hub 配置
  ├─scripts
    ├─run_standalone_train_ascend.sh         # 在 Ascend
    ├─run_distribute_train_ascend.sh         # 在 Ascend
    ├─run_standalone_train_gpu.sh         # 在 Ascend
    ├─run_eval.sh                     # 在 Ascend 处理器或 GPU 上进行评估
    ├─run_eval_gpu.sh
    └─run_infer_310.sh                # 在 Ascend 310 上推理

  ├─src
    ├─model_utils
      ├─__init__.py
      ├─config.py                     # 读取配置
      ├─device_target.py
      ├─local_adapter.py
      └─moxing_adapter.py
    ├─__init__.py
    ├─callback.py                     # 定义回调功能
    ├─dlrm.py                         # DLRM 网络
    ├─dataset.py                      # 创建数据集
    └─preprocess_data.py              # 数据预处理
  ├─npu_config.yaml                   # 默认配置文件
  ├─gpu_config.yaml                   # 默认配置文件
  ├─eval.py                           # 评估网络
  ├─export.py                         # 导出 MindIR 模型
  ├─preprocess.py                     # 预处理用于推理的数据
  ├─postprocess.py                    # 产生推理结果
  └─train.py                          # 训练网络
```

### 脚本参数

在 config.py 中可以同时配置训练参数和评估参数。

- 训练参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Dataset path
  --ckpt_path CKPT_PATH
                        Checkpoint path
  --eval_file_name EVAL_FILE_NAME
                        Auc log file path. Default: "./auc.log"
  --loss_file_name LOSS_FILE_NAME
                        Loss log file path. Default: "./loss.log"
  --do_eval DO_EVAL     Do evaluation or not. Default: True
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

- 评估参数。

  ```参数
  optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path
  --dataset_path DATASET_PATH
                        Dataset path
  --device_target DEVICE_TARGET
                        Ascend or GPU. Default: Ascend
  ```

### 训练过程

#### 训练

- Ascend 处理器上运行

  ```bash
  python train.py \
    --dataset_path='dataset/train' \
    --ckpt_path='./checkpoint' \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=Ascend \
    --do_eval=True > ms_log/output.log 2>&1 &
  ```

  上述 python 命令将在后台运行，您可以通过 `ms_log/output.log` 文件查看结果。

  训练结束后, 您可在默认文件夹`./checkpoint`中找到检查点文件。损失值保存在 loss.log 文件中。

- GPU

  ```bash
  bash scripts/run_standalone_train_gpu.sh DEVICE_ID DEVICE_TARGET DATASET_PATH
  ```

#### 分布式训练

- Ascend 处理器上运行

  ```bash
  bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json
  ```

  上述 shell 脚本将在后台运行分布式训练。请在 `log[X]/output.log` 文件中查看结果。损失值保存在 `loss.log` 文件中。

#### 训练结果

训练结果将保存在示例路径，如以上所述的检查点和输出日志，训练中的损失值保存在 `loss.log` 文件中。

```result
2021-11-08 21:35:30 epoch: 1, step: 76742, loss is 0.3277069926261902
...
```

### 评估过程

#### 评估

- Ascend 处理器上运行评估

  在运行以下命令之前，请检查用于评估的数据集和检查点路径。

  ```bash
  python eval.py \
    --dataset_path='dataset/test' \
    --checkpoint_path='./checkpoint/dlrm.ckpt' \
    --device_target=Ascend > ms_log/eval_output.log 2>&1 &
  OR
  bash scripts/run_eval.sh 0 Ascend /dataset_path /checkpoint_path/dlrm.ckpt
  ```

  上述 python 命令将在后台运行，请在 eval_output.log 路径下查看结果。准确率保存在 acc.log 文件中。

- GPU

    ```bash
    bash scripts/run_eval.sh DEVICE_ID DEVICE_TARGET DATASET_PATH CHECKPOINT_PATH
    ```

#### 评估结果

评估保存在 `acc.log` 文件中。

```result
2021-11-08 21:51:14 EvalCallBack metric {'acc': 0.787175917087641}; eval_time 894s
```

### 推理过程

#### 导出 MindIR

请在 npu_config.yaml 中修改 checkpoint_path, file_name, file_format 三个参数，然后执行

```shell
python export.py --config_path [/path/to/npu_config.yaml]
```

`file_format` 必须在 ["AIR", "MINDIR"] 中选择

#### 在 Ascend310 执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

由于模型过大无法加载，推理尚未完成。

在执行推理前，mindir 文件必须通过 `export.py` 脚本导出。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为 0。

## 模型说明

### 性能

#### 训练性能

| 参数                    | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- |----|
| 模型版本              | DLRM                                                  | DLRM   |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | CPU: Intel(R) Xeon(R) Gold 6226R RAM: 252G GPU: RTX3090 24G |
| 上传日期              | 2021-11-09                           | 2022-02-23 |
| MindSpore版本          | 1.3.0/1.5.0                                           | 1.6.0 |
| 数据集                    | Criteo                                           | Criteo |
| 训练参数        | epoch=1,  batch_size=128, lr=0.15                        |epoch=1,  batch_size=1280, lr=0.15  |
| 优化器                  | SGD                                                      |SGD   |
| 损失函数              | Sigmoid Cross Entropy With Logits                           |Sigmoid Cross Entropy With Logits                           |
| 输出                    | 准确率                                                    |准确率 |
| 损失                       | 0.3277069926261902                                                    | 0.452256|
| 速度| 单卡：144 毫秒/步;                                      | 单卡：38 毫秒/步;|
| 总时长| 单卡：9 小时;                                               | 单卡：20 min;|
| 参数(M)             | 540                                                        | 540 |
| 微调检查点 | 6.1G (.ckpt 文件)                                     | 6.1G (.ckpt 文件)  
| 脚本                    |  |

#### 推理性能

| 参数          | Ascend                      | GPU |
| ------------------- | --------------------------- | --- |
| 模型版本       | DLRM                | DLRM   |
| 资源            | Ascend 910；系统 Euler2.8                  | CPU: Intel(R) Xeon(R) Gold 6226R RAM: 252G GPU: RTX3090 24G |
| 上传日期       | 2021-11-09 | 2022-02-23 |
| MindSpore版本   | 1.3.0/1.5.0      |  1.6.0 |
| 数据集           | Criteo                    | Criteo |
| batch_size          | 16384                        | 16384  |
| 输出             | 准确率                    | 准确率 |
| 总耗时           | 1H50min                  | 155 s|
| 准确率| 0.787175917087641                | 0.7876784245770677 |
| 推理模型 | 6.1G (.ckpt文件)           | 6.1G (.ckpt文件) |

## 随机情况说明

在 `train.py` 设置 `MindSpore` 的随机种子。

## ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
