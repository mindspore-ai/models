# 目录

<!-- TOC -->

- [目录](#目录)
    - [HAKE概述](#HAKE-概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
        - [评估过程](#评估过程)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

## HAKE 概述

HAKE模型是一种层次感知的知识图谱嵌入模型。该模型将实体映射到极坐标系中，极坐标系中的同心圆可以很自然地体现层次感。在该模型中，径向坐标旨在对层次结构不同级别上的实体进行建模，半径较小的实体应该位于较高的级别；角坐标旨在区分层次结构相同级别的实体，并且这些实体的半径大致相同，但角度不同。 实验表明，HAKE可以有效地对知识图中的语义层次进行建模，并且明显优于现有的基准数据集上用于链接预测任务的最新方法。

[论文](https://arxiv.org/pdf/1911.09419.pdf)：Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang. Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction[C]. AAAI 2020.

## 模型架构

HAKE模型总体网络架构见[论文](https://arxiv.org/pdf/1911.09419.pdf)

## 数据集

使用数据集：[WN18RR](https://github.com/MIRALab-USTC/KGE-HAKE)

- 数据集大小：3.78M，共40943个实体，11个关系
    - 训练集：2.98M，共86835个三元组
    - 验证集：108K，共3034个三元组
    - 测试集：112K，共3134个三元组
- 数据格式：三元组文本文件
    - 注：实体使用数字代号，具体实体名可以在entities.dict中查找。

## 环境要求

- 硬件：GPU环境
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

数据集准备完成后，请按照如下步骤开始训练和评估：

```bash
# 运行单卡训练示例
bash scripts/run_standalone_train_gpu.sh 3 data/wn18rr wn18rr/

# 运行分布式训练示例（2卡）
bash scripts/run_distribute_train_gpu.sh 2 data/wn18rr wn18rr/

# 运行评估示例
bash scripts/run_eval_gpu.sh 0 data/wn18rr wn18rr/CKP-120_339.ckpt
```

## 脚本说明

### 脚本和样例代码

```shell
.
└─HAKE
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train_gpu.sh        # GPU单卡训练
    ├─run_distribute_train_gpu.sh        # GPU多卡训练
    ├─run_eval_gpu.sh                    # GPU单个CKPT评估
    └─run_eval_gpu_all.sh                # GPU多个CKPT评估
  ├─src
    ├─model_utils
      ├── config.py                      # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py              # 区分本地/ModelArts训练
      ├── local_adapter.py               # 本地训练获取相关环境变量
      └── moxing_adapter.py              # ModelArts训练获取相关环境变量、交换数据
    ├─config.py                          # 参数配置
    ├─dataset.py                         # 数据预处理
    ├─HAKE_for_train.py                  # HAKE训练和loss
    ├─HAKE_model.py                      # HAKE模型
    └─utils.py                           # 工具类
  ├─default_config.yaml                  # 训练参数配置文件
  ├─eval.py                              # 评估脚本
  └─train.py                             # 训练脚本
```

### 训练脚本参数

```bash
# 单卡训练
usage: bash scripts/run_standalone_train_gpu.sh [--device_id DEVICE_ID] [--data_path DATA_PATH] [--save_path SAVA_PATH]

# 多卡训练
usage: bash scripts/run_distribute_train_gpu.sh [RANK_SIZE] [--data_path DATA_PATH] [--save_path SAVA_PATH]

options:
    --RANK_SIZE                    device num: N
    --max_epochs                   epoch size: N
    --device_id                    device id: N, default is 0
    --batch_size                   training batch size: N
    --negative_sample_size         negative sample size: N
    --hidden_dim                   embedding dimension: N
    --gamma                        global hyper-parameters: N
    --adversarial_temperature      global hyper-parameters, adversarial temperature: N
    --learning_rate                learning rate: N
    --modulus_weight               global hyper-parameters, modulus weight: N
    --phase_weight                 global hyper-parameters, phase weight: N
    --data_path                    path to dataset file: PATH, default is ""
    --save_path                    path to save checkpoint files: PATH, default is ""
```

### 训练过程

- 在`default_config.yaml`中设置选项，包括save_skpt_epoch_every, save_checkpoint_num等

- 运行`run_standalone_train_gpu.sh`，进行HAKE模型的非分布式训练。

    ``` bash
    bash scripts/run_standalone_train_gpu.sh DEVICE_ID DATA_PATH SAVE_PATH
    样例：
    bash scripts/run_standalone_train_gpu.sh 3 data/wn18rr wn18rr/
    ```

- 运行`run_distribute_train_gpu.sh`，进行HAKE模型的分布式训练。RANK_SIZE为卡的数量

    ``` bash
    bash scripts/run_distribute_train_gpu.sh RANK_SIZE DATA_PATH SAVE_PATH
    样例：
    bash scripts/run_distribute_train_gpu.sh 2 data/wn18rr wn18rr/
    ```

### 评估过程

- 运行`run_eval_gpu.sh`，评估HAKE模型。

    ```bash
    bash scripts/run_eval_gpu.sh DEVICE_ID DATA_PATH CKPT_PATH
    样例：
    bash scripts/run_eval_gpu.sh 3 data/wn18rr wn18rr/CKP-120_339.ckpt
    ```

### 结果

评估结果保存在./eval/eval.log文件中。您可以在日志中找到类似以下的结果。

```xml
'MRR': 0.4930160346867213, 'MR': 3769.847479259732, 'HITS@1': 0.44623484365028715, 'HITS@3': 0.5110082961072112, 'HITS@10': 0.5867900446713465
```

## 模型描述

### 性能

#### 训练性能

| 参数          | GPU                                                          |
| ------------- | ------------------------------------------------------------ |
| 资源          | Tesla V100-SXM2-32GB                                         |
| 上传日期      | 2021-11-18                                                   |
| MindSpore版本 | 1.6.0                                                        |
| 数据集        | WN18RR                                                       |
| 训练参数      | epoch=120, batch_size=256                                    |
| 优化器        | Adam                                                         |
| 损失函数      | Self-adversarial Negative Sampling Loss Function             |
| 性能          | 'MRR': 0.493, 'HITS@1': 0.4462, 'HITS@3': 0.511, 'HITS@10': 0.587 |
| 速度          | 152ms/step                                                   |
| 总时长        | 3h24min                                                      |

#### 评估性能

| 参数                 | GPU                          |
| -------------------- | ---------------------------- |
| 资源                 | Tesla V100-SXM2-32GB         |
| 上传日期             | 2021-11-18                   |
| MindSpore版本        | 1.6.0                        |
| 数据集               | WN18RR                       |
| batch_size           | 256                          |
| negative_sample_size | 1024                         |
| hidden_dim           | 500                          |
| 输出                 | MRR, HITS@1, HITS@3, HITS@10 |
| MRR                  | 0.493                        |
| HITS@1               | 0.446                        |
| HITS@3               | 0.511                        |
| HITS@10              | 0.587                        |

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
