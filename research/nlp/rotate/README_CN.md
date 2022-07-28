# 目录

<!-- TOC -->

- [目录](#目录)
- [RotatE概述](#rotate概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单卡训练](#单卡训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
    - [导出过程](#导出过程)
    - [推理过程](#推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估描述](#评估描述)
        - [推理描述](#推理描述)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [RotatE概述](#目录)

RotatE是一个用于链接预测任务的知识图谱嵌入模型。

论文：Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, Jian Tang: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

# [模型架构](#目录)

RotatE模型将每个关系定义为在复矢量空间中从源实体到目标实体的旋转。RotatE模型能够建模和推断各种关系模式，包括：对称/反对称，反演和合成。此外，RotatE模型还提出了一种新颖的自我对抗式负采样技术，可以有效地训练RotatE模型。

# [数据集](#目录)

使用数据集：[WN18RR](<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>)

- 数据集大小：3.78M，共40943个实体，11个关系
    - 训练集：2.98M，共86835个三元组
    - 验证集：108K，共3034个三元组
    - 测试集：112K，共3134个三元组
- 数据格式：三元组文本文件
    - 注：实体使用数字代号，具体实体名可以在entities.dict中查找。

# [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用Ascend/GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练、评估、推理和解释：

- 数据准备

将数据集[WN18RR](<https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>)下载后置于Data/wn18rr文件夹中即可。

- Ascend处理器环境运行

```bash
# 运行单卡训练示例
bash scripts/run_standalone_train.sh [DEVICE_ID] [DEVICE_TARGET] [OUTPUT_PATH] [MAX_STEPS] [LOG_FILE]
# example: bash scripts/run_standalone_train.sh 0 Ascend ./checkpoints/rotate-standalone-ascend/ 80000 output-standalone-ascend.log

# 运行分布式训练示例
bash scripts/run_distribute_train_ascend.sh [DEVICE_NUM] [BATCH_SIZE] [MAX_STEPS] [RANK_TABLE_FILE]
# example: bash scripts/run_distribute_train_ascend.sh 8 64 640000 ./rank_table_8p.json

# 运行评估示例
bash scripts/run_eval.sh [DEVICE_ID] [DEVICE_TARGET] [EVAL_CHECKPOINT] [EVAL_LOG_FILE]
# example: bash scripts/run_eval.sh 0 Ascend ./checkpoints/rotate-standalone-ascend/rotate.ckpt eval-standalone-ascend.log

# 运行推理示例
bash run_infer_310.sh [MINDIR_HEAD_PATH] [MINDIR_TAIL_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
# example: bash run_infer_310.sh ../rotate-head.mindir ../rotate-tail.mindir ../Data/wn18rr/ y 0
```

在裸机环境（本地有Ascend 910 AI 处理器）进行分布式训练时，需要配置当前多卡环境的组网信息文件。
请遵循一下链接中的说明创建json文件：
<https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html#配置分布式环境变量>

- GPU处理器环境运行

```bash
# 运行单卡训练示例
bash scripts/run_standalone_train.sh [DEVICE_ID] [DEVICE_TARGET] [OUTPUT_PATH] [MAX_STEPS] [LOG_FILE]
# example: bash scripts/run_standalone_train.sh 0 GPU ./checkpoints/rotate-standalone-gpu/ 70000 output-standalone-gpu.log

# 运行分布式训练示例
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [BATCH_SIZE] [MAX_STEPS]
# example: bash scripts/run_distribute_train_gpu.sh 8 64 560000

# 运行评估示例
bash scripts/run_eval.sh [DEVICE_ID] [DEVICE_TARGET] [EVAL_CHECKPOINT] [EVAL_LOG_FILE]
# example: bash scripts/run_eval.sh 0 GPU ./checkpoints/rotate-standalone-gpu/rotate.ckpt eval-standalone-gpu.log
```

# [脚本说明](#目录)

## [脚本和样例代码](#目录)

```bash
.
└─rotate
  ├─README.md                              # 模型所有相关说明
  ├─ascend310_infer                        # 实现310推理源代码
  ├─scripts
        ├─run_distribute_train_gpu.sh      # GPU分布式训练shell脚本
        ├─run_distribute_train_ascend.sh   # Ascend分布式训练shell脚本
        ├─run_eval.sh                      # 评估测试shell脚本
        ├─run_infer_310.sh                 # Ascend推理shell脚本
        └─run_standalone_train.sh          # 单卡训练shell脚本
  ├─src
    ├─dataset.py                    # 创建数据集
    └─rotate.py                     # RotatE网络
  ├─eval.py                         # 评估脚本
  ├─export.py                       # 将checkpoint文件导出到mindir
  ├─requirements.txt                # 模型依赖包列表
  ├─postprocess.py                  # 310推理后处理脚本
  ├─preprocess.py                   # 310推理前处理脚本
  └─train.py                        # 训练脚本
```

## [脚本参数](#目录)

在config.py中可以同时配置训练参数和评估参数。

```python
'data_path': "./Data/wn18rr/"            # 数据集存放路径
'output_path': "./checkpoints/"          # checkpoint输出路径
'device_target': 'Ascend'                # 运行设备
'experiment_name': "rotate"              # 实验名称，存储的ckpt将以此命名
'data_url': ""                           # modelarts数据集路径配置
'train_url': ""                          # modelarts训练脚本路径配置
'checkpoint_url': ""                     # modelarts输出路径配置
'eval_checkpoint': ""                    # 测试评估的ckpt路径
'lr': 0.00005                            # 学习率
'gamma': 6.0                             # 正负样本间隔超参数
'max_steps': 80000                       # 训练步数
'batch_size': 512                        # 训练批次大小
'test_batch_size': 8                     # 测试批次大小
'hidden_dim': 500                        # 嵌入维度
'negative_sample_size': 1024             # 负采样个数
'adversarial_temperature': 0.5           # 自对抗温度超参数
'double_entity_embedding': True          # 是否对实体维度加倍
'double_relation_embedding': False       # 是否对关系维度加倍
'use_dynamic_loss_scale': True           # 是否使用动态loss scale
'file_format': "MINDIR"                  # export导出文件格式
```

更多配置细节请参考脚本`config.py`。

## [训练过程](#目录)

### [单卡训练](#目录)

- Ascend处理器环境运行

  ```bash
  bash scripts/run_standalone_train.sh 0 Ascend ./checkpoints/rotate-standalone-ascend/ 80000 output-standalone-ascend.log
  ```

  上述命令将在后台运行，您可以通过ms_log/output-standalone-ascend.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  step374 cost time: 141.10ms loss=0.951583
  step375 cost time: 140.84ms loss=0.946728
  ...
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  ``` bash
  bash scripts/run_standalone_train.sh 0 GPU ./checkpoints/rotate-standalone-gpu/ 70000 output-standalone-gpu.log
  ```

  上述命令将在后台运行，您可以通过ms_log/output-standalone-gpu.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。

### [分布式训练](#目录)

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distribute_train_ascend.sh 8 64 640000 ./rank_table_8p.json
  ```

  上述shell脚本将在后台运行分布训练。您可以通过ms_log/output-distribute-ascend.log文件查看结果。采用以下方式达到损失值：

  ```bash
  step230 cost time: 24.41ms loss=1.524412
  step230 cost time: 24.39ms loss=1.470812
  step230 cost time: 24.41ms loss=1.706939
  step230 cost time: 24.48ms loss=1.649485
  step230 cost time: 24.44ms loss=1.524977
  step230 cost time: 24.44ms loss=1.556086
  step230 cost time: 24.42ms loss=1.654312
  step230 cost time: 24.38ms loss=1.535843
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash scripts/run_distribute_train_gpu.sh 8 64 560000
  ```

  上述shell脚本将在后台运行分布训练。您可以通过ms_log/output-distribute-gpu.log文件查看结果。

## [评估过程](#目录)

- Ascend环境评估WN18RR数据集

  在运行以下命令之前，请检查用于评估的检查点路径。

  ```bash
  bash scripts/run_eval.sh 0 Ascend checkpoints/rotate-standalone-ascend/rotate.ckpt eval-standalone-ascend.log
  ```

  上述python命令将在后台运行，您可以通过ms_log/eval-standalone-ascend文件查看类似如下的结果：

  ```bash
  {'MRR': 0.475194889652354, 'MR': 3239.5397255903, 'HITS@1': 0.4261327377153797, 'HITS@3': 0.4952137843012125, 'HITS@10': 0.5754626675175495}
  ```

  注：对于分布式训练后评估，可以将结果输出到不同的log文件中以示区分。测试数据集的准确性如下：

  ```bash
  {'MRR': 0.4756099890731606, 'MR': 3240.682992980217, 'HITS@1': 0.4262922782386726, 'HITS@3': 0.4948947032546267, 'HITS@10': 0.5749840459476707}
  ```

- GPU处理器环境评估WN18RR数据集

  在运行以下命令之前，请检查用于评估的检查点路径。

  ```bash
  bash scripts/run_eval.sh 0 GPU ./checkpoints/rotate-standalone-gpu/rotate.ckpt eval-standalone-gpu.log
  ```

  上述命令将在后台运行，您可以通过ms_log/eval-standalone-gpu.log文件查看结果。测试数据集的准确性如下：

  ```bash
  {'MRR': 0.4760354072358681, 'MR': 3325.9582003828973, 'HITS@1': 0.42756860242501593, 'HITS@3': 0.49505424377791957, 'HITS@10': 0.5724313975749841}
  ```

## [导出过程](#目录)

```shell
python export.py --eval_checkpoint [EVAL_CHECKPOINT] --file_format [FILE_FORMAT]
```

参数eval_checkpoint为必填项，EXPORT_FORMAT 必须在 ["AIR", "MINDIR"]中选择。

## [推理过程](#目录)

在进行完上述导出模型过程之后，我们可以进行推理过程。请注意AIR模型只能在昇腾910环境上导出，MINDIR可以在任意环境上导出。本模型默认MINDIR格式，batch_size在推理时默认为1，在执行推理过程之前需要手动修改default_config.yaml中的test_batch_size为1。

- 在昇腾310上使用wn18rr数据集进行推理

  先进入到scripts目录下，然后执行下述命令即可进行推理：

  ```bash
  bash run_infer_310.sh ../rotate-head.mindir ../rotate-tail.mindir ../Data/wn18rr/ y 0
  ```

  完成上述命令后，即可在scripts下的eval.log中找到类似以下的结果：

  ```bash
  {'MRR': 0.4752237112953095, 'MR': 3239.202456924059, 'HITS@1': 0.4262922782386726,  'HITS@3': 0.49553286534779833, 'HITS@10': 0.5753031269942566}
  ```

# [模型描述](#目录)

## [性能](#目录)

### [训练性能](#目录)

| 参数          | Ascend                                                       | GPU                                                          |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8     | Tesla V100-SXM2-32GB                                         |
| 上传日期      | 2021-12-01                                                   | 2021-12-01                                                   |
| MindSpore版本 | 1.5.0                                                        | 1.5.0                                                        |
| 数据集        | WN18RR                                                       | WN18RR                                                       |
| 训练参数      | batch_size=512, negative_sample_size=1024, hidden_dim=500, gamma=6.0, alpha=0.5, lr=0.00005, max_steps=80000 | batch_size=512, negative_sample_size=1024, hidden_dim=500, gamma=6.0, alpha=0.5, lr=0.00005, max_steps=70000 |
| 优化器        | Adam                                                         | Adam                                                         |
| 损失函数      | 自对抗负采样Self-adversarial Negative Sampling Loss Function | 自对抗负采样Self-adversarial Negative Sampling Loss Function |
| 速度          | 单卡：141ms/step；8卡：24ms/step                             | 单卡：284ms/step；8卡：90ms/step                             |
| 总时长        | 单卡：294min；8卡：38min                                     | 单卡：360min；8卡：107min                                    |
| 微调检查点    | 156M (.ckpt文件)                                             | 156M (.ckpt文件)                                             |

### [评估描述](#目录)

| 参数          | Ascend                      | GPU                   |
| ------------------- | ------------------- | ------------------- |
| 模型版本       | RotatE            | RotatE     |
| 资源            | Ascend 910, 系统 Euler2.8 | Tesla V100-SXM2-32GB |
| 上传日期       | 2021-12-17 | 2021-12-17 |
| MindSpore版本   | 1.5.0                 | 1.5.0           |
| 数据集             | wn18rr                  | wn18rr            |
| 输出             | score | score |
| MRR           | 单卡：0.475194；8卡：0.475609                            | 单卡：0.476046；8卡：0.476035 |
| MR            | 单卡：3239.53；8卡：3240.68                              | 单卡：3327.17；8卡：3325.95 |
| HITS@1        | 单卡：0.426132；8卡：0.426292                            | 单卡：0.428206；8卡：0.427568 |
| HITS@3        | 单卡：0.495213；8卡：0.494894                            | 单卡：0.494097；8卡：0.495054 |
| HITS@10       | 单卡：0.575462；8卡：0.574984                            | 单卡：0.572112；8卡：0.572431 |
| 推理模型 | 156M (.ckpt文件)          | 156M (.ckpt文件) |

### [推理描述](#目录)

| 参数          | Ascend                    |
| ------------- | ------------------------- |
| 模型版本      | RotatE                    |
| 资源          | Ascend 310, 系统 Euler2.8 |
| 上传日期      | 2021-12-17                |
| MindSpore版本 | 1.5.0                     |
| 数据集        | wn18rr                    |
| 输出          | score                     |
| MRR           | 单卡: 0.475223            |
| MR            | 单卡: 3239.20             |
| HITS@1        | 单卡: 0.426292            |
| HITS@3        | 单卡: 0.495532            |
| HITS@10       | 单卡: 0.515303            |
| 推理模型      | 156M (.ckpt文件)          |

# [随机情况说明](#目录)

- `rotate.py`中Embedding矩阵随机初始化和`dataset.py`中负样本随机采样。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。  
