# 目录

<!-- TOC -->

- [目录](#目录)
- [Roberta initialized Seq2Seq模型简介](#模型简介)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [CelebA上的PGAN](#CelebA上的PGAN)
- [ModelZoo主页](#modelzoo主页)

# 模型简介

RoBERTa initialized Seq2Seq 模型是一个基于 Transformer 的 Seq2Seq 模型，该模型在编码器和解码器上同时加载公开可用的预训练语言模型RoBERTa 的
Checkpoint。该模型在机器翻译、文本摘要、句子拆分和句子融合方面产生了新的最先进的结果。

[论文](https://aclanthology.org/2020.tacl-1.18/)：Leveraging Pre-trained Checkpoints for Sequence Generation Tasks

# 模型架构

Roberta initialized Seq2Seq模型使用seq2seq架构，编码器和解码器均由Transformer组成。对于编码器，继承了BERT
Transformer层的实现，与规范的Transformer层略有不同。BERT使用GELU激活函数，而不是标准的RELU函数。解码器的实现与编码器有所不同，将self-attention
机制MASK为仅查看左侧上下文，同时使用encoder-decoder attention 机制。编码器和解码器都使用RoBERTa Checkpoints，同时，两者之间的权重共享。

# 数据集

使用的数据集: [BBC XSum](<https://github.com/EdinburghNLP/XSum>)

XSum 数据集包含 226,711 条 Wayback 存档的 BBC 文章，涵盖了近十年（2010 年至 2017 年），涵盖了广泛的领域(例如，新闻、政治、体育、天气、商业、技术、科学、健康、家庭、教育、 娱乐和艺术）。

该数据集可用作文本摘要生成任务。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  bash scripts/run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [CONFIG_PATH] [DATA_PATH] [CHECKPOINT_PATH]

  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [RANK_TABLE_FILE] [CONFIG_PATH] [CHECKPOINT_PATH]

  # 运行评估示例
  bash scripts/run_eval.sh  [DEVICE_ID] [DATA_PATH] [CKPT_PATH] [CONFIG_PATH]
  [VOCAB_FILE_PATH] [OUTPUT_FILE]
  # CONFIG_PATH要和训练时保持一致
  ```

对于训练需要传入预训练的RoBERTa Checkpoints，下载pytorch的参数后，可以使用代码`read_ckpt_torch.py`  进行转换。

对于分布式训练，需要提前创建JSON格式的hccl配置文件。该配置文件的绝对路径作为运行分布式脚本的第二个参数。

请遵循以下链接中的说明：

<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

# 脚本说明

## 脚本及样例代码

```shell
.
└─Roberta_Seq2Seq
  ├─README_CN.md
  ├─README.md
#   ├─ascend310_infer
#     ├─build.sh
#     ├─CMakeLists.txt
#     ├─inc
#     │ └─utils.h
#     └─src
#       ├─main.cc
#       └─utils.cc
  ├─cover_torch_ckpt
    └─read_ckpt_torch.py
  ├─scripts
    ├─run_distribute_train.sh
    ├─run_standalone_train.sh
    └─run_eval.sh
  ├─src
    ├─beam_search.py
    ├─dataset.py
    ├─lr_schedule.py
    ├─model_encoder_decoder.py
    ├─model_infer.py
    ├─model_train.py
    ├─process_output.py
    ├─roberta_model.py
    ├─rouge_score.py
    ├─sample_process.py
    ├─tokenization.py
    ├─transform_data.py
    ├─utils.py
    └─model_utils
      ├─config.py
      ├─device_adapter.py
      ├─local_adapter.py
      └─moxing_adapter.py
  ├─default_config.yaml
  ├─eval.py
  ├─export.py
  ├─hccl_tools.py
  ├─requirements.txt
  └─train.py
```

### 准备数据集

- 您可以参考[GitHub](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset)下载并预处理XSum数据集。假设您已获得下列文件，并存在DATA_PATH下：
    - train.json
    - test.json
    - validation.json
- 将原数据转换为MindRecord数据格式进行训练和评估：

```bash
python src/transform_data.py --vocab_file_path [VOCAB_PATH] --data_path [DATA_PATH]
```

## 训练过程

- 在`default_config.yaml`中设置选项，包括loss_scale、学习率和网络超参数。
- 运行`run_standalone_train.sh`，进行模型的单卡训练。

    ``` bash
    bash scripts/run_standalone_train.sh [DEVICE_ID] [EPOCH_SIZE] [CONFIG_PATH] [DATA_PATH] [CHECKPOINT_PATH]
    ```

- 运行`run_distribute_train.sh`，进行Transformer模型的分布式训练。

    ``` bash
    # Ascend environment
    bash scripts/run_distribute_train.sh [DEVICE_NUM] [EPOCH_SIZE] [DATA_PATH] [RANK_TABLE_FILE] [CONFIG_PATH] [CHECKPOINT_PATH]
    ```

## 评估过程

- 运行`run_eval.sh`，评估Roberta initialized Seq2Seq模型，并生成真实的摘要。

    ``` bash
    # Ascend environment
    bash scripts/run_eval.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH] [CONFIG_PATH]
    [VOCAB_FILE_PATH] [OUTPUT_FILE]
    ```

## 推理过程

### 导出MindIR

```shell
python export.py --model_file [CKPT_PATH] --file_name [FILE_NAME] --config_path [CONFIG_PATH]
```

参数ckpt_file为必填项，脚本会在当前目录下生成对应的MINDIR文件。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CONFIG_PATH]
```

- `NEED_PREPROCESS` 表示是否需要对数据集进行预处理, 取值为'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的目录下，...

# 模型描述

## 性能

### 评估性能

#### BBC XSum上的Roberta initialized Seq2Seq

| 参数                       | Ascend 910                                                  |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本                   | Roberta initialized Seq2Seq                                                      |
| 资源                       | Ascend                                                      |
| 上传日期                   | 08/14/2022 (month/day/year)                                 |
| MindSpore版本              | 1.6.1                                                      |
| 数据集                     | BBC XSum                                                      |
| 训练参数                   | batch_size=32, epoch=30, lr=0.00003                                   |
| 优化器                     | Adam                                                        |                                                 |
| 速度                       |2p:578.192 ms; 1p:274.678 ms;                               |
| 输出             | Rouge1, Rouge2, RougeL |
| 结果                   | Rouge1=31.93, Rouge2=11.17, RougeL=25.30                             |
| 脚本                       | [Roberta Seq2Seq script](https://gitee.com/mindspore/models/) |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)