# 目录

<!-- TOC -->

- [GRU](#gru)
    - [论文](#论文)
- [模型结构](#模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
    - [要求](#要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [数据集准备](#数据集准备)
    - [配置文件](#配置文件)
    - [训练过程](#训练过程)
    - [推理过程](#推理过程)
    - [导出MindIR](#导出mindir)
    - [ONNX导出和评估](#onnx导出和评估)
        - [ONNX导出](#onnx导出)
        - [ONNX评估](#onnx评估)
    - [推理过程](#推理过程-1)
        - [用法](#用法)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [其他](#其他)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [GRU](#目录)

门控递归单元（GRU）是一种递归神经网络算法，就像长短期存储器（LSTM）一样。它是由Kyunghyun Cho、Bart van Merrienboer等人在2014年的论文《Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation》中提出的。论文提出了一种新的神经网络模型RNN Encoder-Decoder，该模型由两个递归神经网络（RNN）组成，为了提高翻译任务的效果，我们还参考了另外两篇论文：《Sequence to Sequence Learning with Neural Networks》、《Neural Machine Translation by Jointly Learning to Align and Translate》。

## 论文

[论文1](https://arxiv.org/abs/1406.1078): "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014, Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio

[论文2](https://arxiv.org/pdf/1409.3215.pdf): "Sequence to Sequence Learning with Neural Networks", 2014, Ilya Sutskever, Oriol Vinyals, Quoc V. Le

[论文3](): "Neural Machine Translation by Jointly Learning to Align and Translate", 2014, Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio

# [模型结构](#目录)

GRU模型主要由Encoder和Decoder组成。其中，Encoder由双向GRU cell组成，Decoder主要包括注意力和GRU cell。网络的输入是单词（文本或句子）序列，网络的输出是vocab中每个单词的概率。我们选择最大概率作为我们的预测结果。

# [数据集](#目录)

在这个模型中，我们使用Multi30K数据集作为训练和测试集。其中，训练数据29,000个，每个数据包含1个德语句子及其英语翻译，测试数据包含1000个德语和英语句子。我们还提供了一个预处理脚本来对数据集进行分词并创建vocab文件。

# [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 要求

```txt
nltk
numpy
onnxruntime-gpu
```

按以下方式安装nltk：

```bash
pip install nltk
```

然后下载其他软件包，如下所示：

```python
import nltk
nltk.download()
```

# [快速入门](#目录)

- Ascend、GPU或CPU本地运行

    数据集准备完成后，可以按照以下方式开始训练和评估：

    ```bash
    cd ./scripts
    # 下载数据集
    bash download_dataset.sh

    # 预处理数据集
    bash preprocess.sh [DATASET_PATH]

    # 创建MindRecord
    bash create_dataset.sh [DATASET_PATH] [OUTPUT_PATH]

    # 运行训练示例
    bash run_standalone_train_{platform}.sh [TRAIN_DATASET_PATH]
    # 平台：Ascend或GPU
    python train.py --config_path=[CPU_CONFIG_PATH] --dataset_path=[TRAIN_DATASET_PATH]
    # 平台：CPU

    # 运行分布式训练示例
    bash run_distribute_train_{platform}.sh [RANK_TABLE_FILE] [TRAIN_DATASET_PATH]
    # 平台：Ascend或GPU
    # 如果使用GPU，则不需要[RANK_TABLE_FILE]
    # 如果使用CPU，则不需要此步骤

    # 运行评估示例
    bash run_eval_{platform}.sh [CKPT_FILE] [DATASET_PATH]
    # 平台：Ascend或GPU
    python eval.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # 平台：CPU
    ```

    在数据集准备和训练之后，可以运行quick_start.py显示训练结果。

    ```bash
    # run quick_start.py
    python quick_start.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # 平台：CPU
    # 示例
    python quick_start.py --dataset_path=./data/mindrecord/multi30k_test_mindrecord_32 --ckpt_file=./ckpt_0/0-20_1807.ckpt --device_target=CPU
    ```

- 在ModelArts上运行（如果想在ModelArts中运行，请查看[ModelArts官方文档](https://support.huaweicloud.com/modelarts/)，并按照以下方式开始训练）

    ```python
    # ModelArts上运行8卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"run_distribute=True"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"run_distribute=True"。
    #          在网页上添加"dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32_0"。
    #          在网页上添加其他参数。
    # （2）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （3）在网页上设置代码目录为"/path/gru"。
    # （4）在网页上设置启动文件为"train.py"。
    # （5）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （6）创建作业。
    #
    # ModelArts上运行单卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32_0"。
    #          在网页上添加其他参数。
    # （2）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （3）在网页上设置代码目录为"/path/gru"。
    # （4）在网页上设置启动文件为"train.py"。
    # （5）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （6）创建作业。
    #
    # ModelArts上运行单卡评估
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_trained_ckpt/'"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data/mindrecord/multi30k_train_mindrecord_32_0'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"ckpt_file=/cache/checkpoint_path/model.ckpt"。
    #          在网页上添加"checkpoint_url=s3://dir_to_trained_ckpt/"。
    #          在网页上添加"dataset_path=/cache/data/mindrecord/multi30k_train_mindrecord_32"。
    #          在网页上添加其他参数。
    # （2）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （3）在网页上设置代码目录为"/path/gru"。
    # （4）在网页上设置启动文件为"eval.py"。
    # （5）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （6）创建作业。
    ```

# [脚本说明](#目录)

GRU网络脚本和代码结果如下：

```text
├── gru
  ├── README.md                              // GRU模型介绍
  ├── model_utils
  │   ├──__init__.py                         // 模块初始化文件
  │   ├──config.py                           // 解析参数
  │   ├──device_adapter.py                   // ModelArts的设备适配器
  │   ├──local_adapter.py                    // 本地适配器
  │   ├──moxing_adapter.py                   // ModelArts的Moxing适配器
  ├── src
  │   ├──create_data.py                      // 数据集准备
  │   ├──dataset.py                          // 要馈送到模型中的数据集加载器
  │   ├──gru_for_infer.py                    // GRU评估模型架构
  │   ├──gru_for_train.py                    // GRU训练模型架构
  │   ├──loss.py                             // 损失架构
  │   ├──lr_schedule.py                      // 学习率调度器
  │   ├──parse_output.py                     // 解析output文件
  │   ├──preprocess.py                       // 数据集预处理
  |   ├──rnn_cells.py                        // RNN cell架构
  |   ├──rnns.py                             // RNN层架构
  │   ├──seq2seq.py                          // Seq2seq架构
  |   ├──utils.py                            // RNN工具
  │   ├──tokenization.py                     // 数据集分词
  │   ├──weight_init.py                      // 初始化网络中的权重
  ├── scripts
  │   ├──create_dataset.sh                   // 创建数据集的shell脚本
  │   ├──download_dataset.sh                 // 下载数据集的shell脚本
  │   ├──parse_output.sh                     // 解析eval输出文件计算BLEU的shell脚本
  │   ├──preprocess.sh                       // 预处理数据集的shell脚本
  │   ├──run_distributed_train_ascend.sh     // Ascend分布式训练shell脚本
  │   ├──run_distributed_train_gpu.sh        // GPU分布式训练shell脚本
  │   ├──run_eval_ascend.sh                  // Ascend单机评估shell脚本
  │   ├──run_eval_gpu.sh                     // GPU单机评估shell脚本
  │   ├──run_eval_onnx_gpu.sh                //  GPU单机ONNX模型评估shell脚本
  │   ├──run_infer_310.sh                    // Ascend 310推理的shell脚本
  │   ├──run_standalone_train_ascend.sh      // Ascend单机评估shell脚本
  │   ├──run_standalone_train_gpu.sh         // GPU单机评估shell脚本
  ├── default_config.yaml                    // 配置
  ├── cpu_config.yaml                        // CPU配置
  ├── postprocess.py                         // GRU后处理脚本
  ├── preprocess.py                          // GRU预处理脚本。
  ├── export.py                              // 导出API入口
  ├── eval.py                                // 推理API入口
  ├── eval_onnx.py                           // ONNX推理API入口
  ├── quick_start.py                         // GRU快速启动脚本
  ├── requirements.txt                       // 第三方包需求
  ├── train.py                               // 训练API入口
```

## [数据集准备](#目录)

首先，从WMT16官网下载数据集。

```bash
cd scripts
bash download_dataset.sh
```

下载Multi30k数据集文件后，我们得到了六个数据集文件，如下所示。把它们放在同一个目录中。

```text
train.de
train.en
val.de
val.en
test.de
test.en
```

然后，使用scripts/preprocess.sh对数据集文件进行分词，并获取vocab文件。

```bash
bash preprocess.sh [DATASET_PATH]
```

预处理后，我们得到后缀为".tok"的数据集文件和两个名为vocab.de和vocab.en的vocab文件。
然后，使用scripts/create_dataset.sh来创建格式为mindrecord的数据集文件。

```bash
bash create_dataset.sh [DATASET_PATH] [OUTPUT_PATH]
```

最后，我们将得到multi30k_train_mindrecord_0至multi30k_train_mindrecord_8作为训练集，multi30k_test_mindrecord作为测试集。

## [配置文件](#目录)

可以在config.py中设置训练和评估的参数。所有数据集使用相同的参数名称，参数值可以根据需要更改。

- Ascend和GPU的网络参数

  ```text
    "batch_size": 16,                  # 输入数据集的batch size
    "src_vocab_size": 8154,            # 源数据集词汇表大小
    "trg_vocab_size": 6113,            # 目标数据集词汇表大小
    "encoder_embedding_size": 256,     # Encoder嵌入大小
    "decoder_embedding_size": 256,     # Decoder嵌入大小
    "hidden_size": 512,                # GRU的hidden size
    "max_length": 32,                  # 最大句子长度
    "num_epochs": 30,                  # 总epoch数
    "save_checkpoint": True,           # #是否保存检查点文件
    "ckpt_epoch": 1,                   # 保存检查点文件的频率
    "target_file": "target.txt",       # 目标文件
    "output_file": "output.txt",       # 输出文件
    "keep_checkpoint_max": 30,         # 检查点文件的最大数量
    "base_lr": 0.001,                  # 初始化学习率
    "warmup_step": 300,                # warmup步骤
    "momentum": 0.9,                   # 优化器的动量
    "init_loss_scale_value": 1024,     # 初始化缩放
    'scale_factor': 2,                 # 动态损失缩放的缩放因子
    'scale_window': 2000,              # 动态损失缩放的缩放窗口
    "warmup_ratio": 1/3.0,             # warmup比率
    "teacher_force_ratio": 0.5         # teacher force比率
  ```

- Ascend和GPU的网络参数

  ```text
    "batch_size": 16,                  # 输入数据集的batch size
    "src_vocab_size": 8154,            # 源数据集词汇表大小
    "trg_vocab_size": 6113,            # 目标数据集词汇表大小
    "encoder_embedding_size": 256,     # Encoder嵌入大小
    "decoder_embedding_size": 256,     # Decoder嵌入大小
    "hidden_size": 512,                # GRU的hidden size
    "max_length": 32,                  # 最大句子长度
    "num_epochs": 13,                  # 总epoch数
    "save_checkpoint": True,           # #是否保存检查点文件
    "ckpt_epoch": 1,                   # 保存检查点文件的频率
    "target_file": "target.txt",       # 目标文件
    "output_file": "output.txt",       # 输出文件
    "keep_checkpoint_max": 5,          # 检查点文件的最大数量
    "base_lr": 0.001,                  # 初始化学习率
    "warmup_step": 300,                # warmup步骤
    "momentum": 0.9,                   # 优化器的动量
    "init_loss_scale_value": 1024,     # 初始化缩放
    'scale_factor': 2,                 # 动态损失缩放的缩放因子
    'scale_window': 2000,              # 动态损失缩放的缩放窗口
    "warmup_ratio": 1/3.0,             # warmup比率
    "teacher_force_ratio": 0.5         # teacher force比率
  ```

## [训练过程](#目录)

- 在单个设备上启动任务训练，如果使用Ascend或GPU，则运行shell脚本，如果使用CPU，则运行Python文件。

    ```bash
    cd ./scripts
    # 平台：Ascend或GPU
    bash run_standalone_train_{platform}.sh [DATASET_PATH]
    # 示例：
    bash run_standalone_train_ascend.sh /Muti30k/mindrecord/multi30k_train_mindrecord_32_0

    # 平台：CPU
    python train.py --config_path=[CPU_CONFIG_PATH] --dataset_path=[TRAIN_DATASET_PATH] --device_target=CPU
    # 示例：
    python train.py --config_path=cpu_config.yaml --dataset_path=./data/mindrecord/multi30k_train_mindrecord_32_0 --device_target=CPU
    ```

- 运行GRU分布式训练的脚本。若在多台设备上进行任务训练，在`scripts/`执行以下bash命令：

    ``` bash
    cd ./scripts
    bash run_distributed_train_{platform}.sh [RANK_TABLE_PATH] [DATASET_PATH]
    # 平台：Ascend或GPU
    # 如果使用GPU，则不需要[RANK_TABLE_FILE]
    # 如果使用CPU，则不需要此步骤
    ```

## [推理过程](#目录)

- 运行GRU评估脚本。命令如下所示。

    ``` bash
    cd ./scripts
    # 平台：Ascend或GPU
    bash run_eval_{platform}.sh [CKPT_FILE] [DATASET_PATH]
    # 示例：
    bash run_eval_ascend.sh /data/ckpt_0/0-20_1807.ckpt /data/mindrecord/multi30k_test_mindrecord_32

    # 平台：CPU
    python eval.py --dataset_path=[DATASET_PATH] --ckpt_file=[CKPT_FILE] --device_target=CPU
    # 示例：
    python eval.py --dataset_path=./data/mindrecord/multi30k_test_mindrecord_32 --ckpt_file=./ckpt_0/0-20_1807.ckpt --device_target=CPU
    ```

- 评估后，将获得eval/target.txt和eval/output.txt。然后，使用parse_output.sh来获取翻译。

    ``` bash
    cp eval/*.txt ./
    bash parse_output.sh target.txt output.txt /path/vocab.en
    ```

    我们建议在本地执行此操作，但你也可以通过运行带有此命令"os.system("bash parse_output.sh target.txt output.txt /path/vocab.en")"的python脚本在ModelArts上执行此操作。

- 解析输出后，将获得target.txt.forbleu和output.txt.forbleu。要计算BLEU分数，可以使用[perl脚本](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)并执行以下命令，获取BLEU分数。

    ```bash
    perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
    ```

    我们建议在本地执行此操作，但你也可以通过运行带有此命令"os.system("perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu")"的python脚本在ModelArts上执行此操作。

注：`DATASET_PATH`是MindRecord的路径。即，train: /dataset_path/multi30k_train_mindrecord_0  eval: /dataset_path/multi30k_test_mindrecord

## [导出MindIR](#目录)

- 本地导出

    ```python
    # ckpt_file参数必填，EXPORT_FORMAT取值为["AIR", "MINDIR"]
    python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
    ```

- 在ModelArts上导出（如果想在ModelArts中运行，请查看[ModelArts官方文档](https://support.huaweicloud.com/modelarts/)，并按照以下方式开始）

    ```python
    # ModelArts上运行单卡评估
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_trained_ckpt/'"。
    #          在default_config.yaml文件上设置"file_name='./gru'"。
    #          在default_config.yaml文件上设置"file_format='MINDIR'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在网页上添加"checkpoint_url='s3://dir_to_trained_ckpt/'"。
    #          在网页上添加"file_name='./gru'"。
    #          在网页上添加"file_format='MINDIR'"。
    #          在网页上添加其他参数。
    # （2）在网页上设置代码目录为"/path/gru"。
    # （3）在网页上设置启动文件为"export.py"。
    # （4）在网页上设置自己的"Output file path"、"Job log path"。
    # （5）创建作业。
    ```

## [ONNX导出和评估](#目录)

### ONNX导出

```bash
python export.py --device_target="GPU" --file_format="ONNX" --ckpt_file [CKPT_PATH]
# 示例：python export.py --device_target="GPU" --file_format="ONNX" --ckpt_file models/official/nlp/GRU/0-25_1807.ckpt
```

### ONNX评估

- 运行GRU ONNX评估脚本。命令如下所示。

    ``` bash
    cd ./scripts
    bash run_eval_onnx_gpu.sh [ONNX_CKPT_FILE] [DATASET_PATH]
    # 平台：GPU
    # 示例：
    bash run_eval_onnx_gpu.sh gru.onnx /data/mindrecord/multi30k_test_mindrecord_32
    ```

- 评估后，将获得eval/target.txt和eval/output.txt。然后，使用parse_output.sh来获取翻译。

    ``` bash
    cp eval/*.txt ./
    bash parse_output.sh target.txt output.txt /path/vocab.en
    ```

- 解析输出后，我们将得到target.txt.forbleu和output.txt.forbleu。如需计算BLEU分数，可以使用[perl脚本](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)并执行以下命令。

    ```bash
    perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
    ```

## [推理过程](#目录)

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

### 用法

在推理前，必须通过export.py导出mindir文件。输入文件必须为bin格式。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS`：表示是否需要预处理，其值为'y'或'n'。
`DEVICE_ID`：可选参数，默认值为0。

### 结果

获得target.txt和output.txt后，使用parse_output.sh来获取翻译。

``` bash
bash parse_output.sh target.txt output.txt /path/vocab.en
```

解析输出后，将获得target.txt.forbleu和output.txt.forbleu。如需计算BLEU分数，可以使用[perl脚本](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)并执行以下命令。

```bash
perl multi-bleu.perl target.txt.forbleu < output.txt.forbleu
```

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend                        | GPU                      | CPU |
| -------------------------- | ----------------------------- |---------------------------| -------------------------- |
| 资源                  | Ascend 910; EulerOS 2.8       | GTX1080Ti, Ubuntu 18.04   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz,Ubuntu 18.04 |
| 上传日期             | 06/05/2021  | 06/05/2021| 09/28/2022|
| MindSpore版本         | 1.2.0                         |1.2.0                      | 1.2.0 |
| 数据集                   | Multi30k数据集             | Multi30k数据集         | Multi30k数据集|
| 训练参数       | epoch=30, batch_size=16       | epoch=30, batch_size=16   | epoch=13, batch_size=16 |
| 优化器                 | Adam                          | Adam                      | Adam |
| 损失函数             | NLLLoss                       | NLLLoss                   | NLLLoss |
| 输出                   | 概率                  | 概率              | 概率|
| 速度                     | 35ms/step (单卡)              | 200ms/step (单卡)         | 1465ms/step (单卡) |
| Epoch时间| 64.4s (单卡)                                                 | 361.5s (单卡) | 2640s (单卡) |
| 损失| 3.86888 | 2.533958 | 2.9340835 |
| 参数量 （M）| 21 | 21 | 21 |
| 推理检查点| 272M（.ckpt）| 272M（.ckpt）| 321M（.ckpt）|
| 脚本| [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) |

### 推理性能

| 参数         | Ascend                      | GPU| CPU|
| ------------------- | --------------------------- |---------------------------| ------------------- |
| 资源           | Ascend 910; EulerOS 2.8     | GTX1080Ti, Ubuntu 18.04   | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz,Ubuntu 18.04 |
| 上传日期      | 06/05/2021| 06/05/2021| 09/28/2022|
| MindSpore版本  | 1.2.0                       | 1.2.0                     | 1.2.0 |
| 数据集            | Multi30K                    | Multi30K                  | Multi30K |
| batch_size          | 1                           | 1                         | 1 |
| 输出            | 标签索引                | 标签索引              | 标签索引|
| 准确率           | BLEU: 31.26                 | BLEU: 29.30               | BLEU: 30.19 |
| 推理模型| 272M（.ckpt）          | 272M（.ckpt）        | 321M（.ckpt）|

# [随机情况说明](#目录)

只有一种随机情况。

- 初始化一些模型权重。

为了避免权重初始化的随机性，已经在train.py中设置了一些种子。

# [其他](#其他)

该模型已在Ascend环境下得到验证，尚未在CPU和GPU环境下验证。

# [ModelZoo主页](#目录)

 请浏览官网[主页](https://gitee.com/mindspore/models)。
