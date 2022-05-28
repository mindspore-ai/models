# 目录

- [目录](#目录)
    - [Transformer-XL 概述](#transformer-xl-概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
            - [训练脚本参数](#训练脚本参数)
            - [运行选项](#运行选项)
            - [网络参数](#网络参数)
        - [准备数据集](#准备数据集)
        - [训练过程](#训练过程)
        - [评估过程](#评估过程)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

## Transformer-XL 概述

Transformer-XL是对Transformer的改进，主要是解决长序列的问题。同时结合了RNN序列建模和Transformer自注意力机制的优点，引入循环机制（Recurrence
Mechanism）和相对位置编码（Relative Positional
Encoding），在输入数据的每个段上使用Transformer的注意力模块，并使用循环机制来学习连续段之间的依赖关系。并成功在enwik8、text8等语言建模数据集上取得SoTA效果。

[论文](https://arxiv.org/abs/1901.02860):  Dai Z, Yang Z, Yang Y, et al. Transformer-xl: Attentive language models beyond
a fixed-length context[J]. arXiv preprint arXiv:1901.02860, 2019.

## 模型架构

Transformer-XL主干结构为Transformer，在原有基础上加入了循环机制（Recurrence Mechanism）和相对位置编码（Relative Positional Encoding）

## 数据集

以下数据集包含训练数据集和评估数据集，数据集推荐使用 `bash getdata.sh` 的方式自动下载并预处理。

[enwik8](http://mattmahoney.net/dc/enwik8.zip)

enwik8数据集基于维基百科，通常用于衡量模型压缩数据的能力。包含了100MB未处理的Wikipedia的文本。

如果直接通过链接下载enwik8数据集，请通过下载并执行 [prep_enwik8.py](https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py) 的方式对下载的数据集进行预处理。

数据集大小

- 训练集：共计88,982,818个字符
- 验证集：共计4,945,742个字符
- 测试集：共计36,191个字符

数据集格式：txt文本

数据集目录结构：

```text
└─data
  ├─enwik8
    ├─train.txt       # 训练集
    ├─train.txt.raw   # 训练集(未处理)
    ├─valid.txt       # 验证集
    ├─valid.txt.raw   # 验证集(未处理)
    ├─test.txt        # 测试集
    └─test.txt.raw    # 测试集(未处理)
```

- [text8](http://mattmahoney.net/dc/text8.zip)

text8同样包含了100MB的Wikipedia文本，区别在于在enwik8数据集的基础上移除了26个字母和空格以外的其他字符。

如果直接通过链接下载text8数据集，请通过执行 prep_text8.py 的方式对下载的数据集进行预处理。

数据集大小：

- 训练集：共计89,999,999个字符
- 验证集：共计4,999,999个字符
- 测试集：共计5,000,000个字符

数据集格式：txt文本

数据集目录结构：

```text
└─data
  ├─text8
    ├─train.txt       # 训练集
    ├─train.txt.raw   # 训练集(未处理)
    ├─valid.txt       # 验证集
    ├─valid.txt.raw   # 验证集(未处理)
    ├─test.txt        # 测试集
    └─test.txt.raw    # 测试集(未处理)
```

## 环境要求

- 硬件（Ascend处理器）
    - 使用Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

- 在GPU上运行

数据集准备完成后，请按照如下步骤开始训练和评估：

```bash
# 对参数进行微调: enwik8_base.yaml中对超参数进行调整
# 其中[DATA_NAME]属于缺省参数[enwik8，text8]
# 其中[TRAIN_URL]参数可以设置为一个字符名称，这样会自动按照这个名称在/script/train/下面创建对应的模型训练文件，也可以设置为一个路径，例如 `"/home/mindspore/transformer-xl/enwik8_8p"`  这种方式会将训练的模型单独保存在这个目录下。

# 运行非分布式训练示例
bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR] [DATA_NAME] [TRAIN_URL] [CONFIG_PATH]
# for example: bash run_standalone_train_gpu.sh 0 /home/mindspore/transformer-xl/data/enwik8/ enwik8 experiments /home/mindspore/transformer-xl/yaml/enwik8_base.yaml

# 运行分布式训练示例
bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_DIR] [DATA_NAME] [TRAIN_URL] [CONFIG_PATH]
# for example: bash run_distribute_train_gpu.sh 4 0,1,2,3 /home/mindspore/transformer-xl/data/enwik8/ enwik8 experiments /home/mindspore/transformer-xl/yaml/enwik8_base.yaml

# 运行评估示例
bash run_eval_gpu.sh [DATA_URL] [DATA_NAME] [CKPT_PATH] [CONFIG_PATH] [DEVICE_ID(optional)]
# for example: bash run_eval_gpu.sh  /home/mindspore/transformer-xl/data/enwik8/ enwik8 /home/mindspore/transformer-xl/script/experiments-enwik8/20220416-140816/model7.ckpt /home/mindspore/transformer-xl/yaml/enwik8_base.yaml 0
```

## 脚本说明

### 脚本和样例代码

```text
.
└─Transformer-XL
  ├─README.md             // descriptions about Transformer-XL
  ├─README_CN.md          // descriptions about Transformer-XL
  ├─scripts
    ├─run_distribute_train_gpu.sh   // shell script for distributed training on GPU
    ├─run_standalone_train_gpu.sh   // shell script for training on GPU
    └─run_eval_gpu.sh               // shell script for testing on GPU
  ├─src
    ├─callback
      ├─eval.py           // callback function(eval)
      ├─flag.py           // callback function(flag)
      └─log.py            // callback function(log)
    ├─loss_fn
      └─ProjectedAdaptiveLogSoftmaxLoss.py    // loss
    ├─metric
      └─calc.py               // get bpc and ppl
    ├─model
      ├─attn.py               // Attention code
      ├─dataset.py            // get dataset
      ├─embedding.py          // PositionalEmbedding and AdaptiveEmbedding
      ├─layer.py              // layer code
      ├─mem_transformer.py    // Transformer-XL model
      ├─positionwiseFF.py     // positionwiseFF
      └─vocabulary.py         // construct vocabulary
    ├─model_utils
      ├─config.py             // parameter configuration
      ├─device_adapter.py     // device adapter
      ├─local_adapter.py      // local adapter
      └─moxing_adapter.py     // moxing adapter
    ├─utils
      ├─additional_algorithms.py  // General method
      ├─dataset_util.py           // Interface to get dataset
      └─nnUtils.py                // Basic method
  ├─yaml
      ├─enwik8_base.yaml          // parameter configuration on gpu
      ├─enwik8_large.yaml         // parameter configuration on gpu
      └─text8_large.yaml          // parameter configuration on gpu
  ├─getdata.sh                    // shell script for preprocessing dataset
  ├─eval.py                       // evaluation script
  └─train.py                      // training script
```

### 脚本参数

#### 训练脚本参数

```text
用法:
train.py
如果需要对参数进行设置，可以修改./enwik8_base.yaml文件中的参数实现。
如果需要更改参数配置文件，可以更改/src/model_utils/config.py中line130的--config_path参数。
```

#### 网络参数

```text
数据集和网络参数（训练/微调/评估）:
    n_layer       网络层数: N, 默认值为 12
    d_model       模型维度, 默认值为 512
    n_head        总的注意力头数, 默认值为 8
    d_head        注意力头的维度, 默认值为 64
    d_inner       前馈网络的维度, 默认值为 2048
    dropout       输出层的随机失活概率: Q, 默认值是 0.1
    dropatt       注意力层的随机失活概率: Q, default is 0.0
    max_step      迭代次数: N, 默认值为 400000
    tgt_len       标签特征维度大小, 默认值为 512
    mem_len       记忆特征维度大小, 默认值为 512
    eval_tgt_len  迭代任务中标签特征维度大小, 默认值为 128
    batch_size    输入数据集的批次大小: N, 默认值是 22

学习率参数:
    lr            学习率: Q, 默认值为 0.00025
    warmup_step   热身学习率步数: N, 默认值为 0
```

### 准备数据集

- 运行 `bash getdata.sh` , 脚本会创建 `./data` 目录并将数据集自动下载到该目录下

- 下载数据集并配置好DATA_PATH

### 训练过程

- 通过直接用sh输入参数的方式输入路径，或在`enwik8_base.yaml`中设置选项，确保 'datadir' 路径为数据集路径。设置其他参数包括loss_scale、学习率和网络超参数。

- 运行`run_standalone_train_gpu.sh`，进行Transformer-XL模型的非分布式训练。

    ```
    # 运行非分布式训练示例
    bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR] [DATA_NAME] [TRAIN_URL] [CONFIG_PATH]
    # for example: bash run_standalone_train_gpu.sh 0 /home/mindspore/transformer-xl/data/enwik8/ enwik8 experiments /home/mindspore/transformer-xl/yaml/enwik8_base.yaml
    ```

- 运行`run_distribute_train_gpu.sh`，进行Transformer-XL模型的分布式训练。

    ```
    # 运行分布式训练示例
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_DIR] [DATA_NAME] [TRAIN_URL] [CONFIG_PATH]
    # for example: bash run_distribute_train_gpu.sh 4 0,1,2,3 /home/mindspore/transformer-xl/data/enwik8/ enwik8 experiments /home/mindspore/transformer-xl/yaml/enwik8_base.yaml
    ```

### 评估过程

- 通过直接用sh输入参数的方式输入路径，或在`enwik8_base.yaml`中设置选项，设置 'load_path' 文件路径。

- 运行`run_eval_gpu.sh`，评估Transformer-XL模型。

    ```
    # 运行评估示例
    bash run_eval_gpu.sh [DATA_URL] [DATA_NAME] [CKPT_PATH] [CONFIG_PATH] [DEVICE_ID(optional)]
    # for example: bash run_eval_gpu.sh  /home/mindspore/transformer-xl/data/enwik8/ enwik8 /home/mindspore/transformer-xl/script/experiments-enwik8/20220416-140816/model7.ckpt /home/mindspore/transformer-xl/yaml/enwik8_base.yaml 0
    ```

## 模型描述

### 性能

#### 训练性能

| 参数           | GPU                            |
| ------------- | ------------------------------ |
| 资源           | MindSpore                      |
| 上传日期        | 2022-04-22                     |
| MindSpore版本  | 1.6.1                           |
| 数据集         | enwik8                          |
| 训练参数       | max_step=400000, batch_size=22  |
| 优化器         | Adam                            |
| 损失函数       | Softmax Cross Entropy           |
| BPC分数       | 1.07906                         |
| 速度          | 421.24ms/step(1p,bsz=8)  |
| 损失          | 0.75                            |
| 推理检查点     | 1.45G(.ckpt文件)                |
| 脚本          | Transformer-XL script           |

#### 评估性能

| 参数           | GPU                   |
| ------------- | --------------------------- |
|资源            | MindSpore               |
| 上传日期        | 2022-04-22                |
| MindSpore版本  | 1.6.1                      |
| 数据集         | enwik8                     |
| batch_size    | 22                        |
| 输出           | 损失loss,BPC分数                   |
| 损失loss       | 0.75                      |
| BPC分数       | 1.07906                      |

## 随机情况说明

以下三种随机情况：

- 轮换数据集
- 初始化部分模型权重
- 随机失活运行

train.py已经设置了一些种子，避免数据集轮换和权重初始化的随机性。若需关闭随机失活，将default_config.yaml中相应的dropout_prob参数设置为0。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

