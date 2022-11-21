# 目录

- [目录](#目录)
- [FiBiNET概述](#FiBiNET概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [预处理脚本参数](#预处理脚本参数)
    - [准备数据集](#准备数据集)
        - [处理Criteo数据集](#处理真实世界数据)
        - [处理合成数据集](#生成和处理合成数据)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
    - [评估过程](#评估过程)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [模型表现](#result)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# FiBiNET概述

FiBiNET (Feature Importance and Bilinear feature Interaction NETwork) 新浪微博于2019年提出的一种基于深度学习的广告推荐算法。[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)论文中描述了FiBiNET的实现原理。

# 模型架构

FiBiNET模型训练了宽线性模型和深度学习神经网络，并在Wide&Deep的基础上对神经网络部分增加了动态学习特征重要性的SENET模块 (Squeeze-and-Excitation Network) 与学习特征交叉的Bilinear-Interaction模块。

# 数据集

- [Criteo Kaggle Display Advertising Challenge Dataset](http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz)

# 环境要求

- 硬件（GPU）
    - 准备GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)，如需查看详情，请参见如下资源：
        - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
        - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

1. 克隆代码。

```bash
git clone https://gitee.com/mindspore/models.git
cd models/research/recommend/fibinet
```

2. 下载数据集。

  > 请参考[1](#数据集)获得下载链接。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

3. 使用此脚本预处理数据。处理过程可能需要一小时，生成的MindRecord数据存放在data/mindrecord路径下。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

4. 开始训练。

数据集准备就绪后，即可训练和评估模型。

```bash
# 执行Python脚本
python train.py --data_path=./data/mindrecord --device_target=GPU --eval_while_train=True

# 执行Shell脚本
bash ./script/run_train_gpu.sh './data/mindrecord/' 1 GPU True
```

按如下操作单独评估模型：

```bash
# 执行Python脚本
python eval.py  --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU

# 执行Shell脚本
bash ./script/run_eval_gpu.sh './data/mindrecord/' 1 GPU
```

## 脚本说明

## 脚本和样例代码

```markdown
└── fibinet
    ├── README.md                                 # 所有fibinet模型相关说明与教程
    ├── requirements.txt                          # python环境
    ├── script
    │   ├── common.sh
    │   ├── run_train_gpu.sh                      # GPU处理器单卡训练shell脚本
    │   └── run_eval_gpu.sh                       # GPU处理器单卡评估shell脚本
    ├──src
    │   ├── callbacks.py
    │   ├── datasets.py                           # 创建数据集
    │   ├── generate_synthetic_data.py            # 生成虚拟数据
    │   ├── __init__.py
    │   ├── metrics.py                            # 模型表现评价指标脚本
    │   ├── preprocess_data.py                    # 数据预处理
    │   ├── process_data.py
    │   ├── fibinet.py                            # FiBiNET主体架构
    │   └── model_utils
    │       ├── __init__.py
    │       ├── config.py                         # 获取训练配置信息
    │       └── moxing_adapter.py                 # 参数处理
    ├── default_config.yaml                       # 训练参数配置文件，任何模型相关的参数均建议在此处修改
    ├── train.py                                  # 训练脚本
    ├── eval.py                                   # 评估脚本
    └── export.py
```

## 脚本参数

### 训练脚本参数

```markdown

Used by: train.py

Arguments:

  --device_target                     Device where the code will be implemented, only support GPU currently. (Default:GPU)
  --data_path                         Where the preprocessed data is put in
  --epochs                            Total train epochs. (Default:10)
  --full_batch                        Enable loading the full batch. (Default:False)
  --batch_size                        Training batch size.(Default:1000)
  --eval_batch_size                   Eval batch size.(Default:1000)
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.(Default:10)
  --field_size                        The number of features.(Default:39)
  --vocab_size                        The total features of dataset.(Default:200000)
  --emb_dim                           The dense embedding dimension of sparse feature.(Default:10)
  --deep_layer_dim                    The dimension of all deep layers.(Default:[400,400,400])
  --deep_layer_act                    The activation function of all deep layers.(Default:'relu')
  --keep_prob                         The keep rate in dropout layer.(Default:0.5)
  --dropout_flag                      Enable dropout.(Default:0)
  --output_path                       Deprecated
  --ckpt_path                         The location of the checkpoint file. If the checkpoint file
                                      is a slice of weight, multiple checkpoint files need to be
                                      transferred. Use ';' to separate them and sort them in sequence
                                      like "./checkpoints/0.ckpt;./checkpoints/1.ckpt".
                                      (Default:"./ckpt/")
  --eval_file_name                    Eval output file.(Default:eval.og)
  --loss_file_name                    Loss output file.(Default:loss.log)
  --dataset_type                      The data type of the training files, chosen from [tfrecord, mindrecord, hd5].(Default:mindrecord)
  --vocab_cache_size                  Enable cache mode.(Default:0)
  --eval_while_train                  Whether to evaluate after training each epoch
```

### 预处理脚本参数

```markdown

used by: generate_synthetic_data.py

Arguments:
  --output_file                        The output path of the generated file.(Default: ./train.txt)
  --label_dim                          The label category. (Default:2)
  --number_examples                    The row numbers of the generated file. (Default:4000000)
  --dense_dim                          The number of the continue feature.(Default:13)
  --slot_dim                           The number of the category features.(Default:26)
  --vocabulary_size                    The vocabulary size of the total dataset.(Default:400000000)
  --random_slot_values                 0 or 1. If 1, the id is generated by the random. If 0, the id is set by the row_index mod
                                       part_size, where part_size is the vocab size for each slot
```

```markdown

usage: preprocess_data.py

  --preprocess_data_path              Where the origin sample data is put in (i.e. where the file origin_data is put in)
  --dense_dim                         The number of your continues fields.(default: 13)
  --slot_dim                          The number of your sparse fields, it can also be called category features.(default: 26)
  --threshold                         Word frequency below this value will be regarded as OOV. It aims to reduce the vocab size.(default: 100)
  --train_line_count                  The number of examples in your dataset.
  --skip_id_convert                   0 or 1. If set 1, the code will skip the id convert, regarding the original id as the final id.(default: 0)
  --eval_size                         The percent of eval samples in the whole dataset.
  --line_per_sample                   The number of sample per line, must be divisible by batch_size.
```

## 准备数据集

### 处理Criteo数据集

1. 下载数据集，并将其存放在某一路径下，例如./data/origin_data。

```bash
mkdir -p data/origin_data && cd data/origin_data
wget DATA_LINK
tar -zxvf dac.tar.gz
```

> 从[1](#数据集)获取下载链接。

2. 使用此脚本预处理数据。

```bash
python src/preprocess_data.py  --data_path=./data/ --dense_dim=13 --slot_dim=26 --threshold=100 --train_line_count=45840617 --skip_id_convert=0
```

### 处理合成数据集

1. 以下命令将会生成4000万行虚拟点击数据，格式如下：

> "label\tdense_feature[0]\tdense_feature[1]...\tsparse_feature[0]\tsparse_feature[1]...".

```bash
mkdir -p syn_data/origin_data
python src/generate_synthetic_data.py --output_file=syn_data/origin_data/train.txt --number_examples=40000000 --dense_dim=13 --slot_dim=51 --vocabulary_size=2000000000 --random_slot_values=0
```

2. 预处理生成数据。

```bash
python src/preprocess_data.py --data_path=./syn_data/  --dense_dim=13 --slot_dim=51 --threshold=0 --train_line_count=40000000 --skip_id_convert=1
```

## 训练过程

### 单机训练

运行如下命令训练模型：

```bash
python train.py --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU

# Or

bash ./script/run_train_gpu.sh './data/mindrecord/' 1 GPU False
```

## 评估过程

运行如下命令单独评估模型：

```bash
python eval.py --data_path=./data/mindrecord --dataset_type=mindrecord --device_target=GPU --ckpt_path=./ckpt/fibinet_train-10_41265.ckpt

# Or

bash ./script/run_eval_gpu.sh './data/mindrecord/' 1 GPU
```

## 推理过程

### [导出MindIR](#contents)

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，默认值："./ckpt/fibinet_train-10_41265.ckpt"；

`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中选择，默认值："MINDIR"。

### 模型表现

推理结果保存在脚本执行的当前路径，在eval_output.log中可以看到以下精度计算结果。

```markdown
auc :  0.7814143582416716
```

# 模型描述

## 性能

### 评估性能

| 计算框架               | MindSpore                                       |
| ---------------------- |-------------------------------------------------|
| 处理器                 | GPU                                             |
| 资源                 | A100-SXM4-40GB                                  |
| 上传日期            | 2022-07-29                                      |
| MindSpore版本        | 1.9                                           |
| 数据集                  | [1](#数据集)                                       |
| 训练参数      | Epoch=10,<br />batch_size=1000,<br />lr=0.0001 |
| 优化器                | FTRL,Adam                                       |
| 损失函数       | Sigmoid交叉熵                                      |
| AUC分数        |  0.7814143582416716                                         |
| 速度           | 15.588毫秒/步                                      |
| 损失           | 0.4702615                                          |
| 参数(M)           | 30                                              |
| 推理检查点 | 180MB（.ckpt文件）                               |

所有可执行脚本参见[此处](https://gitee.com/mindspore/models/tree/master/research/recommend/fibinet/script)。

# 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout算子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
