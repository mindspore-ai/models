# 目录

[View English](./README.md)

- [目录](#目录)
- [Soft-Masked BERT](#Soft-MaskedBERT)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [贡献指南](#贡献指南)
    - [贡献者](#贡献者)
- [ModelZoo主页](#ModelZoo主页)

<TOC>

# Soft-Masked BERT
[论文](https://arxiv.org/pdf/2005.07421v1.pdf)：Zhang S, Huang H, Liu J, et al. Spelling error correction with soft-masked BERT[J]. arXiv preprint arXiv:2005.07421, 2020.

# 模型架构

Soft-Masked BERT由一个基于Bi-GRU的检测网络和一个基于BERT的校正网络组成。检测网络预测误差的概率，修正网络预测误差修正的概率，而检测网络利用软掩蔽将预测结果传递给修正网络。

# 数据集

1. 下载[SIGHAN数据集](http://nlp.ee.ncu.edu.tw/resource/csc.html)
1. 解压上述数据集并将文件夹中所有 ''.sgml'' 文件复制至 datasets/csc/ 目录
1. 复制 ''SIGHAN15_CSC_TestInput.txt'' 和 ''SIGHAN15_CSC_TestTruth.txt'' 至 datasets/csc/ 目录
1. [下载](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)至datasets/csc 目录
1. 请确保以下文件在 datasets/csc 中

```text
train.sgml
B1_training.sgml
C1_training.sgml  
SIGHAN15_CSC_A2_Training.sgml  
SIGHAN15_CSC_B2_Training.sgml  
SIGHAN15_CSC_TestInput.txt
SIGHAN15_CSC_TestTruth.txt
```

6. 对数据进行预处理(运行脚本所需要的依赖包请参考requirement.txt安装)

```python
python preprocess_dataset.py
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 依赖
    - 安装所需依赖 pip install -r requirements.txt
- 版本问题
    - 如果出现报错GLIBC版本过低的问题，可以将openCC改为安装较低版本（例如 1.1.0）

# 快速入门

1. 将预处理后数据放在datasets目录。
2. 下载[bert-base-chinese-vocab.txt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt)，放在src/文件夹中。
3. 下载[预训练模型](https://download.mindspore.cn/models/r1.3/bertbase_ascend_v130_cnnews128_official_nlp_loss1.5.ckpt)，放入weight/文件夹。
4. 执行训练脚本。
- 在线下服务器进行训练

```python
# 分布式训练
bash scripts/run_distribute_train.sh [RANK_SIZE] [RANK_START_ID] [RANK_TABLE_FILE] [BERT_CKPT]
#BERT_CKPT：预训练BERT文件名（例如bert_base.ckpt）

# 单机训练
bash scripts/run_standalone_train.sh [BERT_CKPT] [DEVICE_ID] [PYNATIVE]
#BERT_CKPT：预训练BERT文件名（例如bert_base.ckpt）
#DEVICE_ID：运行的机器id
#PYNATIVE：是否使用pynative模式运行（默认False）
```

- 在OpenI进行训练

```text
# (1) 进入[代码仓](https://git.openi.org.cn/OpenModelZoo/SoftMaskedBert),新建训练任务。
# (2) 在网页上设置 "enable_modelarts=True; bert_ckpt=bert_base.ckpt"
# (3) 如果按pynative模式运行，则在网页上设置 "pynative=True"
# (4) 在网页上设置数据集 "SoftMask.zip"
# (5) 在网页上设置启动文件为 "train.py"
# (6) 运行训练作业
```

5. 执行评估脚本。

训练结束后，按照如下步骤启动评估：

```python
# 评估
bash scripts/run_eval.sh [BERT_CKPT_NAME] [CKPT_DIR]
```

# 脚本说明

```text
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── soft-maksed-bert
        ├── README.md                    // softmasked-BERT相关说明
        ├── README_CN.md             // softmasked-BERT中文版相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_distribute_train.sh             // Ascend分布式训练的shell脚本
        │   ├──run_standalone_train.sh          // Ascend单机训练的shell脚本
        │   ├──run_eval.sh                  // Ascend评估的shell脚本
        │   ├──run_infer_310.sh         // Ascend推理shell脚本
        │   ├──run_preprocess.sh      // 运行数据预处理的shell脚本
        ├── src
        │   ├──soft_masked_bert.py          //  soft-maksed bert架构
        │   ├──bert_model.py                    //  BERT架构
        │   ├──dataset.py                          //   数据集处理
        │   ├──finetune_config.py            //   模型超参数
        │   ├──gru.py                                //   GRU架构
        │   ├──tokenization.py                 //   单词分割
        │   ├──util.py                                //   工具
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── postprogress.py       // 310推理后处理脚本
        ├── export.py            // 将checkpoint文件导出
        ├── preprocess_dataset.py            // 数据预处理
```

## 脚本参数

```python
'batch size':36    # batch大小
'epoch':100         # 总计训练epoch数
'learning rate':0.0001            # 初始学习率
'loss function':'BCELoss'        # 训练采用的损失函数
'optimizer':AdamWeightDecay           # 激活函数
```

## 训练过程

### 单机训练

- Ascend处理器环境运行

  ```python
  bash scripts/run_standalone_train.sh [BERT_CKPT] [DEVICE_ID] [PYNATIVE]
  ```

  训练结束后，您可在默认脚本文件夹下找到检查点文件。运行过程如下：

  ```bash
  epoch: 1 step: 152, loss is 3.3235654830932617
  epoch: 1 step: 153, loss is 3.6958463191986084
  epoch: 1 step: 154, loss is 3.585498571395874
  epoch: 1 step: 155, loss is 3.276094913482666
  ...
  ```

### 分布式训练

- Ascend处理器环境运行

  ```python
  bash run_distribute_train.sh [RANK_SIZE] [RANK_START_ID] [RANK_TABLE_FILE] [BERT_CKPT]
  ```

  上述shell脚本将在后台运行分布训练。

  ```bash
  epoch: 1 step: 12, loss is 7.957302093505859
  epoch: 1 step: 13, loss is 7.886098861694336
  epoch: 1 step: 14, loss is 7.781495094299316
  epoch: 1 step: 15, loss is 7.755488395690918
  ...
  ...
  ```

## 推理

### 推理过程

在执行推理之前，需要通过export.py导出mindir文件。输入数据文件为bin格式。

```python
# 导出mindir文件

python export.py --bert_ckpt [BERT_CKPT] --ckpt_dir [CKPT_DIR]

# Ascend310 推理

bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`BERT_CKPT`为必选项, 预训练BERT文件名（例如bert_base.ckpt）
`CKPT_DIR`为必选项, 训练好ckpt的路径 (例如./checkpoint/SoftMaskedBert-100_874.ckpt)
`MINDIR_PATH` 为必选项, 表示模型文件的目录。
`DATA_FILE_PATH` 为必选项, 表示输入数据的目录。
`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`DEVICE_ID` 可选，默认值为 0。

### 推理结果

推理结果保存在项目主目录下，可在acc.log中看到最终精度结果。

```eval log
1 The detection result is precision=0.6733436055469953, recall=0.6181046676096181 and F1=0.6445427728613569
2 The correction result is precision=0.8260869565217391, recall=0.7234468937875751 and F1=0.7713675213675213
3 Sentence Level: acc:0.606364, precision:0.650970, recall:0.433579, f1:0.520487
```

# 模型描述

## 性能

### 训练性能

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | BERT-base                                                |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             |
| 上传日期              | 2022-06-28                                 |
| MindSpore版本          | 1.6.0                                                       |
| 数据集                    | SIGHAN                                                    |
| 训练参数        | epoch=100, steps=6994, batch_size = 36, lr=0.0001              |
| 优化器                  | AdamWeightDecay                                                    |
| 损失函数              | BCELoss                                       |
| 损失                       | 0.0016                                                      |
| 速度                      | 单卡：349.7毫秒/步;  8卡：314.7毫秒/步                          |
| 总时长                 | 单卡：4076分钟;  8卡：458分钟                          |
| 微调检查点 | 459M (.ckpt文件)                                         |
| 脚本                    | [Soft-Masked BERT脚本](https://gitee.com/rafeal8830/soft-maksed-bert/edit/master/README_TEMPLATE_CN.md) |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.7.0                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

# 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## 贡献者

* [c34](https://gitee.com/c_34) (Huawei)

# ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。