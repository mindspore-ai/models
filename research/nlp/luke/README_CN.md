# 目录

- [Luke概述](#Luke概述)
- [模型架构](#模型架构)
- [预训练模型](#预训练模型)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练过程](#训练过程)
- [评估](#评估)
- [导出过程](#导出过程)
- [性能](#性能)
- [ModelZoo主页](#modelzoo主页)

# Luke概述

LUKE (Language Understanding with Knowledge-based Embeddings)是一个新的预先训练好的基于转化器的词和实体的语境化表示。它在重要的NLP基准上取得了最先进的结果，包括SQuAD
v1.1（抽取式问题回答）、CoNLL-2003（命名实体识别）、ReCoRD（cloze-style问题回答）、TACRED（关系分类）和Open Entity（实体类型）。

论文:https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/2020.emnlp-main.523.pdf

源码：https://link.zhihu.com/?target=https%3A//github.com/studio-ousia/luke

# 模型架构

1.作者提出了一种新的专门用于处理与实体相关的任务的上下文表示方法LUKE(Language Understanding with Knowledge-based Embeddings)
。LUKE利用从维基百科中获得的大量实体注释语料库，预测随机mask的单词和实体。 2.作者提出了一种实体感知的自我注意机制，它是对transformer原有的注意机制的有效扩展，该机制在计算注意力分数时考虑到了标记（单词或实体）的类型。

# 预训练模型

下载luke-large

https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing

将其解压放入pre_luke/luke_large

修改convert_luke.py文件 model_type = 'luke-large'

运行python convert_luke.py 转换成ckpt

# 数据集

- squad数据集

squad1.1 https://data.deepai.org/squad1.1.zip

将其解压到suqad_data/squad

- wikipedia数据

https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing

将其解压到enwiki_dataset

数据文件和预训练文件的整体目录结构如下所示

```text
    │──squad_data
       │──squad                      # squadv1.1
          │──train-v1.1.json
          │──dev-v1.1.json
       │──squad_change               # 处理后的数据集
       │──mindrecord                 # 处理后的mindrecord
    │──pre_luke
       │──luke_large                 # luke预训练模型
          │──metadata.json
          │──entity_vocab.tsv
          │──luke.ckpt
    │──enwiki_dataset                # enwiki dataset
          │──enwiki_20160305_redirects.pkl
          │──enwiki_20160305.pkl
          │──enwiki_20181220_redirects.pkl
```

# 环境要求

- 硬件(Ascend)
    - 准备Ascend搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)

- 其他
    - ujson
    - marisa-trie
    - wikipedia2vec==1.0.5
    - transformers==2.3.0

- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

推荐使用绝对路径，运行脚本。

- 在Ascend处理器上运行

```bash
# 处理squad的数据集
bash scripts/run_squad_data_process.sh [DATA] [MODEL_FILE] [WIKIPEDIA]
# 运行训练示例
bash scripts/run_squad_standalone_train.sh [DATA] [MODEL_FILE]
# 运行分布式训练示例
bash scripts/run_squad_distribute_train.sh [RANK_TABLE] [DATA] [MODEL_FILE]
# 运行评估示例
bash scripts/run_squad_eval.sh [DATA] [MODEL_FILE] [CHECKPOINT_FILE]
```

```text
选项：
[DATA]              数据集地址(/home/ma-user/work/squad_data)
[MODEL_FILE]        预训练模型地址(/home/ma-user/work/pre_luke/luke_large)
[WIKIPEDIA]         wikipedia数据地址(/home/ma-user/work/enwiki_dataset)
[CHECKPOINT_FILE]   训练出来的模型地址(/home/ma-user/work/output/luke_squad-2_10973.ckpt)
[RANK_TABLE]        hccl配置文件
```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。 请遵循以下链接中的说明：
<https://gitee.com/mindspore/models/tree/r1.5/utils/hccl_tools>

- 在ModelArt上使用8卡训练

```text
# (1) 上传你的代码和转好的luke模型到 s3 桶上
# (2) 在ModelArts上创建训练任务
# (3) 选择代码目录 /{path}/luke
# (4) 选择启动文件 /{path}/luke/run_squad_train.py
# (5) 执行a或b
#     a. 在 pretrain_main 文件中设置参数
#         1. 设置 ”modelArts=True“
#         3. 设置 “checkpoint_url=luke预训练模型的位置”
#         2. 设置其它参数，其它参数配置可以参考 `src/model_utils/config.yaml`
#     b. 在 网页上设置
#         1. 添加 ”modelArts = True“
#         2. 添加 ”checkpoint_url = obs://jeffery/luke_pre/luke_large/“
#         3. 添加其它参数，其它参数配置可以参考 `src/model_utils/config.yaml`
# (6) 上传你的数据、到s3桶上
# (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
# (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
# (9) 在网页上的’资源池选择‘项目下， 选择8卡规格的资源
# (10) 创建训练作业
# 训练结束后会在'训练输出文件路径'下保存训练的权重
```

# 脚本说明

## 脚本及样例代码

```text
|--Luke
    |--scripts
    |      |--run_squad_data_process.sh         # 处理squad数据集脚本
    |      |--run_squad_eval.sh                 # squad验证脚本
    |      |--run_squad_standalone_train.sh     # squad训练脚本
    |      |--run_squad_distribute_train.sh     # 分布式squad训练脚本
    |--src
    |      |--luke                              # luke模型相关文件
    |      |      |--config.py                  # 配置文件
    |      |      |--model.py                   # luke模型文件
    |      |      |--roberta.py                 # roberta文件
    |      |--model_utils
    |      |      |--local_adapter.py           # 本地适配
    |      |      |--moxing_adapter.py          # mox上传
    |      |      |--config.yaml                # 配置文件
    |      |      |--config_args.py             # config读取文件
    |      |--reading_comprehension             # 阅读理解任务
    |      |      |--dataLoader.py              # 数据加载文件
    |      |      |--dataProcessing.py          # 数据处理文件
    |      |      |--dataset.py                 # 数据集文件
    |      |      |--model.py                   # 模型文件
    |      |      |--squad_get_predictions.py   # squad得到预测文件
    |      |      |--squad_postprocess.py       # squad处理文件
    |      |      |--train.py                   # squad 训练文件
    |      |      |--wiki_link_db.py
    |      |--utils   # 工具包
    |      |      |--entity_vocab.py            # entity vocab文件
    |      |      |--interwiki_db.py            # wiki db文件
    |      |      |--model_utils.py             # 模型工具包
    |      |      |--sentence_tokenizer.py      # 句子分词器
    |      |      |--utils.py                   # 工具包
    |      |      |--word_tokenizer.py          # 单词分词
    |--convert_luke.py                          # 转换luke模型
    |--create_squad_data.py                     # 创建squad数据集
    |--export.py                                # 导出脚本
    |--README_CN.md
    |--requirements.txt
    |--run_squad_eval.py                        # eval squad
    |--run_squad_train.py                       # run squad
```

## 脚本参数

可在run_squad_train.py中修改参数。

```text
train_batch_size                输入数据集的批次大小
num_train_epochs                epoch的数量
decay_steps                     学习率开始衰减的步数
learning_rate                   学习率
warmup_proportion               热身学习比率
weight_decay                    权重衰减
eps                             增加分母，提高小数稳定性
with_negative                   数据集是squad1.1,还是squad2.0
seed                            初始化种子
data                            数据集位置
其他参数详见src/model_utils/config.yaml
```

# 训练过程

数据处理

```bash
python create_squad_data.py --data ./squad_data --model_file ./pre_luke/luke_large/ --wikipedia ./enwiki_dataset
```

训练

- Ascend处理器上运行

```bash
python run_squad_train.py --warmup_proportion 0.09 --num_train_epochs 2 --model_file ./pre_luke/luke_large --data ./squad_data --train_batch_size 8 --learning_rate 12e-6 --dataset_sink_mode False
```

```bash
选项：
--model_file              预训练模型地址
--data                    数据集地址
--wikipedia               wikipedia数据地址
--warmup_proportion       热身学习比率
--num_train_epochs        epoch数量
--train_batch_size        bach size数量
--dataset_sink_mode       数据下沉模式
```

modelarts

可以查看快速入门

# 验证

将训练的ckpt 放入output

启动脚本

```bash
python run_squad_eval.py --data ./squad_data --eval_batch_size 8  --checkpoint_file ./output/luke_squad-2_10973.ckpt --model_file ./pre_luke/luke_large/
```

其中checkpoint_file指的是训练出模型的位置

结果

```text
单卡: {"exact_match": 89.50804162724693, "f1": 95.01586892223338}
多卡: {"exact_match": 89.33774834437087, "f1": 94.89125713127684}
```

# 导出过程

## 用法

- Ascend处理器环境运行

```bash
python export.py --export_batch_size 8 --model_file ./pre_luke/luke_large/ --checkpoint_file ./output/luke_squad-2_10973.ckpt
```

# 性能

## 训练性能

| 参数       | Ascend                                                    |
| ---------- | --------------------------------------------------------- |
| 模型       | Luke                                                     |
| 资源       | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |
| 上传日期   | 2022-01-28                                                |
| 数据集     | SQuAD                                                     |
| 训练参数   | src/model_utils/config.yaml                                            |
| 学习率     | 12e-6                                                      |
| 优化器     | AdamWeightDecay                                                      |
| 损失函数   | SoftmaxCrossEntropy                                       |
| 轮次       | 2                                                         |
| Batch_size | 2*8                                                       |
| 损失       | 0.8173281                                                |
| 速度       | 359毫秒/步                                                |
| 总时长     | 1.5小时                                                     |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。