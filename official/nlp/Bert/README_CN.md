﻿# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [BERT概述](#bert概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [预训练模型](#预训练模型)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [预训练](#预训练)
        - [微调与评估](#微调与评估)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
        - [schema\_file](#schema_file)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估tnews数据集](#ascend处理器上运行后评估tnews数据集)
            - [CPU处理器上运行后评估tnews数据集](#cpu处理器上运行后评估tnews数据集)
            - [Ascend处理器上运行后评估cluener数据集](#ascend处理器上运行后评估cluener数据集)
            - [CPU处理器上运行后评估cluener数据集](#cpu处理器上运行后评估cluener数据集)
            - [Ascend处理器上运行后评估chineseNer数据集](#ascend处理器上运行后评估chinesener数据集)
            - [CPU处理器上运行后评估chineseNer数据集](#cpu处理器上运行后评估chinesener数据集)
            - [Ascend处理器上运行后评估msra数据集](#ascend处理器上运行后评估msra数据集)
            - [Ascend处理器上运行后评估squad v1.1数据集](#ascend处理器上运行后评估squad-v11数据集)
            - [CPU处理器上运行后评估squad v1.1数据集](#cpu处理器上运行后评估squad-v11数据集)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法-2)
        - [结果](#结果)
    - [导出onnx模型与推理](#导出onnx模型与推理)
    - [模型描述](#模型描述)
    - [性能](#性能)
        - [预训练性能](#预训练性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
- [FAQ](#faq)

<!-- /TOC -->

# BERT概述

BERT网络由谷歌在2018年提出，该网络在自然语言处理领域取得了突破性进展。采用预训练技术，实现大的网络结构，并且仅通过增加输出层，实现多个基于文本的任务的微调。BERT的主干代码采用Transformer的Encoder结构。引入注意力机制，使输出层能够捕获高纬度的全局语义信息。预训练采用去噪和自编码任务，即掩码语言模型（MLM）和相邻句子判断（NSP）。无需标注数据，可对海量文本数据进行预训练，仅需少量数据做微调的下游任务，可获得良好效果。BERT所建立的预训练加微调的模式在后续的NLP网络中得到了广泛的应用。

[论文](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.[BERT：深度双向Transformer语言理解预训练](https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[论文](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu.[NEZHA：面向汉语理解的神经语境表示](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.

# 模型架构

BERT的主干结构为Transformer。对于BERT_base，Transformer包含12个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。对于BERT_NEZHA，Transformer包含24个编码器模块，每个模块包含一个自注意模块，每个自注意模块包含一个注意模块。BERT_base和BERT_NEZHA的区别在于，BERT_base使用绝对位置编码生成位置嵌入向量，而BERT_NEZHA使用相对位置编码。

# 数据集

- 生成预训练数据集
    - 下载[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练，
    - 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，使用步骤如下：
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - `WikiExtarctor`提取出来的原始文本并不能直接使用，还需要将数据集预处理并转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert#pre-training-with-bert)代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。
- 生成下游任务数据集
    - 下载数据集进行微调和评估，如中文实体识别任务[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)、中文文本分类任务[TNEWS](https://github.com/CLUEbenchmark/CLUE)、中文实体识别任务[ChineseNER](https://github.com/zjy-ucas/ChineseNER)、英文问答任务[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)、[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)、英文分类任务集合[GLUE](https://gluebenchmark.com/tasks)等。
    - 将数据集文件从JSON格式转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert)代码仓中的run_classifier.py或run_squad.py文件。
- 生成MindRecord数据集
    - 生成预训练mindrecord数据集
        - 如果已按上面步骤下载原始预训练数据集，并使用WikiExtractor提取文本数据，你可以按以下操作获取对应的mindrecord数据集

        ```bash
           bash ./generate_pretrain_mindrecords.sh INPUT_FILES_PATH OUTPUT_FILES_PATH VOCAB_FILE
           比如:
           bash ./generate_pretrain_mindrecords.sh /path/wiki-clean-aa /path/output/ /path/bert-base-uncased-vocab.txt
        ```

        - 如果已将json格式的数据转换为tfrecord数据集，你也可以通过以下方式将tfrecord转换成对应的mindrecord格式

        ```python
           python parallel_tfrecord_to_mindrecord.py --input_tfrecord_dir /path/tfrecords_path --output_mindrecord_dir /path/save_mindrecord_path
        ```

        - 同时也可以按以下操作对tfrecord或mindrecord数据进行可视化

        ```python
            python vis_tfrecord_or_mindrecord.py --file_name /path/train.mindrecord --vis_option vis_mindrecord > mindrecord.txt
            `vis_option` 需要从["vis_tfrecord", "vis_mindrecord"]中选择
            注：在执行之前，需要确保需要已安装tensorflow==1.15.0
        ```

    - 为ner下游任务生成CLUENER和ChineseNER mindrecord数据集
        在生成mindrecord数据集之前，你需要按以上指导下载下游任务对应的CLUENER及ChineseNER原始数据集及[vocab.txt](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/tf_version/vocab.txt)
        - 生成ner下游任务：CLUENER数据集的mindrecord数据集

          ```python
             python generate_cluener_mindrecord.py --data_dir /path/ClueNER/cluener_public/ --vocab_file /path/vocab.txt --output_dir /path/ClueNER/
          ```

        - 生成ner下游任务：ChineseNER数据集的mindrecord数据集

          ```python
             python generate_chinese_mindrecord.py --data_dir /path/ChineseNER/data/ --vocab_file /path/vocab.txt --output_dir /path/ChineseNER/
          ```

        - 生成ner下游任务：ChineseNER数据集在CPU上的mindrecord数据集

          ```python
             python generate_chinese_mindrecord.py --data_dir /path/ChineseNER/data/ --vocab_file /path/vocab.txt --output_dir /path/ChineseNER/ --max_seq_length 128
          ```

    - 为squad下游任务生成SquadV1.1 mindrecord数据集
      在生成mindrecord数据集之前，你需要按以上指导下载下游任务对应的SquadV1.1原始数据集及[vocab.txt](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/pretrained_model/uncased_L-12_H-768_A-12/vocab.txt)
      - 生成squad下游任务：SquadV1.1数据集的mindrecord数据集

          ```python
             python generate_squad_mindrecord.py --vocab_file /path/squad/vocab.txt --train_file /path/squad/train-v1.1.json --predict_file /path/squad/dev-v1.1.json --output_dir /path/squad
          ```

    - 为classifier下游任务生成tnews mindrecord数据集
      在生成mindrecord数据集之前，你需要按以上指导下载下游任务对应的tnews原始数据集及[vocab.txt](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/tf_version/vocab.txt)
      - 生成classifier下游任务：tnews数据集的mindrecord数据集

          ```python
             python generate_tnews_mindrecord.py --data_dir /path/tnews/ --task_name tnews --vocab_file /path/tnews/vocab.txt --output_dir /path/tnews
          ```

# 预训练模型

我们提供了一些预训练权重以供使用

- [Bert-base-zh](https://download.mindspore.cn/models/r1.9/bert_base_ascend_v190_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36.ckpt), 在128句长的中文wiki数据集上进行了训练
- [Bert-large-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_zhwiki_official_nlp_bs3072_loss0.8/), 在128句长的中文wiki数据集上进行了训练
- [Bert-large-en](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1/), 在512句长的英文wiki数据集上进行了训练

# 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend或GPU或CPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

```bash

# 单机运行预训练示例

bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例

bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json

# 单独运行训练评估脚本示例

- GE流程暂不支持边训边推，MLM任务可以单独加载训练ckpt后执行评估脚本
- 修改对应yaml配置文件中`eval_ckpt` and `eval_data_dir`加载路径
bash scripts/run_pretrain_eval_ascend.sh

# 运行微调和评估示例

- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`task_[DOWNSTREAM_TASK]_config.yaml`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier.sh DEVICE_ID(optional)

- NER任务：在scripts/run_ner.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner.sh DEVICE_ID(optional)

- SQUAD任务：在scripts/run_squad.sh中设置任务相关的超参。
-运行`bash scripts/run_squad.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad.sh DEVICE_ID(optional)
```

- 在GPU上运行

```bash

# 单机运行预训练示例

bash run_standalone_pretrain_for_gpu.sh 0 1 /path/cn-wiki-128

# 分布式运行预训练示例

bash scripts/run_distributed_pretrain_for_gpu.sh 8 40 /path/cn-wiki-128

# 运行微调和评估示例

- 如需运行微调任务，请先准备预训练生成的权重文件（ckpt）。
- 在`task_[DOWNSTREAM_TASK]_config.yaml`中设置BERT网络配置和优化器超参。

- 分类任务：在scripts/run_classifier_gpu.sh中设置任务相关的超参。
- 运行`bash scripts/run_classifier_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_classifier_gpu.sh DEVICE_ID(optional)

- NER任务：在scripts/run_ner_gpu.sh中设置任务相关的超参。
- 运行`bash scripts/run_ner_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_ner_gpu.sh DEVICE_ID(optional)

- SQUAD任务：在scripts/run_squad_gpu.sh中设置任务相关的超参。
-运行`bash scripts/run_squad_gpu.sh [DEVICE_ID]`，对BERT-base和BERT-NEZHA模型进行微调。

  bash scripts/run_squad_gpu.sh DEVICE_ID(optional)
```

- 在ModelArts上运行(如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在ModelArt上使用8卡预训练

    ```python
    # (1) 上传你的代码到 s3 桶上
    # (2) 在ModelArts上创建训练任务
    # (3) 选择代码目录 /{path}/bert
    # (4) 选择启动文件 /{path}/bert/run_pretrain.py
    # (5) 执行a或b
    #     a. 在 /{path}/bert/default_config.yaml 文件中设置参数
    #         1. 设置 ”enable_modelarts=True“
    #         2. 设置其它参数，其它参数配置可以参考 `./scripts/run_distributed_pretrain_ascend.sh`
    #     b. 在 网页上设置
    #         1. 添加 ”run_distributed=True“
    #         2. 添加其它参数，其它参数配置可以参考 `./scripts/run_distributed_pretrain_ascend.sh`
    # (6) 上传你的 数据 到 s3 桶上
    # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
    # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
    # (9) 在网页上的’资源池选择‘项目下， 选择8卡规格的资源
    # (10) 创建训练作业
    # 训练结束后会在'训练输出文件路径'下保存训练的权重
    ```

    - 在ModelArts上使用单卡运行下游任务

    ```python
    # (1) 上传你的代码到 s3 桶上
    # (2) 在ModelArts上创建训练任务
    # (3) 选择代码目录 /{path}/bert
    # (4) 选择启动文件 /{path}/bert/run_ner.py(或 run_squad.py 或 run_classifier.py)
    # (5) 执行a或b
    #     a. 在 /path/bert 下的`task_ner_config.yaml`(或 `task_squad_config.yaml` 或 `task_classifier_config.yaml`) 文件中设置参数
    #         1. 设置 ”enable_modelarts=True“
    #         2. 设置其它参数，其它参数配置可以参考 './scripts/'下的 `run_ner.sh`或`run_squad.sh`或`run_classifier.sh`
    #     b. 在 网页上设置
    #         1. 添加 ”enable_modelarts=True“
    #         2. 添加其它参数，其它参数配置可以参考 './scripts/'下的 `run_ner.sh`或`run_squad.sh`或`run_classifier.sh`
    #     注意vocab_file_path，label_file_path，train_data_file_path，eval_data_file_path，schema_file_path填写相对于第7步所选路径的相对路径。
    #     最后必须在网页上添加 “config_path=/path/*.yaml”(根据下游任务选择 *.yaml 配置文件)
    # (6) 上传你的 数据 到 s3 桶上
    # (7) 在网页上勾选数据存储位置，设置“训练数据集”路径（该路径下仅有 数据/数据zip压缩包）
    # (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
    # (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
    # (10) 创建训练作业
    # 训练结束后会在'训练输出文件路径'下保存训练的权重
    ```

在Ascend设备上做分布式训练时，请提前创建JSON格式的HCCL配置文件。

在Ascend设备上做单机分布式训练时，请参考[hccl_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)创建HCCL配置文件。

在Ascend设备上做多机分布式训练时，训练命令需要在很短的时间间隔内在各台设备上执行。因此，每台设备上都需要准备HCCL配置文件。请参考[merge_hccl](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools#merge_hccl)创建多机的HCCL配置文件。

## 脚本说明

## 脚本和样例代码

```shell
.
└─bert
  ├─ascend310_infer
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─ascend_distributed_launcher
        ├─__init__.py
        ├─hyper_parameter_config.ini          # 分布式预训练超参
        ├─get_distribute_pretrain_cmd.py      # 分布式预训练脚本
        --README.md
    ├─run_classifier.sh                       # Ascend设备上单机分类器任务shell脚本
    ├─run_classifier_gpu.sh                   # GPU设备上单机分类器任务shell脚本
    ├─run_ner.sh                              # Ascend设备上单机NER任务shell脚本
    ├─run_ner.sh                              # GPU设备上单机NER任务shell脚本
    ├─run_squad.sh                            # Ascend设备上单机SQUAD任务shell脚本
    ├─run_squad_gpu.sh                        # GPU设备上单机SQUAD任务shell脚本
    ├─run_standalone_pretrain_ascend.sh       # Ascend设备上单机预训练shell脚本
    ├─run_distributed_pretrain_ascend.sh      # Ascend设备上分布式预训练shell脚本
    ├─run_distributed_pretrain_gpu.sh         # GPU设备上分布式预训练shell脚本
    └─run_standaloned_pretrain_gpu.sh         # GPU设备上单机预训练shell脚本
  ├─src
    ├─generate_mindrecord
      ├── generate_chinesener_mindrecord.py   # 为ner下游任务产生ChineseNER对应的mindrecord数据集
      ├── generate_cluener_mindrecord.py      # 为ner下游任务产生CLUENER对应的mindrecord数据集
      ├── generate_pretrain_mindrecord.py     # 为预训练产生预训练mindrecord数据集
      ├── generate_pretrain_mindrecords.sh    # 并行调用generate_pretrain_mindrecord.py产生预训练mindrecord数据集
      ├── generate_squad_mindrecord.py        # 为squad下游任务产生SquadV1.1对应的mindrecord数据集
      └── generate_tnews_mindrecord.py        # 为classifier下游任务产生tnews对应的mindrecord数据集
    ├─model_utils
      ├── config.py                           # 解析 *.yaml参数配置文件
      ├── devcie_adapter.py                   # 区分本地/ModelArts训练
      ├── local_adapter.py                    # 本地训练获取相关环境变量
      └── moxing_adapter.py                   # ModelArts训练获取相关环境变量、交换数据
    ├─tools
      ├── parallel_tfrecord_to_mindrecord.py  # 多线程池将tfrecord数据集转换成mindrecord数据集
      └── vis_tfrecord_or_mindrecord.py       # 可视化tfrecord或mindrecord数据集
    ├─__init__.py
    ├─assessment_method.py                    # 评估过程的测评方法
    ├─bert_for_finetune.py                    # 网络骨干编码
    ├─bert_for_finetune_cpu.py                # 网络骨干编码
    ├─bert_for_pre_training.py                # 网络骨干编码
    ├─bert_model.py                           # 网络骨干编码
    ├─finetune_data_preprocess.py             # 数据预处理
    ├─cluner_evaluation.py                    # 评估线索生成工具
    ├─cluner_evaluation_cpu.py                # 评估线索生成工具
    ├─CRF.py                                  # 线索数据集评估方法
    ├─dataset.py                              # 数据预处理
    ├─finetune_eval_model.py                  # 网络骨干编码
    ├─sample_process.py                       # 样例处理
    ├─utils.py                                # util函数
  ├─pretrain_config.yaml                      # 预训练参数配置
  ├─task_ner_config.yaml                      # 下游任务_ner 参数配置
  ├─task_ner_cpu_config.yaml                  # 下游任务_ner 参数配置
  ├─task_classifier_config.yaml               # 下游任务_classifier 参数配置
  ├─task_classifier_cpu_config.yaml           # 下游任务_classifier 参数配置
  ├─task_squad_config.yaml                    # 下游任务_squad 参数配置
  ├─pretrain_eval.py                          # 训练和评估网络
  ├─run_classifier.py                         # 分类器任务的微调和评估网络
  ├─run_ner.py                                # NER任务的微调和评估网络
  ├─run_pretrain.py                           # 预训练网络
  └─run_squad.py                              # SQUAD任务的微调和评估网络
```

## 脚本参数

### 预训练

```shell
用法：run_pretrain.py  [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                        [--enable_save_ckpt ENABLE_SAVE_CKPT] [--device_target DEVICE_TARGET]
                        [--enable_lossscale ENABLE_LOSSSCALE] [--do_shuffle DO_SHUFFLE]
                        [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                        [--accumulation_steps N]
                        [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                        [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                        [--save_checkpoint_steps N] [--save_checkpoint_num N]
                        [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [train_steps N]

选项：
    --device_target            代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --distribute               是否多卡预训练，可选项为true（多卡预训练）或false。默认为false
    --epoch_size               轮次，默认为1
    --device_num               使用设备数量，默认为1
    --device_id                设备ID，默认为0
    --enable_save_ckpt         是否使能保存检查点，可选项为true或false，默认为true
    --enable_lossscale         是否使能损失放大，可选项为true或false，默认为true
    --do_shuffle               是否使能轮换，可选项为true或false，默认为true
    --enable_data_sink         是否使能数据下沉，可选项为true或false，默认为true
    --data_sink_steps          设置数据下沉步数，默认为1
    --accumulation_steps       更新权重前梯度累加数，默认为1
    --save_checkpoint_path     保存检查点文件的路径，默认为""
    --load_checkpoint_path     加载检查点文件的路径，默认为""
    --save_checkpoint_steps    保存检查点文件的步数，默认为1000
    --save_checkpoint_num      保存的检查点文件数量，默认为1
    --train_steps              训练步数，默认为-1
    --data_dir                 数据目录，默认为""
    --dataset_format           数据集格式，支持tfrecord和mindrecord，默认为mindrecord
    --schema_dir               schema.json的路径，默认为""，当 train_data_file_path 为TFRecord时，
                               需要指定 schema_file，MindRecord则无需指定。
```

### 微调与评估

```shell
用法：run_ner.py   [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--assessment_method ASSESSMENT_METHOD] [--use_crf USE_CRF] [--with_lstm WITH_LSTM]
                    [--device_id N] [--epoch_num N] [--vocab_file_path VOCAB_FILE_PATH]
                    [--label2id_file_path LABEL2ID_FILE_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
选项：
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为f1或clue_benchmark
    --use_crf                         是否采用CRF来计算损失，可选项为true或false
    --with_lstm                       是否在bert后接lstm子网络提升性能，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --label2id_file_path              标注文件，文件中的标注名称必须与原始数据集中所标注的类型名称完全一致
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             如采用f1来评估结果，则为TFRecord文件保存预测；如采用clue_benchmark来评估结果，则为JSON文件保存预测
    --dataset_format                  数据集格式，支持tfrecord和mindrecord格式，默认为mindrecord
    --schema_file_path                模式文件保存路径，当 train_data_file_path 为TFRecord时，需要指定 schema_file,
                                      MindRecord则无需指定。

用法：run_squad.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                    [--device_id N] [--epoch_num N] [--num_class N]
                    [--vocab_file_path VOCAB_FILE_PATH]
                    [--eval_json_path EVAL_JSON_PATH]
                    [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                    [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                    [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                    [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                    [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                    [--train_data_file_path TRAIN_DATA_FILE_PATH]
                    [--eval_data_file_path EVAL_DATA_FILE_PATH]
                    [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       分类数，SQuAD任务通常为2
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --vocab_file_path                 BERT模型训练的词汇表
    --eval_json_path                  保存SQuAD任务开发JSON文件的路径
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存SQuAD训练数据的TFRecord文件，如train1.1.tfrecord
    --eval_data_file_path             用于保存SQuAD预测数据的TFRecord文件，如dev1.1.tfrecord
    --schema_file_path                模式文件保存路径，当 train_data_file_path 为TFRecord时，需要指定 schema_file,
                                      MindRecord则无需指定。

usage: run_classifier.py [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [----do_eval DO_EVAL]
                         [--assessment_method ASSESSMENT_METHOD] [--device_id N] [--epoch_num N] [--num_class N]
                         [--save_finetune_checkpoint_path SAVE_FINETUNE_CHECKPOINT_PATH]
                         [--load_pretrain_checkpoint_path LOAD_PRETRAIN_CHECKPOINT_PATH]
                         [--load_finetune_checkpoint_path LOAD_FINETUNE_CHECKPOINT_PATH]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
                         [--schema_file_path SCHEMA_FILE_PATH]
options:
    --device_target                   任务运行的目标设备，可选项为Ascend或GPU
    --do_train                        是否基于训练集开始训练，可选项为true或false
    --do_eval                         是否基于开发集开始评估，可选项为true或false
    --assessment_method               评估方法，可选项为accuracy、f1、mcc、spearman_correlation
    --device_id                       任务运行的设备ID
    --epoch_num                       训练轮次总数
    --num_class                       标注类的数量
    --train_data_shuffle              是否使能训练数据集轮换，默认为true
    --eval_data_shuffle               是否使能评估数据集轮换，默认为true
    --save_finetune_checkpoint_path   保存生成微调检查点的路径
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型）
    --load_finetune_checkpoint_path   如仅执行评估，提供微调检查点保存路径
    --train_data_file_path            用于保存训练数据的TFRecord文件，如train.tfrecord文件
    --eval_data_file_path             用于保存预测数据的TFRecord文件，如dev.tfrecord
    --dataset_format                  数据集格式，支持tfrecord和mindrecord，默认为mindrecord
    --schema_file_path                模式文件保存路径，当 train_data_file_path 为TFRecord时，需要指定 schema_file,
                                      MindRecord则无需指定。
```

## 选项及参数

可以在yaml配置文件中分别配置预训练和下游任务的参数。

### 选项

```text
config for lossscale and etc.
    bert_network                    BERT模型版本，可选项为base或nezha，默认为base
    batch_size                      输入数据集的批次大小，默认为32
    loss_scale_value                损失放大初始值，默认为2^32
    scale_factor                    损失放大的更新因子，默认为2
    scale_window                    损失放大的一次更新步数，默认为1000
    optimizer                       网络中采用的优化器，可选项为AdamWerigtDecayDynamicLR、Lamb、或Momentum，默认为Lamb
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为128
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同。默认为21128。根据论文，一般对于中文数据集，取值为21128，英文数据集为30522。
    hidden_size                     BERT的encoder层数，默认为768
    num_hidden_layers               隐藏层数，默认为12
    num_attention_heads             注意头的数量，默认为12
    intermediate_size               中间层数，默认为3072
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    BERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减

    Momentum:
    learning_rate                   学习率
    momentum                        平均移动动量
```

### schema_file

当使用TFRecord作为训练、评估数据时，需要用到`schema_file`这一数据，该数据需要自行生成，其具体含义如下：

对于预训练，schema文件包含 `["input_ids"、"input_mask"、"segment_ids"、"next_sentence_labels"、"masked_lm_positions"、"masked_lm_ids"、"masked_lm_weights"]`。

对于 ner 或分类任务，schema文件包含 `["input_ids"、"input_mask"、"segment_ids"、"label_ids"]`。

对于squad任务，训练用到的schema文件包含`["start_positions"、"end_positions"、"input_ids"、"input_mask"、"segment_ids"]`，评估用到的schema文件包含`["input_ids"、"input_mask"、"segment_ids"]`。

当 dataset_format 为 tfrecord 时，`numRows` 是schema文件中唯一可以由用户设置的选项，其他值必须根据数据集设置。

当 dataset_format 为 mindrecord 时，`num_samlpes` 是 yaml 文件中唯一可由用户设置的选项，其他值必须根据数据集设置。

例如，用于预训练的cn-wiki-128数据集的schema文件显示如下：

```json
{
    "datasetType": "TF",
    "numRows": 7680,
    "columns": {
        "input_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "input_mask": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "segment_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [128]
        },
        "next_sentence_labels": {
            "type": "int64",
            "rank": 1,
            "shape": [1]
        },
        "masked_lm_positions": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_ids": {
            "type": "int64",
            "rank": 1,
            "shape": [20]
        },
        "masked_lm_weights": {
            "type": "float32",
            "rank": 1,
            "shape": [20]
        }
    }
}
```

或者，SQuAD数据集的schema文件显示如下：

```json
{
  "datasetType": "TF",
  "numRows": 1000000,
  "columns": {
    "input_ids": {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "input_mask": {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "segment_ids" : {
      "type": "int64",
      "rank": 1,
      "shape": [384]
    },
    "unique_ids": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "start_positions": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "end_positions": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    },
    "is_impossible": {
      "type": "int64",
      "rank": 1,
      "shape": [1]
    }
  }
}
```

## 训练过程

### 用法

#### Ascend处理器上运行

```bash
bash scripts/run_standalone_pretrain_ascend.sh 0 1 /path/cn-wiki-128
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认脚本路径下脚本文件夹中找到检查点文件，得到如下损失值：

```text

# grep "epoch" pretraining_log.txt

epoch: 0.0, current epoch percent: 0.000, step: 1, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0856101e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.000, step: 2, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.0821701e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**如果所运行的数据集较大，建议添加一个外部环境变量，确保HCCL不会超时。
>
> ```bash
> export HCCL_CONNECT_TIMEOUT=600
> ```
>
> 将HCCL的超时时间从默认的120秒延长到600秒。
> **注意**若使用的BERT模型较大，保存检查点时可能会出现protobuf错误，可尝试使用下面的环境集。
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```

### 分布式训练

#### Ascend处理器上运行

在多卡运行之前，您可以按以下操作生成distributed_cmd.sh:

```python
python scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py --run_script_dir ./scripts/run_distributed_pretrain_ascend.sh --hyper_parameter_config_dir ./scripts/ascend_distributed_launcher/hyper_parameter_config.ini --data_dir /path/data_dir/ --hccl_config /path/hccl.json --cmd_file ./distributed_cmd.sh
```

```bash
bash scripts/run_distributed_pretrain_ascend.sh /path/cn-wiki-128 /path/hccl.json
```

以上命令后台运行，您可以在pretraining_log.txt中查看训练日志。训练结束后，您可以在默认LOG*文件夹下找到检查点文件，得到如下损失值：

```text

# grep "epoch" LOG*/pretraining_log.txt

epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08209e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07566e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 0.0, current epoch percent: 0.001, step: 100, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.08218e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 0.0, current epoch percent: 0.002, step: 200, outputs are (Tensor(shape=[1], dtype=Float32, [ 1.07770e+01]), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **注意**训练过程中会根据device_num和处理器总数绑定处理器内核。如果您不希望预训练中绑定处理器内核，请在`scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py`中移除`taskset`相关操作。

## 评估过程

### 用法

> **注意** 如果推理数据集是mindrecord或tfrecord格式，对应yaml中dataset_format参数需要修改为mindrecord或tfrecord。对应sh脚本中：'train_data_file_path'和'eval_data_file_path'参数需要适配修改，同时不需要设置'schema_file_path'

#### Ascend处理器上运行后评估tnews数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/tnews/train.tf_record" \

--eval_data_file_path="/data/tnews/dev.tf_record" \

--schema_file_path="/data/tnews/dataset.json"

```bash
bash scripts/run_classifier.sh
```

以上命令后台运行，您可以在classfier_log.txt中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果(accuracy: 0.55~0.56)：

```text
acc_num XXX, total_num XXX, accuracy 0.555500
```

#### CPU处理器上运行后评估tnews数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/tnews/train.mindrecord" \

--eval_data_file_path="/data/tnews/dev.mindrecord"

```bash
python run_classifier.py --config_path=../../task_classifier_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --num_class=15 --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

如您选择准确性作为评估方法，可得到如下结果(accuracy: 0.55~0.56)：

```text
acc_num XXX, total_num XXX, accuracy 0.554200
```

#### Ascend处理器上运行后评估cluener数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--label_file_path="/data/finetune/cluener/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/cluener/train.tf_record" \

--eval_data_file_path="/data/cluener/dev.tf_record" \

--schema_file_path="/data/cluener/dataset.json"

```bash
bash scripts/run_ner.sh
```

以上命令后台运行，您可以在ner_log.txt中查看训练日志。

如您选择F1作为评估方法，可得到如下结果：

```text
Precision 0.868245
Recall 0.865611
F1 0.866926
```

#### CPU处理器上运行后评估cluener数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--label_file_path="/data/finetune/cluener/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/cluener/train.mindrecord" \

--eval_data_file_path="/data/cluener/dev.mindrecord"

```bash
python run_ner.py --config_path=../../task_ner_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --assessment_method=Accuracy --use_crf=false --with_lstm=false --label_file_path="" --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

如您选择accuracy作为评估方法，可得到如下结果：

```text
acc_num XXX, total_num XXX, accuracy 0.916855
```

#### Ascend处理器上运行后评估chineseNer数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--label_file_path="/data/finetune/chineseNer/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/chineseNer/train.tf_record" \

--eval_data_file_path="/data/chineseNer/dev.tf_record" \

--schema_file_path="/data/chineseNer/dataset.json"

```bash
bash scripts/run_ner.sh
```

以上命令后台运行，您可以在ner_log.txt中查看训练日志。

如您选择F1作为评估方法，可得到如下结果：

```text
F1 0.986526
```

#### CPU处理器上运行后评估chineseNer数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--label_file_path="/data/finetune/chineseNer/label_file" \

--load_pretrain_checkpoint_path="/data/scripts/checkpoint_bert-20000_1.ckpt" \

--train_data_file_path="/data/chineseNer/train.mindrecord" \

--eval_data_file_path="/data/chineseNer/dev.mindrecord"

```bash
python run_ner.py --config_path=../../task_ner_cpu_config.yaml --device_target CPU --do_train=true --do_eval=true --assessment_method=BF1 --use_crf=true --with_lstm=true --label_file_path="" --train_data_file_path="" --eval_data_file_path="" --load_pretrain_checkpoint_path=""
```

如您选择BF1作为评估方法，可得到如下结果：

```text
Precision 0.983121
Recall 0.978546
F1 0.980828
```

#### Ascend处理器上运行后评估msra数据集

您可以采用如下方式，先将MSRA数据集的原始格式在预处理流程中转换为mindrecord格式以提升性能 (请注意label2id_file文件中的标注名称应与数据集msra_dataset.xml文件中的标注名保持一致)：

```python
python src/finetune_data_preprocess.py --data_dir=/path/msra_dataset.xml --vocab_file=/path/vacab_file --save_path=/path/msra_dataset.mindrecord --label2id=/path/label2id_file --max_seq_len=seq_len --class_filter="NAMEX" --split_begin=0.0 --split_end=1.0
```

此后，您可以进行微调再训练和推理流程，

```bash
bash scripts/run_ner.sh
```

以上命令后台运行，您可以在ner_log.txt中查看训练日志。
如您选择MF1（多标签的F1得分）作为评估方法，在微调训练10个epoch之后进行推理，可得到如下结果：

```text
F1 0.931243
```

#### Ascend处理器上运行后评估squad v1.1数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--vocab_file_path="/data/squad/vocab_bert_large_en.txt" \

--load_pretrain_checkpoint_path="/data/scripts/bert_converted.ckpt" \

--train_data_file_path="/data/squad/train.tf_record" \

--eval_json_path="/data/squad/dev-v1.1.json" \

```bash
bash scripts/squad.sh
```

以上命令后台运行，您可以在bant_log.txt中查看训练日志。
结果如下：

```text
{"exact_match": 80.3878923040233284, "f1": 87.6902384023850329}
```

#### CPU处理器上运行后评估squad v1.1数据集

运行以下命令前，确保已设置加载与训练检查点路径。请将检查点路径设置为绝对全路径，例如，

--vocab_file_path="/data/squad/vocab_bert_large_en.txt" \

--load_pretrain_checkpoint_path="/data/scripts/bert_converted.ckpt" \

--train_data_file_path="/data/squad/train.mindrecord" \

--eval_json_path="/data/squad/dev-v1.1.json" \

```bash
python run_squad.py --config_path=../../task_squad_config.yaml --device_target CPU --do_train=true --do_eval=true --vocab_file_path="" --train_data_file_path="" --eval_json_path="" --dataset_format tfrecord --load_pretrain_checkpoint_path=""
```

结果如下：

```text
{"exact_match": 79.62157048249763, "f1": 87.24089125977054}
```

## 导出mindir模型

由于预训练模型通常没有应用场景，需要经过下游任务的finetune之后才能使用，所以当前仅支持使用下游任务模型和yaml配置文件进行export操作。

- 在本地导出

```shell
python export.py --config_path [/path/*.yaml] --export_ckpt_file [CKPT_PATH] --export_file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

- 在ModelArts上导出

```python

# (1) 上传你的代码到 s3 桶上
# (2) 在ModelArts上创建训练任务
# (3) 选择代码目录 /{path}/bert
# (4) 选择启动文件 /{path}/bert/export.py
# (5) 执行a或b
#     a. 在 /path/bert 下的`task_ner_config.yaml`(或 `task_squad_config.yaml` 或 `task_classifier_config.yaml`) 文件中设置参数
#         1. 设置 ”enable_modelarts: True“
#         2. 设置 “export_ckpt_file: ./{path}/*.ckpt”('export_ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
#         3. 设置 ”export_file_name: bert_ner“
#         4. 设置 ”file_format：MINDIR“
#         5. 设置 ”label_file_path：{path}/*.txt“('label_file_path'指相对于第7步所选文件夹的相对路径)
#     b. 在 网页上设置
#         1. 添加 ”enable_modelarts=True“
#         2. 添加 “export_ckpt_file=./{path}/*.ckpt”(('export_ckpt_file' 指待导出的'*.ckpt'权重文件相对于`export.py`的路径, 且权重文件必须包含在代码目录下)
#         3. 添加 ”export_file_name=bert_ner“
#         4. 添加 ”file_format=MINDIR“
#         5. 添加 ”label_file_path：{path}/*.txt“('label_file_path'指相对于第7步所选文件夹的相对路径)
#     最后必须在网页上添加 “config_path=/path/*.yaml”(根据下游任务选择 *.yaml 配置文件)
# (7) 在网页上勾选数据存储位置，设置“训练数据集”路径
# (8) 在网页上设置“训练输出文件路径”、“作业日志路径”
# (9) 在网页上的’资源池选择‘项目下， 选择单卡规格的资源
# (10) 创建训练作业
# 你将在{Output file path}下看到 'bert_ner.mindir'文件

```

参数`export_ckpt_file` 是必需的，`file_format` 必须在 ["AIR", "MINDIR"]中进行选择。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 用法

在执行推理之前，需要通过export.py导出mindir文件。输入数据文件为bin格式。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [LABEL_PATH] [DATA_FILE_PATH] [DATASET_FORMAT] [SCHEMA_PATH] [TASK] [VOCAB_FILE_PATH] [EVAL_JSON_PATH] NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`TASK` 为必选项, 在 [ner|ner_crf|classifier|squad]中取值。
`DEVICE_ID` 可选，默认值为 0。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
Classifier: Accuracy=0.5539  NER: F1=0.931243
```

## 导出onnx模型与推理

当前已支持导出bert分类任务的ONNX模型, 并可通过ONNXRuntime等第三方工具加载ONNX进行推理。

- 导出ONNX

```shell
python export.py --config_path [/path/*.yaml] --file_format ["ONNX"] --export_ckpt_file [CKPT_PATH] --device_target
 [DEVICE_TARGET] --num_class [NUM_CLASS] --export_file_name [EXPORT_FILE_NAME]
```

`CKPT_PATH`为必选项, 是某个分类任务模型训练完毕的ckpt文件路径。
`NUM_CLASS`为必选项, 是该分类任务模型的类别数。
`EXPORT_FILE_NAME`为可选项, 是导出ONNX模型的名字, 如果未设置则ONNX模型会以默认名保存在当前目录下。
`DEVICE_TARGET`为可选项, 是硬件平台类型, 默认为Ascend, 当硬件平台是Ascend时无需设置, 当硬件平台是GPU或CPU时, 应设置该选项为对应平台。

运行结束后, 当前文件目录下会保存bert该分类任务模型的ONNX模型。

- 加载ONNX并推理

```shell
python run_eval_onnx.py --config_path [/path/*.yaml] --eval_data_file_path [EVAL_DATA_FILE_PATH] --export_file_name [EXPORT_FILE_NAME] --device_target [DEVICE_TARGET]
```

`EVAL_DATA_FILE_PATH`为必选项, 是该分类任务所用数据集的eval数据。
`EXPORT_FILE_NAME`为可选项, 是导出ONNX步骤中ONNX的模型名, 此处用于加载指定ONNX模型进行推理。
`DEVICE_TARGET`为可选项, 是硬件平台类型, 默认为Ascend, 当硬件平台是Ascend时无需设置, 当硬件平台是GPU或CPU时, 应设置该选项为对应平台。

## 模型描述

## 性能

### 预训练性能

| 参数                  | Ascend                                                     | GPU                       |
| -------------------------- | ---------------------------------------------------------- | ------------------------- |
| 模型版本              | BERT_base                                                      | BERT_base                  |
| 资源                   |Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8              || NV SMX2 V100-32G          |
| 上传日期              | 2021-07-05                                           | 2021-07-05      |
| MindSpore版本          | 1.3.0                                                      | 1.3.0                     |
| 数据集                    | cn-wiki-128(4000w)                                                | cn-wiki-128               |
| 训练参数        | pretrain_config.yaml                                           | pretrain_config.yaml          |
| 优化器                  | Lamb                                                       | Lamb                  |
| 损失函数             | SoftmaxCrossEntropy                                        | SoftmaxCrossEntropy       |
| 输出              | 概率                                                |                   |
| 轮次                      | 40                                                         |                           |                      |
| Batch_size | 256*8 | 32*8 |
| 损失                       | 1.7                                                        | 1.913                 |
| 速度                      | 284毫秒/步                                               |180毫秒/步             |
| 总时长                 | 63小时                              |                             |
| 参数（M）                 | 110                                                        |                        |
| 微调检查点 | 1.2G（.ckpt文件）                                           |                   |

| 参数                  | Ascend                                                     |
| -------------------------- | ---------------------------------------------------------- |
| 模型版本              | BERT_NEZHA                                                      |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8               |
| 上传日期              | 2021-07-05                                           |
| MindSpore版本          | 1.3.0                                                      |
| 数据集                    | cn-wiki-128(4000w)                                                |
| 训练参数        | pretrain_config.yaml                                           |
| 优化器                  | Lamb                                                        |
| 损失函数             | SoftmaxCrossEntropy                                        |
| 输出              | 概率                                                |
| 轮次                      | 40                                                         |
| Batch_size | 96*8 |
| 损失                       | 1.7                                                        |
| 速度                      | 320毫秒/步                                               |
| 总时长                 | 180小时                              |
| 参数（M）                 | 340                                                        |
| 微调检查点 | 3.2G（.ckpt文件）                                           |

#### 推理性能

| 参数                 | Ascend                        |
| -------------------------- | ----------------------------- |
| 模型版本              |                               |
| 资源                   | Ascend 910；系统 Euler2.8                    |
| 上传日期              | 2021-07-05                    |
| MindSpore版本         | 1.3.0                         |
| 数据集 | cola，1.2W |
| batch_size          | 32（单卡）                        |
| 准确率 | 0.588986 |
| 速度                      | 59.25毫秒/步                              |
| 总时长                 | 15分钟                              |
| 推理模型 | 1.2G（.ckpt文件）              |

# 随机情况说明

run_standalone_pretrain.sh和run_distributed_pretrain.sh脚本中将do_shuffle设置为True，默认对数据集进行轮换操作。

run_classifier.sh、run_ner.sh和run_squad.sh中设置train_data_shuffle和eval_data_shuffle为True，默认对数据集进行轮换操作。

config.py中，默认将hidden_dropout_prob和note_pros_dropout_prob设置为0.1，丢弃部分网络节点。

run_pretrain.py中设置了随机种子，确保分布式训练中每个节点的初始权重相同。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

优先参考[ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ)来查找一些常见的公共问题。

- **Q: 运行过程中发生持续溢出怎么办？**

  **A**： 持续溢出通常是因为使用了较高的学习率导致训练不收敛。可以考虑修改yaml配置文件中的参数，调低`learning_rate`来降低初始学习率或提高`power`加速学习率衰减。

- **Q: 运行报错shape不匹配是什么问题？**

  **A**： Bert模型中的shape不匹配通常是因为模型参数配置和使用的数据集规格不匹配，主要是句长问题，可以考虑修改`seq_length`参数来匹配所使用的具体数据集。改变该参数不影响权重的规格，权重的规格仅与`max_position_embeddings`参数有关。

- **Q: 运行过程中报错Gather算子错误是什么问题？**

  **A**： Bert模型中的使用Gather算子完成embedding操作，操作会根据输入数据的值来映射字典表，字典表的大小由配置文件中的`vocab_size`来决定，一般中文为21128，英文为30522，当实际使用的数据集编码时使用的字典表大小超过配置的大小时，操作gather算子时就会发出越界访问的错误，从而Gather算子会报错中止程序。

- **Q: 修改了yaml文件中的配置，为什么没有效果？**

  **A**：实际运行的参数，由`yaml`文件和`命令行参数`共同控制，使用`ascend_dsitributed_launcher`的情况下，也会受`ini`配置文件的影响。起作用的优先级是**bash参数 > ini文件参数 > yaml文件参数**。

- **Q: 运行过程中出现因get_dataset_size失败导致的RuntimeError？**

  **A**：实际运行的dataset_format参数，由`yaml`文件定义。如果你遇到get_dataset_size失败的问题，你需要校验一下脚本中设置的输入数据格式是否和yaml文件中定义的dataset_format一致。当前dataset_format只支持[tfrecord，mindrecord]。

