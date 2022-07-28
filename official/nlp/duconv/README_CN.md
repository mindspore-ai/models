# 目录

<!-- TOC -->

- [目录](#目录)
- [Proactive-Conversation描述](#Proactive-Conversation描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Proactive-Conversation描述

人机对话研究近年来受到学术界和工业界的广泛关注，目前的对话系统还处于起步阶段，通常是被动地交谈，说话更多是作为回应而不是自己的主动，这与人与人的对话不同，机器的主动对话能力是其实现类似人与人之间自然对话的关键。为了促进主动对话系统的开发，Proactive Conversation模型设置了新的对话任务，该模型由在论文《Proactive Human-Machine Conversation with Explicit Conversation Goals》中提出，主要赋予它主动引导对话的能力（引入新话题或保持当前话题，即模型中的knowledge）。

[论文](https://arxiv.org/abs/1906.05572v2)：Wu W, Guo Z, Zhou X, et al. Proactive human-machine conversation with explicit conversation goals[J]. arXiv preprint arXiv:1906.05572, 2019.

# 模型架构

Proactive Conversation模型包含四个部分：
第一部分：Transformer Encoding layer，对对话过程中的上下文和response作为整体进行编码，得到对话表示；
第二部分：Knowledge encode layer，利用双向GRU对知识进行编码；
第三部分：Knowledge Reasoner，对前面两部分表示进行Attention计算，并获得注意力分布，得到给定对话下的知识表示；
第四部分：Output layer，将对话与知识相结合，输入至MLP进行二分类。

# 数据集

使用的数据集：[DuConv](<https://dataset-bj.cdn.bcebos.com/duconv/train.txt.gz>)

- 数据集大小：
    - dialogs 29858
    - utterances 270399
    - average # utterances per dialog  9.1
    - average # words per utterance 10.6
    - average # words per dialog 96.2
    - average # knowledge per dialogue 17.1

- 数据格式：二进制文件
    - 注：数据将在src/reader.py中进行处理。

- 数据集目录结构

    ```python
    ├──data
        ├──resource
        ├    ├──candidate.train.txt
        ├    ├──candidate.dev.txt
        ├    ├──candidate.test.txt
        ├    ├──sample.train.txt
        ├    ├──sample.dev.txt
        ├    ├──train.txt
        ├    ├──dev.txt
        ├    ├──test.txt
        ├
        ├──train.mindrecord
        ├──test.mindrecord
        ├──dev.mindrecord
        ├──build.train.txt
        ├──build.dev.txt
        ├──candidate_set.txt
        ├──gene.dict
    ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 数据处理

  ```bash
  # 数据集下载
  bash scripts/download_dataset.sh
  # 原始数据样例：
  {"goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"], ["蕾切儿 · 哈伍德", "家人", "休 · 劳瑞"]], "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"], ["休 · 劳瑞", "描述", "医疗剧 中 的 性感 医师"], ["休 · 劳瑞", "出生 日期", "1959 - 6 - 11"], ["休 · 劳瑞", "性别", "男"], ["休 · 劳瑞", "职业", "编剧"], ["休 · 劳瑞", "领域", "明星"], ["蕾切儿 · 哈伍德", "评论", "红 头发 和 古装 最 适合 她 ， 美 的 让 人 心碎 。 请 多 拍 些 大片"], ["蕾切儿 · 哈伍德", "获奖", "香水 _ 提名 _ ( 2007 ； 第33届 ) _ 土星奖 _ 土星奖 - 最佳 女 配角"], ["蕾切儿 · 哈伍德", "家人", "休 · 劳瑞"], ["蕾切儿 · 哈伍德", "性别", "女"], ["蕾切儿 · 哈伍德", "职业", "演员"], ["蕾切儿 · 哈伍德", "领域", "明星"], ["休 · 劳瑞", "评论", "不要 疯狂 迷恋 叔 ， 叔 只是 个 传说 。"], ["休 · 劳瑞", "评分", "9.4"], ["休 · 劳瑞", "星座", "双子座"], ["休 · 劳瑞", "职业", "导演"]], "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？", "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"], "response": "那 你 喜欢 休 · 劳瑞 不 ？ 一个 外国 明星 。"}
  # goal: 表示对话引导者进行对话的路径 knowledge: 提供goal的背景知识 history：历史对话 response：对话回复
  # 数据集构建
  ```

  ```bash
  bash scripts/build_dataset.sh
  #模型所需数据样例：
  0 你 对 明 星 有 没 有 到 迷 恋 的 程 度 呢 ？ [INNER_SEP] 一 般 吧 ， 毕 竟 年 纪 不 小 了 ， 只 是 追 星 而 已 。 给 你 推 荐 一 个 明 星 ， 他 叫 person_topic_a ， 他 也 是 处 女 座 的 ， 你 可 以 关 注 一 下 。  START person_topic_a person_topic_b [PATH_SEP] person_topic_b 家 人 person_topic_a    person_topic_a 评 论 完 美 的 男 人 [KN_SEP] person_topic_a 描 述 医 疗 剧 中 的 性 感 医 师 [KN_SEP] person_topic_a 出 生 日 期 1959 - 6 - 11 [KN_SEP] person_topic_a 性 别 男 [KN_SEP] person_topic_a 职 业 编 剧 [KN_SEP] person_topic_a 领 域 明 星 [KN_SEP] person_topic_b 评 论 红 头 发 和 古 装 最 适 合 她 ， 美 的 让 人 心 碎 。 请 多 拍 些 大 片 [KN_SEP] person_topic_b 获 奖 香 水 _ 提 名 _ ( 2007 ； 第 3 3 届 ) _ 土 星 奖 _ 土 星 奖 - 最 佳 女 配 角 [KN_SEP] person_topic_b 家 人 person_topic_a [KN_SEP] person_topic_b 性 别 女 [KN_SEP] person_topic_b 职 业 演 员 [KN_SEP] person_topic_b 领 域 明 星 [KN_SEP] person_topic_a 评 论 不 要 疯 狂 迷 恋 叔 ， 叔 只 是 个 传 说 。 [KN_SEP] person_topic_a 评 分 9.4 [KN_SEP] person_topic_a 星 座 双 子 座 [KN_SEP] person_topic_a 职 业 导 演
  # 0: 标签
  ```

  ```bash
  # 数据转换-mindrecord
  bash scripts/convert_dataset.sh
  ```

- Ascend处理器环境运行

  ```python
  # 单卡运行训练示例
  bash run_train.sh [TASK_NAME] [DATA_PATH] [OUTPUT_PATH]

  # 运行分布式训练示例
  bash run_train_distribute.sh [TASK_NAME] [DATA_PATH] hccl_8p.json [OUTPUT_PATH]

  # 运行评估示例
  bash run_predict.sh [TASK_NAME] [DATA_PATH] [CHECKPOINT_PATH] [PREDICT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将./train.py 和 ./predict.py 中的`device_target`从`Ascend`改为`GPU`

  ```python
  # 运行训练示例
  export CUDA_VISIBLE_DEVICES=0
  bash run_train_distribute.sh [TASK_NAME] [DATA_PATH] [OUTPUT_PATH]

  # 运行评估示例
  bash run_predict.sh [TASK_NAME] [DATA_PATH] [CHECKPOINT_PATH] [PREDICT_PATH]
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

  ```python
  (1)上传你的代码和数据集到obs上
  (2) 在ModelArts上创建训练任务
  (3) 选择代码目录 /{path}/DuConv_mindspore
  (4) 选择启动文件 /{path}/DuConv_mindspore/train.py
  (5) 执行
     在网页上设置
       1. 设置 use_modelarts=true
       2. 其它超参数
  (6) 在网页上勾选数据存储位置设置训练数据集路径
  (7) 在网页上设置训练输出文件路径和作业日志路
  (8) 在网页上的资源池选择项目下选择8卡或者单卡规格的资源
  (9) 创建训练作业
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── Proactive_conversation
        ├── README.md                      // Proactive_conversation相关说明
        ├── scripts
            ├──download_dataset.sh        // 下载模型数据集脚本
            ├──build_dataset.sh           // 将数据集处理成模型所需数据形式
            ├──convert_dataset.sh         // 将处理好的数据转换成mindrecord
            ├──run_train.sh               // 单卡Ascend的shell脚本
            ├──run_train_distribute.sh    // 分布式Ascend的shell脚本
            ├──run_predict.sh             // 评估的shell脚本
            ├──run_export.sh                  // 模型导出脚本

        ├── src
            ├──utils
                ├──__init__.py
                ├──build_candidate_set_from_corpus.py
                ├──build_dict.py
                ├──construct_candidate.py
                ├──convert_conversation_corpus_to_model_text.py
                ├──convert_session_to_sample.py
                ├──extract.py

            ├──bert.py
            ├──rnns.py
            ├──rnn_encoder.py
            ├──model.py
            ├──reader.py
            ├──datasets.py
            ├──lr_schedule.py
            ├──eval.py
            ├──callbacks.py

        ├── train.py                    // 训练网络
        ├── predict.py                  // 评估网络
        ├── export.py                  // 模型导出
```

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  bash run_train.sh [TASK_NAME] [DATA_PATH] [OUTPUT_PATH]
  ```

  训练结束后，您可在默认脚本文件夹下找到检查点文件,文件内容如下：

  ```bash
  # grep "loss is " log.txt
  epoch:1 step:52, loss is 0.2744
  per step time: 98.912ms
  epcoh:1 step:152, loss is 0.1677
  per step time: 104.442ms
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py > train.log 2>&1 &
  ```

  通过修改train.py中的参数实现GPU环境运行。上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认`./ckpt_0/`脚本文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_train_distribute.sh [TASK_NAME] [DATA_PATH] hccl_8p.json [OUTPUT_PATH]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过LONG_{0,1,2,3,...}/log.txt文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估DuConv数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“/home/DuConv_mindspore/save_model/ckpt0”。

  ```bash
  ##example for evaluate model
  bash run_preict.sh match_kn_gene /DuConv_mindspore/data/test.mindrecord /home/DuConv_mindspore/save_model/ckpt0 predict1p
  ```

  上述python命令将在后台运行，您可以通过./predict/predict_match_kn_gene_rank_?_ckpt.log文件查看结果。测试数据集的准确性如下：

  ```bash
    F1: 31.50%
    BLEU1: 0.285%
    BLEU2: 0.154%
    DISTINCT1: 0.120%
    DISTINCT2: 0.399%
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为所有检查点文件的目录，如“/home/DuConv_mindspore/output8p/save_model/ckpt0”。一个ckpt对应的测试数据集的准确性如下：

  ```bash
    F1: 31.17%
    BLEU1: 0.283%
    BLEU2: 0.153%
    DISTINCT1: 0.123%
    DISTINCT2: 0.405%
  ```

  在predict文件夹含有所有ckpt目录中权重文件的的评估log，每个log的文件名与ckpt文件名对应，需要遍历所有log找到最优精度，通过对应文件名确认ckpt文件，或根据自己的需求，使用所需精度的ckpt。

## 导出过程

### 导出

在导出之前需要修改对应的参数，需要修改的项为 batch_size和ckpt_file。

```shell

bash run_export.sh [TASK_NAME] [CHECK_POINT_PATH] [FILE_FORMAT]

```

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

在昇腾310上进行推理

```shell
# Ascend310 inference

bash run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

`NEED_PREPROCESS` 为必选项, 在[y|n]中取值，表示数据是否预处理为bin格式。
`DEVICE_ID` 可选，默认值为 0。

### 结果

- 推理结果保存在当前路径，可在acc.log中看到最终精度结果。

```eval log
31.71%
```

# 模型描述

## 性能

### 训练性能

| 参数                 | Ascend                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | Proactive-conversation V1                                                | Proactive-conversation V1           |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | rtx-3090       |
| MindSpore版本          | 1.3.0                                                       | 1.3.0                  |
| 数据集                    | DuConv                                                    | DuConv               |
| 训练参数        | epoch=30, steps=7023, batch_size = 128, lr=0.1              | epoch=30, steps=7023, batch_size = 128, lr=0.1    |
| 优化器                  | Adam                                                    | Adam               |
| 损失函数              | Softmax交叉熵                                       | Softmax交叉熵  |
| 输出                    | 概率                                                 | 概率            |
| 速度                      | 单卡：92毫秒/步;  8卡：95毫秒/步                          | 单卡：130ms/步     |
| 总时长                 | 单卡：323.05分钟;  8卡：41.69分钟                          | 单卡：456.49分钟    |                 |
| 微调检查点 | 127M (.ckpt文件)                                         | 127M (.ckpt文件)    |
| 推理模型        | 47M (.mindir文件)                    |      |

### 评估性能

| 参数          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | Proactive-conversation V1                | Proactive-conversation V1                |
| 资源            |  Ascend 910；系统 Euler2.8                  | GPU                         |
| MindSpore 版本   | 1.3.0                       | 1.3.0                       |
| 数据集             | DuConv     | DuConv     |
| batch_size          | 128                         | 128                         |
| 输出             | 概率                 | 概率                 |
| 准确性            | 单卡: 31.50%;  8卡：31.22%   | 单卡：31.16%      |
| 推理模型 | 47MM (.mindir文件)         |  |

# 随机情况说明

在dataset.py中，我们设置了随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
