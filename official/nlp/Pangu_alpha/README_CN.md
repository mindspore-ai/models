# 目录

- [目录](#目录)
- [PanGu-Alpha描述](#pangu-alpha描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [安装要求](#安装要求)
    - [数据集生成](#数据集生成)
        - [增量训练](#增量训练)
    - [训练](#训练)
        - [Ascend上运行训练](#ascend运行训练)
        - [GPU上训练](#GPU上训练)
        - [MoE训练](#moe训练)
            - [异构训练](#异构训练)
            - [同构训练](#同构训练)
        - [增量训练](#增量训练-1)
    - [预测](#预测)
        - [下载检查点](#下载检查点)
        - [分布式预测](#分布式预测)
        - [单机预测](#单机预测)
    - [下游任务评估](#下游任务评估)
        - [下载数据集](#下载数据集)
        - [下载检查点](#下载检查点)
        - [运行评估](#运行评估)
        - [在启用Server的情况下运行评估](#在启用server的情况下运行评估)
        - [2.6B模型零样本的评估结果](#26b模型零样本的评估结果)
    - [Serving](#serving)
        - [准备](#准备)
        - [Ascend 910/Nvidia GPU上单机运行Serving 13B或2.6B模型](#ascend-910nvidia-gpu上单机运行serving-13b或26b模型)
        - [Ascend 910上分布式运行Serving 13B或2.6B模型](#ascend-910上分布式运行serving-13b或26b模型)
        - [Ascend 910上分布式运行8卡多机Serving](#ascend-910上分布式运行8卡多机serving)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
- [ModelZoo主页](#modelzoo主页)
- [要求](#要求)
- [FAQ](#faq)

# [PanGu-Alpha描述](#目录)

我们正在探索训练具有数十亿甚至万亿参数的大模型的最新前沿技术。
基于MindSpore的并行特性，我们采用了高效的模型并行和数据并行技术，如算子级并行，
最大限度地降低通信成本，提高计算效率。
只需少量修改，就可以很容易地扩展到数千个NPU和万亿参数量的模型。

与此同时，我们在PanGu-Alpha语言模型上运行并行训练，证明并行条件下也可以很容易地训练大模型。
 训练技巧总结如下：

1. 算子级模型并行
2. 流水线模型并行
3. 优化器模型并行

有关上述特性，请点击[此处](https://www.mindspore.cn/tutorials/experts/en/master/parallel/overview.html)查看详情。
更多特性敬请期待。

详细技术报告和检查点文件，可点击[此处](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-AIpha)查看。

# [模型架构](#目录)

![](./docs/model.png)

PanGu-α基于Transformer的架构，如今已被广泛用作各种预训练语言模型的骨干网络，如BERT和GPT。
 与之不同，我们在Transformer层之上添加了一个查询层，用来预测下一个token。
 模型的示意图如图1所示。

# [数据集](#目录)

- 开源数据集。

    每个示例中都使用1024个token对上述数据集进行预处理。dataset.py中的列键默认为`input_ids`。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

## 安装要求

下表给出了测试环境、脚本以及MindSpore版本的说明。**注意该模型仅支持图模式。**

| 并行模式     | MindSpore版本| GPU(V100)             | Ascend 910             |
| -----------------  | ----------------- | ---------------------- | -------------------------------- |
| 数据并行     | Master   | 支持             | 支持                       |
| 模型并行    | Master   | 支持             | 支持                       |
| 优化器并行| Master   | 支持             | 支持                       |
| 重计算         | Master   | 支持             | 支持                       |
| 流水线并行 | Master   | 不支持         | 支持                       |

如需获取Pangu_α脚本，可使用`git`工具按照如下操作克隆MindSpore的代码：

```bash
git clone https://gitee.com/mindspore/models.git -b master
cd models/official/nlp/Pangu_alpha
```

请参见[要求](#要求)来安装依赖项。

## 数据集生成

下游任务的格式可能多种多样，因此`preprocess.py`提供了如何处理原始文本文件的基本用法。请使用以下格式准备数据，文件中每行是一段连续的文本：

```text
今天是一个好天气，小明很高兴的背起书包上学去。但是...
突然刮起了狂风暴雨！
```

假设文本数据放在`./data`下且**每个文本文件以'txt'结尾**，我们可以运行以下命令生成seq_length=1025的MindRecord文件。

```bash
python -m src.preprocess --input_glob  'data/*.txt' --tokenizer gpt --eot 50256 --data_column_name input_ids --seq_length 1025
```

脚本用1025个token对每一行进行分词，不足1025个token的部分将被忽略。

输出文件位于`./output`目录下。默认采用Transformer的分词器。注意，`vocab_szie`的值取决于vocab文件。

- tokenizer：用于标记文本，可采用GPT（用于Transformer）或结巴分词器。注意，GPT分词器需要同时使用Transformer、Pytorch或TensorFlow。`结巴`分词器需要添加两个文件vocab.model。单击[此处](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer)下载。
- eod_id：文档结尾ID。
- data_column_name：MindRecord功能列名。
- seq_length：默认值为1025。预处理后，每个示例都会生成序列长度为1025的MindRecord。

### 增量训练

如需在[PCL-Platform](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)发布的ckpt上进行增量训练，请点击[此处](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer)下载`vocab.model`模型。然后运行以下命令，使用和预训练（使用结巴分词器）相同的词汇表对原始文本进行分词。

```bash
python -m src.preprocess --input_glob  data/*.txt --tokenizer jieba --model_file vocab.model --eot 6
```

`vocab.model`的词汇表大小为40000，`vocab.model`值为6。

## [训练](#目录)

### 在Ascend上运行训练

目前，脚本提供了四个默认配置：1.3B、2.6B、13B和200B。以**Ascend**上8卡训练`2.6B`模型为例。

```bash

# 运行分布式训练示例

bash scripts/run_distribute_train.sh DATASET RANK_TABLE RANK_SIZE TYPE MODE STAGE_NUM MICRO_SIZE PER_BATCH RANK_START
#示例：
bash scripts/run_distribute_train.sh /data/pangu_30_step_ba64/ /root/hccl_8p.json 8 fp32 2.6B 1 1 8 0 8

```

上述命令涉及以下`args`：

- DATASET：mindrecord文件父目录的路径。例如：`/home/work/mindrecord/`。
- RANK_TABLE：rank table的详细信息，请点击[此处](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)查看。该.json文件描述了`device id`、`service ip`和`rank`。
- RANK_SIZE：设备编号，也可以表示设备总数。例如，8、16、32 ...
- TYPE：参数初始化类型。参数使用单精度（FP32） 或半精度（FP16）初始化。可以节省设备占用内存。
- MODE：配置模式。通过设置`hidden size`和`layers`，将参数量增至26亿。还可以选择13B（`hidden size`为5120和`layers`为40，训练至少需要16卡）和200B模式。
- STAGE_NUM：流水线阶段的数量。当`stage_num`大于1时，应用流水线并行模式。该配置表示流水线并行模式下子图的数量。
- MICRO_SIZE：流水线并行模式下的微批次大小，其取值应该大于`stage_num`。
- PER_BATCH：每个数据并行的批处理大小，默认为16。
- RANK_START：本机的开始rank_id，在多卡场景下用于表示每台设备的rank_id。
- LOCAL_DEVICE_NUM：本机的设备编号。

以训练2.6B模型为例：

```bash
# 在单机Ascend上运行分布式训练示例

bash scripts/run_distribute_train.sh /path/dataset /path/hccl.json 8 fp32 2.6B 1 1 8 0 8
```

```bash
# 在双机Ascend上运行分布式训练示例

# 设备A
bash scripts/run_distribute_train.sh /path/dataset /path/hccl.json 16 fp32 2.6B 2 4 16 0 8
# 设备B
bash scripts/run_distribute_train.sh /path/dataset /path/hccl.json 16 fp32 2.6B 2 4 16 8 8
```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。
请按照以下链接中的说明操作：
https:gitee.com/mindspore/models/tree/master/utils/hccl_tools.

开始训练后，训练日志将重定向到设备{rank_id}/log{rank_id}.txt（例如，
device0/log0.log）。

### GPU上运行训练

该脚本通过mpirun启动GPU训练，用户可以在任何设备上运行以下命令开始训练。
请注意，多节点训练时，变量`NCCL_SOCKET_IFNAME` `NCCL_IB_HCA`的取值在某些设备上可能不同。如果遇异常，请取消设置或设置NCCL变量。
 点击[链接](https://www.mindspore.cn/docs/zh-CN/r1.9/faq/distributed_configure.html)查看详情。

```bash
# 以下变量可选。
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_HCA=^mlx5_16,mlx5_17
bash scripts/run_distributed_train_gpu.sh RANK_SIZE HOSTFILE DATASET PER_BATCH MOD
```

- RANK_SIZE：设备编号，也可以表示设备总数。例如，8、16、32 ...
- HOSTFILE：描述主机IP及其设备的文本文件。有关更多详细信息，请参见我们的[教程](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html) or [OpenMPI](https://www.open-mpi.org/)。
- DATASET：mindrecord文件父目录的路径。例如：`/home/work/mindrecord/`。
- PER_BATCH：每个数据并行的批处理大小，
- MODE：可以是`1.3B`、`2.6B`、`13B`或`200B`。

### MoE训练

#### 异构训练

目前，脚本提供了四个默认配置：1.3B、2.6B、13B和200B。仅支持Ascend设备。

```bash

# 运行分布式训练示例

bash scripts/run_distribute_train_moe_host_device.sh DATASET RANK_TABLE RANK_SIZE TYPE MODE STAGE_NUM MICRO_SIZE PER_BATCH RANK_START LOCAL_DEVICE_NUM EXPERT_NUM_PER_EP ENABLE_ALLTOALL

```

上述命令涉及以下args：

- DATASET：mindrecord文件父目录的路径。例如：`/home/work/mindrecord/`。
- RANK_TABLE：rank table的详细信息，请点击[此处](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)查看。该.json文件描述了device id、service ip和rank。
- RANK_SIZE：设备编号，也可以是您的设备总数。例如，8、16、32 ...
- TYPE：参数初始化类型。参数使用单精度（FP32） 或半精度（FP16）初始化。可以节省设备占用内存。
- MODE：配置模式。通过设置`hidden size`和`layers`，将参数量增至26亿。还可以选择`13B`（`hidden size`为5120和`layers`为40，训练至少需要16卡）和`200B`模式。
- STAGE_NUM：流水线阶段的数量。当`stage_num`大于1时，应用流水线并行模式。该配置表示流水线并行模式下子图的数量。
- MICRO_SIZE：流水线并行模式下的微批次大小，其取值应该大于stage_num。
- PER_BATCH：每个数据并行的批处理大小，默认为8。
- RANK_START：本机的开始rank_id，在多卡场景下用于表示每台设备的rank_id。
- LOCAL_DEVICE_NUM：本机的设备编号。
- EXPERT_NUM_PER_EP：单维度数据并行的专家数。
- ENABLE_ALLTOALL：启用alltoall通信。默认值0。

以8卡NPU训练60B模型为例：
模型配置与2.6B模型相同，但是没有MoE。
单机使用8卡NPU训练60B模型要求服务器至少有1TB的主机内存。

```bash
# 在单机Ascend上运行分布式训练示例

bash run_distributed_train_moe_host_device.sh /path/dataset /path/hccl.json 8 fp32 2.6B 1 1 2 0 8 36 0
```

#### 同构训练

您也可以使用MoE进行同构训练。

```bash

# 运行分布式训练示例

bash scripts/run_distribute_train_moe.sh DATASET RANK_TABLE RANK_SIZE TYPE MODE STAGE_NUM MICRO_SIZE PER_BATCH RANK_START LOCAL_DEVICE_NUM EXPERT_NUM_PER_EP ENABLE_ALLTOALL

```

各参数含义同`run_distributed_train_moe_host_device.sh`。

### 增量训练

在我们开始增量训练之前，必须完成以下两个步骤：

1. 使用发布的词汇表处理数据集，请参考[数据集生成中的增量训练](#数据集生成中的增量训练)。
2. 参考[下载检查点](#下载检查点)下载检查点和策略文件。每个主机都应拥有完整的检查点文件。

然后运行以下命令，开始`2.6B`模型增量训练：

```bash
export FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_incremental_train.sh DATASET RANK_TABLE 8 fp32 2.6B 8 ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt  ${FILE_PATH}/checkpoint_file filitered
```

## [预测](#目录)

### 下载检查点

请访问[网站](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)下载以下内容：

- 分词器：vocab.txt和vocab.model
- 检查点文件：同等参数量下\*.part\[0-4\]（需要提取）和*.npy
- 策略文件：描述参数如何在不同设备上切片的文件。

这里，我们假设下载的检查点、分词器和策略文件的目录结构如下：

注：以下所指ckpts路径均为引用为`/home/your_path/ckpts`。

```shell
ckpts
├── checkpoint_file
│   ├── filtered_*.ckpt
│   ├── word_embedding.npy
│   ├── top_query_embedding.npy
│   └── position_embedding.npy
├── strategy_load_ckpt
│   └── strategy.ckpt
└── tokenizer
    └── vocab.model
```

我们提供了两种预测方法。第一种是正常的方式，每次迭代都需要将输入填充到一定的长度。
 由于冗余计算，该方法的延迟相当高。为了加快速度性能，我们提供了二次状态重用（增量推理）方法。

默认启用状态重用，您可以通过将'use_past'参数值更改为False来禁用。

### 分布式预测

以Ascend上8卡预测为例。

```bash
export FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_predict.sh 8 /home/config/rank_table_8p.json ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B fp32
```

### 单机预测

以Ascend上或Nvidia GPU上单机预测为例。不同点在于，网络采用半精度（FP16）初始化。

```bash
export FILE_PATH=/home/your_path/ckpts
export DEVICE_TARGET=Ascend # or GPU
bash scripts/run_standalone_predict.sh ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B $DEVICE_TARGET
```

## 下游任务评估

此脚本提供以下任务的评估：

- [C3](https://github.com/nlpdata/c3)

### 下载数据集

单击上述任务的链接，下载数据。以C3为例，将数据集解压缩到
`/home/my_path/data/c3`。

其结构如下：

```text
c3
├── annotation
│   ├── c3-d-dev.txt
│   ├── c3-d-test.txt
│   ├── c3-m-dev.txt
│   └── c3-m-test.txt
├── bert
│   ├── convert_tf_checkpoint_to_pytorch.py
│   ├── extract_features.py
│   ├── __init__.py
│   ├── LICENSE
│   ├── modeling.py
│   ├── optimization.py
│   ├── run_classifier.py
│   └── tokenization.py
├── data
│   ├── c3-d-dev.json
│   ├── c3-d-test.json
│   ├── c3-d-train.json
│   ├── c3-m-dev.json
│   ├── c3-m-test.json
│   └── c3-m-train.json
├── license.txt
└── README.md
```

### 下载检查点

请按照[预测](#预测)中的说明下载检查点。

### 运行评估

大多数参数与[单机预测](#单机预测)相同，
除了最后的`TASK`和`TASK_PATH`参数。目前，只支持c3任务。

```bash
export FILE_PATH=/home/your_path/ckpts
export DEVICE_TARGET=Ascend # or GPU
export TASK=c3
export TASK_PATH=/home/your_c3_data_path
bash scripts/run_standalone_eval.sh ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B $DEVICE_TARGET $TASK $TASK_PATH
```

对于2.6B模型，大约需要13分钟才能得到结果。您可以在`device0/log0.log`下查看日志。
日志内容如下：

```text
数据集c3的指标为{'top1_acc': 0.5452}。
```

如果要使用13B模型，执行以下命令启动8卡评估：

```bash
export FILE_PATH=/home/your_path/ckpts
export DEVICE_TARGET=Ascend # or GPU
export TASK=c3
export TASK_PATH=/home/your_c3_data_path
export RANK_TABLE=/home/rank_table_8p.json
bash scripts/run_distribute_eval.sh 8 $RANK_TABLE ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/ ${FILE_PATH}/checkpoint_file 13B fp32 $TASK $TASK_PATH
```

### 在启用Server的情况下运行评估

#### 启动Server。

参考[Serving](./Serving)启动Server。

用户可以使用以下命令导出mindir文件。

```bash
set -e
export FILE_PATH=/home/your_path/ckpts
export DEVICE_TARGET=Ascend # or GPU
export TASK=c3
bash scripts/run_standalone_export.sh ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/checkpoint_file 2.6B ${DEVICE_TARGET} $TASK
```

您可以在device0/log0.log下查看日志。日志中看到以下输出，
即表示导出已完成：

```text
Export finished and now exit.
```

还需要复制vocab.model文件，用于分词。

```bash
mkdir -p serving_increment/pangu_standalone/pangu/tokenizer
cp -r your_path/vocab.model serving_increment/pangu_standalone/pangu/tokenizer/
```

模型导出后，我们可以使用以下命令启动serving。

```bash
mkdir -p serving_increment/pangu_standalone/pangu/1/
mv device0/* serving_increment/pangu_standalone/pangu/1/
cd serving_increment && bash start_pangu_standalone.sh
```

可以看到如下日志：

```text
* Running on all addresses (0, 0, 0, 0)
* Running on http://127.0.0.1: 5000
* Running on http://your_server_ip:5000
Press CTRL+C to quit
```

#### 启用Server运行评估

```bash
TOKENIZER_PATH=/home/your_path/ckpts/tokenizer/
EVAL_DATA_URL=/home/my_path/data/c3
EVAL_TASK=c3
python predict.py --enable_client --eval_task=$EVAL_TASK \
                  --tokenizer_path=$TOKENIZER_PATH \
                  --eval_data_url=$EVAL_DATA_URL
```

程序完成后（大约需要20分钟），输出以下内容。

```text
数据集c3的指标为{'top1_acc': 0.5432}。
```

### 2.6B模型零样本的评估结果

以下为PanGu-Alpha 2.6B模型的结果。

| 任务名称        | 指标  | 论文| 重现| 平台|
|-------------------|----------|-------|------------|----------|
| C3               | 准确率| 53.42 | 54.32      | Ascend  |

## [Serving](#目录)

### 准备

- 使用Pip工具安装MindSpore和MindSpore Serving1.5或更高版本。
- 如有需要，同步安装flask、flask、jieba、jieba和其他wl包。
- 下载[PanGu-Alpha仓](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)，
  后面的步骤需要`pangu-alpha/strategy_load_ckpt`和`pangu-alpha/tokenizer`。
- 从[PanGu-Alpha仓](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)下载13B或2.6B检查点文件和`*embedding`文件。

  对于13B模型，我们需要`13B_part0`~`13B_part3`、`13B_word_embedding`、`13B_top_query_embedding`，
  以及`13B_position_embedding`。

  对于2.6B，我们需要`2.6B_part0`~`2.6B_part3`、`13B_word_embedding`、`2.6B_top_query_embedding`，
  以及`2.6B_position_embedding`。

  解压所有`13B_part*`或`2.6B_part*`的tar文件，会生成大量*ckpt文件。
  将所有`*embedding`移动到`*.ckpt`文件的同一目录。

### Ascend 910/Nvidia GPU上单机运行Serving 13B或2.6B模型

- 使用脚本/run_standalone_export.sh导出MindIR模型，并将所有device0/*设备移动到
  'serving_increment/pangu_standalone/pangu/1/'目录。

  ```shell
  >>> cd scripts
  >>> bash run_standalone_export.sh ${strategy_file_path} ${ckpt_dir_path} 2.6B Ascend
  ```

  如果我们要导出2.6B模型，需要将`run_standalone_export.sh`中的`MODE`参数值从`13B`更新为`2.6B`。

  在GPU环境中运行时，需要将`run_standalone_export.sh`中的`DEVICE_TARGET`参数值从`Ascend`更新到`GPU`。

  `${strategy_file_path}`是13B模型`pangu-alpha/strategy_load_ckpt/angu_alpha_13B_cktp_strategy.ckpt`
  以及2.6B模型`pangu-alpha/strategy_load_ckpt/angu_alpha_2.6B_cktp_strategy.ckpt`的文件路径。

  `${ckpt_dir_path}`是解压后生成的`*ckpt`文件和`*embedding`文件的目录。

  模型导出会需要几分钟。查看日志device_0/log0.log，确认最后没有异常。
  确认在device_0/中已经生成了mindir文件，即模型导出成功。

  ```shell
  >>> ls device_0
  pangu_alpha_1024_graph.mindir  pangu_alpha_1024_variables  pangu_alpha_1_graph.mindir  pangu_alpha_1_variables
  >>> cd - && mkdir serving_increment/pangu_standalone/pangu/1/
  >>> mv scripts/device_0/* serving_increment/pangu_standalone/pangu/1/
  >>> cd serving_increment
  ```

- 将`pangu_alpha/tokenizer`复制到**serving_increment/pangu_standalone/pangu/tokenizer**目录下。

  所需文件的目录结构如下所示。pangu_alpha_1024_variables和pangu_alpha_1_variables已折叠，便于显示。

  ```shell
  >>> tree pangu_distributed
  pangu_standalone/
  ├── pangu
  │   ├── 1
  │   │   ├── pangu_alpha_1024_graph.mindir
  │   │   ├── pangu_alpha_1024_variables/
  │   │   ├── pangu_alpha_1_graph.mindir
  │   │   └── pangu_alpha_1_variables/
  │   ├── servable_config.py
  │   ├── tokenization_jieba.py
  │   └── tokenizer
  │       └── vocab.model
  └── serving_server.py
  ```

- 运行bash start_pangu_Standone.sh开始新的执行，等待serving和flask server启动成功。

- 如果发生任何错误，可以在serving_server.log、serving_logs/*.log和flask.log中查看日志。
- 确认无误后，在浏览器中访问地址{ip}:5000，等待回复需要一些时间。
- 执行`bash stop_pangu.sh`停止当前执行。

### Ascend 910上分布式运行Serving 13B或2.6B模型

- 生成[rank table文件](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。

  ```shell
  # 生成目录models/utils/hccl_tools/hccl_tools
  >>> python3 models/utils/hccl_tools/hccl_tools.py --device_num "[0,8]"
  >>> mv hccl_8p_01234567*.json serving_increment/pangu_distributed/hccl_8p.json
  ```

- 使用脚本/run_distribute_export.sh导出MindIR模型，并将所有device*设备移动到
  'serving_increment/pangu_distributed/models/'目录下。

  ```shell
  >>> cd scripts
  >>> bash run_distribute_export.sh ${strategy_file_path} ${ckpt_dir_path}
  ```

  如果我们要导出2.6B模型，需要将`run_distribute_export.sh`中的`MODE`参数值从`13B`更新为`2.6B`。

  `${strategy_file_path}`是13B模型`pangu-alpha/strategy_load_ckpt/angu_alpha_13B_cktp_strategy.ckpt`
  以及2.6B模型`pangu-alpha/strategy_load_ckpt/angu_alpha_2.6B_cktp_strategy.ckpt`的文件路径。

  `${ckpt_dir_path}`是解压后生成的*ckpt文件和*embedding文件的目录。

  模型导出会需要几分钟。查看日志device_[0-7]/log[0-7].log，确认最后没有异常。
   确认在device_[0-7]/中已经生成了mindir文件，即模型导出成功。

  ```shell
  >>> cd - && mkdir serving_increment/pangu_distributed/models/
  >>> mv scripts/device_* serving_increment/pangu_distributed/models/
  >>> cd serving_increment
  ```

- 如有需要，更新MindIR文件名serving_increment/pangu_distributed/serving_agent.py。
- 将`pangu-alpha/tokenizer`复制到serving_increment/pangu_distributed/pangu/tokenizer目录下。

  所需文件的目录结构如下所示。device_1至device_7已折叠，便于显示。

  ```shell
  >>> tree pangu_distributed
  pangu_distributed/
  ├── hccl_8p.json
  ├── models
  │   ├── device_0
  │   │   ├── pangu_alpha_1024_graph.mindir
  │   │   ├── pangu_alpha_1024_variables
  │   │   │   ├── data_0
  │   │   │   ├── data_1
  │   │   │   ├── data_2
  │   │   │   ├── data_3
  │   │   │   └── data_4
  │   │   ├── pangu_alpha_1_graph.mindir
  │   │   └── pangu_alpha_1_variables
  │   │       ├── data_0
  │   │       ├── data_1
  │   │       ├── data_2
  │   │       ├── data_3
  │   │       └── data_4
  │   ├── device_1/
  │   ├── device_2/
  │   ├── device_3/
  │   ├── device_4/
  │   ├── device_5/
  │   ├── device_6/
  │   └── device_7/
  ├── pangu
  │   ├── servable_config.py
  │   ├── tokenization_jieba.py
  │   └── tokenizer
  │       └── vocab.model
  ├── serving_agent.py
  └── serving_server.py
  ```

- 运行`bash start_pangu_distributed.sh`开始新的执行，等待serving和flask server启动成功。

- 如果发生任何错误，可以在serving_server.log、serving_agent.logserving_logs/*.log和flask.log中查看日志。
- 确认无误后，在浏览器中访问地址{ip}:5000，等待回复需要一些时间。
- 执行`bash stop_pangu.sh`停止当前执行。

### Ascend 910上分布式运行8卡多机Serving

- 生成[rank table文件](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。
- 在每台设备上，准备检查点文件和嵌入文件。我们也可以使用13B模型作为测试示例。
- 在每台设备上，使用脚本/run_cluster_export.sh导出MindIR模型，并将所有设备*移动到
  'serving_increment/pangu_distributed/models/'目录下。

```shell
>>> cd scripts
>>> bash run_cluster_export.sh ${strategy_file_path} ${ckpt_dir_path} ${rank_table_file} ${rank_size} ${rank_start}
```

如果我们要导出13B模型，需要将`run_distribute_export.sh`中的`MODE`参数值从`200B`更新为`13B`。

`${rank_start}`是每台设备中的第一个 rank id，如0,8,16,24。

模型导出会需要几分钟。查看日志device_[0-7]/log[0-7].log，确认最后没有异常。
 确认在device_[0-7]/中已经生成了mindir文件，即模型导出成功。

```shell
>>> cd - && mkdir serving_increment/pangu_distributed/models/
>>> mv scripts/device_* serving_increment/pangu_distributed/models/
>>> cd serving_increment
```

- 在第一台设备上，更新`serving_increment/pangu_distributed/pangu/servable_config.py`中的`rank_size`和`stage_size`（流水线阶段大小）。
- 在第一台设备上，更新`rank_table_json_file` of `serving_increment/pangu_distributed/serving_server.py`中的参数。
- 如有需要，更新每台设备上的MindIR文件名**serving_increment/pangu_distributed/serving_agent.py**。
- 在每台设备上，将`serving_increment/pangu_distributed/serving_agent.py`
  和`serving_increment/pangu_distributed/serving_server.py`的`distributed_address`更新为第一台设备的IP地址。
- 在第一台设备上，将`pangu-alpha/tokenizer`复制到**serving_increment/pangu_distributed/pangu/tokenizer**目录下。
- 在第一台设备上，运行`bash start_pangu_distributed.sh`，开始新的执行。
- 同时，在其他设备上，运行`python serving_agent.py`，启动serving agent进程。

  ```shell
  >>> unset http_proxy && unset https_proxy
  >>> python pangu_distributed/serving_agent.py > serving_agent.log 2>&1 &
  ```

- 等待serving和flask server启动成功。
- 如果发生任何错误，可以在serving_server.log、serving_agent.logserving_logs/*.log和flask.log中查看日志。
- 确认无误后，在浏览器中访问地址{ip}:5000，等待回复需要一些时间。
- 在每台设备上执行`bash stop_pangu.sh`停止当前执行。

# [脚本说明](#目录)

## 脚本及样例代码

```bash
.
├── docs
│         └── model.png
├── predict.py
├── README.md
├── scripts
│         ├── run_distribute_predict.sh
│         └── run_distribute_train.sh
├── src
│         ├── dataset.py
│         ├── generate.py
│         ├── pangu_alpha_config.py
│         ├── pangu_alpha.py
│         ├── pangu_alpha_wrapcell.py
│         ├── preprocess.py
│         ├── tokenization_jieba.py
│         └── utils.py
└── train.py
```

# [ModelZoo主页](#目录)

注：此模型将被移动到r1.8中的`/models/research/`目录下。

请浏览官网[主页](https://gitee.com/mindspore/models)。

# [要求](#目录)

- Mindpore 1.9.0或更高版本
- jieba 0.42.1
- sentencepiece 0.1.94
- transformers 4.7.0 or later
- mindformers 0.3

对于Serving和flask server，要求如下：

- MindSpore Serving 1.3.0
- flask-apscheduler 1.12.2
- flask 1.1.2

注意: 用户可以使用下述的命令来安装mindformers:

```shell
git clone --branch r0.3 https://gitee.com/mindspore/mindformers.git
cd mindformers
python setup.py install
```

# [FAQ](#目录)

问：发生意外错误。MindRecordOp初始化失败，非法列的列表。

答：这是因为`dataset.py`中的特征列名称与mindrecord中的名称不一致。在`run_distribute_train.sh`中添加`--data_column_name your_feature name`。

问：提示错误`ERROR: device_num must be the power of 2`。

答：并行训练时，卡的数量必须是2的幂次方。例如，如果我们想训练2.6B模型，所使用卡的数量应该是2、4、8、16等。

问：如何修改网络的超参？

答：网络的预定义超参在`src/pangu_alpha_config.py`的函数`set_parse`中设置。
该参数可以设置图层数、隐藏大小等。数据并行数在可`train.py`中通过
`device_num / model_parallel`来设置。

问：对于Ascend设备，在多节点模式下训练模型时，程序长时间卡住？

答：主要有两个原因：

- 网络过于复杂，编译需要的时间长，此时耐心等待。或者将网络的层数缩小到2层的规模，此时预计几分钟内可以打印日志

- 由于某些原因，有些设备无法编译，而有些设备正在等待失败的设备。等待时间可以通过`hccl_connect_time`参数设置，默认为6000秒。如需测试网络和设备是否正常，可以设置为1200秒，层数设置为2，看看是否能得到损失值。

Q：我在模型训练的时候loss持续溢出怎么办?

A：目前网络中除了`Embedding/Loss/LayerNorm`采用`FP32`计算，其余部分默认采用`FP16`计算。如果在网络训练过程中，发现loss突然上升，同时loss scale的值突然下降到1，请参考下面的方式将`softmax`的计算类型设置为`FP32`

在`src/pangu_alpha_config.py`文件中的`PanguAlphaConfig`类中，将`softmax_compute_type`初始值修改为`mstype.float32`
