# 目录

<!-- TOC -->

- [目录](#目录)
- [InceptionV3描述](#inceptionv3描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度ascend)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [启动](#启动)
        - [结果](#结果)
            - [Ascend](#ascend)
            - [CPU](#cpu)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [启动](#启动-1)
        - [结果](#结果-1)
    - [模型导出](#模型导出)
    - [ONNX模型导出及评估](#onnx模型导出及评估)
        - [ONNX模型导出](#onnx模型导出)
        - [ONNX模型评估](#onnx模型评估)
    - [推理过程](#推理过程)
        - [使用方法](#使用方法)
        - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# InceptionV3描述

Google的InceptionV3是深度学习卷积架构系列的第3个版本。InceptionV3主要通过修改以前的Inception架构来减少计算资源的消耗。这个想法是在2015年出版的Rethinking the Inception Architecture for Computer Vision, published in 2015一文中提出的。

[论文](https://arxiv.org/pdf/1512.00567.pdf)： Min Sun, Ali Farhadi, Steve Seitz.Ranking Domain-Specific Highlights by Analyzing Edited Videos[J].2014.

# 模型架构

InceptionV3的总体网络架构如下：

[链接](https://arxiv.org/pdf/1512.00567.pdf)

# 数据集

所用数据集可参照论文。

使用的数据集: [ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G, 120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M，共10个类、6万张32*32彩色图像
    - 训练集：146M，共5万张图像
    - 测试集：29M，共1万张图像
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
- 使用Ascend来搭建硬件环境。
- 框架
- [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
- [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```python
    # 在 ModelArts 上使用8卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在 default_config.yaml 文件中设置 "lr_init=0.00004"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size=250"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "lr_init=0.00004"
    #          在网页上设置 "dataset_path=/cache/data"
    #          在网页上设置 "epoch_size=250"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/inceptionv3"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size=250"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 "epoch_size=250"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/inceptionv3"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "need_modelarts_dataset_unzip=True"
    #          在 default_config.yaml 文件中设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在 default_config.yaml 文件中设置 "checkpoint='./inceptionv3/inceptionv3-rank3_1-247_1251.ckpt'"
    #          在 default_config.yaml 文件中设置 "dataset_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "need_modelarts_dataset_unzip=True"
    #          在网页上设置 "modelarts_dataset_unzip_name='imagenet_original'"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_trained_model/'"
    #          在网页上设置 "checkpoint='./inceptionv3/inceptionv3-rank3_1-247_1251.ckpt'"
    #          在网页上设置 "dataset_path='/cache/data'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 执行a或者b (推荐选择 a)
    #       a. 第一, 将该数据集压缩为一个 ".zip" 文件。
    #          第二, 上传你的压缩数据集到 S3 桶上 (你也可以上传未压缩的数据集，但那可能会很慢。)
    #       b. 上传原始 coco 数据集到 S3 桶上。
    #           (数据集转换发生在训练过程中，需要花费较多的时间。每次训练的时候都会重新进行转换。)
    # (5) 在网页上设置你的代码路径为 "/path/inceptionv3"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

- 在 ModelArts 进行导出 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

1. 使用voc val数据集评估多尺度和翻转s8。评估步骤如下：

    ```python
    # (1) 执行 a 或者 b.
    #       a. 在 base_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 base_config.yaml 文件中设置 "file_name='inceptionv3'"
    #          在 base_config.yaml 文件中设置 "file_format='MINDIR'"
    #          在 base_config.yaml 文件中设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在 base_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在 base_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "file_name='inceptionv3'"
    #          在网页上设置 "file_format='MINDIR'"
    #          在网页上设置 "checkpoint_url='/The path of checkpoint in S3/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
    #          在网页上设置 其他参数
    # (2) 上传你的预训练模型到 S3 桶上
    # (3) 在网页上设置你的代码路径为 "/path/inceptionv3"
    # (4) 在网页上设置启动文件为 "export.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

# 脚本说明

## 脚本和样例代码

```shell
.
└─Inception-v3
  ├─README_CN.md
  ├─README.md
  ├─ascend310_infer                           # 实现310推理源代码
  ├─infer
  ├─modelarts
  ├─scripts
    ├─run_standalone_train_cpu.sh             # 启动CPU训练
    ├─run_standalone_train_gpu.sh             # 启动GPU单机训练（单卡）
    ├─run_distribute_train_gpu.sh             # 启动GPU分布式训练（8卡）
    ├─run_standalone_train.sh                 # 启动Ascend单机训练（单卡）
    ├─run_distribute_train.sh                 # 启动Ascend分布式训练（8卡）
    ├─run_infer_310.sh                        # Ascend推理shell脚本
    ├─run_eval_cpu.sh                         # 启动CPU评估
    ├─run_eval_gpu.sh                         # 启动GPU评估
    ├─run_eval_onnx_gpu.sh                    # 启动GPU下ONNX模型的评估
    └─run_eval.sh                             # 启动Ascend评估
  ├─src
    ├─dataset.py                      # 数据预处理
    ├─inception_v3.py                 # 网络定义
    ├─loss.py                         # 自定义交叉熵损失函数
    ├─lr_generator.py                 # 学习率生成器
    └─model_utils
      ├─config.py                     # 获取.yaml配置参数
      ├─device_adapter.py             # 获取云上id
      ├─local_adapter.py              # 获取本地id
      └─moxing_adapter.py             # 云上数据准备
  ├─default_config.yaml               # 训练配置参数(ascend)
  ├─default_config_cpu.yaml           # 训练配置参数(cpu)
  ├─default_config_gpu.yaml           # 训练配置参数(gpu)
  ├─eval.py                           # 评估网络
  ├─eval_onnx.py                      # 评估导出的ONNX模型
  ├─export.py                         # 导出 AIR,MINDIR模型的脚本
  ├─mindspore_hub_conf.py             # 创建网络模型
  ├─postprogress.py                   # 310推理后处理脚本
  └─train.py                          # 训练网络
```

## 脚本参数

```python
train.py和config.py中主要参数如下：
'random_seed'                # 修复随机种子
'rank'                       # 分布式的本地序号
'group_size'                 # 分布式进程总数
'work_nums'                  # 读取数据的worker个数
'decay_method'               # 学习率调度器模式
"loss_scale"                 # 损失等级
'batch_size'                 # 输入张量的批次大小
'epoch_size'                 # 总轮次数
'num_classes'                # 数据集类数
'ds_type'                    # 数据集类型，如：imagenet, cifar10
'ds_sink_mode'               # 使能数据下沉
'smooth_factor'              # 标签平滑因子
'aux_factor'                 # aux logit的损耗因子
'lr_init'                    # 初始学习率
'lr_max'                     # 最大学习率
'lr_end'                     # 最小学习率
'warmup_epochs'              # 热身轮次数
'weight_decay'               # 权重衰减
'momentum'                   # 动量
'opt_eps'                    # epsilon
'keep_checkpoint_max'        # 保存检查点的最大数量
'ckpt_path'                  # 保存检查点路径
'onnx_file'                  # 保存导出的ONNX模型路径
'is_save_on_master'          # 保存Rank0的检查点，分布式参数
'dropout_keep_prob'          # 保持率，介于0和1之间，例如keep_prob = 0.9，表示放弃10%的输入单元
'has_bias'                   # 层是否使用偏置向量
'amp_level'                  # `mindspore.amp.build_train_network`中参数`level`的选项，level表示混合
                             # 精准训练支持[O0, O2, O3]

```

## 训练过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

修改用到的yaml文件，默认为**default_config.yaml**文件，训练cifar10数据集时，**ds_type: cifar10**，训练imagenet数据集时，**ds_type: imagenet**．

```shell
# 分布式训练示例(8卡)
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CKPT_PATH]
# example: bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cifar10/ ./ckpt/

# 单机训练
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
# example: bash scripts/run_standalone_train.sh 0 /home/DataSet/cifar10/ ./ckpt/
```

> 1. RANK_TABLE_FILE可参考[链接](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools)生成。
>
> 2. 如不需要关于device_num和处理器总数的处理器核绑定操作，请删除scripts/run_distribute_train.sh中的taskset操作。

### 启动

``` launch
# 训练示例
  python:
      Ascend: python train.py --config_path default_config.yaml --dataset_path /dataset/train --platform Ascend
      CPU: python train.py --config_path CONFIG_FILE --dataset_path DATA_PATH --platform CPU

  shell:
      Ascend:
      # 分布式训练示例(8卡)
      bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CKPT_PATH]
      # example: bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cifar10/ ./ckpt/

      # 单机训练
      bash scripts/run_standalone_train.sh [DEVICE_ID] [DATA_PATH] [CKPT_PATH]
      # example: bash scripts/run_standalone_train.sh 0 /home/DataSet/cifar10/ ./ckpt/

      CPU:
      bash script/run_standalone_train_cpu.sh DATA_PATH ./ckpt
```

### 结果

训练结果保存在示例路径。checkpoint默认保存在`ckpt`路径下，训练日志会重定向到`./log.txt`，如下：

#### Ascend

```log
epoch:0 step:1251, loss is 5.7787247
Epoch time:360760.985, per step time:288.378
epoch:1 step:1251, loss is 4.392868
Epoch time:160917.911, per step time:128.631
```

#### CPU

```bash
epoch: 1 step: 390, loss is 2.7072601
epoch time: 6334572.124 ms, per step time: 16242.493 ms
epoch: 2 step: 390, loss is 2.5908582
epoch time: 6217897.644 ms, per step time: 15943.327 ms
epoch: 3 step: 390, loss is 2.5612416
epoch time: 6358482.104 ms, per step time: 16303.800 ms
...
```

## 评估过程

### 用法

使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

```shell
    bash run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
    # example: bash run_eval.sh 0 /home/DataSet/cifar10/ /home/model/inceptionv3/ckpt/inception_v3-rank0-2_1251.ckpt
```

- CPU:

```python
    bash run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

### 启动

``` launch
# 评估示例
  python:
      Ascend: python eval.py --config_path CONFIG_FILE --dataset_path DATA_DIR --checkpoint PATH_CHECKPOINT --platform Ascend
      CPU: python eval.py --config_path CONFIG_FILE --dataset_path DATA_PATH --checkpoint PATH_CHECKPOINT --platform CPU

  shell:
      Ascend: bash run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
      CPU: bash run_eval_cpu.sh DATA_PATH PATH_CHECKPOINT
```

> 训练过程中可以生成检查点。

### 结果

推理结果保存在示例路径，可以在`eval.log`中找到如下结果。

```log
metric:{'Loss':1.778, 'Top1-Acc':0.788, 'Top5-Acc':0.942}
```

## 模型导出

```shell
python export.py --config_path [CONFIG_FILE] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR", "ONNX"]

## ONNX模型导出及评估

### ONNX模型导出

```bash
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format "ONNX"
# example:python export.py --ckpt_file /home/models/official/cv/Inception/inceptionv3/inceptionv3_ascend_v160_imagenet2012_official_cv_top1acc78.69_top5acc94.3.ckpt --device_target "GPU" --file_format "ONNX"
```

### ONNX模型评估

```bash
    bash run_eval_onnx_gpu.sh [DEVICE_ID] [DATA_DIR] [PATH_ONNX]
    # example: bash run_eval_onnx_gpu.sh 2 /home/data/ /home/models/official/cv/Inception/inceptionv3/inceptionv3.onnx
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 使用方法

在推理之前需要在昇腾910环境上完成模型的导出。

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATA_PATH] [LABEL_FILE] [DEVICE_TYPE] [DEVICE_ID]
```

-注意：推理使用ImageNet数据集. 图片的标签是将所在文件夹排序后获得的从0开始的编号。该文件可以利用脚本导出，该脚本可以从`models/utils/cpp_infer/imgid2label.py`取得。

### 结果

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```python
accuracy:78.742
```

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend                                                  |
| -------------------------- | ------------------------------------------------------- |
| 模型版本                   | InceptionV3                                             |
| 资源                       | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8|
| 上传日期                   | 2021-07-05                                              |
| MindSpore版本              | 1.3.0                                                   |
| 数据集                     | 120万张图像                                             |
| Batch_size                 | 128                                                     |
| 训练参数                   | src/model_utils/default_config.yaml                     |
| 优化器                     | RMSProp                                                 |
| 损失函数                   | Softmax交叉熵                                           |
| 输出                       | 概率                                                    |
| 损失                       | 1.98                                                    |
| 总时长（8卡）              | 10小时                                                  |
| 参数(M)                    | 103M                                                    |
| 微调检查点                 | 313M                                                    |
| 训练速度                   | 单卡：1200img/s;8卡：9500 img/s                         |
| 脚本                       | [inceptionv3脚本](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Inception/inceptionv3) |

### 推理性能

| 参数             | Ascend                 |
| ------------------- | --------------------------- |
| 模型版本         | InceptionV3    |
| 资源             |  Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8|
| 上传日期         | 2021-07-05                  |
| MindSpore 版本   | 1.3.0                       |
| 数据集           | 5万张图像                  |
| Batch_size       | 128                         |
| 输出             | 概率                 |
| 准确率           | ACC1[78.8%] ACC5[94.2%]     |
| 总时长           | 2分钟                       |
| 推理模型         | 92M (.onnx文件)            |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
