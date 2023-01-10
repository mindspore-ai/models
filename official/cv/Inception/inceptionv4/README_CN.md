# 目录

- [InceptionV4描述](#inceptionv4描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [InceptionV4描述](#目录)

Inception-v4卷积神经网络架构，在之前Inception网络基础上进行了简化，并且使用了比Inception-v3更多的初始模块。该网络架构首次在2016年发表的Inception-v4、Inception-ResNet和残差连接对学习的影响相关论文中提出。

[论文](https://arxiv.org/pdf/1602.07261.pdf) Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Computer Vision and Pattern Recognition[J]. 2016.

# [模型架构](#目录)

InceptionV4的整体网络架构如下：

[链接](https://arxiv.org/pdf/1602.07261.pdf)

# [数据集](#目录)

有关所使用的数据集，请参考论文。

- 数据集：ImageNet2012
- 数据集大小：125G, 125万张彩色图像，1000个类别
    - 训练集：120G, 120万张图像
    - 测试集：5G, 5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。
- 数据路径：http://www.image-net.org/download-images

# [特性](#目录)

## [混合精度（Ascend）](#目录)

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
    -
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

- [ModelArts](https://support.huaweicloud.com/modelarts/)环境上运行

    ```python
    # Ascend环境上运行8卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"distribute=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在default_config.yaml文件中设置"lr_init=0.00004"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data'"。
    #          在default_config.yaml文件中设置"epoch_size=250"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"distribute=True"。
    #          在网页上添加"lr_init=0.00004"。
    #          在网页上添加"dataset_path=/cache/data"。
    #          在网页上添加"epoch_size=250"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/inceptionv4"。
    # （6）在网页上设置启动文件为“train.py”。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    #
    # Ascend环境上运行单卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data'"。
    #          在default_config.yaml文件中设置"epoch_size=250"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"dataset_path='/cache/data'"。
    #          在网页上添加"epoch_size=250"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/inceptionv4"。
    # （6）在网页上设置启动文件为“train.py”。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    #
    # Ascend环境上运行单卡评估
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
    #          在default_config.yaml文件中设置"checkpoint_path='./inceptionv4/inceptionv4-train-250_1251.ckpt'"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"checkpoint_url='s3://dir_to_your_trained_model/'"。
    #          在网页上添加"checkpoint_path='./inceptionv4/inceptionv4-train-250_1251.ckpt'"。
    #          在网页上添加"dataset_path='/cache/data'"。
    # （2）准备模型代码。
    #          在网页上添加其他参数。
    # （3）上传或复制训练好的模型到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/inceptionv4"。
    # （6）在网页上设置启动文件为"eval.py"。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    ```

- 在ModelArts上导出并开始评估（如果你想在ModelArts上运行，可以参考[ModelArts](https://support.huaweicloud.com/modelarts/)官方文档。

1. 使用voc val数据集在ModelArts上导出并评估多尺度翻转s8。

    ```python
    # （1）执行a或b。
    #       a. 在base_config.yaml文件中设置"enable_modelarts=True"。
    #          在base_config.yaml文件中设置"file_name='inceptionv4'"。
    #          在base_config.yaml文件中设置"file_format='MINDIR'"。
    #          在beta_config.yaml文件中设置"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在base_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在base_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"file_name='inceptionv4'"。
    #          在网页上添加"file_format='MINDIR'"。
    #          在网页上添加"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在网页上添加"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在网页上添加其他参数。
    # （2）上传或复制训练好的模型到S3桶。
    # （3）在网页上设置代码目录为"/path/inceptionv4"。
    # （4）在网页上设置启动文件为"export.py"。
    # （5）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （6）创建作业。
    ```

# [脚本说明](#目录)

## [脚本及示例代码](#目录)

```shell
.
└─Inception-v4
  ├─README.md
  ├─ascend310_infer                     # Ascend 310推理实现
  ├─scripts
    ├─run_distribute_train_gpu.sh       # GPU上运行8卡分布式训练
    ├─run_eval_gpu.sh                   # GPU上运行评估
    ├─run_eval_cpu.sh                   # CPU上运行评估
    ├─run_standalone_train_cpu.sh       # CPU上运行单机(单卡)训练
    ├─run_standalone_train_ascend.sh    # Ascend上运行单机(单卡)训练
    ├─run_distribute_train_ascend.sh    # Ascend上运行分布式(8卡)训练
    ├─run_infer_310.sh                  # 用于Ascend 310上运行推理的shell脚本
    ├─run_onnx_eval.sh                  # 用于onnx评估的shell脚本
    └─run_eval_ascend.sh                # Ascend上运行评估
  ├─src
    ├─dataset.py                      # 数据预处理
    ├─inceptionv4.py                  # 网络定义
    ├─callback.py                     # 评估回调函数
    └─model_utils
      ├─config.py               # 处理配置参数
      ├─device_adapter.py       # 获取云ID
      ├─local_adapter.py        # 获取本地ID
      └─moxing_adapter.py       # 参数处理
  ├─default_config.yaml             # （Ascend上）训练参数配置文件
  ├─default_config_cpu.yaml         # （CPU上）训练参数配置文件
  ├─default_config_gpu.yaml         # （GPU上）训练参数配置文件
  ├─eval.py                         # 评估脚本
  ├─eval_onnx.py                    # 用于评估onnx的脚本
  ├─export.py                       # 导出检查点，支持.onnx、.air、.mindir格式转换
  ├─postprogress.py                 # Ascend 310上推理的后处理脚本
  └─train.py                        # 训练网络
```

## [脚本参数](#目录)

```python
train.py和config.py中的主要涉及如下参数：
'is_save_on_master'          # 是否仅在主设备上保存检查点
'batch_size'                 # 输入批次大小
'epoch_size'                 # 总epoch数
'num_classes'                # 数据集类数
'work_nums'                  # 读取数据的工作线程数
'loss_scale'                 # 损失缩放
'smooth_factor'              # 标签平滑因子
'weight_decay'               # 权重衰减
'momentum'                   # 动量
'amp_level'                  # 精度训练，支持[O0, O2, O3]
'decay'                      # 优化函数中的衰减
'epsilon'                    # 优化器函数中使用的梯度
'keep_checkpoint_max'        # 保留检查点的的最大数量
'save_checkpoint_epochs'     # 每n个epochs保存一次检查点
'lr_init'                    # 初始学习率
'lr_end'                     # 结束学习率
'lr_max'                     # 最大学习率
'warmup_epochs'              # 预热epoch数
'start_epoch'                # 起始epoch范围[1, epoch_size]
```

## [训练过程](#目录)

### 用法

您可以使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

    ```yaml
    ds_type:imagenet
    或
    ds_type:cifar10
    以训练cifar10为例，ds_type参数配置为cifar10。
    ````

    ```bash
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_DIR]
    # 示例：bash scripts/run_distribute_train_ascend.sh ~/hccl_8p.json /home/DataSet/cifar10/

    # 单机训练
    bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATA_DIR]
    # 示例：bash scripts/run_standalone_train_ascend.sh 0 /home/DataSet/cifar10/
    ```

> 注：
> 有关RANK_TABLE_FILE，可参考[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html)。设备IP可参考[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。对于像InceptionV4这样的大型模型，最好设置外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将hccl连接检查时间从默认的120秒延长到600秒。否则，可能会连接超时，因为编译时间会随着模型增大而增加。
>
> 绑核操作取决于`device_num`参数值及处理器总数。如果不需要，删除`scripts/run_distribute_train.sh`脚本中的`taskset`操作任务集即可。

- GPU：

    ```bash
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train_gpu.sh DATA_PATH
    ```

- CPU：

    ```bash
    # shell脚本运行的单机训练示例
    bash scripts/run_standalone_train_cpu.sh DATA_PATH
    ```

### 启动

```bash
# 训练示例
  shell：
      Ascend：
      # 分布式训练示例（8卡）
      bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_DIR]
      # 示例：bash scripts/run_distribute_train_ascend.sh ~/hccl_8p.json /home/DataSet/cifar10/

      # 单机训练
      bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATA_DIR]
      # 示例：bash scripts/run_standalone_train_ascend.sh 0 /home/DataSet/cifar10/

      GPU：
      # 分布式训练示例（8卡）
      bash scripts/run_distribute_train_gpu.sh DATA_PATH
      CPU：
      # shell脚本运行的单机训练示例
      bash scripts/run_standalone_train_cpu.sh DATA_PATH
```

### 结果

训练结果将存储在示例路径中。检查点默认存储在`ckpt_path`路径下，训练日志重定向到`./log.txt`，示例如下：

- Ascend

    ```python
    epoch: 1 step: 1251, loss is 5.4833196
    Epoch time: 520274.060, per step time: 415.887
    epoch: 2 step: 1251, loss is 4.093194
    Epoch time: 288520.628, per step time: 230.632
    epoch: 3 step: 1251, loss is 3.6242008
    Epoch time: 288507.506, per step time: 230.622
    ```

- GPU

    ```python
    epoch: 1 step: 1251, loss is 6.49775
    Epoch time: 1487493.604, per step time: 1189.044
    epoch: 2 step: 1251, loss is 5.6884665
    Epoch time: 1421838.433, per step time: 1136.561
    epoch: 3 step: 1251, loss is 5.5168786
    Epoch time: 1423009.501, per step time: 1137.498
    ```

## [评估过程](#目录)

### 用法

您可以使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：

    ```bash
    bash scripts/run_eval_ascend.sh [DEVICE_ID] [DATA_DIR] [CHECKPOINT_PATH]
    # 示例：bash scripts/run_eval_ascend.sh 0 /home/DataSet/cifar10/ /home/model/inceptionv4/ckpt/inceptionv4-train-250_1251
    ```

- GPU

    ```bash
    bash scripts/run_eval_gpu.sh DATA_DIR CHECKPOINT_PATH
    ```

### 启动

```bash
# 评估示例
  shell：
      Ascend：
            bash scripts/run_eval_ascend.sh [DEVICE_ID] [DATA_DIR] [CHECKPOINT_PATH]
      GPU：
            bash scripts/run_eval_gpu.sh DATA_DIR CHECKPOINT_PATH
```

> 训练过程中会生成检查点。

### 结果

评估结果将存储在示例路径中，您可以在`eval.log`中查看。

- Ascend

```python
metric: {'Loss': 0.9849, 'Top1-Acc':0.7985, 'Top5-Acc':0.9460}
```

- GPU(8卡)

    ```python
    metric: {'Loss': 0.8144, 'Top1-Acc': 0.8009, 'Top5-Acc': 0.9457}
    ```

## 模型导出

```shell
python export.py --config_path [CONFIG_FILE] --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT`取值为["AIR", "MINDIR", "ONNX"]。

## 推理过程

**推理前，请参考[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)设置环境变量。**

### 用法

在执行推理前，我们需要先在Ascend 910环境通过导出脚本导出MINDIR文件。

```shell
# Ascend 310上运行推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

-注：使用ImageNet数据集在Ascend 310上进行推理。图像的标签序号为排序后的文件夹编号，从0开始。

在进行推理之前，我们需要先用脚本导出模型。

```shell
# ONNX推理
bash scripts/run_onnx_eval.sh [DATA_PATH] [DATASET_TYPE] [DEVICE_TYPE] [FILE_TYPE] [ONNX_PATH]
# 示例：bash scripts/run_onnx_eval.sh /path/to/dataset imagenet GPU ONNX /path/to/inceptionv4.onnx
```

-注：使用ImageNet数据集进行ONNX推理。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中查看。

```python
accuracy:80.044
```

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend                                       | GPU                             |
| -------------------------- | --------------------------------------------- | -------------------------------- |
| 模型版本             | InceptionV4                                  | InceptionV4                     |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；操作系统EulerOS 2.8 | NV SMX2 V100-32G                |
| 上传日期             | 11/04/2020                                   | 03/05/2021                      |
| MindSpore版本         | 1.0.0                                        | 1.0.0                           |
| 数据集                   | 120万张图像                                 | 120万张图像                    |
| Batch_size                | 128                                           | 128                              |
| 训练参数       | src/model_utils/default_config.yaml (Ascend)    | src/model_utils/default_config.yaml (GPU)|
| 优化器                 | RMSProp                                      | RMSProp                         |
| 损失函数             | SoftmaxCrossEntropyWithLogits                 | SoftmaxCrossEntropyWithLogits    |
| 输出                   | 概率                                  | 概率                     |
| 损失                      | 0.98486                                       | 0.8144                           |
| 准确率（8卡）             | ACC1[79.85%] ACC5[94.60%]                    | ACC1[80.09%] ACC5[94.57%]       |
| 总时长（8卡）           | 20h                                          | 95h                             |
| 参数量（M）                | 153M                                         | 153M                            |
| 微调检查点| 2135M                                        | 489M                            |
| 脚本                   | [Inceptionv4脚本](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) | [Inceptionv4脚本](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4)|

#### 推理性能

| 参数         | Ascend                                       | GPU                               |
| ------------------- | --------------------------------------------- | ---------------------------------- |
| 模型版本      | InceptionV4                                  | InceptionV4                       |
| 资源           | Ascend 910；CPU 2.60GHz, 192核；内存755G；操作系统EulerOS 2.8| NV SMX2 V100-32G                  |
| 上传日期      | 11/04/2020                                    | 03/05/2021                         |
| MindSpore版本  | 1.0.0                                        | 1.0.0                             |
| 数据集            | 5万张图像                                   | 5万张图像                        |
| Batch_size         | 128                                           | 128                                |
| 输出            | 概率                                  | 概率                       |
| 准确率           | ACC1[79.85%] ACC5[94.60%]                    | ACC1[80.09%] ACC5[94.57%]         |
| 总时长         | 2mins                                        | 2mins                             |
| 推理模型| 2135M（.ckpt文件）                           | 489M（.ckpt文件）                 |

#### 训练性能结果

| **Ascend**| 训练性能|
| :--------: | :---------------: |
|     单卡    |     556img/s    |

| **Ascend**| 训练性能|
| :--------: | :---------------: |
|     8卡    |     4430img/s   |

| **GPU**   | 训练性能|
| :--------: | :---------------: |
|     8卡    |     906img/s   |

# [随机情况说明](#目录)

在`dataset.py`中，我们设置了`create_dataset`函数内的种子，同时还使用了`train.py`中的随机种子。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
