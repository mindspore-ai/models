# 目录

- [目录](#目录)
- [Xception描述](#xception描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
        - [启动](#启动-1)
        - [结果](#结果-1)
    - [导出过程](#导出过程)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [Xception描述](#目录)

谷歌的Xception是继Inception后的又一新版本。使用改良后的深度可分离卷积，效果甚至比Inception-v3更好。该论文发表于2017年。

[论文](https://arxiv.org/pdf/1610.02357v3.pdf) Franois Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) IEEE, 2017.

# [模型架构](#目录)

Xception的整体网络架构如下：

[链接](https://arxiv.org/pdf/1610.02357v3.pdf)

# [数据集](#目录)

使用的数据集可参考论文。

- 数据集大小：125G，125万张彩色图像，1000个分类
    - 训练集：120G，120万张图像
    - 测试集：5G，5万张图像
- 数据格式：RGB
    - 注：数据将在**src/dataset.py**中处理。

# [特性](#目录)

## [混合精度](#目录)

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# [环境要求](#目录)

- 硬件
    - 使用Ascend或GPU搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```shell
.
└─Xception
  ├─README.md
  ├─ascend310_infer                # 请求Ascend 310推理
  ├─scripts
    ├─run_standalone_train.sh      # 在Ascend上进行单机训练（单卡）
    ├─run_distribute_train.sh      # 在Ascend上进行分布式训练（8卡）
    ├─run_train_gpu_fp32.sh        # 在GPU上进行单机训练或FP32算子的分布式训练（单卡或8卡）
    ├─run_train_gpu_fp16.sh        # 在GPU上进行单机训练或FP16算子的分布式训练（单卡或8卡）
    ├─run_eval.sh                  # 在Ascend上进行评估
    ├─run_infer_310.sh             # 在Ascend 310上推理的shell脚本
    └─run_eval_gpu.sh              # 在GPU上进行评估
  ├─src
    ├─model_utils
        ├─config.py                # 解析yaml的参数配置文件
        ├─device_adapter.py        # 本地或ModelArts培训
        ├─local_adapter.py         # 在本地获取相关环境变量
        └─moxing_adapter.py        # 获取相关环境变量，并在ModelArts上传输数据
    ├─dataset.py                   # 数据预处理
    ├─Xception.py                  # 定义网络
    ├─loss.py                      # 定义损失函数CrossEntropy
    └─lr_generator.py              # 学习速率生成器
  ├─default_config.yaml            # 参数配置
  ├─mindspore_hub_conf.py          # Mindpore Hub接口
  ├─train.py                       # 训练网络
  ├─postprogress.py                # Ascend 310推理后的处理脚本
  ├─export.py                      # 导出网络
  └─eval.py                        # 评估网络

```

## [脚本参数](#目录)

训练和评估的参数都可以在`default_config.yaml`中设置。

- 在Ascend中设置

    ```python
    train.py和config.py中主要的参数有：
    'num_classes': 1000                # 数据集类别数
    'batch_size': 128                  # 输入的批处理大小
    'loss_scale': 1024                 # 损失缩放
    'momentum': 0.9                    # 动量
    'weight_decay': 1e-4               # 权重衰减
    'epoch_size': 250                  # 模型epoch总数
    'save_checkpoint': True            # 保存CKPT文件
    'save_checkpoint_epochs': 1        # 每迭代相应次数保存一个ckpt文件
    'keep_checkpoint_max': 5           # 保存CKPT文件的最大数量
    'save_checkpoint_path': "./"       # 保存CKPT文件路径
    'warmup_epochs': 1                 # 热身epoch数
    'lr_decay_mode': "liner"           # 学习率下降方式
    'use_label_smooth': True           # 是否使用标签平滑
    'finish_epoch': 0                  # 完成epoch数
    'label_smooth_factor': 0.1         # 标签平滑因子
    'lr_init': 0.00004                 # 学习率初始化
    'lr_max': 0.4                      # 最大学习率
    'lr_end': 0.00004                  # 最小学习率
    ```

- 在GPU上设置

    ```python
    train.py和config.py主要的参数有：
    'num_classes': 1000                # 数据集类别数
    'batch_size': 64                   # 输入的批处理大小
    'loss_scale': 1024                 # 损失缩放
    'momentum': 0.9                    # 动量
    'weight_decay': 1e-4               # 权重衰减
    'epoch_size': 250                  # 模型epoch总数
    'save_checkpoint': True            # 保存CKPT文件
    'save_checkpoint_epochs': 1        # 每迭代相应次数保存一个ckpt文件
    'keep_checkpoint_max': 5           # 保存CKPT文件的最大数量
    'save_checkpoint_path': "./gpu-ckpt"       # 保存CKPT文件的路径
    'warmup_epochs': 1                 # 热身epoch数
    'lr_decay_mode': "linear"          # 学习率下降方式
    'use_label_smooth': True           # 是否使用标签平滑
    'finish_epoch': 0                  # 完成epoch数
    'label_smooth_factor': 0.1         # 标签平滑因子
    'lr_init': 0.00004                 # 学习率初始化
    'lr_max': 0.4                      # 最大学习率
    'lr_end': 0.00004                  # 最小学习率
    ```

## [训练过程](#目录)

### 用法

您可以使用python命令或运行shell脚本来进行训练。shell脚本的用法如下：

- Ascend：

    ```shell
    # 分布式训练示例（8卡）
    bash scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    bash scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
    ```

- GPU：

    ```shell
    # FP32分布式训练示例（8卡）
    bash scripts/run_train_gpu_fp32.sh DEVICE_NUM DATASET_PATH PRETRAINED_CKPT_PATH(optional)

    # FP32单机训练示例
    bash scripts/run_train_gpu_fp32.sh 1 DATASET_PATH PRETRAINED_CKPT_PATH(optional)

    # FP16分布式训练示例（8卡）
    bash scripts/run_train_gpu_fp16.sh DEVICE_NUM DATASET_PATH PRETRAINED_CKPT_PATH(optional)

    # FP16单机训练示例
    bash scripts/run_train_gpu_fp16.sh 1 DATASET_PATH PRETRAINED_CKPT_PATH(optional)

    # 推理示例
    bash run_eval_gpu.sh DEVICE_ID DATASET_PATH CHECKPOINT_PATH

    #Ascend 310推理示例
    bash run_infer_310.sh MINDIR_PATH DATA_PATH LABEL_FILE DEVICE_ID
    ```

> 注：RANK_TABLE_FILE可以参考[链接](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)，device_ip可以参考[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。

### 启动

``` shell
# 训练示例
  python：
      Ascend：
      python train.py --device_target Ascend --train_data_dir /dataset/train
      GPU：
      python train.py --device_target GPU --train_data_dir /dataset/train

  shell：
      Ascend：
      # 分布式训练示例（8卡）
      bash scripts/run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
      # 单机训练
      bash scripts/run_standalone_train.sh DEVICE_ID DATA_PATH
      GPU：
      # FP16训练示例（8卡）
      bash scripts/run_train_gpu_fp16.sh DEVICE_NUM DATA_PATH
      # FP32训练示例（8卡）
      bash scripts/run_train_gpu_fp32.sh DEVICE_NUM DATA_PATH
```

### 结果

训练结果将存储在示例路径中。默认情况下，CKPT文件将存储在Ascend的`./ckpt_0`和GPU的`./gpu_ckpt`，训练日志将重定向到Ascend的`log.txt`和GPU的`log_gpu.txt`，如下所示。

- Ascend：

    ``` shell
    epoch: 1 step: 1251, loss is 4.8427444
    epoch time: 701242.350 ms, per step time: 560.545 ms
    epoch: 2 step: 1251, loss is 4.0637593
    epoch time: 598591.422 ms, per step time: 478.490 ms
    ```

- GPU：

    ``` shell
    epoch: 1 step: 20018, loss is 5.479554
    epoch time: 5664051.330 ms, per step time: 282.948 ms
    epoch: 2 step: 20018, loss is 5.179064
    epoch time: 5628609.779 ms, per step time: 281.177 ms
    ```

### ModelArts上运行8卡训练

  如果您想在Modelarts中运行，请查看[ModelArts](https://support.huaweicloud.com/modelarts/)的官方文档，您可以按照以下方式开始训练：

  ```python
  # （1）将代码文件夹上传到S3桶中；
  # （2）在网页上单击创建训练任务；
  # （3）在网页上设置代码目录为"/{path}/xception"；
  # （4）在网页上设置启动文件为"/{path}/xception/train.py"；
  # （5）执行a或b。
  #     a. 在/{path}/xception/default_config.yaml中设置参数。
  #         1、设置”enable_modelarts: True“；
  #         2、设置“is_distributed: True”；
  #         3、如果数据是以zip包的形式上传的，请设置“modelarts_dataset_unzip_name: {folder_name}"。
  #         4、设置“folder_name_under_zip_file: {path}”，（解压缩文件夹下的数据集路径，如./ImageNet_Original/train）
  #     b.在网页上添加参数。
  #         1、添加”enable_modelarts=True“；
  #         2、添加“is_distributed: True”；
  #         3、如果数据是以zip包的形式上传的，请添加“modelarts_dataset_unzip_name: {folder_name}"。
  #         4.添加“folder_name_under_zip_file: {path}”，（解压缩文件夹下的数据集路径，如./ImageNet_Original/train）
  # (6)将MindRecord数据集上传至S3桶；
  # （7）在网页上检查数据存储位置"data storage location"，设置数据集路径"Dataset path"；
  # （8）在网页上上设置输出文件路径"Output file path"和任务日志路径"Job log path"；
  # （9）在资源池选项下，选择8卡；
  # （10）创建作业；
  ```

## [评估过程](#目录)

### 用法

您可以使用python命令或运行shell脚本来进行训练。shell脚本的用法如下：

- Ascend：

    ```shell
    bash scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
    ```

- GPU：

    ```shell
    bash scripts/run_eval_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
    ```

### 启动

```shell
# 评估示例
  python:
      Ascend: python eval.py --device_target Ascend --checkpoint_path PATH_CHECKPOINT --test_data_dir DATA_DIR
      GPU: python eval.py --device_target GPU --checkpoint_path PATH_CHECKPOINT --test_data_dir DATA_DIR

  shell：
      Ascend: bash scripts/run_eval.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
      GPU: bash scripts/run_eval_gpu.sh DEVICE_ID DATA_DIR PATH_CHECKPOINT
```

> 可以在训练过程中生成CKPT文件。

### 结果

评估结果将存储在示例路径中，您可以在Ascend的`eval.log`和GPU的`eval_gpu.log`中查看如下结果。

- 在Ascend上进行评估

    ```shell
    result: {'Loss': 1.7797744848789312, 'Top_1_Acc': 0.7985777243589743, 'Top_5_Acc': 0.9485777243589744}
    ```

- 在GPU上进行评估

    ```shell
    result: {'Loss': 1.7846775874590903, 'Top_1_Acc': 0.798735595390525, 'Top_5_Acc': 0.9498439500640204}
    ```

### 在ModelArts上进行单卡评估

如果您想在Modelarts中运行，请查看[ModelArts](https://support.huaweicloud.com/modelarts/)的官方文档，您可以按照以下方式开始训练：

```python
# （1）将代码文件夹xception上传到S3桶；
# （2）在网页上单击创建训练任务；
# （3）在网页上设置代码目录为"/{path}/xception"；
# （4）在网页上设置启动文件为"/{path}/xception/eval.py"
# （5）执行a或b；
#     a. 在/{path}/xception/default_config.yaml中设置参数；
#         1. 设置”enable_modelarts: True“；
#         2. 设置“checkpoint_path: ./{path}/*.ckpt”（'load_checkpoint_path'表示待评估的权重文件相对于`eval.py`文件的路径，权重文件必须包含在代码目录中）；
#         3. 如果数据是以zip包的形式上传的，请设置“modelarts_dataset_unzip_name: {folder_name}"；
#         4.设置“folder_name_under_zip_file: {path}”，（解压缩文件夹下的数据集路径，如./ImageNet_Original/validation_preprocess）。
#     b.在网页上添加参数。
#         1. 添加”enable_modelarts: True“；
#         2. 添加“checkpoint_path: ./{path}/*.ckpt”（'load_checkpoint_path'表示待评估的权重文件相对于`eval.py`文件的路径，权重文件必须包含在代码目录中）；
#         3. 如果数据是以zip包的形式上传的，请添加“modelarts_dataset_unzip_name: {folder_name}"；
#         4、添加“folder_name_under_zip_file: {path}”，（解压缩文件夹下的数据集路径，如./ImageNet_Original/validation_preprocess）。
# （6）将数据集上传到S3桶（非MindRecord格式）；
# （7）在网页上检查数据存储位置"data storage location"，设置数据集路径"Dataset path"；
# （8）在网页上上设置输出文件路径"Output file path"和任务日志路径"Job log path"；
# （9）在资源池选项下，选择单卡；
# （10）创建作业。
```

## [导出过程](#目录)

- 导出到本地

  ```shell
  python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT] --batch_size [BATCH_SIZE]
  ```

`EXPORT_FORMAT`可选值为AIR或MINDIR。

- 在ModelArts上导出（如果您想在ModelArts中运行，请查看[ModelArts](https://support.huaweicloud.com/modelarts/)，您可以以如下方式启动。

  ```python
  # （1）将代码文件夹上传到S3桶中；
  # （2）在网页上单击创建训练任务；
  # （3）在网页上设置代码目录为"/{path}/xception"；
  # （4）在网页上设置启动文件为"/{path}/xception/export.py"；
  # （5）执行a或b；
  #     a. 在/{path}/xception/default_config.yaml中设置参数；
  #         1. 设置”enable_modelarts: True“；
  #         2. 设置”ckpt_file: ./{path}/*.ckpt”（'ckpt_file'参数表示待导出的权重文件相对于`export.py`文件的路径，权重文件必须包含在代码目录中）；
  #         3. 设置”file_name: xception“；
  #         4. 设置”file_format：MINDIR“；
  #     b.在网页上上添加参数；
  #         1. 添加”enable_modelarts=True“；
  #         2. 添加“ckpt_file=./{path}/*.ckpt”（'ckpt_file'参数表示待导出的权重文件相对于`export.py`文件的路径，权重文件必须包含在代码目录中）；
  #         3. 添加”file_name=xception“；
  #         4. 添加”file_format=MINDIR“;
  # （7）查看网页上的数据存储位置"data storage location"，并设置"Dataset path"（必要步骤）；
  # （8）在网页上上设置输出文件路径"Output file path"和任务日志路径"Job log path"；
  # （9）在资源池选项下，选择单卡；
  # （10）创建作业。
  # 您将在{Output file path}下查看xception.mindir。
  ```

## [推理过程](#目录)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在推理之前，我们需要先导出模型。AIR模型只能在Ascend 910中导出，MindIR模型可以在任何环境中导出。
注意当前batch_size只能设置为1。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

-注：ImageNet数据集用于dnsnet121。图像在文件夹排序后从0开始编号。

推理结果将存储在脚本路径中，您可以在**acc.log**中查看如下结果。

```shell
Top_1_Acc: 0.79886%, Top_5_Acc: 0.94882%
```

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend                   | GPU                      |
| -------------------------- | ------------------------- | ------------------------- |
| 模型版本             | Xception                 | Xception                 |
| 资源                  | 华为云ModelArts   | 华为云ModelArts   |
| 上传日期             | 12/10/2020               | 02/09/2021               |
| MindSpore版本         | 1.1.0                    | 1.1.0                    |
| 数据集                   | 120万张图像             | 120万张图像             |
| Batch_size                | 128                      | 64                       |
| 训练参数       | src/config.py            | src/config.py            |
| 优化器                 | 动量                 | 动量                 |
| 损失函数             | CrossEntropySmooth       | CrossEntropySmooth       |
| 损失                      | 1.78                     | 1.78                      |
| 准确率（8卡）             | Top1（79.8%）Top5（94.8%）  | Top1（79.8%）Top5（94.9%）  |
| 速度（8卡）        | 479毫秒/步              | 282毫秒/步              |
| 总时长（8卡）           | 42小时                      | 51小时                      |
| 参数量（M）                | 180M                     | 180M                     |
| 脚本                   | [Xception](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception)| [Xception](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception)|

#### 推理性能

| 参数         | Ascend                     | GPU                     |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本      | Xception                   | Xception                   |
| 资源           | 华为云ModelArts     | 华为云ModelArts     |
| 上传日期      | 12/10/2020                 | 02/09/2021                 |
| MindSpore版本  | 1.1.0                      | 1.1.0                      |
| 数据集            | 5万张图像                 | 5万张图像                 |
| Batch_size         | 128                        | 64                        |
| 准确率           | Top1（79.8%）Top5（94.8%）    | Top1（79.8%）Top5（94.9%）    |
| 总时长         | 3分钟                      | 4.7分钟                      |

# [随机情况说明](#目录)

在`dataset.py`中，我们设置了`create_dataset`函数内的种子。我们还在`train.py`中使用随机种子。

# [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
