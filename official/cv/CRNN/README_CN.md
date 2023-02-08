# 目录

- [目录](#目录)
    - [CRNN描述](#crnn描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
        - [数据集准备](#数据集准备)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
            - [训练脚本参数](#训练脚本参数)
            - [参数配置](#参数配置)
        - [数据集准备](#数据集准备)
    - [训练过程](#训练过程)
        - [训练](#训练)
            - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [训练时评估](#训练时评估)
        - [ONNX评估](#onnx评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend 310上推理](#在ascend310上推理)
        - [结果](#结果)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [MindSpore版本说明](#mindspore版本说明)
    - [ModelZoo主页](#modelzoo主页)

## [CRNN描述](#目录)

CRNN是一种基于图像序列识别的神经网络，应用于场景文本识别。本文研究了场景文本识别问题，这是基于图像序列识别中最重要和最具挑战的任务之一。本文提出了一种新的神经网络结构，将特征提取、序列建模和转录集成到统一框架中。与以前的场景文本识别系统相比，本文提及的架构具有四个特性：（1）端到端可训练，不像现有算法，大多数都是单独训练和调整组件。（2）自然处理任意长度序列，不涉及字符分割或水平尺度标准化。（3）不局限于预定义词典，在无词典和基于词典的场景文本识别任务中都取得了显著的性能。（4）生成有效且更小的模型，在现实的应用场景中更实用。

[论文](https://arxiv.org/abs/1507.05717): Baoguang Shi, Xiang Bai, Cong Yao, "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition", ArXiv, vol. abs/1507.05717, 2015.

## [模型架构](#目录)

CRNN使用vgg16结构进行特征提取，附加两层双向LSTM，最后使用CTC计算损失。有关详细信息，请参见src/crnn.py。

我们提供了2个版本的网络，使用不同的方法将hidden size传到class numbers。您可以通过修改config.yaml中的`model_version`来选择不同版本。

- V1中，RNN之后增加了全连接层。
- V2中，更改最后一个RNN的输出特征大小，输出具有相同分类数的特征。V2中，切换到内置`LSTM` cell，而不是`DynamicRNN`算子，这样GPU和Ascend上都支持该模型。

## [数据集](#目录)

注：可以运行原始论文中提到的数据集脚本，也可以运行在相关域/网络架构中广泛使用的脚本。下面将介绍如何使用相关数据集运行脚本。

我们使用论文中提到的五个数据集。在训练中，使用Jederberg等人发布的合成数据集（[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)和[SynthText](https://github.com/ankush-me/SynthText)）作为训练数据，其中包含800万张训练图像及其对应的地面真值词。在评估中，使用四个流行的场景文本识别基准，即ICDAR 2003([IC03](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions))、ICDAR2013（[IC13](https://rrc.cvc.uab.es/?ch=2&com=downloads))、IIIT 5k-word（[IIIT5k](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)）和街景文本（[SVT](http://vision.ucsd.edu/~kai/grocr/)）。

### [数据集准备](#目录)

对于数据集`IC03`、`IIIT5k`和`SVT`，不能直接在CRNN中使用官网的原始数据集。

- `IC03`，需要根据word.xml从原始图像中裁剪文本。
- `IIIT5k`，需要从matlib数据文件中提取标注。
- `SVT`，需要根据`train.xml`或`test.xml`从原始图像中裁剪文本。

我们提供了`convert_ic03.py`、`convert_iiit5k.py`、`convert_svt.py`作为上述预处理的示例参考。

## [环境要求](#目录)

- 硬件
    - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速入门](#目录)

- 准备好数据集后，可以开始运行训练或评估脚本，如下所示：

    - 在Ascend上运行

        ```shell
        # Ascend分布式训练示例
        $ bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] Ascend [RANK_TABLE_FILE] [RESUME_CKPT]

        # Ascend评估示例
        $ bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID] Ascend

        # Ascend单机训练示例
        $ bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [DEVICE_ID] Ascend [RESUME_CKPT]

        # Ascend 310离线推理
        $ bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DATASET] [DEVICE_ID]

        ```

    - 在GPU上运行

        ```shell
        # GPU分布式训练示例
        $ bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] GPU [RESUME_CKPT]

        # GPU评估示例
        $ bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] GPU

        # GPU单机训练示例
        $ bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] GPU
        ```

    - 在CPU上运行

        ```shell
        # CPU单机训练示例
        $ bash scripts/run_standalone_train_cpu.sh [DATASET_NAME] [DATASET_PATH] [RESUME_CKPT]

        # CPU评估示例
        $ bash scripts/run_eval_cpu.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
        ```

        DATASET_NAME取值范围：`ic03`、`ic13`、`svt`、`iiit5k`、`synth`。

        对于分布式训练，需要提前创建JSON格式的hccl配置文件。

        请按照以下链接中的说明操作：
        [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

- 在Docker上运行

    构建Docker镜像（将版本更改为实际使用的版本）

    ```shell
    # 构建Docker
    docker build -t ssd:20.1.0 . --build-arg FROM_IMAGE_NAME=ascend-mindspore-arm:20.1.0
    ```

    在创建的镜像上创建并启动一个容器层

    ```shell
    # 启动Docker
    bash scripts/docker_start.sh ssd:20.1.0 [DATA_DIR] [MODEL_DIR]
    ```

    然后可以像在Ascend上一样运行。

## [脚本说明](#目录)

### [脚本及样例代码](#目录)

```shell
crnn
├── README.md                                   # CRNN描述
├── convert_ic03.py                             # 转换原始IC03数据集
├── convert_iiit5k.py                           # 转换原始IIIT5K数据集
├── convert_svt.py                              # 转换原始SVT数据集
├── requirements.txt                            # 数据集要求
├── scripts
│   ├── run_standalone_train_cpu.sh             # 在CPU中启动单机训练
│   ├── run_eval_cpu.sh                         # 在CPU中启动评估
│   ├── run_distribute_train.sh                 # 在Ascend或GPU中启动分布式训练（8卡）
│   ├── run_eval.sh                             # 在Ascend或GPU中启动评估
│   └── run_standalone_train.sh                 # 在Ascend或GPU中启动单机训练（单卡）
│   └── run_eval_onnx.sh                        # Eval ONNX模型
├── src
│   ├── model_utils
│       ├── config.py                           # 参数配置
│       ├── moxing_adapter.py                   # ModelArts设备配置
│       └── device_adapter.py                   # 设备配置
│       └── local_adapter.py                    # 本地设备配置
│   ├── crnn.py                                 # CRNN网络定义
│   ├── crnn_for_train.py                       # CRNN网络，带梯度、损失和梯度裁剪
│   ├── dataset.py                              # 训练和评估数据预处理
│   ├── eval_callback.py
│   ├── ic03_dataset.py                         # IC03数据预处理
│   ├── ic13_dataset.py                         # IC13数据预处理
│   ├── iiit5k_dataset.py                       # IIIT5K数据预处理
│   ├── loss.py                                 # CTC损失定义
│   ├── metric.py                               # CRNN网络的准确率指标
│   └── svt_dataset.py                          # SVT数据预处理
└── train.py                                    # 训练脚本
├── eval.py                                     # 评估脚本
├── eval_onnx.py                                # ONNX模型评估脚本
├── default_config.yaml                         # 配置文件

```

### [脚本参数](#目录)

#### 训练脚本参数

```shell
# Ascend或GPU分布式训练
用法：bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM] [RANK_TABLE_FILE](if Ascend) [RESUME_CKPT](optional for resume)

# Ascend或GPU单机训练
用法：bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM] [RESUME_CKPT](optional for resume)

# CPU单机训练
用法：bash scripts/run_standalone_train_cpu.sh [DATASET_NAME] [DATASET_PATH] [RESUME_CKPT](optional for resume)
```

#### 参数配置

可以在default_config.yaml中设置训练和评估的参数。

```yaml
"max_text_length": 23,                       # 最大文本长度
"image_width": 100,                          # 文本图像宽度
"image_height": 32,                          # 文本图像高度
"batch_size": 64,                            # 输入张量的batch size
"epoch_size": 10,                            # 仅对训练有效，始终为1
"hidden_size": 256,                          # LSTM层中的hidden size
"learning_rate": 0.02,                       # 初始学习率
"momentum": 0.95,                            # SGD优化器的动量
"nesterov": True,                            # 在SGD优化程序中启用nesterov
"save_checkpoint": True,                     # 是否保存检查点
"save_checkpoint_steps": 1000,               # 两个检查点之间的步长间隔
"keep_checkpoint_max": 30,                   # 仅保留最后一个keep_checkpoint_max
"save_checkpoint_path": "./",                # 保存检查点的路径
"class_num": 37,                             # 数据集分类数
"input_size": 512,                           # LSTM层中的输入大小
"num_step": 24,                              # LSTM层的步数
"use_dropout": True,                         # 是否使用dropout
"blank": 36                                  # 为分类添加空白
"train_dataset_path": ""                     # 训练集路径
"train_eval_dataset": "svt"                  # 训练集名称，选项[synth, ic03, ic13, svt, iiit5k]
"train_eval_dataset_path": ""                # eval数据集路径
"run_eval": False                            # 在训练时运行评估，默认值为False。
"eval_all_saved_ckpts": False                # 为eval加载所有检查点，默认值为False。
"save_best_ckpt": True                       # 当run_eval为True时保存最佳检查点，默认值为True。
"eval_start_epoch": 5                        # 当run_eval为True时，评估开始轮次，默认值为5。
"eval_interval": 1                           # 当run_eval为True时，评估间隔，默认值为5。
```

### [数据集准备](#目录)

- 可参考[快速入门](#快速入门)中的“生成数据集”自动生成数据集，也可以选择自行生成文本图像数据集。

## [训练过程](#目录)

- 设置`config.py`中的选项，包括学习率和其他网络超参。有关数据集的更多信息，请参阅[MindSpore数据集准备教程](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset.html)。

### [训练](#目录)

- 运行`run_standalone_train.sh`进行CRNN模型的非分布式训练，目前支持Ascend和GPU。

    ``` bash
    bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM](optional) [RESUME_CKPT](optional for resume)
    ```

- 或者在CPU中运行`run_standalone_train_cpu.sh`进行CRNN模型的非分布式训练。

    ``` bash
    bash scripts/run_standalone_train_cpu.sh [DATASET_NAME] [DATASET_PATH] [RESUME_CKPT](optional for resume)
    ```

#### [分布式训练](#目录)

- 在Ascend或GPU上运行`run_distribute_train.sh`进行CRNN模型的分布式训练

    ``` bash
    bash scripts/run_distribute_train.sh [DATASET_NAME] [DATASET_PATH] [PLATFORM] [RANK_TABLE_FILE](if Ascend) [RESUME_CKPT](optional for resume)
    ```

    检查`train_parallel0/log.txt`，将得到以下输出：

    ```shell
    epoch: 10 step: 14110, loss is 0.0029097411
    Epoch time: 2743.688s, per step time: 0.097s
    ```

- 在ModelArts上运行
- 请参考ModelArts[官方指导文档](https://support.huaweicloud.com/modelarts/)。

    ```python
    #  在ModelArts上使用分布式训练DPN：
    #  数据集目录结构

    #  ├── crnn_dataset                                             # 数据集目录
    #    ├──train                                                   # 训练目录
    #      ├── mnt                                                  # 训练集目录
    #      ├── pred_trained                                         # 预训练目录
    #    ├── eval                                                   # eval目录
    #      ├── IIIT5K-Word_V3.0                                     # eval数据集目录
    #      ├── checkpoint                                           # 检查点目录
    #      ├── svt                                                  # 检查点目录

    # （1）执行a（修改yaml文件参数）或b（ModelArts创建训练作业以修改参数）。
    #       a. 设置"enable_modelarts=True"
    #          设置"run_distribute=True"
    #          设置"save_checkpoint_path=/cache/train/checkpoint"
    #          设置"train_dataset_path=/cache/data/mnt/ramdisk/max/90kDICT32px"
    #
    #       b. 在ModelArts界面添加"enable_modelarts=True"参数
    #          在ModelArts界面设置方法a所需的参数
    #          注：path参数不需要用引号括起来。

    # （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
    # （3）在ModelArts界面设置代码路径"/path/crnn"
    # （4）在ModelArts界面设置模型的启动文件"train.py"
    # （5）在ModelArts界面设置模型的数据路径".../crnn_dataset/train"（选择crnn_dataset/train文件夹路径）
    # 模型的输出路径"Output file path"和模型的日志路径"Job log path"
    # （6）开始训练模型

    # 在ModelArts上使用模型推理
    # （1）将训练好的模型放置到桶的对应位置
    # （2）执行a或者b
    #        a. 设置"enable_modelarts=True"
    #          设置"eval_dataset=svt"或eval_dataset=iiit5k
    #          设置"eval_dataset_path=/cache/data/svt/converted/img/"或eval_dataset_path=/cache/data/IIIT5K-Word_V3/IIIT5K/
    #          设置"CHECKPOINT_PATH=/cache/data/checkpoint/checkpoint file name"

    #       b. 在ModelArts界面添加"enable_modelarts=True"参数
    #          在ModelArts界面设置方法a所需的参数
    #          注：path参数不需要用引号括起来。

    # （3）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
    # (4)在ModelArts界面设置代码路径"/path/crnn"
    # (5)在ModelArts界面设置模型的启动文件"eval.py"
    # （6）在ModelArts界面设置模型的数据路径".../crnn_dataset/eval"（选择crnn/eval文件夹路径）
    # 模型的输出路径"Output file path"和模型的日志路径"Job log path"
    # （7）开始模型推理
    ```

#### [断点续训练](#目录)

- 如果想使用断点续训练功能，运行训练脚本时，[RESUME_CKPT]参数指定对应的checkpoint文件即可。

## [评估过程](#目录)

### [评估](#目录)

- 运行`run_eval.sh`进行评估。

    ``` bash
    bash scripts/run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID] [PLATFORM](optional)

    bash scripts/run_eval_cpu.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH]
    ```

    检查`eval/log.txt`，将得到以下输出：

    ```shell
    result: {'CRNNAccuracy': (0.806)}
    ```

### 训练时评估

添加并设置`run_eval`为True来启动shell，还需添加`eval_dataset`来选择评估数据集。如果希望在训练时进行评估，添加eval_dataset_path来启动shell。当`run_eval`为True时，可设置参数选项：`save_best_ckpt`、`eval_start_epoch`、`eval_interval`。

### [ONNX评估](#目录)

- 将模型导出到ONNX：

  ```shell
  python export.py --device_target GPU --ckpt_file /path/to/deeptext.ckpt --file_name crnn.onnx --file_format ONNX --model_version V2
  ```

- 运行ONNX评估脚本：

  ```shell
  bash scripts/run_eval_onnx.sh [DATASET_NAME] [DATASET_PATH] [ONNX_MODEL] [DEVICE_TARGET]
  ```

  评估结果将保存在log.txt文件中，格式如下：

  ```text
  correct num:  2392 , total num:  3000
  result: 0.7973333333333333
  ```

## [推理过程](#目录)

### [导出MindIR](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --device_target [DEVICE_TARGET] --model_version [MODEL_VERSION](required for cpu)
```

必须设置ckpt_file参数。
`FILE_FORMAT`：取值范围["AIR", "MINDIR"]。

- 在ModelArts上导出MindIR

  ```Modelarts
  在ModelArts上导出MindIR
  数据存储方法同训练
  # （1）执行a（修改yaml文件参数）或b（ModelArts创建训练作业以修改参数）
  #       a. 设置"enable_modelarts=True"
  #          设置"file_name=crnn"
  #          设置"file_format=MINDIR"
  #          设置"ckpt_file=/cache/data/checkpoint file name"

  #       b. 在ModelArts界面添加"enable_modelarts=True"参数
  #          在ModelArts界面设置方法a所需的参数
  #          注：path参数不需要用引号括起来。
  # （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
  # （3）在ModelArts界面设置代码路径"/path/crnn"
  # （4）在ModelArts界面设置模型的启动文件"export.py"
  # （5）在ModelArts界面设置模型的数据路径".../crnn_dataset/eval/checkpoint"（选择crnn_dataset/eval/checkpoint文件夹路径）
  # 模型的输出路径"Output file path"和模型的日志路径"Job log path"。
  ```

### 在Ascend 310上推理

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

在执行推理之前，必须在Ascend 910环境上通过导出脚本导出MINDIR文件。下面以使用MINDIR模型推理为例。
当前batch_Size只能设置为1。推理结果只是网络输出，保存在二进制文件中。准确率由`src/metric.`计算。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DATASET] [DEVICE_ID]
```

`MINDIR_PATH`：通过export.py导出的MINDIR模型。  
`DATA_PATH`：数据集的路径。如果必须转换数据，请将路径传递到转换数据。  
`ANN_FILE_PATH`：标注文件的路径。对于转换的数据，标注文件通过转换脚本导出。  
`DATASET`：数据集名称，取值范围为["synth", "svt", "iiit5k", "ic03", "ic13"]。  
`DEVICE_ID`：可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中查看如下结果。

```shell
correct num: 2042 , total num: 3000
result CRNNAccuracy is: 0.806666666666
```

## [模型说明](#目录)

### [性能](#目录)

#### [训练性能](#目录)

| 参数                | Ascend 910                                        | Tesla V100                                        |
| -------------------------- | --------------------------------------------------|---------------------------------------------------|
| 模型版本             | v1.0                                              | v2.0                                              |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8    |  Tesla V100; CPU 2.60GHz, 72核；内存256G；操作系统Ubuntu 18.04.3|
| 上传日期             | 12/15/2020                      | 6/11/2021                       |
| MindSpore版本         | 1.0.1                                             | 1.2.0                                             |
| 数据集                   | Synth                                             | Synth                                             |
| 训练参数       | epoch=10, steps per epoch=14110, batch_size = 64  | epoch=10, steps per epoch=14110, batch_size = 64  |
| 优化器                 | SGD                                               | SGD                                               |
| 损失函数             | CTCLoss                                           | CTCLoss                                           |
| 输出                   | 概率                                      | 概率                                      |
| 损失                      | 0.0029097411                                      | 0.0029097411                                      |
| 速度                     | 118 ms/step (8卡)                                  | 36 ms/step (8卡)                                   |
| 总时长                | 557 mins                                          | 189 mins                                          |
| 参数量（M）            | 83M (.ckpt)                                 | 96M                                              |
| 微调检查点| 20.3M (.ckpt)                                |                                                   |
| 脚本                   | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) |

#### [评估性能](#目录)

| 参数         | SVT                        | IIIT5K                      | SVT                         | IIIT5K                      |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 模型版本      | V1.0                       | V1.0                       | V2.0                       | V2.0                       |
| 资源           | Ascend 910; EulerOS 2.8     | Ascend 910                  | Tesla V100                  | Tesla V100                  |
| 上传日期      | 12/15/2020| 12/15/2020| 6/11/2021 | 6/11/2021 |
| MindSpore版本  | 1.0.1                       | 1.0.1                       | 1.2.0                       | 1.2.0                       |
| 数据集            | SVT                        | IIIT5K                      | SVT                        | IIIT5K                      |
| batch_size          | 1                           | 1                           | 1                           | 1                           |
| 输出            | ACC                        | ACC                        | ACC                        | ACC                        |
| 准确率           | 80.8%                       | 79.7%                       | 81.92%                      | 80.2%                       |
| 推理模型| 83M (.ckpt)           | 83M (.ckpt)           | 96M (.ckpt)           | 96M (.ckpt)           |

| 参数         | IC13        |
| ------------------- |-------------|
| 模型版本      | V1.0        |
| 资源           | Ascend 910  |
| 上传日期      | 02/08/2023  |
| MindSpore版本  | 2.0.0       |
| 数据集            | SYNTH       |
| batch_size          | 16          |
| 输出            | ACC         |
| 准确率           | 92.9%       |
| 推理模型| 110.5M (.ckpt) |

## [随机情况说明](#目录)

dataset.py中设置了“create_dataset”函数内的种子。我们还在train.py中使用随机种子进行权重初始化。

# [MindSpore版本说明](#目录)

由于MindSpore(>1.5)的升级，从旧版本训练的检查点文件无法直接加载到网络中。

## [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
