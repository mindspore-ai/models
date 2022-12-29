# 目录

- [Tacotron2描述](#tacotron2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [推理过程](#推理过程)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [Tacotron2描述](#目录)

Tacotron2是一个TTS模型，包含两个阶段：第一阶段采用序列对序列的方法从文本序列中预测梅尔频谱，
第二阶段应用WaveNet作为声码器，将梅尔频谱转换为波形。我们支持在Ascend上训练和评估Tacotron2模型。

[论文](https://arxiv.org/abs/1712.05884): Jonathan, et al. Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.

# [模型架构](#目录)

Tacotron2实质上是一种包含编码器和解码器的序列到序列模型。编码器由三个卷积神经网络层和一个BiLSTM层实现，解码器使用两个LSTM层来解码下一个状态，在编码器和解码器之间应用位置感知器。然后将解码后的状态反馈到由5个卷积神经网络层组成的PostNet中，用来预测梅尔频谱。最后将预测好的梅尔频谱特征反馈到WaveNet声码器中，合成语音信号。

# [数据集](#目录)

我们接下来将介绍如何使用下面的数据集来运行脚本。

使用的数据集：[The LJ Speech Dataset](<https://keithito.com/LJ-Speech-Dataset>)

- 数据集大小：2.6GB
- 数据格式：13100个音频片段和转录文稿

- 数据集结构如下：

    ```text
    .
    └── LJSpeech-1.1
        ├─ wavs                  //音频片段文件
        └─ metadata.csv           //转录文件
    ```

# [环境要求](#目录)

- 硬件
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行

  ```python
  # 安装Pyhon3安装包
  pip install -r requirements.txt
  # 从数据集生成HDF5文件，输出文件为当前目录下的ljspech.h5。
  python generate_hdf5 --data_path /path/to/LJSpeech-1.1
  ```

  ```shell
  cd scripts
  # 运行单机训练
  bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID]
  # 示例：bash run_standalone_train.sh /path/ljspeech.hdf5 0

  # 运行分布式训练
  bash run_distributed_train.sh [DATASET_PATH] [RANK_TABLE_PATH] [DATANAME] [RANK_SIZE] [DEVICE_BEGIN]
  # 示例：bash run_distributed_train.sh /path/ljspeech.h5 ../hccl_8p_01234567_127.0.0.1.json 8 0

  # 运行评估
  bash run_eval.sh [OUTPUT_PATH] [MODEL_CKPT] [DEVICE_ID] text is set in config.py( can modify text of ljspeech_config.yaml)
  # 示例：bash run_eval.sh /path/output /path/model.ckpt 0
  ```

  对于分布式训练，需要提前创建JSON格式的HCCL配置文件。

  请按照以下链接中的说明操作：

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- 如果您想在Modelarts中运行，请查看[ModelArts](https://support.huaweicloud.com/modelarts/)的官方文档，并按照以下方式开始训练。

    - 在ModelArts上进行单机训练。

      ```python
      # 进行单机训练

      # （1）在网页上添加"config_path='/path_to_code/[DATASET_NAME]_config.yaml'"。
      # （2）执行a或b.
      #       a.在[DATASET_NAME]_config.yaml文件中设置"enable_modelarts=True"。
      #          在[DATASET_NAME]_config.yaml文件中设置"dataset_path='/cache/data/[DATASET_NAME]'"。
      #          在[DATASET_NAME]_config.yaml文件中设置"data_name='[DATASET_NAME]'"。
      #          （可选）在[DATASET_NAME]_config.yaml文件中设置其他参数。
      #       b.在网页上添加"enable_modelarts=True"。
      #          在网页上添加"dataset_path='/cache/data/[DATASET_NAME]'"。
      #          在网页上添加"data_name='[DATASET_NAME]'"；
      #          （可选）在网页上添加其他参数。
      # （3）上传zip数据集到S3桶（您也可以上传源数据集，但可能比较耗时）。
      # （4）在网页上设置代码目录为"/path/to/tacotron2"。
      # （5）在网页上设置启动文件为"train.py"。
      # （6）在网页上设置"Dataset path"、"Output file path"和"Output file path"。
      # （7）创建任务。
      ```

    - 在Modelarts上进行分布式训练

      ```python
      # 运行分布式训练示例

      # （1）在网页上添加"config_path='/path_to_code/[DATASET_NAME]_config.yaml'"。
      # （2）执行a或b。
      #       a.在[DATASET_NAME]_config.yaml文件中设置"enable_modelarts=True"。
      #          在[DATASET_NAME]_config.yaml文件中设置"run_distribute=True"；
      #          在[DATASET_NAME]_config.yaml文件中设置"dataset_path='/cache/data/[DATASET_NAME]'"；
      #          在[DATASET_NAME]_config.yaml文件中设置"data_name='[DATASET_NAME]'"；
      #          在[DATASET_NAME]_config.yaml文件中设置其他参数（可选）；
      #       b.在网页上添加"enable_modelarts=True"；
      #          在网页上添加"run_distribute=True"；
      #          在网页上添加"dataset_path='/cache/data/[DATASET_NAME]'"；
      #          在网页上添加"data_name='[DATASET_NAME]'"；
      #          在网页上添加其他参数（可选）；
      # （3）上传zip数据集到S3桶（您也可以上传源数据集，但它可能比较慢）；
      # （4）在网站的用户上设置代码目录为"/path/to/tacotron2"；
      # （5）在网页上设置启动文件为"train.py"；
      # （6）在网页上设置"Dataset path"、"Output file path"和"Output file path"；
      # （7）创建任务。
      ```

    - 在ModelArts上运行评估

      ```python
      # 运行评估示例

      # （1）在网页上添加"config_path='/path_to_code/[DATASET_NAME]_config.yaml'"；
      # （2）执行a或b；
      #       a.在[DATASET_NAME]_config.yaml文件中设置"enable_modelarts=True"；
      #          在[DATASET_NAME]_config.yaml文件中设置"data_name='[DATASET_NAME]'"；
      #          在[DATASET_NAME]_config.yaml文件中设置"model_ckpt='/cache/checkpoint_path/model.ckpt'"；
      #          在[DATASET_NAME]_config.yaml文件上设置"text='text to synthesize'"；
      #          在[DATASET_NAME]_config.yaml文件上设置"checkpoint_url='s3://dir_to_trained_ckpt/'"；
      #          在[DATASET_NAME]_config.yaml文件中设置其他参数（可选）；
      #       b.在网页上添加"enable_modelarts=True"；
      #          在网页上添加"data_name='[DATASET_NAME]'"；
      #          在网页上添加"model_ckpt=/cache/checkpoint_path/model.ckpt"；
      #          在网页上添加"text='text to synthesize'"；
      #          在网页上添加"checkpoint_url='s3://dir_to_trained_ckpt/'"；
      #          在网页上添加其他参数（可选）；
      # （3）上传或复制预训练的模型到S3桶；
      # （4）上传zip数据集到S3桶（您也可以上传源数据集，但它可能比较慢）；
      # （5）在网页上设置代码目录为"/path/to/tacotron2"；
      # （6）在网页上设置启动文件为"eval.py"；
      # （7）在网页上设置"Dataset path"、"Output file path"和"Output file path"；
      # （8）创建任务。
      ```

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```path

tacotron2/
├── eval.py                             // 评估脚本
├── generate_hdf5.py                    // 从数据集生成HDF5文件
├── ljspeech_config.yaml
├── model_utils
│  ├── config.py                       // 解析参数
│  ├── device_adapter.py               // ModelArts设备适配器
│  ├── __init__.py                     // 初始化文件
│  ├── local_adapter.py                // 本地适配器
│  └── moxing_adapter.py               // ModelArts的装饰器
├── README.md                           // 关于Tacotron2的描述
├── requirements.txt                // 需要的包
├── scripts
│  ├── run_distribute_train.sh         // 进行分布式训练
│  ├── run_eval.sh                     // 进行评估
│  └── run_standalone_train.sh         // 进行单机训练
├── src
│  ├── callback.py                     // 监督训练的回调函数
│  ├── dataset.py                      // 定义数据集和采样器
│  ├── hparams.py                      // Tacotron2配置
│  ├── rnn_cells.py                    // 执行RNN Cells
│  ├── rnns.py                         // 执行带有长度掩码的LSTM
│  ├── tacotron2.py                    // Tacotron2网络
│  ├── text
│  │  ├── cleaners.py                  // 清理文本序列
│  │  ├── cmudict.py                   // 定义CMUdict
│  │  ├── __init__.py                  // 处理文本序列
│  │  ├── numbers.py                   // 正则化数量
│  │  └── symbols.py                   // 编码符号
│  └── utils
│      ├── audio.py                     // 提取音频特征
│      └── convert.py                   // 通过均值归整梅尔频谱
└── train.py                            // 训练入口

```

## [脚本参数](#目录)

训练和评估的参数都可以在[DATASET]_config.yaml中设置。

- LJSpeech-1.1的配置

  ```python
  'pretrain_ckpt': '/path/to/model.ckpt'# 在训练阶段使用预训练好的CKPT文件
  'model_ckpt': '/path/to/model.ckpt'   # 在推理阶段使用预训练好的CKPT文件
  'lr': 0.002                           # 初始学习率
  'batch_size': 16                      # 训练批处理大小
  'epoch_num': 2000                     # 总训练epoch
  'warmup_epochs': 30                   # 热身学习周期数
  'save_ckpt_dir:' './ckpt'             # CKPT文件保存目录
  'keep_checkpoint_max': 10             # 仅保留最后一个keep_checkpoint_max检查点

  'text': 'text to synthesize'          # 指定要在推理时合成的文本
  'dataset_path': '/dir/to/hdf5'        # 指定HDF5文件的目录
  'data_name': 'ljspeech'               # 指定数据集名称
  'audioname': 'text2speech'            # 指定生成音频的文件名
  'run_distribute': False               # 是否进行分布式训练
  'device_id': 0                        # 指定使用的设备
  ```

### [训练过程](#目录)

- 在Ascend上运行

    - 进行单卡任务训练并运行shell脚本

        ```bash
        cd scripts
        bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID] [DATANAME]
        ```

    - 运行Tacotron2分布式训练脚本。在多台设备上进行任务训练，在`scripts/`中执行以下命令。

        ```bash
        cd scripts
        bash run_distributed_train.sh [DATASET_PATH] [RANK_TABLE_PATH] [DATANAME] [RANK_SIZE] [DEVICE_BEGIN]
        ```

    注：`DATASET_PATH`是包含HDF5文件的目录。

### [推理过程](#目录)

- 在Ascend上运行

    - 运行Tacotron2评估的脚本 执行命令如下。

        ```bash
        cd scripts
        bash run_eval.sh [OUTPUT_PATH] [DATANAME] [MODEL_CKPT] [DEVICE_ID]
        ```

    注：`OUTPUT_PATH`是保存评估输出的目录

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Tacotron2                                                     |
| -------------------------- | ---------------------------------------------------------------|
| 资源                  | Ascend 910，EulerOS 2.8             |
| 上传日期             | 12/20/2021                                   |
| MindSpore版本         | 1.3.0                                                         |
| 数据集                   | LJSpeech-1.1                                                |
| 训练参数       | 8p, epoch=2000, batch_size=16 |
| 优化器                 | Adam                                                          |
| 损失函数             | BinaryCrossEntropy，MSE                               |
| 输出                   | 梅尔频谱                                                    |
| 损失                      | 0.33                                                       |
| 速度|1264毫秒/步|
| 训练总时长      | 8卡：24小时19分钟41秒                                 |
| 检查点                | 328.9M（.ckpt）                                             |
| 脚本                   | [Tacotron2](https://gitee.com/mindspore/models/tree/master/official/audio/Tacotron2)|

### 推理性能

| 参数                | Tacotron2                                                      |
| -------------------------- | ----------------------------------------------------------------|
| 资源                  | Ascend 910，EulerOS 2.8                  |
| 上传日期             | 12/20/2021                                |
| MindSpore版本         | 1.3.0                                                          |
| 数据集                   | LJSpeech-1.1                        |
| batch_size                | 1                                                              |
| 输出                   | 梅尔频谱                      |
| 速度      | 单卡：花125秒合成6秒梅尔频谱|

## [随机情况说明](#目录)

只有一种随机的情况。

- 初始化一些模型权重。

为了避免权重初始化的随机性，已经在train.py中设置了一些种子。

# [ModelZoo主页](#目录)

 请查看官方[主页](https://gitee.com/mindspore/models)。
