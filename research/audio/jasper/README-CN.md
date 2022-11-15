# 目录

[View English](./README.md)

<!-- TOC -->

- - [目录](#目录)
  - [jasper介绍](#jasper介绍)
  - [网络模型结构](#网络模型结构)
  - [数据集](#数据集)
  - [环境要求](#环境要求)
  - [文件说明和运行说明](#文件说明和运行说明)
    - [代码目录结构说明](#代码目录结构说明)
    - [模型参数](#模型参数)
    - [训练和评估](#训练和评估)
    - [导出和推理](#导出和推理)
  - [性能](#性能)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
  - [FAQ](#FAQ)
  - [ModelZoo主页](#modelzoo主页)

## [Jasper介绍](#contents)

Japser是一个使用 CTC 损失训练的端到端的语音识别模型。Jasper模型仅仅使用1D convolutions, batch normalization, ReLU, dropout和residual connections这些模块。训练和验证支持CPU,GPU和Ascend。

[论文](https://arxiv.org/pdf/1904.03288v3.pdf): Jason Li, et al. Jasper: An End-to-End Convolutional Neural Acoustic Model.

## [网络模型结构](#contents)

Jasper是一种基于卷积的端到端神经声学模型。在音频处理阶段，将每一帧转换为梅尔尺度谱图特征，声学模型将其作为输入，并输出每一帧词汇表上的概率分布。声学模型具有模块化的块结构，可以相应地进行参数化：Jasper BxR模型有B个块，每个块由R个重复子块组成。

每一个子块应用下面这些操作：
1D-Convolution, Batch Normalization, ReLU activation, Dropout.
每个块输入通过残差连接直接连接到所有后续块的最后一个子块，本文称之为dense residual。每个块的内核大小和过滤器数量都不同，从底层到顶层，过滤器的大小都在增加。不管精确的块配置参数B和R如何，每个Jasper模型都有四个额外的卷积块：一个紧跟在输入层之后，三个在B块末尾。

## [数据集](#contents)

可以基于论文中提到的数据集或在相关领域/网络架构中广泛使用的数据集运行脚本。在下面的部分中，我们将介绍如何使用下面的相关数据集运行脚本。

使用的数据集为: [LibriSpeech](<http://www.openslr.org/12>)

训练集：
train-clean-100: [6.3G] (100小时的无噪音演讲训练集)
train-clean-360.tar.gz [23G] (360小时的无噪音演讲训练集)
train-other-500.tar.gz [30G] (500小时的有噪音演讲训练集)
验证集：
dev-clean.tar.gz [337M] (无噪音)
dev-other.tar.gz [314M] (有噪音)
测试集:
test-clean.tar.gz [346M] (测试集, 无噪音)
test-other.tar.gz [328M] (测试集, 有噪音)
数据格式：wav 和 txt 文件

## [环境要求](#contents)

硬件
  GPU处理器或Ascend处理器
框架
  [MindSpore](https://www.mindspore.cn/install/en)
通过下面网址可以获得更多信息：
 [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
 [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## [文件说明和运行说明](#contents)

### [代码目录结构说明](#contents)

```path
.
└─audio
    └─jasper
        │  eval.py                          //推理文件
        │  labels.json                      //需要用到的字符
        │  pt2mind.py                       //pth转化ckpt文件
        |  create_mindrecord.py             //将数据集转化为mindrecord
        │  README-CN.md                     //中文readme
        │  README.md                        //英文readme
        │  requirements.txt                 //需要的库文件
        │  train.py                         //训练文件
        │  preprocess.py                    //推理预处理
        │  postprocess.py                   //推理后处理
        │
        ├─scripts
        │      download_librispeech.sh      //下载数据集的脚本
        │      preprocess_librispeech.sh    //处理数据集的脚本
        │      run_distribute_train_gpu.sh  //GPU8卡训练
        │      run_eval_cpu.sh              //CPU推理
        │      run_eval_gpu.sh              //GPU推理
        │      run_standalone_train_cpu.sh  //CPU单卡训练
        │      run_standalone_train_gpu.sh  //GPU单卡训练
        │      run_distribute_train_ascend.sh  //Ascend-910 8卡训练
        │      run_standalone_train_ascend.sh  //Ascend-910单卡训练
        │      run_eval_ascend.sh           //Ascend-910评估
        │      run_infer_310.sh             //Ascend-310推理
        |
        ├─ascend310_infer                   //ascend-310推理代码
        │
        ├─src
        │      audio.py                     //数据处理相关代码
        │      callback.py                  //回调以监控训练
        │      cleaners.py                  //数据清理
        │      config.py                    //jasper配置文件
        │      dataset.py                   //数据处理
        │      decoder.py                   //来自第三方的解码器
        │      eval_callback.py             //推理的数据回调
        │      greedydecoder.py             //修改Mindspore代码的greedydecoder
        │      jasper10x5dr_speca.yaml      //jasper网络结构配置
        │      lr_generator.py              //产生学习率
        │      model.py                     //训练模型
        │      model_test.py                //推理模型
        │      number.py                    //数据处理
        │      text.py                      //数据处理
        │      __init__.py
        │
        └─utils
                convert_librispeech.py      //转化数据集
                download_librispeech.py     //下载数据集
                download_utils.py           //下载工具
                inference_librispeech.csv   //推理数据集链接
                librispeech.csv             //全部数据集链接
                preprocessing_utils.py      //预处理工具
                __init__.py
```

### [模型参数](#contents)

训练和推理的相关参数在`config.py`文件

```text
训练相关参数
    epochs                       训练的epoch数量，默认为440
```

```text
数据处理相关参数
    train_manifest               用于训练的数据文件路径，默认为 'data/libri_train_manifest.json'
    val_manifest                 用于测试的数据文件路径，默认为 'data/libri_val_manifest.json'
    batch_size                   批处理大小，默认为64
    labels_path                  模型输出的token json 路径, 默认为 "./labels.json"
    sample_rate                  数据特征的采样率，默认为16000
    window_size                  频谱图生成的窗口大小（秒），默认为0.02
    window_stride                频谱图生成的窗口步长（秒），默认为0.01
    window                       频谱图生成的窗口类型，默认为 'hamming'
    speed_volume_perturb         使用随机速度和增益扰动，默认为False，当前模型中未使用
    spec_augment                 在MEL谱图上使用简单的光谱增强，默认为False，当前模型中未使用
    noise_dir                    注入噪音到音频。默认为noise Inject未添加，默认为''，当前模型中未使用
    noise_prob                   每个样本加噪声的概率，默认为0.4，当前模型中未使用
    noise_min                    样本的最小噪音水平，(1.0意味着所有的噪声，不是原始信号)，默认是0.0，当前模型中未使用
    noise_max                    样本的最大噪音水平。最大值为1.0，默认值为0.5，当前模型中未使用
```

```text
优化器相关参数
    learning_rate                初始化学习率，默认为3e-4
    learning_anneal              对每个epoch之后的学习率进行退火，默认为1.1
    weight_decay                 权重衰减，默认为1e-5
    momentum                     动量，默认为0.9
    eps                          Adam eps，默认为1e-8
    betas                        Adam betas，默认为(0.9, 0.999)
    loss_scale                   损失规模，默认是1024
```

```text
checkpoint相关参数
    ckpt_file_name_prefix        ckpt文件的名称前缀，默认为'DeepSpeech'
    ckpt_path                    ckpt文件的保存路径，默认为'checkpoints'
    keep_checkpoint_max          ckpt文件的最大数量限制，删除旧的检查点，默认是10
```

## [训练和评估](#contents)

### 训练

```text
运行: train.py   [--pre_trained_model_path PRE_TRAINED_MODEL_PATH]
                 [--is_distributed IS_DISTRIBUTED]
                 [--device_target DEVICE_TARGET]
                 [--device_id DEVICE_ID]
参数:
    --pre_trained_model_path    预先训练的模型文件路径，默认为''
    --is_distributed            多卡训练，默认为False
    --device_target             运行代码的设备："GPU" | “CPU” | “Ascend”，默认为"GPU"
    --device_target             运行代码的设备ID，使用Ascend训练时用到，默认为0
```

### 评估

```text
运行: eval.py   [--pretrain_ckpt PRETRAIN_CKPT]
                [--device_target DEVICE_TARGET]

参数:
    --pretrain_ckpt              checkpoint的文件路径, 默认为''
    --device_target              运行代码的设备："GPU" | “CPU”，默认为"GPU"
```

在训练之前，应该下载、处理数据集。

``` bash
bash scripts/download_librispeech.sh
bash scripts/preprocess_librispeech.sh
python create_mindrecord.py //将数据集转成mindrecord格式
```

流程结束后，数据目录结构如下：

```path
    .
    |--LibriSpeech
    │  |--train-clean-100-wav
    │  │--train-clean-360-wav
    │  │--train-other-500-wav
    │  |--dev-clean-wav
    │  |--dev-other-wav
    │  |--test-clean-wav
    │  |--test-other-wav
    |--librispeech-train-clean-100-wav.json,librispeech-train-clean-360-wav.json,librispeech-train-other-500-wav.json,librispeech-dev-clean-wav.json,librispeech-dev-other-wav.json,librispeech-test-clean-wav.json,librispeech-test-other-wav.json
```

src/config中设置数据集的位置。

```shell
...
训练配置
"Data_dir": '/data/dataset',
"train_manifest": ['/data/dataset/librispeech-train-clean-100-wav.json',
                   '/data/dataset/librispeech-train-clean-360-wav.json',
                   '/data/dataset/librispeech-train-other-500-wav.json'],
"mindrecord_format": "/data/jasper_tr{}.md",
"mindrecord_files": [f"/data/jasper_tr{i}.md" for i in range(8)]

评估配置
"DataConfig":{
     "Data_dir": '/data/inference_datasets',
     "test_manifest": ['/data/inference_datasets/librispeech-dev-clean-wav.json'],
}

```

训练之前，需要安装`librosa` and `Levenshtein`
通过官网安装MindSpore并完成数据集处理后，可以开始训练如下：

```shell

# gpu单卡训练
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID]

# cpu单卡训练
bash ./scripts/run_standalone_train_cpu.sh

# gpu多卡训练
bash ./scripts/run_distribute_train_gpu.sh

# Ascend单卡训练
bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID]

# Ascend多卡训练
bash ./scripts/run_distribute_train_ascend.sh [RANK_SIZE] [BEGIN] [RANK_TABLE_FILE]
# 其中，RANK_SIZE：训练的卡数，BEGIN：起始的卡号，RANK_TABLE_FILE：分布式配置文件路径

```

例如，可以这样进行Ascend八卡的训练：`bash ./scripts/run_distribute_train_ascend.sh 8 0 ./hccl_config/hccl_8.json`

评估：

```shell

# cpu评估
bash ./scripts/run_eval_cpu.sh [PATH_CHECKPOINT]

# gpu评估
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [PATH_CHECKPOINT]

# Ascend评估
bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [PATH_CHECKPOINT]

```

## [导出和推理](#contents)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出

在推理之前需要先导出mindir，并准备好训练好的ckpt文件

```shell
# 在Ascend 910上执行导出
DEVICE_ID=[DEVIC_ID] python export.py --pre_trained_model_path [CKPT_FILE] --device_target 'Ascend'
```

### 推理

推理需要在Ascend 310上进行，将导出`jasper_graph.mindir`文件和`jasper_variables/`放置到当前路径，将处理好（从Ascend 910上）的测试集及对应的json文件存好，修改`src/config.py`中的推理配置的"DataConfig"，另外，推理需要用到第三方解码器，需要在环境中安装pytorch

```shell
# 推理配置
"DataConfig": {
    "Data_dir": '/home/dataset/LibriSpeech',
    "test_manifest": ['/home/dataset/LibriSpeech/librispeech-test-clean-wav.json'],
},
"batch_size_infer": 1,
# for preprocess
"result_path": "./preprocess_Result",
# for postprocess
"result_dir": "./result_Files",
"post_out": "./infer_output.txt"

```

执行推理脚本：

```shell
bash scripts/run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件
- `NEED_PREPROCESS` 是否需要进行preprocess，需要设置'y'，不需要设置'n'
- `DEVICE_ID` 是可选的，默认为0

推理完成后，可以在`infer_output.txt`中看到推理结果

## [性能](#contents)

### [训练和测试性能分析](#contents)

#### 训练性能

| 参数                 | Jasper                                                      |
| -------------------------- | ---------------------------------------------------------------|
| 资源                   | NV SMX2 V100-32G              |
| 更新日期              | 7/2/2022 (month/day/year)                                    |
| MindSpore版本           | 1.8.0                                                        |
| 数据集                    | LibriSpeech                                                 |
| 训练参数       | 8p, epoch=440, steps=1088 * epoch, batch_size = 64, lr=3e-4 |
| 优化器                  | Adam                                                           |
| 损失函数              | CTCLoss                                |
| 输出                    | 概率值                                                    |
| 损失值                       | 32-33                                                        |
| 运行速度                      | GPU: 8p 2.7s/step; Ascend: 8p 1.7s/step    |
| 训练总时间       | GPU: 8p around 194h; Ascend 8p around 220h  |
| Checkpoint文件大小                 | 991M (.ckpt file)                                              |
| 代码                   | [Japser script](https://gitee.com/mindspore/models/tree/master/research/audio/jasper) |

#### Inference Performance

| 参数                 | Jasper                                                       |
| -------------------------- | ----------------------------------------------------------------|
| 资源                   | NV SMX2 V100-32G                   |
| 更新日期              | 2/7/2022 (month/day/year)                                 |
| MindSpore版本          | 1.8.0                                                         |
| 数据集                    | LibriSpeech                         |
| 批处理大小                 | 64                                                         |
| 输出                    | 错误率                       |
| 精确度(无噪声)       | GPU: 8p WER: 5.754  CER: 2.151; Ascend: 8p, WER: 4.597  CER: 1.544 |
| 精确度(有噪声)      | GPU: 8p, WER: 19.213 CER: 9.393; Ascend: 8p, WER, 12.871 CER: 5.618 |
| 模型大小        | 330M (.mindir file)                                              |

## [FAQ](#contents)

Q1. 训练中断如何继续训练？  
A1. 将中断时的ckpt文件作为训练脚本的pre_path参数，可以接着该断点继续训练。  
Q2. 训练完成后如何选择ckpt文件去做推理？  
A2. 可以根据输出的log文件，找到loss最小时对应的ckpt文件。  
Q3. src/dataset.py, src/model.py和src/model_test.py中的常量TRAIN_INPUT_PAD_LENGTH如何设置？  
A3. 该常量表示训练输入数据进行padding后的长度，可以适当增大牺牲训练时间以换取更低的错误率。

## [ModelZoo主页](#contents)

 [ModelZoo主页](https://gitee.com/mindspore/models).
