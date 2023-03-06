
# 目录

<!-- TOC -->

- [CTPN描述](#ctpn描述)
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
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [CTPN描述](#目录)

CTPN是一种基于目标检测方法的文本检测模型。在Faster R-CNN的基础上进行了改进，并与双向LSTM结合，因此CTPN对于水平文本检测非常有效。CTPN的另一个亮点是将文本检测任务转化为一系列小规模的文本框检测，这一想法首次在《Detecting Text in Natural Image with Connectionist Text Proposal Network》论文中提出。

[论文](https://arxiv.org/pdf/1609.03605.pdf) Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao, "Detecting Text in Natural Image with Connectionist Text Proposal Network", ArXiv, vol. abs/1609.03605, 2016.

# [模型架构](#目录)

整体网络架构将VGG16作为主干，使用双向LSTM提取小规模文本框的上下文特征，然后使用区域候选网络（Region Proposal Network，RPN）预测边界框和概率。

[链接](https://arxiv.org/pdf/1605.07314v1.pdf)

# [数据集](#目录)

我们使用了6个数据集进行训练，1个数据集用于评估。

- 数据集1：[ICDAR 2013: Focused Scene Text](https://rrc.cvc.uab.es/?ch=2&com=tasks)：
    - 训练集：142 MB，229张图像
    - 测试集：110 MB，233张图像
- 数据集2：[ICDAR 2011: Born-Digital Images](https://rrc.cvc.uab.es/?ch=1&com=tasks)：
    - 训练集：27.7 MB，410张图像
- 数据集3：[ICDAR 2015: Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4&com=tasks)：
    - 训练集：89 MB，1000张图像
- 数据集4：[SCUT-FORU: Flickr OCR Universal Database](https://github.com/yan647/SCUT_FORU_DB_Release)：
    - 训练集：388 MB，1715张图像
- 数据集5：[CocoText v2（MSCCO2017的子集）](https://rrc.cvc.uab.es/?ch=5&com=tasks)：
    - 训练集：13 GB，63686张图像
- 数据集6：[SVT（街景数据集）](https://www.kaggle.com/datasets/nageshsingh/the-street-view-text-dataset)：
    - 训练集：115 MB，349张图像

# [特性](#目录)

# [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```shell
└─ ctpn
  ├── README.md                             # 网络描述
  ├── Ascend310_infer                        # Ascend 310推理应用
  ├── eval.py                               # eval网络
  ├── scripts
  │   ├── eval_res.sh                       # 计算精度和召回
  │   ├── run_distribute_train_ascend.sh    # 在Ascend平台启动分布式训练（8卡）
  │   ├── run_distribute_train_gpu.sh       # 在GPU平台启动分布式训练（8卡）
  │   ├── run_eval_ascend.sh                # 在Ascend平台启动评估
  │   ├── run_eval_gpu.sh                   # 在GPU平台启动评估
  │   ├── run_infer_310.sh                  # Ascend 310推理的shell脚本
  │   ├── run_standalone_train_gpu.sh       # 在GPU平台启动单机训练（单卡）
  │   └── run_standalone_train_ascend.sh    # 在Ascend平台启动单机训练（单卡）
  ├── src
  │   ├── CTPN
  │   │   ├── BoundingBoxDecode.py          # 边界框解码
  │   │   ├── BoundingBoxEncode.py          # 边界框编码
  │   │   ├── __init__.py                   # 初始化包
  │   │   ├── anchor_generator.py           # Anchor生成器
  │   │   ├── bbox_assign_sample.py         # 候选区域生成层
  │   │   ├── proposal_generator.py         #候选区域生成器
  │   │   ├── rpn.py                        # 区域候选网络
  │   │   └── vgg16.py                      # 骨干
  │   ├── model_utils
  │   │   ├── config.py             // 参数配置
  │   │   ├── moxing_adapter.py     // ModelArts设备配置
  │   │   ├── device_adapter.py     // 设备配置
  │   │   ├── local_adapter.py      // 本地设备配置
  │   ├── convert_icdar2015.py              # 转换ICDAR2015数据集标签
  │   ├── convert_svt.py                    # 转换SVT标签
  │   ├── create_dataset.py                 # 创建MindRecord数据集
  │   ├── ctpn.py                           # CTPN网络定义
  │   ├── dataset.py                        # 数据预处理
  │   ├── eval_callback.py                  # 训练时评估回调
  │   ├── eval_utils.py                     # 评估函数
  │   ├── lr_schedule.py                    # 学习率调度器
  │   ├── weight_init.py                    # LSTM初始化
  │   ├── network_define.py                 # 网络定义
  │   └── text_connector
  │       ├── __init__.py                   # 初始化包文件
  │       ├── connect_text_lines.py         # 连接文本行
  │       ├── detector.py                   # 检测框
  │       ├── get_successions.py            # 获取继承候选区域
  │       └── utils.py                      # 常用函数
  ├── postprogress.py                        # Ascend 310推理后处理
  ├── export.py                              # 导出AIR或MindIR模型的脚本
  ├── requirements.txt                       # 需求文件
  ├── train.py                               # 训练网络
  └── default_config.yaml                    # 配置文件

```

## [训练过程](#目录)

### 数据集

要创建数据集，请先下载并处理数据集。我们提供了src/convert_svt.py和src/convert_icdar2015.py来处理SVT和ICDAR2015数据集标签。对于SVT数据集，可以按以下方式处理：

```shell
    python convert_svt.py --dataset_path=/path/img --xml_file=/path/train.xml --location_dir=/path/location
```

对于ICDAR2015数据集，可以按以下方式处理：

```shell
    python convert_icdar2015.py --src_label_path=/path/train_label --target_label_path=/path/label
```

然后修改src/config.py并添加数据集路径。对于每个路径，将IMAGE_PATH和LABEL_PATH添加到config的列表中，如下所示：

```python
    # 创建数据集
    "coco_root": "/path/coco",
    "coco_train_data_type": "train2017",
    "cocotext_json": "/path/cocotext.v2.json",
    "icdar11_train_path": ["/path/image/", "/path/label"],
    "icdar13_train_path": ["/path/image/", "/path/label"],
    "icdar15_train_path": ["/path/image/", "/path/label"],
    "icdar13_test_path": ["/path/image/", "/path/label"],
    "flick_train_path": ["/path/image/", "/path/label"],
    "svt_train_path": ["/path/image/", "/path/label"],
    "pretrain_dataset_path": "",
    "finetune_dataset_path": "",
    "test_dataset_path": "",
```

然后，使用src/create_dataset.py创建数据集，命令如下：

```shell
python src/create_dataset.py
```

### 用法

- Ascend处理器：

    ```default_config.yaml
    预训练集pretraining_dataset_file：/home/DataSet/ctpn_dataset/pretrain/ctpn_pretrain.mindrecord0
    微调集pretraining_dataset_file：/home/DataSet/ctpn_dataset/finetune/ctpn_finetune.mindrecord0
    img_dir:/home/DataSet/ctpn_dataset/ICDAR2013/test

    根据实际路径修改参数
    ```

    ```bash
    # 分布式训练
    bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH]
    # 示例：bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json Pretraining(or Finetune) \
    # /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

    # 单机训练
    bash scrpits/run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID]
    示例：bash scrpits/run_standalone_train_ascend.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

    # 评估：
    bash scripts/run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
    # 示例：bash script/run_eval_ascend.sh /home/DataSet/ctpn_dataset/ICDAR2013/test \
    # /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

- GPU：

    ```bash
    # 分布式训练
    bash scripts/run_distribute_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH]
    # 示例：bash scripts/run_distribute_train_gpu.sh Pretraining(or Finetune) \
    # /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

    # 单机训练
    bash scrpits/run_standalone_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH] [DEVICE_ID]
    示例：bash scrpits/run_standalone_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

    # 评估：
    bash scripts/run_eval_gpu.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
    # 示例：bash script/run_eval_gpu.sh /home/DataSet/ctpn_dataset/ICDAR2013/test \
    # /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

`pretrained_path`应该是在Imagenet2012上训练的vgg16的检查点。dict中的权重名称应该完全一样，且在vgg16训练中启用batch_norm，否则将导致后续步骤失败。有关COCO_TEXT_PARSER_PATH coco_text.py文件，请参考[链接](https://github.com/andreasveit/coco-text)。如需获取vgg16主干，可以使用src/CTPN/vgg16.py中定义的网络结构。如需训练主干，请复制modelzoo/official/cv/VGG/vgg16/src/下的src/CTPN/vgg16.py，并修改vgg16/train.py以适应新的结构。您可以按以下方式处理：

```python
...
from src.vgg16 import VGG16
...
network = VGG16(num_classes=cfg.num_classes)
...

```

为了训练一个更好的模型，你可以修改modelzoo/official/cv/VGG/vgg16/src/config.py中的一些参数，这里我们建议修改"warmup_epochs"，如下所示。你也可以尝试调整其他参数。

```python

imagenet_cfg = edict({
    ...
    "warmup_epochs": 5
    ...
})

```

然后，您可以使用ImageNet2012训练它。
> 注：
> RANK_TABLE_FILE文件，请参考[链接](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)。如需获取设备IP，请点击[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。对于InceptionV4等大模型，最好导出外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将hccl连接检查时间从默认的120秒延长到600秒。否则，连接可能会超时，因为随着模型增大，编译时间也会增加。
>
> 处理器绑核操作取决于`device_num`和总处理器数。如果不希望这样做，请删除`scripts/run_distribute_train.sh`中的`taskset`操作。
>
> TASK_TYPE包含预训练和微调。对于预训练，我们使用ICDAR2013、ICDAR2015、SVT、SCU-FORU、CocoText v2。对于微调，我们使用ICDAR2011，
ICDAR2013和SCU-FORU，可以提高精度和召回率。在执行微调时，我们使用预训练中的检查点训练作为PRETRAINED_PATH。
> 有关COCO_TEXT_PARSER_PATH coco_text.py，请参考[链接](https://github.com/andreasveit/coco-text)。
>

### 启动

```bash
# 训练示例
  shell:
    Ascend处理器：
      # 分布式训练示例（8卡）
      bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [TASK_TYPE] [PRETRAINED_PATH]
      # 示例：bash scripts/run_distribute_train_ascend.sh /home/hccl_8p_01234567_10.155.170.71.json Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

      # 单机训练
      bash run_standalone_train_ascend.sh [TASK_TYPE] [PRETRAINED_PATH]
      # 示例：bash scrpits/run_standalone_train_ascend.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0

  shell:
    GPU：
      # 分布式训练示例（8卡）
      bash run_distribute_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH]
      # 示例：bash scripts/run_distribute_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt

      # 单机训练
      bash run_standalone_train_gpu.sh [TASK_TYPE] [PRETRAINED_PATH]
      # 示例：bash scrpits/run_standalone_train_gpu.sh Pretraining(or Finetune) /home/DataSet/ctpn_dataset/backbone/0-150_5004.ckpt 0
```

### 结果

训练结果将存储在示例路径中。默认情况下，检查点将存储在`ckpt_path`中，训练日志将重定向到`./log`，损失也将重定向到`./loss_0.log`，如下所示。

```python
377 epoch: 1 step: 229 ,rpn_loss: 0.00355
399 epoch: 2 step: 229 ,rpn_loss: 0.00327
424 epoch: 3 step: 229 ,rpn_loss: 0.00910
```

- 在ModelArts上运行
- 请参考ModelArts[官方指导文档](https://support.huaweicloud.com/modelarts/)。

```python
#  在ModelArts上使用分布式训练DPN：
#  数据集目录结构

#  ├── ctpn_dataset              # 目录
#    ├──train                         # 训练目录
#      ├── pretrain               # 预训练数据集目录
#      ├── finetune               # 微调数据集目录
#      ├── backbone               # 预训练目录（如存在）
#    ├── eval                    # eval目录
#      ├── ICDAR2013              # ICDAR2013图像目录
#      ├── checkpoint           # ckpt文件目录
#      ├── test                  # ckpt文件目录
#          ├── ctpn_test.mindrecord       # MindRecord测试图像
#          ├── ctpn_test.mindrecord.db    # mindrecord.db测试图像

# （1）执行a（修改yaml文件参数）或b（ModelArts创建训练作业以修改参数）。
#       a. 设置"enable_modelarts=True"
#          设置"run_distribute=True"
#          设置"save_checkpoint_path=/cache/train/checkpoint/"
#          设置"finetune_dataset_file=/cache/data/finetune/ctpn_finetune.mindrecord0"
#          设置"pretrain_dataset_file=/cache/data/finetune/ctpn_pretrain.mindrecord0"
#          设置"task_type=Pretraining" or task_type=Finetune
#          设置"pre_trained=/cache/data/backbone/pred file name" Without pre-training weights  pre_trained=""
#
#       b. 在ModelArts界面上添加"enable_modelarts=True"参数
#          在ModelArts界面设置方法a所需的参数
#          注：path参数不需要用引号括起来。

# （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
# （3）在ModelArts界面设置代码路径"/path/ctpn"。
# （4）在ModelArts界面设置模型的启动文件"train.py"
# （5）在ModelArts界面设置模型的数据路径".../ctpn_dataset/train"（选择ctpn_dataset/train文件夹路径）
# 模型的输出路径"Output file path"和模型的日志路径"Job log path"
# （6）开始训练模型

# 在ModelArts上使用模型推理
# （1）将训练好的模型放置到桶的对应位置
# （2）执行a或者b
#       a. 设置"enable_modelarts=True"
#          设置"dataset_path=/cache/data/test/ctpn_test.mindrecord"
#          设置"img_dir=/cache/data/ICDAR2013/test"
#          设置"checkpoint_path=/cache/data/checkpoint/checkpoint file name"

#       b. 在ModelArts界面添加"enable_modelarts=True"参数
#          在ModelArts界面设置方法a所需的参数
#          注：path参数不需要用引号括起来。

# （3）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
# （4）在ModelArts界面设置模型的数据路径"/path/ctpn"
# （5）在ModelArts界面设置模型的启动文件"eval.py"
# （6）在ModelArts界面设置模型的数据路径".../ctpn_dataset/eval"（选择FSNS/eval文件夹路径）
# 模型的输出路径"Output file path"和模型的日志路径"Job log path"
# （7）开始模型推理
```

## [评估过程](#目录)

### 用法

您可以使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend处理器：

    ```bash
    bash run_eval_ascend.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
    # 示例：bash script/run_eval_ascend.sh /home/DataSet/ctpn_dataset/ICDAR2013/test /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

- GPU：

    ```bash
    bash run_eval_gpu.sh [IMAGE_PATH] [DATASET_PATH] [CHECKPOINT_PATH]
    # 示例：bash script/run_eval_gpu.sh /home/DataSet/ctpn_dataset/ICDAR2013/test /home/DataSet/ctpn_dataset/ctpn_final_dataset/test/ctpn_test.mindrecord /home/model/cv/ctpn/train_parallel0/ckpt_0/
    ```

评估后，将得到submit_ctpn-xx_xxxx.zip存档文件，其中包含检查点文件的名称。如需评估，可以使用ICDAR2013网络提供的脚本。如需下载Deteval脚本，请点击[链接](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)。
下载脚本后，解压缩并将其放在ctpn/scripts下，然后执行eval_res.sh。您将得到以下文件：

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

然后可以运行scripts/eval_res.sh来计算评估结果。

```base
bash eval_res.sh
```

### 训练时评估

如果希望在训练时进行评估，可以添加`run_eval`以启动shell并将其设置为True。当`run_eval`为True时，可设置参数选项：`eval_dataset_path`、`save_best_ckpt`、`eval_start_epoch`、`eval_interval`。

### 结果

评估结果将存储在示例路径中，您可以在`log`中查看如下结果。

```text
{"precision": 0.90791, "recall": 0.86118, "hmean": 0.88393}
```

GPU评估结果如下：

```text
{"precision": 0.9346, "recall": 0.8621, "hmean": 0.8969}
```

## 模型导出

```shell
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

- 在Modelart上导出MindIR

    ```Modelarts
    在ModelArts上导出MindIR
    数据存储方法同训练
    # （1）执行a（修改yaml文件参数）或b（ModelArts创建训练作业以修改参数）。
    #       a. 设置"enable_modelarts=True"
    #          设置"file_name=ctpn"
    #          设置"file_format=MINDIR"
    #          设置"ckpt_file=/cache/data/checkpoint file name"

    #       b. 在ModelArts界面添加"enable_modelarts=True"参数
    #          在ModelArts界面设置方法a所需的参数
    #          注：path参数不需要用引号括起来。
    # （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"
    # （3）在ModelArts界面设置代码路径"/path/ctpn"。
    # （4）在ModelArts界面设置模型的启动文件"export.py"
    # （5）在ModelArts界面设置模型的数据路径".../ctpn_dataset/eval/checkpoint"（选择CNNCTC_Data/eval/checkpoint文件夹路径）
    # 模型的输出路径"Output file path"和模型的日志路径"Job log path"
    ```

    `EXPORT_FORMAT`：取值范围["AIR", "MINDIR"]。

## [推理过程](#目录)

### 用法

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

在执行推理之前，必须在Ascend 910环境上通过导出脚本导出AIR文件。

```shell
# Ascend 310推理
bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]
```

推理后，将得到submit.zip存档文件。如需评估，可以使用ICDAR2013网络提供的脚本。如需下载Deteval脚本，请点击[链接](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9zdGFuZGFsb25lcy9zY3JpcHRfdGVzdF9jaDJfdDFfZTItMTU3Nzk4MzA2Ny56aXA=)。
下载脚本后，解压缩并将其放在ctpn/scripts下，然后执行eval_res.sh。您将得到以下文件：

```text
gt.zip
readme.txt
rrc_evalulation_funcs_1_1.py
script.py
```

然后可以运行scripts/eval_res.sh来计算评估结果。

```base
bash eval_res.sh
```

### 结果

评估结果将存储在示例路径中，您可以在`log`中查看如下结果。

```text
{"precision": 0.88913, "recall": 0.86082, "hmean": 0.87475}
```

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend                                                       | GPU                                             |
| -------------------------- | ------------------------------------------------------------ |------------------------------------------------------------ |
| 模型版本             | CTPN                                                         | CTPN                                                     |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8 | Tesla V100 PCIE 32GB；CPU 2.60GHz; 104核；内存790G；EulerOS 2.0    |
| 上传日期             | 02/06/2021                                                   | 09/20/2021                                                   |
| MindSpore版本         | 1.1.1                                                        | 1.5.0                                                        |
| 数据集                   | 16930张图像                                                | 16930张图像                                                |
| Batch_size                 | 2                                                            | 2                                                            |
| 训练参数       | default_config.yaml                                          | default_config.yaml                                          |
| 优化器                 | 动量                                                    | 动量                                                    |
| 损失函数             | SoftmaxCrossEntropyWithLogits用于分类, SmoothL2Loss用于bbox回归| SoftmaxCrossEntropyWithLogits用于分类, SmoothL2Loss用于bbox回归|
| 损失                      | ~0.04                                                        | ~0.04                                                       |
| 总时长（8卡）           | 6h                                                           | 11h                                                           |
| 脚本                   | [ctpn](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) | [ctpn](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1)    |

#### 推理性能

| 参数         | Ascend                                        | GPU                |
| ------------------- | --------------------------------------------- | --------------------------- |
| 模型版本      | CTPN                                          | CTPN                 |
| 资源           | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8  | Tesla V100 PCIE 32GB；CPU 2.60GHz; 104核；内存790G；EulerOS 2.0|
| 上传日期      | 02/06/2021                                    | 09/20/2021                 |
| MindSpore版本  | 1.1.1                                         | 1.5.0              |
| 数据集            | 229张图像                                   |229张图像                 |
| Batch_size          | 1                                             |1                         |
| 准确率           | precision=0.9079, recall=0.8611 F-measure:0.8839 | precision=0.9346, recall=0.8621 F-measure:0.8969 |
| 总时长         | 1 min                                         |1 min                      |
| 推理模型| 135M（.ckpt）                            | 135M（.ckpt）          |

#### 训练性能结果

| **Ascend** | 训练性能|
| :--------: | :---------------: |
|     1p     |     10 img/s      |

| **Ascend** | 训练性能|
| :--------: | :---------------: |
|     8p     |     84 img/s     |

| **GPU** | 训练性能|
| :--------: | :---------------: |
|     1p     |     6 img/s      |

| **GPU**| 训练性能|
| :--------: | :---------------: |
|     8p     |     52 img/s     |

# [随机情况说明](#目录)

我们在train.py中将种子设置为1。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
