# Contents

- [TSN 介绍](#TSN-介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本介绍](#脚本介绍)
    - [脚本以及简单代码](#脚本以及简单代码)
    - [脚本参数](#脚本参数)
    - [训练步骤](#训练步骤)
        - [训练](#训练)
    - [评估步骤](#评估步骤)
        - [评估](#评估)
    - [ONNX评估](#ONNX评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型介绍](#模型介绍)
    - [性能](#性能)  
        - [评估性能](#评估性能)
- [随机事件介绍](#随机事件介绍)
- [ModelZoo 主页](#ModelZoo-主页)

# [TSN 介绍](#contents)

TSN网络用于视频分类，是一种双流网络架构。TSN网络主要解决数据集不足以及视频中存在大量冗余信息，针对数据集问题，TSN网络使用了四种数据：RGB、RGB-DIFF、Flow、Warped-flow；对于连续视频帧中存在大量冗余信息，作者采用稀疏采样的策略选取视频帧。

[Paper](https://arxiv.org/abs/1608.00859): Limin Wang, Yuanjun Xiong, Zhe Wang, Dahua Lin, Xiaoou Tang, Luc Van Gool. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. aeXiv preprint arXiv:1608.00859, 2016.

# [模型架构](#contents)

TSN主要使用BNInception网络作为基础，再根据作者提出的数据稀疏采样对网络做简单修改。

# [数据集](#contents)

Dataset used:

[UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)和[标签信息](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)

数据处理：对于Flow类型数据提取方法，使用dense_flow进行提取，[dense_flow安装方法](https://github.com/open-mmlab/mmaction/blob/master/INSTALL.md)。

# [环境要求](#contents)

- 硬件（Ascend/GPU）
    - 需要准备具有Ascend或GPU处理能力的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [快速开始](#contents)

在通过官方网站安装MindSpore之后，你可以通过如下步骤开始训练以及评估：

- running on Ascend with default parameters

  ```bash
  # 单卡训练
  python train.py --data_url "" --train_url "" --dataset ucf101 --train_list_path "" --train_list "" --modality "" --pretrained_path "" --pre_trained_name "" --run_distribute False --run_modelarts False

  # 多卡训练
  #For flow models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] Flow [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]

  #For warmup_flow models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] Flow [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]

  #For rgb models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] RGB [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]
  ```

- flow、warmup_flow、rgb三者之间是数据集不同(ucf101数据处理不同)

# [脚本介绍](#contents)

## [脚本以及简单代码](#contents)

```path
├── TSN
    ├── scripts
        ├── run_distribute_train_ascend.sh       //training on Ascend with 8P
        ├── run_distribute_gpu.sh                //training on GPU with 8P
        ├── run_eval_gpu.sh.sh               //evaling Flow+RGB on GPU
        ├── run_standalone_train_gpu.sh      //training on GPU with 1P
        ├── run_test_onnx_gpu.sh                  //testing onnx Flow or RGB on GPU
        ├── run_test_gpu.sh                  //testing Flow or RGB on GPU
    ├── src
        ├──basic_ops.py                     // basic operation
        ├──config.py                        // parameter
        ├──dataset.py                       // create dataset
        ├──metrics.py                       // calculation accuracy
        ├──models.py                        // network
        ├──network.py                       // backbone
        ├──transforms.py                    // process dataset
        ├──tsn_for_train.py                 // clip
        ├──util.py                          // util
        ├──video_funcs.py                   // util for test
    ├── eval_scores.py                      // fusion accuracy
    ├── requirement.txt                     // requirement
    ├── test_net.py                         // tesing network performance
    ├── test_net_onnx.py                    // tesing network onnx performance
    ├── train.py                            // traing network
```

## [脚本参数](#contents)

训练以及评估的参数可以在config.py中设置

- config for TSN

  ```bash
     tsn_flow = edict({
    'learning_rate': 0.005,
    'epochs': 340,
    'lr_steps': 70,
    'gamma': 0.1,
    'dropout': 0.3,
    'num_segments': 3,
    })
  ```

如需查看更多信息，请查看`config.py`.

## [训练步骤](#contents)

### 训练

- running on Ascend

  ```bash
  #1P训练
  python train.py --data_url "" --train_url "" --dataset ucf101 --train_list_path "" --train_list "" --modality "" --pretrained_path "" --pre_trained_name "" --run_distribute False --run_modelarts False

  #8P训练
  #For flow models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] Flow [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]

  #For warmup_flow models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] Flow [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]

  #For rgb models:
  bash run_distribute_train.sh  [RANK_TABLE_FILE] [TRAIN_URL] [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] RGB [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]

  ```

- running on GPU

  ```bash
  #1P训练
  bash run_standalone_train_gpu.sh [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] [MODALITY] [PRETRAINED_PATH] [PRETRAINED_PATH_NAME] [DEVICE_ID]

  #8P训练
  bash run_distribute_train_gpu.sh [DATASET_PATH] [DATASET] [TRAIN_LIST_PATH] [TRAIN_LIST] [MODALITY] [PRETRAINED_PATH] [PRETRAINED_PATH_NAME]
  ```

  [DATASET_PATH]：data所在文件夹/data/data_extracted/ucf101/tvl1
  [DATASET]:ucf101
  [TRAIN_LIST_PATH]:data所在文件夹/data/data_extracted/ucf101/
  [TRAIN_LIST]:ucf101_train_split_1_rawframes.txt
  [MODALITY]: RGB或者Flow
  [PRETRAINED_PATH]:ckpt预训练文件所在文件夹
  [PRETRAINED_PATH_NAME]:ckpt预训练文件tsn_rgb.ckpt或者tsn_flow.ckpt

- flow、warmup_flow、rgb三者之间是数据集不同(ucf101数据处理不同)

  8P训练时需要将RANK_TABLE_FILE放在scripts文件夹中，RANK_TABLE_FILE[生成方法](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

  训练时，训练过程中的epch和step以及此时的loss和精确度会呈现在终端上：

  ```bash
    epoch: 1 step: 1, loss is 4.61211
    epoch: 1 step: 2, loss is 4.601388
    epoch: 1 step: 3, loss is 4.611535
    epoch: 1 step: 4, loss is 4.533617
    epoch: 1 step: 5, loss is 4.6036034
    epoch: 1 step: 6, loss is 4.5772705
    epoch: 1 step: 7, loss is 4.6626453
    epoch: 1 step: 8, loss is 4.5711393
    epoch: 1 step: 9, loss is 4.4796357
    epoch: 1 step: 10, loss is 4.471799
    epoch: 1 step: 11, loss is 4.3990126
    epoch: 1 step: 12, loss is 4.680209
    epoch: 1 step: 13, loss is 4.3441525
    epoch: 1 step: 14, loss is 4.5748014
    epoch: 1 step: 15, loss is 4.4087944
    epoch: 1 step: 16, loss is 4.4659142
  ...
  ```

  -Ascend:模型的checkpoint存储在train_parallel0/checkpoint路径中

  -GPU:模型的checkpoint存储在tsn/checkpoint路径中

## [评估步骤](#contents)

### 评估

- 在Ascend上使用ucf101 测试集进行评估

  在使用命令运行时，需要传入模型参数地址、模型参数名称、空域卷积方式、预测时段。

  ```bash
  #For flow models:
  python test_net.py --weights "" --device_id 0 --dataset "" --modality Flow --test_list "" --dataset_path "" --save_scores ""

  #For warmup_flow models:
  python test_net.py --weights "" --device_id 0 --dataset "" --modality Flow --test_list "" --dataset_path "" --save_scores ""

  #For rgb models:
  python test_net.py --weights "" --device_id 0 --dataset "" --modality RGB --test_list "" --dataset_path "" --save_scores ""

  python eval_scores.py score_files_flow score_files_warmup_flow score_files_rgb
  ```

 - score_files_flow、score_files_warmup_flow、score_files_rgb为在test_net.py中生成的npz文件

  以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以如下方式呈现：

  ```bash
  RGB:Accuracy 86.0%
  Flow:Accuracy 87.6%
  Warmup_Flow:Accuracy 87.4%
  RGB+Flow+Warmup_Flow:Accuracy 93.1%
  ```

- 在GPU上使用ucf101 测试集进行评估

  在使用命令运行时，需要传入模型参数地址、模型参数名称、使用设备id。同时需将各文件按如下格式放置。

  ```path
  ├── root_path
      ├── tsn
      ├── data
          ├──data_extracted
              ├──ucf101
                  ├──tvl1                      // RGB和Flow图片存放路径
                  ├──ucf101_val_split_1_rawframes.txt         // 标签存放路径
  ```

  ```bash
  #For flow models:
  用法：bash run_test_gpu.sh [ROOT_PATH] [CKPT_PATH] [MODALITY] [DEVICE_ID]
  实例：bash run_test_gpu.sh /path1/ /path2/ucf101_bninception_Flow-340_597.ckpt Flow 0

  #For rgb models:
  用法：bash run_test_gpu.sh [ROOT_PATH] [CKPT_PATH] [MODALITY] [DEVICE_ID]
  实例：bash run_test_gpu.sh /path1/ /path2/ucf101_bninception_RGB-340_597.ckpt RGB 0
  ```

    - `[ROOT_PATH]` 模型存放的根路径，即tsn文件夹的上层路径
    - `[CKPT_PATH]` ckpt文件的存放路径
    - `[MODALITY]` 数据集格式，此处为RGB或者Flow
    - `[DEVICE_ID]` 使用的设备id

  Flow和RGB模型评估完成后，会分别生成一个npz文件，需分别将npz文件放置在/path/tsn/checkpoint/RGB以及/path/tsn/checkpoint/Flow目录中，然后执行如下命令评估flow+rgb。

  ```bash
  #For RGB+Flow:
  用法：bash run_eval_gpu.sh [ROOT_PATH] [RGB_NAME] [FLOW_NAME]
  实例：bash run_eval_gpu.sh /path/ score_warmupRGB.npz score_warmupFlow.npz
  ```

  - `[ROOT_PATH]` 模型存放的根路径，即tsn文件夹的上层路径
  - `[RGB_NAME]` 推理RGB生成的npz文件的名称
  - `[FLOW_NAME]` 推理Flow生成的npz文件的名称

RGB和Flow的评估结果保存在示例路径中，文件名为“~/{MODALITY}_test.log”。您可在此路径下的日志找到如下结果：

- 使用ucf101_RGB评估tsn

  ```text
  RGB:Accuracy 85.5%
  ```

- 使用ucf101_Flow评估tsn

  ```text
  Flow:Accuracy 88.4%
  ```

RGB+Flow的评估结果保存在示例路径中，文件名为“~/eval_score.log”。您可在此路径下的日志找到如下结果：

  ```text
  RGB+Flow:Accuracy 93.7%
  ```

## ONNX评估

### 导出onnx模型

```bash
python export.py --ckpt_path /path/ucf101_bninception_RGB-21_597.ckpt --modality RGB --platform GPU --file_format ONNX
```

- `ckpt_file` ckpt文件路径
- `modality` 数据集格式，RGB或者Flow
- `platform` 目前仅支持GPU或CPU
- `file_format` 导出模型格式，此处为ONNX

### 运行ONNX模型评估

```bash
用法：bash run_test_onnx_gpu.sh [ONNX_PATH] [MODALITY] [DATA_DIR] [TEST_LIST] [SCORE_SAVE_PATH]
实例：bash run_test_onnx_gpu.sh /path/tsn_RGB.onnx RGB /path/ucf101/ /path/ucf101_val_split_1_rawframes.txt /path/scores_RGB_onnx
 ```

- `[ONNX_PATH]` onnx模型路径
- `[MODALITY]` 数据集格式，此处为RG或者Flow
- `[DATA_DIR]` 数据集路径
- `[TEST_LIST]` 模型标签的路径
- `[SCORE_SAVE_PATH]` npz文件保存的路径

flow和rgb模型评估完成后，会分别生成一个npz文件，需分别将npz文件放置在/path/tsn/checkpoint/RGB以及/path/tsn/checkpoint/Flow中，然后执行如下命令评估flow+rgb。

  ```bash
  #For RGB+Flow:
  用法：bash run_eval_gpu.sh [ROOT_PATH] [RGB_NAME] [FLOW_NAME]
  实例：bash run_eval_gpu.sh /path/ scores_RGB_onnxRGB.npz scores_Flow_onnxFlow.npz
  ```

- `[ROOT_PATH]` 模型存放的根路径，即tsn文件夹的上层路径
- `[RGB_NAME]` 推理RGB生成的npz文件的名称
- `[FLOW_NAME]` 推理Flow生成的npz文件的名称

### 结果

RGB和Flow评估结果保存在示例路径中，文件名为“~/{MODALITY}_test.log”。您可在此路径下的日志找到如下结果：

- 使用ucf101_RGB评估tsn

```text
RGB:Accuracy 85.5%
```

- 使用ucf101_Flow数据集评估tsn

```text
Flow:Accuracy 88.4%
```

RGB+Flow的评估结果保存在示例路径中，文件名为“~/eval_score.log”。您可在此路径下的日志找到如下结果：

```text
RGB+Flow:Accuracy 93.7%
```

## [导出mindir模型](#contents)

```shell
python export.py --device_id [DEVICE_ID] --weights [WEIGHT] --modality [Flow]
```

## [推理过程](#contents)

### 用法

执行推断之前，minirir文件必须由export.py导出。输入文件必须为bin格式

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DATA_PATH] [MODALITY] [TEST_LIST] [DEVICE_TARGET] [DEVICE_ID]
```

### 结果

```shell
cat acc.log
```

# [模型介绍](#contents)

## [性能](#contents)

### 评估性能

#### TSN on ucf101(Flow)

| Parameters                 | ModelArts                                                |GPU |
| -------------------------- | ---------------------------------------------------------|----|
| Model Version              | TSN                                                      |TSN |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G          | NV SMX2 V100-32G         |
| uploaded Date              | 08/11/2021 (month/day/year)                              |01/06/2022(month/day/year)|
| MindSpore Version          | 1.3.0                                                    |1.6.0|
| Dataset                    | UCF101                                                   |UCF101|
| Training Parameters        | epoch=340, steps=75, batch_size=8, lr=0.001              |epoch=340, steps=75, batch_size=8, lr=0.005  |
| Optimizer                  | SGD                                                      |SGD|
| Loss Function              | SoftmaxCrossEntropyWithLogits                            |SoftmaxCrossEntropyWithLogits |
| outputs                    | accuracy                                                 |accuracy|
| Loss                       | 0.183                                                    |0.1615
| Speed                      | 8pc: 300.601 ms/step;                                    |8pc:250 ms/step|
| Total time                 | 8pc: 4h;                                                 |8pc:2.5h|
| Scripts                    | [TSN script](https://gitee.com/mindspore/models/tree/master/research/cv/tsn) |

### Inference Performance

#### TSN on ucf101

| Parameters          | Ascend(Flow + Warped Flow + RGB) |GPU (Flow+RGB) |
| ------------------- | ----------------------------|-------------------|
| Model Version       | TSN                         |TSN                |
| Resource            | Ascend 910                  | V100                  |
| Uploaded Date       | 08/11/2021 (month/day/year) |01/06/2022(month/day/year) |
| MindSpore Version   | 1.3.0                       |1.6.0              |
| Dataset             | UCF101                      |UCF101             |
| batch_size          | 16                          |16                 |
| outputs             | accuracy                    |acuuracy           |
| accuracy            | 93.1%                       |93.7%              |
| Model for inference | about 40M(.ckpt fil)        |about 40M(.ckpt fil)|

# [随机事件介绍](#contents)

我们在train.py中设置了随机种子

# [ModelZoo 主页](#contents)

 请查看官方网站 [homepage](https://gitee.com/mindspore/models).