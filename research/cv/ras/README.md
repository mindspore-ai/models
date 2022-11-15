# 目录

<!-- TOC -->

- [目录](#目录)
- [RAS概述](#RAS概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
  - [脚本和样例代码](#脚本和样例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
      - [用法](#用法)
      - [结果](#结果)
  - [导出过程](#导出过程)
      - [导出](#导出)
  - [推理过程](#推理过程)
      - [用法](#用法-1)
      - [结果](#结果-1)
      - [Ascend310推理](#Ascend310推理)
  - [评估推理结果](#评估推理结果)
  - [ONNX模型导出及评估](#onnx模型导出及评估)
      - [ONNX模型导出](#onnx模型导出)
      - [ONNX模型评估](#onnx模型评估)
- [模型描述](#模型描述)
  - [评估精度](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RAS概述

首先，该模型提出了两种初始预测策略，一种是新设计的多尺度上下文模块，另一种是结合手工制作的显著性先验。
其次，利用残差学习逐步细化，只学习各边输出中的残差，这可以在较少的卷积参数下实现，从而获得较高的紧凑性和效率。
最后，进一步设计了一种新的自上而下的反向注意块来指导上述侧输出残差学习。具体来说，利用当前预测的显著区域来去除其侧输出特征，从而可以有效地从这些未删除的区域中学习到缺失的目标部分和细节，从而实现检测更完整和更高的精度

[论文](https://ieeexplore.ieee.org/abstract/document/8966594)

## 模型架构

RAS总体网络架构如下:

[链接](https://ieeexplore.ieee.org/abstract/document/8966594)

## 数据集

训练集

- DUTS-Train
    - image 10553张
    - ground truth 10553张
- 注：数据集在src/dataset_train.py中处理

测试集

- DUTS-Test
    - 共10218张图片
    - images 5109张
    - ground truth 5109张
- ECSSD
    - 共2000张图片
    - images 1000张
    - ground truth 1000张
- DUT-OMRON
    - 共10336张图片
    - images 5168张
    - ground truth 5168张
- HKU-IS
    - 共8894张图片
    - images 4447张
    - ground truth 4447张
- 注：数据集在src/dataset_test.py中处理

## 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html) 的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件：昇腾处理器（Ascend/GPU）
    - 使用昇腾处理器或GPU来搭建硬件环境。

- 框架
  - [MindSpore](https://www.mindspore.cn/install)

- 如需查看详情，请参见如下资源：

  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)

  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 脚本说明

## 脚本和样例代码

``` python
├── RAS
  ├── Readme.md
  ├── ascend310_infer   # 310推理目录
  ├── scripts
  │   ├──run_distribute_train.sh # 使用昇腾处理器进行八卡训练的shell脚本
  │   ├──run_train.sh    # 使用昇腾处理器进行单卡训练的shell脚本
  │   ├──run_eval.sh  # 使用昇腾处理器进行评估的单卡shell脚本
  │   ├──run_distribute_train_gpu.sh # 使用GPU进行多卡训练的shell脚本
  │   ├──run_train_gpu.sh    # 使用GPU进行单卡训练的shell脚本
  │   ├──run_eval_gpu.sh  # 使用GPU进行评估的单卡shell脚本
  │   ├──run_eval_onnx_gpu.sh  # 使用GPU对导出的onnx模型进行评估的单卡shell脚本
  │   ├──run_infer_310.sh   # 310推理脚本
  ├──src
  │   ├──dataset_train.py #创建训练数据集
  │   ├──dataset_test.py # 创建推理数据集
  │   ├──loss.py        #RAS训练使用的loss函数
  │   ├──model.py       #RAS网络模型
  │   ├──resnet50.py  #RAS使用的boneback
  │   ├──TrainOneStepMyself.py  #自定义训练，参数更新过程
  ├── train.py # 训练脚本
  ├── eval.py # 推理脚本
  ├── eval_onnx.py # onnx推理脚本
  ├── export.py
  ├── export_onnx.py # onnx导出脚本
```

### 脚本参数

- 配置RAS和DUTS-Train数据集。

  ``` python
    'epoch'：35                          //训练epoch数
    'learning_rate' : 0.00005           //训练学习率
    'batchsize' : 10                    //训练时batch大小
    'image_height' : 352                //训练图片的高度
    'image_width'  : 352                //训练图片的宽度
    'gt_height'  : 352                  //训练ground truth图片高度
    'gt_width'   : 352                  //训练ground truth图片宽度
    'print_flag' : 20                   //训练时每print_flag个step输出一次loss
    'device_id'  : 5                   //训练时硬件的ID
    'data_url'   : xxx                 //数据路径
    'onnx_file'  : xxx                 //导出的onnx模型路径
    'pretrained_model':xxx             //resnet50预训练模型路径 在eval该参数为"pre_model"
  ```

## 训练过程

### 用法

- **注：在建立训练数据路径时，在目录最后一级创建两个文件夹images(存放训练图片),labels(存放GT);modelarts模式下无需建立目录，直接存放images.zip,labels.zip即可**
- Ascend处理器环境运行

``` python
  - 直接使用python3在终端进行运行 ：
     如：python3 -u train.py --is_modelarts NO --distribution_flag NO --device_target Ascend --device_id 5 --lr 0.00005 --data_url '' --pretrained_model '' --train_url ''> output.log 2>&1 &
         is_modelarts 为是否在modelarts运行
         distribution_flag 为是否分布式训练
         device_target 为硬件环境(默认为Ascend)
         device_id 为硬件环境中的芯片ID
         lr 为学习率
         data_url 为训练数据路径
         pretrained_model 为resnet50预训练模型路径
         train_url 为输出的ckpt保存路径
    - bash运行
        bash script/run_train.sh device_id lr data_url pretrained_model train_url    //单卡训练
        bash script/run_distribute_train.sh json_file rank_size data_url pretrained_model train_url    //多卡分布式训练
        注：json_file 为多卡训练的json_file文件路径
           rank_size 为多卡训练时需要的卡数
```

- GPU处理器环境运行

  resnet50预训练模型下载路径：[https://download.mindspore.cn/model_zoo/r1.3/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0.ckpt](https://download.mindspore.cn/model_zoo/r1.3/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0/resnet50_gpu_v130_imagenet_official_cv_bs32_acc0.ckpt)

``` python
  - 直接使用python3在终端进行运行 ：
     如：python3 -u train.py --is_modelarts NO --distribution_flag NO --device_target GPU --device_id 5 --lr 0.00005 --data_url '' --pretrained_model '' --train_url ''> output.log 2>&1 &
         device_id 为硬件环境中的芯片ID
         lr 为学习率
         data_url 为训练数据路径
         pretrained_model 为resnet50预训练模型路径
         train_url 为输出的ckpt保存路径
  - bash运行
    bash script/run_train_gpu.sh device_id lr data_url pretrained_model train_url    //单卡训练
    bash script/run_distribute_train_gpu.sh rank_size data_url pretrained_model train_url    //多卡分布式训练
        rank_size 为多卡训练时需要的卡数
```

### 结果

训练结果保存在示例路径中。检查点默认保存在`--train_url`中，训练日志重定向到`output/output.log`，内容如下：

``` python
epoch:1, learning_rate:0.00005000,iter [20/10553],Loss    ||  0.7333122
The Consumption of per step is 2.555 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [40/10553],Loss    ||  0.5926424
The Consumption of per step is 0.143 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [60/10553],Loss    ||  0.46602067
The Consumption of per step is 0.141 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [80/10553],Loss    ||  0.38317975
The Consumption of per step is 0.133 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [100/10553],Loss    ||  0.29325977
The Consumption of per step is 0.136 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [120/10553],Loss    ||  0.31571442
The Consumption of per step is 0.130 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [140/10553],Loss    ||  0.3087693
The Consumption of per step is 0.124 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [160/10553],Loss    ||  0.26840287
The Consumption of per step is 0.133 s
+++++++++++++++++++++++++++++++++++++++++++++++++
epoch:1, learning_rate:0.00005000,iter [180/10553],Loss    ||  0.27287382
The Consumption of per step is 0.136 s
```

## 导出过程

### 导出

``` python
python export.py --pre_model [PRE_MODEL_PATH] --ckpt_file [CKPT_PATH] --file_format [FILE_FORMAT]
```

- `PRE_MODEL_PATH` 为resnet50预训练模型路径
- `CKPT_PATH` 为训练保存的ckpt路径
- `FILE_FORMAT` 为导出的文件格式，默认“MINDIR”，可选“AIR”、 “ONNX”、 “MINDIR”

执行完后会在当前路径生成RAS.mindir文件。

## 推理过程

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### 用法

- **注：在推理数据路径的最后一级目录下建立文件夹images和gts,分别将图片和groundtruth存入其中;modelarts模式下无需建立images，直接存储images.zip和gts.zip**
- Ascend处理器环境运行

``` python
# 推理示例
  python3 -u eval.py --is_modelarts NO --device_target Ascend --device_id 5 --data_url xxx --model_path xxx --pre_model xxx
        device_id 为要进行推理的机器的ID
        data_url  为推理数据路径
        model_path 为训练保存的ckpt路径
        pre_model 为网络resnet50预训练模型路径
  bash 运行
        bash script/run_eval.sh device_id data_url train_url model_path pre_model
```

- GPU处理器环境运行

``` python
# 推理示例
  python3 -u eval.py --is_modelarts NO --device_target GPU --device_id 0 --data_url xxx --model_path xxx --pre_model xxx
        device_id 为要进行推理的机器的ID
        data_url  为推理数据路径
        model_path 为训练保存的ckpt路径
        pre_model 为网络resnet50预训练模型路径
  bash 运行
        bash script/run_eval_gpu.sh device_id data_url train_url model_path pre_model
```

### 结果

推理结果保存在示例路径中，可以在`--train_url`中找到如下结果,日志可在`output/eval_output.log`中找到：

``` python
该推理过程结束后,会在--train_url中生成结果图片,为了评估推理结果，需要将图片继续进行处理
```

### Ascend310推理

在进行推理之前我们需要先导出模型。`mindir`文件需要通过`export.py`导出。

在Ascend310上进行推理：

``` sh
# Ascend310 inference
bash run_infer_310.sh [GEN_MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID] [GT_PATH]
```

- `GEN_MINDIR_PATH`代表生成的mindir文件
- `DATA_PATH`代表数据集路径
- `NEED_PREPROCESS`表示是否需要预处理，它的值为'y'或者'n'
- `DEVICE_ID`表示执行推理的设备id
- `GT_PATH`可选，groundtruth路径，若不为空则对结果进行评估

推理结果保存在当前路径的`result_Files/`目录下，推理时间则保存在`time_Result/`目录下，若对结果进行了评估，则可以在当前路径`evaluation.log`中查看评估结果。

## 评估推理结果

推理完成后，要对结果进行处理，为了方便，已经将评估部分加入到推理中，在推理完成后即可看到
该推理结果的Fmeasure，在推理的log中可以找到

## ONNX模型导出及评估

### ONNX模型导出

```bash
    python export_onnx.py --device_target [DEVICE_TARGET] --pre_model [PRE_MODEL] --ckpt_file [CKPT_FILE]
        ckpt_file 为训练保存的ckpt路径
        pre_model 为网络resnet50预训练模型路径
    # example: python export_onnx.py --device_target "GPU" --pre_model resnet50_gpu_v130_imagenet_official_cv_bs32_acc0.ckpt --ckpt_file ras_ascend_v170_dutstrain_research_cv_ECSSD91_DUTStest81_DUTOMRON75_HKUIS90.ckpt
```

### ONNX模型评估

```bash
    bash script/run_eval_onnx_gpu.sh [data_url] [save_url] [onnx_file]
        data_url  为推理数据路径
        save_url  为生成结果图片的路径
        onnx_file 为导出的onnx文件路径
    # example: bash script/run_eval_onnx_gpu.sh dataset/HKU-IS/ ./output_hku_is ras_onnx.onnx
```

# 模型描述

## 评估精度

| 参数列表 | Ascend 910 | GPU |
| -------------------------- | ----------------------------- | ------ |
| 模型版本 | RAS | RAS |
| 资源 |  Ascend 910；CPU 2.60HGHz 系统 Euler2.8  | RTX-3090 |
| 上传日期 | 2021-11-30 | 2021-12-23 |
| MindSpore版本 | 1.5 | 1.5 |
| 数据集 | ECSSD DUTS-Test  DUT-OMRON HUK-IS |  ECSSD DUTS-Test  DUT-OMRON HUK-IS |
| batch_size | 1 | 1 |
| 输出 | 3位有效数字小数 | 3位有效数字小数 |
| F-measure | 0.921  0.820  0.749  0.907 | 0.920 0.819 0.751 0.906 |

## 随机情况说明

train.py中使用了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。