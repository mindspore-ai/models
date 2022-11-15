<!-- TOC -->

- [目录](#目录)
- [SRCNN描述](#SRCNN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [ONNX评估](#ONNX评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ILSVRC2013_DET_train上的SRCNN](#ILSVRC2013_DET_train上的SRCNN)
        - [推理性能](#推理性能)
      - [Set5上的SRCNN](#Set5上的SRCNN)
    - [使用流程](#使用流程)
        - [在910推理](#在910推理)
        - [在310推理](#在310推理)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SRCNN描述

SRCNN是一个超分网络，学习低分辨率和高分辨率图像之间的端到端映射，结构简单，性能优于最先进的方法。

[论文](https://arxiv.org/abs/1501.00092)：Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks. 2014.

# 模型架构

SRCNN的网络结构仅包含三个卷积层，如下：

<center>
 <img src="./imgs/SRCNN.png" style="zoom:80%;" />
</center>

SRCNN首先使用双三次(bicubic)插值将低分辨率图像放大成目标尺寸，接着通过三层卷积网络拟合非线性映射，最后输出高分辨率图像结果。

# 数据集

使用的数据集：ILSVRC2013_DET_train

- 训练集

    ILSVRC2013_DET_train: 395918 images, 200 classes （目前官网404，可以在一些论坛找到该数据集）
- 验证集
    - Set5: 5 images
    - Set14: 14 images
    - BSDS200: 200 images
    - download_url: [https://gitee.com/a1085728420/srcnn-dataset](https://gitee.com/a1085728420/srcnn-dataset)

- 数据格式：RGB

  - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 数据预处理：为了加速数据读取，这里采用mindrecord来存储数据。因此需要先预处理数据。

  ILSVRC2013_DET_train数据集解压后，内部是各个分类的压缩包，因为SRCNN是超分网络，不需要进行分类，因此我们需要将tar文件解压，并进入所有的目录中将图片拷贝到一个指定的文件夹中去。

```bash
#! /bin/bash

tar -xvf ILSVRC2013_DET_train.tar
root=`pwd`
mkdir train
cd ILSVRC2013_DET_train
for file in `ls *tar`
do
    tar -xvf $file
done
for img in `find . -type f -name "*.JPEG" -maxdepth 1`
do
    mv $img $root/train
done

for dir in `find . -type d -maxdepth 1`
do
    cd $dir
    for img in `find . -type f -name "*.JPEG" -maxdepth 1`
    do
        mv $img $root/train
    done
    cd $root/ILSVRC2013_DET_train
done
```

可以参考该脚本，先进入数据集ILSVRC2013_DET_train.tar所在文件夹，然后执行该脚本，最后的图片就会在train文件夹下。

假设生成的train文件放在/dataset目录下。进入仓库所在目录，然后执行命令：

```bash
cd srcnn
python src/create_dataset.py --src_folder=/dataset/train --output_folder=/dataset/mindrecord
```

这样之后在mindrecord文件夹中会存放有对应的mindrecord格式的数据集。训练时的输入就是该数据集。

- Ascend 910环境运行

    - 单卡训练

    运行分布式训练脚本如下：

    ```bash
    cd srcnn/scripts
    bash run_single_train.sh device_id /dataset/mindrecord /output_path /pretrained_path(option)
    ```

    单卡训练脚本可以有3个或4个参数，
        - device_id代表在选择哪一张卡，是一个整数
        - /dataset/mindrecord是mindrecord格式的数据集
        - /output_path是生成的checkpoint文件存在的路径
        - 还有一个可选参数是预训练的checkpoint所在路径，可以导入已经训练好的checkpoint

    - 分布式训练

    对于分布式训练，需要提前创建JSON格式的hccl配置文件。请遵循以下链接中的说明：<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

    运行分布式训练脚本如下：

    ```bash
    cd srcnn/scripts
    bash run_distribute_train_ascend.sh /dataset/mindrecord /output_path begin_device device_num /hccl.json /pretrained_path(option)
    ```

    - /dataset/mindrecord是mindrecord格式的数据集

    - /output_path是生成的checkpoint文件存在的路径，

    - begin_device和device_num代表从begin_device开始，连续device_num张卡用于训练

    - hccl.json 是对应的hccl配置文件

    - 还有一个可选参数是预训练的checkpoint所在路径，可以导入已经训练好的checkpoint。

- GPU处理器环境运行

  为了在GPU处理器环境运行，请将配置文件config.yaml中的`device_target`从`Ascend`改为`GPU`

  ```bash
  cd srcnn/scripts
  bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_PATH] [OUTPUT_PATH] [PRE_TRAINED](optional)
  ```

  - DEVICE_NUM是GPU的数量，VISIABLE_DEVICES是可见的设备，DATA_PATH是mindrecord的路径，OUTPUT_PATH是输出的checkpoint保存的路径，PRE_TRAINED是可选的，是预训练的checkpoint所在路径，可以导入已经训练好的checkpoint。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

  - 在 ModelArts 上使用8卡训练 ImageNet 数据集

    ```python
    # (1) 上传mindrecord数据集到 S3 桶上。
    # (2) 在网页上设置你的代码路径为 "/path/SRCNN"
    # (3) 在网页上设置启动文件为 "train.py"
    # (4) 在网页上设置data_url和train_url，data_url指向S3桶里的mindrecord数据集，train_url表示训练结果输出到S3桶的位置,并设置运行参数enable_modelarts=True; run_distribute="True" ; device_num=8
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

```bash
└── SRCNN
    ├── README_CN.md    // 模型相关说明
    ├── ascend310_infer    // 实现310推理源码
    ├── config.yaml     // 配置文件
    ├── eval.py        // 评估脚本
    ├── export.py     // 将checkpoint文件导出到air/mindir
    ├── postprocess.py    // 310推理预处理数据
    ├── preprocess.py    // 310推理后处理数据
    ├── eval_onnx.py          // ONNX评估脚本
    ├── scripts
    │   ├── run_distribute_train_ascend.sh // 分布式到Ascend的shell脚本
    │   ├── run_distribute_train_gpu.sh    // 分布式到GPU处理器的shell脚本
    │   ├── run_eval_ascend.sh        // Ascend评估的shell脚本
    │   ├── run_eval_gpu.sh           // GPU处理器评估的shell脚本
    │   ├── run_infer_310.sh        // Ascend推理shell脚本
    │   ├── run_onnx_eval_gpu.sh                  // ONNX评估的shell脚本
    │   └── run_single_train_ascend.sh    // Ascend的单卡训练脚本
    ├── src
    │   ├── create_dataset.py        // 创建mindrecord数据集
    │   ├── dataset.py            // 创建数据集
    │   ├── metric.py            // PSNR指标
    │   ├── SRCNN.py            // 定义网络
    │   ├── model_utils          // model_utils
    │   └── utils.py            // 工具类
    ├── test.py                // 测试脚本
    └── train.py               // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置SRCNN:

  ```yaml
  # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
  enable_modelarts: "False"
  # Url for modelarts
  data_url: ""
  train_url: ""
  checkpoint_url: ""
  # Path for train
  data_path: "/cache/data/"
  output_path: "/cache/train/"
  pre_trained_path: ''
  pretrained_ckpt_path: '/cache/pretrained_ckpt/pretrained.ckpt'
  checkpoint_path: '/cache/checkpoint_path.ckpt'
  enable_profiling: False
  device_target: Ascend
  test_pic: ''
  # Path for create_dataset
  src_folder: ''
  output_folder: ''
  # Path for 310
  image_path: ''
  output_path: ''
  image_width: 512
  image_height: 512
  predict_path: ''
  result_path: ''
  # ==============================================================================
  # options
  device_num: 1
  lr: 0.0001
  patch_size: 33
  batch_size: 16
  epoch_size: 100
  save_checkpoint: True
  keep_checkpoint_max: 100
  run_distribute: False
  filter_weight: False
  scale: 2
  stride: 99
  ```

更多配置细节请参考脚本`config.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --data_path=/mindrecord --output_path=/output_path > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  epoch: 1 step: 39, loss is 0.024799313
  epoch time: 36501.857 ms, per step time: 935.945 ms
  epoch: 2 step: 39, loss is 0.023685196
  epoch time: 24.655 ms, per step time: 0.632 ms
  epoch: 3 step: 39, loss is 0.014360293
  epoch time: 23.282 ms, per step time: 0.597 ms
  epoch: 4 step: 39, loss is 0.009940531
  epoch time: 23.495 ms, per step time: 0.602 ms
  epoch: 5 step: 39, loss is 0.00808487
  epoch time: 23.282 ms, per step time: 0.597 ms
  epoch: 6 step: 39, loss is 0.0055490895
  epoch time: 23.437 ms, per step time: 0.601 ms
  epoch: 7 step: 39, loss is 0.00447102
  epoch time: 23.211 ms, per step time: 0.595 ms
  epoch: 8 step: 39, loss is 0.0043020057
  epoch time: 23.414 ms, per step time: 0.600 ms
  epoch: 9 step: 39, loss is 0.004429098
  epoch time: 23.418 ms, per step time: 0.600 ms
  epoch: 10 step: 39, loss is 0.0019083639
  epoch time: 23.278 ms, per step time: 0.597 ms
  epoch: 11 step: 39, loss is 0.0026995507
  epoch time: 23.245 ms, per step time: 0.596 ms
  epoch: 12 step: 39, loss is 0.0032862143
  epoch time: 23.579 ms, per step time: 0.605 ms
  epoch: 13 step: 39, loss is 0.0026632298
  epoch time: 23.513 ms, per step time: 0.603 ms
  epoch: 14 step: 39, loss is 0.0013880945
  epoch time: 24.802 ms, per step time: 0.636 ms
  epoch: 15 step: 39, loss is 0.0015905896
  epoch time: 23.525 ms, per step time: 0.603 ms
  epoch: 16 step: 39, loss is 0.0017711241
  epoch time: 23.521 ms, per step time: 0.603 ms
  epoch: 17 step: 39, loss is 0.0020061864
  epoch time: 23.510 ms, per step time: 0.603 ms
  epoch: 18 step: 39, loss is 0.0020959028
  epoch time: 23.802 ms, per step time: 0.610 ms
  epoch: 19 step: 39, loss is 0.001098047
  epoch time: 23.779 ms, per step time: 0.610 ms
  epoch: 20 step: 39, loss is 0.0016573562
  epoch time: 23.856 ms, per step time: 0.612 ms
  ```

  训练结束后，可以在输出目录中的`ckpt_0/`脚本文件夹下找到检查点文件。

- GPU处理器环境运行

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python train.py --data_path=/mindrecord --output_path=/output_path > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，可以在输出目录中的`ckpt_0/`脚本文件夹下找到检查点文件。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_single_train.sh device_id /dataset/mindrecord /output_path
  ```

  上述shell脚本将在后台运行分布训练。您可以通过device[X]/train*.log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "epoch" device*/train*.log
  device0/train0.log: 'epoch_size': 100,
  device0/train0.log:epoch: 1 step: 123384, loss is 0.00069705985
  device0/train0.log:epoch time: 188839.725 ms, per step time: 1.531 ms
  device0/train0.log:epoch: 2 step: 123384, loss is 0.0012211615
  device0/train0.log:epoch time: 122672.782 ms, per step time: 0.994 ms
  device0/train0.log:epoch: 3 step: 123384, loss is 0.0009864292
  device0/train0.log:epoch time: 121672.042 ms, per step time: 0.986 ms
  device0/train0.log:epoch: 4 step: 123384, loss is 0.0011620418
  device0/train0.log:epoch time: 119603.594 ms, per step time: 0.969 ms
  device0/train0.log:epoch: 5 step: 123384, loss is 0.0011751236
  device0/train0.log:epoch time: 122682.467 ms, per step time: 0.994 ms
  device0/train0.log:epoch: 6 step: 123384, loss is 0.00058690057
  device0/train0.log:epoch time: 121046.948 ms, per step time: 0.981 ms
  device0/train0.log:epoch: 7 step: 123384, loss is 0.00080206565
  device0/train0.log:epoch time: 122695.601 ms, per step time: 0.994 ms
  ...
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash run_distribute_train_gpu.sh 8 (0,1,2,3,4,5,6,7) /dataset/mindrecord /output_path
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel中找到log。

## 评估过程

### 评估

- 在Ascend环境或GPU处理器评估

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径。注意这里的数据集不需要经过处理，直接使用Set5数据集或者Set14数据集等数据集的绝对路径作为输入参数即可。

  ```bash
  python eval.py --data_path=/eval_dataset_path --checkpoint_path=/check_point  > eval.log 2>&1 &
  OR
  bash run_eval_ascend.sh [DEVICE_ID] /eval_dataset_path /check_point
  ## [DEVICE_ID]为整数，为执行npu-smi info后空闲卡的编号
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "result" eval.log
  result  {'PSNR': 36.76535539859346}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，

### ONNX评估

- 导出ONNX模型

  ```bash
  python export.py --checkpoint_path=/path/to/checkpoint.ckpt --device_target=GPU --file_format="ONNX"
  ```

  上述命令运行后，将在当前目录下生成srcnn.onnx

- 运行ONNX模型评估

  ```bash
  bash scripts/run_onnx_eval_gpu.sh DATA_PATH DEVICE_ID ONNX_MODEL_PATH
      DATA_PATH 为推理数据路径
      DEVICE_ID 为GPU设备id
      ONNX_MODEL_PATH 为onnx模型路径
  #example: bash scripts/run_onnx_eval_gpu.sh /path/to/data 0 /path/to/srcnn.onnx
  ```

- 上述命令将在后台运行，运行结束后可以通过文件`eval_onnx.log`查看结果。将会得到如下精度：

  ```bash
  # Set5
  PSNR: 42.2718
  # Set14
  PSNR: 35.3049
  #BSDS200
  PSNR: 36.1434
  ```

## 导出过程

### 导出

在导出之前需要指定所需导出模型文件格式和checkpoint文件路径，MINDIR为默认导出格式：

```shell
python export.py --checkpoint_path /checkpoint_path --file_format="MINDIR"
```

并且注意因为310推理只支持静态shape，所以需要指定图片的形状，可以通过修改config.yaml的image_width和image_height来改变。
file_format 可在["AIR","MINDIR"]中选择。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用CIFAR-10数据集进行推理

  在执行下面的命令之前，我们需要先修改cifar10的配置文件。修改的项包括batch_size和val_data_path。LABEL_FILE参数只对ImageNet数据集有用，可以传任意值。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
  after allreduce eval: top1_correct=9252, tot=10000, acc=92.52%
  ```

# 模型描述

## 性能

### 评估性能

#### ILSVRC2013_DET_train上的SRCNN

| 参数          | Ascend                                                   | GPU                                  |
| ------------- | -------------------------------------------------------- | ------------------------------------ |
| 模型版本      | Inception V1                                             | Inception V1                         |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8   | NV SMX2 V100-32G                     |
| 上传日期      | 2021-12-08                                               | 2021-12-08                           |
| MindSpore版本 | 1.3.0                                                    | 1.3.0                                |
| 数据集        | ILSVRC2013_DET_train, scale:2                            | ILSVRC2013_DET_train, scale:2        |
| 训练参数      | epoch=20, batch_size = 16, lr=0.0001                     | epoch=20, batch_size = 16, lr=0.0001 |
| 优化器      | Adam                                                       | Adam                                 |
| 损失函数      | MSE Loss                                                 | MSE Loss                             |
| 输出          | 图片                                                     | 图片                                 |
| 损失          | 0.00163                                                  | 0.00179                              |
| PSNR          | SET5:36.71                                               |                                     |
|               | SET14:32.586                                             |                                     |
|               | BSD200:33.809                                            |                                    |
| 总时长        | 单卡：2小时15分钟 <br/>8卡：16分钟                         | 1 h 8ps                             |
| 性能规格（ms/step）   | 单卡：0.67                                        |                                     |

### 推理性能

#### Set5上的SRCNN

| 参数           | Ascend                        | GPU          |
| -------------- | ----------------------------- | ------------ |
| 模型版本       | Inception V1                  | Inception V1 |
| 资源           | Ascend 910；系统 Euler2.8     | GPU          |
| 上传日期       | 2021-12-08                    | 2021-12-08   |
| MindSpore 版本 | 1.3.0                         | 1.3.0        |
| 数据集         | Set5                          | Set5         |
| batch_size     | 1                             | 1            |
| 输出           | 图片                          | 图片         |
| PSNR           | 单卡： 36.70<br /> 8卡: 36.63 | 36.72        |

## 使用流程

### 在910推理

执行test.py推理脚本：

```bash
python test.py --data_path=/picture_path --checkpoint_path=/checkpoint_path
```

picture_path是一个指向图片的绝对路径，checkpoint_path是指向checkpoint文件的绝对路径。

执行完成后，对应的图片会被存放到picture_path路径下。

### 在310推理

```bash
cd srcnn/scripts
bash run_infer_310.sh /srcnn.mindir /data_path/ ../result/
```

其中srcnn.mindir是导出的mindir文件的绝对路径,data_path是数据集所在的目录的绝对路径，会将数据集中所有图片预测的结果放在result中。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。