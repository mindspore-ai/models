# 目录

- [目录](#目录)
    - [MobileNetV1描述](#mobilenetv1描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [特性](#特性)
        - [混合精度（Ascend）](#混合精度ascend)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
        - [用法](#用法)
        - [运行](#运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法)
        - [运行](#运行)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

## [MobileNetV1说明](#目录)

MobileNetV1是应用于移动和嵌入视觉的高效卷积神经网络。MobileNets基于流线型架构，使用深度可分离卷积来构建轻量级深度神经网络。

[论文](https://arxiv.org/abs/1704.04861) Howard A G , Zhu M , Chen B , et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. 2017.

## [模型架构](#目录)

MobileNetV1的整体网络架构如下：

[链接](https://arxiv.org/abs/1704.04861)

## [数据集](#目录)

您可以基于原始论文中提到的数据集运行脚本，也可以采用在相关域/网络架构中广泛使用的脚本。接下来我们将介绍如何使用下面的数据集运行脚本。

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：224*224张彩色图像，1000个类别
    - 训练集：1,281,167张图像
    - 测试集：50,000张图像
- 数据格式：.jpeg
    - 注：数据将在dataset.py中处理。

使用的数据集：[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- 数据集大小：175M，60,000张32*32彩色图像，10个类别
    - 训练集：146M，50,000张图像
    - 测试集：29M，10,000张图像
- 数据格式：二进制文件
    - 注：数据将在dataset.py中处理。

- 下载数据集，目录结构如下：

    ```ImageNet2012
    └─ImageNet_Original
        ├─train                # 训练数据集
        └─validation_preprocess # 评估数据集
    ```

    ```cifar10
    └─cifar10
        ├─cifar-10-batches-bin  # 训练数据集
        └─cifar-10-verify-bin  # 评估数据集
    ```

## 特性

### 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

## 环境要求

- 硬件（Ascend，GPU，CPU）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

- [ModelArts](https://support.huaweicloud.com/modelarts/)环境上运行

    ```bash
    # Ascend环境上运行8卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"distribute=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data'"。
    #          在default_config.yaml文件中设置"epoch_size=90"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"distribute=True"。
    #          在网页上添加"dataset_path=/cache/data"。
    #          在网页上添加"epoch_size=90"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/mobilenetv1"。
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
    #          在default_config.yaml文件中设置"epoch_size=90"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"dataset_path='/cache/data'"。
    #          在网页上添加"epoch_size=90"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/mobilenetv1"。
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
    #          在default_config.yaml文件中设置"checkpoint='./mobilenetv1/mobilenetv1_trained.ckpt'"。
    #          在default_config.yaml文件中设置"dataset_path='/cache/data'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='ImageNet_Original'"。
    #          在网页上添加"checkpoint_url='s3://dir_to_your_trained_model/'"。
    #          在网页上添加"checkpoint='./mobilenetv1/mobilenetv1_trained.ckpt'"。
    #          在网页上添加"dataset_path='/cache/data'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）上传或复制训练好的模型到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，将MindRecord数据集压缩到一个zip文件中。
    #          再将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/mobilenetv1"。
    # （6）在网页上设置启动文件为"eval.py"。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    ```

- 在ModelArts上导出并开始评估（如果你想在ModelArts上运行，可以参考[ModelArts](https://support.huaweicloud.com/modelarts/)官方文档。

1. 使用voc val数据集在ModelArts上导出并评估多尺度翻转s8。

    ```python
    # （1）执行a或b。
    #       a. 在base_config.yaml文件中设置"enable_modelarts=True"。
    #          在base_config.yaml文件中设置"file_name='mobilenetv1'"。
    #          在base_config.yaml文件中设置"file_format: 'MINDIR'"。
    #          在beta_config.yaml文件中设置"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在base_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在base_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"file_name='mobilenetv1'"。
    #          在网页上添加"file_format: 'MINDIR'"。
    #          在网页上添加"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在网页上添加"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在网页上添加其他参数。
    # （2）上传或复制训练好的模型到S3桶。
    # （3）在网页上设置代码目录为"/path/mobilenetv1"。
    # （4）在网页上设置启动文件为"export.py"。
    # （5）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （6）创建作业。
    ```

## 脚本说明

### 脚本及样例代码

```python
├── MobileNetV1
  ├── README.md              # MobileNetV1描述
  ├── scripts
  │   ├──run_distribute_train.sh        # 用于分布式训练的shell脚本
  │   ├──run_distribute_train_gpu.sh    # 用于GPU上运行分布式训练的shell脚本
  │   ├──run_standalone_train.sh        # 用于单机训练的shell脚本
  │   ├──run_standalone_train_gpu.sh    # 用于GPU上单机训练的shell脚本
  │   ├──run_eval.sh                # 用于评估的shell脚本
  ├── src
  │   ├──dataset.py                 # 创建数据集
  │   ├──lr_generator.py            # 学习率配置
  │   ├──mobilenet_v1_fpn.py        # MobileNetV1架构
  │   ├──CrossEntropySmooth.py      # 损失函数
  │   └──model_utils
  │      ├──config.py               # 处理配置参数
  │      ├──device_adapter.py       # 获取云ID
  │      ├──local_adapter.py        # 获取本地ID
  │      └──moxing_adapter.py       # 参数处理
  ├── default_config.yaml               # （使用CIFAR-10数据集）训练参数配置文件
  ├── default_config_imagenet.yaml      # （使用ImageNet数据集）训练参数配置文件
  ├── default_config_gpu_imagenet.yaml  # GPU上（使用ImageNet数据集）训练参数配置文件
  ├── train.py                      # 训练脚本
  ├── eval.py                       # 评估脚本
```

## [训练过程](#目录)

### 用法

您可以使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend：bash run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

  示例：bash run_distribute_train.sh cifar10 /root/hccl_8p_01234567_10.155.170.71.json /home/DataSet/cifar10/cifar-10-batches-bin/

  示例：bash run_distribute_train.sh imagenet2012 /root/hccl_8p_01234567_10.155.170.71.json /home/DataSet/ImageNet_Original/

- CPU：bash run_train_cpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
- GPU（单卡）：bash run_standalone_train_gpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
- GPU（分布式训练）：bash run_distribute_train_gpu.sh [cifar10|imagenet2012] [CONFIG_PATH] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）

对于使用Ascend进行分布式训练，需要提前创建JSON格式的hccl配置文件。

请按照[https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明操作。

### 启动

```shell
# 训练示例
  Python：
      Ascend：python train.py --device_target Ascend --dataset_path [TRAIN_DATASET_PATH] > log.txt 2>&1 &
      CPU：python train.py --device_target CPU --dataset_path [TRAIN_DATASET_PATH] > log.txt 2>&1 &
      GPU（单卡）：python train.py --device_target GPU --dateset [DATASET] --dataset_path [TRAIN_DATASET_PATH] --config_path [CONFIG_PATH] > log.txt 2>&1 &
      GPU（分布式训练）：
      mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
        python train.py --config_path=$2 --dataset=$1 --run_distribute=True \
        --device_num=$DEVICE_NUM --dataset_path=$PATH1 &> log.txt &

  shell：
     Ascend：bash run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
     # 示例：bash run_distribute_train.sh cifar10 ~/hccl_8p.json /home/DataSet/cifar10/cifar-10-batches-bin/
     # 示例：bash run_distribute_train.sh imagenet2012 ~/hccl_8p.json /home/DataSet/ImageNet_Original/

     CPU：bash run_train_cpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
     GPU（单卡）：bash run_standalone_train_gpu.sh [cifar10|imagenet2012] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
     GPU（分布式训练）：bash run_distribute_train_gpu.sh [cifar10|imagenet2012] [CONFIG_PATH] [DATASET_PATH] [PRETRAINED_CKPT_PATH]（可选）
```

### 结果

训练结果将存储在示例路径中。检查点默认存储在`ckpt_*`下。如使用Ascend运行训练，训练日志写入`./train_parallel*/log`。

```shell
epoch: 89 step: 1251, loss is 2.1829057
Epoch time: 146826.802, per step time: 117.368
epoch: 90 step: 1251, loss is 2.3499017
Epoch time: 150950.623, per step time: 120.664
```

训练结果将存储在示例路径中。检查点默认存储在ckpt_*下。如使用GPU运行分布式训练，训练日志写入`./train_parallel/log.txt`。

```shell
epoch: 89 step: 1251, loss is 2.44095
Epoch time: 322114.519, per step time: 257.486
epoch: 90 step: 1251, loss is 2.2521682
Epoch time: 320744.265, per step time: 256.390
```

## [评估过程](#目录)

### 用法

您可以使用Python或Shell脚本开始训练。如果是训练或微调，则不要设置`[CHECKPOINT_PATH]` shell脚本的用法如下：

- Ascend：bash run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]

  示例：bash run_eval.sh cifar10 /home/DataSet/cifar10/cifar-10-verify-bin/ /home/model/mobilenetv1/ckpt/cifar10/mobilenetv1-90_1562.ckpt 0

  示例：bash run_eval.sh imagenet2012 /home/DataSet/ImageNet_Original/ /home/model/mobilenetv1/ckpt/imagenet2012/mobilenetv1-90_625.ckpt 0

- CPU：bash run_eval_cpu.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]

### 启动

```shell
# 评估示例
  Python：
      Ascend：python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH]
      CPU：python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH] --device_target CPU
      GPU：python eval.py --dataset [cifar10|imagenet2012] --dataset_path [VAL_DATASET_PATH] --checkpoint_path [CHECKPOINT_PATH] --config_path [CONFIG_PATH] --device_target GPU

  Shell：
      Ascend：bash run_eval.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID]
      # 示例：bash run_eval.sh cifar10 /home/DataSet/cifar10/cifar-10-verify-bin/ /home/model/mobilenetv1/ckpt/cifar10/mobilenetv1-90_1562.ckpt 0
      # 示例：bash run_eval.sh imagenet2012 /home/DataSet/ImageNet_Original/ /home/model/mobilenetv1/ckpt/imagenet2012/mobilenetv1-90_625.ckpt 0

      CPU：bash run_eval_cpu.sh [cifar10|imagenet2012] [DATASET_PATH] [CHECKPOINT_PATH]
```

> 训练过程中会生成检查点。

### 结果

推理结果将存储在示例路径中，您可以在`eval/log`中查看。

```shell
Ascend
result: {'top_5_accuracy': 0.9010016025641026, 'top_1_accuracy': 0.7128004807692307} ckpt=./train_parallel0/ckpt_0/mobilenetv1-90_1251.ckpt
```

```shell
GPU
result: {'top_5_accuracy': 0.9011217948717949, 'top_1_accuracy': 0.7129206730769231} ckpt=./ckpt_1/mobilenetv1-90_1251.ckpt
```

## 推理过程

**推理前，请参考[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)设置环境变量。**

### [导出MindIR](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`EXPORT_FORMAT`取值为["AIR", "MINDIR"]。

### Ascend 310上运行推理

进行推理前，我们需要通过`export.py`脚本导出mindir文件。如下为导出MindIR模型进行推理的例子。
ImageNet 2012数据集当前的batch size只能设置为1。

```shell
# Ascend 310上运行推理
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_TYPE] [DEVICE_ID](可选)
```

- `MINDIR_PATH` 指定MindIR或AIR模型的路径。
- `DATASET_PATH` 指定cifar10/imagenet2012数据集的路径。
- `DEVICE_TYPE` 表示要运行的目标设备，可选['Ascend', 'GPU', 'CPU']。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中查看。

```bash
'top1 acc': 0.71966
'top5 acc': 0.90424
```

## 模型描述

### [性能](#目录)

#### 训练性能

| 参数                | MobileNetV1                                                     | MobileNetV1                               |
| -------------------------- | -----------------------------------------------------------------| -------------------------------------------|
| 模型版本             | V1                                                              | V1                                        |
| 资源                  | Ascend 910*4；CPU 2.60GHz, 192核；内存755G；操作系统EulerOS 2.8 | GPU NV SMX2 V100-32G                      |
| 上传日期             | 11/28/2020                                                      | 06/26/2021                                |
| MindSpore版本         | 1.0.0                                                           | 1.2.0                                      |
| 数据集                   | ImageNet2012                                                     | ImageNet2012                               |
| 训练参数       | src/config.py                                                   | default_config_gpu_imagenet.yaml          |
| 优化器                 | 动量                                                        | 动量                                  |
| 损失函数             | SoftmaxCrossEntropy                                              | SoftmaxCrossEntropy                        |
| 输出                   | 概率                                                     | 概率                               |
| 损失                      | 2.3499017                                                        | 2.2521682                                  |
| 准确率                  | ACC1[71.28%]                                                    | ACC1[71.29%]                              |
| 总时长                | 225min                                                         | --                                         |
| 参数量（M）                | 3.3M                                                           | --                                         |
| 微调检查点| 27.3M                                                          | --                                         |
| 脚本                   | [链接](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv1)

## [随机情况说明](#目录)

<!-- 在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了`train.py`中的随机种子。-->
在train.py中，我们设置了numpy.random，mindspore.common.Initializer，mindspore.ops.composite.random_ops，及mindspore.nn.probability.distribution所使用的种子。

## [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。

