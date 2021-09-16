# 目录

# 1 简述

## 1.1 ResNeXt101说明

ResNeXt是一个简单、高度模块化的图像分类网络架构。ResNeXt的设计为统一的、多分支的架构，该架构仅需设置几个超参数。此策略提供了一个新维度，我们将其称为“基数”（转换集的大小），它是深度和宽度维度之外的一个重要因素。ResNeXt模型用一种平行堆叠相同与哦噗结构的blocks代替原来ResNet的三层卷积的block，在不明显增加参数数量级的情况下提升了模型的准确率，同时由于拓扑结构相同，超参数也减少了，便于模型移植。ResNeXt有不同的网络层数，常用的有50-layer、101-layer。

本次提供的是101-layer的ResNeXt网络。

[论文](https://arxiv.org/abs/1611.05431)：  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

## 1.2 模型架构

ResNeXt整体网络架构如下：

[链接](https://arxiv.org/abs/1611.05431)

## 1.3 特性

### 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

## 1.4 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 1.5 脚本说明

**脚本及样例代码**(代码总目录resnext101)

```python
resnext101
├── ascend310_infer
│   ├── inc
│   └── src
├── infer
│   ├── convert
│   │     ├── aipp.config  #转换OM文件的aipp配置文件
│   │     └── convert_om.sh
│   ├── data
│   │     ├── images  #推理相关数据保存
│   │     └── models  #存放om模型
│   ├── mxbase
│   └── sdk
│   │     ├── classification_task_metric.py  
│   │     ├── main.py
│   │     ├── resnext101.pipeline
│   │     ├── run.sh
├── modelarts
│     └── start_train.py    #modelarts上的启动文件
├── scripts
│   ├── docker_start.sh
│   ├── run_distribute_train.sh #启动Ascend分布式训练(8卡)
│   ├── run_distribute_train_for_gpu.sh       #启动GPU分布式训练（8卡）
│   ├── run_eval.sh                            #启动Ascend评估
│   ├── run_infer_310.sh
│   ├── run_standalone_train.sh               #启动Ascend单机训练（单卡）
│   └── run_standalone_train_for_gpu.sh       #启动GPU单机训练（单卡）
├── src
│   ├── _pycache_
│   ├── backbone
│   │     ├── _pycache_
│   │     ├── _init_.py     #初始化
│   │     └── resnet.py     # ResNeXt骨干网络,包括ResNet、ResNeXt50和ResNeXt101
│   ├── model_utils
│   │     ├── _pycache_
│   │     ├── config.py          #参数配置
│   │     ├── device_adapter.py  #设备配置
│   │     ├── local_adapter.py   #本地设备配置
│   │     └── moxing_adapter.py  #modelarts设备配置
│   ├── utils
│   │     ├── init.py   #初始化
│   │     ├── auto_mixed_precision.py #混合精度
│   │     ├── cunstom_op.py           #网络操作
│   │     ├── logging.py              #打印日志
│   │     ├── optimizers_init_.py     #获取参数
│   │     ├── sampler.py              #分布式采样器
│   │     └── var_init.py             #计算增益值
│   ├── _init_
│   ├── crossentropy.py           # ImageNet2012数据集的损失定义
│   ├── dataset.py                # 数据预处理
│   ├── head.py                   # 头文件
│   ├── image_classification.py   # 图像分类网络架构，包括ResNext50和ResNeXt101
│   └──lr_generator.py  # 生成每个步骤的学习率
├── creat_imagenet2012_label.py
├── default_config.yaml #参数配置文件
├── Dockerfile
├── eval.py    #测试精度脚本入口
├── export.py  #AIR导出脚本
├── mindspore_hub_conf.py  #MindSpore
├── postprocess.py  #310推理后处理脚本
├── README.md
├── README_CN
├── requirements.txt
└── train.py                            # 训练模型代码

```

# 2 训练

## 2.1 数据集

使用的数据集：[ImageNet](http://www.image-net.org/)

  (1)数据集大小：约125G, 共1000个类，包含120万张彩色图像
   - 训练集：120G，120万张图像
   - 测试集：5G，5万张图像

  (2)数据格式：RGB图像。
   - 注：数据在src/dataset.py中处理

## 2.2 训练过程

2.2.1**将源码上传至训练服务器任意目录（如：“/home/HwHiAiUser”）。**

```shell
# 在环境上执行
cd /home/HwHiAiUser/resnext101
```

2.2.2编译镜像

```shell
docker build -t docker_image --build-arg FROM_IMAGE_NAME=base_image:tag .
```

参数说明：

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| docker_image | 镜像名称，请根据实际写入。##自己定义的一个新的镜像的名字     |
| base_image   | 基础镜像，可从Ascend hub上下载。https://ascendhub.huawei.com/#/detail/ |
| tag          | 镜像的版本，请根据实际配置，如：21.0.2                       |

a. 首先我们要先看自己的裸机是x86架构还是arm架构,

```shell
uname -a
```

b.我们根据上面的选择下载基础镜像的版本

命令行实例

```shell
docker build -t resnext101ms --build-arg FROM_IMAGE_NAME=ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:21.0.2 .
```

参数说明：

最后的"."别忘记

2.2.3启动容器

命令如下

```shell
bash scripts/docker_start.sh docker_image data_dir model_dir
```

命令行实例

```shell
bash scripts/docker_start.sh resnext101ms infer/data/images/imagenet /home/HwHiAiUser/resnext101/
```

2.2.4训练

用法：

```shell
Ascend:
    # 分布式训练示例（8卡）
    bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # 单机训练
    bash run_standalone_train.sh DEVICE_ID DATA_PATH

```

参数说明：

RANK_TABLE_FILE的获取：

 (1)通过[hccl_tools](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools)找到hccl_tools.py,将其复制在/root/RMX/resnext101/src中

 (2)命令行如下（在裸机上）：

```shell
cd /home/HwHiAiUser/resnext101/src
python hccl_tools.py --device_num "[0,8)"
```

经过这一步我们得到hccl_8p_01234567_182.138.104.162.json文件

```shell
# Ascend分布式训练示例（在容器中，8卡）
bash scripts/run_distribute_train.sh /home/HwHiAiUser/resnext101/src/hccl_8p_01234567_127.0.0.1.json infer/data/images/imagenet/train
# Ascend单机训练示例/
bash scripts/run_standalone_train.sh 0 infer/data/images/imagenet/train
```

训练结果如下所示（8卡训练结果部分展示）：

```shell
2021-08-16 04:46:32,538:INFO:epoch[138], iter[1391250], loss:2.0361586, mean_fps:746.92imgs/sec
2021-08-16 05:15:12,908:INFO:epoch[139], iter[1401259], loss:2.1895633, mean_fps:744.70imgs/sec
2021-08-16 05:43:47,957:INFO:epoch[140], iter[1411268], loss:1.9028324, mean_fps:747.01imgs/sec
2021-08-16 06:12:23,129:INFO:epoch[141], iter[1421277], loss:1.7602392, mean_fps:746.95imgs/sec
2021-08-16 06:40:58,388:INFO:epoch[142], iter[1431286], loss:1.9320352, mean_fps:746.92imgs/sec
2021-08-16 07:09:33,538:INFO:epoch[143], iter[1441295], loss:1.8037808, mean_fps:746.96imgs/sec
2021-08-16 07:38:13,170:INFO:epoch[144], iter[1451304], loss:1.9797318, mean_fps:745.02imgs/sec
2021-08-16 08:06:48,281:INFO:epoch[145], iter[1461313], loss:1.9492955, mean_fps:746.98imgs/sec
2021-08-16 08:35:23,370:INFO:epoch[146], iter[1471322], loss:1.8807195, mean_fps:746.99imgs/sec
2021-08-16 09:03:58,498:INFO:epoch[147], iter[1481331], loss:1.7080936, mean_fps:746.97imgs/sec
2021-08-16 09:32:33,580:INFO:epoch[148], iter[1491340], loss:1.7306125, mean_fps:746.99imgs/sec
2021-08-16 10:01:12,860:INFO:epoch[149], iter[1501349], loss:1.825752, mean_fps:745.17imgs/sec
2021-08-16 10:01:13,192:INFO:end network train...
```

2.2.5开始测试

您可以通过Shell脚本训练：

```shell
##命令行
bash scripts/run_eval.sh DEVICE_ID DATA_PATH CHECKPOINT_FILE_PATH PLATFORM
```

```shell
##样例
bash scripts/run_eval.sh 0 infer/data/images/imagenet/val /home/HwHiAiUser/resnext101/output/2021-08-02_time_02_13_45/ckpt_0/0-100_10009.ckpt Ascend
```

测试结果示例如下：

```shell
2021-08-02 01:38:49,282:INFO:load model /home/HwHiAiUser/resnext101/LOG0/output/2021-07-31_time_08_27_42/ckpt_0/0-150_1251.ckpt success
2021-08-02 01:41:01,396:INFO:Inference Performance: 799.19 img/sec
2021-08-02 01:41:01,397:INFO:before results=[[39723], [47211], [49920]]
2021-08-02 01:41:01,397:INFO:after results=[[39723]
 [47211]
 [49920]]
2021-08-02 01:41:01,398:INFO:after allreduce eval: top1_correct=39723, tot=49920,acc=79.57%(TOP1)
2021-08-02 01:41:01,398:INFO:after allreduce eval: top5_correct=47211, tot=49920,acc=94.57%(TOP5)
```

2. 2.6转换AIR模型

用法

```shell
python export.py --device_target [PLATFORM] --checkpoint_file_path [CKPT_PATH] --filename [FILE_NAME] --file_format [EXPORT_FORMAT] --config_path [CONFIG_PATH]
```

示例

```shell
python export.py --device_target Ascend --checkpoint_file_path /home/HwHiAiUser/resnext101/output/2021-07-31_time_08_27_42/ckpt_0/0-150_1251.ckpt --file_format AIR
```

结果生成resnext101.air的文件

## 2.3 迁移学习指导

2.3.1 数据集准备。

数据集收集要求如下：

  (1) 获取数据。
  如果要使用自己的数据集，需要将数据集放到对应目录下。数据集的存放位置可如下示例（以cifar10为例）：
       训练集：/dataset/cifar10/train
       验证集：/dataset/cifar10/val
 训练数据集和验证数据集以文件名中的train和val加以区分。
该数据集的训练过程脚本只作为一种参考示例。

```python
数据集文件结构。
请用户自行准备好图片数据集，包含训练集和验证集两部分（训练集10个类别，验证集10个类别），目录参考：
├── cifar10
│    ├──train      ##共有10个类别，目录从0到9
│    │    ├──0
│    │          ├──xxx.jpeg
│    │          ├──xxx.jpeg
│    │          ├──xxx.jpeg
│    │    ├──1
│    │    ├──2
│    ├──val       ##共有10个类别，目录从0到9
│    │    ├──0
│    │          ├──xxx.jpeg
│    │          ├──xxx.jpeg
│    │          ├──xxx.jpeg
│    │    ├──1
│    │    ├──2
```

2.3.2 加载预训练模型并修改训练脚本参数。

加载预训练模型，train.py中加载预训练模型的代码如下所示。

```python
# network
config.logger.important_info('start create network')
# get network and init
network = get_network(network=config.network,  num_classes=config.num_classes, platform=config.device_target)
load_pretrain_model(config.checkpoint_file_path, network, config)##加载预训练模型

```

通过设置default_config.ymal中修改checkpoint_file_path参数，该参数的值设置为训练好的模型ckpt文件的路径，实现从预训练模型中加载权重。若使用自有数据集训练时，类别个数不等于1000，需同时设置“class_nums”和filter_weight。

例如使用名为“resnet-90_5004.ckpt”的预训练模型进行迁移学习，自有数据集类别数为10，则首先需要修改default_config.ymal文件类别数。

default_config.yaml中的修改如下：

```python
device_target: "Ascend"
checkpoint_file_path: "xx/xxxxxx.ckpt"  ##迁移学习预训练模型.ckpt文件
Training optionsimage_size: [224,224]
num_classes: 10   ##迁移学习类别数
batch_size: 1
lr: 0.4
lr_scheduler: "cosine_annealing"
lr_epochs: [30,60,90,120]
lr_gamma: 0.1
eta_min: 0
T_max: 150
max_epoch: 150
warmup_epochs: 1
weight_decay: 0.0001
momentum: 0.9
is_dynamic_loss_scale: 0
loss_scale: 1024
label_smooth: 1
label_smooth_factor: 0.1
per_batch_size: 128
ckpt_interval: 5
ckpt_save_max: 5
is_save_on_master: 1
filter_weight: true ###迁移学习时这个参数设置为true，其他情况时这个参数为False
rank_save_ckpt_flag: 0
outputs_dir: ""
log_path: "./output_log"
```

2.3.3 迁移学习训练如下：

```shell
Ascend:
    # Ascend单机迁移学习示例
    bash run_distribute_train.sh [DEVICE_ID] [DATA_PATH] [CHECKPOINT_FILE_PATH]
    # Ascend分布式迁移学习示例（8卡）
    bash run_standalone_train.sh [RANK_TABLE_FILE] [DATA_PATH] [CHECKPOINT_FILE_PATH]
```

参数说明:

| 参数                   | 说明                                           |
| ---------------------- | ---------------------------------------------- |
| [DEVICE_ID]            | 设备ID。如0、1、2、3、4、5、6、7               |
| [DATA_PATH]            | 迁移学习训练集路径，如：/dataset/cifar10/train |
| [CHECKPOINT_FILE_PATH] | 预训练模型的ckpt文件路径。                     |

迁移学习示例如下：

```shell
# Ascend单机迁移学习示例
bash scripts/run_standalone_train.sh 0 /dataset/cifar10/train LOG0/output/2021-07-31_time_08_27_42/ckpt_0/0-150_1251.ckpt
# Ascend分布式迁移学习示例（8卡）
bash scripts/run_distribute_train.sh /home/HwHiAiUser/resnext101/src/hccl_8p_01234567_127.0.0.1.json
LOG0/output/2021-07-31_time_08_27_42/ckpt_0/0-150_1251.ckpt
```

2.3.4 模型训练。
     请参考“模型训练”。

# 3 推理

## 3.1 准备容器环境

1. 推理数据准备

（1）获取数据。

   请用户自行准备好数据集，例如我们以imagenet数据集为例。
   数据集的存储目录格式如下。

```python
|---- train
      |----xxxxxx.jpeg
|---- train_label.txt
|---- train_label.txt
|---- val
      |----xxxxxx.jpeg
|---- val_label.txt
```

2. 将软件包上传至推理服务器任意目录（如”/root/REE“）

   ```shell
   ##在环境上执行
   unzip resnext101.zip
   cd root/REE/resnext101/infer
   ```

3. 启动容器

   进入到infer目录中，启动容器

   ```shell
   cd root/REE/resnext101/infer
   bash docker_start_infer.sh infer_image model_dir data_dir
   ```

   参数说明：

   | 参数        | 说明                                        |
   | ----------- | ------------------------------------------- |
   | infer_image | 镜像的实际名称,如：resnext101msarm_infer    |
   | model_dir   | 推理模型的路径，如：/root/RMX/resnext/      |
   | data_path   | 数据路径，如：“/data/imagenet_val/imagenet" |

   命令行实例

   ```shell
   bash infer/docker_start_infer.sh infer_image model_dir /data/imagnet_val
   ```

   说明：MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：”/usr/local/sdk_home“。

## 3.2 模型转换

进入到推理容器环境，具体操作见”准备容器环境“

操作步骤：

1. AIR模型为在昇腾910服务器上导出的模型，导出AIR模型的详细步骤请参考“模型训练过程中”的最后一步--“将.ckpt转换为.air”

2. 准备AIPP配置文件。

AIPP需要配置aipp.config文件，在ATC转换的过程中插入AIPP算子，即可与DVPP处理后的数据无缝对接，AIPP参数配置请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中“ATC工具使用指南”。

aipp.config

```shell
aipp_op {
    aipp_mode: static
    input_format : RGB888_U8

    rbuv_swap_switch : true

    mean_chn_0 : 0
    mean_chn_1 : 0
    mean_chn_2 : 0
    min_chn_0 : 123.675
    min_chn_1 : 116.28
    min_chn_2 : 103.53
    var_reci_chn_0 : 0.0171247538316637
    var_reci_chn_1 : 0.0175070028011204
    var_reci_chn_2 : 0.0174291938997821
}
```

将.air模型转换为.om模型的命令行如下：

```shell
bash convert/convert_om.sh air_path aipp_cfg_path om_path
```

参数说明：

| 参数          | 说明                                            |
| ------------- | ----------------------------------------------- |
| air_path      | .air模型文件路径                                |
| aipp_cfg_path | aipp配置文件路径                                |
| om_path       | 生成的OM文件名，转换脚本会在此基础上添加.om后缀 |

convert_om.sh脚本内容

```shell
#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 INPUT_AIR_PATH OUTPUT_OM_PATH_NAME"
  echo "Example: "
  echo "         bash convert_om.sh ./models/ssd-500_458_on_coco.air ./models/ssd-500_458_on_coco"

  exit 255
fi

input_air_path=$1
output_om_path=$2
aipp_cfg=$3

export ASCEND_ATC_PATH=/usr/local/Ascend/atc/bin/atc
export LD_LIBRARY_PATH=/usr/local/Ascend/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/atc/python/site-packages:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/atc/python/site-packages/schedule_search.egg
export ASCEND_OPP_PATH=/usr/local/Ascend/opp

export ASCEND_SLOG_PRINT_TO_STDOUT=1

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"
echo "AIPP cfg file path: ${aipp_cfg}"

atc --input_format=NCHW \
--framework=1 \
--model=${input_air_path} \
--output=${output_om_path} \
--soc_version=Ascend310 \
--disable_reuse_memory=0 \
--output_type=FP32\
--insert_op_conf=${aipp_cfg} \
--precision_mode=allow_fp32_to_fp16  \
--op_select_implmode=high_precision
```

---结束

## 3.3 sdk推理

**前提条件**：进入推理容器环境

### 推理过程

步骤1. 准备模型推理文件

  (1)、 将“infer/sdk”目录下的所有文件都拷贝到“/usr/local/sdk_home/mxManufacture-2.0.2/samples/mxManufacture/python”目录。同名文件会被直接覆盖。进入到该目录下，包含如下四个文件：

```python
classification_task_metric.py  ##计算精度的文件
main.py
resnext101.pipeline
run.sh ##运行推理服务脚本文件
```

  (2)、  进入/usr/local/sdk_home/mxManufacture/samples/mxManufacture/python/pipeline”目录。

   “modelPath”、“postProcessConfigPath”、"labelPath"和"postProcessLibPath"这三个参数需要修改

```shell
{
    "im_resnet50": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc0": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_imagedecoder0"
        },
        "mxpi_imagedecoder0": {
            "props": {
                "handleMethod": "opencv"
            },
            "factory": "mxpi_imagedecoder",
            "next": "mxpi_imageresize0"
        },
        "mxpi_imageresize0": {
            "props": {
                "handleMethod": "opencv",
                "resizeType": "Resizer_Stretch",
                "resizeHeight": "224",
                "resizeWidth": "224"
            },
            "factory": "mxpi_imageresize",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "modelPath": "../../../../../data/models/0-150_1251.om",##路径根据实际修改
                "waitingTime": "2000",
                "outputDeviceId": "-1"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_classpostprocessor0"
        },
        "mxpi_classpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "./resnet101.cfg", ##路径根据实际修改
                "labelPath": "./imagenet1000_clsidx_to_labels.names",##路径根据实际修改
                "postProcessLibPath": "../../../lib/modelpostprocessors/libresnet50postprocess.so"##路径根据实际修改
            },
            "factory": "mxpi_classpostprocessor",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_classpostprocessor0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "appsink0": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
    }
}

```

参数说明：

参数说明：

- resizeHeight：模型高度，请根据模型的实际尺寸输入。
- resizeWidth：模型宽度，请根据模型的实际尺寸输入。
- modelPath：模型路径，请根据模型实际路径修改。
- postProcessLibPath：后处理插件.so的路径

  (3)、 进入“mxManufacture/samples/mxManufacture/python”目录。根据实际情况修改main.py文件中的 **pipeline**、 **dir_name**和 ***res_dir_name***文件路径。

```shell
# import StreamManagerApi.py
from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./resnext101.pipeline", 'rb') as f:##根据实际情况修改
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()

    dir_name = sys.argv[1]
    res_dir_name = sys.argv[2]
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
```

步骤2. 运行推理服务。

进入“mxManufacture/samples/mxManufacture/python”目录，执行推理命令。

```shell
bash run.sh /data/imagenet/val/ resnext101_8p_result
```

参数说明：

- 参数1：验证集路径。注意在传参时验证集目录后需要加“/”。
- 参数2：推理结果保存路径。

![image-20210804191648475](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210804191648475.png)

步骤3. 性能统计。

  (1). 打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。

  mxManufacture-2.0.1/config/sdk.conf

![image-20210804191839865](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210804191839865.png)

  (2). 执行run.sh脚本。

```shell
bash run.sh /data/imagenet/val/ resnext101_8p_result
```

  a. 在日志目录“/usr/local/sdk_home/mxManufacture/logs/”查看性能统计结果。

  b.进入“/mxManufacture/samples/mxManufacture/python”目录。

  (3). 将软件包中classification_task_metric.py文件拷贝到该目录下。
  (4). 执行命令计算推理精度。

```shell
python3.7 classification_task_metric.py resnext101_8p_result/ /data/val_lable.txt ./ ./resnext101_8p_result.json
```

参数说明：

- 第一个参数（resnext101_8p_result/）：推理结果保存路径。
- 第二个参数（./val_label.txt）：验证集标签文件。
- 第三个参数（./）：精度结果保存目录。
- 第四个参数（./result.json）：结果文件。

步骤4. 查看推理精度结果。

```shell
cat result.json
```

result.json示例如下：

```shell
{
  "title": "Overall statistical evaluation",
  "value": [
    {
      "key": "Number of images",
      "value": "50000"
    },
    {
      "key": "Number of classes",
      "value": "5"
    },
    {
      "key": "Top1 accuracy",
      "value": "79.57%"
    },
    {
      "key": "Top2 accuracy",
      "value": "89.02%"
    },
    {
      "key": "Top3 accuracy",
      "value": "92.13%"
    },
    {
      "key": "Top4 accuracy",
      "value": "93.61%"
    },
    {
      "key": "Top5 accuracy",
      "value": "94.58%"
    }
  ]
}
```

## 3.4 mxBase推理

### 推理过程

mxBase是MindX SDK中的一个组件，其头文件和SO均集成在mxVison的run包中。mxBase提供的API可查看mxVison的开发资料和头文件，本章将通过链接SO的方式进行编译。

操作步骤：

步骤1. 将”resnext101/infer/mxbase“目录中的所有文件拷贝到”/usr/local/sdk_home/mxManufacture-2.0.2/samples/mxManufacture/C++“目录，进入该目录，修改main_opencv.cpp文件中的配置文件路径。

   ```shell
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../models/imagenet1000_clsidx_to_labels.names"; ##根据实际存放位置进行修改
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../models/resnext101/resnext101_8p.om";##根据实际存放位置进行修改
    auto resnext101 = std::make_shared<Resnext101ClassifyOpencv>();
    APP_ERROR ret = resnext101->Init(initParam);
    if (ret != APP_ERR_OK) {
           LogError << "Resnext101Classify init failed, ret=" << ret << ".";
           return ret;
     }
   ```

步骤2.编译工程

   ```shell
    source ~/.bashrc
    doc2unix 'find . *.sh'
    bash build.sh
   ```

步骤3.运行推理服务。

   (1). 建立保存结果的文件夹“result”

```shell
  mkdir result
```

   (2). 运行推理程序脚本

```shell
 bash run.sh
```

  run.sh中的内容

```shell

  export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/lib/modelpostprocessors:
  ${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/
  Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
  echo $LD_LIBRARY_PATH
  ./resnext101  /data/imagenet_val/
```

 参数说明：image_path 推理图片路径，如“../data/images/imagenet/test.jpg”

   (3). 在result中查看预测结果

----结束

# 4 在Modelarts上的应用

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和模型冻结，具体操作如下：

- 任务类型: 图像分类

- 支持的框架引擎: Ascend-Powered-Engine-Mindspore-1.2.0-python3.7-aarch64

- 算法输入: obs数据集路径

- 算法输出: 训练生成的ckpt模型

## 4.1 数据集准备

下载cifar10数据集用于modelarts上的模型训练，数据集目录为

```shell
|-----train  #共10个类别
      |----0
           |----xxx.jpeg
           |----xxx.jpeg
           ......
           |----1
           |----2
           |----3
           .....
|-----val    #共10个类别
      |----0
      |----1
      |----2
      |----3
      .....
```

## 4.2 训练过程

4.2.1 先创建一个obs桶，如：”resnext101“

![image-20210823111026181](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210823111026181.png)

4.2.2 创建成功后，打开你创建的obs桶，创建如下所示四个文件夹

![image-20210823111312698](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210823111312698.png)

```python
|-----code#存放ResNext101所有的代码，将模型的代码全部上传到这个文件夹
|-----dataset   #存放数据集
|-----logs      #用于存放训练日志
|-----output    #用于存放训练模型.ckpt、冻结模型.air等
```

说明：

上传代码前先把default_config.yaml中的enable_modelarts参数设置为true，这个参数设置为true，表示可以在modelarts上应用开发，将启动文件从"code/resnext101/modelarts"目录下拷贝到"code"目录下。

4.2.3 打开ModelArts管理控制平台

  (1)、 ”算法管理“ -->"我的算法"-->创建算法

![image-20210823112512400](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210823112512400.png)

参数说明

| 参数     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 名称     | 你创建的算法的名称，可自定义                                 |
| AI引擎   | 选择Ascend-Powered-Engine-Mindspore-1.3.0-python3.7-aarch64  |
| 代码目录 | 选择你的代码所在的目录                                       |
| 启动文件 | 选择modelarts目录下的"train_start.py"文件                    |
| 超参     | 可以在modelarts界面上设置超参数，这些超参数的名称和类型必须与你模型中使用到的参数保持一致，也即与default_config.yaml中的参数名称保持一致 |

  (2)、 创建训练作业

”训练管理“-->”训练作业“--创建训练作业

![image-20210823114942949](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210823114942949.png)

创建训练作业名称，在我的算法中选择自己创建的算法，填写下面界面中对应的参数

![image-20210823115050328](C:\Users\任美香\AppData\Roaming\Typora\typora-user-images\image-20210823115050328.png)

参数说明

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| 名称         | 训练作业的名称，也可自定义                                   |
| 训练输入     | 点击”选择数据存储位置“按钮从你创建的obs桶中选择模型的训练数据 |
| 训练输出     | 选择存放模型输出的目录，如obs桶中的output目录                |

  (3)、单击“提交”，完成训练作业的创建。
   训练作业一般都需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

  (4)、输出文件夹目录下可以看到模型格式.ckpt模型文件和.air模型文件

 ----结束

## 4.3 查看训练日志

  (1)、在Modelarts管理控制台，在左侧导航栏中选择“训练管理->训练作业（New）",默认进入”训练作业“列表。
  (2)、在训练作业列表中，您可以单击作业名称，查看该作业的详情。
  详情中包含作业的基本信息、训练参数、日志详情和资源占用情况。

## 4.4 迁移学习

  (1)、数据集准备
  请参见“训练 > 迁移学习指导”，准备训练所需数据集，将其和标签文件上传至对应OBS桶中。

  (2)、在modelarts界面上设置参数如下：

```python
checkpoint_file_path=/cache/checkpoint_path/xxx.ckpt
checkpoint_url设置为在obs中ckpt文件的目录   如：obs://resnext101ms-rmx/output/2021-08-16_time_11_30_14/ckpt_0/
在modelarts的界面上设置代码的路径 "/path/ResNeXt"。
在modelarts的界面上设置模型的启动文件 "train.py" 。
在modelarts的界面上设置模型的数据路径 "Dataset path" ,
模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
开始模型的迁移学习。
```

  (3)、创建训练作业，进行迁移学习。
  请参考“创建训练作业”章节。

----结束

