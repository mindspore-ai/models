# 目录

- [目录](#目录)
    - [EGNet描述](#egnet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
        - [数据集预处理](#数据集预处理)
    - [预训练模型](#预训练模型)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
                - [DUTS-TR上的EGNet(Ascend)](#duts-tr上的egnetascend)
                - [DUTS-TR上的EGNet(GPU)](#duts-tr上的egnetgpu)
            - [推理性能](#推理性能)
                - [显著性检测数据集上的EGNet(Ascend)](#显著性检测数据集上的egnetascend)
                - [显著性检测数据集上的EGNet(GPU)](#显著性检测数据集上的egnetgpu)
    - [ModelZoo主页](#modelzoo主页)

## EGNet描述

EGNet是用来解决静态目标检测问题，它由边缘特征提取部分、显著性目标特征提取部分以及一对一的导向模块三部分构成，利用边缘特征帮助显著性目标特征定位目标，使目标的边界更加准确。在6个不同的数据集中与15种目前最好的方法进行对比，实验结果表明EGNet性能最优。

[EGNet的pytorch源码](https://github.com/JXingZhao/EGNet)，由论文作者提供。具体包含运行文件、模型文件和数据处理文件，此外还带有数据集、初始化模型和预训练模型的获取途径，可用于直接训练以及测试。

[论文](https://arxiv.org/abs/1908.08297): Zhao J X, Liu J J, Fan D P, et al. EGNet: Edge guidance network for salient object detection[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 8779-8788.

## 模型架构

EGNet网络由三个部分组成，NLSEM（边缘提取模块）、PSFEM（目标特征提取模块）、O2OGM（一对一指导模块），原始图片通过两次卷积输出图片边缘信息，与此同时，对原始图像进行更深层次的卷积操作提取salient object，然后将边缘信息与不同深度提取出来的显著目标在一对一指导模块中分别FF（融合），再分别经过卷积操作得到不同程度的显著性图像，最终输出了一张融合后的显著性检测图像。

## 数据集

数据集统一放在一个目录，下面的文件夹以此为基础创建。

- 训练集：[DUTS-TR数据集](http://saliencydetection.net/duts/download/DUTS-TR.zip)，210MB，共10533张最大边长为400像素的彩色图像，均从ImageNet DET训练/验证集中收集。

创建名为“DUTS-TR”的文件夹，根据以上链接下载数据集放入文件夹，并解压到当前路径。

```bash
├──DUTS-TR
    ├──DUTS-TR-Image
    ├──DUTS-TR-Mask
```

- 测试集：[DUTS-TE数据集](http://saliencydetection.net/duts/download/DUTS-TE.zip)，32.3MB，共5019张最大边长为400像素的彩色图像，均从ImageNet DET测试集和SUN数据集中收集。

创建名为“DUTS-TE”的文件夹，根据以上链接下载数据集放入文件夹，并解压到当前路径。

```bash
├──DUTS-TE
    ├──DUTS-TE-Image
    ├──DUTS-TE-Mask
```

- 测试集：[SOD数据集](https://www.elderlab.yorku.ca/?smd_process_download=1&download_id=8285)，21.2MB，共300张最大边长为400像素的彩色图像，此数据集是基于Berkeley Segmentation Dataset（BSD）的显著对象边界的集合。

创建名为“SOD”的文件夹，根据以上链接下载数据集放入文件夹，并解压到当前路径。

```bash
├──SOD
    ├──Imgs
```

- 测试集：[ECSSD数据集](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip，http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip)，64.6MB，共1000张最大边长为400像素的彩色图像。

创建名为“ECSSD”的文件夹，根据以上链接下载数据集的原图及groundtruth放入文件夹，并解压到当前路径。

```bash
├──ECSSD
    ├──ground_truth_mask
    ├──images
```

- 测试集：[PASCAL-S数据集](https://academictorrents.com/download/6c49defd6f0e417c039637475cde638d1363037e.torrent)，175 MB，共10个类、850张32*32彩色图像。该数据集与其他显著物体检测数据集区别较大, 没有非常明显的显著物体, 并主要根据人类的眼动进行标注数据集, 因此该数据集难度较大。

根据以上链接下载数据集并解压到当前路径。在数据集根目录创建名为“PASCAL-S”以及Imgs的文件夹，将datasets/imgs/pascal和datasets/masks/pascal放入到Imgs文件夹中。

```bash
├──PASCAL-S
    ├──Imgs
```

- 测试集：[DUTS-OMRON数据集](http://saliencydetection.net/dut-omron/)，107 MB，共5168张最大边长为400像素的彩色图像。数据集中具有一个或多个显著对象和相对复杂的背景，具有眼睛固定、边界框和像素方面的大规模真实标注的数据集.

创建名为“DUTS-OMRON-image”的文件夹，根据以上链接下载数据集放入文件夹，并解压到当前路径。

```bash
├──DUTS-OMRON-image
    ├──DUTS-OMRON-image
    ├──pixelwiseGT-new-PNG
```

- 测试集：[HKU-IS数据集](https://i.cs.hku.hk/~gbli/deep_saliency.html)，893MB，共4447张最大边长为400像素的彩色图像。数据集中每张图像至少满足以下的3个标准之一:1)含有多个分散的显著物体; 2)至少有1个显著物体在图像边界; 3)显著物体与背景表观相似。

创建名为“HKU-IS”的文件夹，根据以上链接下载数据集放入文件夹，并解压到当前路径。

```bash
├──HKU-IS
    ├──imgs
    ├──gt
```

### 数据集预处理

运行dataset_preprocess.sh脚本，对数据集进行了格式统一，裁剪以及生成对应的lst文件。其中测试集生成test.lst，训练集生成test.lst和train_pair_edge.lst。

```shell
# DATA_ROOT 所有数据集存放的根目录
# OUTPUT_ROOT 结果目录
bash dataset_preprocess.sh [DATA_ROOT] [OUTPUT_ROOT]
```

1. 处理后的DUTS-TR数据集目录如下。DUTS-TE-Mask存放groundtruth，DUTS-TE-Image存放原图，test.lst是数据中的图片文件列表，train_pair_edge.lst是记录数据集中图片、groundtruth和边缘图的文件列表。

```bash
├──DUTS-TR
    ├──DUTS-TR-Image
    ├──DUTS-TR-Mask
    ├──test.lst
    ├──train_pair_edge.lst
```

2. 处理后的DUTS-TE数据集目录如下。DUTS-TE-Mask存放groundtruth，DUTS-TE-Image存放原图，test.lst是数据中的图片文件列表。

```bash
├──DUTS-TE
    ├──DUTS-TE-Image
    ├──DUTS-TE-Mask
    ├──test.lst
```

3. 处理后的除DUTS-TE的5个测试集统一成如下格式(以HKU-IS为例)，ground_truth_mask存放groundtruth，images存放原图，test.lst是数据中的图片文件列表。。

```bash
├──HKU-IS
    ├──ground_truth_mask
    ├──images
    ├──test.lst
```

4. test.lst是数据中的图片文件列表，train_pair_edge.lst是包含图片、groundtruth和边缘图的文件列表。

```bash
test.lst文件格式如下(以HKU-IS为例)

    0004.png
    0005.png
    0006.png
    ....
    9056.png
    9057.png
```

```bash
train_pair_edge.lst文件格式如下(DUTS-TR)

    DUTS-TR-Image/ILSVRC2012_test_00007606.jpg DUTS-TR-Mask/ILSVRC2012_test_00007606.png DUTS-TR-Mask/ILSVRC2012_test_00007606_edge.png
    DUTS-TR-Image/n03770439_12912.jpg DUTS-TR-Mask/n03770439_12912.png DUTS-TR-Mask/n03770439_12912_edge.png
    DUTS-TR-Image/ILSVRC2012_test_00062061.jpg DUTS-TR-Mask/ILSVRC2012_test_00062061.png DUTS-TR-Mask/ILSVRC2012_test_00062061_edge.png
    ....
    DUTS-TR-Image/n02398521_31039.jpg DUTS-TR-Mask/n02398521_31039.png DUTS-TR-Mask/n02398521_31039_edge.png
    DUTS-TR-Image/n07768694_14708.jpg DUTS-TR-Mask/n07768694_14708.png DUTS-TR-Mask/n07768694_14708_edge.png
```

## 预训练模型

pytorch预训练模型（包括vgg16, resnet50)

VGG主干网络选用vgg16的结构，包含13个卷积层和3个全连接层，本模型不使用全连接层。

下载 [VGG16预训练模型](https://download.mindspore.cn/thirdparty/vgg16_20M.pth)

ResNet主干网络选用resnet50的结构，包含卷积层和全连接层在内共有50层，本模型不使用全连接层。整体由5个Stage组成，第一个Stage对输出进行预处理，后四个Stage分别包含3,4,6,3个Bottleneck。

下载 [ResNet50预训练模型](https://download.mindspore.cn/thirdparty/resnet50_caffe.pth)

mindspore预训练模型

下载pytorch预训练模型，再运行如下脚本，得到对应的mindspore模型。注：运行该脚本需要同时安装pytorch环境(测试版本号为1.3，CPU 或 GPU)

```bash
# MODEL_NAME: 模型名称vgg或resnet
# PTH_FILE: 待转换模型文件绝对路径
# MSP_FILE: 输出模型文件绝对路径
bash convert_model.sh [MODEL_NAME] [PTH_FILE] [MSP_FILE]
```

## 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend/GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

在default_config.yaml进行相关配置，其中train_path项设置训练集存放路径，base_model项设置主干网络类型（vgg或resnet)，test_path项设置测试集存放路径，vgg和resnet项设置预训练模型存放路径。scripts中的脚本文件里也可传递参数，且可以覆盖掉default_config.yaml中设置的参数。注：所有脚本运行需先进入scripts目录。

- Ascend处理器环境运行

```shell

# 运行训练示例
bash run_standalone_train.sh

# 运行分布式训练示例
bash run_distribute_train.sh 8 [RANK_TABLE_FILE]

# 运行评估示例
bash run_eval.sh
```

- GPU处理器环境运行

```shell

# 运行训练示例
bash run_standalone_train_gpu.sh

# 运行分布式训练示例
# DEVICE_NUM: 使用的显卡数量，如: 8
# USED_DEVICES: 使用的显卡id列表，需和显卡数量对应，如: 0,1,2,3,4,5,6,7
bash run_distribute_train_gpu.sh [DEVICE_NUM] [USED_DEVICES]

# 运行评估示例
bash run_eval_gpu.sh
```

## 脚本说明

### 脚本及样例代码

```bash
├── model_zoo
    ├── EGNet
        ├── README_CN.md                    # EGNet中文README文件
        ├── model_utils                     # config，modelarts等配置脚本文件夹
        │   ├──config.py                    # 解析参数配置文件
        ├── scripts
        │   ├──run_standalone_train.sh      # 启动Ascend单机训练（单卡）
        │   ├──run_distribute_train.sh      # 启动Ascend分布式训练（8卡）
        │   ├──run_eval.sh                  # 启动Ascend评估
        │   ├──run_standalone_train_gpu.sh  # 启动GPU单机训练（单卡）
        │   ├──run_distribute_train_gpu.sh  # 启动GPU分布式训练（多卡）
        │   ├──run_eval_gpu.sh              # 启动GPU评估
        │   ├──run_infer_310.sh             # 启动Ascend310评估
        │   ├──dataset_preprocess.sh        # 对数据集预处理并生成lst文件
        │   ├──convert_model.sh             # 转换预训练模型
        ├── ascend310_infer
        ├── src
        │   ├──dataset.py                   # 加载数据集
        │   ├──egnet.py                     # EGNet的网络结构
        │   ├──vgg.py                       # vgg的网络结构，vgg16版本
        │   ├──resnet.py                    # resnet的网络结构，resnet50版本
        │   ├──sal_edge_loss.py             # 损失定义
        │   ├──train_forward_backward.py    # 前向传播和反向传播定义
        ├── pretrained_model_convert        # pytorch预训练模型转换成mindspore模型  
        │   ├──pth_to_msp.py                # pth文件转换成ckpt文件
        │   ├──resnet_msp.py                # mindspore下resnet预训练模型的网络结构
        │   ├──resnet_pth.py                # pytorch下resnet预训练模型的网络结构
        │   ├──vgg_msp.py                   # mindspore下vgg预训练模型的网络结构
        │   ├──vgg_pth.py                   # pytorch下vgg预训练模型的网络结构
        ├── sal2edge.py                     # 预处理，把显著图像转化为边缘图像
        ├── data_crop.py                    # 数据裁剪并生成test.lst文件
        ├── train.py                        # 训练脚本
        ├── eval.py                         # 评估脚本
        ├── preprocess.py                   # Ascend310评估预处理脚本
        ├── postprocess.py                  # Ascend310评估后处理脚本
        ├── export.py                       # 模型导出脚本
        ├── default_config.yaml             # 参数配置文件
        ├── requirements.txt                # 其他依赖包
```

### 脚本参数

在default_config.yaml中可以同时配置训练参数和评估参数。

- 配置EGNet，这里列出一些关键参数

```text
device_target: "Ascend"                                 # 运行设备 ["Ascend", "GPU"]
base_model: "resnet"                                    # 主干网络，["vgg", "resnet"]
batch_size: 1                                           # 训练批次大小
n_ave_grad: 10                                          # 梯度累积step数
epoch_size: 30                                          # 总计训练epoch数
image_height: 200                                       # 输入到模型的图像高度
image_width: 200                                        # 输入到模型的图像宽度
train_path: "./data/DUTS-TR/"                           # 训练数据集的路径
test_path: "./data"                                     # 测试数据集的根目录
vgg: "/home/EGnet/EGnet/model/vgg16.ckpt"               # vgg预训练模型的路径
resnet: "/home/EGnet/EGnet/model/resnet50.ckpt"         # resnet预训练模型的路径
model: "EGNet/run-nnet/models/final_vgg_bone.ckpt"      # 测试时使用的checkpoint文件
```

更多配置细节请参考 default_config.yaml。

## 训练过程

### 训练

- Ascend处理器环境运行

```bash
bash run_standalone_train.sh
```

- 线上modelarts训练

线上单卡训练需要配置如下参数

online_train_path（obs桶中训练集DUTS-TR的存储路径）

```bash
├──DUTS-TR
    ├──DUTS-TR-Image
    ├──DUTS-TR-Mask
```

online_pretrained_path（obs桶中预训练模型的存储路径）

```bash
├──pretrained
    ├──resnet_pretrained.ckpt
    ├──vgg_pretrained.ckpt
```

base_model（选择的预训练模型（vgg or resnet））

train_online = True（设定为线上训练）

上述python命令将在后台运行，您可以通过./EGNet/run-nnet/logs/log.txt文件查看结果。

训练结束后，您可在默认./EGNet/run-nnet/models/文件夹下找到检查点文件。

- GPU处理器环境运行

```bash
bash run_standalone_train_gpu.sh
```

### 分布式训练

- Ascend处理器环境运行

```bash
bash run_distribute_train.sh 8 [RANK_TABLE_FILE]
```

线下运行分布式训练请参照[mindspore分布式并行训练基础样例（Ascend）](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_ascend.html)

- 线上modelarts分布式训练

线上分布式训练需要的参数配置与单卡训练基本一致，只需要新增参数is_distributed = True

上述shell脚本将在后台运行分布训练。您可以通过train/train.log文件查看结果。

- GPU处理器环境运行

```bash
# DEVICE_NUM: 使用的显卡数量，如: 8
# USED_DEVICES: 使用的显卡id列表，需和显卡数量对应，如: 0,1,2,3,4,5,6,7
bash run_distribute_train_gpu.sh [DEVICE_NUM] [USED_DEVICES]
```

## 评估过程

### 评估

- Ascend处理器环境运行

```bash
bash run_eval.sh
```

- GPU处理器环境运行，需修改default_config.yaml中的model参数为需要评估的模型路径

```text
model: "EGNet/run-nnet/models/final_vgg_bone.ckpt"      # 测试时使用的checkpoint文件
```

```bash
bash run_eval_gpu.sh
```

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

- Ascend310处理器环境运行，需修改default_config.yaml文件：
1. 修改infer_path路径为所有推理数据集存放的根目录
2. infer_image_root路径为推理数据集的原图
3. 修改sal_mode为数据集类型(e对应ECSSD，t对应DUTS-TE，d对应DUT-OMRON，h对应HKU-IS，p对应PASCAL-S, s对应SOD)

```text
infer_path: "./data_crop"                               # 所有推理数据集存放的根目录
infer_image_root: "./data_crop/SOD/images/"             # 推理数据集的原图
```

```bash
bash run_infer_310.sh [MINDIR_PATH]
```

## 导出过程

### 导出

在导出之前需要修改default_config.yaml配置文件的配置项ckpt_file或传入参数--ckpt_file.
在default_config.yaml 中修改base_model为骨干网络 ，如"resnet"或 "vgg".

```shell
python export.py --ckpt_file=[CKPT_FILE]
```

## 模型描述

### 性能

#### 评估性能

##### DUTS-TR上的EGNet(Ascend)

| 参数                 | Ascend                                                     | Ascend                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | EGNet（VGG）                                                | EGNet（resnet）           |
| 资源                   | Ascend 910（单卡/8卡）              | Ascend 910（单卡/8卡）                |
| 上传日期              | 2021-12-25                                 | 2021-12-25 |
| MindSpore版本          | 1.3.0                                                       | 1.3.0                  |
| 数据集                    | DUTS-TR                                                    | DUTS-TR               |
| 训练参数        | epoch=30, steps=1050, batch_size = 10, lr=2e-5              | epoch=30, steps=1050, batch_size=10, lr=5e-5    |
| 优化器                  | Adam                                                    | Adam               |
| 损失函数              | Binary交叉熵                                       | Binary交叉熵  |
| 速度                      | 单卡：593.460毫秒/步 ;  8卡：460.952毫秒/步                         | 单卡：569.524毫秒/步;  8卡：466.667毫秒/步       |
| 总时长                 | 单卡：5h3m ;   8卡： 4h2m                         | 单卡：4h59m ;  8卡：4h5m     |
| 微调检查点 | 412M (.ckpt文件)                                         | 426M (.ckpt文件)    |
| 脚本                    | [EGNet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) | [EGNet 脚本](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) |

##### DUTS-TR上的EGNet(GPU)

| 参数                 | GPU                                                      | GPU                    |
| -------------------------- | ----------------------------------------------------------- | ---------------------- |
| 模型版本              | EGNet（VGG）                                                | EGNet（resnet）           |
| 资源                   | GeForce RTX 2080 Ti（单卡） V100（多卡）             | GeForce RTX 2080 Ti（单卡） V100（多卡）               |
| 上传日期              | 2021-12-02                                 | 2021-12-02 |
| MindSpore版本          | 1.3.0                                                       | 1.3.0                  |
| 数据集                    | DUTS-TR                                                    | DUTS-TR               |
| 训练参数        | epoch=30, steps=1050, batch_size = 10, lr=2e-5              | epoch=30, steps=1050, batch_size=10, lr=5e-5    |
| 优化器                  | Adam                                                    | Adam               |
| 损失函数              | Binary交叉熵                                       | Binary交叉熵  |
| 速度                      | 单卡：1148.571毫秒/步 ;  2卡：921.905毫秒/步                          | 单卡：1323.810毫秒/步;  2卡：1057.143毫秒/步      |
| 总时长                 | 单卡：10h3m ;  2卡：8h4m                          | 单卡：11h35m ;  2卡：9h15m      |
| 微调检查点 | 412M (.ckpt文件)                                         | 426M (.ckpt文件)    |
| 脚本                    | [EGNet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) | [EGNet 脚本](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) |

#### 推理性能

##### 显著性检测数据集上的EGNet(Ascend)

| 参数          | Ascend                      | Ascend                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | EGNet（VGG）                | EGNet（resnet）               |
| 资源            |  Ascend 910                  | Ascend 910                          |
| 上传日期       | 2021-12-25 | 2021-12-25 |
| MindSpore 版本   | 1.3.0                       | 1.3.0                       |
| 数据集             | SOD, 300张图像     | SOD, 300张图像     |
| 评估指标（单卡）            | MaxF:0.865 ; MAE:0.154 ; S:0.731 | MaxF:0.876 ; MAE:0.145 ; S:0.738  |
| 评估指标（多卡）            | MaxF:0.866 ; MAE:0.153 ; S:0.736 | MaxF:0.879 ; MAE:0.144 ; S:0.740  |
| 数据集             | ECSSD, 1000张图像     | ECSSD, 1000张图像     |
| 评估指标（单卡）            | MaxF:0.936 ; MAE:0.074 ; S:0.863 | MaxF:0.947 ; MAE:0.064 ; S:0.876  |
| 评估指标（多卡）            | MaxF:0.935 ; MAE:0.080 ; S:0.859 | MaxF:0.945 ; MAE:0.068 ; S:0.873  |
| 数据集             | PASCAL-S, 850张图像     | PASCAL-S, 850张图像     |
| 评估指标（单卡）            | MaxF:0.877 ; MAE:0.118 ; S:0.765 | MaxF:0.886 ; MAE:0.106 ; S:0.779  |
| 评估指标（多卡）            | MaxF:0.878 ; MAE:0.119 ; S:0.765 | MaxF:0.888 ; MAE:0.108 ; S:0.778  |
| 数据集             | DUTS-OMRON, 5168张图像     | DUTS-OMRON, 5168张图像     |
| 评估指标（单卡）            | MaxF:0.782 ; MAE:0.142 ; S:0.752 | MaxF:0.799 ; MAE:0.133 ; S:0.767  |
| 评估指标（多卡）            | MaxF:0.781 ; MAE:0.145 ; S:0.749 | MaxF:0.799 ; MAE:0.133 ; S:0.764  |
| 数据集             | HKU-IS, 4447张图像     | HKU-IS, 4447张图像     |
| 评估指标（单卡）            | MaxF:0.919 ; MAE:0.073 ; S:0.867 | MaxF:0.929 ; MAE:0.063 ; S:0.881  |
| 评估指标（多卡）            | MaxF:0.914 ; MAE:0.079 ; S:0.860 | MaxF:0.925 ; MAE:0.068 ; S:0.876  |

##### 显著性检测数据集上的EGNet(GPU)

| 参数          | GPU                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | EGNet（VGG）                | EGNet（resnet）               |
| 资源            |  GeForce RTX 2080 Ti                  | GeForce RTX 2080 Ti                          |
| 上传日期       | 2021-12-02 | 2021-12-02 |
| MindSpore 版本   | 1.3.0                       | 1.3.0                       |
| 数据集             | DUTS-TE, 5019张图像     | DUTS-TE, 5019张图像     |
| 评估指标（单卡）            | MaxF:0.852 ; MAE:0.094 ; S:0.819 | MaxF:0.862 ; MAE:0.089 ; S:0.829  |
| 评估指标（多卡）            | MaxF:0.853 ; MAE:0.098 ; S:0.816 | MaxF:0.862 ; MAE:0.095 ; S:0.825  |
| 数据集             | SOD, 300张图像     | SOD, 300张图像     |
| 评估指标（单卡）            | MaxF:0.877 ; MAE:0.149 ; S:0.739 | MaxF:0.876 ; MAE:0.150 ; S:0.732  |
| 评估指标（多卡）            | MaxF:0.876 ; MAE:0.158 ; S:0.734 | MaxF:0.874 ; MAE:0.153 ; S:0.736  |
| 数据集             | ECSSD, 1000张图像     | ECSSD, 1000张图像     |
| 评估指标（单卡）            | MaxF:0.940 ; MAE:0.069 ; S:0.868 | MaxF:0.947 ; MAE:0.064 ; S:0.876  |
| 评估指标（多卡）            | MaxF:0.938 ; MAE:0.079 ; S:0.863 | MaxF:0.947 ; MAE:0.066 ; S:0.878  |
| 数据集             | PASCAL-S, 850张图像     | PASCAL-S, 850张图像     |
| 评估指标（单卡）            | MaxF:0.881 ; MAE:0.110 ; S:0.771 | MaxF:0.879 ; MAE:0.112 ; S:0.772  |
| 评估指标（多卡）            | MaxF:0.883 ; MAE:0.116 ; S:0.772 | MaxF:0.882 ; MAE:0.115 ; S:0.774  |
| 数据集             | DUTS-OMRON, 5168张图像     | DUTS-OMRON, 5168张图像     |
| 评估指标（单卡）            | MaxF:0.787 ; MAE:0.139 ; S:0.754 | MaxF:0.799 ; MAE:0.139 ; S:0.761  |
| 评估指标（多卡）            | MaxF:0.789 ; MAE:0.144 ; S:0.753 | MaxF:0.800 ; MAE:0.143 ; S:0.762  |
| 数据集             | HKU-IS, 4447张图像     | HKU-IS, 4447张图像     |
| 评估指标（单卡）            | MaxF:0.923 ; MAE:0.067 ; S:0.873 | MaxF:0.928 ; MAE:0.063 ; S:0.878  |
| 评估指标（多卡）            | MaxF:0.921 ; MAE:0.074 ; S:0.868 | MaxF:0.928 ; MAE:0.067 ; S:0.878  |

## ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。
