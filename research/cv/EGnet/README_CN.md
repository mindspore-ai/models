
# 目录

- [目录](#目录)
- [EGNet描述](#EGNet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
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
            - [DUTS-TR上的EGNet](#DUTS-TR上的EGNet)
        - [推理性能](#推理性能)
            - [显著性检测数据集上的EGNet](#显著性检测数据集上的EGNet)
- [ModelZoo主页](#modelzoo主页)

# EGNet描述

EGNet是用来解决静态目标检测问题，它由边缘特征提取部分、显著性目标特征提取部分以及一对一的导向模块三部分构成，利用边缘特征帮助显著性目标特征定位目标，使目标的边界更加准确。在6个不同的数据集中与15种目前最好的方法进行对比，实验结果表明EGNet性能最优。

[论文](https://arxiv.org/abs/1908.08297): Zhao J X, Liu J J, Fan D P, et al. EGNet: Edge guidance network for salient object detection[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 8779-8788.

# 模型架构

EGNet网络由三个部分组成，NLSEM（边缘提取模块）、PSFEM（目标特征提取模块）、O2OGM（一对一指导模块），原始图片通过两次卷积输出图片边缘信息，与此同时，对原始图像进行更深层次的卷积操作提取salient object，然后将边缘信息与不同深度提取出来的显著目标在一对一指导模块中分别FF（融合），再分别经过卷积操作得到不同程度的显著性图像，最终输出了一张融合后的显著性检测图像。

# 数据集

使用的数据集：[显著性检测数据集](<https://blog.csdn.net/studyeboy/article/details/102383922?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163031601316780274127035%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163031601316780274127035&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-5-102383922.first_rank_v2_pc_rank_v29&utm_term=DUTS-TE%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8B%E8%BD%BD&spm=1018.2226.3001.4187>)

- 数据集大小：
    - 训练集：DUTS-TR数据集，210MB，共10533张最大边长为400像素的彩色图像，均从ImageNet DET训练/验证集中收集。
    - 测试集：SOD数据集，21.2MB，共300张最大边长为400像素的彩色图像，此数据集是基于Berkeley Segmentation Dataset（BSD）的显著对象边界的集合。
    - 测试集：ECSSD数据集，64.6MB，共1000张最大边长为400像素的彩色图像。
    - 测试集：PASCAL-S数据集，175 MB，共10个类、850张32*32彩色图像。该数据集与其他显著物体检测数据集区别较大, 没有非常明显的显著物体, 并主要根据人类的眼动进行标注数据集, 因此该数据集难度较大。
    - 测试集：DUTS-OMRON数据集，107 MB，共5168张最大边长为400像素的彩色图像。数据集中具有一个或多个显著对象和相对复杂的背景，具有眼睛固定、边界框和像素方面的大规模真实标注的数据集。
    - 测试集：HKU-IS数据集，893MB，共4447张最大边长为400像素的彩色图像。数据集中每张图像至少满足以下的3个标准之一:1)含有多个分散的显著物体; 2)至少有1个显著物体在图像边界; 3)显著物体与背景表观相似。
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

```shell
# 数据集进行裁剪
python data_crop.py --data_name=[DATA_NAME]  --data_root=[DATA_ROOT] --output_path=[OUTPUT_PATH]

# 运行训练示例
bash run_standalone_train.sh

# 运行分布式训练示例
bash run_distribute_train.sh 8 [RANK_TABLE_FILE]

# 运行评估示例
bash run_eval.sh
```

训练集路径在default_config.yaml中的data项设置

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── EGNet
        ├── README.md                     # EGNet相关说明
        ├── model_utils                   # config，modelarts等配置脚本文件夹
        │   ├──config.py                  # 解析参数配置文件
        ├── scripts
        │   ├──run_train.sh               # 启动Ascend单机训练（单卡）
        │   ├──run_distribute_train.sh    # 启动Ascend分布式训练（8卡）
        │   ├──run_eval.sh                # 启动Ascend评估
        ├── src
        │   ├──dataset.py                 # 加载数据集
        │   ├──egnet.py                   # EGNet的网络结构
        │   ├──vgg.py                     # vgg的网络结构
        │   ├──resnet.py                  # resnet的网络结构
        │   ├──sal_edge_loss.py           # 损失定义
        │   ├──train_forward_backward.py  # 前向传播和反向传播定义
        ├── sal2edge.py                   # 预处理，把显著图像转化为边缘图像
        ├── data_crop.py                  # 数据裁剪
        ├── train.py                      # 训练脚本
        ├── eval.py                       # 评估脚本
        ├── export.py                     # 模型导出脚本
        ├── default_config.yaml           # 参数配置文件
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置EGNet和DUTS-TR数据集。

```text
dataset_name: "DUTS-TR"                      # 数据集名称
name: "egnet"                                # 网络名称
pre_trained: Ture                            # 是否基于预训练模型训练
lr_init: 5e-5(resnet) or 2e-5(vgg)           # 初始学习率
batch_size: 10                               # 训练批次大小
epoch_size: 30                               # 总计训练epoch数
momentum: 0.1                                # 动量
weight_decay:5e-4                            # 权重衰减值
image_height: 200                            # 输入到模型的图像高度
image_width: 200                             # 输入到模型的图像宽度
train_data_path: "./data/DUTS-TR/"           # 训练数据集的相对路径
eval_data_path: "./data/SOD/"            # 评估数据集的相对路径
checkpoint_path: "./EGNet/run-nnet/models/"  # checkpoint文件保存的相对路径
```

更多配置细节请参考 src/config.py。

# 训练过程

## 训练

- 数据集进行裁剪：

```bash
python data_crop.py --data_name=[DATA_NAME]  --data_root=[DATA_ROOT] --output_path=[OUTPUT_PATH]
```

- Ascend处理器环境运行

```bash
python train.py --mode=train --base_model=vgg --vgg=[PRETRAINED_PATH]
python train.py --mode=train --base_model=resnet --resnet=[PRETRAINED_PATH]
```

- 线上modelarts训练

线上单卡训练需要配置如下参数

online_train_path（obs桶中训练集DUTS-TR的存储路径）

```bash
├──DUTS-TR
    ├──DUTS-TR-Image  
    ├──DUTS-TR-Mask
    ├──train_pair.lst
    ├──train_pair_edge.lst
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

## 分布式训练

- Ascend处理器环境运行

```bash
bash run_distribute_train.sh 8 [RANK_TABLE_FILE]
```

线下运行分布式训练请参照[mindspore分布式并行训练基础样例（Ascend）](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_ascend.html)

- 线上modelarts分布式训练

线上训练需要的参数配置与单卡训练基本一致，只需要新增参数is_distributed = True

上述shell脚本将在后台运行分布训练。您可以通过train/train.log文件查看结果。

# 评估过程

## 评估

- Ascend处理器环境运行

```bash
python eval.py --model=[MODEL_PATH] --sal_mode=[DATA_NAME] --test_fold=[TEST_DATA_PATH] --base_model=vgg
python eval.py --model=[MODEL_PATH] --sal_mode=[DATA_NAME] --test_fold=[TEST_DATA_PATH] --base_model=resnet
```

数据集文件结构

```bash
├──NAME
    ├──ground_truth_mask  
    ├──images
    ├──test.lst
```

# 导出过程

## 导出

在导出之前需要修改default_config.ymal配置文件.需要修改的配置项为ckpt_file.

```shell
python export.py --ckpt_file=[CKPT_FILE]
```

# 模型描述

## 性能

### 评估性能

#### DUTS-TR上的EGNet

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
| 脚本                    | [EGNnet脚本]() | [EGNet 脚本]() |

### 推理性能

#### 显著性检测数据集上的EGNet

| 参数          | Ascend                      | Ascend                         |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本       | EGNet（VGG）                | EGNet（resnet）               |
| 资源            |  Ascend 910                  | Ascend 910                          |
| 上传日期       | 2021-12-25 | 2021-12-25 |
| MindSpore 版本   | 1.3.0                       | 1.3.0                       |
| 数据集             | SOD, 300张图像     | SOD, 300张图像     |
| 评估指标（单卡）            | MaxF:0.8659637 ; MAE:0.1540910 ; S:0.7317967 | MaxF:0.8763882 ; MAE:0.1453154 ; S:0.7388669  |
| 评估指标（多卡）            | MaxF:0.8667928 ; MAE:0.1532886 ; S:0.7360025 | MaxF:0.8798361 ; MAE:0.1448086 ; S:0.74030272  |
| 数据集             | ECSSD, 1000张图像     | ECSSD, 1000张图像     |
| 评估指标（单卡）            | MaxF:0.9365406 ; MAE:0.0744784 ; S:0.8639620 | MaxF:0.9477927 ; MAE:0.0649923 ; S:0.8765208  |
| 评估指标（多卡）            | MaxF:0.9356243 ; MAE:0.0805953 ; S:0.8595030 | MaxF:0.9457578 ; MAE:0.0684581 ; S:0.8732929  |
| 数据集             | PASCAL-S, 850张图像     | PASCAL-S, 850张图像     |
| 评估指标（单卡）            | MaxF:0.8777129 ; MAE:0.1188116 ; S:0.7653073 | MaxF:0.8861882 ; MAE:0.1061731 ; S:0.7792912  |
| 评估指标（多卡）            | MaxF:0.8787268 ; MAE:0.1192975 ; S:0.7657838 | MaxF:0.8883396 ; MAE:0.1081997 ; S:0.7786236  |
| 数据集             | DUTS-OMRON, 5168张图像     | DUTS-OMRON, 5168张图像     |
| 评估指标（单卡）            | MaxF:0.7821059 ; MAE:0.1424146 ; S:0.7529001 | MaxF:0.7999835 ; MAE:0.1330678 ; S:0.7671095  |
| 评估指标（多卡）            | MaxF:0.7815770 ; MAE:0.1455649 ; S:0.7493499 | MaxF:0.7997979 ; MAE:0.1339806 ; S:0.7646356  |
| 数据集             | HKU-IS, 4447张图像     | HKU-IS, 4447张图像     |
| 评估指标（单卡）            | MaxF:0.9193007 ; MAE:0.0732772 ; S:0.8674455 | MaxF:0.9299341 ; MAE:0.0631132 ; S:0.8817522  |
| 评估指标（多卡）            | MaxF:0.9145629 ; MAE:0.0793372 ; S:0.8608878 | MaxF:0.9254014; MAE:0.0685441 ; S:0.8762386  |

# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。
