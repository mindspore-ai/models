# SSC_ResNet50(Semi-Supervised for Resnet50)

## 目录

<!-- TOC -->

- [目录](#目录)
- [概述](#概述)
- [论文](#论文)
- [特性](#特性)
    - [混合精度（Ascend）](#混合精度（Ascend）)
    - [半监督](#半监督)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器环境运行](#ascend处理器环境运行)
            - [GPU处理器环境运行](#gpu处理器环境运行)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器环境运行](#ascend处理器环境运行-1)
            - [GPU处理器环境运行](#gpu处理器环境运行-1)
        - [结果](#结果-1)
- [性能](#性能)
    - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## 概述

resnet50：残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

SSC-ResNet：通过半监督+主动学习的方式，进行resnet50分类网络的训练，在有限的标注数据上，充分利用大量无标注数据，进行伪标签的学习，提升模型能力。用于解决实际用户在标注预算有限的情况下，利用主动学习，挑选最具价值数据进行标注，并且补充大量无/少成本的无标注数据，进行半监督学习，达到利用少量标注数据+大量无标注数据，得到和全量标注一样的模型性能。SSC-Resnet50，仅使用25%ImageNet的有标注数据，达到全监督的模型性能。

## 论文

1. [论文](https://arxiv.org/pdf/1512.03385.pdf)：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun."Deep Residual Learning for Image Recognition"
2. [论文](https://arxiv.org/pdf/2011.11183.pdf)：Li J, Xiong C, Hoi S C H."Comatch: Semi-supervised learning with contrastive graph regularization"

## 特性

### 混合精度（Ascend）

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。 以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子

### 半监督

在有限的标注数据上，充分利用大量无标注数据，进行伪标签的学习，提升模型能力。

## 数据集

使用的数据集：[ImageNet2012](<http://www.image-net.org/>)

- 数据集大小：1000类224*224彩色图像
    - 训练集：12w张彩色图像，共120G
    - 测试集：5w张彩色图像，共5G

## 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。
    - 内存 > 400G（8卡Ascend 运行需要占用约370G内存左右）

- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)

- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

- 10%随机数据生成

```text
python generate_anno.py --data [TRAIN_DATASET_PATH] \
                        --annotation [ANNOTATION] \
                        --num_class [NUM_CLASSES] \
                        --num_per_class [ALL_LABELED_SAMPLES_COUNT]
```

- Ascend处理器环境运行

```text
# 使用8卡训练
bash run_distribute_train_ascend_8p.sh [ANNOTATION] [EXP_DIR] [RANK_TABLE_FILE] [PRE_TRAINED](option)
# 使用4卡训练
bash run_distribute_train_ascend_4p.sh [ANNOTATION] [EXP_DIR] [RANK_TABLE_FILE] [PRE_TRAINED](option)
```

- GPU处理器环境运行

```text
# 使用8卡训练
bash run_distribute_train_gpu_8p.sh [ANNOTATION] [EXP_DIR] [PRE_TRAINED](option)
# 使用4卡训练
bash run_distribute_train_gpu_4p.sh [ANNOTATION] [EXP_DIR] [PRE_TRAINED](option)
```

## 脚本说明

### 脚本及样例代码

```shell
├── class_to_idx.json                            # ImageNet id与类别对应json
├── default_config.yaml                          # 参数配置
├── eval.py                                      # 评估模型
├── generate_anno.py                             # 生成anno json
├── merge_final_anno.py                          # 合并初始数据与价值筛选数据
├── README.md
├── requirements.txt                             # python环境
├── scripts
│   ├── generate_anno.sh                         # 生成anno json
│   ├── run_distribute_train_ascend_4p.sh        # Ascend 4卡一键式训练
│   ├── run_distribute_train_ascend_8p.sh        # Ascend 8卡一键式训练
│   ├── run_distribute_train_gpu_4p.sh           # GPU 4卡一键式训练
│   ├── run_distribute_train_gpu_8p.sh           # GPU 8卡一键式训练
│   ├── run_distribute_train_model_ascend.sh     # Ascend多卡训练模型
│   ├── run_distribute_train_model_gpu.sh        # GPU多卡训练模型
│   ├── run_eval_ascend.sh                       # Ascend评估
│   ├── run_eval_gpu.sh                          # GPU评估脚本
│   ├── select_sample_ascend.sh                  # Ascend数据筛选
│   └── select_sample_gpu.sh                     # GPU数据筛选
├── select_sample.py                             # 数据筛选
├── src
│   ├── dataAugment.py                           # 数据增强
│   ├── dataset.py                               # 数据集
│   ├── __init__.py
│   ├── model_utils
│   │   ├── config.py                            # 加载配置
│   │   ├── __init__.py
│   │   ├── local_adapter.py                     # 本地设备配置
│   ├── network
│   │   ├── grad_clip.py                         # 梯度裁剪
│   │   ├── __init__.py
│   │   ├── model.py                             # loss模型
│   │   └── resnet.py                            # resnet模型
│   ├── utils.py                                 # 数据处理引用
│   └── warmup_cosine_annealing_lr.py            # 学习率
└── train.py                                     # 训练模型
```

### 脚本参数

```yaml
device_target: "Ascend"                     # 设备类型
device_id: 0                                # 设备id
device_num: 1                               # 设备数量
rank_id: 0                                  # 使用设备编号
exp_dir: ""                                 # 输出路径
print_freq: 50                              # 训练打印频率
# Train options
is_distributed: False                       # 分布式
pre_trained: ""                             # 预训练模型

epochs: 410                                 # 训练轮数
warm_epochs: 10                             # 学习率预热轮数
start_epoch: 0                              # 训练的起始epoch，恢复训练时使用

# lr options
lr: 0.1                                     # 学习率
momentum: 0.9                               # 动量优化器
weight_decay: 0.0001                        # 衰减权重

# Model options
temperature: 0.1                            # 相似缩放温度
low_dim: 128                                # 特征尺度
moco_m: 0.996                               # 更新动量编码器的动量
K: 30000                                    # 内存库大小和动量队列
thr: 0.4                                    # 伪标签置信阈值
contrast_th: 0.4                            # 伪标签图连接阈值
lam_u: 10                                   # 无监督交叉熵损失的权重
lam_c: 1                                    # 无监督对比损失的权重
num_hist: 128                               #
alpha: 0.9                                  # 构建伪标签时模型预测的权重

# Dataset options
workers: 10                                 # 数据加载时的并行工程数
batch_size: 20                              # 输入张量的批次大小
num_clas: 1000                              # 数据集类别数量
annotation: "/path/to/annotation.json"      # 标注文件路径
unlabel_label: 4                            # 无标数据:有标数据的比例
unlabel_aug: True                           # 无标数据增强
unlabel_randomaug_count: 2                  # 无标数据增强数量
unlabel_randomaug_intensity: 10             # 无标数据增强强度
label_aug: True                             # 有标数据增强
label_randomaug_count: 3                    # 有标数据增强数量
label_randomaug_intensity: 5                # 有标数据增强强度

# Eval options
test_root: "/path/to/dataset/val/"          # 评估数据集路径
eval_pre_trained: "/path/to/pretrained.ckpt" # 评估模型路径
folder: 0                                   # 评估模型 0；评估文件夹 1
```

## 训练过程

### 用法

#### Ascend处理器环境运行

`run_distribute_train_ascend_8p.sh`执行脚本中的主要内容如下：

```text
# 使用10%的数据进行训练，得到模型M1
bash run_distribute_train_model_ascend.sh [DEVICE_NUM] [EXP_DIR] [RANK_TABLE_FILE] [ANNOTATION] [PRE_TRAINED](option)

# 使用M1模型进行所有数据的价值排序
bash select_sample_ascend.sh [DEVICE_NUM] [EXP_DIR] [RANK_TABLE_FILE] [PRE_TRAINED]

# 根据价值排序的数据，使用top 15%数据，与1中的数据合并成25%的数据
python merge_final_anno.py --class_to_id [ID_JSON] \
                           --txt_root_path [EXP_DIR] \
                           --base_json [ANNOTATION]

# 使用5中的25%数据，训练得到模型M2，即为最终输出模型
bash run_distribute_train_model_ascend.sh [DEVICE_NUM] [EXP_DIR] [RANK_TABLE_FILE] [ANNOTATION] [PRE_TRAINED]
```

分布式训练需要提前创建JSON格式的HCCL配置文件。

具体操作，参见[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)中的说明。

#### GPU处理器环境运行

`run_distribute_train_gpu_8p.sh`执行脚本中的主要内容如下：

```text
# 使用10%的数据进行训练，得到模型M1
bash run_distribute_train_model_gpu.sh [DEVICE_NUM] [EXP_DIR] [ANNOTATION] [PRE_TRAINED](option)

# 使用M1模型进行所有数据的价值排序
bash select_sample_ascend.sh [DEVICE_NUM] [EXP_DIR] [PRE_TRAINED]

# 根据价值排序的数据，使用top 15%数据，与1中的数据合并成25%的数据
python merge_final_anno.py --class_to_id [ID_JSON] \
                           --txt_root_path [EXP_DIR] \
                           --base_json [ANNOTATION]

# 使用5中的25%数据，训练得到模型M2，即为最终输出模型
bash run_distribute_train_model_gpu.sh [DEVICE_NUM] [EXP_DIR] [ANNOTATION] [PRE_TRAINED]
```

#### 结果

训练checkpoint存在`EXP_DIR/base_model`和`EXP_DIR/final_model`中，你可以从`scripts/train*/device0/`中找到log.log日志，里面有如下结果：

```shell
INFO -  rank: 0, epoch:   0, step:    0, loss_x: 7.946, loss_u: 0.000, loss_contrast: 4.311, batch_time: 230.31, data_time: 1.75, lr: 0.000028, overflow: 0.00000, scaling_sens: 65536.
INFO -  rank: 0, epoch:   1, step:    0, loss_x: 6.758, loss_u: 0.000, loss_contrast: 7.450, batch_time: 4.34, data_time: 3.82, lr: 0.010028, overflow: 0.00000, scaling_sens: 65536.
INFO -  rank: 0, epoch:   2, step:    0, loss_x: 7.144, loss_u: 0.000, loss_contrast: 8.184, batch_time: 3.97, data_time: 3.47, lr: 0.020028, overflow: 0.00000, scaling_sens: 65536.
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
bash run_eval_ascned.sh [DEVICE_ID] [CKPT_PATH] [TEST_ROOT] [FOLDER]
```

#### GPU处理器环境运行

```bash
bash run_eval_gpu.sh [DEVICE_ID] [CKPT_PATH] [TEST_ROOT] [FOLDER]
```

### 结果

评估结果保存在`scripts/eval_1p_*/device*/`路径下，结果存储在`log.txt`中。评估完成后，可在日志中找到如下结果：

```bash
acc1:>=76.5
```

## 性能

### 评估性能

| 参数           | Ascend                             | GPU                                |
|---------------|------------------------------------|------------------------------------|
| 模型版本       | ResNet50(backbone)                 | ResNet50(backbone)                 |
| 资源           | Ascend910                          | GPU                                |
| 上传日期       | 2022/05/10                         | 2022/05/10                         |
| Mindspore版本  | 1.5.2                              | 1.5.2                              |
| 数据集         | ImageNet                           | ImageNet                           |
| 训练参数       | epoch=410\*2, lr=0.1, batch_size=20| epoch=410\*2, lr=0.1, batch_size=20|
| 优化器         | SGD                                | SGD                                |
| 损失函数       | 交叉熵                              | 交叉熵                             |
| 输出           | 概率                               | 概率                               |
| 损失           | 1.074                              | 1.074                             |
| 速度           | 1.56s/step(8卡)                    | 2.28s/step(8卡)                    |
| 总时长         | 270h\*2                            | 370h\*2                            |
| 参数(M)        | 25.5                               | 25.5                               |
| 微调检查点     | 720.27M(ckpt)                       | 720.27M(ckpt)                      |
| 配置文件       | [链接](https://gitee.com/mindspore/models/blob/master/research/cv/ssc_resnet50/default_config.yaml) | [链接](https://gitee.com/mindspore/models/blob/master/research/cv/ssc_resnet50/default_config.yaml) |

## 随机情况说明

在`train.py`中，使用了mindspore的set_seed接口所使用的随机种子。

## ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
