# 目录

- [目录](#目录)
- [ArtTrack描述](#arttrack描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [MPII数据集](#mpii数据集)
        - [单卡](#单卡)
        - [多卡](#多卡)
        - [推理与评估](#推理与评估)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [MPII数据集](#mpii数据集-1)
            - [环境](#环境)
            - [训练](#训练)
                - [单卡](#单卡-1)
                - [多卡](#多卡-1)
            - [推理与评估](#推理与评估-1)
    - [评估过程](#评估过程)
        - [MPII数据集](#mpii数据集-2)
            - [评估](#评估)
            - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [MPII上的ArtTrack](#mpii上的arttrack)
- [ModelZoo主页](#modelzoo主页)

# ArtTrack描述

ArtTrack是2017年提出的基于卷积网络的多人跟踪模型。现实的场景十分复杂，通常包括快速运动，外观和服装的大幅度变异，以及人与人之间的遮挡等因素。为了利用可用的图像信息，ArtTrack使用ResNet端到端地将身体关节和特定人的联系起来。并将这些关联合并到一个框架中，在时间维度上把关节（人）关联起来。为了提高视频推理的效率，采用了基于局部组合优化的快速推理方法，通过建立一个稀疏模型，使变量之间的连接数量保持在最小。

[论文](https://arxiv.org/abs/1612.01465) ： Insafutdinov E, Andriluka M, Pishchulin L, et al. Arttrack: Articulated
multi-person tracking in the wild[C]//Proceedings of the IEEE conference on computer vision and pattern recognition.
2017: 6457-6465.

# 模型架构

ArtTrack的总体网络架构如下：
[链接](https://arxiv.org/abs/1612.01465)

# 数据集

使用的数据集：

- [MPII](http://human-pose.mpi-inf.mpg.de/)
    MPII人体姿态数据集是一种用于评估关节式人体姿态估计的基准。该数据集包含大约25000张图像，其中包含超过40000名带有标注的人体关节的人。这些图像是根据人类日常活动的分类系统收集的。总的来说，该数据集覆盖410个人类活动，每个图像都提供一个活动标签。

解压后的MPII如图所示：

```text
mpii
├── images
│  ├── 099890470.jpg
│  ├── 099894296.jpg
│  ├── 099914957.jpg
│  ├── 099920730.jpg
│  ├── 099927653.jpg
│  └── ....
└── mpii_human_pose_v1_u12_1.mat
```

# 环境要求

- 硬件（GPU）
    - 使用GPU训练模型。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

## MPII数据集

### 单卡

```bash
# 用法 bash scripts/run_train_single_gpu.sh TARGET CONFIG_PATH [OPTION] ...
bash scripts/run_train_single_gpu.sh mpii_single config/mpii_train_single_gpu.yaml
```

### 多卡

```bash
# 用法 bash scripts/run_train_multiple_gpu.sh TARGET CONFIG_PATH CUDA_VISIBLE_DEVICES DEVICE_NUM [OPTION] ...
bash scripts/run_train_multiple_gpu.sh mpii_single config/mpii_train_multiple_gpu.yaml "0,1,2,3,4,5,6,7" 8
```

### 推理与评估

```bash
# 推理与评估
# 用法 bash scripts/eval.sh TARGET CKPT_PATH OUTPUT_PATH
# 根据实际情况替换ckpt文件
bash scripts/eval.sh mpii_single ckpt/rand_0/arttrack-1_356.ckpt out/prediction.mat

# 只推理
python eval.py mpii_single --config config/mpii_eval.yaml --option "load_ckpt=ckpt/rank_0/arttrack-1_356.ckpt" --output "out/prediction.mat"

# 只评估
python eval.py mpii_single --config config/mpii_eval.yaml --accuracy  --prediction "out/prediction.mat"
```

# 脚本说明

## 脚本及样例代码

```text
ArtTrack
├── config
│  ├── coco_pairwise.yaml                   # coco数据集pairwise配置
│  ├── coco_eval.yaml                       # coco数据集推理配置
│  ├── coco_train_multiple_gpu.yaml         # coco数据集多卡训练配置
│  ├── coco_train_single_gpu.yaml           # coco数据集单卡训练配置
│  ├── mpii_eval.yaml                       # mpii数据集推理配置
│  ├── mpii_train_multiple_gpu.yaml         # mpii数据集多卡训练配置
│  ├── mpii_train_single_gpu.yaml           # mpii数据集单卡训练配置
│  └── tf2ms.json                           # tf模型参数转mindspord映射表
├── environment.yaml                        # conda环境
├── log.yaml                                # log配置
├── patch
│  ├── 0001-fix-lib.patch                   # tf代码补丁
│  ├── 0002-pybind11.patch                  # tf代码补丁
│  └── 0003-split-dataset.patch             # tf代码补丁
├── preprocess.py                           # 预处理脚本
├── README_CN.md                            # 模型相关说明
├── requirements.txt                        # 依赖列表
├── scripts
│  ├── download.sh                          # 下载预训练和数据集
│  ├── eval.sh                              # 推理与评估
│  ├── prepare.sh                           # 环境预处理
│  ├── run_train_multiple_gpu.sh            # 多卡训练
│  └── run_train_single_gpu.sh              # 单卡训练
├── src
│  ├── __init__.py
│  ├── tool
│  │  ├── __init__.py
│  │  ├── preprocess
│  │  │  ├── __init__.py
│  │  │  ├── crop.py                        # mpii图片放缩
│  │  │  ├── mat2json.py                    # 由mat索引转为json
│  │  │  ├── pairwise_stats.py              # 生成pairwise
│  │  │  ├── parts.json
│  │  │  ├── preprocess_single.py           # mpii 预处理
│  │  │  ├── split.py                       # 数据集划分
│  │  │  ├── tf2ms.py                       # tf ckpt转为mindspore模型
│  │  │  └── utils.py                       # 预处理工具
│  │  ├── eval
│  │  │  ├── __init__.py
│  │  │  ├── coco.py                        # coco评估
│  │  │  ├── pck.py                         # mpii评估
│  │  │  └── multiple.py                    # coco推理
│  │  └── decorator.py
│  ├── config.py                            # 配置相关
│  ├── dataset
│  │  ├── __init__.py
│  │  ├── coco.py                           # coco数据集
│  │  ├── mpii.py                           # mpii数据集
│  │  ├── pose.py                           # 数据集加载
│  │  └── util.py                           # 数据集工具
│  ├── log.py                               # 日志配置
│  ├── model
│  │  ├── losses.py                         # 自定义loss
│  │  ├── pose.py                           # 模型
│  │  ├── predict.py                        # 推理
│  │  └── resnet
│  │     ├── __init__.py
│  │     ├── resnet.py                      # ResNet
│  │     └── util.py                        # ResNet工具
│  ├── multiperson
│  │  ├── detections.py                     # 目标发现
│  │  ├── predict.py                        # 推理
│  │  └── visualize.py                      # 可视化
│  └── args_util.py                         # 命令行工具
├── eval.py                                 # 推理与评估
└── train.py                                # 训练工具
```

## 脚本参数

```yaml
# 数据集
dataset:
    path: out/train_index_dataset.json
    type: mpii_raw
    parallel: 1
    # need about 13G GPU memory
    batch_size: 16
    mirror: true
    padding: yes
    shuffle: yes

# mindspore context
context:
    # GRAPH
    # mode: 0

    # PYNATIVE
    mode: 1
    device_target: GPU

# mindspore parallel context
# 若该字段存在使用数据并行
parallel_context:
    parallel_mode: data_parallel

# 训练轮次
epoch: 25
# 是否是训练
train: yes

# 关键点数
num_joints: 14

# 数据集中关键点编号，与all_joints_names对应
all_joints: [ [ 0, 5 ], [ 1, 4 ], [ 2, 3 ], [ 6, 11 ], [ 7, 10 ], [ 8, 9 ], [ 12 ], [ 13 ] ]
# 关键点名称
all_joints_names: [ 'ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead' ]

# 评估阈值
pck_threshold: 2

# 数据集处理时热力图阈值
pos_dist_thresh: 17
# 全局放缩
global_scale: 0.8452830189

# 是否使用location refinement
location_refinement: true
# location refinement是否使用huber loss
locref_huber_loss: true
# huber loss权重
locref_loss_weight: 0.05
locref_stdev: 7.2801

# 是否使用intermediate supervision
intermediate_supervision: no
# intermediate supervision 在第3个block的层数
intermediate_supervision_layer: 12
# intermediate supervision 的in channel
intermediate_supervision_input: 1024

# 限制图片大小
max_input_size: 600

# 学习率
multi_step:
    - [ 0.05,0.2,0.02,0.01 ]
    - [ 500,2700,4600,6500 ]
```

更多配置细节请参考`config`目录下配置文件。

## 训练过程

### MPII数据集

#### 环境

```bash
# conda 环境
conda env create -n art-track python=3.7.5
conda activate art-track
pip install -r requeirments.txt

# 下载ResNet101预训练模型
bash scripts/download.sh pretrained_resnet101

# 安装依赖，转换模型
bash scripts/prepare.sh env

# 下载mpii数据集
bash scripts/download.sh dataset_mpii

# mpii数据集预处理
bash scripts/prepare.sh mpii
```

#### 训练

##### 单卡

```bash
# 用法 bash scripts/run_train_single_gpu.sh TARGET CONFIG_PATH [OPTION] ...
bash scripts/run_train_single_gpu.sh mpii_single config/mpii_train_single_gpu.yaml
```

##### 多卡

```bash
# 用法 bash scripts/run_train_multiple_gpu.sh TARGET CONFIG_PATH CUDA_VISIBLE_DEVICES DEVICE_NUM [OPTION] ...
bash scripts/run_train_multiple_gpu.sh mpii_single config/mpii_train_multiple_gpu.yaml "0,1,2,3,4,5,6,7" 8
```

#### 推理与评估

```bash
# 推理与评估
# 用法 bash scripts/eval.sh TARGET CKPT_PATH OUTPUT_PATH
# 根据实际情况替换ckpt文件
bash scripts/eval.sh mpii_single ckpt/rand_0/arttrack-1_356.ckpt out/prediction.mat

# 推理
python eval.py mpii_single --config config/mpii_eval.yaml --option "load_ckpt=ckpt/rank_0/arttrack-1_356.ckpt" --output "out/prediction.mat"

# 评估
python eval.py mpii_single --config config/mpii_eval.yaml --accuracy  --prediction "out/prediction.mat"
```

## 评估过程

### MPII数据集

#### 评估

```bash
# 推理与评估
# 用法 bash scripts/eval.sh TARGET CKPT_PATH OUTPUT_PATH
# 根据实际情况替换ckpt文件
bash scripts/eval.sh mpii_single ckpt/rand_0/arttrack-1_356.ckpt out/prediction.mat

# 只推理
python eval.py mpii_single --config config/mpii_eval.yaml --option "load_ckpt=ckpt/rank_0/arttrack-1_356.ckpt" --output "out/prediction.mat"

# 只评估
python eval.py mpii_single --config config/mpii_eval.yaml --accuracy  --prediction "out/prediction.mat"
```

#### 结果

```text
& ankle & knee & hip & wrist & elbow & shoulder & chin & forehead & total
& 66.0 & 70.9 & 74.7 & 65.5 & 72.4 & 83.4 & 87.0 & 84.2 & 74.1
```

# 模型描述

## 性能

### 评估性能

#### MPII上的ArtTrack

| 参数          | GPU V100                                                                    |
| ------------- | --------------------------------------------------------------------------- |
| 模型版本      | ArtTrack                                                                    |
| 资源          | Telsa GPU V100                                                              |
| 上传日期      | 2022-01-11                                                                  |
| MindSpore版本 | 1.5.0                                                                       |
| 数据集        | MPII                                                                        |
| 训练参数      | epoch=25, steps per epoch=356, batch_size=16                                |
| 优化器        | SGD                                                                         |
| 损失函数      | SigmoidCrossEntropyWithLogits                                               |
| 输出          | 关键点坐标                                                                  |
| 损失          | 0.016214851                                                                 |
| 速度          | 1292.854毫秒/步(8卡)                                                        |
| 总时长        | 3.2小时                                                                     |
| 微调检查点    | 496M (.ckpt文件)                                                            |
| 脚本          | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/ArtTrack) |

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
