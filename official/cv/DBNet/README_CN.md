# DBNet

***

论文：[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

标签：文本检测

***

## 模型简介

近年来，基于分割的方法在场景文本检测中非常流行，因为分割结果可以更准确地描述各种形状的场景文本，如曲线文本。然而，二值化的后处理对于基于分割的检测是必不可少的，该检测将由分割方法生成的概率图转换为文本的边界框/区域。DBNet论文中提出了一个名为可微分二值化（DB）的模块，它可以在分割网络中执行二值化过程。与DB模块一起优化的分割网络可以自适应地设置二值化阈值，这不仅简化了后处理，而且提高了文本检测的性能。

![img](https://user-images.githubusercontent.com/22607038/142791306-0da6db2a-20a6-4a68-b228-64ff275f67b3.png)

## 数据集

使用的数据集：[ICDAR2015](<https://rrc.cvc.uab.es/?ch=4&com=downloads>)

- 数据集大小：132M
    - 训练集：
        - 图片：88.5M(1000张图片)
        - 标签：157KB
    - 测试集：
        - 图片：43.3M(500张图片)
        - 标签：244KB
- 数据格式：图片，标签

在官网下载完成后将数据集组织成如下结构：

```text
└─ICDAR2015
    ├─ch4_training_images                                       # 训练集原始图片
    ├─ch4_training_localization_transcription_gt                # 训练集标签
    ├─ch4_test_images                                           # 验证集原始图片
    └─Challenge4_Test_Task1_GT                                  # 验证集标签
```

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。参考[MindSpore](https://www.mindspore.cn/install/en)安装运行环境
- 版本依赖 MindSpore >= 1.9

```shell
git clone https://gitee.com/mindspore/models.git
cd models/official/cv/DBNet
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## BenchMark

### 精度

| Model | pretrained Model | config | Train Set | Test Set | Device Num | Epoch | Test Size | Recall | Precision | Hmean | CheckPoint | Graph Train Log |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| DBNet-R18 | [R18](https://download.mindspore.cn/thirdparty/dbnet/resnet18-5c106cde.ckpt) | [cfg](config/dbnet/config_resnet18_1p.yaml) | ICDAR2015 Train | ICDAR2015 Test | 1 | 1200 | 736 | 78.63 | 84.21 | 81.32 | [download]() | [download]() |
| DBNet-R50 | [R50](https://download.mindspore.cn/thirdparty/dbnet/resnet50-19c8e357.ckpt) | [cfg](config/dbnet/config_resnet50_1p.yaml) | ICDAR2015 Train | ICDAR2015 Test | 1 | 1200 | 736 | 81.05 | 88.07 | 84.41 | [download]() | [download]() |
| DBNet-MobileNetv3 | [M3]() | [cfg](config/dbnet/config_mobilenetv3_1p.yaml) | ICDAR2015 Train | ICDAR2015 Test | 1 | 1200 | 736 | 73,.05 | 77.02 | 74.96 | [download]() | [download]() |

### 性能

| device | Model     | dataset   | Params(M) | PyNative train 1P bs=16 FPS | PyNative train 8P bs=8 FPS | PyNative infer FPS | Graph train 1P bs=16 FPS | Graph train 8P bs=8 FPS | Graph infer FPS |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Ascend | DBNet-R18 | ICDAR2015 |  11.78 M  |  95  |  381  |    -     |   150    |   610   | 40.62  |
|  GPU   | DBNet-R18 | ICDAR2015 |  11.78 M  |  18    |   81     |   -    |  23  |    104 |  30.97  |
| Ascend | DBNet-R50 | ICDAR2015 |  24.28 M  |  59    |   237  |   -     |  100    |   460    |  33.88  |
|  GPU   | DBNet-R50 | ICDAR2015 |  24.28 M  |  15      |   74    |  -    |  20  |  102 |  23.95  |
| Ascend | DBNet-M3 | ICDAR2015 |  1.77 M  |   60  |   241  |   -     |  117    |   550    |  39.11 |
|  GPU   | DBNet-M3 | ICDAR2015 |  1.77 M  |  15      |   75    |  -    |  18.5  |  98 |  30.21 |

本模型受数据处理影响较大，在不同机器上性能数据波动较大，以上数据仅供参考。

以上数据是在

Ascend 910 32G 8卡；系统： Euler2.8；内存：756 G；ARM 96核 CPU；

GPU v100 PCIE 32G 8卡；系统： Ubuntu 18.04；内存：502 G；x86 72核 CPU。

机器上进行的实验。

## 快速入门

### 参数文件说明

```text
    config/config_base.yaml: 公共参数文件，数据集路径、优化器、训练策略等参数通常在该文件设置
    config/dbnet/*.yaml: backbone训练策略配置文件
    config/dbnet++/*.yaml: 带dcn的backbone训练策略配置文件
```

注意：dbnet/*.yaml和/dbnet++/*.yaml的参数会覆盖config_base.yaml的参数，用户可根据需求合理配置

单卡resnet18为backbone训练ICDAR2015数据为例：

1. 在config/config_base.yaml中配置训练、推理数据集路径，这个路径需要是绝对路径。

```text
load_mindrecord: True    # 是否将数据预处理成mindrecord格式，整个训练会快一点
mindrecord_path: "/path/dbnet/dataset"    # mindrecord保存的路径，需要是绝对路径，只需生成一次
train:
    img_dir: /data/ICDAR2015/ch4_training_images
    gt_dir: /data/ICDAR2015/ch4_training_localization_transcription_gt
eval:
    img_dir: /data/ICDAR2015/ch4_test_images
    gt_dir: /data/ICDAR2015/Challenge4_Test_Task1_GT
```

2. 在config/dbnet/config_resnet18_1p.yaml配置backbone_ckpt预训练路径

```text
backbone:
    backbone_ckpt: "/data/pretrained/resnet18-5c106cde.ckpt"
```

### 单卡训练：

```shell
bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID] [LOG_NAME](optional)

# 单卡训练 resnet18
bash scripts/run_standalone_train.sh config/dbnet/config_resnet18_1p.yaml 0 db_r18_1p

# 单卡训练 resnet50
bash scripts/run_standalone_train.sh config/dbnet/config_resnet50_1p.yaml 0 db_r50_1p

# 单卡训练 mobilenetv3 large
bash scripts/run_standalone_train.sh config/dbnet/config_mobilenetv3_1p.yaml 0 db_m3_1p
```

### 多卡训练：

```shell
# 1.8之前Ascend上使用for循环启动
bash scripts/run_distribution_train_ascend.sh [RANK_TABLE_FILE] [DEVICE_NUM] [CONFIG_PATH]

# 1.8之后Ascend和GPU一样可以用mpirun启动
bash scripts/run_distribution_train.sh [DEVICE_NUM] [CONFIG_PATH] [LOG_NAME](optional)

# 8卡训练 resnet18
bash scripts/run_distribution_train.sh 8 config/dbnet/config_resnet18_8p.yaml db_r18_8p

# 8卡训练 resnet50
bash scripts/run_distribution_train.sh 8 config/dbnet/config_resnet50_8p.yaml db_r50_8p

# 8卡训练 mobilenetv3 large
bash scripts/run_distribution_train.sh 8 config/dbnet/config_mobilenetv3_8p.yaml db_m3_8p
```

### 推理评估：

```shell
bash scripts/run_eval.sh [CONFIG_PATH] [CKPT_PATH] [DEVICE_ID] [LOG_NAME](optional)

# 推理resnet18
bash scripts/run_eval.sh config/dbnet/config_resnet18_1p.yaml your_ckpt_path 0 eval_r18
```

## 训练

### 单卡训练

```shell
bash scripts/run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID] [LOG_NAME](optional)
# CONFIG_PATH：配置文件路径
# DEVICE_ID：执行训练使用的卡号
# LOG_NAME: 日志和结果保存目录名，默认是 train
```

执行上述命令将在后台运行，您可以通过 [LOG_NAME].txt文件查看结果。

训练结束后，您可在[LOG_NAME]对应路径下找到checkpoint文件。

### 分布式训练

```shell
bash scripts/run_distribution_train.sh [DEVICE_NUM] [CONFIG_PATH] [LOG_NAME](optional)
# DEVICE_NUM：执行训练使用的卡数
# CONFIG_PATH：配置文件路径
# LOG_NAME: 日志和结果保存目录名，默认是 distribution_train
```

执行上述命令将在后台运行，您可以通过[LOG_NAME].txt文件查看结果。

## 断点训练

如果想使用断点训练功能，只需要在config文件resume_ckpt加入需要继续训练的ckpt路径训练即可。

### ModelArts 上训练

1. 在config文件中配置 ModelArts 参数：

- 设置 enable_modelarts=True
- 设置OBS数据集路径 data_url: <数据集在OBS中的路径>
- 设置OBS训练回传路径 train_url: <输出文件在OBS中的路径>

2. 按照[ModelArts教程](https://support.huaweicloud.com/modelarts/index.html)执行训练。

## 在线推理

### 推理评估

```shell
bash scripts/run_eval.sh [CONFIG_PATH] [CKPT_PATH] [DEVICE_ID] [LOG_NAME](optional)
# CONFIG_PATH：配置文件路径
# CKPT_PATH：checkpoint路径
```

执行上述python命令将在后台运行，您可以通过[LOG_NAME].txt文件查看结果。

## 离线推理

### 导出过程

```shell
python export.py --config_path=[CONFIG_PATH] --ckpt_path=[CKPT_PATH]
```

将在当前路径下找到对应MINDIR文件。

### 310推理

推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。

```shell
bash scripts/run_cpp_infer.sh [MINDIR_PATH] [CONFIG_PATH] [OUTPUT_DIR] [DEVICE_TARGET] [DEVICE_ID]
# MINDIR_PATH: export生成的MindIR文件路径
# CONFIG_PATH: 配置文件路径
# OUTPUT_DIR: 数据前处理和结果保存路径
# DEVICE_TARGET：可以是[Ascend, GPU, CPU]中一个，310推理选择Ascend
# DEVICE_ID：执行推理使用的卡号
```

## 免责声明

models仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore models中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过 Gitee 与我们联系。非常感谢您对这个社区的理解和贡献。

## 致谢

此版本DBNet借鉴了一些优秀的开源项目，包括

https://github.com/MhLiao/DB.git

https://gitee.com/yanan0122/dbnet-and-dbnet_pp-by-mind-spore.git

## FAQ

优先参考 [Models FAQ](https://gitee.com/mindspore/models#FAQ) 来查找一些常见的公共问题。

Q: 遇到内存不够或者线程数过多的WARNING怎么办?

A: 调整config文件中的`num_workers`： 并行数，`prefetch_size`：缓存队列长度， `max_rowsize`：一条数据在MB里最大内存占用，batchsize 16 最小设9。
一般CPU占用过多需要减小`num_workers`；内存占用过多需要减小`num_workers`，`prefetch_size`和`max_rowsize`。

Q: GPU环境上loss不收敛怎么办?

A: 将`config`文件中的`mix_precision`改成`False`。

Q: TotalText有接口为什么没有配置文件？

A: TotalText需要使用在SynthText数据集上预训练的参数进行finetune，目前暂未提供SynthText数据集上预训练的参数文件。
