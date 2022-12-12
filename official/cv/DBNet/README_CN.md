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

### 性能

| device | Model     | dataset   | Params(M) | PyNative train 1P bs=16 (ms/step) | PyNative train 8P bs=8 (ms/step) | PyNative infer(FPS)| Graph train 1P bs=16 (ms/step) | Graph train 8P bs=8 (ms/step) | Graph infer(FPS) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Ascend | DBNet-R18 | ICDAR2015 |  11.78 M  |  370     |   530   |    -     |   224    |   195   |   40.62   |
|  GPU   | DBNet-R18 | ICDAR2015 |  11.78 M  |  710    |   880     |   -    |  560  |   435    |   30.97   |
| Ascend | DBNet-R50 | ICDAR2015 |  24.28 M  |  524      |   680     |   -     |  273    |   220    |   33.88   |
|  GPU   | DBNet-R50 | ICDAR2015 |  24.28 M  |  935      |   1054    |  -    |  730  |  547    |   23.95   |

本模型受数据处理影响较大，在不同机器上性能数据波动较大，以上数据仅供参考。

以上数据是在

Ascend 910 32G 8卡；系统： Euler2.8；内存：756 G；ARM 96核 CPU；

GPU v100 PCIE 32G 8卡；系统： Ubuntu 18.04；内存：502 G；x86 72核 CPU。

机器上进行的实验。

## 快速入门

单卡训练：

```shell
bash run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID] [LOG_NAME](optional)
```

多卡训练：

```shell
bash run_distribution_train.sh [DEVICE_NUM] [CONFIG_PATH] [LOG_NAME](optional)
```

推理评估：

```shell
bash run_eval.sh [CONFIG_PATH] [CKPT_PATH] [DEVICE_ID] [LOG_NAME](optional)
```

如需修改device或者其他配置，请对应修改配置文件里的对应项。

## 训练

### 单卡训练

```shell
bash run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID] [LOG_NAME](optional)
# CONFIG_PATH：配置文件路径，默认使用Ascend，如需修改请在config文件里修改 device_target
# DEVICE_ID：执行训练使用的卡号
# LOG_NAME: 保存日志和输出文件夹的名字，默认是standalone_train
```

执行上述命令将在后台运行，您可以通过[LOG_NAME].txt文件查看结果。

训练结束后，您可在[LOG_NAME]对应路径下找到checkpoint文件。

### 分布式训练

```shell
bash run_distribution_train.sh [DEVICE_NUM] [CONFIG_PATH] [LOG_NAME](optional)
# DEVICE_NUM：执行训练使用的卡数
# CONFIG_PATH：配置文件路径，默认使用Ascend，如需修改请在config文件里修改 device_target
# LOG_NAME: 保存日志和输出文件夹的名字，默认是distribution_train
```

执行上述命令将在后台运行，您可以通过[LOG_NAME].txt文件查看结果。

### ModelArts 上训练

1. 在config文件中配置 ModelArts 参数：

- 设置 enable_modelarts=True
- 设置OBS数据集路径 data_url: <数据集在OBS中的路径>
- 设置OBS训练回传路径 train_url: <输出文件在OBS中的路径>

2. 按照[ModelArts教程](https://support.huaweicloud.com/modelarts/index.html)执行训练。

## 在线推理

### 推理评估

```shell
bash run_eval.sh [CONFIG_PATH] [CKPT_PATH] [DEVICE_ID] [LOG_NAME](optional)
# CONFIG_PATH：配置文件路径，默认使用Ascend，如需修改请在config文件里修改 device_target
# DEVICE_ID：执行推理使用的卡号
# LOG_NAME: 保存日志和输出文件夹的名字，默认是eval
```

执行上述python命令将在后台运行，您可以通过[LOG_NAME].txt文件查看结果。

## 离线推理

### 导出过程

```shell
python export.py --config_path=[CONFIG_PATH] --ckpt_path=[CKPT_PATH]
```

将在配置文件里`output_dir`对应路径下找到对应MINDIR文件。

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
