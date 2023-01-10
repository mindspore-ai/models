# 目录

- [CycleGAN描述](#cyclegan描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练](#训练)
    - [评估](#评估)
    - [ONNX评估](#onnx评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend 310上推理](#在ascend310上推理)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

# [CycleGAN描述](#目录)

图到图转换是视觉和图像问题，目的在于使用配对图像作为训练集，并（让机器）学习从输入图像到输出图像的映射。但是，许多任务无法获得配对训练数据。CycleGAN不需要对训练数据进行配对，只需要提供不同域的图像，就可以成功地训练不同域之间的图像映射。CycleGAN共享两个生成器，每个生成器都有一个判别符。

[论文](https://arxiv.org/abs/1703.10593): Zhu J Y , Park T , Isola P , et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. 2017.

![CycleGAN Imgs](imgs/objects-transfiguration.jpg)

# [模型架构](#目录)

CycleGAN包含两个生成网络和两个判别网络。

# [数据集](#目录)

下载CycleGAN数据集并创建你自己的数据集。我们提供data/download_cyclegan_dataset.sh来下载数据集。

# [环境要求](#目录)

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend、GPU或CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

整体代码结构如下：

```markdown
.CycleGAN
├─ README.md                            # 关于CycleGAN的说明
├─ data
  └─ download_cyclegan_dataset.sh.py    # 下载数据集
├── scripts
  └─ run_train_ascend.sh                # 启动Ascend训练（单卡）
  └─ run_train_standalone_gpu.sh        # 启动GPU训练（单卡）
  └─ run_train_distributed_gpu.sh       # 启动GPU训练（8卡）
  └─ run_infer_310.sh                   # 启动Ascend 310推理
├─ imgs
  └─ objects-transfiguration.jpg        # CycleGAN图像
├─ Ascend310_infer
  ├─ src
    ├─ main.cc                         # Ascend 310推理源码
    └─ utils.cc                        # Ascend 310推理源码
  ├─ inc
    └─ utils.h                         # Ascend 310推理源码
  ├─ build.sh                          # Ascend 310推理源码
  ├─ CMakeLists.txt                    # Ascend 310推理程序的CMakeLists
  └─ fusion_switch.cfg                 # 使用BatchNorm2d代替InstanceNorm2d
├─ src
  ├─ __init__.py                       # 初始化文件
  ├─ dataset
    ├─ __init__.py                     # 初始化文件
    ├─ cyclegan_dataset.py             # 创建CycleGAN数据集
    └─ distributed_sampler.py          # 数据集迭代器
  ├─ models
    ├─ __init__.py                     # 初始化文件
    ├─ cycle_gan.py                    # CycleGAN模型定义
    ├─ losses.py                       # CycleGAN损失函数定义
    ├─ networks.py                     # CycleGAN子网定义
    ├─ resnet.py                       # ResNet生成网络
    └─ depth_resnet.py                 # 更好生成网络
  └─ utils
    ├─ __init__.py                     # 初始化文件
    ├─ args.py                         # 解析参数
    ├─ reporter.py                     # Reporter类
    └─ tools.py                        # CycleGAN工具
├─ eval.py                             # 生成图像，顺序A->B以及B->A
├─ train.py                            # 训练脚本
├─ export.py                           # 导出MindIR脚本
├─ preprocess.py                       # Ascend 310推理的数据预处理脚本
└─ postprocess.py                      # Ascend 310推理的数据后处理脚本
```

## [脚本参数](#目录)

train.py和config.py中的主要参数如下：

```python
"platform": Ascend       # 运行平台，支持GPU、Ascend和CPU。
"device_id": 0           # 设备ID，默认值为0。
"model": "resnet"        # 生成器模型。
"pool_size": 50          # 存储以前生成图像的缓冲区大小，默认值为50。
"lr_policy": "linear"    # 学习率策略，默认值为linear。
"image_size": 256        # 输入image_size，默认值为256。
"batch_size": 1          # batch_size，默认值为1。
"max_epoch": 200         # 训练的epoch大小，默认值为200。
"in_planes": 3           # 输入通道，默认值为3。
"ngf": 64                # 生成器模型过滤器数量，默认值为64。
"gl_num": 9              # 生成器模型残差块数量，默认值为9。
"ndf": 64                # 判别模型过滤器数量，默认值为64。
"dl_num": 3              # 判别模型残差块数量，默认值为9。
"outputs_dir": "outputs" # 模型保存在此处，默认值为./outputs。
"dataroot": None         # 图像路径（应具有子文件夹trainA、trainB、testA、testB等）。
"load_ckpt": False       # 是否加载预训练的ckpt。
"G_A_ckpt": None         # G_A的预训练检查点文件路径。
"G_B_ckpt": None         # G_B的预训练检查点文件路径。
"D_A_ckpt": None         # D_A的预训练检查点文件路径。
"D_B_ckpt": None         # D_B的预训练检查点文件路径。
```

## [训练](#目录)

- 使用默认参数在Ascend上运行

    ```bash
    bash scripts/run_train_ascend.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size建议为200
    ```

- 使用默认参数在GPU上运行

    ```bash
    bash scripts/run_train_standalone_gpu.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size建议为200
    ```

- 使用默认参数在GPU8卡运行

    ```bash
    bash scripts/run_train_distributed_gpu.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size建议为600
    ```

- 使用默认参数在CPU上运行

    ```bath
    python train.py --platform CPU --dataroot [DATA_PATH] --use_random False --max_epoch [EPOCH_SIZE] --print_iter 1 pool_size 0
    ```

## [评估](#目录)

```bash
python eval.py --platform [PLATFORM] --dataroot [DATA_PATH] --G_A_ckpt [G_A_CKPT] --G_B_ckpt [G_B_CKPT]
```

**注：您将在"./outputs_dir/predict"中获得以下结果。**

## [ONNX评估](#目录)

首先，导出模型：

```bash
python export.py --platform GPU --model ResNet --G_A_ckpt /path/to/GA.ckpt --G_B_ckpt /path/to/GB.ckpt --export_file_name /path/to/<prefix> --export_file_format ONNX
```

您将获得两个`.onnx`文件：`/path/to/<prefix>_AtoB.onnx`和`/path/to/<prefix>_BtoA.onnx`

接下来，使用与导出中相同的`export_file_name`运行ONNX eval：

```bash
python eval_onnx.py --platform [PLATFORM] --dataroot [DATA_PATH] --export_file_name /path/to/<prefix>
```

**注：您将在"./outputs_dir/predict"中获得以下结果。**

## [推理过程](#目录)

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

### [导出MindIR](#目录)

```bash
python export.py --G_A_ckpt [CKPT_PATH_A] --G_B_ckpt [CKPT_PATH_A] --export_batch_size 1 --export_file_name [FILE_NAME] --export_file_format [FILE_FORMAT]
```

### [在Ascend 310上推理](#目录)

在进行推理之前，必须通过`export.py`导出mindir文件。当前batch_Size只能设置为1。

```shell
# Ascend 310推理
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATA_MODE] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

- `DATA_PATH`：必填项，必须指定原始数据路径。
- `DATA_MODE`：CycleGAN的转换方向，其值为'AtoB'或'BtoA'。
- `NEED_PREPROCESS`：表示是否需要预处理，其值为'y'或'n'。
- `DEVICE_ID`：可选参数，默认值为0。

例如，在Ascend上：

```bash
bash ./scripts/run_infer_310.sh ./310_infer/CycleGAN_AtoB.mindir ./data/horse2zebra AtoB y Ascend 0
```

### [结果](#目录)

推理结果保存在当前路径中，您可以在infer_output_img文件中查看如下结果。

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

我们在Ascend上使用Depth Resnet生成器，在GPU上使用Resnet生成器。

| 参数                | 单卡Ascend/GPU                                          | 8卡GPU                                                     |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| 模型版本             | CycleGAN                                                    | CycleGAN                                                    |
| 资源                  | Ascend 910/NV SMX2 V100-32G                                 | NV SMX2 V100-32G x 8                                        |
| MindSpore版本         | 1.2                                                         | 1.2                                                         |
| 数据集                   | horse2zebra                                                 | horse2zebra                                                 |
| 训练参数       | epoch=200, steps=1334, batch_size=1, lr=0.0002              | epoch=600, steps=166, batch_size=8, lr=0.0002               |
| 优化器                 | Adam                                                        | Adam                                                        |
| 损失函数             | Mean Sqare Loss & L1 Loss                                   | Mean Sqare Loss & L1 Loss                                   |
| 输出                   | 概率                                                | 概率                                                |
| 速度                     | 单卡Ascend：123 ms/step; 单卡GPU：190 ms/step             | 190 ms/step                                                 |
| 总时长                | 单卡Ascend：9.6h; 单卡GPU：14.9h;                         | 5.7h                                                        |
| 微调检查点| 44M（.ckpt）                                           | 44M（.ckpt）                                           |

### 评估性能

| 参数         | 单卡Ascend/GPU          |
| ------------------- | --------------------------- |
| 模型版本      | CycleGAN                    |
| 资源           | Ascend 910/NV SMX2 V100-32G |
| MindSpore版本  | 1.2                         |
| 数据集            | horse2zebra                 |
| batch_size          | 1                           |
| 输出            | 转换后的图像         |

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
