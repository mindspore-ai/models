# UNet3+

<!-- TOC -->

- [UNet3+](#UNet3+)
- [UNet3+介绍](#UNet3+介绍)
- [模型结构](#模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [Mindir推理](#Mindir推理)
        - [导出模型](#导出模型)
        - [在Ascend310执行推理](#在Ascend310执行推理)
        - [onnx推理](#onnx推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## UNet3+介绍

UNET 3+:  A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION 利用了全尺度的跳跃连接 (skip connection) 和深度监督(deep supervisions)来完成医学图像语义分割的任务。全尺度的跳跃连接把来自不同尺度特征图中的高级语义与低级语义结合；而深度监督则从多尺度聚合的特征图中学习层次表示，特别适用于不同规模的器官。除了提高精度外，本文所提出的 UNet 3 + 还可以减少网络参数，提高计算效率。

[论文](https://arxiv.org/abs/2004.08790)：Huang H , Lin L , Tong R , et al. UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation[J]. arXiv, 2020.

## 模型结构

与UNet和UNet++相比，UNet 3+通过重新设计跳跃连接、利用多尺度的深度监督将多尺度特征结合起来，这使得它只需要比它们更少的参数，却可以产生更准确的位置感知和边界增强的分割图。

无论是U-Net中的直接连接还是U-Net ++中的密集嵌套连接，都缺乏从全尺度探索足够信息的能力，因此不能明确地得知器官的位置和边界。U-Net 3+ 中的每个解码器层都融合了来自编码器的较小和相同尺度的特征图以及来自解码器的较大尺度的特征图，它们捕获了全尺度下的细粒度语义和粗粒度语义。

## 数据集

数据集：[**LiTS2017**](<https://competitions.codalab.org/competitions/15595>)

Liver tumor Segmentation Challenge (LiTS，肝脏肿瘤病灶区 CT 图像分割挑战大赛) 数据集，包含来自全球各地的医院提供的对比增强过的CT图像。 共有训练集 131例，测试集 70例，其中测试集未公布标签。论文中从131例训练集中选出103例和28例分别用于训练和验证。数据集源格式为 'nii' 。

UNet3+ 处理的数据为RGB图像，故训练前应该将源数据预处理为图片。

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估：

```bash
###参数配置请修改 default_config.yaml 文件

#通过 python 命令行运行单卡训练脚本。
python train.py > log.txt 2>&1 &

#通过 bash 命令启动单卡训练。
bash ./scripts/run_train.sh [root path of code]
#训练日志将输出到 log.txt 文件

#Ascend多卡训练。
bash ./scripts/run_distribute_train.sh [root path of code] [rank size] [rank start id] [rank table file]

# 通过 python 命令行运行推理脚本。
# pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
python eval.py > eval_log.txt 2>&1 &

#通过 bash 命令启动推理。
bash ./scripts/run_eval.sh [root path of code]
#推理日志将输出到 eval_log.txt 文件
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```text
├── model_zoo
    ├── README.md                            // 所有模型的说明文件
    ├── UNet3+
        ├── README_CN.md                     // UNet3+ 的说明文件
        ├── ascend310_infer                  // 310推理主代码文件夹
        |   ├── CMakeLists.txt               // CMake设置文件
        |   ├── build.sh                     // 编译启动脚本
        |   ├── inc
        |       ├── utils.h                  // 工具类头文件
        |   ├── src
        |       ├── main.cc                  // 推理代码文件
        |       ├── utils.cc                 // 工具类文件
        ├── scripts
        │   ├──run_distribute_train.sh       // Ascend 8卡训练脚本
        │   ├──run_eval.sh                   // 推理启动脚本
        │   ├──run_train.sh                  // 训练启动脚本
        |   ├──run_infer_310.sh              // 310推理启动脚本
            |——run_eval_onnx.sh              // onnx推理启动脚本
        ├── src
        │   ├──config.py                     // 配置加载文件
        │   ├──dataset.py                    // 数据集处理
        │   ├──models.py                     // 模型结构
        │   ├──logger.py                     // 日志打印文件
        │   ├──util.py                       // 工具类
        ├── default_config.yaml              // 默认配置信息，包括训练、推理、模型冻结等
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 推理脚本
        ├── export.py                        // 将权重文件导出为 MINDIR 等格式的脚本
        ├── dataset_preprocess.py            // 数据集预处理脚本
        ├── postprocess.py                   // 310精度计算脚本
        ├── preprocess.py                    // 310预处理脚本
        |—— eval_onnx.py                     // onnx推理脚本

```

### 脚本参数

```text
模型训练、推理、冻结等操作的参数均在 default_config.yaml 文件中进行配置。
关键参数默认如下：
--aug: 是否启用数据增强，（1 for True, 0 for False）
--epochs: 训练轮数
--lr: 学习率
--batch_size: 批次大小
```

### 训练过程

#### 训练

- 数据集预处理

  进行网络训练和推理之前，您应该先进行数据集预处理。

  ```python
  ###参数配置请修改 default_config.yaml 文件，其中 source_path 指源 nii 格式数据集根目录，该目录下应该有 "CT" 和 “seg”
  #两个文件夹，分别指源数据及对于语义分割标注结果；dest_path 指您期望存储处理后的图片数据的目录，不存在会自动创建；buffer_path
  #指缓冲区目录，在处理完成后该文件夹会被递归删除。
  python dataset_preprocess.py
  ```

- Ascend处理器环境运行

  ```bash
  ###参数配置请修改 default_config.yaml 文件
  #通过 python 命令行运行单卡训练脚本。
  python train.py > log.txt 2>&1 &

  #通过 bash 命令启动单卡训练。
  bash ./scripts/run_train.sh [root path of code]
  #上述命令均会使脚本在后台运行，日志将输出到 log.txt，可通过查看该文件了解训练详情

  #Ascend多卡训练。
  bash ./scripts/run_distribute_train.sh [root path of code] [rank size] [rank start id] [rank table file]
  ```

  训练完成后，您可以在 output_path 参数指定的目录下找到保存的权重文件，训练过程中的部分 loss 收敛情况如下（8卡并行）：

  ```text
  # grep "epoch time:" log.txt
  epoch: 170 step: 960, loss is 0.51230466
  epoch time: 58413.158 ms, per step time: 60.847 ms
  epoch time: 58448.345 ms, per step time: 60.884 ms
  epoch time: 58446.879 ms, per step time: 60.882 ms
  epoch time: 58480.166 ms, per step time: 60.917 ms
  epoch time: 58409.484 ms, per step time: 60.843 ms
  epoch: 175 step: 960, loss is 0.50975895
  epoch time: 58429.310 ms, per step time: 60.864 ms
  epoch time: 58543.156 ms, per step time: 60.982 ms
  epoch time: 58455.628 ms, per step time: 60.891 ms
  epoch time: 58453.604 ms, per step time: 60.889 ms
  epoch time: 58422.367 ms, per step time: 60.857 ms
  epoch: 180 step: 960, loss is 0.51502335
  epoch time: 58416.837 ms, per step time: 60.851 ms
  [WARNING] SESSION(53798,fffed29421e0,python):2021-11-01-15:55:11.115.617 [mindspore/ccsrc/backend/session/ascend_session.cc:1380] SelectKernel] There are 42 node/nodes used reduce precision to selected the kernel!
  2021-11-01 15:56:54,111 :INFO: epoch: 180, Dice: 97.20967
  2021-11-01 15:56:56,486 :INFO: update best result: 97.20967
  2021-11-01 15:56:56,709 :INFO: update best checkpoint at: ./output/unet_2021-11-01_time_12_56_54/0_best_map.ckpt
  epoch time: 62822.634 ms, per step time: 65.440 ms
  2021-11-01 15:59:10,762 :INFO: epoch: 181, Dice: 97.1946
  epoch time: 66539.150 ms, per step time: 69.312 ms
  2021-11-01 16:01:30,357 :INFO: epoch: 182, Dice: 97.19583
  epoch time: 64837.935 ms, per step time: 67.540 ms
  2021-11-01 16:03:46,606 :INFO: epoch: 183, Dice: 97.33418
  2021-11-01 16:03:46,608 :INFO: update best result: 97.33418
  2021-11-01 16:03:46,828 :INFO: update best checkpoint at: ./output/unet_2021-11-01_time_12_56_54/0_best_map.ckpt
  epoch time: 65825.663 ms, per step time: 68.568 ms
  2021-11-01 16:06:07,652 :INFO: epoch: 184, Dice: 97.15482
  epoch: 185 step: 960, loss is 0.5108043
  epoch time: 62547.918 ms, per step time: 65.154 ms
  2021-11-01 16:08:26,350 :INFO: epoch: 185, Dice: 97.32324
  epoch time: 62356.042 ms, per step time: 64.954 ms
  2021-11-01 16:10:40,546 :INFO: epoch: 186, Dice: 97.008
  epoch time: 66353.477 ms, per step time: 69.118 ms
  2021-11-01 16:13:00,183 :INFO: epoch: 187, Dice: 97.37989
  2021-11-01 16:13:00,186 :INFO: update best result: 97.37989
  2021-11-01 16:13:00,408 :INFO: update best checkpoint at: ./output/unet_2021-11-01_time_12_56_54/0_best_map.ckpt
  ...
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于推理评估的权重文件路径是否正确。

- Ascend处理器环境运行

  ```bash
  ###参数配置请修改 default_config.yaml 文件
  # 通过 python 命令行运行推理脚本。
  # pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
  python eval.py > eval_log.txt 2>&1 &

  #通过 bash 命令启动推理。
  bash ./scripts/run_eval.sh [root path of code]
  #推理日志将输出到 eval_log.txt 文件
  ```

  运行完成后，您可以在 output_path 指定的目录下找到推理运行日志。

## Mindir推理

### [导出MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --device_target [DEVICE_TARGET]
```

- `ckpt_file`为必填项。
- `file_format` 必须在 ["AIR", "MINDIR","ONNX"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATA_PATH` 为测试集所在的路径。
- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为 'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### onnx推理

在执行推理前，onnx文件必须通过`export.py`脚本导出。

```shell
# onnx inference
bash run_eval_onnx.sh [DATASET_PATH] [ONNX_MODEL] [DEVICE_TARGET]
```

- `DATASET_PATH` 为测试集所在的路径
- `ONNX_MODEL` 为导出的onnx模型所在路径
- `DEVICE_TARGET` 必须在['Ascend','CPU','GPU']中选择

### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

## 模型描述

### 性能

#### 评估性能

UNet3+ on “LiTS2017 ”

| Parameters                 | UNet3+                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G             |
| uploaded Date              | 1/11/2021 (month/day/year)                                   |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | LiTS2017                                                     |
| Training Parameters        | epoch=200, batch_size=2, lr=3e-4, aug=1                      |
| Optimizer                  | Adam                                                         |
| Loss Function              | BCEDiceLoss                                                  |
| outputs                    | image with segmentation mask                                 |
| Loss                       | 0.5271476                                                    |
| Accuracy                   | 97.71%                                                       |
| Total time                 | 8p：2h44m (without validation)                               |
| Checkpoint for Fine tuning | 8p: 19.30MB(.ckpt file)                                      |
| Scripts                    | [UNet3+脚本](https://gitee.com/mindspore/models/tree/master/research/cv/UNet3+) |

## 随机情况说明

train.py 和 eval.py 中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
