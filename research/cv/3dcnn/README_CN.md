# 目录

<!-- TOC -->

- [目录](#目录)
- [3DCNN描述](#3DCNN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [导出过程](#导出过程)
    - [推理过程](#推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 3DCNN描述

3DCNN是2018年提出的基于3D卷积神经网络，用于自动分割胶质瘤,并在BraTs 2017训练数据集上进行验证。由于胶质瘤的位置、结构和形状在不同患者之间存在显著差异，3DCNN模型通过从两个接受域尺度提取特征来获取多尺度上下文信息。为了充分利用肿瘤的结构，3DCNN将坏死和非增强肿瘤(NCR/NET)、瘤周水肿(ED)和GD-增强肿瘤(ET)的不同病变区域分层。此外，模型利用紧密连接的卷积块进一步提高性能，使用局部训练模式来训练模型，以缓解肿瘤区域和非肿瘤区域不平衡问题。

[论文](https://arxiv.org/pdf/1802.02427v2.pdf) ： Chen, Lele, et al. "MRI tumor segmentation with densely connected 3D CNN." Medical Imaging 2018: Image Processing. Vol. 10574. International Society for Optics and Photonics, 2018.

# 模型架构

3DCNN的总体网络架构如下：
[链接](https://arxiv.org/pdf/1802.02427v2.pdf)

# 数据集

使用的数据集：

- [BraTS 2017](<http://braintumorsegmentation.org/>) 是BraTS 2017挑战赛的训练数据集。数据集具有210名患者的MRI扫描图像，每名患者有FLAIR、T1、T1-CE和T2四种扫描数据。本模型只用使用BraTS 2017下HGG的数据集。
    - 训练集：168名患者MRI扫描图像
    - 测试集：42名患者MRI扫描图像
    - 注：数据集通过在src/n4correction.py中进行N4ITK偏差修正处理，得到了修正后的数据集[N4Correction BraTS 2017](<https://pan.baidu.com/s/1Sshq6-3uNxCb6OzMwWobkg/>) (提取码：1111)。本模型使用数据集是通过N4ITK偏差修正处理后的数据集。

特别说明：

BraTS 2017原始数据集的文件目录结构如下所示：

```text
├── MICCAI_BraTS17_Data_Training
    ├── HGG
        ├─ Brats17_2013_2_1
            ├─ Brats17_2013_2_1_flair.nii.gz
            ├─ Brats17_2013_2_1_t1.nii.gz
            ├─ Brats17_2013_2_1_t1ce.nii.gz
            ├─ Brats17_2013_2_1_t2.nii.gz
            └─ Brats17_2013_2_1_seg.nii.gz
        ├─ ...
```

经过N4ITK修正后数据集的文件目录结构如下所示：

```text
├── MICCAI_BraTS17_Data_Training
    ├── HGG
        ├─ Brats17_2013_2_1
            ├─ Brats17_2013_2_1_flair.nii.gz
            ├─ Brats17_2013_2_1_flair_corrected.nii.gz
            ├─ Brats17_2013_2_1_t2.nii.gz
            ├─ Brats17_2013_2_1_t2_corrected.nii.gz
            ├─ Brats17_2013_2_1_t1.nii.gz
            ├─ Brats17_2013_2_1_t1_corrected.nii.gz
            ├─ Brats17_2013_2_1_t1ce.nii.gz
            ├─ Brats17_2013_2_1_t1ce_corrected.nii.gz
            └─ Brats17_2013_2_1_seg.nii.gz
        ├─ ...
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

    ```bash
    # 分布式训练
    用法：bash run_distribute_train.sh [DATA_PATH] [TRAIN_PATH] [RANK_TABLE_FILE]

    # 单机训练
    用法：bash run_standalone_train.sh [DATA_PATH] [TRAIN_PATH] [DEVICE_ID]

    # 运行评估示例
    用法：bash run_eval.sh [DATA_PATH] [TEST_PATH] [CKPT_PATH] [DEVICE_ID]
    ```

- GPU处理器环境运行

  ```bash
  # MindRecord数据集生成 (本次结果在SAMPLE_NUM=400下得到)
  用法：bash convert_dataset.sh [DATA_PATH] [TRAIN_PATH] [MINDRECORD_PATH] [SAMPLE_NUM]

  # 分布式训练
  用法：bash run_distribute_train_gpu.sh [MINDRECORD_PATH] [CONFIG_PATH]

  # 单机训练
  用法：bash run_standalone_train_gpu.sh [MINDRECORD_PATH] [CONFIG_PATH] [DEVICE_ID]

  # 运行评估示例
  用法：bash run_eval_gpu.sh [DATA_PATH] [TEST_PATH] [CKPT_PATH] [CONFIG_PATH] [DEVICE_ID]
  ```

# 脚本说明

## 脚本及样例代码

```bash
.
└── 3dcnn
    ├── ascend_310_infer                  # 310推理
    ├── README.md                         # 所有模型相关说明
    ├── scripts
    │   ├── run_310_infer.sh              # 用于310推理的shell脚本
    │   ├── run_distribute_train.sh       # 启动Ascend分布式训练（8卡）
    │   ├── run_eval.sh                   # 启动Ascend评估
    │   ├── run_standalone_train.sh       # 启动Ascend单机训练（单卡）
    │   ├── convert_dataset.sh            # 生成mindrecord格式的数据集
    │   ├── run_distribute_train_gpu.sh   # 启动GPU分布式训练（8卡）
    │   ├── run_standalone_train_gpu.sh   # 启动GPU单机训练（单卡）
    │   ├── run_eval_gpu.sh               # 启动GPU评估
    │   ├── run_onnx_eval.sh              # ONNX推理shell脚本
    ├── src
    │   ├── config.py                     # yaml文件解析
    │   ├── dataset.py                    # 创建数据集
    │   ├── initializer.py                # glorot_normal权重初始化
    │   ├── loss.py                       # 损失定义
    │   ├── lr_schedule.py                # 动态学习率生成器
    │   ├── models.py                     # 3DCNN架构
    │   ├── n4correction.py               # N4ITK偏差修正数据集
    │   ├── mindrecord_generator.py       # mindrecord格式数据集生成
    │   ├── test.txt                      # 测试数据集
    │   ├── train.txt                     # 训练数据集
    ├── train.py                          # 训练脚本
    ├── eval.py                           # 评估脚本
    ├── eval_onnx.py                      # ONNX评估脚本
    ├── export.py                         # 推理模型导出脚本
    ├── preprocess.py                     # 310推理数据预处理
    ├── postprocess.py                    # 310推理数据后处理
    ├── config.yaml                       # Ascend训练参数配置
    ├── config_gpu.yaml                   # GPU训练参数配置
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置3DCNN和BraTS 2017数据集。

  ```python
  'data_path':Path('~your_path/BraTS17/HGG/')    # 训练和评估数据集的绝对全路径
  'mindrecord_path': Path                        # mindrecord数据集绝对全路径
  'train_path': "./src/train.txt"                # 训练集路径
  'test_path': "./src/test.txt"                  # 测试集路径
  'ckpt_path':'./dense24-5_4200.ckpt'            # checkpoint文件保存的绝对全路径
  'correction': 'True'                           # 是否修正数据集
  'model': 'dense24'                             # 模型名字
  'use_optimizer': 'SGD'                         # 使用的优化器
  'use_dynamic_lr': True                         # 是否使用动态学习率
  'use_loss_scale': False                        # 是否使用loss scale
  'use_mindrecord': False                        # 是否使用mindrecord格式的数据集
  'epoch_size':5                                 # 总计训练epoch数
  'batch_size':2                                 # 训练批次大小
  'num_classes':5                                # 数据集类数
  'offset_width':12                              # 评估模型滑动窗口的图像宽度
  'offset_height':12                             # 评估模型滑动窗口的图像高度
  'offset_channel':12                            # 评估模型滑动窗口的图像通道数
  'width_size':38                                # 输入到模型的图像宽度
  'height_size':38                               # 输入到模型的图像高度
  'channel_size':38                              # 输入到模型的图像通道数
  'pred_size':12                                 # 模型输出的图像宽度，高度，通道数
  'lr':0.005                                     # 学习率
  'loss_scale': 128.0                            # loss_scale的值
  'momentum':0.9                                 # 动量
  'weight_decay': 0.001                          # 权重衰减值
  'warmup_step':6720                             # 热身步数
  'warmup_ratio':0                               # 热身率
  'keep_checkpoint_max':5                        # 只保存最后一个keep_checkpoint_max检查点
  'device_target':'Ascend'                       # 运行设备
  'device_id':0                                  # 用于训练或评估数据集的设备ID使用run_distribute_train.sh进行分布式训练时可以忽略。
  ```

更多配置细节请参考脚本`config.yaml/config_gpu.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

    ```bash
    # 分布式训练
    用法：bash run_distribute_train.sh [DATA_PATH] [TRAIN_PATH] [RANK_TABLE_FILE]

    # 单机训练
    用法：bash run_standalone_train.sh [DATA_PATH] [TRAIN_PATH] [DEVICE_ID]
    ```

- GPU处理器环境运行

    ```bash
    # MindRecord数据集生成 (本次结果在SAMPLE_NUM=400下得到)
    用法：bash convert_dataset.sh [DATA_PATH] [TRAIN_PATH] [MINDRECORD_PATH] [SAMPLE_NUM]

    # 分布式训练（默认八卡）
    用法：bash run_distribute_train_gpu.sh [MINDRECORD_PATH] [CONFIG_PATH]

    # 单机训练
    用法：bash run_standalone_train_gpu.sh [MINDRECORD_PATH] [CONFIG_PATH] [DEVICE_ID]
    ```

    分布式训练需要提前创建JSON格式的HCCL配置文件。

    具体操作，参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) 中的说明。

### 结果

- 使用修正后的BraTS 2017数据集3DCNN

    ```bash
    # 分布式训练结果（8P）
    epoch: 1 step: 4200, loss is 0.432982
    epoch time: 1591579.692 ms, per step time: 378.948 ms
    epoch: 2 step: 4200, loss is 0.6555363
    epoch time: 1446606.528 ms, per step time: 344.430 ms
    epoch: 3 step: 4200, loss is 0.23770252
    epoch time: 1446406.834 ms, per step time: 344.383 ms
    epoch: 4 step: 4200, loss is 0.51601326
    epoch time: 1446402.737 ms, per step time: 344.382 ms
    epoch: 5 step: 4200, loss is 0.07842843
    epoch time: 1446404.419 ms, per step time: 344.382 ms
    ```

## 评估过程

### 评估

- Ascend处理器环境运行

```bash
# 分布式训练
用法：bash run_eval.sh [DATA_PATH] [TEST_PATH] [CKPT_PATH] [DEVICE_ID]
```

- GPU处理器环境运行

```bash
# 分布式训练
用法：bash run_eval_gpu.sh [DATA_PATH] [TEST_PATH] [CKPT_PATH] [CONFIG_PATH] [DEVICE_ID]
```

### 结果

- Ascend处理器环境运行结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```bash
mean dice whole:
[0.99771376 0.81746066]
mean dice core:
[0.99902476 0.77739729]
mean dice enhance:
[0.99929174 0.75328799]
```

- GPU处理器环境运行结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```bash
mean dice whole:
[0.99780174 0.82293876]
mean dice core:
[0.99905320 0.78296646]
mean dice enhance:
[0.99930122 0.75698223]
```

## 导出过程

### 导出

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file 是必需的，EXPORT_FORMAT 必须在 ["AIR", "MINDIR", "ONNX"]中进行选择。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在执行推理之前，需要通过export.py导出mindir文件

- 在昇腾310上使用修正后的BraST 2017训练数据集进行推理

  执行推理的命令如下所示，其中`MINDIR_PATH`是mindir文件路径；`DATASET_PATH`是数据集路径；`TEST_PATH`是推理数据路径；`DEVICE_ID`可选，默认值为0。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 推理
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [TEST_PATH] [DEVICE_ID]
  ```

### ONNX推理

在执行推理之前，需要通过export.py导出onnx文件

- 在GPU上使用修正后的BraST 2017训练数据集进行推理

  执行推理的命令如下所示，其中`DATA_PATH`是数据集路径；`TEST_PATH`是推理数据路径；`ONNX_PATH`是ONNX文件路径；`DEVICE_ID`可选，默认值为0。

  ```shell
  # ONNX 推理
  bash run_onnx_eval.sh [DATA_PATH] [TEST_PATH] [ONNX_PATH] [CONFIG_PATH] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过eval_onnx.log文件查看结果。

# 模型描述

## 性能

### 评估性能

#### BraTS2017上的3DCNN

| 参数          | Ascend 910                                                   | GPU V100                                                     |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 模型版本      | 3DCNN                                                        | 3DCNN                                                        |
| 资源          | Ascend 910；CPU：2.60GHz，192核；内存：755G                  | Telsa GPU V100                                               |
| 上传日期      | 2021-10-16                                                   | 2021-11-11                                                   |
| MindSpore版本 | r1.3                                                         | r1.5                                                         |
| 数据集        | BraTS 2017                                                   | BraTS 2017                                                   |
| 训练参数      | epoch=5, steps per epoch=4200, batch_size=2                  | epoch=5, steps per epoch=8400, batch_size=2                  |
| 优化器        | SGD                                                          | Adam                                                         |
| 损失函数      | Softmax交叉熵                                                | Softmax交叉熵                                                |
| 输出          | 概率                                                         | 概率                                                         |
| 损失          | 0.07842843                                                   | 0.08314727                                                   |
| 速度          | 345毫秒/步(8卡)                                              | 339毫秒/步(8卡)                                              |
| 总时长        | 2.9小时                                                      | 3.9小时                                                      |
| 微调检查点    | 9.1M (.ckpt文件)                                             | 9.1M (.ckpt文件)                                             |
| 脚本          | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/3dcnn) | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/3dcnn) |

### 推理性能

#### BraTS2017上的3DCNN

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | 3DCNN                |
| 资源            |  Ascend 310；系统 Eulerosv2r8                  |
| 上传日期       | 2021-11-11 |
| MindSpore 版本   | r1.5                       |
| 数据集             | BraTS 2017     |
| 输出             | 概率                 |
| 准确性            | whole: 81.74%; core: 77.73%; enhance: 75.32% |
| 推理模型 | 3.26M (.mindir文件)         |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子，mindrecord_generator.py使用了随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
