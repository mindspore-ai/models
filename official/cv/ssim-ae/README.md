# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [SSIM-AE描述](#SSIM-AE描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
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
  - [推理过程](#推理过程)
    - [推理](#推理)
- [模型描述](#模型描述)
  - [性能](#性能)
      - [评估性能](#评估性能)
      - [texture图像上的SSIM-AE](#texture图像上的SSIM-AE)
      - [MVTec AD图像上的SSIM-AE](#MVTecAD图像上的SSIM-AE)
      - [推理性能](#推理性能)
          - [texture图像上的SSIM-AE推理](#texture图像上的SSIM-AE推理)
          - [MVTec AD图像上的SSIM-AE推理](#MVTecAD图像上的SSIM-AE)
  - [使用流程](#使用流程)
    - [推理](#推理-1)
    - [继续训练预训练模型](#继续训练预训练模型)
    - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# SSIM-AE描述

自动编码器已经成为无监督的缺陷检测的常用方法。一般通过自动编码器重建图像，然后对重建图和原图在像素级别上进行比较，距离大于一定的阈值则认为是缺陷。但基于距离的损失函数在图像中某些边缘区域重建不准确时造成了较大的误差，且当缺陷在强度上大致一致而在结构上区别较大时，基于距离的损失函数无法检测出这些缺陷。通过采用更复杂的编码器也无法改善这些问题，基于此，该论文采用结构相似性感知损失函数来检查像素之间的相互依赖性，并同时考虑亮度，对比度和结构信息。

[论文](https://www.researchgate.net/publication/326222902)：Improving Unsupervised Defect Segmentation by Applying Structural Similarity To Autoencoders

# 模型架构

SSIM-AE由一连串对称的卷积层和反卷积层组成，具体的网络结构如下表：

| Layer      | Output Size | Kernel | Stride | Padding |
| ---------- | ----------- | :----: | ------ | ------- |
| Input      | 128*128\*1  |        |        |         |
| Conv1      | 64\*64*32   |  4*4   | 2      | 1       |
| Conv2      | 32\*32*32   |  4*4   | 2      | 1       |
| Conv3      | 32\*32*32   |  3*3   | 1      | 1       |
| Conv4      | 16\*16*64   |  4*4   | 2      | 1       |
| Conv5      | 16\*16*64   |  3*3   | 1      | 1       |
| Conv6      | 8\*8*128    |  4*4   | 2      | 1       |
| Conv7      | 8\*8*64     |  3*3   | 1      | 1       |
| Conv8      | 8\*8*32     |  3*3   | 1      | 1       |
| Conv9      | 1*1\*d      |  8*8   | 1      | 0       |
| ConvTrans1 | 8\*8*32     |  8*8   | 1      | 0       |
| Conv10     | 8\*8*64     |  3*3   | 1      | 1       |
| Conv11     | 8\*8*128    |  3*3   | 1      | 1       |
| ConvTrans2 | 16\*16*64   |  4*4   | 2      | 1       |
| Conv12     | 16\*16*64   |  3*3   | 1      | 1       |
| ConvTrans3 | 32\*32*32   |  4*4   | 2      | 1       |
| Conv13     | 32\*32*32   |  3*3   | 1      | 1       |
| ConvTrans4 | 64\*64*32   |  4*4   | 2      | 1       |
| ConvTrans5 | 128*128\*1  |  4*4   | 2      | 1       |

# 数据集

本实现使用了论文中的两个[编织品纹理](ftp://ftp.softronics.ch/visapp_textures/textures.zip)数据集及[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)数据集中的metal_nut类和cable类。

编织品纹理数据集

- 说明：编织品纹理数据集包含两种纹理数据：texture1和texture2。
- 数据集大小：45MB，共2个类、300张灰度图像
    - 训练集：30MB，共200张图像
    - 测试集：15MB，共100张图像
- 数据格式：二进制文件（PNG）
  目录结构如下

```bash
.
└─texture_1
  └─train
    └─000.png
    └─001.png
    ...
  └─test
    └─000.png
    └─001.png
    ...
  └─ground_truth
    └─defective
      └─000.png
      └─001.png
      ...
```

MVTec AD数据集

- 说明：
- 数据集大小：4.9G，共15个类、5354张高分辨图像
    - 训练集：3.4G，共3629张图像
    - 测试集：1.5G，共1725张图像
- 数据格式：二进制文件（PNG）和RGB
  MVTec AD其中一个类的目录结构如下：

```bash
.
└─metal_nut
  └─train
    └─good
      └─000.png
      └─001.png
      ...
  └─test
    └─bent
      └─000.png
      └─001.png
       ...
    └─color
      └─000.png
      └─001.png
       ...
    ...
  └─ground_truth
    └─bent
      └─000_mask.png
      └─001_mask.png
      ...
    └─color
      └─000_mask.png
      └─001_mask.png
      ...
    ...
```

- 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
  - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行在texture_1上的训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 训练:在config.yaml中设置参数如下
  grayscale: true
  do_aug: true
  data_augment:
    im_resize: 256
    crop_size: 128
    rotate_angle: 45
    p_horizontal_flip: 0.3
    p_vertical_flip: 0.3
  z_dim: 100
  # 推理:在config.yaml中设置checkpoint路径
  chcekpoint_path: ./checkpoints/carpet/ssim_autocoder_43-200_62.ckpt
  ```

  ```python
  # 运行训练示例
  python train.py --dataset_path=./texture_1
  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh [DATASET_NAME] [RANK_TABLE_FILE][DEVICE_NUM]
  # example: bash scripts/run_train.sh texture_1 ./hccl_8p_01234567_127.0.0.1.json 8
  # 仅支持四卡或八卡同时训练，DEVICE_NUM只接受4或8作为输入
  # 训练将在后台运行，你可以在./train_parallel0下找到0号卡的日志文件(其他卡同理)
  # 你也可以输入命令 tail -f ./train_parallel0/log实时更新并查看日志
  # 910运行推理示例
  # 先在config.yaml中设置好checkpoint_path，texture数据集时设置grayscale: true, im_resize: 256,
  # crop_size: 128, z_dim: 100, ssim_threshold: 0.5, stride: 32
  python eval.py --dataset_path=./texture_1
  # 310运行推理示例
  # texture数据集时设置grayscale: true, im_resize: 256, crop_size: 128, ssim_threshold: 0.5, stride: 32
  bash scripts/run_infer_310.sh ../texture1.mindir ../texture_1 n 0
  ```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。

请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/r1.5/utils/hccl_tools.>

- CPU处理器环境运行

  ```yaml
  # 训练:在config.yaml中设置参数如下
  grayscale: true
  do_aug: true
  data_augment:
    im_resize: 256
    crop_size: 128
    rotate_angle: 45
    p_horizontal_flip: 0.3
    p_vertical_flip: 0.3
  z_dim: 100
  # 运行训练示例
  python train.py --dataset_path=./texture_1 --device_target=CPU
  # 运行评估示例
  # 先在config.yaml中设置好checkpoint_path，texture数据集时设置grayscale: true, im_resize: 256,
  # crop_size: 128, z_dim: 100, ssim_threshold: 0.5, stride: 32
  python eval.py --dataset_path=./texture_1 --device_target=CPU
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 texture1数据集

      ```python
      # (1) 在网页上设置代码目录为"/[桶名称]/ssim-ae"，config.yaml的设置参考上文
      # (2) 在网页上设置启动文件为"/[桶名称]/ssim-ae/train.py"
      # (3) 上传texture_1数据集到obs桶上(数据集目录结构参考上文)
      # (4) 在网页上设置数据存储位置为"/[桶名称]/ssim-ae/texture_1"
      # (5) 在网页上设置训练输出位置为"/[桶名称]/[你期望的输出路径]"
      # (6) 在网页上设置如下的运行参数
      #     distribute = true
      #     model_arts = ture
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 texture1数据集

      ```python
      # (1) 在网页上设置 代码目录为"/[桶名称]/ssim-ae"
      # (2) 在网页上设置 启动文件为"/[桶名称]/ssim-ae/train.py"
      # (3) 在网页上设置 数据存储位置为"/[桶名称]/ssim-ae/texture_1"
      # (4) 在网页上设置 "model_arts = ture"
      # (5) config.yaml 文件中设置 "checkpoint_url='s3://[桶名称]/ssim-ae/model.ckpt'"
      #     config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/model.ckpt'"
      #     config.yaml 文件中设置  grayscale: true, im_resize: 256, crop_size: 128,
      #                           z_dim: 100, ssim_threshold: 0.5
      # (6) 上传你的预训练模型到 S3 桶上
      # (7) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (8) 在网页上设置"训练输出文件路径"等
      # (9) 创建训练作业
      ```

您也可以将`$dataset_path`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                       // 所有模型相关说明
    ├── ssim-ae
        ├── README.md                   // ssim-ae相关说明
        ├── ascend310_infer             // 实现310推理源代码
        ├── scripts
        │   ├──run_train.sh             // 分布式到Ascend的shell脚本
        │   ├──run_infer_310.sh         // Ascend推理shell脚本
        ├── src
        │   ├──dataset.py                // 创建数据集
        │   ├──network.py                // ssim-ae架构
        │   ├──config.yaml               // 参数配置
        ├── train.py                     // 训练脚本
        ├── eval.py                      // 评估脚本
        ├── export.py                    // 将checkpoint文件导出到air/mindir
        ├── postprogress.py              // 310推理后处理
        ├── preprogress.py               // 310推理前处理
```

## 脚本参数

在config.yaml中可以同时配置训练参数和评估参数。

- 配置SSIM-AE和texture数据集。

  ```yaml
  grayscale: true                    # 是否为灰度图
  do_aug: true                       # 是否需要数据增强
  data_augment:
    augment_num: 10000               # 数据增强后得到的图片的数量
    im_resize: 266                   # 在数据增强之前图片缩放的大小
    crop_size: 256                   # 输入网络的图片大小
    rotate_angle: 0                  # 旋转的角度
    p_horizontal_flip: 0.3           # 水平翻转的概率
    p_vertical_flip: 0.3             # 垂直翻转的概率
  z_dim: 500                         # encoder最后一层输出的频道数
  epochs: 300                        # 训练模型的轮次数
  batch_size: 128                    # 训练的批处理大小
  lr: 2.0e-4                         # 训练的学习率
  decay: 1.0e-5                      # Adam优化器的权重衰减
  flc: 32                            # 第一个隐藏层的通道数
  stride: 32                         # 步长
  ssim_threshold: 0.5                # 设置阈值
  percent: 98                        # 取阈值的百分位数
  checkpoint_path:                   # checkpoint文件保存的绝对全路径
  checkpoint_url:                    # obs桶内checkpoint文件存放路径
  ckpt_file:                         # 容器内checkpoint文件路径
  ```

## 训练过程

### 训练

- Ascend处理器单卡运行

  ```bash
  # 请先在config.yaml中设置好参数
  python train.py --dataset_path=[DATASET_PATH]
  ```

- Ascend处理器多卡运行

  ```bash
  # 请先在config.yaml中设置好参数
  bash scripts/run_distribute_train.sh [DATASET_NAME] [RANK_TABLE_FILE] [DEVICE_NUM]
  ```

- CPU处理器运行

  ```bash
  # 请先在config.yaml中设置好参数
  python train.py --dataset_path=[DATASET_PATH] --device_target=CPU
  ```

  训练结束后，您可在`./result/checkpoint`下找到检查点文件。

### ModelArt训练

- 在 ModelArts 上使用8卡训练

  ```python
  # (1) 在网页上设置代码目录为"/[桶名称]/ssim-ae"，config.yaml的设置参考上文
  # (2) 在网页上设置启动文件为"/[桶名称]/ssim-ae/train.py"
  # (3) 上传texture_1数据集到obs桶上(数据集目录结构参考上文)
  # (4) 在网页上设置数据存储位置为"/[桶名称]/ssim-ae/[数据集名称]"
  # (5) 在网页上设置训练输出位置为"/[桶名称]/[你期望的输出路径]"
  # (6) 在网页上设置如下的运行参数
  #     distribute = true
  #     model_arts = ture
  # (7) 创建训练作业
  ```

- 在 ModelArts 上使用单卡训练

  ```python
  # (1) 在网页上设置代码目录为"/[桶名称]/ssim-ae"，config.yaml的设置参考上文
  # (2) 在网页上设置启动文件为"/[桶名称]/ssim-ae/train.py"
  # (3) 上传texture_1数据集到obs桶上(数据集目录结构参考上文)
  # (4) 在网页上设置数据存储位置为"/[桶名称]/ssim-ae/[数据集名称]"
  # (5) 在网页上设置训练输出位置为"/[桶名称]/[你期望的输出路径]"
  # (6) 在网页上设置如下的运行参数
  #     model_arts = ture
  # (7) 创建训练作业
  ```

训练结束后，您可在`/[桶名称]/result/checkpoint`下找到检查点文件。

## 推理过程

### Ascend910处理器环境推理

- 在Ascend910环境运行时评估texture1数据集

  在运行以下命令之前，请检查用于推理的检查点路径。请将检查点路径设置为绝对全路径，例如`username/ssim-ae/ssim_autocoder_22-257_8.ckpt`。

  ```bash
  # 在config.yaml中设置好checkpoint_path，texture数据集时设置grayscale: true, im_resize: 256,
  # crop_size: 128, z_dim: 100, ssim_threshold: 0.5, stride: 32
  python eval.py --dataset_path=./texture_1
  OR
  bash run_eval.sh [DATASET] [CHECKPOINT]
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如`username/ssim-ae/ssim_autocoder_22-257_8.ckpt`。测试数据集的准确性如下：

  ```python
  # texture1数据集 下线单卡
  AUC: 0.9652512237942968
  ```

- 在Ascend910环境运行时评估MVTec AD数据集

  以`metel_nut`数据集为例：

  ```python
  # 先在config.yaml中设置好checkpoint_path，设置grayscale: false, im_resize: 266,
  # crop_size: 256, z_dim: 500, ssim_threshold为空, percent: 75, stride: 32
  python eval.py --dataset_path=./metal_nut
  OR
  bash run_eval.sh [DATASET] [CHECKPOINT]
  ```

  测试数据集的输出指标如下：

  ```python
  # metel_nut数据集 线下单卡
  ssim_threshold: 0.262414
  ok: 0.5777573895957538
  nok: 0.7690432884405635
  acc: 0.741026847593246
  ```

### Ascend310处理器环境推理

   在推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用texture1数据集进行推理

  在执行下面的命令之前，我们需要先修改配置文件。修改的项包括grayscale、ssim_threshold、im_resize、crop_size、stride。
  使用texture数据集时设置 grayscale: true, im_resize: 256, crop_size: 128, ssim_threshold: 0.5, stride =32。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```python
  # Ascend310 inference
  # bash run_infer_310.sh [MINDIR_PATH][DATASET_PATH][NEEED_PROCESS] [DEVICE_ID]
  bash run_infer_310.sh ../texture1.mindir ../datasets/texture1/ n 2
  after allreduce eval: AUC: 0.9652486992579182
  ```

- 在昇腾310环境运行时评估MVTec AD数据集

  以`metel_nut`数据集为例：在config.yaml中设置grayscale: false, im_resize: 266, crop_size: 256, z_dim: 500, ssim_threshold: 0.262414, percent: 75, stride: 32。如使用cable数据集，其他配置一样，需要修改ssim_threthold: 0.402691。

  ```python
  bash run_infer_310.sh [MINDIR_PATH][DATASET_PATH][NEEED_PROCESS] [DEVICE_ID]
  # 例子：bash run_infer_310.sh ../metal.mindir ../datasets/metal_nut n
  ```

  测试数据集的输出指标如下：

  ```python
  # metel_nut数据集
  ok: 0.5780363268113555
  nok: 0.7689179562236865
  Acc: 0.7409607261739751
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件config.yaml，修改的项包括grayscale、z_dim和crop_size。
使用texture数据集需设置 ： grayscale: true, zdim: 100, crop_size: 128;
使用MVTec数据集需设置 ： grayscale: false, zdim: 500, crop_size: 256。

```shell
python export.py --ckpt_path [CKPT_PATH] --file_name [NAME]
```

# 模型描述

## 性能

### 训练性能

#### texture_1上的SSIM-AE

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | SSIM-AE                                                      |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8     |
| 上传日期      | 2021-12-30                                                   |
| MindSpore版本 | 1.5.0                                                        |
| 数据集        | texture_1                                                    |
| 训练参数      | epoch=200, steps=78, batch_size = 128, lr=2.0e-4             |
| 优化器        | Adam                                                         |
| 损失函数      | SSIM                                                         |
| 输出          | 概率                                                         |
| 损失          | 单卡：0.227；八卡：0.287                                     |
| 速度          | 单卡：498毫秒/步;  八卡：543毫秒/步(均为Model_Arts)          |
| 总时长        | 单卡：144分钟;  八卡：52分钟(均为Model_Arts)                 |
| 精度          | AUC 单卡：0.939；八卡：0.928(均为Model_Arts)                 |
| 参数(M)       | 1M                                                           |
| 微调检查点    | 12.2M (.ckpt文件)                                            |
| 脚本          | [ssim-ae脚本](https://gitee.com/mindspore/models/tree/master/official/cv/ssim-ae) |

#### MVTec AD(metal_nut)图像上的SSIM-AE

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | SSIM-AE                                                      |
| 资源          | Ascend 910；CPU 2.60GHz，56核；内存 314G；系统 Euler2.8      |
| 上传日期      | 2021-12-30                                                   |
| MindSpore版本 | 1.5.0                                                        |
| 数据集        | MVTec AD(metal_nut)                                          |
| 训练参数      | epoch=300, steps=78, batch_size=128, lr=2.0e-4               |
| 优化器        | Adam                                                         |
| 损失函数      | SSIM                                                         |
| 输出          | 概率                                                         |
| 损失          | 0.231                                                        |
| 速度          | 单卡：2476毫秒/步;  八卡：1994毫秒/步(均为Model_Arts)        |
| 总时长        | 单卡：654分钟;  八卡：114分钟(均为Model_Arts)                |
| 精度(ok)      | 单卡：0.939；八卡：0.928(均为Model_Arts)                     |
| 精度(nok)     | 单卡：0.939；八卡：0.928(均为Model_Arts)                     |
| 参数(M)       | 2.59M                                                        |
| 微调检查点    | 31.15M (.ckpt文件)                                           |
| 脚本          | [ssim-ae脚本](https://gitee.com/mindspore/models/tree/master/official/cv/ssim-ae) |

### 推理性能

#### texture_1图像上的SSIM-AE推理

| 参数           | Ascend                          |
| -------------- | ------------------------------- |
| 模型版本       | SSIM-AE                         |
| 资源           | Ascend 910；系统 Euler2.8       |
| 上传日期       | 2021-12-30                      |
| MindSpore 版本 | 1.5.0                           |
| 数据集         | texture_1                       |
| batch_size     | 128                             |
| 输出           | 概率                            |
| 准确性         | 单卡：AUC 0.965；8卡：AUC 0.957 |

#### MVTec AD图像(metal_nut)上的SSIM-AE推理

| 参数          | Ascend                               |
| ------------- | ------------------------------------ |
| 模型版本      | SSIM-AE                              |
| 资源          | Ascend 910；系统 Euler2.8            |
| 上传日期      | 2021-12-30                           |
| MindSpore版本 | 1.5.0                                |
| 数据集        | metal_nut                            |
| batch_size    | 128                                  |
| 输出          | 概率                                 |
| 准确性        | 单卡: ok: 0.56, nok: 0.77, acc: 0.74 |

# 随机情况说明

未设置随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。