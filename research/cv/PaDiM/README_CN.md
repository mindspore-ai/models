# 目录

<!-- TOC -->

- [目录](#目录)
- [PaDiM描述](#PaDiM描述)
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
        - [加载预训练权重](#加载预训练权重)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [MVTec-AD上训练PaDiM](#MVTec-AD上训练PaDiM)
        - [评估性能](#评估性能)
            - [MVTec-AD上评估PaDiM](#MVTec-AD上评估PaDiM)
        - [推理性能](#评估性能)
            - [MVTec-AD上推理PaDiM](#MVTec-AD上推理PaDiM)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PaDiM描述

PaDiM是2020年提出的基于预训练神经网络的工业异常检测模型。PaDiM训练时仅使用正常样本，训练过程中不对网络参数进行更新(无反向传播)，PaDiM 使用于预先训练的 CNN 来进行分块嵌入，然后使用多元高斯分布来得到正常类别的概率表示。此外，还利用了 CNN 不同层的语义来更高的定位缺陷。

[论文](https://arxiv.org/abs/2011.08785v1)：Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier.PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.2020.

# 模型架构

![架构](picture/PaDiM.png)
PaDiM使用预训练的WideResNet50作为Encoder, 并去除layer4之后的层。

# 数据集

使用的数据集：[MVTec AD](<https://www.mvtec.com/company/research/datasets/mvtec-ad/>)

- 数据集大小：4.9G，共15个类、5354张图片(尺寸在700x700~1024x1024之间)
    - 训练集：共3629张
    - 测试集：共1725张
- 数据格式：二进制文件
    - 注：数据将在src/dataset.py中处理。
- 目录结构:

  ```text
  data
  ├── bottle
  │   ├── ground_truth
  │   │   ├── broken_large
  │   │   │   ├── 000_mask.png
  │   │   │   └── ......
  │   │   ├── broken_small
  │   │   │   ├── 000_mask.png
  │   │       └── ......
  │   ├── test
  │   │   ├── broken_large
  │   │   │   ├── 000.png
  │   │   │   └── ......
  │   │   └── good
  │   │       ├── 000.png
  │   │       └── ......
  │   └── train
  │       └── good
  │           ├── 000.png
  │           └── ......
  ├── cable
  │   ├── ground_truth
  │   │   ├── bent_wire
  │   │   │   ├── 000_mask.png
  ......
  ```

# 特性

## 混合精度

采用混合精度的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  python train.py --dataset_path eg./data/mvtec_anomaly_detection/ --device_id 0 --pre_ckpt_path eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --class_name bottle --save_path eg./mvtec_result/ > train.log
  或
  bash run_train.sh [dataset_path] [pre_ckpt_path] [save_path] [class_name] [device_id]

  # 运行评估示例
  python eval.py --dataset_path eg./data/mvtec_anomaly_detection/ --device_id 0 --pre_ckpt_path eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --class_name bottle --save_path eg./mvtec_result/ > eval.log
  或
  bash run_eval.sh [dataset_path] [pre_ckpt_path] [save_path] [class_name] [device_id]

  # 运行推理示例
  bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CLASS_NAME]
  ```

# 脚本说明

## 脚本及样例代码

```text

  ├── PaDiM
      ├── README.md                    // PaDiM相关说明
      ├── ascend310_infer              // 实现310推理源代码
      ├── scripts
      │   ├── run_310_infer.sh         // 推理脚本
      │   ├── run_eval.sh              // 评估脚本
      │   └── run_train.sh             // 训练脚本
      |   └── run_all_mvtec.sh         // 训练所有的Mvtec数据集
      ├── src
      │   ├── dataset.py               // 数据集加载
      │   ├── model.py                 // 模型加载
      │   ├── operator.py              // 数据操作
      │   └── pthtockpt.py             // pth转ckpt
      ├── eval.py                      // 评估脚本
      ├── export.py                    // 推理模型导出脚本
      ├── postprocess.py               // 310后处理脚本
      ├── preprocess.py                // 310前处理脚本
      └── train.py                     // 训练脚本
```

## 脚本参数

  ```yaml
  --dataset_path:数据集路径
  --class_name:数据类别
  --device_id:设备序号
  --pre_ckpt_path:预训练路径
  --save_path:中间特征保存路径
  ```

## 训练过程

### 加载预训练权重

pytorch的WideResNet50预训练模型，[点击获取](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth)

```bash
python src/pthtockpt.py --pth_path ./models/wide_resnet50_2/wide_resnet50_2-95faca4d.pth
```

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --dataset_path eg./data/mvtec_anomaly_detection/ --device_id 0 --pre_ckpt_path eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --class_name bottle --save_path eg./mvtec_result/ > train.log
  或
  bash run_train.sh [dataset_path] [pre_ckpt_path] [save_path] [class_name] [device_id]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  对于Mvtec数据集，可以通过执行以下命令，进行Mvtec中全部类别数据的训练与推理。

  ```bash
  bash run_all_mvtec.sh [dataset_path] [pre_ckpt_path] [save_path] [device_id]
  ```

## 评估过程

### 评估

- 在Ascend环境运行评估

  ```shell
  python eval.py --dataset_path eg./data/mvtec_anomaly_detection/ --device_id 0 --pre_ckpt_path eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --class_name bottle --save_path eg./mvtec_result/ > eval.log
  或
  bash run_eval.sh [dataset_path] [pre_ckpt_path] [save_path] [class_name] [device_id]
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```shell
  # bottle类参考精度
  img_auc: 0.998, pixel_auc: 0.982
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。

```shell
python export.py --device_id 0 --ckpt_file eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --file_format MINDIR --device_target Ascend
```

将checkpoint文件导出成onnx格式模型。

```shell
python export.py --device_id 0 --ckpt_file eg./models/wide_resnet50_2/wide_resnet50_2.ckpt --file_format ONNX --device_target GPU
```

## 推理过程

### 推理

在运行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。

- 在昇腾310上使用MVTec AD数据集进行推理

  执行推理的命令如下所示, 其中``MINDIR_PATH``是mindir文件路径；

  ``DATASET_PATH``是推理数据集路径, 为数据类(如bottle)的父级目录；

  ``NEED_PREPROCESS``表示数据集是否需要预处理，一般选择'y'；

  ``DEVICE_ID``可选，默认值为0；

  ``CLASS_NAME``表示数据类型，可取：bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.  

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [CLASS_NAME]
  # 例：bash run_infer_310.sh ./PathCore.mindir ../data/ y 0 bottle
  ```

  推理的精度结果保存在acc_[CLASS_NAME].log日志文件中。

在运行推理之前我们需要先导出模型。ONNX可以在任意环境上导出。

- 在昇腾910/RTX3090上使用MVTec AD数据集进行推理

  执行推理的命令如下所示, 其中``ONNX_PATH``是ONNX文件路径；

  ``DATASET_PATH``是推理数据集路径, 为数据类(如bottle)的父级目录；

  ``DEVICE_ID``可选，默认值为0；

  ``SAVA_PATH``是训练结果保存路径；

  在script路径下执行如下命令；

  ```shell
  # inference
  bash run_all_mvtec_onnx.sh [DATASET_PATH] [ONNX_PATH] [SAVA_PATH] [DEVICE_ID]
  # 例：bash run_all_mvtec_onnx.sh eg./data/mvtec_anomaly_detection/ eg./PaDiM.onnx eg./data/mvtec_result/ 0
  ```

  推理的精度结果保存在[CLASS_NAME]_eval文件夹中的eval.log中。

# 模型描述

## 性能

### 训练性能

#### MVTec-AD上训练PaDiM

| 参数          | Ascend                                                        |
| ------------- | --------------------------------------------------------------|
| 模型版本      | PaDiM                                                          |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8        |
| 上传日期      | 2022-5-10                                                      |
| MindSpore版本 | 1.6.1                                                         |
| 数据集        | MVTec AD                                                       |
| 训练参数      | epoch=1, steps依数据类型而定, batch_size = 32                   |
| 速度          | 95毫秒/步                                                      |
| 总时长        | 依数据类型10-15min                                              |

### 评估性能

#### MVTec-AD上评估PaDiM

| 参数           | Ascend                           |
| ------------------- | --------------------------- |
| 模型版本       | PaDiM                             |
| 资源           | Ascend 910；系统 Euler2.8         |
| 上传日期       | 2022-5-10                         |
| MindSpore 版本 | 1.6.1                             |
| 数据集         | MVTec AD                          |
| batch_size     | 1                                |
| bottle_auc     | img_auc: 0.998, pixel_auc: 0.982 |
| cable_auc      | img_auc: 0.923, pixel_auc: 0.968 |
| capsule_auc    | img_auc: 0.915, pixel_auc: 0.986 |
| carpet_auc     | img_auc: 0.999, pixel_auc: 0.990 |
| grid_auc       | img_auc: 0.957, pixel_auc: 0.968 |
| hazelnut_auc   | img_auc: 0.939, pixel_auc: 0.980 |
| leather_auc    | img_auc: 1.000, pixel_auc: 0.990 |
| metal_nut_auc  | img_auc: 0.992, pixel_auc: 0.972 |
| pill_auc       | img_auc: 0.942, pixel_auc: 0.963 |
| screw_auc      | img_auc: 0.844, pixel_auc: 0.985 |
| tile_auc       | img_auc: 0.974, pixel_auc: 0.939 |
| toothbrush_auc | img_auc: 0.972, pixel_auc: 0.988 |
| transistor_auc | img_auc: 0.977, pixel_auc: 0.975 |
| wood_auc       | img_auc: 0.989, pixel_auc: 0.943 |
| zipper_auc     | img_auc: 0.909, pixel_auc: 0.985 |

### 推理性能

#### MVTec-AD上推理PaDiM

| 参数           | Ascend                           |
| ------------------- | --------------------------- |
| 模型版本       | PaDiM                             |
| 资源           | Ascend 310；系统 Euler2.8         |
| 上传日期       | 2022-5-10                         |
| MindSpore 版本 | 1.6.1                             |
| 数据集         | MVTec AD                          |
| bottle_auc     | img_auc: 0.998, pixel_auc: 0.982 |
| cable_auc      | img_auc: 0.923, pixel_auc: 0.968 |
| capsule_auc    | img_auc: 0.915, pixel_auc: 0.986 |
| carpet_auc     | img_auc: 0.999, pixel_auc: 0.990 |
| grid_auc       | img_auc: 0.957, pixel_auc: 0.968 |
| hazelnut_auc   | img_auc: 0.939, pixel_auc: 0.980 |
| leather_auc    | img_auc: 1.000, pixel_auc: 0.990 |
| metal_nut_auc  | img_auc: 0.992, pixel_auc: 0.972 |
| pill_auc       | img_auc: 0.942, pixel_auc: 0.963 |
| screw_auc      | img_auc: 0.844, pixel_auc: 0.985 |
| tile_auc       | img_auc: 0.974, pixel_auc: 0.939 |
| toothbrush_auc | img_auc: 0.972, pixel_auc: 0.988 |
| transistor_auc | img_auc: 0.977, pixel_auc: 0.975 |
| wood_auc       | img_auc: 0.989, pixel_auc: 0.943 |
| zipper_auc     | img_auc: 0.909, pixel_auc: 0.985 |

# 随机情况说明

无

# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。
