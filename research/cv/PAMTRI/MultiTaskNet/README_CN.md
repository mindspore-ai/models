# 目录

<!-- TOC -->

- [目录](#目录)
- [MultiTaskNet描述](#MultiTaskNet描述)
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
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [VeRi上的MulTiTaskNet](#VeRi上的MulTiTaskNet)
        - [推理性能](#推理性能)
            - [VeRi上的MulTiTaskNet](#VeRi上的MulTiTaskNet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MultiTaskNet描述

多任务网络(MultiTaskNet)使用嵌入姿势表示的多任务网络用于联合车辆再识别和属性分类。

# 数据集

使用的数据集：[VeRi](https://vehiclereid.github.io/VeRi/)

- 数据集大小：961M，共776个id、5万张彩色图像
    - 训练集：600M, 37778张图像
    - 测试集：213M, 11579张图像(test) 42.6M, 1678张图像(query)

- 数据格式：RGB
    - 注：数据将在src/dataset/dataset.py中处理。

- 目录结构如下：

    ```bash
    ├── data
        ├── veri
        |    ├── image_train
        |    |    ├── ...
        |    ├── image_test
        |    |    ├── ...
        |    ├── image_query
        |    |    ├── ...
        |    ├── heatmap_train
        |    |   ├── ...
        |    ├── heatmap_test
        |    |   ├── ...
        |    ├── heatmap_query
        |    |   ├── ...
        |    ├── segment_train
        |    |   ├── ...
        |    ├── segment_test
        |    |   ├── ...
        |    ├── segment_query
        |    |   ├── ...
        |    ├── label_train.csv
        |    ├── label_test.csv
        |    ├── label_query.csv
    ```

其中`*.csv`标签已经提供，点击[获取](https://github.com/NVlabs/PAMTRI/tree/master/MultiTaskNet/data/veri)，`image_train`、`image_test`和`image_query`是数据集VeRI中数据。其他未提供的数据由第一个模型PoseEstNet中的`run_trans_gpu(ascend).sh`脚本生成。

生成过程如下：

```bash
#Ascend环境
bash PoseEstNet/scripts/run_trans_ascend.sh /data/veri PoseEstNet.ckpt device_id
#GPU环境
bash PoseEstNet/scripts/run_trans_gpu.sh /data/veri PoseEstNet.ckpt gpu_id

```

注意：
数据集路径有二级文件夹veri, PoseEstNet.ckpt在[PoseEstNet](../PoseEstNet/)中训练生成的ckpt文件夹中，请选择精度最高的ckpt

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

```bash
# 安装依赖包
pip install -r requirements.txt
```

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  bash scripts/run_single_train_ascend.sh [DATA_PATH] [PRETRAINED_PATH] [DEVICE_ID] [HEATMAP_SEGMENT]
  # example: bash scripts/run_single_train_ascend.sh ../data/ ../MultiTask_pretrained.ckpt 1 s
  # example: bash scripts/run_single_train_ascend.sh ../data/ ../MultiTask_pretrained.ckpt 1 h

  # 运行分布式训练示例
  bash scripts/run_distribute_train_ascend.sh [DATASET_PATH] [PRETRAIN_CKPT_PATH] [RANK_TABLE_FILE] [HEATMAP_SEGMENT]
  # example: bash scripts/run_distribute_train_ascend.sh ./data/ ./MultiTask_pretrained.ckpt ./scripts/rank_table_8pcs.json h
  # example: bash scripts/run_distribute_train_ascend.sh ./data/ ./MultiTask_pretrained.ckpt ./scripts/rank_table_8pcs.json s

  # 运行评估示例
  python eval.py --ckpt_path path/ckpt --root path/dataset --device_id device_id --heatmapaware --segmentaware > eval.log 2>&1 &
  # example: python eval.py --ckpt_path path/ckpt --root ./data/ --device_id 0 --heatmapaware True > eval.log 2>&1 &
  # example: python eval.py --ckpt_path path/ckpt --root ./data/ --device_id 0 --segmentaware True > eval.log 2>&1 &
  或
  bash scripts/run_eval_ascend.sh [CKPT_PATH] [DATASET_NAME] [DEVICE_ID] [HEATMAP_SEGMENT]
  # example: bash scripts/run_eval_ascend.sh ./*.ckpt ./data/ 0 h
  # example: bash scripts/run_eval_ascend.sh ./*.ckpt ./data/ 0 s

  # 运行推理示例
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID] [NEED_HEATMAP] [NEED_SEGMENT]
  # bash run_310_infer.sh ../MultiTask_Mindir.mindir ../data/ y Ascend 0 y n
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

```python
# 运行训练示例
bash scripts/run_single_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH] [DEVICE_ID] [HEATMAP_SEGMENT]
# example: bash scripts/run_single_train_gpu.sh ../data/ ../MultiTask_pretrained.ckpt 1 s
# example: bash scripts/run_single_train_gpu.sh ../data/ ../MultiTask_pretrained.ckpt 1 h

# 运行分布式训练示例
bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [PRETRAIN_CKPT_PATH] [RANK_SIZE] [HEATMAP_SEGMENT]
# example: bash scripts/run_distribute_train_gpu.sh ./data/ ./MultiTask_pretrained.ckpt 8 h
# example: bash scripts/run_distribute_train_gpu.sh ./data/ ./MultiTask_pretrained.ckpt 8 s

# 运行评估示例
python eval.py --ckpt_path path/ckpt --device_target GPU --root path/dataset --device_id device_id --heatmapaware --segmentaware > eval.log 2>&1 &
# example: python eval.py --ckpt_path path/ckpt --device_target GPU --root ./data/ --device_id 0 --heatmapaware True > eval.log 2>&1 &
# example: python eval.py --ckpt_path path/ckpt --device_target GPU --root ./data/ --device_id 0 --segmentaware True > eval.log 2>&1 &
或
bash scripts/run_eval_gpu.sh [DATASET_NAME] [CKPT_PATH] [DEVICE_ID] [HEATMAP_SEGMENT]
# example: bash scripts/run_eval_gpu.sh ./data/ ./*.ckpt 0 h
# example: bash scripts/run_eval_gpu.sh ./data/ ./*.ckpt 0 s

```

# 脚本说明

## 脚本及样例代码

```bash
├── cv
    ├── PAMTRI
        ├── README.md                                   // PAMTRI描述
        ├── MultiTaskNet
            ├── ascend_310_infer                    // 实现310推理源代码
            ├── scripts
            |   ├── run_distribute_train_ascend.sh      // 分布式到Ascend的shell脚本
            |   ├── run_distribute_train_gpu.sh         // 分布式到GPU的shell脚本
            |   ├── run_single_train_ascend.sh          // 单卡到Ascend的shell脚本
            |   ├── run_single_train_gpu.sh             // 单卡到GPU的shell脚本
            |   ├── run_eval_ascend.sh                  // Ascend评估的shell脚本
            |   ├── run_eval_gpu.sh                     // GPU评估的shell脚本
            |   ├—— run_onnx_eval_gpu.sh                // ONNX推理shell脚本
            |   ├── run_infer_310.sh                    // Ascend推理shell脚本
            ├── src
            |   ├── dataset
            |   |   ├── data_loader.py                  // 加载数据
            |   |   ├── data_manager.py                 // 数据处理
            |   |   ├── dataset.py                      // 数据处理
            |   |   ├── transforms.py                   // 数据转换
            |   ├── loss
            |   |   ├── cross_entropy_loss.py           // 带标签平滑的交叉熵公式
            |   |   ├── hard_mine_triplet_loss.py       // Triplet Loss函数
            |   |   ├── loss.py                         // 损失函数
            |   ├── model
            |   |   ├── DenseNet.py                     // 模型
            |   ├── scheduler
            |   |   ├── lr.py                           // 学习率
            |   ├── utils
            |   |   ├── evaluate.py                     // 推理函数
            |   |   ├── save_callback.py                // 边训练边推理的实现
            |   |   ├── pthtockpt.py                    // pth格式的预训练模型转换为ckpt
            ├── eval.py                             // 精度验证脚本
            ├—— eval_onnx.py                        // ONNX精度验证脚本
            ├── train.py                            // 训练脚本
            ├── export.py                           // 推理模型导出脚本
            ├── preprocess.py                       // 310推理前处理脚本
            ├── postprocess.py                      // 310推理后处理脚本
            ├── README.md                           // MultiTaskNet描述
        ├── PoseEstNet
```

## 脚本参数

```bash
--device_target: 运行代码的设备, 默认为"Ascend"
--root: 数据集所在路径
--pre_trained: 是否要预训练
--heatmapaware: 是否使用热图
--segmentaware: 是否使用分段
--pre_ckpt_path: 预训练路径
--distribute: 是否进行分布式训练
--ckpt_path: 推理所需要的ckpt所在路径
```

## 训练过程

### 加载预训练权重

pytorch的DensenNet121预训练模型，[点击获取](https://download.pytorch.org/models/densenet121-a639ec97.pth)

```bash
python src/utils/pthtockpt.py --pth_path /path/densenet121-a639ec97.pth
```

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --distribute False --pre_ckpt_path path/pretain_ckpt --root path/dataset --heatmapaware --segmentaware > train_ascend.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train_ascend.log文件查看结果。训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train_ascend.log
  epoch:1 step:1136, loss is 1.4842823
  epcoh:2 step:1136, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  ```bash
  python train.py --distribute False --pre_ckpt_path path/pretain_ckpt --device_target GPU --root path/dataset --heatmapaware --segmentaware > train_gpu.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train_gpu.log文件查看结果。训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train_gpu.log
  epoch:1 step:1136, loss is 1.4842823
  epcoh:2 step:1136, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distribute_train_ascend.sh [DATASET_PATH] [PRETRAIN_CKPT_PATH] [DATASET_PATH] [RANK_TABLE_FILE] [HEATMAP_SEGMENT]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过device[X]/train[X].log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "loss is" device0/train0.log
  epoch: 1 step: 284, loss is 2.924
  epoch: 2 step: 284, loss is 1.293
  ...
  # grep "loss is" device1/train1.log
  epoch: 1 step: 284, loss is 2.652
  epoch: 2 step: 284, loss is 1.242
  ...
  ```

- GPU处理器环境运行

  ```bash
  bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [PRETRAIN_CKPT_PATH] [RANK_SIZE] [HEATMAP_SEGMENT]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过distribute_train_gpu.log文件查看结果。采用以下方式达到损失值：

  ```bash
  #
  epoch: 1 step: 284, loss is 2.924
  epoch: 2 step: 284, loss is 1.293
  ...
  #
  epoch: 1 step: 284, loss is 2.652
  epoch: 2 step: 284, loss is 1.242
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估VeRi数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/PAMTRI/MultiTaskNet/MultiTaskNet-1_142.ckpt”。

  ```bash
  python eval.py --ckpt_path path/ckpt --root path/dataset --device_id device_id --heatmapaware --segmentaware > eval_ascend.log 2>&1 &
  OR
  bash scripts/run_eval_ascend.sh [CKPT_PATH] [DATASET_NAME] [DEVICE_ID] [HEATMAP_SEGMENT]
  ```

  上述python命令将在后台运行，您可以通过eval_ascend.log文件查看结果。测试数据集的准确性如下：

  ```bash
  Computing CMC and mAP
  Results ----------
  mAP: 31.01%
  CMC curve
  Rank-1  : 53.34%
  Rank-2  : 62.16%
  Rank-3  : 68.41%
  Rank-4  : 71.99%
  Rank-5  : 74.85%
  ...
  Rank-50 : 95.71%
  ------------------
  Compute attribute classification accuracy
  Color classification accuracy: 90.25%
  Type classification accuracy: 87.27%
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/PAMTRI/MultiTaskNet/device0/ckpt/best.ckpt”。测试数据集的准确性如下：

  ```bash
  Computing CMC and mAP
  Results ----------
  mAP: 31.01%
  CMC curve
  Rank-1  : 53.34%
  Rank-2  : 62.16%
  Rank-3  : 68.41%
  Rank-4  : 71.99%
  Rank-5  : 74.85%
  ...
  Rank-50 : 95.71%
  ------------------
  Compute attribute classification accuracy
  Color classification accuracy: 90.25%
  Type classification accuracy: 87.27%
  ```

- 在GPU环境运行时评估VeRi数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/PAMTRI/MultiTaskNet/MultiTaskNet-1_142.ckpt”。

  ```bash
  python eval.py --ckpt_path path/ckpt --device_target GPU --root path/dataset --device_id device_id --heatmapaware --segmentaware > eval_gpu.log 2>&1 &
  OR
  bash scripts/run_eval_gpu.sh [CKPT_PATH] [DATASET_NAME] [DEVICE_ID] [HEATMAP_SEGMENT]
  ```

  上述python命令将在后台运行，您可以通过eval_gpu.log文件查看结果。测试数据集的准确性如下：

  ```bash
  Computing CMC and mAP
  Results ----------
  mAP: 31.01%
  CMC curve
  Rank-1  : 53.34%
  Rank-2  : 62.16%
  Rank-3  : 68.41%
  Rank-4  : 71.99%
  Rank-5  : 74.85%
  ...
  Rank-50 : 95.71%
  ------------------
  Compute attribute classification accuracy
  Color classification accuracy: 90.25%
  Type classification accuracy: 87.27%
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/PAMTRI/MultiTaskNet/ckpt_0/best.ckpt”。测试数据集的准确性如下：

  ```bash
  Computing CMC and mAP
  Results ----------
  mAP: 31.01%
  CMC curve
  Rank-1  : 53.34%
  Rank-2  : 62.16%
  Rank-3  : 68.41%
  Rank-4  : 71.99%
  Rank-5  : 74.85%
  ...
  Rank-50 : 95.71%
  ------------------
  Compute attribute classification accuracy
  Color classification accuracy: 90.25%
  Type classification accuracy: 87.27%
  ```

## 导出过程

### 导出

将checkpoint文件导出成mindir格式模型。

```shell
python export.py --root /path/dataset --ckpt_path /path/ckpt --segmentaware --heatmapaware
# example: python export.py --root ./data/ --ckpt_path /path/ckpt --heatmapaware True
# example: python export.py --root ./data/ --ckpt_path /path/ckpt --segmentaware True
```

`heatmapaware`和`segmentaware`根据训练时候的参数而定。

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。

- 在昇腾310上使用VeRi数据集进行推理
    执行推理的命令如下所示，其中`MINDIR_PATH`是mindir文件路径；`DATASET_PATH`是推理数据集路径；`NEED_PREPROCESS`表示数据集是否需要预处理，一般选择`y`；`DEVICE_TARGET`是设备类型，默认`Ascend`；`DEVICE_ID`可选，默认值为0；`NEED_HEATMAP` 表示是否使用热图；`NEED_SEGMENT`是否使用分段。

    ```shell
    # Ascend310 inference
    bash scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID] [NEED_HEATMAP] [NEED_SEGMENT]
    ```

### ONNX推理

在进行推理之前我们需要先导出模型。

- 在GPU环境上使用VeRi数据集进行推理

    执行推理的命令如下所示，其中`ONNX_PATH`是onnx文件路径；`DATASET_PATH`是推理数据集路径；`DEVICE_ID`可选，默认值为0；`HEATMAP_SEGMEN` 表示选择使用热图还是分段。

    ```bash
    bash run_onnx_eval_gpu.sh DATASET_NAME ONNX_PATH DEVICE_ID HEATMAP_SEGMENT
    ```

# 模型描述

## 性能

### 评估性能

#### VeRi上的MulTiTaskNet

| 参数                 | Ascend                                                      | GPU |
| -------------------------- | ----------------------------------------------------------- | -------------------------- |
| 模型版本              | MultiTaskNet                                                | MultiTaskNet |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8             | GPU: Geforce RTX3090；CPU 2.90GHz，64核；内存 251G；Ubuntu18.04 |
| 上传日期              | 2021-09-30                                 | 2021-02-28 |
| MindSpore版本          | 1.3.0                                                       | 1.5.0 |
| 数据集                    | VeRi                                                    | Veri |
| 训练参数        | epoch=10, steps=1136(八卡：142), batch_size = 32, lr=0.0005              | epoch=10, steps=1136(八卡：142), batch_size = 32, lr=0.0003 |
| 优化器                  | adam                                                    | adam |
| 损失函数              | CrossEntropyLabelSmooth和TripletLoss                                       | CrossEntropyLabelSmooth和TripletLoss |
| 输出                    | 概率                                                 | 概率 |
| 损失                       | 100.14                                                      | 1.59 |
| 速度                      | 8卡：2503毫秒/步(segment) 1364.692毫秒/步(heatmap)                        | 单卡：347毫秒/步(segment)   8卡：384毫秒/步(segment)  单卡：950毫秒/步(heatmap)  8卡：1020毫秒/步(heatmap) |
| 总时长                 | 8卡：2小时50分钟(segment)  2小时50分钟(heatmap)                       | 单卡：1小时58分钟(segment) 8卡：1小时10分钟(segment) 单卡：5小时20分钟(heatmap) 8卡：3小时15分钟(heatmap) |
| 微调检查点 | 118M (.ckpt文件)                                         | 114M(.ckpt文件  segment)  115M(.ckpt文件  heatmap) |
| 推理模型        | 40M (.mindir文件)                     |  |
| 脚本                    | [MultiTaskNet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/MultiTaskNet) | [MultiTaskNet脚本](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/MultiTaskNet) |

### 推理性能

#### VeRi上的MulTiTaskNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | Inception V1                |
| 资源            |  Ascend 910；系统 Euler2.8                  |
| 上传日期       | 2021-09-30 |
| MindSpore 版本   | 1.3.0                       |
| 数据集             | VeRi     |
| batch_size          | 32                         |
| 输出             | 概率                 |
| 准确性            | top-1:75.57%; Color acc: 93.20%; Type acc: 91.50% |
| 推理模型 | 40M (.mindir文件)         |

# 随机情况说明

data_loader.py中使用根据id随机采样策略，train.py中使用了随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
