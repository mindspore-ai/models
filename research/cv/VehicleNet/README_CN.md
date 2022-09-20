# 目录

<!-- TOC -->

- [目录](#目录)
- [VehicleNet描述](#vehiclenet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [前处理](#前处理)
        - [推理](#推理)
        - [进行onnx推理](#进行onnx推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [VehicleNet上的VehicleNet](#vehiclenet上的vehiclenet)
        - [推理性能](#推理性能)
            - [VeRi-776上的VehicleNet](#veri-776上的vehiclenet)
    - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# VehicleNet描述

车辆重新识别（re-id）是为了在不同的摄像头中发现感兴趣的汽车，通常被视为图像检索问题的子任务。它可以应用于公共场所进行交通分析，从而方便了交通拥堵管理和流量优化。然而，车辆re-id仍然具有挑战性，因为它固有地包含多个类内变体，例如视点，照明和遮挡。因此，考虑到现实场景的多样性和复杂性，车辆re-id系统需要一个鲁棒性和分辨力强的视觉表示。为了克服数据的局限性和多源数据集的使用，作者通过公共数据集构建一个大规模的数据集，称为VehicleNet，并通过两阶段渐进学习的方式学习车辆表示的常识，利用公共数据集重新识别车辆的动机，将识别不同车型的常识传递到最终车型中。

[论文](https://arxiv.org/pdf/2004.06305.pdf)：Zheng Z, Tao R, Wei Y , et al. VehicleNet: Learning Robust Visual Representation for Vehicle Re-identification[J]. 2020.

# 模型架构

基本骨架模型采用ResNet50。改进：将原来的平均池化层替换为自适应平均池化层，自适应平均池化层根据高度和宽度通道来输出输入特征图的平均值；添加一个512维的全连接层和一个批处理归一化层以减小特征尺寸，删除原始分类器，添加一个全连接层以输出最终的分类预测。

# 数据集

[CompCars]

- 数据集大小：12G
    - 训练集：共136708张图像
- 数据格式：txt文件(label)
    - 注：数据将在src/dataset.py中处理。

[CityFlow]

- 数据集大小：3.3G
    - 训练集：共52717张图像
- 数据格式：txt文件(label)
    - 注：数据将在src/dataset.py中处理。

[VehicleID]

- 数据集大小：7.8G
    - 训练集：共221567张图像
- 数据格式：txt文件(label)
    - 注：数据将在src/dataset.py中处理。

[VeRi-776]

- 数据集大小：1.1G
    - 训练集：共37746张图像
    - test集：共11579张图像
    - query集：共1678张图像
- 数据格式：txt文件(label、camera)
    - 注：数据将在src/dataset.py中处理。

注：在Imagenet数据集上预训练ResNet50，迁移至该模型VehicleNet。
pretrained-ckpt文件地址https://download.mindspore.cn/model_zoo/r1.1/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/

# 数据集混合方式

## 目录结构

```bash
├── VehicleNet
    ├── VeCc
        ├── image_train
        ├── name_train.txt
    ├── VeRi
        ├── image_train
            ├── XX_XX_XX_XX.jpg             // 图片
        ├── image_test
            ├── XX_XX_XX_XX.jpg             // 图片
        ├── image_query
            ├── XX_XX_XX_XX.jpg             // 图片
        ├── name_train.txt
            ├── XX_XX_XX_XX.jpg XX          // 一行表示：图片命名 label
        ├── name_train_second.txt
            ├── XX_XX_XX_XX.jpg XX          // 一行表示：图片命名 label
        ├── name_test.txt
            ├── XX_XX_XX_XX.jpg XX XX       // 一行表示：图片命名 camera label
        ├── name_query.txt
            ├── XX_XX_XX_XX.jpg XX XX       // 一行表示：图片命名 camera label
    ├── VeId
        ├── image_train
        ├── name_train.txt
    ├── VeCf
        ├── image_train
        ├── name_train.txt
```

## 标签序号重排

各数据集的name_train.txt中标签序号：
VeCc：0-4445
VeRi：4446-5020
VeId：5021-31348
VeCf：31348-31788

原始数据集目录结构：

```bash
├── VehicleNet
    ├── dataset
        ├── dataset_txt     // 将各数据集的name_train.txt重命名为数据集名_name_train.txt放在该文件夹下
        ├── CityFlow        // 原始数据集
        ├── CompCars
        ├── VehicleID
        ├── VeRi-776
        ├── VehicleNet      // 新的数据集，详情见上一小节
            ├── VeRi        // 将原始数据集的image_train、image_test和image_test文件夹复制过来
            ├── VeCc        // 将原始数据集的image_train文件夹复制过来，下面同理
            ├── VeCf
            ├── VeId
    ├── src
    ├── script
    ├── train.py
    ├── eval.py
    ├── ...
```

切换至src目录，运行sort_dataset_label.py

```python
python sort_dataset_label.py
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行单机训练示例
  bash run_standalone_train.sh [DEVICE_ID]

  # 运行分布式训练示例
  bash run_distribute_train.sh [RANK_SIZE]

  # 运行评估示例
  bash run_eval.sh [DEVICE_ID] [CKPT_PATH]
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                            // 所有模型相关说明
    ├── posenet
        ├── README.md                        // vehiclenet相关说明
        ├── scripts
        │   ├──run_standalone_train.sh       // 单机到Ascend处理器的shell脚本
        │   ├──run_distribute_train.sh       // 分布式到Ascend处理器的shell脚本
        │   ├──run_eval.sh                   // Ascend评估的shell脚本
        ├── src
        │   ├──autoaugment.py                // 数据增强
        │   ├──lr_generator.py               // 学习率策略
        │   ├──re_ranking.py                 // 评估阶段用到的重排序
        │   ├──save_callback.py              // 边训练边推理设置
        │   ├──dataset.py                    // 数据集转换成mindrecord格式，创建数据集及数据预处理
        │   ├──vehiclenet.py                 // vehiclenet架构
        │   ├──loss.py                       // vehiclenet的损失函数定义
        │   ├──config.py                     // 参数配置
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 评估脚本
        ├── export.py                        // 将checkpoint文件导出到mindir、air下
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置VehicleNet。

  ```python
  # common_config
  'device_target': 'Ascend',          # 运行设备Ascend
  'device_id': 0,                     # 用于训练或评估数据集的设备ID使用run_distribute_train.sh进行分布式训练时可以忽略
  'pre_trained': True,                # 是否基于预训练模型训练
  'dataset_path': '../dataset/VehicleNet/',               # 数据集路径
  'mindrecord_dir': '../dataset/VehicleNet_mindrecord',   # mindrecord文件路径
  # 'mindrecord_dir': "/cache/dataset_train/device_" + os.getenv('DEVICE_ID') + "/VehicleNet_mindrecord",
  'save_checkpoint': True,            # 是否保存检查点文件
  'pre_trained_file': '../checkpoint/resnet50.ckpt', # 加载预训练checkpoint文件保存的路径
  'checkpoint_dir': '../checkpoint',  # checkpoint文件夹路径
  'save_checkpoint_epochs': 5,        # 保存检查点间隔epoch数
  'keep_checkpoint_max': 10           # 保存的最大checkpoint文件数

  # 一阶段训练
  'name': 'VehicleNet',      # 数据集名字
  'num_classes': 31789,      # 类别
  'epoch_size': 80,          # 迭代次数
  'batch_size': 24,          # 批处理大小
  'lr_init': 0.02,           # 初始学习率
  'weight_decay': 0.0001,    # 权重衰减率
  'momentum': 0.9,

  # 二阶段训练
  'name': 'VeRi_train',      # 数据集名字
  'num_classes': 575,        # 类别
  'epoch_size': 40,          # 迭代次数
  'batch_size': 24,          # 批处理大小
  'lr_init': 0.02,           # 初始学习率
  'weight_decay': 0.0001,    # 权重衰减率
  'momentum': 0.9

  # 推理
  'name': 'VeRi_test',       # 数据集名字
  'num_classes': 200,        # 类别
  'checkpoint_dir': './checkpoint'  # 保存路径
  ```

  注：预训练checkpoint文件'pre_trained_file'在ModelArts环境下需调整为对应的绝对路径
  比如"/home/work/user-job-dir/vehiclenet/resnet50.ckpt"

更多配置细节请参考脚本`config.py`。

## 训练过程

### 单机训练

- Ascend处理器环境运行

  ```bash
  bash run_standalone_train.sh [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过train_alone.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式得到损失值：

  ```bash
  epoch: 36 step: 1572, loss is 1.0608034
  epoch time: 137466.433 ms, per step time: 87.447 ms
  epoch: 37 step: 1572, loss is 1.191042
  epoch time: 137462.859 ms, per step time: 87.445 ms
  ...
  ```

  模型检查点保存在checkpoint文件夹下。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_distribute_train.sh [RANK_SIZE]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过device[X]/train[X].log文件查看结果。采用以下方式达到损失值：

  ```bash
  epoch: 1 step: 2337, loss is 6.6500106
  epoch time: 1249107.232 ms, per step time: 534.492 ms
  epoch: 2 step: 2337, loss is 6.67811
  epoch time: 1151065.230 ms, per step time: 492.540 ms
  ...
  epoch: 1 step: 2337, loss is 6.639152
  epoch time: 1249047.422 ms, per step time: 534.466 ms
  epoch: 2 step: 2337, loss is 6.4480543
  epoch time: 1151189.549 ms, per step time: 492.593 ms
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估VeRi-776数据集

  在运行以下命令之前，请检查用于评估的检查点路径。
  请将检查点路径设置为相对路径，例如“../checkpoint/second_train_vehiclenet-40_196.ckpt”。

  ```bash
  bash run_eval.sh [DEVICE_ID] [CKPT_PATH]
  ```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  Rank@1:0.955304 Rank@5:0.961485 Rank@10:0.980930 mAP:0.851210
  ```

## 导出过程

### 导出

在执行推理之前，需要通过export.py导出onnx、mindir或air文件。

```shell
export DEVICE_ID=0
python export.py --ckpt_url [CKPT_URL] --device_id [DEVICE_ID] --file_format [File_FORMAT] --device_target [DEVICE_TARGET]
```

ckpt_url和device_id为必填项,file_format、device_target为选填，如onnx的GPU导出，可为--file_format ONNX --device_target GPU。

## 推理过程

### 前处理

在执行推理前，需要进行数据集预处理，将image和label转换为bin文件。

```shell
python precess.py --dataset_path [DATASET_PATH] --result_path [RESULT_PATH]
```

dataset_path为数据集路径，result_path为输出路径。

### 推理

在还行推理之前我们需要先导出模型，mindir可以在本地环境上导出。batch_size只支持1。

- 在昇腾310上使用VeRi-776数据集的test集和query集进行推理

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [TEST_BIN_PATH] [QUERY_BIN_PATH] [MINDIR_PATH] [TEST_DATA_PATH] [QUERY_DATA_PATH] [TEST_LABEL] [QUERY_LABEL] [TEST_OUT_PATH] [QUERY_OUT_PATH] [DEVICE_ID]
  Rank@1:0.943981 Rank@5:0.963051 Rank@10:0.973182 mAP:0.835471
  ```

### 进行onnx推理

在GPU环境运行时评估VeRi-776数据集

  在运行以下命令之前，请检查用于评估的检查点路径。
  请将检查点路径设置为相对路径，例如“../checkpoint/vehiclenet.onnx”。

  ```bash
  bash run_eval_onnx.sh [ONNX_PATH] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看结果。测试数据集的准确性如下：

  ```text
  Rank@1:0.949344 Rank@5:0.963647 Rank@10:0.973182 mAP:0.836560
  ```

# 模型描述

## 性能

### 评估性能

#### VehicleNet上的VehicleNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-10-09                                 |
| MindSpore版本          | 1.3.1                                                 |
| 数据集                    | VehicleNet(4个数据集整合)                                                    |
| 训练参数        | epoch_size=80, batch_size=24, lr_init=0.02              |
| 优化器                  | SGD                                                    |
| 损失函数              | 交叉熵损失函数                                       |
| 输出                    | Rank1, mAP                                                 |
| 损失                       | 0.0123                                                      |
| 速度                      | 8卡：492毫秒/步                          |
| 总时长                 | 8卡：23h                         |
| 参数(M)             | 25.5                                                        |
| 微调检查点 | 284.72M (.ckpt文件)                                         |
| 推理模型        | 94.15M (.mindir文件)                     |
| 脚本                    | <https://gitee.com/mindspore/models/tree/master/research/cv/VehicleNet> |

#### VeRi-776上的VehicleNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 资源                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G             |
| 上传日期              | 2021-10-09                                 |
| MindSpore版本          | 1.3.1                                                 |
| 数据集                    | VeRi-776                                                   |
| 训练参数        | epoch_size=40, batch_size=24, lr_init=0.02              |
| 优化器                  | SGD                                                    |
| 损失函数              | 交叉熵损失函数                                       |
| 输出                    | Rank1, mAP                                                 |
| 损失                       | 0.0123                                                      |
| 速度                      | 8卡：97毫秒/步                          |
| 总时长                 | 8卡：40m                          |
| 参数(M)             | 25.5                                                        |
| 微调检查点 | 284.72M (.ckpt文件)                                         |
| 推理模型        | 94.15M (.mindir文件)                     |
| 脚本                    | <https://gitee.com/mindspore/models/tree/master/research/cv/VehicleNet> |

### 推理性能

#### VeRi-776上的VehicleNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 资源            | Ascend 910                  |
| 上传日期       | 2021-10-09 |
| MindSpore 版本   | 1.3.1                 |
| 数据集             | VeRi-776     |
| batch_size          | 1                         |
| 输出             | Rank1, mAP                 |
| 准确性            | 8卡：96.78%(Rank1) 83.41%(mAP)   |
| 推理模型 | 94.15M (.mindir文件)         |

## 迁移学习

待补充

# 随机情况说明

在train.py中，我们设置了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
