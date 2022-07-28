# 目录

<!-- TOC -->

- [目录](#目录)
- [nnunet描述](#nnunet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [贡献指南](#贡献指南)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# nnunet描述

nnunet是第一个旨在处理医学数据集多样性的分割方法。它简化了相关设置，并自动化执行某些决策，为任何给定的数据集设计一个成功的分割流程。

[论文](https://arxiv.org/abs/1904.08128v2)：Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

# 模型架构

nnUNet能够根据任务不同来选择2D或是3D的Unet，Unet由编码器和解码器构成，并且每个编解码之间有Skip-connection连接。

# 数据集

使用的数据集：[Medical Segmentation Decathlon-Task04](<http://medicaldecathlon.com/.>)

- 数据集大小：27.1M，共3个类、394
    - 训练集：21.68M，共315个3D体素文件
    - 测试集：12.5M，共79个3D体素文件
- 数据格式：nii.gz文件
    - 注：数据将在src/nnUNet中处理。

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

## 训练数据准备

```text
  在训练网络之前，需要进行以下预处理步骤：

  #1.需要提前创建目录：
  mkdir -p ./src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/

  #2.将下载好Task04数据集放在下面路径下
  ./src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task04_Hippocampus
  # example：├─Task04_Hippocampus
  #            ├─imagesTr      训练集
  #            ├─imagesTs      测试集
  #            ├─labelsTr      训练集标签
  #            ├─dataset.json  数据集的字典信息
```

```shell
  #2.导入环境变量
  cd nnUNet
  export nnUNet_raw_data_base="./src/nnUNetFrame/DATASET/nnUNet_raw" #原始数据目录
  export nnUNet_preprocessed="./src/nnUNetFrame/DATASET/nnUNet_preprocessed"#数据预处理结果
  export RESULTS_FOLDER="./src/nnUNetFrame/DATASET/nnUNet_trained_models"#结果保存路径
  #Note 导入的路径是相对路径 工作目录在/nnUNet下
```

```shell
  #3.转换数据集和数据预处理
  python ./src/nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i ./src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task04_Hippocampus
  python ./src/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 4
```

Note:如果提示ModuleNotFoundError: No module named 'src'，可以尝试执行以下命令

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## 单卡训练

```shell
  # 运行训练示例
  bash scripts/run_standalone_train_ascend.sh [NETWORK]
```

- 2d训练

```shell
bash scripts/run_standalone_train_ascend.sh 2d
```

- 3d_fullres训练

```shell
bash scripts/run_standalone_train_ascend.sh 3d_fullres
```

上述python命令将在后台运行，您可以通过train.log文件或在配置RESULTS_FOLDER下查看结果。
训练结束后，您可在配置的RESULTS_FOLDER找到检查点文件。可以通过grep命令查看日志loss信息：

```shell
  grep "loss" train.log
  #
  2021-12-12 22:06:46.793184: train loss : 4.1443
  2021-12-12 22:06:49.264263: validation loss: 2.6799
  ...
```

Note:如果训练结束出现"Exception in Thread-4...",的输出是第三库batchgenerators和当前环境某些方面不兼容的原因，
参见官方[issue](https://github.com/MIC-DKFZ/nnUNet/issues/696),
不影响训练结果，训练的结果保存在RESULTS_FOLDER下。

# 脚本说明

## 脚本及样例代码

```text
├──nnUNet
    ├── README_CN.md                        // nnUNet相关说明
    ├── ascend310_infer                     // 实现310推理源代码
    ├── model_utils
    │   ├──config.py                        // 训练配置
    │   ├──device_adapter.py                // 获取云上id
    │   ├──local_adapter.py                 // 获取本地id
    │   └──moxing_adapter.py                // 参数处理
    ├── scripts
        ├──run_standalone_train_ascend.sh   // 在Ascend上训练（单卡）
        ├──run_eval_ascend.sh               // 在Ascend上评估
        ├──run_infer_310.sh                 // 在310上推理脚本
    ├── src
    │   ├──nnunet
    │   │  ├──evaluation                    // 验证精度指标
    │   │  ├──experiment_planning           // 网络执行计划生成
    │   │  ├──inference                     // 推理
    │   │  ├──network_architecture          // 基本网络结构
    │   │  ├──postprocessing                // 网络结果分析
    │   │  ├──preprocessing                 // 网络数据准备变换
    │   │  ├──run                           // 运行加载文件
    │   │  ├──training                      // 训练网络结构
    │   │  ├──utilities                     // 常用工具
    │   │  ├──configuration.py              // 配置函数
    │   │  ├──generate_testset.py           // 交叉验证数据集提取
    │   │  ├──paths.py                      // 路径导入设置
    │   ├──nnUNetFrame                      // 存放数据集
    ├── export.py                           // 将checkpoint文件导出到air/mindir
    ├── postprocess.py                      // 网络310推理后处理脚本
    ├── config_Task004_2D_inference.yaml    // 推理参数配置文件
    ├── config_Task004_3D_inference.yaml    // 推理参数配置文件
    ├── run.py                              // 训练脚本
    ├── eval.py                             // 评估脚本
```

## 评估过程

### 评估

- 在Ascend环境运行时评估Task04数据集,会在相应文件夹（src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/inferTs）下生成推理完成的nii.gz文件

将训练集的第0折数据作为测试集，执行下列命令

```shell
python src/nnunet/generate_testset.py
```

将会得到相应评估用的交叉验证数据

```text
.
└─Task004_Hippocampus
  ├── dataset.json                      // 记录数据集的相关信息
  ├── imagesTr                          // 训练数据集
  ├── imagesTs                          // 线上封闭测试数据
  ├── labelsTr                          // 训练数据集标签
  ├── inferTs                           // 待推理测试集的推理结果
  ├── imagesVal                         // 交叉验证数据图像
  ├── labelsVal                         // 交叉验证标签
```

```shell
  bash scripts/run_eval_ascend.sh [NETWORK]
```

- 2d网络验证

```shell
 bash scripts/run_eval_ascend.sh 2d
```

- 3d 网络验证

```shell
bash scripts/run_eval_ascend.sh 3d_fullres
```

Note:如果推理3d_fullres网络之前推理过2d网络，需要清空inferTs下的结果，保证输出的文件夹是空的。

- 获得相关的Dice指标，需要执行下面命令,相应结果在inferTs的summary.json进行查看。
- 对于Task04任务 0是背景类， 1是"Anterior"类，2是"Posterior"类，Dice指标的范围为0-1，值越高表示分割结果越好。

```shell
  python src/nnunet/evaluation/evaluator.py -ref src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/labelsVal  -pred src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/inferTs -l 0 1 2
```

```text
    "author": "Fabian",
    "description": "",
    "id": "40aa19943a86",
    "name": "",
    "results": {
        "all": [],
        "mean": {
            "0": {
                "Accuracy": 0.9973668936871827,
                "Dice": 0.9986101370753704,
                "False Discovery Rate": 0.0020563276498204953,
                "False Negative Rate": 0.0007217845405626533,
                "False Omission Rate": 0.013241090373866577,
                "False Positive Rate": 0.03692303683366614,
                "Jaccard": 0.997225828125836,
                "Negative Predictive Value": 0.9867589096261333,
                "Precision": 0.9979436723501794,
                "Recall": 0.9992782154594373,
                "Total Positives Reference": 60177.36538461538,
                "Total Positives Test": 60255.096153846156,
                ......
```

## 导出过程

### 导出

可利用配置文件导出模型为MINDIR格式，Task04_Hippocampus的配置文件分别为config_Task004_2D_infrence.yaml,config_Task004_3D_infrence.yaml

```shell
python export.py --config_path [CONFIG_PATH]
```

- 2d 模型导出

```shell
python export.py --config_path config_Task004_2D_infrence.yaml
```

- 3d 模型导出

```shell
python export.py --config_path config_Task004_3D_infrence.yaml
```

## 推理

### 推理过程

推理流程参见上面介绍的[评估](#评估过程)，产生分割结果用scripts/run_eval_ascend.sh脚本，需要获得指标使用src/nnunet/evaluation/evaluator.py 脚本

### 310 推理

如果要执行3D网络的推理，需要执行[评估](#评估过程)流程，获得相应记录的数据。因为网络的推理会产生大量的滑动窗等中间数据。可以在src/nnunet/preprocess_Result文件夹下面检查数据是否完整

```text
.
└─src
    ├──nnunet
       ├── preprocess_Result                    // 记录数据集的相关信息
           ├── aggregated_nb_of_predictions     // 滑动窗记录矩阵
           ├── data_shape                       // 推理过程中的数据大小
           ├── bboxes                           // 滑动窗crop后的数据
           ├── dct                              // 记录原始nii的相关信息
           ├── location                         // 每个滑动窗的坐标
           ├── slicer                           // 截取的slicer
```

将src/nnunet/preprocess_Result拷贝到scripts同级目录下，310服务器文件组织如下

```text
.
├── ascend310_infer                    // 310 推理 C文件
├── preprocess_Result                  // 310 推理需要的bin文件
├── scripts                            // 推理脚本
├── src                                // 代码路径
├── export.py                          // Mindir 导出脚本
├── postprocess.py                     // 310 推理后处理脚本
├── nnUNet_2d.mindir                   // 310 推理 mindir文件
├── nnUNet_3d_fullres.mindir           // 310 推理 mindir
```

1.执行脚本导出

命令参见如上[导出](#导出过程)

```shell
python export.py --config_path [Config_Path]
```

2.在Ascend310执行推理

例如要对3D卷积网络进行推理的话，您需要获得相应的MINDIR[文件](#导出过程)，放在nnUNet目录下，然后执行类似的如下命令即可。

```shell
cd scripts
bash run_infer_310.sh nnUNet_3d_fullres ../nnUNet_3d_fullres.mindir  # 3d_fullres网络推理
bash run_infer_310.sh nnUNet_2d ../nnUNet_2d.mindir # 2d 网络推理
```

将会在scripts文件夹下面获得相应结果:

```text
└─scripts
    ├── result_Files                                                               // 推理的结果以二进制的方式保存
        ├──inferTs                                                                 // 还原的分割结果图
    ├── time_Result                                                                // 时间结果
```

3.查看结果

在scripts/result_Files/inferTs下 summary.json文件,您可以得到如下的结果：

```text
{
    "author": "Fabian",
    "description": "",
    "id": "40aa19943a86",
    "name": "",
    "results": {
        "all": [],
        "mean": {
            "0": {
                "Accuracy": 0.9973668936871827,
                "Dice": 0.9986101370753704,
                ......
```

# 模型描述

## 性能

### 训练性能

| Parameters                   | Ascend                      |
|------------------------------|-----------------------------|
| Model Version                | nnUNet_2d                   |
| Resource                     | Ascend 910                  |
| Uploaded Date                | 06/10/2022 (month/day/year) |
| MindSpore Version            | 1.5.0                       |
| Dataset                      | Task004_hippocampus         |
| batch_size                   | 366                         |
| outputs                      | .nii.gz                     |
| Dice["Anterior","Posterior"] | [0.9535,0.9492]             |
| Time for training            | 5h17m55s                    |

| Parameters                   | Ascend                      |
|------------------------------|-----------------------------|
| Model Version                | nnUNet_3d_fullres           |
| Resource                     | Ascend 910                  |
| Uploaded Date                | 06/10/2022 (month/day/year) |
| MindSpore Version            | 1.5.0                       |
| Dataset                      | Task004_hippocampus         |
| batch_size                   | 9                           |
| outputs                      | .nii.gz                     |
| Dice["Anterior","Posterior"] | [0.9356,0.9211]             |
| Time for training            | 3h53m36s                    |

### 推理性能

| Parameters          | Ascend                   |
| ------------------- |--------------------------|
| Model Version       | nnUNet_2d                |
| Resource            | Ascend 310               |
| Uploaded Date       | 06/10/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                    |
| Dataset             | Task004_hippocampus      |
| batch_size          | 366                      |
| outputs             | .nii.gz                  |
| Dice["Anterior","Posterior"]              | [0.9535,0.9492]          |
| Model for inference | 14.1M (.MINDIR file)     |

| Parameters          | Ascend                      |
| ------------------- |-----------------------------|
| Model Version       | nnUNet_3d_fullres           |
| Resource            | Ascend 310                  |
| Uploaded Date       | 06/10/2022 (month/day/year) |
| MindSpore Version   | 1.5.0                       |
| Dataset             | Task004_hippocampus         |
| batch_size          | 9                           |
| outputs             | .nii.gz                     |
| Dice["Anterior","Posterior"]              | [0.9356,0.9211]             |
| Model for inference | 21.6M (.MINDIR file)        |

# 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

# ModelZoo主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
