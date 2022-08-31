# 目录

<!-- TOC -->

- [目录](#目录)
- [DecoMR描述](#decomr描述)
    - [模型架构](#模型架构)
- [数据集](#数据集)
    - [使用的数据集](#使用的数据集)
    - [数据组织](#数据组织)
    - [数据预处理](#数据预处理)
    - [其他数据](#其他数据)
    - [本项目需要用到的其他数据组织](#本项目需要用到的其他数据组织)
- [预训练模型](#预训练模型)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练过程](#训练过程)
    - [单卡训练](#单卡训练)
    - [分布式训练](#分布式训练)
- [推理](#推理)
    - [推理过程](#推理过程)
    - [推理结果](#推理结果)
- [性能](#性能)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [其他情况说明](#其他情况说明)
- [贡献指南](#贡献指南)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DecoMR

## DecoMR描述

**3D Human Mesh Regression with Dense Correspondence**  
[Wang Zeng, Wanli Ouyang, Ping Luo, Wentao Liu, Xiaogang Wang]  
CVPR 2020 ，论文可从[DecoMR](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_3D_Human_Mesh_Regression_With_Dense_Correspondence_CVPR_2020_paper.pdf)下载

### 模型架构

DecoMR模型提出一种model-free的三维人体网格估计框架，它显式地建立了网格与局部图像特征在UV空间（即用于三维网格纹理映射的二维空间）中的密集对应关系。实验表明，所提出的局部特征对齐和连续UV Map在多个公共基准上优于现有的基于3D网格的方法。

## 数据集

### 使用的数据集

- [UP-3D](http://files.is.tuebingen.mpg.de/classner/up/): 该数据集用于训练和测试。可以从链接[UP-3D zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip)下载数据集的压缩包，包括训练集和测试集。解压完后, 请在config.py完成路径配置。

### 数据组织

```text
├── up-3d
│   ├── _image.png                           # up-3d数据
│   ├── _body.pkl                            # 姿态和形状标注
│   ├── _joints.npy                          # 关键点标注
│   ├── trainval.txt                         # 训练集数据编号
│   ├── test.txt                             # 测试集数据编号
```

### 数据预处理

数据下载和解压完成后，运行如下命令完成up-3d数据集的预处理，从而生成项目所需要的数据标注和gt_iuv_img。

  ```shell
  python preprocess_dataset.py --train_files --eval_files --gt_iuv
  ```

### 其他数据

本项目所需的一些其他必要数据，请从论文指定链接[data](https://drive.google.com/drive/folders/1xWBVfQa7OZ14VgT9BVO9Lj_kDqRAcQ-e)链接进行下载，下载完解压到./DecoMR即可使用。还需要从[Unite the People repository](https://github.com/classner/up)通过以下脚本下载SMPL模板：

  ```shell
  wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl --directory-prefix=data
  ```

此外，需要在[male and female models](https://smpl.is.tue.mpg.de/)下载性别模型，下载完后解压到data目录。

#### 本项目需要用到的其他数据组织

```text
├── data
    ├── uv_sampler
    │   ├── paras_h0064_w0064_BF.npz                # BF 64x64参数
    │   ├── paras_h0064_w0064_SMPL.npz              # SMPL 64x64参数
    │   ├── paras_h0128_w0128_BF.npz                # BF 128x128参数
    │   ├── smpl_boundry_free_template.obj          # BF SMPL模板
    │   ├── smpl_fbx_template.obj                   # 通用SMPL模板
    ├── basicmodel_f_lbs_10_207_0_v1.0.0.pkl        # 女性SMPL模型参数
    ├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl        # 男性SMPL模型参数
    ├── basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl  # 通用SMPL模型参数
    ├── BF_ref_map_64.npy                           # BF参照图数据
    ├── J_regressor_extra.npy                       # 关节点回归器
    ├── namesUPlsp.txt                              # 测试集图像编号
    ├── reference_mesh.obj                          # 参照网格参数
    ├── segm_per_v_overlap.pkl                      # 重合顶点分割参数
    ├── SMPL_ref_map_64.npy                         # SMPL参照图数据
    ├── vertex_texture.npy                          # 顶点纹理数据
    ├── weight_p24_h0128_w0128_BF.npy               # BF 128x128uv权重参数
```

## 预训练模型

pytorch预训练模型(resnet50)

ResNet主干网络选用resnet50的结构，包含卷积层和全连接层在内共有50层，本模型不使用全连接层。整体由5个Stage组成，第一个Stage对输出进行预处理，后四个Stage分别包含3,4,6,3个Bottleneck。

下载 [ResNet50预训练模型](https://download.pytorch.org/models/resnet50-19c8e357.pth)

mindspore预训练模型

下载pytorch预训练模型，再运行如下脚本，得到对应的mindspore模型，将mindspore模型输出到data文件夹中。注：运行该脚本需要同时安装pytorch环境(测试版本号为1.3，CPU 或 GPU)

```bash
# MODEL_NAME: 模型名称vgg或resnet
# PTH_FILE: 待转换模型文件绝对路径
# MSP_FILE: 输出模型文件绝对路径
bash convert_model.sh [MODEL_NAME] [PTH_FILE] [MSP_FILE]
```

## 环境要求

- 具体的python第三方库，见requirements.txt文件
- 安装依赖：本项目用到opendr来渲染3d网格，需要安装opendr，安装前需要安装依赖：

  ```shell
  sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
  sudo apt-get install libosmesa6-dev
  sudo apt-get install gfortran
  pip install --force-reinstall pip==19
  pip install -r requirements.txt
  ```

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 进入script文件夹，运行：

  ```bash
  # 运行单卡训练示例
  bash ./run_train_standalone_gpu.sh up-3d 3 5 30 16 './ckpt'

  # 运行分布式训练示例GPU
  bash ./run_train_distribute_gpu.sh up-3d 8 5 30 16 './ckpt'

  # 运行评估示例
  bash ./run_eval.sh up-3d 16
  ```

## 脚本说明

### 脚本及样例代码

```bash
├── DecoMR
    ├─ README.md                                 # 模型相关说明
    ├─ preprocess_datasets.py                    # 训练测试图像预处理
    ├─ preprocess_surreal.py                     # surreal训练测试图像预处理
    ├─ eval.py                                   # 评估脚本
    ├─ train.py                                  # 训练脚本
    ├─ pretrained_model_convert
    │  ├─ pth_to_msp.py                          # pth文件转换成ckpt文件
    │  ├─ resnet_msp.py                          # mindspore下resnet预训练模型的网络结构
    │  ├─ resnet_pth.py                          # pytorch下resnet预训练模型的网络结构
    ├─ scripts
    │  ├─ convert_model.sh                       # 转换预训练模型
    │  ├─ run_eval.sh                            # 启动评估
    │  ├─ run_train_distribute_gpu.sh            # 启动多卡训练
    │  ├─ run_train_standalone_gpu.sh            # 启动单卡训练
    ├─ datasets
    │  ├─ preprocess
    │     ├─ generate_gt_iuv.py                  # gt_iuv图像生成
    │     ├─ surreal.py                          # surreal数据集预处理
    │     └─ up_3d.py                            # up-3d数据集预处理
    ├─ base_dataset.py                           # 数据集加载
    └─ surreal_dataset.py                        # surreal数据集加载
    ├─ models
    │  ├─ dense_cnn.py                           # 网络定义
    │  ├─ DMR.py                                 # CNet和LNet定义端到端的组合
    │  ├─ geometric_layers.py                    # 几何变换层
    │  ├─ grid_sample.py                         # grid_sample算子
    │  ├─ layer.py                               # 网络层
    │  ├─ resnet.py                              # resnet的网络结构，resnet50版本
    │  ├─ smpl.py                                # smpl模板
    │  ├─ upsample.py                            # 上采样网络
    │  ├─ uv_generator.py                        # uv图像生成
    │  ├─ TrainOneStepDP.py                      # CNet单步训练
    │  ├─ TrainOneStepEnd.py                     # LNet单步训练
    │  ├─ WithLossCellDP.py                      # CNet损失
    │  └─ WithLossCellEnd.py                     # LNet损失
    ├─ utils
    │  ├─ config.py                              # 数据路径
    │  ├─ imutils.py                             # 图像处理函数
    │  ├─ objfile.py                             # obj文件读取
    │  ├─ renderer.py                            # iuv图像渲染
    │  ├─ train_options.py                       # 训练参数
```

### 脚本参数

具体参数说明和修改见untils中train_options文件。

## 训练过程

### 单卡训练

- 使用单卡训练运行下面的命令:

  ```bash
  python train.py --dataset=up-3d --device_id=3 --num_epochs_dp=5 --num_epochs_end=30 --batch_size=16 --ckpt_dir='./ckpt'
  或：
  bash ./run_train_standalone_gpu.sh up-3d 3 5 30 16 './ckpt'
  ```

### 分布式训练

- 使用分布式训练运行下面的命令:

  ```bash
  python train.py --run_distribute=True --ngpus=8 --dataset=up-3d --num_epochs_dp=5 --num_epochs_end=30 --batch_size=16 --ckpt_dir='./ckpt'
  或：
  bash ./run_train_distribute_gpu.sh up-3d 8 5 30 16 './ckpt'
  ```

## 推理

### 推理过程

- 使用如下命令进行推理:

  ```bash
  python eval.py --dataset=up-3d --batch_size=16
  或：
  bash ./run_eval.sh up-3d 16
  ```

### 推理结果

> *** Final Results ***  
> Shape Error: 202.17

## 性能

### 训练性能

训练性能如下：

| Parameters                 | GPU                                                          |
| -------------------------- |--------------------------------------------------------------|
| Model Version              | DecoMR                                                       |
| Resource                   | PCIE 3090-24G                                                |
| uploaded Date              | 06/28/2022 (month/day/year)                                  |
| MindSpore Version          | 1.8.0                                                        |
| Dataset                    | up-3d                                                        |
| Training Parameters        | epoch_dp=5, epoch_end=30, steps per epoch=55, batch_size=128 |
| Optimizer                  | Adam                                                         |
| Loss Function              | MSE, BSE, L1                                                 |
| outputs                    | probability                                                  |
| Loss                       |                                                              |
| Speed                      | 2245 ms/step（8pcs）                                           |
| Total time                 | 1.8h                                                         |
| Parameters (M)             | 236.9                                                        |
| Checkpoint for Fine tuning | (.ckpt file)                                                 |
| Scripts                    |

### 推理性能

 推理性能如下：

| Parameters          | GPU                         |
|---------------------|-----------------------------|
| Model Version       | DecoMR                      |
| Resource            | GPU                         |
| Uploaded Date       | 06/28/2022 (month/day/year) |
| MindSpore Version   | 1.8.0                       |
| Dataset             | up-3d                       |
| batch_size          | 16                          |
| outputs             | probability                 |
| shape error         | 202.17                      |
| Model for inference |                             |

## 随机情况说明

使用了train.py中的随机种子。

## 其他情况说明

由于本项目比较依赖计算资源，大多数训练报错情况为数据加载线程数太大或者batch_size太大所导致，解决方法为调小train_options中的num_workers，或者减小batch_size

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
