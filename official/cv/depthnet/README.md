# 目录

<!-- TOC -->

- [目录](#目录)
- [DepthNet描述](#DepthNet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
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
            - [NYU上的DepthNet](#NYU上的DepthNet)
        - [推理性能](#推理性能)
            - [NYU上的DepthNet](#NYU上的DepthNet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# DepthNet描述

深度估计(Depth Estimation)的主要任务就是从给定的RGB图像获取对应的深度图，是3D场景理解的重要课题。而单目深度估计，由于尺度的不确定性等因素，其难度往往更大。由NYU的Eigen等人提出的单目深度估计的经典工作，采用了coarse-to-fine的训练策略，首先利用一个网络基于RGB图片的全局信息，粗略估计深度图。然后又利用一个网络，精细地估计深度图的局部深度信息。

[论文](https://arxiv.org/abs/1406.2283)：Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. David Eigen, Christian Puhrsch, Rob Fergus.

# 模型架构

具体而言，本工作的单目深度估计网络(Depth Net)由CoarseNet和FineNet两部分组成。首先，对CoarseNet，输入一张RGB图片，经过一系列卷积层后，最后再经过两个全连接层输出粗略估计的深度图(Coarse Depth)。FineNet部分，在RGB图片经过一个卷积层后，再与前面输入的Coarse Depth拼接后，组成新的Feature Map，经过若干卷积层后，得到更精细的深度图。

# 数据集

对于该工作的Mindspore复现和验证，我们使用了由[Junjie Hu](https://github.com/JunjH/Revisiting_Single_Depth_Estimation)提供的预处理好的[NYU数据集](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing)，采用了[1](Structure-Aware Residual Pyramid Network for Monocular Depth Estimation. Xiaotian Chen, Xuejin Chen, Zheng-Jun Zha)和[2](Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries. Junjie Hu, Mete Ozay, Yan Zhang, Takayuki Okatani.)等工作中对NYU数据集常用的预处理方式。共有284个场景的数据作为训练数据集，654张图片作为测试集评估精度结果。
数据文件存储路径如下：

```text
├── NYU
    ├── Train
        ├── basement_0001a_out
            ├── 1.jpg
            ├── 1.png
            ├── 2.jpg
            ├── 2.png
              ....
        ├── basement_0001b_out
              ....
    ├── Test
        ├── 00000_colors.png
        ├── 00000_depth.png
        ├── 00001_colors.png
        ├── 00001_depth.png
              ....

```

训练数据集中，RGB图片是以.jpg格式存储的，深度图数据是以.png格式存储的。深度值 z=pixel_value / 255.0 x 10.0 (m)
测试集中，RGB图片和深度图均以.png格式存储，深度值 z=pixel_value / 1000.0 (m) 数据集具体读取方法可以参考data_loader.py文件。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore，或者使用运道(ROMA)已经配置好的开发环境。

- 数据及模型准备

  ```bash
  # 开始训练

  # 单P训练环境使用命令：
  ###bash脚本命令
  cd scripts
  bash run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
  例如：bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU 0
  ### 或者用python命令
  python train.py --data_url ~/mindspore_dataset/NYU --device_id 0 > train.log 2>&1 &

  # 模型评估测试，使用.ckpt模型
  ## bash脚本命令
  cd scripts
  bash run_eval.sh [DATASET_PATH] [COARSENET_MODEL_PATH] [FINENET_MODEL_PATH]
  例如：bash run_eval.sh ~/mindspore_dataset/NYU ~/Model/Ckpt/FinalCoarseNet.ckpt ~/Model/Ckpt/FinalFineNet.ckpt

  ## 或者用python命令
  python eval.py --test_data ~/mindspore_dataset/NYU --coarse_ckpt_model ~/Model/Ckpt/FinalCoarseNet.ckpt --fine_ckpt_model ~/Model/Ckpt/FinalFineNet.ckpt> eval.log 2>&1 &

  # coarse网络.ckpt模型转换为.mindir和.air格式
  ## bash脚本命令
  cd scripts
  bash run_export_coarse_model.sh
  ## 或者用python命令
  python export.py --coarse_or_fine coarse

  # fine网络.ckpt模型转换为.mindir和.air格式
  ## bash脚本命令
  cd scripts
  bash run_export_fine_model.sh
  ## 或者用python命令
  python export.py --coarse_or_fine fine

  # 模型推理
  cd scripts
  bash run_infer_310.sh ../Model/MindIR/FinalCoarseNet.mindir ../Model/MindIR/FinalFineNet.mindir ../NYU/Test/ 0

  # 8P分布式环境训练：
  ###bash脚本命令
  cd scripts
  bash run_distributed_train_ascend.sh [DATASET_PATH] [RANK_TABLE_FILE]
  例如：bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU ~/rank_table_8pcs.json

  # 模型评估测试，使用.ckpt模型
  ## bash脚本命令
  cd scripts
  bash run_eval.sh [DATASET_PATH] [COARSENET_MODEL_PATH] [FINENET_MODEL_PATH]
  例如：bash run_eval.sh ~/mindspore_dataset/NYU ~/Model/Ckpt/FinalCoarseNet_rank0.ckpt ~/Model/Ckpt/FinalFineNet_rank0.ckpt
  ## 或者用python命令
  python eval.py --test_data ~/mindspore_dataset/NYU --coarse_ckpt_model ~/Model/Ckpt/FinalCoarseNet_rank0.ckpt --fine_ckpt_model ~/Model/Ckpt/FinalFineNet_rank0.ckpt > eval.log 2>&1 &
  ```

# 脚本说明

## 脚本及样例代码

```text
├── ModelZoo_DepthNet_MS_MTI
        ├── ascend310_infer               // 模型推理
            ├── CmakeLists.txt            // 模型推理编译cmakelist
            ├── build.sh                  // build脚本编译cmakelist文件
            ├── src
                ├── main.cc               // 模型推理主函数
                ├── utils.cc              // 文件操作函数
            ├── inc
                ├── utils.h               // 文件操作头文件
        ├── scripts
            ├── run_eval.sh               // 运行评估脚本
            ├── run_export_coarse_model.sh  // 导出coarse模型脚本
            ├── run_export_fine_model.sh    // 导出fine模型脚本
            ├── run_infer_310.sh            // 模型推理脚本
            ├── run_standalone_train_ascend.sh  // 运行单卡训练脚本
            ├── run_distributed_train_ascend.sh  // 8P环境分布式训练脚本
        ├── src
             ├── data_loader.py           // 读取数据
             ├── loss.py                  // 定义损失函数以及评估指标
             ├── net.py                   // 定义网络结构
        ├── README.md                     // DepthNet使用文档
        ├── eval.py                       // 评估测试
        ├── export.py                     // 网络.ckpt模型转化为.mindir和.air格式
        ├── postprocess.py                // 推理后图片后处理
        ├── preprocess.py                 // 推理前图片预处理
        ├── train.py                      // 训练文件
```

### 训练过程

- Ascend处理器环境运行

  ```bash
  ### bash脚本命令
  cd scripts
  bash run_standalone_train_ascend.sh [DATASET_PATH] [DEVICE_ID]
  例如：bash run_standalone_train_ascend.sh ~/mindspore_dataset/NYU 0
  ### 或者用python命令
  python train.py --data_url ~/mindspore_dataset/NYU --device_id 0 > train.log 2>&1 &
  ```

  运行上述命令后，您可以通过`train.log`文件查看结果 。

  ```bash
  # python train.log
  traing coarse net, step: 0 loss:1.73914, time cost: 54.1325149361328
  traing coarse net, step: 10 loss:1.606946, time cost: 0.051651954650878906
  traing coarse net, step: 20 loss:1.5636182, time cost: 0.06647920608520508
  ...
  traing coarse net, step: 14150 loss:0.39416388, time cost: 0.04835963249206543
  traing coarse net, step: 14160 loss:0.38534725, time cost: 0.04690909385681152
  traing coarse net, step: 14170 loss:0.39199725, time cost: 0.04682588577270508
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估NYU数据集

 将最终训练好的.ckpt的Coarse和Fine模型分别命名为FinalCoarseNet.ckpt和FinalFineNet.ckpt，放在./Model/Ckpt文件夹下，加载训练好的.ckpt模型，然后进行评估。

  ```bash
  ## bash脚本命令
  cd scripts
  bash run_eval.sh [DATASET_PATH]
  例如：bash run_eval.sh ~/mindspore_dataset/NYU
  ## 或者用python命令
  python eval.py --test_data ~/mindspore_dataset/NYU > eval.log 2>&1 &
  ```

## 导出过程

### 模型导出

```bash
# bash 脚本命令
## coarse model导出：
cd scripts
bash run_export_coarse_model.sh

## fine model导出:
cd scripts
bash run_export_fine_model.sh

# 或者使用python命令：
## coarse model 导出：
python export.py --coarse_or_fine coarse

## fine model 导出:
python export.py --coarse_or_fine fine
```

## 推理过程

### 推理

- 在推理环境运行时评估NYU数据集

  在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。

- 进入scripts目录，按照下面命令执行模型推理。

  使用以下命令

```bash
 bash run_infer_310.sh [MINDIR1_PATH] [MINDIR2_PATH] [DATA_PATH] [DEVICE_ID]
```

其中MINDIR1_PATH为coarse_net路径，MINDIR2_PATH为fine_net路径，DATA_PATH为测试集路径

  ```bash
  cd scripts
  bash run_infer_310.sh ../Model/MindIR/FinalCoarseNet.mindir ../Model/MindIR/FinalFineNet.mindir ../NYU/Test/ 0
  ```

推理的结果保存在当前目录下，preprocess_Result文件夹为预处理后的图片，result_Files模型推理后的图片结果，在acc.log日志文件中可以找到类似以下的结果。

# 模型描述

## 性能

### 评估性能

#### NYU上的DepthNet

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | DepthNet                                            |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 720G；系统 Euler2.8             |
| 上传日期              | 2021-12-25                                 |
| MindSpore版本          | 1.5.1                                                       |
| 数据集                    | NYU                                              |
| coarse_net训练参数        | epoch=20, batch_size = 32, lr=0.0001    |
| fine_net训练参数        | epoch=10, batch_size = 32, lr=0.00001    |
| 优化器                  | Adam                                                 |
| 损失函数              | L2 Loss和ScaleInvariant Loss的组合 |
| 输出                    | 全景深度图                                         |
| 损失 | 0.2 |
| 速度 | 640batch/s（单卡） |
| 总时长 | 360min@1P(coarse) + 360min@1P(fine) |
| 参数 | 84.5M(.ckpt文件) |

####

### 推理性能

#### NYU上的DepthNet

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | DepthNet                |
| 资源            |  Ascend 310；系统 Ubuntu 18.04.3 LTS 4.15.0-45.generic x86_64               |
| 上传日期       | 2021-12-25 |
| MindSpore 版本   | 1.5.1                       |
| 数据集             | NYU测试集 |
| batch_size          | 1                      |
| 输出             | delta1_loss，delta2_loss，delta3_loss, abs_relative_loss, sqr_relative_loss, rmse_linear_loss, rmse_log_loss |
| 指标            | delta1_loss:  0.618 delta2_loss:  0.880 delta3_loss:  0.965 abs_relative_loss: 0.228 sqr_relative_loss:  0.224  rmse_linear_loss:  0.764 rmse_log_loss:  0.272
 |
| 推理模型 | coarse_net: 84M (.mindir文件) fine_net: 482K (.mindir文件)    |

# 随机情况说明

train.py中设置了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

