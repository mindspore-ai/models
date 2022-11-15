# R(2+1)D

<!-- TOC -->

- [R(2+1)D](#R(2+1)D)
- [R(2+1)D介绍](#R(2+1)D介绍)
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
    - [导出Mindir](#导出Mindir)
    - [在Ascend310执行推理](#在Ascend310执行推理)
    - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## R(2+1)D介绍

​    R(2+1)D:  A Closer Look at Spatiotemporal Convolutions for Action Recognition；来自 Facebook Resaerch &Dartmouth Colledge，通过对动作识别中的各种卷积块进行深度探究，提出了用于视频动作识别的新型网络结构：R(2+1)D。本文的灵感来源是，对单帧视频进行 2D 卷积仍然可以获得接近 3D 时空卷积方法中的 SOTA 结果。论文表明，将三维卷积滤波器分解为单独的时空分量可以显著提高精度，因此设计了一个新的时空卷积块 “R(2+1)D” ，它产生的 CNN 可以达到 Sports-1M，Kinetics，UCF101 和 HMDB51 上的与最优性能相当或更优的结果。

[论文](https://arxiv.org/abs/1711.11248v1)：Du T ,  Wang H ,  Torresani L , et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition[C]// IEEE/CVF Conference on Computer Vision and Pattern Recognition. 0.

## 模型结构

本文最重要的结构就是 (2+1)D 卷积，把 3 维时空卷积分解成 2 维空间卷积和 1 维时间卷积，那么卷积核大小就变成了N’×1×d×d + M×t×1*1，超参数 M 决定了信号在时、空卷积之间投影的子空间个数。分解之后，两个子卷积之间会多出来一个非线性操作，和原来同样参数量的3维卷积相比，倍增了非线性操作，相当于给网络扩容。而且时空分解让优化的过程也分解开来，事实上之前发现，3 维时空卷积把空间信息和动态信息拧在一起，不容易优化，而 2+1 维卷积更容易优化，loss 会更低。

网络输入的 5 维数据（NCTHW）依次经过 5 次 (2+1)D 卷积操作，然后经过一个时空信息池化层，最后经过一个全连接层得到最后的视频动作分类结果。

## 数据集

- [预训练数据集 Kinetics400](https://deepmind.com/research/open-source/kinetics)

   Kinetics400 数据集包括了四百种人体动作类别，每一种类别都至少有400个视频片段，每个片段都取自不同的 Youtube 视频，持续大概十秒。视频后缀共有 'mp4', 'webm', 'mkv' 三种。

  官网已有训练集和验证集的标注，测试集标注未公布。

- [迁移学习数据集 UCF101](https://www.kaggle.com/pevogam/ucf101)

  UCF101动作识别数据集，包含101个类别共13320个视频，均为 avi 格式。该数据集是由用户真实上传的视频构成，视频中包含摄像机运动和杂乱背景。不过该数据集未被分割为训练集和验证集，需要用户自行分割。

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估（训练和评估之前请先执行数据集预处理和预训练模型转换步骤）：

```bash
###参数配置请修改 default_config.yaml 文件

#启动训练：
#通过 python 命令行运行Ascend单卡训练脚本。
python train.py --is_distributed=0 --device_target=Ascend > train_ascend_log.txt 2>&1 &
#通过 bash 命令启动Ascend单卡训练。
bash ./scripts/run_train_ascend.sh device_id
e.g. bash ./scripts/run_train_ascend.sh 0

#Ascend多卡训练。
bash ./scripts/run_distribute_train_ascend.sh rank_size rank_start_id rank_table_file
e.g. bash ./scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json

#启动推理
# default_config.yaml 文件中的 eval_ckpt_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
# 通过 python 命令行运行Ascend推理脚本。
python eval.py --device_target=Ascend > eval_ascend_log.txt 2>&1 &
#通过 bash 命令启动Ascend推理。
bash ./scripts/run_eval_ascend.sh device_id
e.g. bash ./scripts/run_eval_ascend.sh 0
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```tex
├── model_zoo
    ├── README.md                            // 所有模型的说明文件
    ├── R(2+1)D
        ├── README_CN.md                     // R(2+1)D 的说明文件
        ├── ascend310_infer                  // 310推理主代码文件夹
        |   ├── CMakeLists.txt               // CMake设置文件
        |   ├── build.sh                     // 编译启动脚本
        |   ├── inc
        |   |   ├── utils.h                  // 工具类头文件
        |   ├── src
        |   |   ├── main.cc                  // 推理代码文件
        |   |   ├── utils.cc                 // 工具类文件
        ├── scripts
        │   ├──run_distribute_train_ascend.sh // Ascend 8卡训练脚本
        │   ├──run_eval_ascend.sh            // Ascend 推理启动脚本
        │   ├──run_train_ascend.sh           // Ascend 单卡训练启动脚本
        |   ├──run_infer_310.sh              // 310推理启动脚本
        ├── src
        │   ├──config.py                     // 配置加载文件
        │   ├──dataset.py                    // 数据集处理
        │   ├──models.py                     // 模型结构
        │   ├──logger.py                     // 日志打印文件
        │   ├──utils.py                      // 工具类
        ├── default_config.yaml              // 默认配置信息，包括训练、推理、模型冻结等
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 推理脚本
        ├── dataset_process.py               // 数据集预处理脚本
        ├── export.py                        // 将权重文件冻结为 MINDIR 等格式的脚本
        ├── transfer_pth.py                  // 将pth权重文件转换为ckpt权重的脚本
        ├── postprocess.py                   // 310精度计算脚本
        ├── preprocess.py                    // 310预处理脚本
```

### 脚本参数

```tex
模型训练、推理、冻结等操作及模型部署环境的参数均在 default_config.yaml 文件中进行配置。
关键参数（针对 UCF101 数据集的迁移学习）默认如下：
num_classes: 101
layer_num: 34
epochs: 30
batch_size: 8
lr: 0.001
momentum: 0.9
weight_decay: 0.0005
```

### 数据集预处理

由于获取到的原始数据集为视频格式（网络处理的输入为图片格式），而且很可能还没有分类，因此执行模型训练和推理之前需要进行数据集预处理。数据集预处理脚本为 ”dataset_process.py“，其参数在 default_config.yaml 文件中进行配置，本节仅涉及其中的 “splited” ，“source_data_dir”，“output_dir” 三个参数。

注意：数据集预处理需要的 decord 包在 aarch64 架构的机器上无法安装，请使用 x86 架构的机器。

- Kinetics400 数据集的预处理

  通过[官网](https://deepmind.com/research/open-source/kinetics)下载数据集后，我们会得到一个未分类的训练集和一个已分类的验证集，请使用官方脚本对训练集进行分类，然后按如下目录结构组织数据集：

  ```tex
  ├── xxx/kinetics400/                      // 数据集根目录
      │   ├──train                          // 训练集
      │   │   ├── abseiling
      │   │   ├── air_drumming
      │   │   ├── ....
      │   ├──val                             // 验证集
      │   │   ├── abseiling
      │   │   ├── air_drumming
      │   │   ├── ....
  ```

  然后将 default_config.yaml 文件中的 “splited” ，“source_data_dir”，“output_dir” 参数依次设置为 1、“xxx/kinetics400/”、“./kinetics400_img”，最后执行 python dataset_process.py ，等待执行完毕即可。

- UCF101 数据集的预处理

  通过[此处](https://www.kaggle.com/pevogam/ucf101)下载数据集后，我们会得到一个已分类的视频数据集，但是该数据集没有被进一步分割为训练集和验证集。请按如下目录结构组织数据集：

  ```tex
  ├── xxx/ucf101/                            // 数据集根目录
      │   ├──ApplyEyeMakeup
      │   ├──ApplyLipstick
      │   ├──....
  ```

  然后将 default_config.yaml 文件中的 “splited” ，“source_data_dir”，“output_dir” 参数依次设置为 0、“xxx/ucf101/”、“./ucf101_img”，最后执行 python dataset_process.py ，等待执行完毕即可。

### 预训练模型转换

请从[此处](https://cv.gluon.ai/model_zoo/action_recognition.html)下载训练网络所需的预训练模型 [r2plus1d_v1_resnet34_kinetics400.ckpt](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/torch/models/r2plus1d_v1_resnet34_kinetics400-5102fd17.pth)，然后执行 transfer_pth.py 脚本，将该模型转换为 mindspore 所需的格式。

```python
#请将下载好的预训练模型 r2plus1d_v1_resnet34_kinetics400-5102fd17.pth 放在 transfer_pth.py 同级目录下，然后执行下述命令
python transfer_pth.py
```

### 训练过程

#### 训练

- ModelArts环境运行

  以UCF101数据集为例，经过前述 “数据集预处理”步骤后，我们会得到如下结构的数据集：

  ```tex
  ├── xxx/ucf101/                            // 数据集根目录
      │   ├──train                           // 训练集
      │   │   ├── ApplyEyeMakeup
      │   │   ├── ApplyLipstick
      │   │   ├── ....
      │   ├──val                             // 验证集
      │   │   ├── ApplyEyeMakeup
      │   │   ├── ApplyLipstick
      │   │   ├── ....
  ```

  请将根目录 "xxx/ucf101/" 直接压缩为 zip 格式的压缩包，并命名为 ucf101_img.zip，然后上传到 OBS 桶中。

  您的 OBS 桶应该按如下结构组织：

  ```tex
  ├── xxx/R2plus1D/                                  // 根目录
  │   ├──code                                        // 代码目录
  │   │   ├──README_CN.md                            // R(2+1)D 的说明文件
  │   │   ├──scripts
  │   │   │  ├──run_distribute_train_ascend.sh        // Ascend 8卡训练脚本
  │   │   │  ├──run_eval_ascend.sh                    // Ascend 推理启动脚本
  │   │   │  ├──run_train_ascend.sh                   // Ascend 单卡训练启动脚本
  │   │   ├──src
  │   │   │  ├──config.py                             // 配置加载文件
  │   │   │  ├──dataset.py                            // 数据集处理
  │   │   │  ├──models.py                             // 模型结构
  │   │   │  ├──logger.py                             // 日志打印文件
  │   │   │  ├──utils.py                              // 工具类
  │   │   ├──default_config.yaml                      // 默认配置信息，包括训练、推理、模型冻结等
  │   │   ├──train.py                                 // 训练脚本
  │   │   ├──eval.py                                  // 推理脚本
  │   │   ├──dataset_process.py                       // 数据集预处理脚本
  │   │   ├──export.py                                // 将权重文件冻结为 MINDIR 等格式的脚本
  │   │   ├──transfer_pth.py                          // 将pth权重文件转换为ckpt权重的脚本
  │   ├──dataset
  │   │   ├──ucf101_img.zip                           // 数据集
  │   ├──pretrain
  │   │   ├──r2plus1d_v1_resnet34_kinetics400.ckpt    // 转换好的预训练模型
  │   ├──outputs                                      // 存储训练日志等文件
  ```

  然后进入ModelArts控制台 ---> 训练管理 ---> 训练作业 ---> 创建，具体参数如下：

  算法来源： 常用框架

  ​     AI引擎  Ascend-Powered-Engine MindSpore-1.3.0-c78-python3.7-euleros2.8-aarch64
  ​     代码目录 /xxx/R2plus1D/code
  ​     启动文件 /xxx/R2plus1D/code/train.py

  数据来源：数据存储位置

  ​     数据存储位置：/xxx/R2plus1D/dataset

  训练输出位置：/xxx/R2plus1D/outputs

  运行参数：

  ​      train_url：/xxx/R2plus1D/outputs

  ​      data_url：/xxx/R2plus1D/dataset

  ​      use_modelarts：1

  ​      outer_path：obs://xxx/R2plus1D/outputs

  ​      dataset_root_path：obs://xxx/R2plus1D/dataset

  ​      pack_file_name：ucf101_img.zip

  ​      pretrain_path：obs://xxx/R2plus1D/pretrain/

  ​      ckpt_name：r2plus1d_v1_resnet34_kinetics400.ckpt

  ​      is_distributed：0

  作业日志路径：/xxx/R2plus1D/outputs

  然后选择规格："Ascend : 1 * Ascend-910 CPU：24 核 256GiB"  创建训练任务即可。

  如果您想创建8卡训练任务，只需要将上述运行参数中的 "is_distributed" 设置为 1，然后选择规格 “Ascend : 8 * Ascend-910 CPU：192 核 2048GiB” 创建训练任务即可。

- Ascend处理器环境运行

  ```bash
  ###参数配置请修改 default_config.yaml 文件
  #通过 python 命令行运行Ascend单卡训练脚本。
  python train.py --is_distributed=0 --device_target=Ascend > train_ascend_log.txt 2>&1 &

  #通过 bash 命令启动Ascend单卡训练。
  bash ./scripts/run_train_ascend.sh device_id
  e.g. bash ./scripts/run_train_ascend.sh 0

  #Ascend多卡训练
  bash ./scripts/run_distribute_train_ascend.sh rank_size rank_start_id rank_table_file
  e.g. bash ./scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json
  #Ascend多卡训练将会在代码根目录创建ascend_work_space文件夹，并在该工作目录下独立运行、保存相关训练信息。
  ```

  训练完成后，您可以在 output_path 参数指定的目录下找到保存的权重文件，训练过程中的部分 loss 收敛情况如下（8卡并行）：

  ```tex
  ###参数配置请修改 default_config.yaml 文件
  ......
  epoch time: 214265.706 ms, per step time: 1290.757 ms
  epoch: 20 step: 6, loss is 0.004272789
  epoch: 20 step: 16, loss is 0.062011678
  epoch: 20 step: 26, loss is 1.2200212
  epoch: 20 step: 36, loss is 0.20649293
  epoch: 20 step: 46, loss is 0.110879965
  epoch: 20 step: 56, loss is 0.019843677
  epoch: 20 step: 66, loss is 0.0016696296
  epoch: 20 step: 76, loss is 0.028821332
  epoch: 20 step: 86, loss is 0.022604007
  epoch: 20 step: 96, loss is 0.050388362
  epoch: 20 step: 106, loss is 0.03981915
  epoch: 20 step: 116, loss is 1.77048
  epoch: 20 step: 126, loss is 0.46865237
  epoch: 20 step: 136, loss is 0.006930205
  epoch: 20 step: 146, loss is 0.01725213
  epoch: 20 step: 156, loss is 0.15757804
  epoch: 20 step: 166, loss is 0.77281004
  epoch time: 185916.069 ms, per step time: 1119.976 ms
  [WARNING] MD(19783,fffdd97fa1e0,python):2021-12-02-16:04:30.120.203 [mindspore/ccsrc/minddata/dataset/engine/datasetops/device_queue_op.cc:725] DetectPerBatchTime] Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset pipeline, which might result `GetNext` timeout problem. You may test dataset processing performance(with creating dataset iterator) and optimize it.
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.416 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.498 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.525 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.546 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.566 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.585 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] SESSION(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.736 [mindspore/ccsrc/backend/session/ascend_session.cc:1205] SelectKernel] There has 1 node/nodes used reduce precision to selected the kernel!
  2021-12-02 16:09:50,480 :INFO: epoch: 20, accuracy: 97.52252
  2021-12-02 16:09:54,156 :INFO: update best result: 97.52252
  2021-12-02 16:10:47,968 :INFO: update best checkpoint at: ./output/2021-12-02_time_14_41_28/0_best_map.ckpt
  epoch: 21 step: 10, loss is 0.30040452
  epoch: 21 step: 20, loss is 0.04393909
  epoch: 21 step: 30, loss is 0.26733813
  epoch: 21 step: 40, loss is 0.35622913
  epoch: 21 step: 50, loss is 0.14869432
  epoch: 21 step: 60, loss is 0.45824617
  epoch: 21 step: 70, loss is 0.031756364
  epoch: 21 step: 80, loss is 0.07024868
  epoch: 21 step: 90, loss is 0.3892364
  epoch: 21 step: 100, loss is 3.364152
  epoch: 21 step: 110, loss is 0.48548156
  epoch: 21 step: 120, loss is 2.4292169
  epoch: 21 step: 130, loss is 0.24383453
  epoch: 21 step: 140, loss is 0.31997812
  epoch: 21 step: 150, loss is 0.0057518715
  epoch: 21 step: 160, loss is 0.009464129
  epoch time: 62759.866 ms, per step time: 378.071 ms
  ......
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于推理评估的权重文件路径是否正确。

- Ascend处理器环境运行

  ```bash
  ### 参数配置请修改 default_config.yaml 文件
  #  default_config.yaml 文件中的 eval_ckpt_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
  # 通过 python 命令行运行Ascend推理脚本。
  python eval.py --device_target=Ascend > eval_ascend_log.txt 2>&1 &
  #通过 bash 命令启动Ascend推理。
  bash ./scripts/run_eval_ascend.sh device_id
  e.g. bash ./scripts/run_eval_ascend.sh 0
  ```

  运行完成后，您可以在 output_path 指定的目录下找到推理运行日志。部分推理日志如下：

  ```tex
  2021-12-01 15:41:35,434:INFO:Args:
  2021-12-01 15:41:35,434:INFO:--> use_modelarts: 0
  2021-12-01 15:41:35,434:INFO:--> data_url:
  2021-12-01 15:41:35,434:INFO:--> train_url:
  2021-12-01 15:41:35,434:INFO:--> outer_path: s3://output/
  2021-12-01 15:41:35,434:INFO:--> num_classes: 101
  2021-12-01 15:41:35,434:INFO:--> layer_num: 34
  2021-12-01 15:41:35,434:INFO:--> epochs: 30
  2021-12-01 15:41:35,435:INFO:--> batch_size: 8
  2021-12-01 15:41:35,435:INFO:--> lr: 0.001
  2021-12-01 15:41:35,435:INFO:--> momentum: 0.9
  2021-12-01 15:41:35,435:INFO:--> weight_decay: 0.0005
  2021-12-01 15:41:35,435:INFO:--> dataset_root_path: /opt/npu/data/R2p1D/dataset/ucf101_img/
  2021-12-01 15:41:35,435:INFO:--> dataset_name: ucf101
  2021-12-01 15:41:35,435:INFO:--> val_mode: val
  2021-12-01 15:41:35,435:INFO:--> pack_file_name:
  2021-12-01 15:41:35,435:INFO:--> eval_while_train: 1
  2021-12-01 15:41:35,435:INFO:--> eval_steps: 1
  2021-12-01 15:41:35,435:INFO:--> eval_start_epoch: 20
  2021-12-01 15:41:35,435:INFO:--> save_every: 1
  2021-12-01 15:41:35,436:INFO:--> is_save_on_master: 1
  2021-12-01 15:41:35,436:INFO:--> ckpt_save_max: 5
  2021-12-01 15:41:35,436:INFO:--> output_path: ./output/
  2021-12-01 15:41:35,436:INFO:--> pretrain_path: /opt/npu/data/R2p1D/code_check/
  2021-12-01 15:41:35,436:INFO:--> ckpt_name: r2plus1d_v1_resnet34_kinetics400.ckpt
  2021-12-01 15:41:35,436:INFO:--> resume_path:
  2021-12-01 15:41:35,436:INFO:--> resume_name:
  2021-12-01 15:41:35,436:INFO:--> resume_epoch: 0
  2021-12-01 15:41:35,436:INFO:--> eval_ckpt_path: /opt/npu/data/R2p1D/code_check/
  2021-12-01 15:41:35,436:INFO:--> eval_ckpt_name: r2plus1d_best_map.ckpt
  2021-12-01 15:41:35,436:INFO:--> export_batch_size: 1
  2021-12-01 15:41:35,436:INFO:--> image_height: 112
  2021-12-01 15:41:35,437:INFO:--> image_width: 112
  2021-12-01 15:41:35,437:INFO:--> ckpt_file: ./r2plus1d_best_map.ckpt
  2021-12-01 15:41:35,437:INFO:--> file_name: r2plus1d
  2021-12-01 15:41:35,437:INFO:--> file_format: MINDIR
  2021-12-01 15:41:35,437:INFO:--> source_data_dir: /data/dataset/UCF-101/
  2021-12-01 15:41:35,437:INFO:--> output_dir: ../dataset/ucf101_img/
  2021-12-01 15:41:35,437:INFO:--> splited: 0
  2021-12-01 15:41:35,437:INFO:--> device_target: Ascend
  2021-12-01 15:41:35,437:INFO:--> is_distributed: 0
  2021-12-01 15:41:35,437:INFO:--> rank: 0
  2021-12-01 15:41:35,437:INFO:--> group_size: 1
  2021-12-01 15:41:35,437:INFO:--> config_path: /opt/npu/data/R2p1D/code_check/src/../default_config.yaml
  2021-12-01 15:41:35,438:INFO:--> save_dir: ./output/2021-12-01_time_15_41_35
  2021-12-01 15:41:35,438:INFO:--> logger: <LOGGER R2plus1D (NOTSET)>
  2021-12-01 15:41:35,438:INFO:
  2021-12-01 15:41:37,742:INFO:load validation weights from /opt/npu/data/R2p1D/code_check/r2plus1d_best_map.ckpt
  2021-12-01 15:41:46,801:INFO:loaded validation weights from /opt/npu/data/R2p1D/code_check/r2plus1d_best_map.ckpt
  2021-12-01 15:41:48,838:INFO:cfg.steps_per_epoch: 333
  [WARNING] DEVICE(11897,fffe56ffd1e0,python):2021-12-01-15:42:03.243.372 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[OneHot] reduce precision from int64 to int32
  [WARNING] DEVICE(11897,fffe56ffd1e0,python):2021-12-01-15:42:03.243.467 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[OneHot] reduce precision from int64 to int32
  [WARNING] SESSION(11897,fffe56ffd1e0,python):2021-12-01-15:42:06.867.318 [mindspore/ccsrc/backend/session/ascend_session.cc:1806] SelectKernel] There are 1 node/nodes used reduce precision to selected the kernel!
  2021-12-01 15:44:53,346:INFO:Final Accuracy：{'top_1_accuracy': 0.9786036036036037, 'top_5_accuracy': 0.9981231231231231}
  2021-12-01 15:44:53,347:INFO:validation finished....
  2021-12-01 15:44:53,456:INFO:All task finished!
  ```

## Mindir推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出Mindir

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
e.g. python export.py --ckpt_file ./r2plus1d_best_map.ckpt --file_name r2plus1d --file_format MINDIR
```

- `ckpt_file`为必填项。
- `file_format` 必须在 ["AIR", "MINDIR"]中选择。

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```bash
# Ascend310 inference
bash run_infer_310.sh [model_path] [data_path] [out_image_path]
e.g. bash run_infer_310.sh ../r2plus1d.mindir ../dataset/ ../outputs
```

- `model_path` mindir文件所在的路径。
- `data_path` 数据集所在路径（该路径下可以只含有预处理过的val目录）。
- `out_image_path` 存储预处理数据的目录，不存在会自动创建。

### 结果

推理结果保存在 ascend310_infer 文件夹下，您可以在acc.log中看到以下精度计算结果。

```tex
Accuracy: 0.9774774774774775
```

## 模型描述

### 性能

#### 评估性能

Validation for R(2+1)D

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G             |
| uploaded Date              | 11/27/2021 (month/day/year)                                  |
| MindSpore Version          | 1.5.0                                                        |
| Dataset                    | UCF101                                                       |
| Training Parameters        | num_classes=101, layer_num=34, epochs=30, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005 |
| Optimizer                  | SGD                                                          |
| Loss Function              | SoftmaxCrossEntropyWithLogits                                |
| outputs                    | 输入视频属于各类别的概率                                     |
| Loss                       | 0.2231637                                                    |
| Accuracy                   | top_1=0.9786, top_5=0.9981                                   |
| Total time                 | 8p：1h58m (without validation), 1p：3h (without validation)  |
| Checkpoint for Fine tuning | 8p: 706MB(.ckpt file)                                        |
| Scripts                    | [R(2+1)D脚本](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d) |

## 随机情况说明

train.py 、 eval.py 和 preprocess.py 中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
