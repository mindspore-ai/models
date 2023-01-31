# 目录 <!-- TOC -->

- [目录](#目录-)
- [PvNet描述](#pvnet描述)
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
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [LINEMOD上的PVNet](#linemod上的pvnet)
            - [](#)
        - [推理性能](#推理性能)
            - [LINEMOD上的PvNet](#linemod上的pvnet-1)
    - [性能说明](#性能说明)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PvNet描述

PvNet是2019年浙江大学CAD&CG国家重点实验室的一篇 6D Pose Estimation领域的CVPR oral论文。对于6D Pose Estimation任务，其目标是检测出物体在3D空间的位置和姿态，随着近年来计算机视觉算法的提升，对3D空间中物体状态的检测越来越受到关注，2018 ECCV会议的最佳论文奖也授予给了6D Pose Estimation领域的论文。Pvnet提出了一种基于向量场投票的方法来预测关键点的位置，即每个像素预测一个指向物体关键点的方向向量，其相比于其它方法对遮挡截断物体的估计效果鲁棒性有极大提升。

[论文](https://zju3dv.github.io/pvnet/)：Sida Peng, Y. Liu, Qixing Huang, Hujun Bao, Xiaowei Zhou."PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation."*2018, 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

# 模型架构

PvNet是一种Encode-Decode的网络结构，通过输入一张rgb图，输出目标物体的语义分割及指向物体关键点的向量场，随后通过Ransac Voting的方法从方向向量场中计算出物体的关键点。

# 数据集

使用的数据集：[LINEMOD](https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet)

- 数据集大小：1.8G，共13个物体

    - 训练集/测试集：划分请参照数据下train.txt/val.txt/test.txt ，不同物体图像数量有所差异。

使用的数据集：[LINEMOD_ORIG](https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%5FORIG%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet)

- 数据集大小：3.8G，共13个物体

    合成数据集共1万张图像，已包含于LINEMOD链接中。
    渲染数据集为13个物体，每个包含1万张图像，生成渲染数据方法请参照如下：

    - [pvnet-rendering](https://github.com/zju3dv/pvnet-rendering)

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
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

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 数据及模型准备

  ```python
  # 原始数据需生成对应真实数据/渲染数据/合成数据的pickle文件
  # 转换脚本参照model_utils/generateposedb.py
  python model_utils/generateposedb.py

  # 原始数据需转换为MindRecord格式数据
  # 转换脚本参照model_utils/data2mindrecord.py
  python model_utils/data2mindrecord.py

  # 下载resnet18预训练模型并转换为mindspore格式
  # pytorch官方版本resnet18[http://download.pytorch.org/models/resnet18-5c106cde.pth]
  # 转换脚本参照model_utils/pth2ms.py
  python model_utils/pth2ms.py
  ```

- Ascend处理器环境运行

  ```text
  # 添加数据集路径,以训练LINEMOD为例
  data_url:"/data/bucket-4609/dataset/pvnet/data2mindrecord/"

  # 保存模型文件路径,以训练LINEMOD为例
  train_url:"/data/bucket-4609/dataset/pvnet/trained/"

  # 添加训练物体名称
  cls_name:"cat"

  # 添加数据集名称
  dataset_name:"LINEMOD"

  # 添加预训练模型
  # 预训练模型置于根目录下，设置为None则不使用预训练模型
  pretrained_path:"./resnet18-5c106cde.ckpt"

  # 推理前添加checkpoint路径参数
  ckpt_file:"./model/pvnet-199_681.ckpt"
  ```

  ```python
  # 运行训练示例
  python train.py --cls_name=ape > train.log 2>&1 &

  # 运行分布式训练示例
  export RANK_SIZE=8
  bash scripts/run_distribute.sh --cls_name ape --distribute 1 --data_url= ~/pvnet/data2mindrecord/ --train_url= ~/pvnet/trained/

  # 运行评估示例
  bash scripts/run_eval.sh

  # 运行推理示例
  bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [CLS_NAME] [DEVICE_ID]
  ```

默认使用LINEMOD数据集。如需查看更多详情，请参考指定脚本。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 LINEMOD 数据集

      ```python
      # (1) 执行a或者b
      #       a. 在 pvnet_linemod_config.yaml 文件中设置distribute=1, 并设置其它参数如
      #          cls_name,batch_size,data_url,train_url
      #       b. 在网页上设置 "distribute=1"
      #          在网页上设置 "train_url=/bucket-xxxx/linemod/trained/"
      #          在网页上设置 "data_url=/bucket-xxxx/linemod/dataset/"
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "~/pvnet"
      # (5) 在网页上设置启动文件为 "~/pvnet/train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                            // 所有模型相关说明
    ├── pvnet
        ├── README.md                        // pvnet相关说明
        ├── ascend310_infer                  // 实现310推理源代码

        ├── model_utils
        │   ├──config.py                     // 读取配置文件脚本
        │   ├──data_file_utils.py            // 基础工具函数脚本
        │   ├──data2mindrecord.py            // 原始数据到mindrecord格式转换脚本
        │   ├──generate_posedb.py            // 原始数据生成pickle文件脚本
        │   ├──pth2ms.py                     // 预训练模型转换mindspore脚本

        ├── scripts
        │   ├──run_distribute.sh             // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh                   // Ascend评估的shell脚本
        │   ├──run_infer_310.sh              // Ascend推理shell脚本

        ├── src
        │   ├──lib
        │   │   ├──voting                   // ransac voting相关代码
        │   ├──dataset.py                   // 数据集相关脚本
        │   ├──evaluation_dataset.py        // 评估数据集脚本
        │   ├──evaluation_utils.py          // 评估工具脚本
        │   ├──loss_scale.py                // 动态loss_scale脚本
        │   ├──model_reposity.py            // 网络模型脚本
        │   ├──net_utils.py                 // 网络工具脚本
        │   ├──resnet.py                    // 特征提取网络脚本

        ├── train.py                        // 训练脚本
        ├── eval.py                         // 评估脚本
        ├── postprogress.py                 // 310推理后处理脚本
        ├── export.py                       // 将checkpoint文件导出mindir
        ├── pvnet_linemod_config.yaml       // 参数配置yaml文件
        ├── requirements.txt                // 依赖库说明

```

## 脚本参数

在pvnet_linemod_config.yaml中可以同时配置训练参数和评估参数。

- 配置PvNet和LINEMOD数据集。

  ```python
  'data_url': "./pvnet/"                           # 训练数据集的绝对全路径
  'train_url': "./trained/"                        # 训练模型保存路径
  'group_size': 1                                  # 训练总卡数
  'rank': 0                                        # 当前训练卡号
  'device_target': "Ascend"                        # 运行设备
  'distribute': False                              # 是否分布式训练
  'cls_name': "cat"                                # 训练物体类别
  'vote_num': 9                                    # 投票关键点数量
  'workers_num': 16                                # 多线程数
  'batch_size': 16                                 # 训练批次大小
  'epoch_size': 200                                # 训练轮次
  'learning_rate': 0.005                           # 训练学习率
  'learning_rate_decay_epoch': 20                  # 学习率衰减轮次
  'learning_rate_decay_rate': 0.5                  # 学习率衰减倍数
  'pretrained_path': "./resnet18-5c106cde.ckpt"    # 预训练模型路径
  'loss_scale_value': 1024                         # 动态loss_scale初始值
  'scale_factor': 2                                # 动态loss_scale倍数
  'scale_window': 1000                             # 动态loss_scale 更新频率
  'dataset_name': "LINEMOD"                        # LINEMOD训练数据集名称
  'dataset_dir': "~/pvnet/data/"                   # LINEMOD数据集路径
  'origin_dataset_name': "LINEMOD_ORIG"            # LINEMOD原始数据集名称
  'img_width': 640                                 # 数据集图片宽
  'img_height': 480                                # 数据集图片高
  'ckpt_file': "./train_cat-199_618.ckpt"          # 模型保存文件
  'eval_dataset': "./"                             # 评估数据集路径
  'result_path': "./scripts/result_Files"          # 310推理结果保存路径
  'file_name': "pvnet"                             # 生成mindir文件前缀
  'file_format': "MINDIR"                          # 推理模型转换格式
  'keep_checkpoint_max': 10                        # 最大模型保存数量
  'img_crop_size_width': 480                       # 数据增强图片宽
  'img_crop_size_height': 360                      # 数据增强图片高
  'rotation': True                                 # 数据增强图像是否旋转
  'rot_ang_min': -30                               # 物体旋转的角度范围
  'rot_ang_max': 30
  'crop': True                                     # 数据增强图像是否进行裁剪
  resize_ratio_min: 0.8                            # 图像缩放的比例的范围
  resize_ratio_max: 1.2
  overlap_ratio: 0.8                               # 目标占物体的比例
  brightness: 0.1                                  # 调整图像亮度的参数
  contrast: 0.1                                    # 调整图像对比度的参数
  saturation: 0.05                                 # 调整图像饱和度的参数
  hue: 0.05                                        # 调整图像色度的参数
  ```

更多配置细节请参考文件`pvnet_linemod_config.yaml`。

### 训练过程

- Ascend处理器环境运行

  ```bash
  # 单机训练
  python train.py >train.log
  ```

  可在配置文件pvnet_linemod_config.yaml中修改相关配置，如rank, cls_name。

  运行上述python命令后，您可以通过`train.log`文件查看结果 。

  ```bash
  # 分布式训练
  Usage：bash scripts/run_distribute.sh --cls_name [cls_name] --distribute [distribute]
  #example: bash ./scripts/run_distribute.sh --cls_name ape --distribute 1
  ```

  需在run_distribute.sh里设置可训练卡数RANK_SIZE，其余参数在配置文件pvnet_linemod_config.yaml中修改。

  上述shell脚本将在后台运行分布训练。您可以通过device[X]/train.log文件查看结果。采用以下方式达到损失值：

  ```bash
  # grep "total" device[X]/train.log
  Rank:0/2, Epoch:[1/200], Step[80/308] cost:0.597510814666748.s total:0.28220612
  Rank:0/2, Epoch:[1/200], Step[160/308] cost:0.41454052925109863.s total:0.20701535
  Rank:0/2, Epoch:[1/200], Step[240/308] cost:0.2790074348449707.s total:0.15037575
  ...
  Rank:1/2, Epoch:[1/200], Step[80/308] cost:1.0746071338653564.s total:0.27446517
  Rank:1/2, Epoch:[1/200], Step[160/308] cost:1.1847755908966064.s total:0.20768473
  Rank:1/2, Epoch:[1/200], Step[240/308] cost:0.9300284385681152.s total:0.13899626
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估LINEMOD数据集

  在运行以下命令之前，请检查用于评估的参数，需要修改的配置项为 cls_name和 ckpt_file。请将检查点路径设置为绝对全路径，例如"/username/dataset/cat/train_cat-199_618.ckpt" 。

  ```bash
  bash scripts/run_eval.sh
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件pvnet_linemod_config.yaml.
需要修改的配置项为 cls_name, file_name和 ckpt_file.

```shell
python export.py
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 使用LINEMOD数据集进行推理

  在执行下面的命令之前，我们需要先修改配置文件。修改的项包括cls_name,eval_dataset和result_path。

  推理的结果保存在scripts目录下，在postprocess.log日志文件中可以找到类似以下的结果。

  ```shell
  # Run inference
  bash scripts/run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [CLS_NAME] [DEVICE_TYPE] [DEVICE_ID]
  # example:bash scripts/run_infer_cpp.sh ./can.mindir ./LINEMOD/can/JPEGImages/ can Ascend 0
  Processing object:can, 2D projection error:0.9960629921259843, ADD:0.8622047244094488
  ```

# 模型描述

## 性能

### 评估性能

#### LINEMOD上的PVNet

| 参数                    | Ascend                                                      |
| ----------------------- | ----------------------------------------------------------- |
| 模型版本                | PvNet                                                       |
| 资源                    | Ascend 910；CPU 2.60GHz，192核；内存 720G；系统 Euler2.8    |
| 上传日期                | 2021-12-25                                                  |
| MindSpore版本           | 1.5.0                                                       |
| 数据集                  | LINEMOD                                                     |
| 训练参数                | epoch=200, batch_size = 16, lr=0.0005                       |
| 优化器                  | Adam                                                        |
| 损失函数                | Smoth L1 Loss，SoftmaxCrossEntropyWithLogits                |
| 输出                    | 分割概率及投票向量场                                        |
| 损失                    | 0.005                                                       |
| 速度                    | 990毫秒/步（8卡）                                           |
| 总时长                  | 547分钟（8卡）                                              |
| 参数                    | 148.5M(.ckpt文件)                                           |

####

### 推理性能

#### LINEMOD上的PvNet

| 参数                | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本            | PvNet                       |
| 资源                |  Ascend 310；系统 Euler2.8  |
| 上传日期            | 2021-12-25                  |
| MindSpore 版本      | 1.5.0                       |
| 数据集              | 4类物体，每类约1000张图像   |
| batch_size          | 1                           |
| 输出                | 2D projection 达标率，ADD达标率 |
| 准确性              | 2D projection 达标率：单卡:  99.5%;  8卡：99.7%；ADD达标率：单卡: 70.4 %;  8卡：66.7% |
| 推理模型            | 49.6M (.mindir文件)         |

## 性能说明

本文档提供LINEMOD数据集四类物体(cat/ape/cam/can)的性能验证，PvNet网络其本身已具备足够的泛化性，对于LINEMOD数据集其余物体，用户可参照已提供四类物体，修改相应配置参数如物体名称即可训练。

# 随机情况说明

train.py中设置了随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
