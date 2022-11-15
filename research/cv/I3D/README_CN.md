# 目录

<!-- TOC -->

- [目录](#目录)
- [i3d描述](#i3d描述)
- [模型架构](#模型架构)
- [数据集以数据预处理](#数据集及数据预处理)
- [预训练模型转换](#预训练模型转换)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [导出过程](#导出过程)
    - [推理过程](#推理过程)
    - [onnx模型导出与推理](#onnx模型导出与推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [modelzoo主页](#modelzoo主页)
 <!-- /TOC -->

# i3d描述

于2017年提出的I3D是由3D卷积+Two-Stream方法的结合形成一个新的网络架构。I3D（Two-Stream Inflated 3D ConvNets）模型是由2DCNN Inception-V1扩张而来，并且可以使用在ImageNet上预训练的参数，实验结果表明这个模型在各个标准数据集上都取得了当时最好的结果。另外论文中还公布了新的Human Action Video 数据：Kinetics，它有400个人类动作类以及每个类超过400个片段，收集自现实的、具有挑战性的网络视频。

Carreira J ,  Zisserman A . Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset[J]. IEEE, 2017.
[论文链接](https://arxiv.org/pdf/1705.07750.pdf)

# 模型架构

I3D的模型架构主要涉及 Two-stream构造，使用两个通过ImageNet预训练好的卷积（2D）模型，一个做RGB数据处理，一个做optical flow数据处理。作者根据以上各个模型的优缺点，设计了一个基于3D卷积的双流模型（Two-stream Inflated 3D ConvNets）（模型结构图见论文PDF）
因为是3D卷积模型，没有像2D卷积那样成熟的预训练参数。作者遍借鉴了成熟的2D卷积网络——Inception-v1，将该网络中的2D换为3D。H，W对应的参数都直接从Inception中获取，但D参数需要通过训练得到。

# 数据集及数据预处理

本代码使用的数据集：

- 请注意，数据集的预处理需要较高性能的CPU和较长的时间，可以直接下载已经处理完成的数据集。[下载链接](https://pan.baidu.com/s/1vMIfc_s4tkr913YK87tbBg) （提取码：ms22）
- 同一数据集的rgb和flow的annotation文件是相同的。
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) 该数据集主要来自电影，还有一小部分来自公共数据库，如搜索引擎和视频网站的视频。该数据集包含6849个剪辑，分为51个动作类别，每个类别至少包含101个剪辑。
- 以下为HMDB51数据集预处理的步骤：

    - 在链接中下载```hmdb51_org1.rar```和```test_train_splits.rar```并解压，得到所有的视频片段和训练测试集的划分。
    - 运行```src/tools```下的```generate_rgb_jpgs.py```，将视频文件转换为rgb图片。```avi_video_dir_path```是原始视频文件的路径，```jpg_video_dir_path```是保存rgb图片的路径。

      ```bash

      python -m generate_rgb_jpgs avi_video_dir_path jpg_video_dir_path hmdb51

      ```

    - 运行```src/tools```下的```convert_rgb_to_flow.py```，将RGB图片转换为flow图片。```RGB_PATH```是rgb图片的路径，```FLOW_PATH```是保存flow图片的路径。注意，如果运行时报错 module 'cv2' has no attribute 'DualTVL1OpticalFlow_create'，请尝试安装较早版本的opencv-python库，如3.x.x.xx的版本。

      ```bash

      python convert_rgb_to_flow.py --rgb_path=[RGB_PATH] --flow_path=[FLOW_PATH]

      ```

    - 运行```src/tools```下的```hmdb51_json.py```生成annotation文件。```annotation_dir_path```即为前面解压的```test_train_splits.rar```，```jpg_video_dir_path```即为上一步得到的rgb图片```dst_json_path```即为生成的json文件的保存路径。**注意，flow图片也使用此json文件，不需要再次生成。**

      ```bash

      python -m hmdb51_json annotation_dir_path jpg_video_dir_path dst_json_path

      ```

    - 运行```src/tools```下的```generate_n_frames.py```生成n_frames文件。```jpg_video_directory```即为之前得到的rgb图片和flow图片的路径。**注意，rgb数据集和flow数据集均需要生成n_frames文件。**

      ```bash

      python generate_n_frames.py jpg_video_directory

      ```

- [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) 该数据集具有101个动作类别，共计13320个视频，其在动作方面提供多样性，并且存在以相机运动、物体外观、物体比例、姿势等方面变化，它也是迄今为止最具挑战性的数据集，这套数据集旨在通过学习和探索新的显示动作以加快行为识别。 该数据集的101个动作类别被分为25组，每组包含4-7个动作视频，同一组的视频会存在相似的场景。
- 以下为UCF101数据集预处理的步骤（与HMDB51相似）：

    - 在链接中下载```UCF101.rar```和```UCF101TrainTestSplits-RecognitionTask.zip```并解压，得到所有的视频片段和训练测试集的划分。
    - 运行```src/tools```下的```generate_rgb_jpgs.py```，将视频文件转换为rgb图片。```avi_video_dir_path```是原始视频文件的路径，```jpg_video_dir_path```是保存rgb图片的路径。

      ```bash
      python -m generate_rgb_jpgs avi_video_dir_path jpg_video_dir_path ucf101
      ```

    - 运行```src/tools```下的```convert_rgb_to_flow.py```，将RGB图片转换为flow图片。```RGB_PATH```是rgb图片的路径，```FLOW_PATH```是保存flow图片的路径。注意，如果运行时报错 module 'cv2' has no attribute 'DualTVL1OpticalFlow_create'，请尝试安装较早版本的opencv-python库，如3.x.x.xx的版本。

      ```bash
      python convert_rgb_to_flow.py --rgb_path=[RGB_PATH] --flow_path=[FLOW_PATH]
      ```

    - 运行```src/tools```下的```ucf101_json.py```生成annotation文件。```annotation_dir_path```即为前面解压的```test_train_splits.rar```，```jpg_video_dir_path```即为上一步得到的rgb图片```dst_json_path```即为生成的json文件的保存路径。**注意，FLOW图片也使用此json文件，不需要再次生成。**

      ```bash
      python -m hmdb51_json annotation_dir_path jpg_video_dir_path dst_json_path
      ```

    - 运行```src/tools```下的```generate_n_frames.py```生成n_frames文件。```jpg_video_directory```即为之前得到的rgb图片和flow图片的路径。**注意，rgb数据集和flow数据集均需要生成n_frames文件。**

      ```bash
      python generate_n_frames.py jpg_video_directory
      ```

- 建议将处理好的数据集按以下目录结构进行整理：

    ```text
    ├── datasets
    │   ├── flow
    │   │   ├─ hmdb51
    │   │   │   ├─ jpg
    │   │   │   │   ├─ brush_hair
    │   │   │   │   ├─ cartwheel
    │   │   │   │   ├─ ...
    │   │   │   ├─ annotation
    │   │   │   │   ├─ hmdb51_1.json
    │   │   │   │   ├─ hmdb51_2.json
    │   │   │   │   ├─ hmdb51_3.json
    │   │   ├─ ucf101
    │   │   │   ├─ jpg
    │   │   │   │   ├─ ApplyEyeMakeup
    │   │   │   │   ├─ ApplyLipstick
    │   │   │   │   ├─ ...
    │   │   │   ├─ annotation
    │   │   │   │   ├─ ucf101_01.json
    │   │   │   │   ├─ ucf101_02.json
    │   │   │   │   ├─ ucf101_03.json
    │   ├── rgb
    │   │   ├─ hmdb51
    │   │   │   ├─ jpg
    │   │   │   │   ├─ brush_hair
    │   │   │   │   ├─ cartwheel
    │   │   │   │   ├─ ...
    │   │   │   ├─ annotation
    │   │   │   │   ├─ hmdb51_1.json
    │   │   │   │   ├─ hmdb51_2.json
    │   │   │   │   ├─ hmdb51_3.json
    │   │   ├─ ucf101
    │   │   │   │   ├─ ApplyEyeMakeup
    │   │   │   │   ├─ ApplyLipstick
    │   │   │   │   ├─ ...
    │   │   │   ├─ annotation
    │   │   │   │   ├─ ucf101_01.json
    │   │   │   │   ├─ ucf101_02.json
    │   │   │   │   ├─ ucf101_03.json
    ```

# 预训练模型转换

- 网盘链接中提供原始Pytorch预训练模型(.pt文件，基于ImageNet训练)和已转换完成的Mindspore预训练模型(.ckpt文件)。[下载链接](https://pan.baidu.com/s/1vMIfc_s4tkr913YK87tbBg) （提取码：ms22）
- 论文原作者提供了由TensorFlow框架在ImageNet数据集上训练得到的预训练模型 [下载链接](https://github.com/deepmind/kinetics-i3d/tree/master/data/checkpoints) 。这里也提供社区的Pytorch预训练模型链接 [下载链接](https://github.com/hassony2/kinetics_i3d_pytorch/tree/master/model )。 这里所用的Pytorch预训练模型是由论文作者所提供的的文件转换而来，由于进行复现时并未进行TensorFlow和Pytorch预训练模型的转换工作，Pytorch预训练模型为直接从社区下载，故不提供转换代码和脚本。
- Pytorch环境下的预训练模型需要经过转换才能被Mindspore代码读取，目前仅支持转换Pytorch下的预训练模型。 运行```src/tools```下的```PRETRAINED_PATH```是Pytorch预训练模型的路径，```SAVE_PATH```是保存转换完成的模型的路径。**注意，此代码只能在同时装有Pytorch和Mindspore的环境下运行。**

    ```bash
    python pt2ckpt.py --pretrained_path=[PRETRAINED_PATH] --save_path=[SAVE_PATH]
    ```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 其余所需的Python包请见requirements.txt。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行（参数详细含义及可选范围请参考[脚本参数](#脚本参数)）

    ```bash
    # 分布式训练
      [rank_size]是所要使用的设备的数量，[rank_table_file]是HCCL配置文件的路径，[dataset]是所用数据集的名称，
      [mode]是训练模式，[num_epochs]是训练epoch数，[video_path]是数据集路径，[annotation_path]是JSON文件路径，[checkpoint_path]是预训练权重的路径，
      其余参数可在config.py中修改。如果出现HCCL配置文件无法找到的情况，请分别尝试相对路径和绝对路径。
      用法：bash ./scripts/run_distribute_train.sh [rank_size] [rank_table_file] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]

    # 单卡训练
      [device_id]是训练设备的编号，[dataset]是所用数据集的名称，[mode]是训练模式，[num_epochs]是训练epoch数，
      [video_path]是数据集路径，[annotation_path]是JSON文件路径，[checkpoint_path]是预训练权重的路径，
      其余参数可在config.py中修改。
      用法：bash ./scripts/run_standalone_train.sh [device_id] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]

    # 运行单模式（RGB/FLOW）评估
      [device_id]是评估设备的编号，[mode]是评估模式，[dataset]是所用数据集的名称，[ckpt_path]是需要评估的模型的路径，
      [video_path]是数据集路径，[annotation_path]是JSON文件路径，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_single_eval.sh [device_id] [mode] [dataset] [ckpt_path] [video_path] [annotation_path]

    # 运行多卡单模式（RGB/FLOW）评估
      [ckpt_start_id]是评估ckpt文件的起始编号，[mode]是评估模式，[dataset]是所用数据集的名称，[video_path]是数据集路径，
      [annotation_path]是JSON文件路径，[output_ckpt_path]是需要评估的模型的上层路径，[train_steps]是模型训练时每个epoch的step，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_distributed_single-mode_eval.sh [ckpt_start_id] [mode] [dataset] [annotation_path] [video_path] [output_ckpt_path] [train_steps]

    # 运行联合模式（RGB+FLOW）评估
      [device_id]是评估设备的编号，[dataset]是所用数据集的名称，[video_path]是rgb数据集路径，[video_path_joint_flow]是flow数据集路径，
      [annotation_path]是rgb数据集的JSON文件路径，[annotation_path_joint_flow]是flow数据集的JSON文件路径，[rgb_path]是需要评估的rgb模型的路径，
      [flow_path]是需要评估的flow模型的路径，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_joint_eval.sh [device_id] [dataset] [video_path] [video_path_joint_flow] [annotation_path] [annotation_path_joint_flow] [rgb_path] [flow_path]
    ```

# 脚本说明

## 脚本及样例代码

- 脚本文件和代码文件的目录结构如下：

    ```text
    └── i3d
        ├── ascend_310_infer                  # 310推理
        │        ├── build.sh                 # 编译推理代码脚本
        │        ├── inc
        │        │   └── utils.h              # 配置文件
        │        └── src
        │            ├── CMakeLists.txt       # CmakeList文件
        │            ├── main.cc              # 推理代码
        │            └── utils.cc             # 推理代码
        ├── README_CN.md                      # 所有模型相关说明
        ├── scripts
        │   ├── run_310_infer.sh              # 用于310推理的shell脚本
        │   ├── run_310_infer_all.sh          # 进行四个模型的310推理的shell脚本
        │   ├── run_preprocess.sh             # 用于前处理的shell脚本（8卡）
        │   ├── run_distributed_single-mode_eval.sh   # 启动多卡单模式（RGB/FLOW）评估，可一次评估8个ckpt文件
        │   ├── run_preprocess_all.sh         # 进行四个模型的前处理的shell脚本
        │   ├── run_distribute_train.sh       # 启动Ascend分布式训练（8卡）
        │   ├── run_standalone_train.sh       # 启动Ascend单机训练（单卡）
        │   ├── run_single_eval.sh            # 启动单模式（RGB/FLOW）评估
        │   ├── run_joint_eval.sh             # 启动联合模式（RGB+FLOW）评估
        │   ├── run_eval_onnx.sh             # 启动联合模式（RGB+FLOW）评估
        ├── src
        │   ├── factory
        │   │   ├── data_factory.py           # 获得数据集对象
        │   │   ├── model_factory.py          # 获得模型对象
        │   ├── getdataset
        │   │   ├── hmdb51.py                 # 对HMDB51数据集进行读取和处理
        │   │   ├── ucf101.py                 # 对UCF101数据集进行读取和处理
        │   ├── pretrained
        │   │   ├── flow_imagenet.ckpt        # flow模式下基于ImageNet的预训练模型（Mindspore）
        │   │   ├── flow_imagenet.pt          # flow模式下基于ImageNet的预训练模型（Pytorch）
        │   │   ├── rgb_imagenet.ckpt         # rgb模式下基于ImageNet的预训练模型（Mindspore）
        │   │   ├── rgb_imagenet.pt           # rgb模式下基于ImageNet的预训练模型（Pytorch）
        │   ├── tools
        │   │   ├── convert_rgb_to_flow.py    # 用于将rgb图片转换为flow图片
        │   │   ├── generate_n_frames.py      # 用于生成数据集的n_frames文件
        │   │   ├── generate_rgb_jpgs.py      # 用于将视频转换至rgb图片
        │   │   ├── hmdb51_json.py            # 用于生成HMDB51数据集的json文件
        │   │   ├── ucf101_json.py            # 用于生成UCF101数据集的json文件
        │   │   ├── pt2ckpt.py                # 用于将预训练模型从Pytorch转换至Mindspore
        │   ├── transforms
        │   │   ├── spatial_transforms.py     # 在空间上对图片进行处理
        │   │   ├── target_transforms.py      # 对数据集标签进行处理
        │   │   ├── temporal_transforms.py    # 在时间线上对图片进行处理
        │   ├── i3d.py                        # I3D模型架构
        │   ├── utils.py                      # 部分用于代码运行的函数
        ├── ma-pre-start.sh                   # 在openI平台上运行时自动运行的脚本
        ├── train.py                          # 训练脚本
        ├── eval.py                           # 评估脚本
        ├── eval__onnx.py                           # 评估脚本
        ├── export.py                         # 推理模型导出脚本
        ├── config.py                         # Ascend训练参数配置
        ├── postprocess.py                    # 后处理
        ├── preprocess.py                     # 前处理
        ├── preprocess_Result                 # 前处理结果
        │        ├── data                     # 前处理后数据
        │        └── label                    # 前处理后标签
    ```

## 脚本参数

在所有sh脚本文件中只能配置部分参数。在config.py中可以配置所有和训练有关的参数，在eval.py中可配置所有和评估有关的参数，在export.py中可配置所有和导出有关的参数。

- config.py

  ```text
  'video_path':''                              # 由视频转换得到的图片文件的路径
  'annotation_path': ''                        # annotation文件的路径
  'save_dir': './output_standalone/'           # 训练结果的保存路径，当为分布式训练时，路径将自动变为'./output_distribute/'
  'mode': 'rgb'                                # 训练模式（'rgb'或'flow'）
  'dataset':'hmdb51'                           # 训练使用的数据集（'hmdb51'或'ucf101'）
  'num_val_samples': 1                         # 每项活动的验证样本数量
  'num_classes': 400                           # 数据集的类别数（HMDB51：51 UCF101：101 当载入训练模型时，此参数设置为400）
  'spatial_size': 224                          # 最终输入模型的图片的高度和宽度
  'train_sample_duration': 64                  # 训练时输入的图片相对于视频持续时间
  'test_sample_duration': 64                   # 测试时输入的图片相对于视频持续时间（这一参数对训练没有影响）
  'checkpoint_path': ''                        # 预训练模型的路径
  'finetune_num_classes': 51                   # 这一参数仅在载入预训练模型是使用，用于微调的类数。使用此参数时，将num_classes设置为400，将此参数设置为数据集对应的类别数。
  'finetune_prefixes': 'logits,Mixed_5'        # 要微调的层的前缀，用逗号分隔
  'dropout_keep_prob': 0.5                     # Dropout保留的概率
  'optimizer': 'adam'                          # 使用的优化器
  'lr': 0.001                                  # 学习率（初始值）
  'lr_de_rate': 0.5                            # 学习率下降的倍数
  'lr_de_epochs': 4                            # 在这些迭代之后，学习率将会下降
  'momentum': 0.9                              # 动量值
  'weight_decay': 1e-8                         # 权重衰减值
  'batch_size': 8                              # 批量大小
  'start_epoch': 0                             # 起始迭代数，仅与微调有关
  'num_epochs': 32                             # 迭代次数
  'checkpoint_frequency': 1                    # 保存模型的频率
  'checkpoints_num_keep': 24                   # 保存模型的数量
  'no_eval': False('store_true')               # 禁用评估
  'sink_mode':False('store_true')              # 启用数据下沉模式
  'distributed': False('store_true')           # 启用分布式训练
  'context': 'gr'                              # mindspore的模式（'gr'：GRAPH_MODE或'py'：PYNATIVE_MODE）
  'device': 'Ascend'                           # 使用的设备的类型
  'device_id': 0                               # 用于训练或评估数据集的设备ID，使用run_distribute_train.sh进行分布式训练时可以忽略
  'num_workers': 16                            # 多线程加载的线程数
  'amp_level': 'O3'                            # 混合精度训练的等级
  'device_num': 8                              # 分布式训练时使用的设备数量
  'has_back': False                            # 调整学习率
  'data_url'                                   # openI平台的默认参数
  'train_url'                                  # openI平台的默认参数
  'openI': False                               # 是否在openI平台上训练
  ```

- eval.py

  ```text
  'video_path':''                              # 由视频转换得到的图片文件的路径（如果是joint模式，则为rgb图片的路径）
  'annotation_path': ''                        # annotation文件的路径（如果是joint模式，则为rgb的json文件的路径）
  'video_path_joint_flow': ''                  # joint模式下flow图片的路径
  'annotation_path_joint_flow': ''             # joint模式下flow的json文件的路径
  'dataset':'hmdb51'                           # 评估使用的数据集（'hmdb51'或'ucf101'）
  'test_mode': 1                               # 评估模式（'rgb'或'flow'或'joint'）
  'flow_path': 400                             # flow模型的路径
  'rgb_path': 224                              # rgb模型的路径
  'num_classes': 51                            # 数据集的类别数（HMDB51：51 UCF101：101 当载入训练模型时，此参数设置为400）
  'num_val_samples': 1                         # 每项活动的验证样本数量
  'batch_size': 8                              # 批量大小
  'test_sample_duration': 64                   # 测试时输入的图片相对于视频持续时间（这一参数对训练没有影响）
  'spatial_size': 224                          # 最终输入模型的图片的高度和宽度
  'context': 'gr'                              # mindspore的模式（'gr'：GRAPH_MODE或'py'：PYNATIVE_MODE）
  'device_target': 'Ascend'                    # 使用的设备的类型
  'device_id': 0                               # 使用的设备的ID
  'num_workers': 16                            # 多线程加载的线程数
  'dropout_keep_prob': 0.5                     # Dropout保留的概率
  'data_url'                                   # openI平台的默认参数
  'ckpt_url'                                   # openI平台的默认参数
  'result_url'                                 # openI平台的默认参数
  'openI': False                               # 是否在openI平台上评估
  ```

- export.py

  ```text
  'checkpoint_path': ''                        # 训练好的模型的路径
  'file_name': 'i3d_minddir'                   # 要导出的文件的名称
  'file_format': 'MINDIR'                      # 导出文件的格式（'AIR'或'MINDIR'）
  'batch_size': 8                              # 批量大小
  'device_target': 'Ascend'                    # 使用的设备的类型
  'device_id': 0                               # 使用的设备的ID
  'mode': 'rgb'                                # 导出模式（'rgb'或'flow'）
  'spatial_size': 224                          # 最终输入模型的图片的高度和宽度
  'test_sample_duration': 64                   # 输入的图片相对于视频持续时间
  'num_classes': 51                            # 数据集的类别数（UCF101：101 HMDB51：51）
  'dropout_keep_prob':0.5                      # Dropout保留的概率
  ```

## 训练过程

### 训练

- Ascend处理器环境运行（参数详细含义及可选范围请参考[脚本参数](#脚本参数)）

    ```bash
    # 分布式训练
      [rank_size]是所要使用的设备的数量，[rank_table_file]是HCCL配置文件的路径，[dataset]是所用数据集的名称，
      [mode]是训练模式，[num_epochs]是训练epoch数，[video_path]是数据集路径，[annotation_path]是JSON文件路径，[checkpoint_path]是预训练权重的路径，
      其余参数可在config.py中修改。如果出现HCCL配置文件无法找到的情况，请分别尝试相对路径和绝对路径。
      用法：bash ./scripts/run_distribute_train.sh [rank_size] [rank_table_file] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]

    # 单卡训练
      [device_id]是训练设备的编号，[dataset]是所用数据集的名称，[mode]是训练模式，[num_epochs]是训练epoch数，
      [video_path]是数据集路径，[annotation_path]是JSON文件路径，[checkpoint_path]是预训练权重的路径，
      其余参数可在config.py中修改。
      用法：bash ./scripts/run_standalone_train.sh [device_id] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]
    ```

- 分布式训练的指令实例如下（通常情况下只需要替换最后三个路径参数，其余可以不变）：

    ```bash
    # UCF101数据集，rgb模式：
      bash ./scripts/run_distribute_train.sh 8 hccl_8p.json ucf101 rgb 40 ./dataset/rgb/ucf101/jpg ./dataset/rgb/ucf101/annotation/ucf101_01.json ./src/pretrained/rgb_imagenet.ckpt  

    # UCF101数据集，flow模式：
      bash ./scripts/run_distribute_train.sh 8 hccl_8p.json ucf101 flow 60 ./dataset/flow/ucf101/jpg ./dataset/flow/ucf101/annotation/ucf101_01.json ./src/pretrained/flow_imagenet.ckpt

    # HMDB51数据集，rgb模式：
      bash ./scripts/run_distribute_train.sh 8 hccl_8p.json hmdb51 rgb 40 ./dataset/rgb/hmdb51/jpg ./dataset/rgb/hmdb51/annotation/hmdb51_1.json ./src/pretrained/rgb_imagenet.ckpt

    # HMDB51数据集，flow模式：
      bash ./scripts/run_distribute_train.sh 8 hccl_8p.json hmdb51 flow 60 ./dataset/flow/hmdb51/jpg ./dataset/flow/hmdb51/annotation/hmdb51_1.json ./src/pretrained/flow_imagenet.ckpt
    ```

    分布式训练需要提前创建JSON格式的HCCL配置文件（即运行脚本所需的[rank_table_file]）。

    具体操作，参见[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) 中的说明。

### 结果

- HMDB51 rgb模式

    ```text
    # 分布式训练结果（8p）
    epoch: 1 step: 56, loss is 14.464891
    epoch time: 308488.012 ms, per step time: 5508.715 ms
    epoch: 2 step: 56, loss is 6.3676977
    epoch time: 127061.488 ms, per step time: 2268.955 ms
    epoch: 3 step: 56, loss is 4.588459
    epoch time: 127578.249 ms, per step time: 2278.183 ms
    epoch: 4 step: 56, loss is 2.4345832
    epoch time: 122955.708 ms, per step time: 2195.638 ms
    epoch: 5 step: 56, loss is 3.6290636
    epoch time: 126541.872 ms, per step time: 2259.676 ms
    ...
    ...
    ```

- HMDB51 flow模式

    ```text
    # 分布式训练结果（8p）
    epoch: 1 step: 56, loss is 15.534042
    epoch time: 288257.892 ms, per step time: 5147.462 ms
    epoch: 2 step: 56, loss is 5.654072
    epoch time: 119883.500 ms, per step time: 2140.777 ms
    epoch: 3 step: 56, loss is 4.9139385
    epoch time: 135799.968 ms, per step time: 2424.999 ms
    epoch: 4 step: 56, loss is 1.7731495
    epoch time: 136061.834 ms, per step time: 2429.676 ms
    epoch: 5 step: 56, loss is 2.97477
    epoch time: 121972.934 ms, per step time: 2178.088 ms
    ...
    ...
    ```

- UCF101 rgb模式

    ```text
    # 分布式训练结果（8p）
    epoch: 1 step: 148, loss is 4.1663394
    epoch time: 371017.286 ms, per step time: 2506.874 ms
    epoch: 2 step: 148, loss is 2.9134295
    epoch time: 238646.173 ms, per step time: 1612.474 ms
    epoch: 3 step: 148, loss is 1.7739947
    epoch time: 228499.503 ms, per step time: 1543.916 ms
    epoch: 4 step: 148, loss is 1.1962743
    epoch time: 241279.213 ms, per step time: 1630.265 ms
    epoch: 5 step: 148, loss is 1.1757829
    epoch time: 232278.085 ms, per step time: 1569.447 ms
    ...
    ...
    ```

- UCF101 flow模式

    ```text
    # 分布式训练结果（8p）
    epoch: 1 step: 148, loss is 3.0665674
    epoch time: 223182.687 ms, per step time: 1507.991 ms
    epoch: 2 step: 148, loss is 2.4819078
    epoch time: 90178.436 ms, per step time: 609.314 ms
    epoch: 3 step: 148, loss is 2.622658
    epoch time: 91545.521 ms, per step time: 618.551 ms
    epoch: 4 step: 148, loss is 0.45071584
    epoch time: 90573.475 ms, per step time: 611.983 ms
    epoch: 5 step: 148, loss is 0.5305861
    epoch time: 92125.260 ms, per step time: 622.468 ms
    ...
    ...
    ```

## 评估过程

### 评估

- Ascend处理器环境运行

```bash
    # 单模式（RGB/FLOW）评估
      [device_id]是评估设备的编号，[mode]是评估模式，[dataset]是所用数据集的名称，[ckpt_path]是需要评估的模型的路径，
      [video_path]是数据集路径，[annotation_path]是JSON文件路径，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_single_eval.sh [device_id] [mode] [dataset] [ckpt_path] [video_path] [annotation_path]

    # 运行多卡单模式（RGB/FLOW）评估
      [ckpt_start_id]是评估ckpt文件的起始编号，[mode]是评估模式，[dataset]是所用数据集的名称，[video_path]是数据集路径，
      [annotation_path]是JSON文件路径，[output_ckpt_path]是需要评估的模型的上层路径，[train_steps]是模型训练时每个epoch的step，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_distributed_single-mode_eval.sh [ckpt_start_id] [mode] [dataset] [annotation_path] [video_path] [output_ckpt_path] [train_steps]

      注意：每次运行结束此脚本后，请及时查看output_eval下summary.txt的评估结果，脚本每次运行时会清空此目录下的所有文件。
      说明：模型的训练时会默认保存多个ckpt文件，rgb模式保存最后8个，flow模式保存最后16个，此脚本可以同时对8个脚本文件进行评估，运行结束后会除了会保存每个ckpt的评估日志外，还会把每个ckpt的评估精度汇总到summary.txt。评估结果保存在output_eval目录下。运行脚本的指令如下：
      HMDB51 rgb：bash ./scripts/run_distributed_single-mode_eval.sh 33 rgb hmdb51 ./rgb/hmdb51/annotation/hmdb51_1.json ./rgb/hmdb51/jpg ./output_distribute/hmdb51_rgb_device0/checkpoints 56
             flow：bash ./scripts/run_distributed_single-mode_eval.sh 45 flow hmdb51 ./flow/hmdb51/annotation/hmdb51_1.json ./flow/hmdb51/jpg ./output_distribute/hmdb51_flow_device0/checkpoints 56
      UCF101 rgb：bash ./scripts/run_distributed_single-mode_eval.sh 33 rgb ucf101 ./rgb/ucf101/annotation/ucf101_01.json ./rgb/ucf101/jpg ./output_distribute/ucf101_rgb_device0/checkpoints 148
             flow：bash ./scripts/run_distributed_single-mode_eval.sh 45 flow ucf101 ./flow/ucf101/annotation/ucf101_01.json ./flow/ucf101/jpg ./output_distribute/ucf101_flow_device0/checkpoints 148
      以第一条指令为例，训练hmdb51的rgb模式保存了8个ckpt文件，epoch数为40，所以保存的模型从33到40一共8个。56为每个epoch的step数，[output_ckpt_path]路径下为8个保存的ckpt文件。flow模式需要两次运行此脚本，当epoch数为60时，起始ckpt编号分别为45和53。

    # 联合模式（RGB+FLOW）评估
      [device_id]是评估设备的编号，[dataset]是所用数据集的名称，[video_path]是rgb数据集路径，[video_path_joint_flow]是flow数据集路径，
      [annotation_path]是rgb数据集的JSON文件路径，[annotation_path_joint_flow]是flow数据集的JSON文件路径，[rgb_path]是需要评估的rgb模型的路径，
      [flow_path]是需要评估的flow模型的路径，其余参数可在eval.py中修改。
      用法：bash ./scripts/run_joint_eval.sh [device_id] [dataset] [video_path] [video_path_joint_flow] [annotation_path] [annotation_path_joint_flow] [rgb_path] [flow_path]

    # 或直接使用以下指令进行评估
      HMDB51 rgb： python eval.py --dataset=hmdb51 --video_path=[rgb jpg PATH] --annotation_path=[json PATH] --rgb_path=[rgb ckpt PATH] --test_mode=rgb --num_classes=51 --device_id=[device id]
             flow：python eval.py --dataset=hmdb51 --video_path=[flow jpg PATH] --annotation_path=[json PATH] --flow_path=[flow ckpt PATH] --test_mode=flow --num_classes=51 --device_id=[device id]
      UCF101 rgb： python eval.py --dataset=ucf101 --video_path=[rgb jpg PATH] --annotation_path=[json PATH] --rgb_path=[rgb ckpt PATH] --test_mode=rgb --num_classes=101 --device_id=[device id]
             flow：python eval.py --dataset=ucf101 --video_path=[flow jpg PATH] --annotation_path=[json PATH] --flow_path=[flow ckpt PATH] --test_mode=flow --num_classes=101 --device_id=[device id]
      rgb+flow HMDB51：python eval.py --dataset=hmdb51 --video_path=[rgb jpg PATH] --annotation_path=[json PATH] --video_path_joint_flow=[flow jpg PATH] --annotation_path_joint_flow=[json PATH] --rgb_path=[rgb ckpt PATH] --flow_path=[flow ckpt PATH] --test_mode=joint --num_classes=51 --device_id=[device id]
               UCF101：python eval.py --dataset=ucf101 --video_path=[rgb jpg PATH] --annotation_path=[json PATH] --video_path_joint_flow=[flow jpg PATH] --annotation_path_joint_flow=[json PATH] --rgb_path=[rgb ckpt PATH] --flow_path=[flow ckpt PATH] --test_mode=joint --num_classes=101 --device_id=[device id]
```

### 结果 (Ascend910 8p)

- Accuracy of HMDB51 （based on Imagenet pre-trained Inception-v1）

| 评估模式   | Mindspore精度 | 论文精度 | 8p训练epoch数 | 保存模型数 |
| --------- | -------------| -------| ------------ |--------- |
| rgb       | 0.524        | 0.498  | 40           | 8        |
| flow      | 0.591        | 0.619  | 60           | 16       |
| joint     | 0.652        | 0.664  | -            | -        |

- Accuracy of UCF101 （based on Imagenet pre-trained Inception-v1）

| 评估模式   | Mindspore精度 | 论文精度 | 8p训练epoch数 | 保存模型数 |
| --------- | -------------| -------| ------------ |--------- |
| rgb       | 0.854        | 0.845  | 40           | 8        |
| flow      | 0.868        | 0.906  | 60           | 16       |
| joint     | 0.925        | 0.934  | -            | -        |

## 导出过程

### 导出

- 参数名称及含义请参考 **脚本参数** 部分，请注意，参数 CHECKPOINT_PATH 为必填项， FILE_FORMAT 必须在 ["AIR", "MINDIR", "ONNX"]中选择，MODE必须在 ["flow", "rgb"]中选择。导出用HMDB51数据集训练的模型时， NUM_CLASS 为51。导出用UCF101数据集训练的模型时， NUM_CLASS 为101。

```bash
python export.py --checkpoint_path=[CHECKPOINT_PATH] --file_name=[FILE_NAME] --file_format=[FILE_FORMAT] --mode=[MODE] --num_class=[NUM_CLASS]
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在310机器中执行推理前，需要通过export.py导出mindir文件

此外，预处理的数据集需要进行前处理，利用生成在preprocess_Result中的clips与对应的labels进行推理

前处理在910上进行，运行如下脚本：

```bash
[rank_size]是所要使用的设备的数量，[rank_table_file]是HCCL配置文件的路径，[dataset]是所用数据集的名称，
[mode]是训练模式，[num_epochs]是训练epoch数，[video_path]是数据集路径，[annotation_path]是JSON文件路径，[checkpoint_path]是预训练权重的路径，
其余参数可在config.py中修改。如果出现文件无法找到的情况，请分别尝试相对路径和绝对路径
用法：bash ./scripts/run_preprocess.sh [rank_size] [rank_table_file] [dataset] [mode] [num_epochs] [video_path] [annotation_path] [checkpoint_path]
```

运行实例如下所示：

```bash
#hmdb51 rgb
bash ./scripts/run_preprocess.sh 8 hccl_8p.json hmdb51 rgb 40 ./rgb/hmdb51/jpg ./rgb/hmdb51/annotation/hmdb51_1.json ./src/pretrained/rgb_imagenet.ckpt
#ucf101 rgb
bash ./scripts/run_preprocess.sh 8 hccl_8p.json ucf101 rgb 40 ./rgb/ucf101/jpg ./rgb/ucf101/annotation/ucf101_01.json ./src/pretrained/rgb_imagenet.ckpt
#hmdb51 flow
bash ./scripts/run_preprocess.sh 8 hccl_8p.json hmdb51 flow 60 ./flow/hmdb51/jpg ./flow/hmdb51/annotation/hmdb51_1.json ./src/pretrained/flow_imagenet.ckpt
#ucf101 flow
bash ./scripts/run_preprocess.sh 8 hccl_8p.json ucf101 flow 60 ./flow/ucf101/jpg ./flow/ucf101/annotation/ucf101_01.json ./src/pretrained/flow_imagenet.ckpt
```

如果出现文件无法找到的情况，请分别尝试相对路径和绝对路径

preprocess_Result应存放于本项目根目录，其下分别有两个子文件夹，data与label，前者存放前处理产生的bin文件，后者存放前处理生成的label的npy文件

data文件的下的hmdb51与ucf101均有flow与rgb两个子文件夹，将相应的bin数据文件均在其中

注：因为受前处理过程的某些特性，在生成的bin文件中会有概率出现一个大小与其他bin文件不同的文件（往往是最后一个文件），如果遇到这种情况，手动删去即可，不然会影响后续的推理流程

preprocess_Result具体的文件结构如下所示：

```text
├── data
│   ├── hmdb51
│   │   ├── flow
│   │   └── rgb
│   └── ucf101
│       ├── flow
│       └── rgb
└── label
```

preprocess_Result在稍后的run_infer_310.sh中会作为标定的文件路径

具体使用该路径的代码在infer()与cal_acc()函数中，如果使用其他路径亦可从中更改

或者，直接运行全部前处理脚本run_proprecess_all:

```bash
bash ./scripts/run_preprocess_all.sh
```

具体的参数可以在该bash文件中进行修改

如果run_preprocess_all.sh正常运行，在script中会生成后一个preprocess_Result文件夹，结构与上文相同：

```text
├── data
│   ├── hmdb51
│   │   ├── flow
│   │   └── rgb
│   └── ucf101
│       ├── flow
│       └── rgb
└── label
```

将结果放置在310机器的项目根目录中即可

随后在310机器中，运行如下推理脚本：

```bash
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET] [DEVICE_ID] [MODE]
```

其中[MINDIR_PATH]是先前导出的mindir模型文件的路径，
[DATASET]是填写推理用到的的数据集（"hmdb51"或"ucf101"），
[DEVICE_ID]是设备编号，
[mode]是推理模式（"rgb"或"flow"）。

即可进行对应模型的推理

以下是310进行hmdb51的rgb模式下的推理部分结果（可在对应的acc_HMDB51_rgb.log中查询）：

```text
...
predictions shape after reshape (8, 51)
label shape: (8,)
predictions shape before reshape (408,)
predictions shape after reshape (8, 51)
label shape: (8,)
eval result: top_1 81.364%
```

也可运行run_infer_310_all.sh执行全部四类模型的推理：

```bash
bash ./scripts/run_infer_310_all.sh [MINDIR_PATH_HMDB51_RGB] [MINDIR_PATH_HMDB51_FLOW] [MINDIR_PATH_UCF101_RGB] [MINDIR_PATH_UCF101_FLOW] [DEVICE_ID]
```

推理的结果可以在相应的log文件中查询得到（infer_hmdb51_rgb、infer_hmdb51_flow.log、infer_ucf101_rgb.log、infer_ucf101_flow.log）

精度结果可以在相应的log文件中查询得到（acc_hmdb51_rgb.log、acc_hmdb51_flow.log、acc_ucf101_rgb.log、acc_ucf101_flow.log）

## onnx模型导出与推理

### onnx模型文件导出

```bash
python export.py --checkpoint_path /path/to/I3D.ckpt --file_name /path/to/I3D --file_format ONNX --batch_size 8 --device GPU --device_id 0 --mode rgb
```

### 评估

```bash
# joint模式
bash ./scripts/run_eval_onnx.sh [test_mode] [device_target] [device_id] [dataset] [video_path] [video_path_joint_flow] [annotation_path] [annotation_path_joint_flow] [rgb_onnx_path] [flow_onnx_path]
# 示例
bash ./scripts/run_eval_onnx.sh joint GPU 0 hmdb51 ./data/rgb/hmdb51/jpg ./data/flow/hmdb51/jpg ./data/rgb/hmdb51/annotation/hmdb51_1.json ./data/flow/hmdb51/annotation/hmdb51_1.json ./i3d_h_rgb.onnx ./i3d_h_flow.onnx
```

```bash
# rgb模式, flow模式请修改对应参数
bash ./scripts/run_eval_onnx.sh [test_mode] [device_target] [device_id] [dataset] [video_path]  [annotation_path] [rgb_onnx_path]
# 示例
bash ./scripts/run_eval_onnx.sh rgb GPU 0 hmdb51 ./data/rgb/hmdb51/jpg ./data/rgb/hmdb51/annotation/hmdb51_1.json ./i3d_h_rgb.onnx
```

# 模型描述

## 性能

### 训练性能

#### HMDB51数据集

| 参数          | Ascend 910                                                   |
| ------------- | ------------------------------------------------------------ |
| 模型         | I3D                                                           |
| 机器          | 8p Ascend 910                                                |
| 框架版本       | MindSpore 1.5.0                                               |
| 数据集        | HMDB51                                                       |
| 训练参数      | epoch=40(rgb) 60(flow), steps_per_epoch=56, batch_size=8      |
| 优化器        | Adam                                                         |
| 损失函数      | Softmax交叉熵                                                  |
| 速度          | rgb: 2200ms/step ,  flow: 2100ms/step                         |
| 训练总时长     | rgb: 1.30h ,       flow: 2.06h                                |

#### UCF101数据集

| 参数          | Ascend 910                                                   |
| ------------- | ------------------------------------------------------------ |
| 模型         | I3D                                                           |
| 机器           | 8p Ascend 910                                                |
| 框架版本        | MindSpore 1.5.0                                               |
| 数据集         | UCF101                                                        |
| 训练参数       | epoch=40(rgb) 60(flow), steps_per_epoch=148, batch_size=8     |
| 优化器         | Adam                                                          |
| 损失函数       | Softmax交叉熵                                                  |
| 速度          | rgb: 1550ms/step , flow: 600ms/step                           |
| 训练总时长     | rgb: 2.80h ,       flow: 1.44h                               |

# 在云平台openI上进行训练和推理任务

## 训练任务

启动文件为train.py，不同数据集和模式，对应的参数如下:

```text
启动文件 : train.py
AI引擎 : MindSpore-1.5.1-c79-python3.7-euleros2.8-aarch64
规格 : Ascend: 8 * Ascend-910(32GB) | ARM: 192 核 2048GB
hmdb51 rgb  : openI=True, dataset=hmdb51, mode=rgb, num_epochs=40, distributed=True
hmdb51 flow : openI=True, dataset=hmdb51, mode=flow, num_epochs=60, distributed=True
ucf101 rgb  : openI=True, dataset=ucf101, mode=rgb, num_epochs=40, distributed=True
ucf101 flow : openI=True, dataset=ucf101, mode=flow, num_epochs=60, distributed=True
```

## 推理任务

启动文件为eval.py，不同数据集和模式，对应的参数如下(云上仅支持单模式评估):

```text
启动文件 : eval.py
AI引擎 : MindSpore-1.5.1-c79-python3.7-euleros2.8-aarch64
规格 : Ascend: 1 * Ascend-910(32GB) | ARM: 24 核 256GB
hmdb51 rgb  : openI=True, dataset=hmdb51, test_mode=rgb, rgb_path=[要评估的模型的名字，如i3d-40_56.ckpt，不需要上级目录]
hmdb51 flow : openI=True, dataset=hmdb51, test_mode=flow, flow_path=[要评估的模型的名字，如i3d-60_56.ckpt，不需要上级目录]
ucf101 rgb  : openI=True, dataset=ucf101, test_mode=rgb, rgb_path=[要评估的模型的名字，如i3d-40_148.ckpt，不需要上级目录]
ucf101 flow : openI=True, dataset=ucf101, test_mode=flow, flow_path=[要评估的模型的名字，如i3d-60_148.ckpt，不需要上级目录]
```

# 随机情况说明

在```train.py```中和```eval.py```中，我们设置了```mindspore```、```mindspore.dataset```、```random```和```np.random```的随机种子。

# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。

