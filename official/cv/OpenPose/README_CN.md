# 目录

- [目录](#目录)
- [OpenPose描述](#openpose描述)
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
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)

# [OpenPose描述](#目录)

OpenPose网络提出了一种利用部分亲和场（PAF）自下而上的人体姿态估计算法，区别于先检测人再返回关键点和骨架的自上而下的算法。OpenPose的优势在于计算时间不会随着图像中人数的增加而显著增加，但自上而下的算法是基于检测结果，运行时间随人数线性增长。

[论文](https://arxiv.org/abs/1611.08050):  Zhe Cao,Tomas Simon,Shih-En Wei,Yaser Sheikh,"Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields",The IEEE Conference on Computer Vision and Pattern Recongnition(CVPR),2017

# [模型架构](#目录)

首先，通过基线CNN网络，提取输入图像的特征图。本文作者采用了VGG-19网络的前10层。
然后，在多阶段CNN流水线中处理特征图，生成部分置信图和部分亲和场。
最后，生成的置信图和部分亲和场通过贪婪二分匹配算法处理，获得图像中每个人的姿势。

# [数据集](#目录)

准备数据集，包括训练集、验证集和注释。训练集和验证集示例位于"dataset"目录中，可用的数据集包括COCO2014、COCO2017数据集。
在本文提供的训练脚本中，以COCO2017数据集为例，在训练过程中进行数据预处理。如您使用其他格式的数据集，请修改数据集加载和预处理方法。

- 从COCO2017数据官网下载数据，并解压缩。

    ```bash
        wget http://images.cocodataset.org/zips/train2017.zip
        wget http://images.cocodataset.org/zips/val2017.zip
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    ```

- 创建掩码数据集。

    运行python gen_ignore_mask.py。

    ```python
        python gen_ignore_mask.py --train_ann=[TRAIN_ANN] --val_ann=[VAL_ANN] --train_dir=[IMGPATH_TRAIN] --val_dir=[IMGPATH_VAL]
        示例：python gen_ignore_mask.py --train_ann /home/DataSet/coco/annotations/person_keypoints_train2017.json --val_ann /home/DataSet/coco/annotations/person_keypoints_val2017.json --train_dir /home/DataSet/coco/train2017 --val_dir /home/DataSet/coco/val2017
    ```

- 在根目录下生成dataset文件夹，包含以下文件：

   ```python
   ├── dataset
       ├── annotations
           ├─person_keypoints_train2017.json
           └─person_keypoints_val2017.json
       ├─ignore_mask_train
       ├─ignore_mask_val
       ├─train2017
       └─val2017
   ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 下载MindSpore版本的VGG19模型：
    - [vgg19-0-97_5004.ckpt](https://download.mindspore.cn/model_zoo/converted_pretrained/vgg/vgg19-0-97_5004.ckpt)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

  ```python
  # 训练示例
  python train.py --imgpath_train coco/train2017 --jsonpath_train coco/annotations/person_keypoints_train2017.json --maskpath_train coco/ignore_mask_train --vgg_path vgg19-0-97_5004.ckpt > train.log 2>&1 &
  OR
  bash run_standalone_train.sh [IAMGEPATH_TRAIN] [JSONPATH_TRAIN] [MASKPATH_TRAIN] [VGG_PATH] [DEVICE_ID]
  # example: bash run_standalone_train.sh coco/train2017 coco/annotations/person_keypoints_train2017.json coco/ignore_mask_train vgg19-0-97_5004.ckpt 0

  # 运行分布式训练示例
  bash run_distribute_train.sh [RANK_TABLE_FILE] [IMGPATH_TRAIN] [JSONPATH_TRAIN] [MASKPATH_TRAIN] [VGG_PATH]
  # 示例：bash run_distribute_train.sh ~/hccl_8p.json coco/train2017 coco/annotations/person_keypoints_train2017.json coco/ignore_mask_train vgg19-0-97_5004.ckpt

  # 运行评估示例
  python eval.py --model_path ckpt/0-8_663.ckpt --imgpath_val coco/val2017 --ann coco/annotations/person_keypoints_val2017.json > eval.log 2>&1 &
  或
  bash scripts/run_eval_ascend.sh [MODEL_PATH] [IMPATH_VAL] [ANN] [DEVICE_ID]
  # 示例：bash scripts/run_eval_ascend.sh ckpt/0-8_663.ckpt coco/val2017 coco/annotations/person_keypoints_val2017.json 0
  ```

[RANK_table_FILE]为多卡信息配置表在环境中的路径。配置表可以由工具[hccl_tool](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)自动生成。

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```python
├── openpose
    ├── README.md                        // OpenPose描述
    ├── scripts
    │   ├──run_standalone_train.sh       // 用于Ascend上运行分布式训练的shell脚本
    │   ├──run_distribute_train.sh       // 用于Ascend上运行分布式训练（8卡）的shell脚本
    │   ├──run_eval_ascend.sh            // 用于Ascend上运行评估的shell脚本
    ├── src
    │   ├── model_utils
    │       ├── config.py                           # 参数配置
    │       ├── moxing_adapter.py                   # ModelArts设备配置
    │       └── device_adapter.py                   # 设备配置
    │       └── local_adapter.py                    # 本地设备配置
    │   ├──openposenet.py                // OpenPose架构
    │   ├──loss.py                       // 损失函数
    │   ├──dataset.py                    // 数据预处理
    │   ├──utils.py                      // 实用工具
    │   ├──gen_ignore_mask.py            // 生成掩码数据脚本
    ├── export.py                        // 模型转换脚本
    ├── train.py                         // 训练脚本
    ├── eval.py                          // 评估脚本
    ├── mindspore_hub_config.py          // Hub配置文件
    ├── default_config.yaml              // 默认配置文件
```

## [脚本参数](#目录)

训练和评估的参数都可以在default_config.yaml中设置。

- config for openpose

  ```default_config.yaml
  'imgpath_train': 'path to dataset'               # 训练和评估数据集的绝对全路径
  'vgg_path': 'path to vgg model'                  # vgg19模型的绝对全路径
  'save_model_path': 'path of saving models'       # 输出模型的绝对全路径
  'load_pretrain': 'False'                         # 是否基于预训练模型进行训练
  'pretrained_model_path':''                       # 加载预训练模型路径
  'lr': 1e-4                                       # 初始学习率
  'batch_size': 10                                 # 训练batch size
  'lr_gamma': 0.1                                  # 达到lr_steps的学习率比例
  'lr_steps': '100000,200000,250000'               # lr*lr_gamma时的步骤
  'loss scale': 16384                              # 混合精度损失缩放
  'max_epoch_train': 60                            # 总训练epochs数
  'insize': 368                                    # 用作模型输入的图像大小
  'keep_checkpoint_max': 1                         # 仅保留最后一个keep_checkpoint_max检查点
  'log_interval': 100                              # 打印日志的时间间隔
  'ckpt_interval': 5000                            # 输出模型保存的时间间隔
  ```

有关更多配置详细信息，请参见脚本`default_config.yaml`。

## [训练过程](#目录)

### 训练

- 在Ascend上运行

  ```python
  python train.py --imgpath_train coco/train2017 --jsonpath_train coco/annotations/person_keypoints_train2017.json --maskpath_train coco/ignore_mask_train --vgg_path vgg19-0-97_5004.ckpt > train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以在`train.log`文件中查看结果。

  训练后，默认在脚本文件夹下生成检查点文件。得到如下损失值：

  ```log
  # grep "epoch " train.log
  epoch[0], iter[23], mean loss is 0.292112287
  epoch[0], iter[123], mean loss is 0.060355084
  epoch[0], iter[223], mean loss is 0.026628130
  ...
  ```

  模型检查点文件默认保存在default_config.yaml目录中：'save_model_path'。

- 在ModelArts上运行
- 如果您想在ModelArts上训练模型，可参考官方指导文档[ModelArts](https://support.huaweicloud.com/modelarts/).

```ModelArts
#  在ModelArts上使用分布式训练DPN的示例：
#  数据集目录结构
#   ├── openpose_dataset
#       ├── annotations
#           ├─person_keypoints_train2017.json
#           └─person_keypoints_val2017.json
#       ├─ignore_mask_train
#       ├─ignore_mask_val
#       ├─train2017
#       └─val2017
#       └─checkpoint
#       └─pre_trained
#
# （1）执行a（修改yaml文件参数）或b（ModelArts上创建训练作业并修改参数）。
#       a. 设置"enable_modelarts=True"。
#          设置"vgg_path=/cache/data/pre_trained/vgg19-0-97_5004.ckpt"。
#          设置"maskpath_train=/cache/data/ignore_mask_train2017"。
#          设置"jsonpath_train=/cache/data/annotations/person_keypoints_train2017"。
#          设置"save_model_path=/cache/train/checkpoint"。
#          设置"imgpath_train=/cache/data/train2017"。
#
#       b. 在ModelArts界面上添加"enable_modelarts=True"。
#          在ModelArts界面上设置方法a所需的参数。
#          注：path参数不需要用引号括起来。

# （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"。
# (3)在ModelArts界面上设置代码路径"/path/openpose"。
# (4)在ModelArts界面上设置模型的启动文件"train.py"。
# (5)在ModelArts界面上设置模型的数据路径".../openpose_dataset"。
# 模型的输出路径“输出文件路径”和“作业日志路径”。
# （6）开始训练模型。

# 在ModelArts上运行模型推理的示例
# （1）将训练好的模型放到对应的桶位置。
# （2）执行a或b。
#        a. 设置"enable_modelarts=True"。
#          设置"ann=/cache/data/annotations/person_keypoints_val2017"。
#          设置"output_img_path=/cache/data/output_imgs/"。
#          设置"imgpath_val=/cache/data/val2017"。
#          设置"model_path=/cache/data/checkpoint/0-80_663.ckpt"。

#       b. 在ModelArts界面上添加"enable_modelarts=True"。
#          在ModelArts界面上设置方法a所需的参数。
#          注：path参数不需要用引号括起来。

# （3）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"。
# (4)在ModelArts界面上设置代码路径"/path/openpose"。
# (5)在ModelArts界面上设置模型的启动文件"eval.py"。
# (6)在ModelArts界面上设置模型的数据路径".../openpose_dataset"。
# 模型的输出文件路径和作业日志路径。
# （7）启动模型推理。
```

## [评估过程](#目录)

### 评估

- 在Ascend上运行

  运行以下命令前，请检查用于评估的检查点路径。检查点路径需要设置为绝对全路径，例如，"username/openpose/outputs/\*time*\/0-6_30000.ckpt"。

  ```python
  # 运行评估示例
  python eval.py --model_path ckpt/0-8_663.ckpt --imgpath_val coco/val2017 --ann coco/annotations/person_keypoints_val2017.json > eval.log 2>&1 &
  或
  bash scripts/run_eval_ascend.sh [MODEL_PATH] [IMPATH_VAL] [ANN] [DEVICE_ID]
  # bash scripts/run_eval_ascend.sh ckpt/0-8_663.ckpt coco/val2017 coco/annotations/person_keypoints_val2017.json 0
  ```

  上述Python命令将在后台运行，您可以通过文件"eval.log"查看结果。测试数据集的准确率如下：

  ```log
  # grep "AP" eval.log

  {'AP': 0.40250956300341397, 'Ap .5': 0.6658941566481336, 'AP .75': 0.396047897339743, 'AP (M)': 0.3075356543635785, 'AP (L)': 0.533772768618845, 'AR': 0.4519836272040302, 'AR .5': 0.693639798488665, 'AR .75': 0.4570214105793451, 'AR (M)': 0.32155148866429945, 'AR (L)': 0.6330360460795242}

  ```

- 在Modelart上导出MindIR

  ```ModelArts
  在ModelArts上导出MindIR的示例
  数据目录结构同训练
  # （1）执行a（修改yaml文件参数）或b（ModelArts上创建训练作业并修改参数）。
  #       a. 设置"enable_modelarts=True"。
  #          设置"file_name=openpose"。
  #          设置"file_format=MINDIR"。
  #          设置"ckpt_file=/cache/data/checkpoint file name"。

  #       b. 在ModelArts界面上添加"enable_modelarts=True"。
  #          在ModelArts界面上设置方法a所需的参数。
  #          注：path参数不需要用引号括起来。
  # （2）设置网络配置文件路径"_config_path=/The path of config in default_config.yaml/"。
  # (3)在ModelArts界面上设置代码路径"/path/openpose"。
  # (4)在ModelArts界面上设置模型的启动文件"export.py"。
  # (5)在ModelArts界面上设置模型的数据路径".../openpose_dataset/checkpoint"。
  # 模型的输出文件路径和作业日志路径。
  ```

# [模型说明](#目录)

## [性能](#目录)

### 评估性能

| 参数                | Ascend
| -------------------------- | -----------------------------------------------------------
| 模型版本             | OpenPose
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；操作系统EulerOS 2.8
| 上传日期             | 12/14/2020
| MindSpore版本         | 1.0.1
| 训练参数       | epoch=60(1pcs)/80(8pcs), steps=30k(1pcs)/5k(8pcs), batch_size=10, init_lr=0.0001
| 优化器                 | Adam（单卡）/动量（8卡）
| 损失函数             | MSE
| 输出                   | 人体姿势
| 速度                     | 单卡：35fps；8卡：230fps
| 总时长                | 单卡：22.5h；8卡：5.1h
| 微调检查点| 602.33M（.ckpt文件）
