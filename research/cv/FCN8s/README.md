# 目录

- [目录](#目录)
- [FCN8s描述](#FCN8s描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [启动](#启动)
        - [结果](#结果)
    - [推理过程](#推理过程)
        - [导出ONNX](#导出ONNX)
        - [在GPU执行ONNX推理](#在GPU执行ONNX推理)
        - [结果](#结果)
- [模型说明](#模型说明)
    - [训练性能](#训练性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo 主页](#modelzoo-主页)

# FCN8s描述

FCN主要用用于图像分割领域，是一种端到端的分割方法。FCN丢弃了全连接层，使得其能够处理任意大小的图像，且减少了模型的参数量，提高了模型的分割速度。FCN在编码部分使用了VGG的结构，在解码部分中使用反卷积/上采样操作恢复图像的分辨率。FCN-8s最后使用8倍的反卷积/上采样操作将输出分割图恢复到与输入图像相同大小。

[Paper]: Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

# 模型架构

FCN-8s使用丢弃全连接操作的VGG16作为编码部分，并分别融合VGG16中第3,4,5个池化层特征，最后使用stride=8的反卷积获得分割图像。

# 数据集

- Dataset used:

    [PASCAL VOC 2012](<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>)

# 环境要求

- 硬件（Ascend/GPU）
    - 需要准备具有Ascend或GPU处理能力的硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需获取更多信息，请查看如下链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本和样例代码

```text
├── model_zoo
    ├── README.md                     // descriptions about all the models
    ├── FCN8s
        ├── README.md                 // descriptions about FCN
        ├── ascend310_infer           // 实现310推理源代码
        ├── scripts
            ├── run_train.sh
            ├── run_standalone_train.sh
            ├── run_standalone_train_gpu.sh             // train in gpu with single device
            ├── run_distribute_train_gpu.sh             // train in gpu with multi device
            ├── run_eval.sh
            ├── run_eval_onnx.sh         //用于ONNX推理的shell脚本
            ├── run_infer_310.sh         // Ascend推理shell脚本
            ├── build_data.sh
        ├── src
        │   ├──data
        │       ├──build_seg_data.py       // creating dataset
        │       ├──dataset.py          // loading dataset
        │   ├──nets
        │       ├──FCN8s.py            // FCN-8s architecture
        │   ├──loss
        │       ├──loss.py            // loss function
        │   ├──utils
        │       ├──lr_scheduler.py            // getting learning_rateFCN-8s
        │   ├──model_utils
        │       ├──config.py                     // getting config parameters
        │       ├──device_adapter.py            // getting device info
        │       ├──local_adapter.py            // getting device info
        │       ├──moxing_adapter.py          // Decorator
        ├── default_config.yaml               // Ascend parameters config
        ├── gpu_default_config.yaml           // GPU parameters config
        ├── train.py                 // training script
        ├── postprogress.py          // 310推理后处理脚本
        ├── export.py                // 将checkpoint文件导出到air/mindir
        ├── eval.py                  //  evaluation script
        ├── eval_onnx.py             //  onnx评估
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```text
  # dataset
  'data_file': '/data/workspace/mindspore_dataset/FCN/FCN/dataset/MINDRECORED_NAME.mindrecord', # path and name of one mindrecord file
  'train_batch_size': 32,
  'crop_size': 512,
  'image_mean': [103.53, 116.28, 123.675],
  'image_std': [57.375, 57.120, 58.395],
  'min_scale': 0.5,
  'max_scale': 2.0,
  'ignore_label': 255,
  'num_classes': 21,

  # optimizer
  'train_epochs': 500,
  'base_lr': 0.015,
  'loss_scale': 1024.0,

  # model
  'model': 'FCN8s',
  'ckpt_vgg16': '',
  'ckpt_pre_trained': '',

  # train
  'save_steps': 330,
  'keep_checkpoint_max': 5,
  'ckpt_dir': './ckpt',
```

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```bash
# Ascend单卡训练示例
python train.py --device_id device_id
or
bash scripts/run_standalone_train.sh [DEVICE_ID]
# example: bash scripts/run_standalone_train.sh 0

#Ascend八卡并行训练
bash scripts/run_train.sh [DEVICE_NUM] rank_table.json
# example: bash scripts/run_train.sh 8 ~/hccl_8p.json

# GPU单卡训练示例
python train.py  \
--config_path=gpu_default_config.yaml  \
--device_target=GPU
or
bash scripts/run_standalone_train_gpu.sh DEVICE_ID

# GPU八卡训练示例
export RANK_SIZE=8
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout  \
python train.py  \
--config_path=gpu_default_config.yaml \
--device_target=GPU
or
bash run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]

# GPU评估示例
python eval.py  \
--config_path=gpu_default_config.yaml \
--device_target=GPU
```

### 结果

训练时，训练过程中的epch和step以及此时的loss和精确度会呈现log.txt中:

```text
epoch: * step: **, loss is ****
...
```

此模型的checkpoint会在默认路径下存储

如果要在modelarts上进行模型的训练，可以参考modelarts的[官方指导文档](https://support.huaweicloud.com/modelarts/) 开始进行模型的训练和推理，具体操作如下：

```ModelArts
#  在ModelArts上使用分布式训练示例:
#  数据集存放方式

#  ├── VOC2012                                                     # dir
#    ├── VOCdevkit                                                 # VOCdevkit dir
#      ├── Please refer to VOCdevkit structure  
#    ├── benchmark_RELEASE                                         # benchmark_RELEASE dir
#      ├── Please refer to benchmark_RELEASE structure
#    ├── backbone                                                  # backbone dir
#      ├── vgg_predtrained.ckpt
#    ├── predtrained                                               # predtrained dir
#      ├── FCN8s_1-133_300.ckpt
#    ├── checkpoint                                                # checkpoint dir
#      ├── FCN8s_1-133_300.ckpt
#    ├── vocaug_mindrecords                                        # train dataset dir
#      ├── voctrain.mindrecords0
#      ├── voctrain.mindrecords0.db
#      ├── voctrain.mindrecords1
#      ├── voctrain.mindrecords1.db
#      ├── voctrain.mindrecords2
#      ├── voctrain.mindrecords2.db
#      ├── voctrain.mindrecords3
#      ├── voctrain.mindrecords3.db
#      ├── voctrain.mindrecords4
#      ├── voctrain.mindrecords4.db
#      ├── voctrain.mindrecords5
#      ├── voctrain.mindrecords5.db
#      ├── voctrain.mindrecords6
#      ├── voctrain.mindrecords6.db
#      ├── voctrain.mindrecords7
#      ├── voctrain.mindrecords7.db

# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式
#       a. 设置 "enable_modelarts=True"
#          设置 "ckpt_dir=/cache/train/outputs_FCN8s/"
#          设置 "ckpt_vgg16=/cache/data/backbone/vgg_predtrain file"  如果没有预训练 ckpt_vgg16=""
#          设置 "ckpt_pre_trained=/cache/data/predtrained/pred file" 如果无需继续训练 ckpt_pre_trained=""
#          设置 "data_file=/cache/data/vocaug_mindrecords/voctrain.mindrecords0"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/FCN8s"
# (4) 在modelarts的界面上设置模型的启动文件 "train.py"
# (5) 在modelarts的界面上设置模型的数据路径 ".../VOC2012"(选择VOC2012文件夹路径)
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path"
# (6) 开始模型的训练

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置
# (2) 选择a或者b其中一种方式
#       a. 设置 "enable_modelarts=True"
#          设置 "data_root=/cache/data/VOCdevkit/VOC2012/"
#          设置 "data_lst=./ImageSets/Segmentation/val.txt"
#          设置 "ckpt_file=/cache/data/checkpoint/ckpt file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号

# (3) 设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (4) 在modelarts的界面上设置代码的路径 "/path/FCN8s"
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py"
# (6) 在modelarts的界面上设置模型的数据路径 ".../VOC2012"(选择VOC2012文件夹路径) ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path"
# (7) 开始模型的推理
```

## 评估过程

### 启动

在Ascend或GPU上使用PASCAL VOC 2012 验证集进行评估

在使用命令运行前，请检查用于评估的checkpoint的路径。请设置路径为到checkpoint的绝对路径，如 "/data/workspace/mindspore_dataset/FCN/FCN/model_new/FCN8s-500_82.ckpt"。

```python
python eval.py
```

```bash
bash scripts/run_eval.sh DATA_ROOT DATA_LST CKPT_PATH
# example: bash scripts/run_eval.sh /home/DataSet/voc2012/VOCdevkit/VOC2012 \
# /home/DataSet/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt /home/FCN8s/ckpt/fcn8s_ascend_v180_voc2012_official_cv_meanIoU62.7.ckpt
```

### 结果

以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以类似如下方式呈现：

```text
mean IoU 0.638887018016709
```

# 推理过程

## 导出ONNX

```bash
python export.py --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT] --config_path [CONFIG_PATH]
```

例如：python expor
--ckpt_file /root/zj/models/research/cv/FCN8s/checkpoint/fcn8s_ascend_v180_voc2012_official_cv_meanIoU62.7.ckpt  --file_format ONNX  --config_path /root/zj/models/research/cv/FCN8s/default_config.yaml

参数ckpt_file为必填项， `EXPORT_FORMAT` 可选 ["AIR", "MINDIR", "ONNX"]. config_path 为相关配置文件.

在modelarts上导出ONNX

```Modelarts
在ModelArts上导出ONNX示例
数据集存放方式同Modelart训练
# (1) 选择a(修改yaml文件参数)或者b(ModelArts创建训练作业修改参数)其中一种方式。
#       a. 设置 "enable_modelarts=True"
#          设置 "file_name=fcn8s"
#          设置 "file_format=ONNX"
#          设置 "ckpt_file=/cache/data/checkpoint file name"

#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置方法a所需要的参数
#          注意：路径参数不需要加引号
# (2)设置网络配置文件的路径 "_config_path=/The path of config in default_config.yaml/"
# (3) 在modelarts的界面上设置代码的路径 "/path/fcn8s"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 ".../VOC2012/checkpoint"(选择VOC2012/checkpoint文件夹路径) ,
# MindIR的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
```

## 在GPU执行ONNX推理

在执行推理前，ONNX文件必须通过 `export.py` 脚本导出。以下展示了使用ONNX模型执行推理的示例。

```bash
# ONNX inference
bash scripts/run_eval_onnx.sh [ONNX_PATH][DATA_ROOT] [DATA_LST]
```

例如：bash scripts/run_eval_onnx.sh /root/zj/models/research/cv/FCN8s/fcn8s.onnx /root/zj/models/research/cv/FCN8s/dataset/VOC2012 /root/zj/models/research/cv/FCN8s/dataset/VOC2012/ImageSets/Segmentation/val.txt

## 结果

- eval on GPU

以上的python命令会在终端上运行，你可以在终端上查看此次评估的结果。测试集的精确度会以类似如下方式呈现:

```text
mean IoU 0.6388868594659682
```

## 310推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

# 模型说明

## 训练性能

| 参数     | Ascend                                                       |
| -------- | ------------------------------------------------------------ |
| 模型名称 | FCN8s                                                        |
| 运行环境 | TITAN Xp 12G                                                 |
| 上传时间 | 2022-09-22                                                   |
| 数据集   | PASCAL VOC 2012                                              |
| 训练参数 | default_config.yaml                                          |
| 优化器   | Momentum                                                     |
| 损失函数 | Softmax Cross Entropy                                        |
| 最终损失 | 0.036                                                        |
| 速度     | 1pc: 455.460 ms/step;                                        |
| mean IoU | 0.6388868594659682                                           |
| 脚本     | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/FCN8s) |

# 随机情况的描述

我们在 `train.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models) 。
