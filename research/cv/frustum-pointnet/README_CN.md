# 目录

<!-- TOC -->

- [目录](#目录)
- [Frustum Pointnets描述](#frustum-pointnets描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [数据集处理](#数据集处理)
    - [训练模型](#训练模型)
    - [评估模型](#评估模型)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单卡训练](#单卡训练)
        - [多卡训练](#多卡训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Frustum Pointnets描述

结合成熟的2d物体检测技术和先进的3d深度学习技术。在其工作流水线上，首先使用RGB图像，构建region proposals，生成的2d bounding box定义了3d frustum region。然后基于该furstum regions的3d点云，使用PointNet / PointNet ++网络实现3D实例分割和amodal 3D边界框估计。

[论文](https://arxiv.org/abs/1711.08488v1)：Qi, Charles R., et al. "Frustum pointnets for 3d object detection from rgb-d data." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

# 模型架构

Fustum PointNets结构，利用二维图片和点云数据进行三维检测，先在二维图片生成检测框，再利用截锥反投影到点云上。得到点云上的视锥体，接着用 PointNet 变形进行语义分割，排除掉一些不用的点，解决阻塞和扰乱等问题，接着将得到的点使用 PointNet 的另一变形进行回归，生成边界框。

# 数据集

使用的数据集：[KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark)

- 数据集大小：数据集可以分为Road、City、Residential、Campus、Person几类，数据集总大小约为180G。
    - 训练集
        - [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
        - [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
        - [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
        - [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
    - 测试集
        - [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
        - [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
        - [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)

下载数据集于主目录`dataset`文件夹中，数据集目录结构如下：

```bash
.
├── dataset
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
│
```

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

## 数据集处理

- 生成训练数据集：

```bash
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --car_only
```

- 生成评估数据集：

从`https://github.com/prclibo/kitti_eval`下载`evaluate_object_3d_offline.cpp`,`compile.sh`,`mail.h`到`./kitti/kitti_eval`文件夹，通过Boost和Linux dirent.h依赖编译`evaluate_object_3d_offline.cpp`，确保`./kitti/kitti_eval`文件夹中生成`evaluate_object_3d_offline`文件。

```bash

sudo apt-get install libboost-all-dev
g++ -o2 evaluate_object_3d_offline.cpp -o evaluate_object_3d_offline

```

- 预处理数据集于主目录`kitti`文件夹中，数据集目录结构如下：

```bash
.
├── image_sets
    ├── test.txt
    ├── train.txt
    ├── trainval.txt
    ├── val.txt
├── kitti_val
├── rgb_detections
    ├── rgb_detection_train.txt
    ├── rgb_detection_val.txt
├── kitti_object.py
├── kitti_util.py
├── prepare_data.py
├── frustum_caronly_val.pickle
├── frustum_caronly_train.pickle
├── frustum_caronly_val_rgb_detection.pickle

```

image_sets & rgb_detections txt file can be download here: https://github.com/charlesq34/frustum-pointnets/tree/master/kitti

## 训练模型

- GPU单卡训练

```bash
bash scripts/run_standalone_train_gpu.sh [LOG_DIR] [DEVICE_TARGET] [DEVICE_ID]
# example bash scripts/run_standalone_train_gpu.sh log GPU 0
```

训练日志保存在`$LOG_DIR/train.log`中，查看日志信息可以通过如下命令：

```bash
tail -f $LOG_DIR/train.log
```

- GPU多卡训练

```bash
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM]
# example: bash scripts/run_distributed_train_gpu.sh 8
```

训练日志保存在`$LOG_DIR/train.log`中，查看日志信息可以通过如下命令：

```bash
tail -f $LOG_DIR/train.log
```

- Ascend单卡训练

```bash
bash scripts/run_standalone_train.sh [LOG_DIR] [DEVICE_TARGET] [DEVICE_ID]
# example bash scripts/run_standalone_train.sh log Ascend 0
```

训练日志保存在`$LOG_DIR/train.log`中，查看日志信息可以通过如下命令：

```bash
tail -f $LOG_DIR/train.log
```

- Ascend多卡训练

```bash
bash scripts/run_distributed_train_ascend.sh [RANK_TABLE_FILE]
# example: bash scripts/run_distributed_train_ascend.sh ./hccl_8p.json
```

训练日志保存在`train{device_id}.log`中，查看日志信息可以通过如下命令：

```bash
tail -f train0.log
```

## 评估模型

```bash
bash [OUTPUT_PATH] [PRETRAINDE_CKPT] [DEVICE_TARGET] [DEVICE_ID]
# example: bash scripts/run_eval_gpu.sh eval_result net.ckpt Ascend 0
```

评估日志为`the Fpointnet_eval.log`。

# 脚本说明

## 脚本及样例代码

```bash
.
├── README.md                                       // Frustum Pointnets中文描述文档
├── dataset                                         // 数据集目录
│   └── README.md                                   // 数据集说明文档
├── kitti
│   ├── kitti_eval                                  // kitti官方评估工具(https://github.com/prclibo/kitti_eval)
│   │   ├── README.md
│   │   ├── compile.sh
│   │   ├── evaluate_object_3d_offline
│   │   ├── evaluate_object_3d_offline.cpp
│   │   └── mail.h
│   ├── image_sets                                  // 数据集标注文件
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── trainval.txt
│   │   └── val.txt
│   ├── kitti_object.py                             // 数据集对象脚本
│   ├── kitti_util.py                               // 数据集工具脚本
│   ├── prepare_data.py                             // 数据预处理脚本
│   └── rgb_detections                              // RGB标注文件
│       ├── rgb_detection_train.txt
│       └── rgb_detection_val.txt
├── scripts
│   ├── command_prep_data.sh                        // 数据预处理shell脚本
│   ├── run_distributed_train_gpu.sh                // 多卡训练shell脚本-GPU
    ├── run_distributed_train_ascend.sh             // 多卡训练shell脚本-Ascend
│   ├── run_eval.sh                                 // 模型评估shell脚本
│   └── run_standalone_train.sh                     // 单卡训练shell脚本
├── src
│   ├── datautil.py                                 // 数据集处理工具脚本
│   ├── frustum_pointnets_v1.py                     // Frustum Pointnets脚本
│   ├── model_util.py                               // 模型相关脚本
├── train
│   ├── __init__.py
│   ├── box_util.py                                 // 框处理脚本
│   ├── datautil.py                                 // 数据处理脚本
│   ├── train_util.py                               // 训练脚本
│   └── provider.py
├── train_net.py                                    // 模型训练脚本
├── eval.py                                         // 模型评估脚本
```

## 脚本参数

在train_net.py中可以配置训练参数。

```bash
usage: train_net.py [-h] [--name NAME] [--model MODEL] [--log_dir LOG_DIR] [--num_point NUM_POINT] [--max_epoch MAX_EPOCH] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--momentum MOMENTUM] [--optimizer OPTIMIZER] [--decay_step DECAY_STEP] [--loss_per_epoch LOSS_PER_EPOCH]
               [--decay_rate DECAY_RATE] [--objtype OBJTYPE] [--weight_decay WEIGHT_DECAY] [--no_intensity] [--train_sets TRAIN_SETS] [--val_sets VAL_SETS]
               [--restore_model_path RESTORE_MODEL_PATH] [--keep_checkpoint_max KEEP_CHECKPOINT_MAX] [--disable_datasink_mode]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           tensorboard writer name
  --model MODEL         Model name [default: frustum_pointnets_v1_ms]
  --log_dir LOG_DIR     Log dir [default: log]
  --num_point NUM_POINT
                        Point Number [default: 2048]
  --max_epoch MAX_EPOCH
                        Epoch to run [default: 200]
  --batch_size BATCH_SIZE
                        Batch Size during training [default: 32]
  --learning_rate LEARNING_RATE
                        Initial learning rate [default: 0.001]
  --momentum MOMENTUM   Initial learning rate [default: 0.9]
  --optimizer OPTIMIZER
                        adam or momentum [default: adam]
  --decay_step DECAY_STEP
                        Decay step for lr decay [default: 200000]
  --loss_per_epoch LOSS_PER_EPOCH
                        times to print loss value per epoch
  --decay_rate DECAY_RATE
                        Decay rate for lr decay [default: 0.7]
  --objtype OBJTYPE     caronly or carpedcyc
  --weight_decay WEIGHT_DECAY
                        Weight Decay of Adam [default: 1e-4]
  --no_intensity        Only use XYZ for training
  --train_sets TRAIN_SETS
  --val_sets VAL_SETS
  --restore_model_path RESTORE_MODEL_PATH
                        Restore model path e.g. log/model.ckpt [default: None]
  --keep_checkpoint_max KEEP_CHECKPOINT_MAX
                        max checkpoints to save [default: 5]
  --disable_datasink_mode
                        disable datasink mode [default: False]

```

在eval.py中可以配置评估参数。

```bash
usage: eval.py [-h] [--gpu GPU] [--num_point NUM_POINT] [--model MODEL] [--model_path MODEL_PATH] [--batch_size BATCH_SIZE] [--output OUTPUT]
               [--data_path DATA_PATH] [--from_rgb_detection] [--idx_path IDX_PATH] [--dump_result] [--return_all_loss] [--objtype OBJTYPE] [--sensor SENSOR]
               [--dataset DATASET] [--split SPLIT] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU to use [default: GPU 0]
  --num_point NUM_POINT
                        Point Number [default: 1024]
  --model MODEL         Model name [default: frustum_pointnets_v1]
  --model_path MODEL_PATH
                        model checkpoint file path [default: log/model.ckpt]
  --batch_size BATCH_SIZE
                        batch size for inference [default: 32]
  --output OUTPUT       output file/folder name [default: test_results]
  --data_path DATA_PATH
                        frustum dataset pickle filepath [default: None]
  --from_rgb_detection  test from dataset files from rgb detection.
  --idx_path IDX_PATH   filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]
  --dump_result         If true, also dump results to .pickle file
  --return_all_loss     only return total loss default
  --objtype OBJTYPE     caronly or carpedcyc
  --sensor SENSOR       only consider CAM_FRONT
  --dataset DATASET     kitti or nuscenes or nuscenes2kitti
  --split SPLIT         v1.0-mini or val
  --debug               debug mode

```

更多配置细节请参考脚本`train.py`和`test.py`。

## 训练过程

### 单卡训练

- GPU处理器环境运行

  ```python
  bash scripts/run_standalone_train.sh [LOG_DIR] [DEVICE_TARGET] [DEVICE_ID]
  # example bash scripts/run_standalone_train.sh log GPU 0
  ```

  上述python命令将在后台运行，您可以通过`$LOG_DIR/train.log`文件查看结果。

  训练过程的日志如下所示：

  ```log
  epoch: 1 step: 100, loss is 165.117431640625
  epoch: 1 step: 200, loss is 40.0536003112793
  epoch: 1 step: 300, loss is 64.46954345703125
  epoch: 1 step: 400, loss is 43.03189468383789
  ```

- Ascend处理器环境运行

  ```bash
  bash scripts/run_standalone_train.sh [LOG_DIR] [DEVICE_TARGET] [DEVICE_ID]
  # example bash scripts/run_standalone_train.sh log Ascend 0
  ```

  训练日志保存在`$LOG_DIR/train.log`中，查看日志信息可以通过如下命令：

  ```bash
  tail -f $LOG_DIR/train.log
  ```

  训练过程的日志如下所示：

  ```log
  epoch: 1 step: 100, loss is 86.37467193603516
  epoch: 1 step: 200, loss is 41.17808532714844
  epoch: 1 step: 300, loss is 103.21517944335938
  epoch: 1 step: 400, loss is 21.88726806640625
  epoch: 1 step: 500, loss is 33.24089050292969
  epoch: 1 step: 600, loss is 103.6310806274414
  ```

### 多卡训练

- GPU处理器环境运行

  ```bash
  bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM]
  # example: bash scripts/run_distributed_train_gpu.sh 8
  ```

  上述shell脚本将在后台运行分布训练。您可以在`$LOG_DIR/train.log`文件下查看结果。

  ```log
  epoch: 1 step: 100, loss is 165.117431640625
  epoch: 1 step: 200, loss is 40.0536003112793
  epoch: 1 step: 300, loss is 64.46954345703125
  epoch: 1 step: 400, loss is 43.03189468383789
  ```

- Ascned处理器环境运行

  ```bash
  bash scripts/run_distributed_train_ascend.sh [RANK_TABLE_FILE]
  # example: bash scripts/run_distributed_train_ascend.sh ./hccl_8p.json
  ```

  上述shell脚本将在后台运行分布训练。您可以在`train{device_id}.log`文件下查看结果。

  ```log
  epoch: 1 step: 100, loss is 37.824676513671875
  epoch: 1 step: 200, loss is 14.480133056640625
  Train epoch time: 134025.120 ms, per step time: 580.195 ms
  save checkpoint acc 0.13 > best.ckpt
  {"eval_accuracy": 0.7726795430086097, "eval_box_IoU_(ground/3D)": [0.5129166744193252, 0.44038378219215235], "eval_box_estimation_accuracy_(IoU=0.7)": 0.1256377551020408, "Best Test acc: %f(Epoch %d)": [0.1256377551020408, 1]}
  ```

## 评估过程

### 评估

- 评估kitti数据集

  ```bash
  bash scripts/run_eval.sh [OUTPUT_PATH] [PRETRAINDE_CKPT] [DEVICE_TARGET] [DEVICE_ID]
  # example: bash scripts/run_eval_gpu.sh results net.ckpt Ascend 0
  ```

  上述python命令将在后台运行，您可以通过`theFpointnet_eval.log`文件查看结果。测试数据集的日志文件如下：

  ```bash
  2022-09-22T15:11:35.125338:loading test dataset ...
  Number of point clouds: 12538
  segmentation accuracy 0.9020370261903813
  box IoU(ground) 0.7931940339174086
  box IoU(3D) 0.740433778730871
  box estimation accuracy (IoU=0.7) 0.7633593874621152
  Average pos ratio: 0.436544
  Average pos prediction ratio: 0.472163
  Average npoints: 1024.000000
  Mean points: x0.026006 y0.987028 z24.961711
  Max points: x16.745872 y10.476355 z79.747406
  Min points: x-20.209103 y-3.865781 z0.000000
  {'Average pos ratio:': 0.4365437663253709, 'Average pos prediction ratio:': 0.4721632954568113, 'Average npoints:': 1024.0, 'Mean points:': array([ 0.02600595,  0.98702844, 24.96171088]), 'Max points:': array([16.7458725 , 10.4763546 , 79.74740601]), 'Min points:': array([-20.20910263,  -3.86578107,   0.        ])}
  mkdir: cannot create directory ‘msp_output_764/plot’: File exists
  PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
  ==> 1 page written on 'car_detection.pdf'.
  PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
  ==> 1 page written on 'car_detection_ground.pdf'.
  Thank you for participating in our evaluation!
  Loading detections...
  number of files for evaluation: 3769
  done.
  save msp_output_764/plot/car_detection.txt
  car_detection AP: 100.000000 100.000000 100.000000
  Finished 2D bounding box eval.
  Going to eval ground for class: car
  save msp_output_764/plot/car_detection_ground.txt
  car_detection_ground AP: 87.600319 85.757347 77.970871
  Finished Birdeye eval.
  Going to eval 3D box for class: car
  save msp_output_764/plot/car_detection_3d.txt
  car_detection_3d AP: 84.155861 73.772659 66.403107
  PDFCROP 1.38, 2012/11/02 - Copyright (c) 2002-2012 by Heiko Oberdiek.
  ==> 1 page written on 'car_detection_3d.pdf'.
  Finished 3D bounding box eval.
  Your evaluation results are available at:
  msp_output_764
  ```

# 模型描述

## 性能

### 训练性能

| 参数          | GPU       | Ascend |
| ------------- | --------- | ----- |
| MindSpore版本 | 1.8.0                                                 | 1.8.1 |
| ckpt模型      | 25.2MB (.ckpt文件)                                    | 25.2MB (.ckpt文件) |
| 上传日期      | 2022-09-23                                            | 2022-11-03 |
| 优化器        | Adam                                                  | Adam |
| 总时长        | 单卡：12小时19分钟；八卡：3小时                       |单卡：20小时；八卡：5小时
| 损失          | 1.02                                                  | 1.03 |
| 损失函数      | NLLLoss, huber_loss                                   | NLLLoss, huber_loss |
| 数据集        | KITTI                                                 | KITTI |
| 模型版本      | frustum pointnets                                     | frustum pointnets |
| 训练参数      | epoch=200, steps per epoch=1849, batch_size = 32      | epoch=200, steps per epoch=1849, batch_size = 32 |
| 资源          | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G | Ascend 910 |
| 输出          | 坐标，特征向量，尺寸                                  |  坐标，特征向量，尺寸 |
| 速度          | 单卡：118毫秒/步；八卡：230毫秒/步                    | 单卡：267毫秒/步；八卡：350毫秒/步 |

### 推理性能

| 参数          | GPU                    | Ascend |
| ------------- | ------------------------- | ----- |
| 模型版本      | frustum pointnets              | frustum pointnets |
| 资源          | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G | Ascend 910 |
| 上传日期      | 2022-09-23                | 2022-11-03 |
| MindSpore版本 | 1.8.0                     | 1.8.1 |
| 数据集        | KITTI              | KITTI |
| batch_size    | 32                       | 32 |
| 输出          | 概率                      | 概率 |
| 准确性        | 76.96%               | 78.57% |

# 随机情况说明

在train_net.py中，我们设置了“set_seed”的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。