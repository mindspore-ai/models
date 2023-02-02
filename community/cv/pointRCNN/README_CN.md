# 目录

<!-- TOC -->

- [目录](#目录)
- [PointRCNN描述](#pointrcnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [工具安装](#工具安装)
    - [模型转化](#模型转化)
    - [训练模型](#训练模型)
    - [评估模型](#评估模型)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PointRCNN描述

PointRCNN 3D对象检测器以自下而上的方式从原始点云直接生成精确的3D箱体提案，然后通过所提出的基于箱的3D箱回归损失在规范坐标中对其进行细化。PointRCNN是第一个仅使用原始点云作为输入进行3D对象检测的两级3D对象检测器。

[论文](https://arxiv.org/abs/1812.04244)：Shi, Shaoshuai, Xiaogang Wang, and Hongsheng Li. "Pointrcnn: 3d object proposal generation and detection from point cloud." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

# 模型架构

第一阶段，通过PointNet++ 进行特征的提取，基于提取到的特征可以进行前景和背景的分割， 在每个前景点上进行3D框的预测。 这一步预测是比较粗糙的， 主要是为了提取出proposal。 前景点可能很多， 预测的框就比较多， 为了避免这一问题， 会基于框的打分以及NMS过滤掉大部分的框。

第二阶段，在上一步提取到的proposal的基础上进行进一步的refine。
值得注意的是， 在对框的预测上没有直接采用回归的方式， 而是把要回归的量转换成许多离散的bin， 预测实际值属于哪一个bin即可。 这样就把一把连续的回归问题转换成一个有限类别的分类问题， 减小了预测的难度。

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

下载数据集于主目录`kitti`文件夹中，数据集目录结构如下：

```bash
.
├── kitti
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
│
```

其中,ImgaeSets下的文件可以在[image_sets](https://github.com/charlesq34/frustum-pointnets/tree/master/kitti/image_sets)下载。

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，由于该网络目前缺少算子的反向计算过程，因此暂不支持训练过程。评估模型之前，需下载Pytorch版本的`PointRCNN.pth`模型，将模型转化为Mindsproe模型后，进行推理评估。

## 工具安装

```shell
cd src
git clone https://github.com/traveller59/kitti-object-eval-python
```

请参考src/kitti-object-eval-python/README.md的说明安装相关依赖

## 模型转化

请将`PointRCNN.pth`文件放在项目主目录下，运行`pth2ckpt.py`脚本。

```bash
python pth2ckpt.py
```

脚本正常运行后，项目主目录下生成`PointRCNN.ckpt`文件。

## 训练模型

由于模型训练相关算子反向计算未实现，因此暂不支持训练过程。

## 评估模型

```bash
bash scripts/run_eval_gpu.sh [CKPT]
# example: bash scripts/run_eval_gpu.sh PointRCNN.ckpt
```

评估日志为`eval_gpu.log`。

# 脚本说明

## 脚本及样例代码

```bash
.
├── README_CN.md
├── config                                  // config文件
│   └── default.yaml                        // config文件
├── data                                    // 数据存放文件夹
├── eval.py                                 // 推理主文件
├── kitti                                   // KITTI数据集标签
│   └── KITTI
│       └── ImageSets
│           ├── test.txt
│           ├── train.txt
│           ├── trainval.txt
│           └── val.txt
├── reqeuiment.txt                          // 依赖文件
├── pth2ckpt.py                             // 参数转换脚本
├── scripts                                 // 脚本文件
│   ├── build_and_install.sh                // 编译脚本
│   └── eval.sh                             // 推理脚本
├── src                                     // 模型主文件
│   ├── __init__.py
│   ├── _init_path.py
│   ├── datautil.py
│   ├── generate_aug_scene.py
│   ├── generate_gt_database.py
│   ├── layer_utils.py
│   ├── lib                                         // 网络库文件
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── datasets
│   │   │   ├── kitti_dataset.py
│   │   │   └── kitti_rcnn_dataset.py
│   │   ├── net
│   │   │   ├── __init__.py
│   │   │   ├── ms_loss.py
│   │   │   ├── point_rcnn.py
│   │   │   ├── pointnet2_msg.py
│   │   │   ├── rcnn_net.py
│   │   │   ├── rpn.py
│   │   │   └── train_functions.py
│   │   ├── rpn
│   │   │   ├── proposal_layer.py
│   │   │   └── proposal_target_layer.py
│   │   └── utils
│   │       ├── bbox_transform.py
│   │       ├── calibration.py
│   │       ├── iou3d                               // iou3d cuda算子
│   │       │   ├── iou3d_utils.py
│   │       │   ├── setup.py                        // 编译脚本
│   │       │   └── src
│   │       │       ├── iou3d.cpp
│   │       │       ├── iou3d_kernel.cu
│   │       │       ├── ms_ext.cpp
│   │       │       └── ms_ext.h
│   │       ├── kitti_utils.py
│   │       ├── loss_utils.py
│   │       ├── object3d.py
│   │       └── roipool3d                           // roipool3d cuda算子
│   │           ├── roipool3d_utils.py
│   │           ├── setup.py                        // 编译脚本
│   │           └── src
│   │               ├── ms_ext.cpp
│   │               ├── ms_ext.h
│   │               ├── roipool3d.cpp
│   │               └── roipool3d_kernel.cu
│   ├── pointnet2_lib                               // pointnet2依赖库
│   │   ├── pointnet2_train.log
│   │   ├── src
│   │   │   ├── __init.__.py
│   │   │   ├── callbacks.py
│   │   │   ├── dataset.py
│   │   │   ├── layers.py
│   │   │   ├── lr_scheduler.py
│   │   │   ├── pointnet2.py
│   │   │   ├── pointnet2_cuda                      // pointnet2第三方cuda算子
│   │   │   │   ├── ball_query.cpp
│   │   │   │   ├── ball_query_gpu.cu
│   │   │   │   ├── ball_query_gpu.h
│   │   │   │   ├── cuda_utils.h
│   │   │   │   ├── group_points.cpp
│   │   │   │   ├── group_points_gpu.cu
│   │   │   │   ├── group_points_gpu.h
│   │   │   │   ├── interpolate.cpp
│   │   │   │   ├── interpolate_gpu.cu
│   │   │   │   ├── interpolate_gpu.h
│   │   │   │   ├── ms_ext.cpp
│   │   │   │   ├── ms_ext.h
│   │   │   │   ├── pointnet2_api.cpp
│   │   │   │   ├── sampling.cpp
│   │   │   │   ├── sampling_gpu.cu
│   │   │   │   └── sampling_gpu.h
│   │   │   ├── pointnet2_utils.py
│   │   │   ├── provider.py
│   │   │   └── setup.py                            // 编译脚本
│   └── train_utils
│       └── train_utils.py                          // 训练用工具脚本

```

## 脚本参数

在eval.py中可以配置评估参数。

```bash
usage: eval.py [-h] [--cfg_file CFG_FILE] --eval_mode EVAL_MODE [--eval_all]
               [--test] [--ckpt CKPT] [--rpn_ckpt RPN_CKPT]
               [--rcnn_ckpt RCNN_CKPT] [--batch_size BATCH_SIZE]
               [--workers WORKERS] [--extra_tag EXTRA_TAG]
               [--output_dir OUTPUT_DIR] [--ckpt_dir CKPT_DIR] [--save_result]
               [--save_rpn_feature] [--random_select]
               [--start_epoch START_EPOCH]
               [--rcnn_eval_roi_dir RCNN_EVAL_ROI_DIR]
               [--rcnn_eval_feature_dir RCNN_EVAL_FEATURE_DIR] [--set ...]

evaluate PointRCNN Model

optional arguments:
  -h, --help            show this help message and exit
  --cfg_file CFG_FILE   specify the config for evaluation
  --eval_mode EVAL_MODE
                        specify the evaluation mode
  --eval_all            whether to evaluate all checkpoints
  --test                evaluate without ground truth
  --ckpt CKPT           specify a checkpoint to be evaluated
  --rpn_ckpt RPN_CKPT   specify the checkpoint of rpn if trained separated
  --rcnn_ckpt RCNN_CKPT
                        specify the checkpoint of rcnn if trained separated
  --batch_size BATCH_SIZE
                        batch size for evaluation
  --workers WORKERS     number of workers for dataloader
  --extra_tag EXTRA_TAG
                        extra tag for multiple evaluation
  --output_dir OUTPUT_DIR
                        specify an output directory if needed
  --ckpt_dir CKPT_DIR   specify a ckpt directory to be evaluated if needed
  --save_result         save evaluation results to files
  --save_rpn_feature    save features for separately rcnn training and
                        evaluation
  --random_select       sample to the same number of points
  --start_epoch START_EPOCH
                        ignore the checkpoint smaller than this epoch
  --rcnn_eval_roi_dir RCNN_EVAL_ROI_DIR
                        specify the saved rois for rcnn evaluation when using
                        rcnn_offline mode
  --rcnn_eval_feature_dir RCNN_EVAL_FEATURE_DIR
                        specify the saved features for rcnn evaluation when
                        using rcnn_offline mode
  --set ...             set extra config keys if needed

```

更多配置细节请参考脚本`eval.py`。

## 训练过程

由于缺少相关算子，因此训练过程没有实现。

## 评估过程

### 评估

- 在GPU环境运行时评估kitti数据集

  ```bash
  bash scripts/run_eval_gpu.sh [CKPT]
  # example: bash scripts/run_eval_gpu.sh PointRCNN.ckpt
  ```

  评估结果保存在文件`eval_gpu.log`中，具体内容（部分）如下：

  ```bash
  2022-10-08 21:06:37,063   INFO  final average cls acc refined: 0.000
  2022-10-08 21:06:37,064   INFO  total roi bbox recall(thresh=0.100): 12772 / 14385 = 0.887869
  2022-10-08 21:06:37,064   INFO  total roi bbox recall(thresh=0.300): 12691 / 14385 = 0.882238
  2022-10-08 21:06:37,064   INFO  total roi bbox recall(thresh=0.500): 12498 / 14385 = 0.868822
  2022-10-08 21:06:37,065   INFO  total roi bbox recall(thresh=0.700): 9659 / 14385 = 0.671463
  2022-10-08 21:06:37,065   INFO  total roi bbox recall(thresh=0.900): 55 / 14385 = 0.003823
  2022-10-08 21:06:37,065   INFO  total bbox recall(thresh=0.100): 12775 / 14385 = 0.888078
  2022-10-08 21:06:37,066   INFO  total bbox recall(thresh=0.300): 12697 / 14385 = 0.882656
  2022-10-08 21:06:37,066   INFO  total bbox recall(thresh=0.500): 12600 / 14385 = 0.875912
  2022-10-08 21:06:37,066   INFO  total bbox recall(thresh=0.700): 11384 / 14385 = 0.791380
  2022-10-08 21:06:37,066   INFO  total bbox recall(thresh=0.900): 2299 / 14385 = 0.159819
  2022-10-08 21:06:37,067   INFO  Averate Precision:
  2022-10-08 21:06:51,313   INFO  Car AP@0.70, 0.70, 0.70:
  bbox AP:97.5822, 89.4141, 86.9551
  bev  AP:90.2099, 85.4985, 80.1179
  3d   AP:88.8803, 78.4792, 77.3416
  aos  AP:97.57, 89.29, 86.77
  Car AP@0.70, 0.50, 0.50:
  bbox AP:97.5822, 89.4141, 86.9551
  bev  AP:96.0476, 89.7339, 88.9142
  3d   AP:96.0148, 89.6869, 88.7116
  aos  AP:97.57, 89.29, 86.77
  ```

# 模型描述

## 性能

### 推理性能

| 参数          | GPU                                                   |
| ------------- | ----------------------------------------------------- |
| 模型版本      | PointRCNN                                     |
| 资源          | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G |
| 上传日期      | 2022-09-23                                            |
| MindSpore版本 | 1.8.0                                                 |
| 数据集        | KITTI                                                 |
| batch_size    | 32                                                    |
| 输出          | 概率                                                  |
| 准确性        | 76.96%                                                |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
