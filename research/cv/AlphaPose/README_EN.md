# Contents

<!-- TOC -->

- [Alphapose Description](#alphapose-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [mixed precision](#mixed-precision)
- [Environmental requirements](#environmental-requirements)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Scripts and sample code](#scripts-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
    - [Evaluation process](#evaluation-process)
    - [310 Inference Process](#310-inference-process)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Evaluation performance](#evaluation-performance)
        - [Inference performance](#inference-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# Alphapose Description

## Overview

AlphaPose was proposed by Lu Cewu's team of Shanghai Jiaotong University, and the author proposed the Regional Multi-Person Pose Estimation (RMPE) framework. Mainly include symmetric spatial transformer network (SSTN), Parametric Pose Non-Maximum-Suppression (NMS), and Pose-Guided Proposals Generator (PGPG). And use symmetric spatial transformer network (SSTN), deep proposals generator (DPG), parametric pose nonmaximum suppression (p-NMS) three techniques to solve the problem of multi-person pose estimation in complex scenes.

For details of the AlphaPose model network, please refer to [Paper 1](https://arxiv.org/pdf/1612.00137.pdf)，The Mindspore implementation of the AlphaPose model network is based on the Pytorch version released by the Lu Cewu team of Shanghai Jiaotong University. For details, please refer to (<https://github.com/MVIG-SJTU/AlphaPose>.

## paper

1. [paper](https://arxiv.org/pdf/1804.06208.pdf)：Fang H S , Xie S , Tai Y W , et al. RMPE: Regional Multi-person Pose Estimation

# Model Architecture

The overall network architecture of AlphaPose is as follows:
[Link](https://arxiv.org/abs/1612.00137)

# Dataset

Datasets used: [COCO2017](https://cocodataset.org/#download)

- Dataset size:
    - Training set: 19.56G, 118,287 images
    - Test set: 825MB, 5,000 images
- Data format: JPG file
    - Note: Data is processed in src/dataset.py

# Features

## mixed precision

The training method using [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.6/enable_mixed_precision.html) uses support for single-precision and half-precision data to improve the training speed of deep learning neural networks , while maintaining the network accuracy that single-precision training can achieve. Mixed-precision training increases computational speed and reduces memory usage while enabling training of larger models on specific hardware or enabling larger batches of training.
Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore background will automatically reduce the precision to process the data. You can open the INFO log and search for "reduce precision" to view operators with reduced precision.

# Environmental requirements

- Hardware (Ascend)
    - Prepare the Ascend processor to build the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Pre-trained models

  AlphaPose model uses ResNet-50 network trained on ImageNet as backbone. You can run the Resnet training script in [official model zoo](https://gitee.com/mindspore/models/tree/master/official/cv/resnet) to get the model weight file or download trained checkpoint from [here](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs32_acc77.06/). The pre-training file name should be resnet50.ckpt.

- Dataset preparation

  The Alphapose network model uses the COCO2017 dataset for training and inference. The dataset can be downloaded from the [official website](https://cocodataset.org/) official website.

- Configuration

  Set desired configuration in ```default_config.yaml``` file or create new one.

- Ascend processor environment to run

```bash
# Distributed training
bash scripts/run_distribute_train.sh --is_model_arts False --run_distribute True

# Stand-alone training
bash scripts/run_standalone_train.sh --device_id 0

# Run the evaluation example
bash scripts/run_eval.sh [DEVICE_TARGET] [CONFIG] [CKPT_PATH] [DATASET]

# run demo
bash scripts/run_demo.sh
```

- GPU environment to run

```bash
# Distributed training
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [dataset_dir] [pretrained_backbone]

# Stand-alone training
bash scripts/run_standalone_train_gpu.sh [config_file] [dataset_dir] [pretrained_backbone]

# Run the evaluation example
bash scripts/run_eval.sh [DEVICE_TARGET] [CONFIG] [CKPT_PATH] [DATASET]
```

# Script description

## Scripts and sample code

```text

└──AlphaPose
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # Start Ascend distributed training (8 cards)
    ├── run_distribute_train_gpu.sh        # Start GPU distributed training (8 cards)
    ├── run_demo.sh                        # Start the demo (single card)
    ├── run_eval.sh                        # Start Ascend eval
    ├── run_standalone_train.sh            # Start Ascend stand-alone training (single card)
    └── run_standalone_train_gpu.sh        # Start GPU stand-alone training (single card)
  ├── src
    ├── utils
        ├── coco.py                        # COCO dataset evaluation tools
        ├── fn.py                          # Drawing human poses based on key points
        ├── inference.py                   # Heatmap keypoint prediction
        ├── nms.py                         # nms
        └── transforms.py                  # Image processing transformation
    ├── config.py                          # parameter configuration
    ├── dataset.py                         # data preprocessing
    ├── DUC.py                             # Network part structure DUC
    ├── FastPose.py                        # Backbone network definition
    ├── network_with_loss.py               # Loss function definition
    ├── SE_module.py                       # Network part structure SE
    └── SE_module.py                       # Part of the network structure ResNet50
  ├── demo.py                              # demo
  ├── data_to_bin.py                       # Convert the images in the dataset to binary
  ├── default_config.yaml                  # Default configuration file
  ├── requirements.txt                     # pip requirements
  ├── export.py                            # Convert ckpt model file to mindir
  ├── postprocess.py                       # Post-processing precision
  ├── eval.py                              # Evaluate the network
  └── train.py                             # train the network
```

## script parameters

Configure relevant parameters in ```default_config.yaml```.

- Configure model related parameters:

```python
MODEL_INIT_WEIGHTS = True                                 # Initialize model weights
MODEL_PRETRAINED = 'resnet50.ckpt'                        # pretrained model
MODEL_NUM_JOINTS = 17                                     # number of key points
MODEL_IMAGE_SIZE = [192, 256]                             # image size
```

- Configure network related parameters:

```python
NETWORK_NUM_LAYERS = 50                                   # Resnet backbone network layers
NETWORK_DECONV_WITH_BIAS = False                          # network deconvolution bias
NETWORK_NUM_DECONV_LAYERS = 3                             # The number of network deconvolution layers
NETWORK_NUM_DECONV_FILTERS = [256, 256, 256]              # Deconvolution layer filter size
NETWORK_NUM_DECONV_KERNELS = [4, 4, 4]                    # Deconvolution layer kernel size
NETWORK_FINAL_CONV_KERNEL = 1                             # Final convolutional layer kernel size
NETWORK_HEATMAP_SIZE = [48, 64]                           # Heatmap size
```

- Configure training related parameters:

```python
TRAIN_SHUFFLE = True                                      # training data in random order
TRAIN_BATCH_SIZE = 64                                     # training batch size
TRAIN_BEGIN_EPOCH = 0                                     # Test dataset filename
DATASET_FLIP = True                                       # The dataset is randomly flipped
DATASET_SCALE_FACTOR = 0.3                                # dataset random scale factor
DATASET_ROT_FACTOR = 40                                   # Dataset random rotation factor
TRAIN_BEGIN_EPOCH = 0                                     # number of initial cycles
TRAIN_END_EPOCH = 270                                     # number of final cycles
TRAIN_LR = 0.001                                          # initial learning rate
TRAIN_LR_FACTOR = 0.1                                     # Learning rate reduction factor
```

- Configure test related parameters:

```python
TEST_BATCH_SIZE = 32                                      # test batch size
TEST_FLIP_TEST = True                                     # flip validation
TEST_USE_GT_BBOX = False                                  # Use gt boxes
```

- Configure nms related parameters:

```python
TEST_OKS_THRE = 0.9                                       # OKS threshold
TEST_IN_VIS_THRE = 0.2                                    # Visualization Threshold
TEST_BBOX_THRE = 1.0                                      # candidate box threshold
TEST_IMAGE_THRE = 0.0                                     # image threshold
TEST_NMS_THRE = 1.0                                       # nms threshold
```

- Configure demo related parameters:

```python
detect_image = "images/1.jpg"                             # Detect pictures
yolo_image_size = [416, 416]                              # yolo network input image size
yolo_ckpt = "yolo/yolo.ckpt"                              # yolo network weight
fast_pose_ckpt = "fastpose.ckpt"                          # fastpose network weights
yolo_threshold = 0.1                                      # bbox threshold
```

## training process

### usage

#### Ascend processor environment running

```bash
# Distributed training
bash scripts/run_distribute_train.sh --is_model_arts False --run_distribute True

# Stand-alone training
bash scripts/run_standalone_train.sh --device_id 0

# Run the evaluation example
bash scripts/run_eval.sh checkpoint_path device_id
```

#### GPU environment

```bash
# Distributed training
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [dataset_dir] [pretrained_backbone]

# Stand-alone training
bash scripts/run_standalone_train_gpu.sh [config_file] [dataset_dir] [pretrained_backbone]

# Run the evaluation example
bash scripts/run_eval.sh [DEVICE_TARGET] [CONFIG] [CKPT_PATH] [DATASET]
```

### result

- Train Alphapose with COCO2017 dataset

```text
Distributed training results (8P)
epoch:1 step:292, loss is 0.001391
epoch:2 step:292, loss is 0.001326
epoch:3 step:292, loss is 0.001001
epoch:4 step:292, loss is 0.0007763
epoch:5 step:292, loss is 0.0006757
...
epoch:288 step:292, loss is 0.0002837
epoch:269 step:292, loss is 0.0002367
epoch:270 step:292, loss is 0.0002532
```

## evaluation process

### usage

#### Ascend processor environment running

The corresponding model inference can be performed by changing the "TEST_MODEL_FILE" file in the config file.

```bash
# evaluate
bash scripts/run_eval.sh [DEVICE_TARGET] [CONFIG] [CKPT_PATH] [DATASET]
```

#### GPU environment

```bash
# Run the evaluation example
bash scripts/run_eval.sh [DEVICE_TARGET] [CONFIG] [CKPT_PATH] [DATASET]
```

### result

Alphapose is evaluated using val2017 in the COCO2017 dataset folder as follows:

```text
coco eval results saved to /cache/train_output/multi_train_poseresnet_v5_2-140_2340/keypoints_results.pkl
AP: 0.723
```

## 310 Inference Process

### usage

#### export model

```python
# export model
python export.py --ckpt_url [ckpt_url] --device_target [device_target] --device_id [device_id] --file_name [file_name] --file_format [file_format]
```

#### Ascend310 processor environment running

```bash
# 310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

#### Acquire accuracy

```bash
# Acquire accuracy
more acc.log
```

### result

```text
AP: 0.723
```

# Model description

## performance

### Evaluation performance

#### Performance parameters on coco2017

| parameter                 | Ascend                                      | GPU                |
| -------------------------- | ------------------------------------------ | ------------------ |
| model version              | ResNet50                                   | ResNet50           |
| resource                   | Ascend 910 ；CPU 2.60GHz，192核；内存：755G  | 8p RTX 3090 24GB   |
| upload date                | 2020-12-16                                 | 2022-02-16         |
| MindSpore version          | 1.3                                        | 1.6                |
| data set                    | coco2017                                  | coco2017
| training parameters        | epoch=270, steps=2336, batch_size = 64, lr=0.001 | epoch=270, batch_size = 128, lr=0.001 |
| optimizer                  | Adam                                       | Adam               |
| loss function              | Mean Squared Error                         | Mean Squared Error |
| output                    | heatmap                                     | heatmap            |
| loss                       | 0.00025                                    | 0.00026            |
| speed                      | 单卡：138.9毫秒/步;  8卡：147.28毫秒/步        | 8p: 441 ms/step    |
| total duration                 | 单卡：24h22m36s;  8卡：3h13m31s          | 8p: 04h 48m 00s    |
| parameter(M)             | 13.0                                         | 13.0               |
| Fine-tune checkpoints | 389.64M (.ckpt文件)                              | 338M (.ckpt)       |
| inference model        | 57.26M (.om文件),  112.76M(.MINDIR文件)          | -                  |

### Inference performance

#### Performance parameters on coco2017

| parameter          | Ascend                   | GPU           |
| ------------------- | ----------------------- | ------------ |
| model version       | ResNet50                | ResNet50     |
| resource            | Ascend 910              | RTX 3090 24 GB |
| upload date       | 2020-12-16                | 2022-02-16   |
| MindSpore Version | 1.3                       | 1.6          |
| data set             | coco2017               | coco2017     |
| batch_size          | 32                      | 32           |
| output             | heatmap                  | heatmap      |
| accuracy            | 单卡: 72.3%;  8卡：72.5% | 72.2 %       |
| inference model | 389.64M (.ckpt文件)         | 338M (.ckpt)  |

# Random Seed Description

The seed in the "create_dataset" function is set in dataset.py, and the initial network weights are used in model.py.

# ModelZoo Homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models).
