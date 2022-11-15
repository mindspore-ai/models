# Contents

<!-- TOC -->

[查看中文](./README_CN.md)

- [Simple Baselines Description](#simple_baselines-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage1)
        - [Result](#result1)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage2)
        - [Result](#result2)
    - [Inference Process](#inference-process)
        - [Model Export](#model-export)
        - [Infer on Ascend310](#infer-ascend310)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random State](#description-of-random-state)
- [ModelZoo Homepage](#ModelZoo-homepage)

<!-- /TOC -->

# Simple Baselines Description

## Overview

Simple Baselines proposed by Bin Xiao, Haiping Wu, and Yichen Wei from Microsoft Research Asia. The authors believe that
the current popular human pose estimation and tracking methods are too complicated. The existing human pose estimation and
pose tracking models seem to be quite different in structure, but It's really close in terms of performance. The author proposes
a simple and effective baseline method by adding a deconvolution layer on the backbone network ResNet, which is precisely the
simplest method to estimate the heatmap from the high and low resolution feature maps, thereby helping to stimulate and evaluate
new ideas in the field.

For more details refer to [paper](https://arxiv.org/pdf/1804.06208.pdf).
Mindspore implementation is based on [original pytorch version](https://github.com/microsoft/human-pose-estimation.pytorch) released by Microsoft Asia Research Institute.

## Paper

[Paper](https://arxiv.org/pdf/1804.06208.pdf): Bin Xiao, Haiping Wu, Yichen Wei "Simple baselines for human pose estimation and tracking"

# Model Architecture

The overall network architecture of simple baselines is [here](https://arxiv.org/pdf/1804.06208.pdf).

# Dataset

Dataset used: [COCO2017](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download)

- Dataset size：
    - Train：19.56GB, 57k images, 149813 person instances
    - Test：825MB, 5k images, 6352 person instances
- Data Format：JPG
    - Note: Data is processed in src/dataset.py

# Features

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training
method accelerates the deep learning neural network training process by using both the single-precision and half-precision
data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision
training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained
on specific hardware. For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle
it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# Environment Requirements

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information about MindSpore, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# Quick Start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation.

- Dataset preparation

The simple baselines uses the COCO2017 dataset for training and evaluation. Download dataset from [official website](https://cocodataset.org/).

- Running on Ascend

```shell
# Distributed training
bash scripts/run_distribute_train.sh RANK_TABLE

# Standalone training
bash scripts/run_standalone_train.sh DEVICE_ID

# Evaluation
bash scripts/run_eval.sh
```

- Running on GPU

```shell
# Distributed training
bash scripts/run_distribute_train_gpu.sh DEVICE_NUM

# Standalone training
bash scripts/run_standalone_train_gpu.sh DEVICE_ID

# Evaluation
bash scripts/run_eval_gpu.sh DEVICE_ID
```

# Script Description

## Script and Sample Code

```text
.
└──simple_baselines
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # train on Ascend
    ├── run_distribute_train_gpu.sh        # train on GPU
    ├── run_eval.sh                        # eval on Ascend
    ├── run_eval_gpu.sh                    # eval on GPU
    ├── run_standalone_train.sh            # train on Ascend
    ├── run_standalone_train_gpu.sh        # train on GPU
    └── run_infer_310.sh
  ├── src
    ├── utils
        ├── coco.py                        # COCO dataset evaluation results
        ├── nms.py
        └── transforms.py                  # Image processing conversion
    ├── config.py
    ├── dataset.py                         # Data preprocessing
    ├── network_with_loss.py               # Loss function
    ├── pose_resnet.py                     # Backbone network
    └── predict.py                         # Heatmap key point prediction
  ├── export.py
  ├── postprocess.py
  ├── preprocess.py
  ├── eval.py
  └── train.py
```

## Script Parameters

Before training configure parameters and paths in src/config.py.

- Model parameters:

```text
config.MODEL.INIT_WEIGHTS = True                                 # Initialize model weights
config.MODEL.PRETRAINED = 'resnet50.ckpt'                        # Pre-trained model
config.MODEL.NUM_JOINTS = 17                                     # Number of key points
config.MODEL.IMAGE_SIZE = [192, 256]                             # Image size
```

- Network parameters:

```text
config.NETWORK.NUM_LAYERS = 50                                   # Resnet backbone layers
config.NETWORK.DECONV_WITH_BIAS = False                          # Network deconvolution bias
config.NETWORK.NUM_DECONV_LAYERS = 3                             # Number of network deconvolution layers
config.NETWORK.NUM_DECONV_FILTERS = [256, 256, 256]              # Deconvolution layer filter size
config.NETWORK.NUM_DECONV_KERNELS = [4, 4, 4]                    # Deconvolution layer kernel size
config.NETWORK.FINAL_CONV_KERNEL = 1                             # Final convolutional layer kernel size
config.NETWORK.HEATMAP_SIZE = [48, 64]
```

- Training parameters:

```text
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.LR = 0.001
config.TRAIN.LR_FACTOR = 0.1                 # learning rate reduction factor
config.TRAIN.LR_STEP = [90, 120]
config.TRAIN.NUM_PARALLEL_WORKERS = 8
config.TRAIN.SAVE_CKPT = True
config.TRAIN.CKPT_PATH = "./model"           # directory of pretrained resnet50 and to save ckpt
config.TRAIN.SAVE_CKPT_EPOCH = 3
config.TRAIN.KEEP_CKPT_MAX = 10
```

- Evaluation parameters:

```text
config.TEST.BATCH_SIZE = 32
config.TEST.FLIP_TEST = True
config.TEST.USE_GT_BBOX = False
```

- nms parameters:

```text
config.TEST.OKS_THRE = 0.9                                       # OKS threshold
config.TEST.IN_VIS_THRE = 0.2                                    # Visualization threshold
config.TEST.BBOX_THRE = 1.0                                      # Candidate box threshold
config.TEST.IMAGE_THRE = 0.0                                     # Image threshold
config.TEST.NMS_THRE = 1.0                                       # nms threshold
```

## Training Process

### Usage

- Ascend

```shell
# Distributed training 8p
bash scripts/run_distribute_train.sh RANK_TABLE

# Standalone training
bash scripts/run_standalone_train.sh DEVICE_ID

# Evaluation
bash scripts/run_eval.sh
```

- GPU

```shell
# Distributed training
bash scripts/run_distribute_train_gpu.sh DEVICE_NUM

# Standalone training
bash scripts/run_standalone_train_gpu.sh DEVICE_ID

# Evaluation
bash scripts/run_eval_gpu.sh DEVICE_ID
```

### Result

- Use COCO2017 dataset to train simple_baselines

```text
# Standalone training results （1P）
epoch:1 step:2340, loss is 0.0008106
epoch:2 step:2340, loss is 0.0006160
epoch:3 step:2340, loss is 0.0006480
epoch:4 step:2340, loss is 0.0005620
epoch:5 step:2340, loss is 0.0005207
...
epoch:138 step:2340, loss is 0.0003183
epoch:139 step:2340, loss is 0.0002866
epoch:140 step:2340, loss is 0.0003393
```

## Evaluation Process

### Usage

The corresponding model inference can be performed by changing the "config.TEST.MODEL_FILE" file in the config.py file.
Use val2017 in the COCO2017 dataset folder to evaluate simple_baselines.

- Ascend

```shell
# Evaluation
bash scripts/run_eval.sh
```

- GPU

```shell
# Evaluation
bash scripts/run_eval_gpu.sh DEVICE_ID
```

### Result

results will be saved in keypoints_results.pkl

```text
AP: 0.704
```

## Inference Process

### Model Export

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

- Export in local

```shell
python export.py
```

- Export in ModelArts (If you want to run in modelarts, please check [modelarts official document](https://support.huaweicloud.com/modelarts/).

```text
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/simple_pose" on the website UI interface.
# (4) Set the startup file to /{path}/simple_pose/export.py" on the website UI interface.
# (5) Perform a .
#     a. setting parameters in /{path}/simple_pose/default_config.yaml.
#         1. Set ”enable_modelarts: True“
#         2. Set “TEST.MODEL_FILE: ./{path}/*.ckpt”('TEST.MODEL_FILE' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”EXPORT.FILE_NAME: simple_pose“
#         4. Set ”EXPORT.FILE_FORMAT：MINDIR“
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see simple_pose.mindir under {Output file path}.
```

`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### 310 inference

Before performing inference, the mindir file must bu exported by export.py script. We only provide an example of inference using MINDIR model.
When the network is processing the dataset, if the last batch is not enough, it will not be automatically supplemented. Better set batch_size to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` indicates that dataset is processed in binary format, value are "y" or "n".
- `DEVICE_ID` optional, default value is 0.

### Result

The inference results are saved in the current path in `acc.log` file.

```text
AP: 0.7139169694686592
```

# Model Description

## Performance

| Parameters          | Ascend 910                  | GPU 1p | GPU 8p |
| ------------------- | --------------------------- | ------------ | ------------ |
| Model               | simple_baselines            | simple_baselines | simple_baselines |
| Environment         | Ascend 910；CPU 2.60GHz，192cores；RAM：755G  | Ubuntu 18.04.6, 1p RTX3090, CPU 2.90GHz, 64cores, RAM 252GB; Mindspore 1.5.0 | Ubuntu 18.04.6, 8pcs RTX3090, CPU 2.90GHz, 64cores, RAM 252GB; Mindspore 1.5.0 |
| Upload date (Y-M-D) | 2021-03-29                  | 2021-12-29 | 2021-12-29 |
| MindSpore Version   | 1.1.0                       | 1.5.0 | 1.5.0 |
| Dataset             | COCO2017                    | COCO2017 | COCO2017 |
| Training params     | epoch=140, batch_size=64    | epoch=140, batch_size=64 | epoch=140, batch_size=64 |
| Optimizer           | Adam                        | Adam | Adam |
| Loss function       | Mean Squared Error          | Mean Squared Error | Mean Squared Error |
| Output              | heatmap                     | heatmap | heatmap |
| Final Loss          |                             | 0.27 | 0.27 |
| Training speed      | 1pc: 251.4 ms/step          | 184 ms/step | 285 ms/step |
| Total training time |                             | 17h | 3.5h |
| Accuracy            | AP: 0.704                   | AP: 0.7143 | AP: 0.7143 |

# Description of Random State

Random seed is set inside "create_dataset" function in dataset.py.
Initial network weights are used in model.py.

# ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).
