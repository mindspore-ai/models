# Contents

<!-- TOC -->

- [Contents](#contents)
- [EfficientNet-B2 Description](#efficientnet-b2-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Startup](#startup)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Startup](#startup-1)
        - [Result](#result-1)
    - [ONNX Inference](#onnx-inference)
        - [Exporting ONNX](#exporting-onnx)
        - [ONNX Inference on the GPU](#onnx-inference-on-the-gpu)
    - [Inference on Ascend 310 Processor](#inference-on-ascend-310-processor)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo](#modelzoo)

<!-- /TOC -->

# EfficientNet-B2 Description

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. (2019)

[Paper](https://arxiv.org/abs/1905.11946): Mingxing Tan, Quoc V. Le.EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks. 2019.

# Model Architecture

The overall EfficientNet architecture is described at:

[Link](https://arxiv.org/abs/1905.11946)

# Dataset

Used dataset: [ImageNet](http://www.image-net.org/)

- Dataset size: 146 GB, 1,330,000 color images of 1000 classes
    - Training set: 140 GB, 1,280,000 images
    - Test set: 6 GB, 50,000 images
- Data format: RGB
    - Note: Data is processed in **src/dataset.py**.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Script Description

## Script and Sample Code

```python
├── EfficientNet - B2
├── README_CN.md  # Description of EfficientNet-B2
├── scripts
│   ├──run_standalone_train.sh  # Shell script for single-device training
│   ├──run_standalone_train_gpu.sh  # Shell script for single-device training (GPU)
│   ├──run_distribute_train.sh  # Shell script for 8-device training
│   ├──run_train_gpu.sh  # Shell script for 8-device training (GPU)
│   ├──run_eval_gpu.sh  # Shell script for GPU evaluation
│   ├──run_infer_onnx.sh  # Shell script for ONNX evaluation
│   ├──run_distribute_resume.sh  # Shell script for resuming training
│   └──run_eval.sh  # Shell script for evaluation
├── src
│   ├──models  # EfficientNet-B2 architecture
│   │   ├──effnet.py
│   │   └──layers.py
│   ├──config.py  # Parameter configuration
│   ├──dataset.py  # Dataset creation
│   ├──loss.py  # Loss function
│   ├──lr_generator.py  # Learning rate configuration
│   └──Monitor.py  # Monitor network loss and other data.
├── eval.py  # Evaluation script
├── export.py  # Model format conversion script
└── train.py  # Training script
```

## Script Parameters

The parameters used during model training and evaluation can be set in the **config.yaml** file.

```python
'class_num': 1000,  # Number of classes in the dataset.
'batch_size': 256,  # Batch size.
'loss_scale': 1024,  # Loss scale.
'momentum': 0.9,  # Momentum parameter.
'weight_decay':1e-5,  # Weight decay rate.
'epoch_size': 350,  # Number of model epochs.
'save_checkpoint': True,  # Specifies whether to save a CKPT file.
'save_checkpoint_epochs': 1,  # Number of epochs for saving a CKPT file.
'keep_checkpoint_max': 5,  # Maximum number of CKPT files that can be saved.
'save_checkpoint_path': "./checkpoint",  # Path for storing the CKPT file.
'opt': 'rmsprop',  # Optimizer.
'opt_eps': 0.001,  # Optimizer parameter for improving value stability.
'warmup_epochs': 2,  # Number of warm-up epochs.
'lr_decay_mode': 'liner',  # Learning rate decay mode.
'use_label_smooth: True,  # Specifies whether to use label smoothing.
'label_smooth_factor':0.1,  # Label smoothing factor.
'lr_init': 0.0001,  # Initial learning rate.
'lr_max': 0.13,  #Maximum learning rate
'lr_end': 0.00001,  # End learning rate.
```

## Training Process

### Startup

You can use Python or shell scripts for training.

```shell
# Training example
  python:
      Single-device training (Ascend): python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]
      Single-device training (GPU): python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --dataset_target [DEVICE_TARGET]

  shell:
      Single-device training (Ascend): bash ./run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
      8-device parallel training (Ascend): bash ./run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]
      Single-device training (GPU): bash ./run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR]
      8-device parallel training (GPU): bash ./run_train_gpu.sh [DEVICE_NUM] [DEVICE_ID(0,1,2,3,4,5,6,7)] [DATA_DIR]
```

### Result

The CKPT file is stored in the `./checkpoint` directory, and training logs are recorded in the `log.txt` directory. An example of a training log is as follows:

```shell
epoch 1: epoch time: 1301358.75, per step time: 2082.174, avg loss: 5.814
epoch 2: epoch time: 645634.656, per step time: 1033.015, avg loss: 4.786
epoch 3: epoch time: 645646.679, per step time: 1033.035, avg loss: 4.152
epoch 4: epoch time: 645604.903, per step time: 1032.968, avg loss: 3.719
epoch 5: epoch time: 645621.756, per step time: 1032.995, avg loss: 3.342
```

## Evaluation Process

### Startup

You can use Python or shell scripts for evaluation.

```shell
# Evaluation example
  python:
      Ascend evaluation example: python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]
      GPU evaluation example: python eval.py [DEVICE_ID] [PATH_CHECKPOINT] [DATA_DIR] [DEVICE_TARGET]

  shell:
      Ascend evaluation example: bash ./run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
      GPU evaluation example: bash run_eval_gpu.sh [DEVICE_ID] [PATH_CHECKPOINT] [DATA_DIR] [DEVICE_TARGET]
```

> The CKPT file can be generated during training.

### Result

You can view the evaluation results in `eval_log.txt`.

```shell
result: {'Loss': 1.7495090191180889, 'Top_1_Acc': 0.7979567307692308, 'Top_5_Acc': 0.9468549679487179} ckpt = ./checkpoint/model_0/Efficientnet_b2-rank0-350_625.ckpt
```

## ONNX Inference

### Exporting ONNX

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 260 --height 260 --file_format ONNX --device_target GPU
```

### ONNX Inference on the GPU

Before inference, the ONNX file must be exported through the `export.py` script. The following shows an example of using the ONNX model to perform inference.

```bash
# ONNX inference
bash scripts/run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [DEVICE_TARGET]
```

## Inference on Ascend 310 Processor

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

# Model Description

## Training Performance

| Parameter                       | Ascend                                | GPU                             |
| -------------------------- | ------------------------------------- | ------------------------------------- |
| Model name                   | EfficientNet                          | EfficientNet              |
| Model version                   | B2                           | B2                         |
| Operating environment                   | HUAWEI CLOUD Modelarts                     | Ubuntu 18.04 GeForce RTX 3090 |
| Upload date                   | 2021-8-17                             | 2021-11-20                  |
| Dataset                     | imagenet                              | imagenet                     |
| Training parameters                   | src/config.py                         | src/config.py            |
| Optimizer                     | RMSProp                              | RMSProp                       |
| Loss function                   | CrossEntropySmooth         | CrossEntropySmooth |
| Final loss                   | 1.75                                  | 1.76                              |
| Accuracy (8-device)                | Top1[79.80%], Top5[94.69%]               | Top1[79.57%], Top5[94.72%] |
| Total training duration (8-device)            | 64.87h                                    | 105h                               |
| Total evaluation duration                 | 1min                                    | 2min                               |

# Random Seed Description

We set random seeds in the `dataset.py` and `train.py` scripts.

# ModelZoo

For details, please visit the [official website](https://gitee.com/mindspore/models).
