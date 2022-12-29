# Contents

<!-- TOC -->

- [Contents](#contents)
- [EfficientNet-B3 Description](#efficientnet-b3-description)
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

# EfficientNet-B3 Description

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth, width, and resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. (2019)

[Paper](https://arxiv.org/abs/1905.11946): Mingxing Tan, Quoc V. Le.EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 2019.

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
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Script Description

## Script and Sample Code

```python
├── EfficientNet-B3
  ├── README_CN.md                 # Description of EfficientNet-B3
  ├── scripts
  │   ├──run_standalone_train.sh       # Shell script for single-device training
  │   ├──run_standalone_train_gpu.sh   # Shell script for single-device training (GPU)
  │   ├──run_distribute_train.sh       # Shell script for 8-device training (Ascend)
  │   ├──run_distribute_train_gpu.sh   # Shell script for 8-device training (GPU)
  │   └──run_eval.sh                   # Shell script for Asced evaluation
  │   └──run_infer_onnx.sh             # Shell script for ONNX evaluation
  │   └──run_eval_gpu.sh               # Shell script for GPU evaluation
  ├── src
  │   ├──models                    # EfficientNet-B3 architecture
  │   │   ├──effnet.py
  │   │   └──layers.py
  │   ├──config.py                 # Parameter configuration
  │   ├──dataset.py                # Dataset creation
  │   ├──loss.py                   # Loss function
  │   ├──lr_generator.py           # Learning rate configuration
  │   └──Monitor.py                # Monitor network loss and other data.
  ├── eval.py                      # Evaluation script
  ├── infer_onnx.py                # ONNX evaluation
  ├── export.py                    # Script for converting the model format
  └── train.py                     # Training script
```

## Script Parameters

The parameters used during model training and evaluation can be set in the **config.yaml** file.

```python
'class_num': 1000,                        # Number of classes in the dataset.
'batch_size': 128,                        # Data batch size.
'loss_scale': 1024,                       # Loss scale.
'momentum': 0.9,                          # Momentum parameter.
'weight_decay': 1e-5,                     # Weight decay rate.
'epoch_size': 350,                        # Number of model iterations.
'save_checkpoint': True,                  # Specifies whether to save a CKPT file.
'save_checkpoint_epochs': 1,             # Number of epochs for saving a CKPT file.
'keep_checkpoint_max': 5,                 # Maximum number of CKPT files that can be saved.
'save_checkpoint_path': "./checkpoint",   # Path for storing the CKPT file.
'opt': 'rmsprop',                         # Optimizer.
'opt_eps': 0.001,                         # Optimizer parameter for improving value stability.
'warmup_epochs': 2,                       # Number of warmup epochs.
'lr_decay_mode': 'liner',                # Learning decay mode.
'use_label_smooth: True,                 # Specifies whether to use label smoothing.
'label_smooth_factor': 0.1,               # Label smoothing factor.
'lr_init': 0.0001,                        # Initial learning rate.
'lr_max': 0.1,                            # Maximum learning rate.
'lr_end': 0.00001,                        # End learning rate.
```

## Training Process

### Startup

You can use Python or shell scripts for training.

```shell
# Training example
  python:
      Single-device training (Ascend): python train.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR]
      Single-device training (GPU): python train.py --device_target GPU --device_id [DEVICE_ID] --dataset_path [DATA_DIR]

  shell:
      Single-device training (Ascend): bash ./run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
      8-device parallel training (Ascend): bash ./run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]
      Single-device training (GPU): bash ./run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR]
      8-device parallel training (GPU): bash ./run_distribute_train_gpu.sh [RANK_SIZE] [DATA_DIR]
```

### Result

The CKPT file is stored in the `./checkpoint` directory, and training logs are recorded in the `log.txt` directory. An example of a training log is as follows:

```shell
epoch 1: epoch time: 1345745.321, per step time: 1075.735, avg loss: 5.192
epoch 2: epoch time: 599782.047, per step time: 479.442, avg loss: 4.452
epoch 3: epoch time: 599652.285, per step time: 479.338, avg loss: 3.966
epoch 4: epoch time: 599624.245, per step time: 479.315, avg loss: 3.782
epoch 5: epoch time: 599762.047, per step time: 479.426, avg loss: 3.661
```

## Evaluation Process

### Startup

You can use Python or shell scripts for evaluation.

```shell
# Evaluation example
  python:
      python eval.py --device_id [DEVICE_ID] --dataset_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

  shell:
      sh ./run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
```

> The CKPT file can be generated during training.

### Result

You can view the evaluation results in `eval_log.txt`.

```shell
result: {'Loss': 1.7236452958522699, 'Top_1_Acc': 0.8045072115384615, 'Top_5_Acc': 0.9503806089743589}, ckpt = ./checkpoint/model_0/Efficientnet_b3-rank0-350_1251.ckpt
```

## ONNX Inference

### Exporting ONNX

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 300 --height 300 --file_format ONNX --device_target GPU
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

| Parameter                       | Ascend                                | GPU                                |
| -------------------------- | ------------------------------------- |------------------------------------- |
| Model name                   | EfficientNet                          |EfficientNet                          |
| Model version                   | B3                           |B3                           |
| Operating environment                   | HUAWEI CLOUD Modelarts                     |Tesla V100; EulerOS 2.8                |
| Dataset                     | imagenet                              |imagenet                              |
| Training parameters                   | src/config.py                         |src/config.py                         |
| Optimizer                     | RMSProp                              |RMSProp                              |
| Loss function                   | CrossEntropySmooth         |CrossEntropySmooth         |
| Final loss                   | 1.72                                  |1.72                                  |
| Accuracy (8-device)                | Top1[80.5%], Top5[95%]               |Top1[80.5%], Top5[95%]               |

# Random Seed Description

We set random seeds in the `dataset.py` and `train.py` scripts.

# ModelZoo

For details, please visit the [official website](https://gitee.com/mindspore/models).
