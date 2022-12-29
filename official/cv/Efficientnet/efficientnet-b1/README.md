# Contents

<!-- TOC -->

- [Contents](#contents)
- [EfficientNet-B1 Description](#efficientnet-b1-description)
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
- [Inference Process](#inference-process)
    - [Exporting MindIR](#exporting-mindir)
    - [Inference on Ascend 310 Processor](#inference-on-ascend-310-processor)
    - [Exporting ONNX](#exporting-onnx)
    - [ONNX Inference on the GPU](#onnx-inference-on-the-gpu)
    - [Result](#result-2)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo](#modelzoo)

<!-- /TOC -->

# EfficientNet-B1 Description

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

```text
├─ EfficientNet-B1
│   ├─ README_CN.md                     # Description of EfficientNet-B1
│   ├─ ascend310_infer                  # Script for inference on Ascend 310 AI Processor
│   │   ├─ inc
│   │   │   └─ utils.h
│   │   └─ src
│   │       ├─ build.sh
│   │       ├─ CMakeLists.txt
│   │       ├─ main.cc
│   │       └─ utils.cc
│   ├─ scripts
│   │   ├─ run_infer_310.sh             # Shell script for inference on Ascend 310 AI Processors
│   │   ├─ run_standalone_train.sh      # Shell script for single-device training
│   │   ├─ run_distribute_train.sh      # Shell script for 8-device training
│   │   └─ run_eval.sh                  # Shell script for evaluation
│   │   └─ run_infer_onnx.sh            # Shell script for ONNX inference
│   ├─ src
│   │   ├─ model_utils                  # Script for ModelArts training adaptation
│   │   │   └─ moxing_adapter.py
│   │   ├─ models                       # EfficientNet architecture
│   │   │   ├─ effnet.py
│   │   │   └─ layers.py
│   │   ├─ callback.py                  # Callback function
│   │   ├─ config.py                    # Parameter configuration
│   │   ├─ dataset.py                   # Dataset creation
│   │   ├─ loss.py                      # Loss function
│   │   └─ utils.py                     # Tool function script
│   ├─ create_imagenet2012_label.py     # Create an ImageNet2012 label.
│   ├─ eval.py                          # Evaluation script
│   ├─ infer_onnx.py                    # ONNX evaluation
│   ├─ export.py                        # Script for converting the model format
│   ├─ postprocess.py                   # Script for post-processing after inference on Ascend 310 AI Processors
│   └─ train.py                         # Training script
```

## Script Parameters

The parameters used during model training and evaluation can be set in the **config.py** file.

```text
'train_url': None,                      # Training output path (bucket)
'train_path': None,                     # Training output path
'data_url': None,                       # Training dataset path (bucket)
'data_path': None,                      # Training dataset path
'checkpoint_url': None,                 # Checkpoint path (bucket)
'checkpoint_path': None,                # Checkpoint path
'eval_data_url': None,                  # Path of the inference dataset (bucket)
'eval_data_path': None,                 # Path of the inference dataset
'eval_interval': 10,                    # Inference interval during training
'modelarts': False,                     # Specifies whether to use ModelArts.
'run_distribute': False,                # Specifies whether to perform multi-device training.
'device_target': 'Ascend',              # Training platform
'begin_epoch': 0,                       # Training start epoch
'end_epoch': 350,                       # Training end epoch
'total_epoch': 350,                     # Total number of training epochs
'dataset': 'imagenet',                  #Dataset name
'num_classes': 1000,                    # Number of classes in the dataset
'batchsize': 128                        # Batch size
'input_size': 240,                      # Input size
'lr_scheme': 'linear',                  # Learning rate decay solution
'lr': 0.15,                             # Maximum learning rate
'lr_init': 0.0001,                      # Initial learning rate
'lr_end': 5e-5   ,                      # Final learning rate
'warmup_epochs': 2,                     #Number of warm-up epochs
'use_label_smooth: True,               # Specifies whether to use label smoothing
'label_smooth_factor': 0.1,             # Label smoothing factor
'conv_init': 'TruncatedNormal',         # Initialization mode of a convolution layer
'dense_init': 'RandomNormal',           # Initialization mode of a fully-connected layer
'optimizer': 'rmsprop',                 # Optimizer
'loss_scale': 1024,                     # Loss scale
'opt_momentum': 0.9,                    # Momentum parameter
'wd': 1e-5,                             # Weight decay rate
'eps': 0.001                            # Epsilon
'save_ckpt': True,                      # Specifies whether to save the CKPT file
'save_checkpoint_epochs': 1,            # Save a CKPT file for each epoch.
'keep_checkpoint_max': 10,              # Maximum number of CKPT files that can be saved.
```

## Training Process

### Startup

You can use Python or shell scripts for training.

```bash
# Training example
  python:
      Single-device training (Ascend):
          python train.py --data_path [DATA_DIR] --train_path [TRAIN_OUTPUT_PATH] --model efficientnet-b1 --run_distribute False

  shell:
      Single-device training (Ascend): bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]
      8-device parallel training (Ascend):
          bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]
```

### Result

The CKPT file for multi-device training is stored in the `./checkpoint` directory, and the CKPT file for single-device training is stored in the specified directory. Training logs are recorded in `log`. An example of a training log is as follows:

```text
epoch: [ 1/350], epoch time: 2709470.652, steps: 625, per step time: 4335.153, avg loss: 5.401, lr:[0.050]
epoch: [ 2/350], epoch time: 236883.599, steps: 625, per step time: 379.014, avg loss: 4.142, lr:[0.100]
epoch: [ 3/350], epoch time: 236615.708, steps: 625, per step time: 378.585, avg loss: 3.724, lr:[0.100]
epoch: [ 4/350], epoch time: 236606.486, steps: 625, per step time: 378.570, avg loss: 3.133, lr:[0.099]
epoch: [ 5/350], epoch time: 236639.009, steps: 625, per step time: 378.622, avg loss: 3.225, lr:[0.099]
```

## Evaluation Process

### Startup

You can use Python or shell scripts for evaluation.

```bash
# Evaluation example
  python:
      python eval.py --data_path [DATA_DIR] --checkpoint_path [PATH_CHECKPOINT]

  shell:
      bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

> The CKPT file can be generated during training.

### Result

You can view the evaluation results in `eval_log`.

```bash
{'Loss': 1.8175019884720827, 'Top_1_Acc': 0.7914495192307693, 'Top_5_Acc': 0.9445458333333333}
```

# Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

## Exporting MindIR

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 240 --height 240 --file_format MINDIR
```

## Inference on Ascend 310 Processor

Before inference, the MindIR file must be exported through the `export.py` script. The following shows an example of using the MindIR model to perform inference.

```bash
# Ascend310 inference
bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

## Exporting ONNX

```bash
python export.py --checkpoint_path [CHECKPOINT_FILE_PATH] --file_name [OUTPUT_FILE_NAME] --width 240 --height 240 --file_format ONNX
```

## ONNX Inference on the GPU

Before inference, the ONNX file must be exported through the `export.py` script. The following shows an example of using the ONNX model to perform inference.

```bash
# ONNX inference
bash scripts/run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [DEVICE_TARGET]
```

## Result

The inference result is saved in the current path where the script is executed. You can view the accuracy computation result of Ascend 310 AI Processor in **acc.log** and the ONNX inference accuracy computation result in **infer_onnx.log**.

# Model Description

## Training Performance

| Parameter                       | Ascend                                |
| -------------------------- | ------------------------------------- |
| Model name                   | EfficientNet                          |
| Model version                   | B1                           |
| Operating environment                   | HUAWEI CLOUD Modelarts                     |
| Upload date                   | 2021-12-06                             |
| Dataset                     | imagenet                              |
| Training parameters                   | src/config.py                         |
| Optimizer                     | RMSProp                              |
| Loss function                   | CrossEntropySmooth         |
| Final loss                   | 1.82                                 |
| Accuracy (8-device)                | Top1[79.1%], Top5[94.4%]               |
| Total training duration (8-device)            | 25.1h                                    |
| Total evaluation duration                 | 84s                                    |
| Parameters (M)                | 30M                                   |
| Script                      | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1)|

# Random Seed Description

We also set a random seed in the `train.py` script.

# ModelZoo

For details, please visit the [official website](https://gitee.com/mindspore/models).
