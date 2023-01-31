# Contents

<!-- TOC -->

- [Contents](#contents)
- [OCRNet Description](#ocrnet-description)
    - [Overview](#overview)
    - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Scripts and Sample Code](#scripts-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Ascend Runs](#ascend-runs)
            - [GPU Runs](#gpu-runs)
            - [Inference While Training](#inference-while-training)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Ascend](#ascend)
            - [GPU](#gpu)
        - [Result](#result-1)
    - [Reasoning](#reasoning)
        - [Export MindIR](#export-mindir)
        - [Perform Inference](#perform-inference)
        - [Result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [Performance of OCRNet on Cityscapes](#performance-of-ocrnet-on-cityscapes)
- [Random Situation Description](#random-situation-description)
- [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

# OCRNet Description

## Overview

OCRNet is a semantic segmentation network proposed by Microsoft Research Institute and Chinese Academy of Sciences.
OCRNet uses a new object contextual information that explicitly enhances the contribution of pixels
from the same class of objects when constructing the contextual information,
and was presented in both July 2019 and January 2020 Cityscapes leaderboard submissions
Achieved the first place in the semantic segmentation task.
The related work "Object-Contextual Representations for Semantic Segmentation" has been included in ECCV 2020.

## Paper

[Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/pdf/1909.11065)

# Model Architecture

The overall architecture of OCRNet is as follows:

![OCRNet](figures/OCRNet.png)

# Dataset

1. Dataset used [Cityscapes](https://www.cityscapes-dataset.com/)

The Cityscapes dataset contains 5000 high-quality pixel-level finely annotated images of urban and urban scenes.
The images are divided into three groups according to the 2975/500/1525 split,
which are used for training, validation and testing respectively.
There are a total of 30 classes of entities in the dataset, 19 of which are used for validation.

2. Structural schema after dataset download

```text
Cityscapes
├─ leftImg8bit
│  ├─ train
│  │  └─ [city_folders]
│  └─ val
│     └─ [city_folders]
├─ gtFine
│  ├─ train
│  │  └─ [city_folders]
│  └─ val
│     └─ [city_folders]
├─ train.lst
└─ val.lst
```

# Environment Requirements

- Hardware（Ascend or GPU）
    - Prepare the hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Quick Start

Install required libraries from **requirements.txt**.

Training starts with trained backbone [hrnet_w48](https://download.mindspore.cn/model_zoo/official/cv/ocrnet/ImageNet-torch-pretrained-hrnet48.ckpt).
This backbone converted from [torch HRNet-W48-C](https://github.com/HRNet/HRNet-Image-Classification) by script.

```bash
python convert_from_torch.py --torch_path [INPUT_TORCH_MODEL] --mindspore_path [OUT_MINDSPORE_MODEL]
```

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Ascend runs

```bash
# Distribute training
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# Distribute training, resume training from the specified period
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# Standalone training
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK]

# Standalone training，resume training from the specified period
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK]

# Evaluation
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

- GPU runs

```bash
# Distribute training
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]

# Distribute training, resume training from the specified period
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]

# Standalone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]

# Standalone training, resume training from the specified period
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]

# Evaluation
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

If you want to train the model on ModelArts, you can refer to the
[Official Guidance Document](https://support.huaweicloud.com/modelarts/)
of ModelArts to start model training and inference. The specific operations are as follows:

```text
# train model
1. Create a job
2. Select the data set storage location
3. Select the output storage location
2. In the model parameter list, add parameters as follows:
     data_url [autocomplete]
     train_url [autocomplete]
     checkpoint_url [CHECKPOINT_PATH_OBS]
     modelarts True
     device_target Ascend
     run_distribute [True/False]
     eval_callback [True/False]
     # For details of other optional parameters, please refer to the train.py script
3. Select the appropriate number of processors
4. Start running

# evaluate the model
1. Create a job
2. Select the data set storage location
3. Select the output storage location
2. In the model parameter list, add parameters as follows:
     data_url [autocomplete]
     train_url [autocomplete]
     checkpoint_url [CHECKPOINT_PATH_OBS]
     modelarts True
     device_target Ascend
3. Select a single processor
4. Start running
```

# Script Description

## Scripts and Sample Code

```text
└─ OCRNet
   ├─ ascend310_infer                       # 310 Inference related scripts
   │  ├─ inc
   │  │  └─ utils.py
   │  └─ src
   │     ├─ build.sh
   │     ├─ CMakeLists.txt
   │     ├─ main.cc
   │     └─ utils.cc
   ├─ scripts
   │  ├─ ascend310_inference.sh             # Start Ascend310 inference (single card)
   │  ├─ run_standalone_train.sh            # Start Ascend standalone training (single card)
   │  ├─ run_standalone_train_gpu.sh        # Start GPU standalone training (single card)
   │  ├─ run_distribute_train.sh            # Start Ascend distribute training (8 cards)
   │  ├─ run_distribute_train_gpu.sh        # Start GPU distribute training (8 cards)
   │  └─ run_eval.sh                        # Start Ascend standalone evaluation (single card)
   │  └─ run_eval_gpu.sh                    # Start GPU standalone evaluation (single card)
   ├─ src
   │  ├─ model_utils
   │  │  └─ moxing_adapter.py               # ModelArts device configuration
   │  ├─ config.py                          # Parameter configuration
   │  ├─ basedataset.py                     # Dataset generator base class
   │  ├─ cityscapes.py                      # Cityscapes dataset generator
   │  ├─ loss.py                            # Loss function
   │  ├─ callback.py                        # Inference callback function during training
   │  ├─ seg_hrnet_ocr.py                   # OCRNet network structure
   │  ├─ seg_hrnet.py                       # HRNet network structure
   │  └─ utils.py                           # Parameter initialization function
   ├─ convert_from_torch.py                 # Backbone converter
   ├─ export.py                             # 310 reasoning, export mindir
   ├─ preprocess.py                         # 310 inference, data preprocessing
   ├─ postprocess.py                        # 310 Inference, calculate mIoU
   ├─ requirements.txt                      # Required libraries
   ├─ train.py                              # Train the model
   └─ eval.py                               # Eval the model
```

## Script Parameters

Both training parameters and evaluation parameters can be configured in the configuration file.
Some parameters should be configured as args for train.py or eval.py. You can see it by running `python train.py --help`

```python
hrnetv2_w48_configuration = {
    "data_url": None,                           # Dataset OBS storage path
    "data_path": None,                          # Dataset local machine storage path
    "train_url": None,                          # Training output OBS storage path
    "train_path": None,                         # Training output local machine storage path
    "checkpoint_url": None,                     # OBS storage path of the checkpoint file
    "checkpoint_path": None,                    # Checkpoint file local machine storage path
    "run_distribute": False,                    # Whether it is distributed operation
    "device_target": "Ascend",                  # Operating platform
    "workers": 8,
    "modelarts": False,                         # Does it run on ModelArts
    "lr": 0.0013,                               # Base learning rate
    "lr_power": 4e-10,                          # Learning rate adjustment factor
    "save_checkpoint_epochs": 20,               # How often to store checkpoints
    "keep_checkpoint_max": 20,                  # Save the number of checkpoints
    "total_epoch": 1000,                        # Total training cycle
    "begin_epoch": 0,                           # Start epoch
    "end_epoch": 1000,                          # End epoch
    "batchsize": 4,                             # Input tensor batch size
    "eval_callback": False,                     # Whether to use train-time inference
    "eval_interval": 50,                        # Frequency of inference during training
    "train": {
        "train_list": "/train.lst",             # Training set file storage path list
        "image_size": [512, 1024],              # Training input image size
        "base_size": 2048,                      # Base size of training images
        "multi_scale": True,                    # Whether to randomly scale the image
        "flip": True,                           # Whether to flip the image
        "downsample_rate": 1,                   # Downsampling rate
        "scale_factor": 16,                     # Scale factor
        "shuffle": True,                        # Whether to shuffle
        "param_initializer": "TruncatedNormal", # Parameter initialization method
        "opt_momentum": 0.9,                    # Momentum of Optimizer
        "wd": 0.0005,                           # Weight decay
        "num_samples": 0                        # Number of samples
    },
    "dataset": {
        "name": "Cityscapes",                   # Dataset name
        "num_classes": 19,                      # Number of categories
        "ignore_label": 255,                    # Category label values not considered
        "mean": [0.485, 0.456, 0.406],          # Mean RGB normalization
        "std": [0.229, 0.224, 0.225],           # Std RGB normalization

    },
    "eval": {
        "eval_list": "/val.lst",                # Validation set file storage path list
        "image_size": [1024, 2048],             # Evaluate the input image size
        "base_size": 2048,                      # Evaluate the base image size
        "batch_size": 1,                        # Evaluate image base size
        "num_samples": 0,                       # Number of samples
        "flip": False,                          # Whether to flip the image
        "multi_scale": False,                   # Whether to use multi-dimensional feature maps
        "scale_list": [1]                       # Enlarge the size list
    },
    "model": {                                  # Model related parameters
        "name": "seg_hrnet_w48",                # Model name
        "extra": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {                         # Stage1 parameters
                "NUM_MODULES": 1,               # High-resolution module quantity
                "NUM_BRANCHES": 1,              # Number of branches
                "BLOCK": "BOTTLENECK",          # Residual block type
                "NUM_BLOCKS": [4],              # The number of residual blocks in each branch
                "NUM_CHANNELS": [64],           # The number of feature map channels of each branch
                "FUSE_METHOD": "SUM"            # Branch fusion
            },
            "STAGE2": {                         # stage2 parameters
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [48, 96],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {                         # stage3 parameters
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {                         # stage4 parameters
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192, 384],
                "FUSE_METHOD": "SUM"
            }
        },
        "ocr": {                                # ocr module parameters
            "mid_channels": 512,
            "key_channels": 256,
            "key_channels": 256,
            "dropout": 0.05,
            "scale": 1
        }
    },
    "loss": {
        "loss_scale": 10,                       # Loss scale
        "use_weights": True,
        "balance_weights": [0.4, 1]
    },
}
```

## Training Process

### Usage

#### Ascend Runs

```bash
# Distribute training
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK](optional)

# Distributed training, resume training from the specified period
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK](optional)

# Standalone training
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [EVAL_CALLBACK](optional)

# Standalone training, resume training from the specified period
bash scripts/run_standalone_train.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH] [EVAL_CALLBACK](optional)
```

Distributed training requires creating an HCCL configuration file in JSON format in advance.

For specific operations, see the instructions in [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

The training results are saved in the example path, and the folder name starts with "train" or "train_parallel".
You can find the checkpoint file along with the results in the log at this path, as shown below.

If you want to change the running card number when running a single-card use case,
you can set the environment variable `export DEVICE_ID=x`.

#### GPU Runs

```bash
# Distribute training
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]

# Distribute training, resume training from the specified period
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]

# Standalone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH]

# Standalone training, resume training from the specified period
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH] [BEGIN_EPOCH]
```

#### Inference While Training

If inference during training on Ascend is required, you can pass `True` for the `EVAL_CALLBACK`
parameter when executing the shell script. The default value is `False`.

### Result

Train OCRNet with Cityscapes dataset

```text
# Distribute training logs（8p GPUs）
...
epoch: 2 step: 124, loss is 0.32486626505851746
epoch time: 74857.710 ms, per step time: 603.691 ms
epoch: 2 step: 124, loss is 0.5980132222175598
epoch: 2 step: 124, loss is 0.2903040051460266
epoch time: 74845.289 ms, per step time: 603.591 ms
epoch time: 74872.818 ms, per step time: 603.813 ms
epoch: 2 step: 124, loss is 3.1416678428649902
epoch time: 74860.086 ms, per step time: 603.710 ms
epoch: 3 step: 124, loss is 0.5978401303291321
epoch time: 75155.596 ms, per step time: 606.094 ms
epoch: 3 step: 124, loss is 0.7612668871879578
epoch time: 75150.912 ms, per step time: 606.056 ms
epoch: 3 step: 124, loss is 0.5324980020523071
epoch: 3 step: 124, loss is 0.6240115165710449
epoch: 3 step: 124, loss is 0.5367609858512878
...
```

## Evaluation Process

### Usage

#### Ascend

```bash
# Run the evaluation
bash scripts/run_eval.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

#### GPU

```bash
# Run the evaluation
bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT_PATH]
```

### Result

The evaluation results are saved in the example path in a folder named "eval". You can find the following results in the log file at this path:

```text
Total number of images:  500
=========== Validation Result ===========
===> mIoU: 0.8066736268581052
===> IoU array:
 [0.98223745 0.85388544 0.93150755 0.55247141 0.62441439 0.69092572
 0.73535771 0.81997183 0.93085611 0.64866909 0.95282953 0.83939804
 0.66192086 0.95695508 0.88400934 0.92132691 0.82947832 0.71646621
 0.79411792]
=========================================
```

## Reasoning

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

```bash
python export.py --device_id [DEVICE_ID] --checkpoint_file [CKPT_PATH] --file_name [FILE_NAME] --file_format MINDIR --device_target [TARGET_DEVICE]
```

Where **TARGET_DEVICE** GPU or Ascend

### Perform Inference

Before performing inference, the `export.py` script must be passed to this mindir file.
The following shows an example of performing inference using the mindir model.
Currently only inference with batchsize of 1 is supported for the Cityscapes dataset.

```bash
bash scripts/run_cpp_infer.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

- `MINDIR_PATH` The storage path of the mindir file
- `DATA_PATH` The storage path of the original dataset of Cityscapes
- `DEVICE_TYPE` can choose from [Ascend, GPU, CPU]
- `DEVICE_ID` card number

The script is divided into three steps:

1. `preprocess.py` preprocesses the original dataset and stores the processed dataset in binary form under the path `./preprocess_Result/`;
2. `ascend310_infer/src/main.cc` executes the inference process and stores the prediction results in binary form under the path `./result_Files/`. The inference log can be viewed in `infer.log`;
3. `postprocess.py` uses the prediction results and corresponding labels to calculate mIoU, and the calculation results can be viewed in `acc.log`.

### Result

```text
Total number of images:  500
=========== 310 Inference Result ===========
miou: 0.7880364289865892
iou array:
 [0.98327649 0.86189605 0.92990512 0.53712174 0.63041064 0.68390911
 0.71874631 0.80141863 0.92871439 0.63142162 0.94527287 0.83139662
 0.6455081  0.95468034 0.81087329 0.87612221 0.74120989 0.67898836
 0.78182036]
============================================
```

# Model Description

## Performance

### Evaluation Performance

#### Performance of OCRNet on Cityscapes

|Parameters          | Ascend 910                                    |GPU (8p)                                        |
|--------------------|-----------------------------------------------|------------------------------------------------|
|Model Version       | OCRNet                                        | OCRNet                                         |
|Resources           | Ascend 910; CPU 2.60GHz                       | 8x V100-PCIE                                   |
|Upload Date         | 2021-12-12                                    | 2022-05-20                                     |
|MindSpore Version   | 1.2                                           | 1.6.1                                          |
|Datasets            | Cityscapes                                    | Cityscapes                                     |
|Training parameters | epoch=1000, steps per epoch=248, batch_size=3 | epoch=1000, steps per epoch=124, batch_size=3  |
|Optimizer           | SGD                                           | SGD                                            |
|Loss Function       | Softmax Cross Entropy                         | Softmax Cross Entropy                          |
|Output              | mIoU                                          | mIoU                                           |
|Loss                | 0.06756218                                    | 0.137                                          |
|Speed               | 279ms/step (4 cards)                          | 606ms/step (8 cards)                           |
|Total Duration      | 19.4 Hours                                    | 21 Hours                                       |

# Random Situation Description

A random seed is used in `train.py`.

# Disclaimer

Models only provide scripts for transforming models. We don't own these models, nor are we responsible for their quality and maintenance. The transformation of these models is only used for non-commercial research and teaching purposes.

To the model owner: If you do not want to include the model in MindSpore models or want to convert it in any way, we will delete or update all public content as required. Please contact us through Gitee. Thank you very much for your understanding and contribution to this community.

# ModelZoo homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models).
