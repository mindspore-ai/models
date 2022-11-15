# Contents

<!-- TOC -->

- [Contents](#contents)
- [Single-path-nas description](#single-path-nas-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Scripts Description](#scripts-description)
    - [Scripts and sample code](#scripts-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Standalone training](#standalone-training)
        - [Distributed training](#distributed-training)
    - [Evaluation process](#evaluation-process)
        - [Evaluate](#evaluate)
    - [Export process](#export-process)
        - [Export](#export)
    - [Inference process](#inference-process)
        - [Inference](#inference)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training performance](#training-performance)
            - [Single-Path-NAS on ImageNet-1k](#single-path-nas-on-imagenet-1k)
        - [Inference performance](#inference-performance)
            - [Single-Path-NAS on ImageNet-1k](#single-path-nas-on-imagenet-1k-1)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# Single-path-nas description

The author of single-path-nas used a large 7x7 convolution to represent the three convolutions of 3x3, 5x5, and 7x7.
The weights of the smaller convolution layers are shared with the larger ones. The largest kernel becomes a "superkernel".
This way when training the model we do not need to choose between different paths, instead we pass the data through a
layer with shared weights among different sub-kernels. The search space is a block-based straight structure.
Like in the ProxylessNAS and the FBNet, the Inverted Bottleneck block is used as the cell,
and the number of layers is 22 as in the MobileNetV2. Each layer has only two searchable hyper-parameters:
expansion rate and kernel size. The others hyper-parameters are fixed. For example, the filter number of each layer in
the 22nd layer is fixed. Like FBNet, it is slightly changed from MobileNetV2. The used kernel sizes in the paper are
only 3x3 and 5x5 like in the FBNet and ProxylessNAS, and 7x7 kernels are not used. The expansion ratio in the paper has
only two choices of 3 and 6. Both the kernel size and expansion ratio have only 2 choices.
The Single-Path-NAS paper uses the techniques described in Lightnn's paper.
In particular, it describes using a continuous smooth function to represent the discrete choice,
and the threshold is a group Lasso term. This paper uses the same technique as ProxylessNAS to express skip connection,
which is represented by a zero layer.
Paper: https://zhuanlan.zhihu.com/p/63605721

# Dataset

Dataset used：[ImageNet2012](http://www.image-net.org/)

- Dataset size：a total of 1000 categories, 224\*224 color images
    - Training set: 1,281,167 images in total
    - Test set: 50,000 images in total
- Data format：JPEG
    - Note: The data is processed in dataset.py.
- Download the dataset and prepare the directories structure as follows：

```text
└─dataset
    ├─train                 # Training dataset
    └─val                   # Evaluation dataset
```

# Features

## Mixed Precision

The [mixed-precision](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)
training method uses single-precision and half-precision data to improve the training speed of
deep learning neural networks, while maintaining the network accuracy that can be achieved by single-precision training.
Mixed-precision training increases computing speed and reduces memory usage, while supporting training larger models or
allowing larger batches for training on specific hardware.

# Environment Requirements

- Hardware（Ascend, GPU）
    - Prepare hardware environment with Ascend processor or CUDA based GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the links below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Quick Start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- For the Ascend hardware

  ```bash
  # Run the training example
  python train.py --device_id=0 --device_target="Ascend" --data_path=/imagenet/train > train.log 2>&1 &

  # Run the standalone training example
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATA_PATH]

  # Run a distributed training example
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DEVICE_NUM] [DATA_PATH]

  # Run evaluation example
  python eval.py --checkpoint_path=./ckpt_0 --val_data_path=/imagenet/val --device_id=0 --device_target="Ascend"> ./eval.log 2>&1 &

  # Run the inference example
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  For distributed training, you need to create an **hccl** configuration file in JSON format in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- For the GPU hardware

  ```bash
  # Run the training example
  python train.py --device_id=0 --device_target="GPU" --data_path=/imagenet/train > train.log 2>&1 &

  # Run the standalone training example
  bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] > train.log 2>&1 &

  # Run a distributed training example
  bash ./scripts/run_distributed_train_gpu.sh [CUDA_VISIBLE_DEVICES] [DEVICE_NUM] [DATA_PATH]

  # Run evaluation example
  python eval.py --device_target="GPU" --device_id=0 --val_data_path="/path/to/imagenet/val/" --checkpoint_path ./ckpt_0 > ./eval.log 2>&1 &
  ```

# Scripts Description

## Scripts and sample code

```text
├── model_zoo
  ├── scripts
  │   ├──run_distribute_train_ascend.sh              // Shell script for running the Ascend distributed training
  │   ├──run_distribute_train_gpu.sh          // Shell script for running the GPU distributed training
  │   ├──run_standalone_train_ascend.sh              // Shell script for running the Ascend standalone training
  │   ├──run_standalone_train_gpu.sh          // Shell script for running the GPU standalone training
  │   ├──run_eval_ascend.sh                          // Shell script for running the Ascend evaluation
  │   ├──run_eval_gpu.sh                      // Shell script for running the GPU evaluation
  │   ├──run_infer_310.sh                     // Shell script for running the Ascend 310 inference
  ├── src
  │   ├──lr_scheduler
  │   │   ├──__init__.py
  │   │   ├──linear_warmup.py                 // Definitions for the warm-up functionality
  │   │   ├──warmup_cosine_annealing_lr.py    // Definitions for the cosine annealing learning rate schedule
  │   │   ├──warmup_step_lr.py                // Definitions for the exponential learning rate schedule
  │   ├──__init__.py
  │   ├──config.py                            // Parameters configuration
  │   ├──CrossEntropySmooth.py                // Definitions for the cross entropy loss function
  │   ├──dataset.py                           // Functions for creating a dataset
  │   ├──spnasnet.py                          // Single-Path-NAS architecture.
  │   ├──utils.py                             // Auxiliary functions
  ├── create_imagenet2012_label.py            // Creating ImageNet labels
  ├── eval.py                                 // Evaluate the trained model
  ├── export.py                               // Export model to other formats
  ├── postprocess.py                          // Postprocess for the Ascend 310 inference.
  ├── README.md                               // Single-Path-NAS related instruction in English
  ├── README_CN.md                            // Single-Path-NAS related instruction in Chinese
  ├── train.py                                // Train the model.
```

## Script parameters

Training parameters and evaluation parameters can be configured in a `config.py` file.

- Parameters of a Single-Path-NAS model for the ImageNet-1k dataset.

  ```python
  'name':'imagenet'                        # dataset
  'pre_trained':'False'                    # Whether to start using a pre-trained model
  'num_classes':1000                       # Number of classes in a dataset
  'lr_init':0.26                           # Initial learning rate, set to 0.26 for single-card training, and 1.5 for eight-card parallel training.
  'batch_size':128                         # training batch size
  'epoch_size':180                         # Number of epochs
  'momentum':0.9                           # Momentum
  'weight_decay':1e-5                      # Weight decay value
  'image_height':224                       # Height of the model input image
  'image_width':224                        # Width of the model input image
  'keep_checkpoint_max':40                 # Number of checkpoints to keep
  'checkpoint_path':None                   # The absolute path to the checkpoint file or a directory, where the checkpoints are saved

  'lr_scheduler': 'cosine_annealing'       # Learning rate scheduler ['cosine_annealing', 'exponential']
  'lr_epochs': [30, 60, 90]                # Key points for the exponential schedular
  'lr_gamma': 0.3                          # Learning rate decay for the exponential scheduler
  'eta_min': 0.0                           # Minimal learning rate
  'T_max': 180                             # Number of epochs for the cosine
  'warmup_epochs': 0                       # Number of warm-up epochs
  'is_dynamic_loss_scale': 1               # Use dynamic loss scale manager (scale manager is not used for GPU)
  'loss_scale': 1024                       # Loss scale value
  'label_smooth_factor': 0.1               # Factor for labels smoothing
  'use_label_smooth': True                 # Use label smoothing
  ```

For more configuration details, please refer to the script `config.py`.

## Training process

### Standalone training

- Using an Ascend processor environment

  ```bash
  python train.py --device_id=0 --device_target="Ascend" --data_path=/imagenet/train > train.log 2>&1 &
  ```

  The above python command will run in the background, and the result can be viewed through the generated train.log file.

- Using an GPU environment

  ```bash
  python train.py --device_id=0 --device_target="GPU" --data_path=/imagenet/train > train.log 2>&1 &
  ```

  The above python command will run in the background, and the result can be viewed through the generated train.log file.

### Distributed training

- Using an Ascend processor environment

  ```bash
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DEVICE_NUM] [DATA_PATH]
  ```

  The above shell script will run distributed training in the background.

- Using a GPU environment

  ```bash
  bash ./scripts/run_distributed_train_gpu.sh [CUDA_VISIBLE_DEVICES] [DEVICE_NUM] [DATA_PATH]
  ```

> TRAIN_PATH - Path to the directory with the training subset of the dataset.

The above shell scripts will run the distributed training in the background.
Also `train_parallel` folder will be created where the copy of the code,
the training log files and the checkpoints will be stored.

## Evaluation process

### Evaluate

- Evaluate the model on the ImageNet-1k dataset using the Ascend environment

  “./ckpt_0” is a directory, where the trained model is saved in the .ckpt format.

  ```bash
  python eval.py --checkpoint_path=./ckpt_0 --device_id=0 --device_target="Ascend" --val_data_path/imagenet/val > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [DATA_PATH] [CKPT_FILE/CKPT_DIR]
  ```

- Evaluate the model on the ImageNet-1k dataset using the GPU environment

  “./ckpt_0” is a directory, where the trained model is saved in the .ckpt format.

  ```bash
  python eval.py --checkpoint_path=./ckpt_0 --device_id=0 --device_target="GPU" --val_data_path/imagenet/val > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [CKPT_FILE/CKPT_DIR]
  ```

> CKPT_FILE_OR_DIR - Path to the trained model checkpoint or to the directory, containing checkpoints.
>
> VALIDATION_DATASET - (optional) Path to the validation subset of the dataset.

## Export process

### Export

  ```shell
  python export.py --ckpt_file [CKPT_FILE] --device_target [DEVICE_TARGET]
  ```

> DEVICE_TARGET: Ascend or GPU

## Inference process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Inference

Before inference, we need to export the model first.
MINDIR can be exported in any environment, and the AIR model can only be exported in the Ascend 910 environment.
The following shows an example of using the MINDIR model to run the inference.

- Use ImageNet-1k dataset for inference on the Ascend 310

  The results of the inference are stored in the scripts directory,
  and results similar to the following can be found in the acc.log log file.

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.74214, top5 accuracy: 0.91652.
  ```

# Model description

## Performance

### Training performance

#### Single-Path-NAS on ImageNet-1k

| Parameter                  | Ascend                                                                                  | GPU                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Model                      | single-path-nas                                                                         | single-path-nas                                                                         |
| Resource                   | Ascend 910                                                                              | V100 GPU, Intel Xeon Gold 6226R CPU @ 2.90GHz                                           |
| Upload date                | 2021-06-27                                                                              | -                                                                                       |
| MindSpore version          | 1.2.0                                                                                   | 1.5.0                                                                                   |
| Dataset                    | ImageNet-1k Train, 1,281,167 images in total                                            | ImageNet-1k Train, 1,281,167 images in total                                            |
| Training parameters        | epoch=180, batch_size=128, lr_init=0.26 (0.26 for a single card, 1.5 for eight cards)   | epoch=180, batch_size=128, lr_init=0.26 (0.26 for a single card, 1.5 for eight cards)   |
| Optimizer                  | Momentum                                                                                | Momentum                                                                                |
| Loss function              | Softmax cross entropy                                                                   | Softmax cross entropy                                                                   |
| Output                     | Probability                                                                             | Probability                                                                             |
| Classification accuracy    | Eight cards: top1:74.21%, top5:91.712%                                                  | Single card: top1=73.9%, top5=91.62% ; Eight cards: top1=74.01%, top5=91.66%            |
| Speed                      | Single card: milliseconds/step; eight cards: 87.173 milliseconds/step                   | Single card: 221 ms/step; Eight cards: 263 ms/step                                      |

### Inference performance

#### Single-Path-NAS on ImageNet-1k

| Parameter                  | Ascend                                        | GPU (8 card)                                | GPU (1 card)                               |
| -------------------------- | --------------------------------------------- | ------------------------------------------- | ------------------------------------------ |
| Model                      | single-path-nas                               | single-path-nas                             | single-path-nas                            |
| Resource                   | Ascend 310                                    | V100 GPU                                    | V100 GPU                                   |
| Upload date                | 2021-06-27                                    | -                                           | -                                          |
| MindSpore version          | 1.2.0                                         | 1.5.0                                       | 1.5.0                                      |
| Dataset                    | ImageNet-1k Val, a total of 50,000 images     | ImageNet-1k Val, a total of 50,000 images   | ImageNet-1k Val, a total of 50,000 images  |
| Classification accuracy    | top1: 74.214%, top5: 91.652%                  | top1: 74.01%, top5: 91.66%                  | top1: 73.9%, top5: 91.62%                  |
| Speed                      | Average time 7.67324 ms of infer_count 50000  | 1285 images/second                          | 1285 images/second                         |

# ModelZoo homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models)
