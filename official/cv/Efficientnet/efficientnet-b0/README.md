# Contents

- [EfficientNet-B0 Description](#efficientnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [EfficientNet-B0 Description](#contents)

[Paper](https://arxiv.org/abs/1905.11946): Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 2019.

# [Model architecture](#contents)

The overall network architecture of EfficientNet-B0 is show below:

[Link](https://arxiv.org/abs/1905.11946)

# [Dataset](#contents)

Dataset used:

1. [ImageNet](http://www.image-net.org/)
- Dataset size: 146G, 133W colorful images in 1000 classes
    - Train: 140G, 128W images
    - Test: 6G, 5W images
- Data format: RGB images

2. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- Dataset size: ~180MB, 6W colorful images in 10 classes
    - Train: 150MB, 5W images
    - Test: 30MB, 1W images
- Data format: RGB images（Binary Version）

Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware CPU/GPU
    - Prepare hardware environment with CPU/GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Training on GPU:

```bash
bash run_standalone_train_gpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [PRETRAINED_CKPT_PATH](optional)
```

- Evaluation on GPU:

```bash
cd scripts
bash run_eval_cpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [CHECKPOINT_PATH]
```

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
└─efficientnet
  ├─README.md
  ├─scripts
    ├─run_train_cpu.sh                    # launch training with cpu platform
    ├─run_standalone_train_gpu.sh         # launch standalone training with gpu platform
    ├─run_distribute_train_gpu.sh         # launch distributed training with gpu platform
    ├─run_eval_cpu.sh                     # launch evaluating with cpu platform
    └─run_eval_gpu.sh                     # launch evaluating with gpu platform
  ├─src
    ├─callbacks.py                        # callback for model evaluation during training
    ├─config.py                           # parameter configuration
    ├─dataset.py                          # data preprocessing
    ├─efficientnet.py                     # network definition
    ├─loss.py                             # Customized loss function
    ├─transform_utils.py                  # random augment utils
    ├─transform.py                        # random augment class
  ├─efficientnet_b0_cifar10_config.yaml   # config for EfficientNet-B0 CIFAR10 training
  ├─efficientnet_b0_imagenet_config.yaml  # config for EfficientNet-B0 ImageNet training
  ├─efficientnet_b1_imagenet_config.yaml  # config for EfficientNet-B1 ImageNet training
  ├─eval.py                               # eval net
  ├─train.py                              # train net
  ├─export.py                             # export net
  ├─mindspore_hub_conf.py                 # midnspore hub configuration file
  └─requirements.txt                      # pip requirements

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in one of configuration files:

- ```efficientnet_b0_cifar10_config.yaml```
- ```efficientnet_b0_imagenet_config.yaml```
- ```efficientnet_b1_imagenet_config.yaml```

1. ImageNet Config for GPU:

```python
'random_seed': 1,                # fix random seed
'model': 'efficientnet_b0',      # model name
'drop': 0.2,                     # dropout rate
'drop_connect': 0.2,             # drop connect rate
'opt_eps': 0.001,                # optimizer epsilon
'lr': 0.064,                     # learning rate LR
'batch_size': 128,               # batch size
'decay_epochs': 2.4,             # epoch interval to decay LR
'warmup_epochs': 5,              # epochs to warmup LR
'decay_rate': 0.97,              # LR decay rate
'weight_decay': 1e-5,            # weight decay
'epochs': 600,                   # number of epochs to train
'workers': 8,                    # number of data processing processes
'amp_level': 'O0',               # amp level
'opt': 'rmsprop',                # optimizer
'num_classes': 1000,             # number of classes
'gp': 'avg',                     # type of global pool, "avg", "max", "avgmax", "avgmaxc"
'momentum': 0.9,                 # optimizer momentum
'warmup_lr_init': 0.0001,        # init warmup LR
'smoothing': 0.1,                # label smoothing factor
'bn_tf': False,                  # use Tensorflow BatchNorm defaults
'keep_checkpoint_max': 10,       # max number ckpts to keep
'loss_scale': 1024,              # loss scale
'resume_start_epoch': 0,         # resume start epoch
```

2. CIFAR-10 Config for CPU/GPU

```python
'random_seed': 1,                # fix random seed
'model': 'efficientnet_b0',      # model name
'drop': 0.2,                     # dropout rate
'drop_connect': 0.2,             # drop connect rate
'opt_eps': 0.0001,               # optimizer epsilon
'lr': 0.0002,                    # learning rate LR
'batch_size': 32,                # batch size
'decay_epochs': 2.4,             # epoch interval to decay LR
'warmup_epochs': 5,              # epochs to warmup LR
'decay_rate': 0.97,              # LR decay rate
'weight_decay': 1e-5,            # weight decay
'epochs': 150,                   # number of epochs to train
'workers': 8,                    # number of data processing processes
'amp_level': 'O0',               # amp level
'opt': 'rmsprop',                # optimizer
'num_classes': 10,               # number of classes
'gp': 'avg',                     # type of global pool, "avg", "max", "avgmax", "avgmaxc"
'momentum': 0.9,                 # optimizer momentum
'warmup_lr_init': 0.0001,        # init warmup LR
'smoothing': 0.1,                # label smoothing factor
'bn_tf': False,                  # use Tensorflow BatchNorm defaults
'keep_checkpoint_max': 10,       # max number ckpts to keep
'loss_scale': 1024,              # loss scale
'resume_start_epoch': 0,         # resume start epoch
```

## [Training Process](#contents)

### Usage

1. GPU

```bash
    # distribute training
    bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [PRETRAINED_CKPT_PATH](optional)
    # standalone training
    bash run_standalone_train_gpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [PRETRAINED_CKPT_PATH](optional)
```

2. CPU

```bash
    bash run_train_cpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [PRETRAINED_CKPT_PATH](optional)
```

### Launch Example

```bash
# distributed training example(8p) for GPU
cd scripts
bash run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 ImageNet ./dataset efficientnet_b0

# standalone training example for GPU
cd scripts
bash run_standalone_train_gpu.sh ImageNet ./dataset efficientnet_b0

# training example for CPU
cd scripts
bash run_train_cpu.sh ImageNet ./dataset efficientnet_b0
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

1. CPU

```bash
bash run_eval_cpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [CHECKPOINT_PATH]
```

2. GPU

```bash
bash run_eval_gpu.sh [DATASET_TYPE] [DATASET_PATH] [MODEL_NAME] [CHECKPOINT_PATH]
```

#### Launch Example

```bash
# Evaluation with checkpoint for GPU
cd scripts
bash run_eval_gpu.sh ImageNet ./dataset efficientnet_b0 ./checkpoint/efficientnet_b0-600_1251.ckpt

# Evaluation with checkpoint for CPU
cd scripts
bash run_eval_cpu.sh ImageNet ./dataset efficientnet_b0 ./checkpoint/efficientnet_b0-600_1251.ckpt
```

#### Result

Evaluation result will be stored in the model_path/eval path. Under this, you can find result like the following in log.

```text
acc=76.96%(TOP1)
```

# [Model description](#contents)

## [Performance in ImageNet](#contents)

### Training Performance

| Parameters                 | efficientnet_b0           | efficientnet_b1           |
| -------------------------- | ------------------------- | ------------------------- |
| Resource                   | NV SMX2 V100-32G          | GPU(Tesla V100-PCIE 32G)；CPU：2.70GHz 52cores；RAM：1.5T |
| uploaded Date              | 10/26/2020                | 10/22/2021                |
| MindSpore Version          | 1.0.0                     | 1.6.0.20211119            |
| Dataset                    | ImageNet                  | ImageNet                  |
| Training Parameters        | src/config.py             | src/config.py             |
| Optimizer                  | rmsprop                   | rmsprop                   |
| Loss Function              | LabelSmoothingCrossEntropy| LabelSmoothingCrossEntropy|
| Loss                       | 1.8886                    | 1.8227                    |
| Accuracy                   | 76.96%(TOP1)              | 78.44%(TOP1)              |
| Total time                 | 132 h 8ps                 | 164 h 28 m  (8p: 252ms/step, 1p：239ms/step) |
| Checkpoint for Fine tuning | 64 M(.ckpt file)          | 100 M(.ckpt file)         |

### Inference Performance

| Parameters        | efficientnet_b0  | efficientnet_b1  |
| ----------------- | ---------------- | ---------------- |
| Resource          | NV SMX2 V100-32G | RTX 3090 24G     |
| uploaded Date     | 10/26/2020       | 10/22/2021       |
| MindSpore Version | 1.0.0            | 1.6.0.20211119   |
| Dataset           | ImageNet         | ImageNet         |
| batch_size        | 128              | 128              |
| outputs           | probability      | probability      |
| Accuracy          | acc=76.96%(TOP1) | acc=78.44%(TOP1) |

## [Performance in CIFAR-10](#contents)

### Training Performance

| Parameters                 | efficientnet_b0            |
| -------------------------- | -------------------------- |
| Resource                   | NV GTX 1080Ti-12G          |
| uploaded Date              | 06/28/2021                 |
| MindSpore Version          | 1.3.0                      |
| Dataset                    | CIFAR-10                   |
| Training Parameters        | src/config.py              |
| Optimizer                  | rmsprop                    |
| Loss Function              | LabelSmoothingCrossEntropy |
| Loss                       | 1.2773                     |
| Accuracy                   | 97.75%(TOP1)               |
| Total time                 | 2 h 4ps                    |
| Checkpoint for Fine tuning | 47 M(.ckpt file)           |

### Inference Performance

| Parameters        | efficientnet_b0  |
| ----------------- | ---------------- |
| Resource          | NV GTX 1080Ti-12G|
| uploaded Date     | 06/28/2021       |
| MindSpore Version | 1.3.0            |
| Dataset           | CIFAR-10         |
| batch_size        | 128              |
| outputs           | probability      |
| Accuracy          | acc=93.12%(TOP1) |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
