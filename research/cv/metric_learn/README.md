# Contents

<!-- TOC -->

- [Contents](#contents)
- [Metric_Learn Description](#metric_learn-description)
    - [Description](#description)
    - [Paper](#paper)
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
        - [Usage](#usage)
            - [Running on Ascend](#running-on-ascend)
            - [Running on GPU](#running-on-gpu)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [ResNet50-Triplet on SOP](#resnet50-triplet-on-sop)
            - [ResNet50-Quadruplet on SOP](#resnet50-quadruplet-on-sop)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# Metric_Learn description

## Description

Metric learning is a feature space mapping method, that is, for a given data set, a metric ability can be learned, so
that in the feature space, samples of the same category have a smaller feature distance, and samples of different
categories have a larger feature distance. The characteristic distance. In deep learning, the basic metric learning
methods all use pairs and groups of samples for loss calculation. This type of method is called pair-based deep metric
learning. For example, in the process of training a model, we randomly select two samples, extract features, and
calculate the distance between features. If the two samples belong to the same category, then we hope that the distance
between them should be as small as possible; if the two samples belong to different categories, then we hope that the
distance between them should be as large as possible. According to this principle, many different types of pair-based
losses are derived. These losses are used to calculate the distance between sample pairs, and various optimization
methods are used to update the model according to the generated loss. The metric learning method based on deep neural
network has improved a lot of performance on many visual tasks, such as: face recognition, face verification, pedestrian
re-recognition and image retrieval, etc.

The following is an example of MindSpore using Triplet loss and Quadruptlet loss to tune ResNet50 in the SOP data set.
For Triplet loss, please refer to [Paper 1](https://arxiv.org/abs/1503.03832). Quadruptlet loss is a variant of Triplet
loss, which can be referred to [Paper 2](https://arxiv.org/abs/1704.01719).

In order to train the metric learning model, we need a neural network model as a skeleton model (ResNet50) and the
metric learning cost function for the optimization. The Residual Neural Network (ResNet) was proposed by five Chinese
including He Kaiming from Microsoft Research, and the effect is very significant. The entire network only needs to learn
the difference between input and output, which simplifies the learning objectives and difficulty. The structure of
ResNet greatly improves the speed of neural network training and greatly improves the accuracy of the model. Because of
this, ResNet is very popular and is often used as a backbone network in various fields. Here, the ResNet-50 structure is
selected as the backbone network for metric learning.

We first load the ResNet-50-ImageNet
[model weights](https://www.mindspore.cn/resources/hub/details/en?MindSpore/ascend/1.3/resnet50_v1.3_imagenet2012) as a pre-trained model, then modify the classification layer to use
the softmax function to fine-tune the model on the SOP dataset, and finally utilize the metric learning loss
(eg: triplet, quadruplet) to further finetune the model.

The following is to pre-train a pretrain model on the SOP data
set, and then use the triplet and quadruplet cost functions to fine-tune the pretrain model obtained from softmax. Use
the 8-card Ascend 910 to train the network model. It only takes 30 cycles to use the SOP data. In the 5184 categories of
the collection, the accuracy of TOP1 reached 73.9% and 74.3%.

## Paper

1. [Paper 1](https://arxiv.org/abs/1503.03832): CVPR2015 F Schroff, Kalenichenko D,Philbin J."FaceNet: A Unified
   Embedding for Face Recognition and Clustering"

2. [Paper 2](https://arxiv.org/abs/1704.01719): CVPR2017 Chen W, Chen X, Zhang J."Beyond triplet loss: A deep quadruplet
   network for person re-identification"

3. [Paper 3](https://arxiv.org/abs/1909.03909): Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, Tao Mei:
   "Deep Metric Learning with Density Adaptivity"

# Model Architecture

Please refer to this paper for the overall network architecture of ResNet:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

# Dataset

Dataset used: Stanford Online Products.

[Homepage](https://cvgl.stanford.edu/projects/lifted_struct/).

[Link to download](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip).

Paper: [Deep Metric Learning with Density Adaptivity](https://arxiv.org/abs/1909.03909)

The Stanford Online Products (SOP) data set contains a total of 120053 product images and 22634 categories. We divide it
into three data sets and use half of the data sets for experiments.

```bash
# Training dataset
cd Stanford_Online_Products && sed '1d' Ebay_train.txt | awk -F' ' '{print $4" "$2}' > train.txt
cd Stanford_Online_Products && sed '1d' Ebay_test.txt | awk -F' ' '{print $4" "$2}' > test.txt
cd Stanford_Online_Products && head -n 29437 train.txt > train_half.txt
cd Stanford_Online_Products && head -n 30003 test.txt > test_half.txt
cd Stanford_Online_Products && head -n 1012 train.txt > train_tiny.txt
cd Stanford_Online_Products && head -n 1048 test.txt > test_tiny.txt
```

- Complete data set size: 22634 categories, 120053 images in total
    - Training set: 59551 images, 11318 categories
    - Test set: 60502 images, 11316 categories

- Half of the data set size: 10368 categories, 59440 images in total
    - Training set: 29437 images, 5184 categories
    - Test set: 30003 images, 5184 categories

- Small data set size: 320 classes, 2060 images in total
    - Training set: 1012 images, 160 categories
    - Test set: 1048 images, 160 categories

- Download the data set. The directory structure is as follows:

```text
├─Stanford_Online_Products
```

# Features

## Mixed precision

The [mixed precision](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html) training
method uses single-precision and half-precision data to improve the training speed of deep learning neural networks,
while maintaining the network accuracy that can be achieved by single-precision training. Mixed-precision training
improves computing speed and reduces memory usage, while supporting training larger models or achieving larger batches
of training on specific hardware. Taking the FP16 operator as an example, if the input data type is FP32, the MindSpore
background will automatically reduce the accuracy to process the data. Users can open the INFO log and search for "
reduce precision" to view the operators with reduced precision.

# Environment Requirements

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# Quick Start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Running on Ascend

```bash
# Distributed training
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] [LOSS_NAME]

# Standalone training
bash run_standalone_train.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID] [LOSS_NAME]

# Run evaluation example
bash run_eval.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID]
```

- Running on GPU

```bash
# Distributed training
bash run_distribute_train_gpu.sh [DATASET_PATH] [LOSS_NAME] [CHECKPOINT_PATH]

# Standalone training
bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID] [LOSS_NAME] [CHECKPOINT_PATH]

# Run evaluation example
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID] [LOSS_NAME](optional)
```

# Script Description

## Script and Sample Code

```text
.
└──metric_learn
  ├── README.md
  ├── README_CN.md
  ├── ascend310_infer
  ├── scripts
    ├── run_distribute_train.sh      # Start Ascend distributed training (8 cards)
    ├── run_distribute_train_gpu.sh  # Start GPU distributed training (8 cards)
    ├── run_standalone_train.sh      # Start Ascend standalone training (single card)
    ├── run_standalone_train_gpu.sh  # Start GPU standalone training (single card)
    ├── run_infer_310.sh             # Start Ascend 310 inference shell script
    ├── run_eval.sh                  # Start Ascend evaluation
    └── run_eval_gpu.sh              # Start GPU evaluation
  ├── src
    ├── config.py                    # Parameter configuration
    ├── dataset.py                   # Data preprocessing
    ├── loss.py                      # Definition of measurement loss
    ├── lr_generator.py              # Generate the learning rate of each step
    ├── resnet.py                    # Backbone network ResNet50 definition code
    └── utility.py                   # Dataset reading
  ├── eval.py                        # Evaluation code
  ├── export.py                      # Model conversion
  ├── postprocess.py                 # 310 inference post-processing script
  ├── preprocess.py                  # 310 inference Pre-processing
  └── train.py                       # Training code
```

## Script Parameters

Configure the training parameters in config.py.

- Configure ResNet50, Softmax's pre-training parameters on the SOP dataset.

```python
"class_num":5184,                # Number of data set classes
"batch_size":80,                 # Enter the batch size of the tensor
"loss_scale":1024,               # Loss level
"momentum":0.9,                  # Momentum
"weight_decay":1e-4,             # Weight decay
"epoch_size":30,                 # This value is only applicable to training; it is fixed to 1 when applied to inference
"pretrain_epoch_size":0,         # The period size of the model that has been trained before loading the pre-training checkpoint; the actual training period size Equal to epoch_size minus pretrain_epoch_size
"save_checkpoint":True,          # Whether to save checkpoints
"save_checkpoint_epochs":10,     # Period interval between two checkpoints; by default, the last checkpoint will be saved after the last step is completed
"keep_checkpoint_max":1,         # Only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path":"./",     # Checkpoint save path
"warmup_epochs":0,               # Warm-up cycle number
"lr_decay_mode":"steps”          # Decay mode can be step, strategy and default
"lr_init":0.01,                  # Initial learning rate
"lr_end":0.0001,                 # Final learning rate
"lr_max":0.3,                    # Maximum learning rate
```

- Configure the fine-tuning parameters of ResNet50, Triplet loss on the SOP dataset

```python
"class_num":5184,                # Number of data set classes
"batch_size":60,                 # Enter the batch size of the tensor
"loss_scale":1024,               # Loss level
"momentum":0.9,                  # Momentum
"weight_decay":1e-4,             # Weight decay
"epoch_size":30,                 # This value is only applicable to training; it is fixed to 1 when applied to inference
"pretrain_epoch_size":0,         # The period size of the model that has been trained before loading the pre-training checkpoint; the actual training period size Equal to epoch_size minus pretrain_epoch_size
"save_checkpoint":True,          # Whether to save checkpoints
"save_checkpoint_epochs":10,     # Period interval between two checkpoints; by default, the last checkpoint will be saved after the last step is completed
"keep_checkpoint_max":1,         # Only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path":"./",     # Checkpoint save path
"warmup_epochs":0,               # Warm-up cycle number
"lr_decay_mode":"const”          # Decay mode can be step, strategy and default
"lr_init":0.01,                  # Initial learning rate
"lr_end":0.0001,                 # Final learning rate
"lr_max":0.0001                  # Maximum learning rate
```

- Configure ResNet50, Quadruplet loss fine-tuning parameters on the SOP dataset

```python
"class_num":5184,                # Number of data set classes
"batch_size":60,                 # Enter the batch size of the tensor
"loss_scale":1024,               # Loss level
"momentum":0.9,                  # Momentum
"weight_decay":1e-4,             # Weight decay
"epoch_size":30,                 # This value is only applicable to training; it is fixed to 1 when applied to inference
"pretrain_epoch_size":0,         # The period size of the model that has been trained before loading the pre-training checkpoint; the actual training period size Equal to epoch_size minus pretrain_epoch_size
"save_checkpoint":True,          # Whether to save checkpoints
"save_checkpoint_epochs":10,     # Period interval between two checkpoints; by default, the last checkpoint will be saved after the last step is completed
"keep_checkpoint_max":1,         # Only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path":"./",     # Checkpoint save path
"warmup_epochs":0,               # Warm-up cycle number
"lr_decay_mode":"const”          # Decay mode can be step, strategy and default
"lr_init":0.01,                  # Initial learning rate
"lr_end":0.0001,                 # Final learning rate
"lr_max":0.0001,                 # Maximum learning rate
```

## Training Process

### Usage

#### Running on Ascend

```bash
# Distributed training
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH] [LOSS_NAME]

# Standalone training
bash run_standalone_train.sh [DATASET_PATH] [CKPT_PATH] [DEVICE_ID] [LOSS_NAME]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance. Please follow the
instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

#### Running on GPU

```bash
# Distributed training
bash run_distribute_train_gpu.sh [DATASET_PATH] [LOSS_NAME] [CHECKPOINT_PATH]

# Standalone training
bash run_standalone_train_gpu.sh [DATASET_PATH] [DEVICE_ID] [LOSS_NAME] [CHECKPOINT_PATH]
```

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under
this, you can find checkpoint file together with result like the following in log. If you want to change device_id for
standalone training, you can set environment variable export DEVICE_ID=x or set device_id=x in context.

### Result

- Pre-training ResNet50 on SOP dataset using softmax

```text
# Distributed training result (8P)
epoch: 1 step: 46, loss is 8.5783054
epoch: 2 step: 46, loss is 8.0682616
epoch: 3 step: 46, loss is 7.8836588
epoch: 4 step: 46, loss is 7.80090446
epoch: 5 step: 46, loss is 7.80853784
...
```

- Fine-tuning ResNet50 on SOP dataset using Tripletloss

```text
# Distributed training result (8P)
epoch: 1 step: 62, loss is 0.357934
epoch: 2 step: 62, loss is 0.2891967
epoch: 3 step: 62, loss is 0.2131956
epoch: 4 step: 62, loss is 0.2302577
epoch: 5 step: 62, loss is 0.197817
...
```

- Fine-tuning ResNet50 on SOP dataset using Quadruptletloss

```text
# Distributed training result (8P)
epoch:1 step:62, loss is 1.7601055
epoch:2 step:62, loss is 1.6955021
epoch:3 step:62, loss is 1.5707983
epoch:4 step:62, loss is 1.462166
epoch: 5 step:62, loss is 1.393667
...
```

## Evaluation Process

### Usage

#### Running on Ascend

```bash
# Evaluation
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

```bash
# Evaluation example
bash run_eval.sh ~/Stanford_Online_Products ~/ResNet50.ckpt
```

#### Running on GPU

```bash
# Evaluation
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID] [LOSS_NAME](optional)
```

```bash
# Evaluation example
bash run_eval_gpu.sh ~/Stanford_Online_Products ~/ResNet50.ckpt 0 quadruplet
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like
the following in log.

#### Ascend

- Evaluate the results of ResNet50-triplet using the SOP dataset

```text
result: {'acc': 0.739} ckpt=~/ResNet50_triplet.ckpt
```

- Evaluate the results of ResNet50-quadrupletloss using the SOP dataset

```text
result: {'acc': 0.743} ckpt=~/ResNet50_quadruplet.ckpt
```

#### GPU

- Evaluate the results of ResNet50-triplet using the SOP dataset

```text
result: {'acc': 0.637} ckpt=~/ResNet50_triplet.ckpt
```

- Evaluate the results of ResNet50-quadrupletloss using the SOP dataset

```text
result: {'acc': 0.706} ckpt=~/ResNet50_quadruplet.ckpt
```

# Inference Process

## Export MindIR

Modify the export file ckpt_file and run.

```bash
python export.py --ckpt_file [CKPT_PATH]
```

## Infer on Ascend310

Before performing inference, the mindir file must be export.py exported through a script. The following shows an example
of using the mindir model to perform inference.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir file path
- `DATASET_PATH` SOP dataset path
- `DEVICE_ID` Optional, the default value is 0.

## Result

The inference result is saved in the current path of script execution, and you can see the following accuracy
calculation results in acc.log.

```text
'eval_recall:': 0.743
```

# Model Description

## Performance

### Evaluation Performance

#### ResNet50-Triplet on SOP

| Parameters                 | Ascend 910                                     | NVIDIA GeForce RTX 3090 (8 pcs)                      | NVIDIA GeForce RTX 3090 (Single)                     |
| -------------------------- |------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model version              | ResNet50-Triplet                               | ResNet50-Triplet                                     | ResNet50-Triplet                                     |
| Resource                   | Ascend 910; CPU: 2.60GHz, 192 cores; RAM: 755G | GeForce RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB | GeForce RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB |
| Uploaded Date              | 2021-03-25                                     | 2022-06-27                                           | 2022-06-27                                           |
| MindSpore Version          | 1.1.1-alpha                                    | 1.7.0                                                | 1.7.0                                                |
| Dataset                    | Stanford_Online_Products                       | Stanford_Online_Products                             | Stanford_Online_Products                             |
| Training Parameters        | epoch=30, steps per epoch=62, batch_size = 60  | epoch=4, batch_size=9                                | epoch=4, batch_size=9                                |
| Optimizer                  | Momentum                                       | Momentum                                             | Momentum                                             |
| Loss Function              | Triplet loss                                   | Triplet loss                                         | Triplet loss                                         |
| Outputs                    | Probability                                    | Probability                                          | Probability                                          |
| Loss                       | 0.115702                                       | 0.000001                                             | 0.000001                                             |
| Speed                      | 110 ms/step (8pcs)                             | 46 ms/step (8pcs)                                    | 24 ms/step                                           |
| Total time                 | 21 minutes                                     | 21 minutes                                           | 11 minutes                                           |

#### ResNet50-Quadruplet on SOP

| Parameters                 | Ascend 910                                     | NVIDIA GeForce RTX 3090 (8 pcs)                      | NVIDIA GeForce RTX 3090 (Single)                     |
| -------------------------- | ---------------------------------------------- |------------------------------------------------------|------------------------------------------------------|
| Model version              | ResNet50-Quadruplet                            | ResNet50-Quadruplet                                  | ResNet50-Quadruplet                                  |
| Resource                   | Ascend 910; CPU: 2.60GHz, 192 cores; RAM: 755G | GeForce RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB | GeForce RTX 3090, CPU 2.90 GHz, 64 cores, RAM 252 GB |
| Uploaded Date              | 2021-03-25                                     | 2022-06-27                                           | 2022-06-27                                           |
| MindSpore Version          | 1.1.1-alpha                                    | 1.7.0                                                | 1.7.0                                                |
| Dataset                    | Stanford_Online_Products                       | Stanford_Online_Products                             | Stanford_Online_Products                             |
| Training Parameters        | epoch=30, steps per epoch=62, batch_size = 60  | epoch=40, batch_size=20, lr_max=0.0001               | epoch=40, batch_size=20, lr_max=0.0001               |
| Optimizer                  | Momentum                                       | Momentum                                             | Momentum                                             |
| Loss Function              | Quadruplet loss                                | Quadruplet loss                                      | Quadruplet loss                                      |
| Outputs                    | Probability                                    | Probability                                          | Probability                                          |
| Loss                       | 0.81702                                        | 0.08166                                              | 0.34311                                              |
| Speed                      | 90 ms/step (8pcs)                              | 48 ms/step (8pcs)                                    | 24 ms/step                                           |
| Total time                 | 12 minutes                                     | 96 minutes                                           | 47 minutes                                           |

# Description of Random Situation

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).