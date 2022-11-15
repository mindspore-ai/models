# README

- [Content](#content)
- [AutoAugment Description](#autoaugment-description)
    - [Overview](#overview)
    - [AutoAugment paper](#autoaugment-paper)
- [Model architecture](#model-architecture)
    - [WideResNet paper](#wideresnet-paper)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quick start](#quickstart)
- [Script description](#script-description)
- [Script parameters](#script-parameters)
- [Script usage](#script-usage)
    - [AutoAugment operator usage](#autoaugment-operator-usage)
    - [Training script usage](#training-script-usage)
    - [Evaluation script usage](#evaluation-script-usage)
    - [Export script usage](#export-script-usage)
    - [Inference script usage](#inference-script-usage)
- [Model description](#model-description)
- [Random case description](#random-case-description)
- [ModelZoo homepage](#modelzoo-homepage)

## AutoAugment Description

### Overview

Data augmentation is an important means to improve the accuracy and
generalization ability of image classifiers. Traditional data augmentation
methods mainly rely on manual design and use fixed augmentation processes
(such as combined applications ```RandomCrop``` and ```RandomHorizontalFlip```
image transformation operators). Different from traditional methods,
AutoAugment proposes an effective strategy space design for data augmentation,
enabling researchers to use different search algorithms (such as reinforcement
learning, evolutionary algorithms, and even random search, etc.) for specific
models and data. Set automatic customization augmentation process. Specifically,
the strategy space proposed by AutoAugment mainly covers the following concepts:

| Concept name| Concept brief                                                                                                                                                                                                                                    |
|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Operation   | Image transformation operators (such as translation, rotation, etc.), the operators selected by AutoAugment do not change the size and type of the input image; each operator has two searchable parameters, which are probability and magnitude.|
| Probability | The probability of randomly applying a certain image transformation operator, if not applied, it will directly return to the input picture.                                                                                                      |
| Magnitude   | Apply the strength of a certain image transformation operator, such as the number of pixels to translate, the angle of rotation, etc.                                                                                                            |
| Subpolicy   |Each sub-strategy contains two operators; when the sub-strategy is applied, the two operators transform the input image in order according to probability and magnitude.                                                                          |
| Policy      | Each strategy contains several sub-strategies. When data is augmented, the strategy randomly selects a sub-strategy for each picture.                                                                                                            |

Since the number of operators is limited and the probability and magnitude
parameters of each operator can be discretized, the strategy space proposed
by AutoAugment can lead to a finite state discrete search problem. In
particular, experiments show that the strategy space proposed by AutoAugment
also has a certain transferability, that is, the strategy obtained by using
a certain model and data set combination search can be migrated to other models
for the same data set, or using a certain data set The searched strategy can be
transferred to other similar data sets. This example is mainly implemented for
the strategy space proposed by AutoAugment. Developers can use the"good
strategies" listed in the AutoAugment paper to augment the data set based on
this example, or design a search algorithm based on this example to
automatically customize the augmentation process.

### AutoAugment paper

Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation strategies
from data." Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. 2019.

## Model architecture

In addition to implementing the strategy space proposed by AutoAugment,
this example also provides a simple implementation of the Wide-ResNet
model for developers' reference.

### WideResNet paper

Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv
preprint arXiv:1605.07146 (2016).

## Dataset

This example uses two datasets, Cifar10 and SVHN, as an example to introduce the use of
AutoAugment and verify the effectiveness of this example.

- CIFAR10

  This example uses
  [CIFAR-10 binary version](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz),
  and its directory structure is as follows:

  ```text
  cifar-10-batches-bin
  ├── batches.meta.txt
  ├── data_batch_1.bin
  ├── data_batch_2.bin
  ├── data_batch_3.bin
  ├── data_batch_4.bin
  ├── data_batch_5.bin
  ├── readme.html
  └── test_batch.bin
  ```

- SVHN

  You can download Street View House Numbers (SVHN) dataset [here](http://ufldl.stanford.edu/housenumbers/).

  We use the following files, containing cropped digits:

  ```text
  SVHN
  ├── train_32x32.mat
  ├── extra_32x32.mat
  ├── test_32x32.mat
  ```

  Before the training procedure, you need to process SVHN mat files using the svhn_preprocess.py script:

  ```bash
  python svhn_preprocess.py --mat_path=/path/to/SVHN/ --result_dir=/path/to/result/dir/
  ```

  This will extract the numbers and sort them into the directories corresponding to each digit.

## Environmental requirements

- Hardware（Ascend）
    - Prepare the Ascend processor to build the hardware environment.
- Software
    -[MindSpore](https://www.mindspore.cn/install/)
- For details, please refer to the following resources:：
    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## Quick start

After completing the preparation of the computing device and the framework
environment, the developer can run the following commands to train and evaluate
this example.

- Ascend processor environment operation

  ```bash

  # Distribute Ascend training
  bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH]

  # Standalone Ascend training
  bash run_standalone_train.sh [DATASET_PATH]

  # Evaluation on Ascend
  bash run_eval.sh [CHECKPOINT_PATH] [DATASET_PATH]

  ```

- GPU processor environment operation

  ```bash

  # Distribute GPU training
  bash ./scripts/run_distribute_train_gpu.sh [DATASET_PATH]

  # Standalone GPU training
  bash ./scripts/run_standalone_train_gpu.sh [DATASET_PATH]

  # Evaluation on GPU
  bash ./scripts/run_eval_gpu.sh [CHECKPOINT_PATH] [DATASET_PATH]
  ```

Distributed Ascend training requires the creation of an HCCL configuration
file in JSON format in advance. For specific operations, please refer to the
instructions in
[hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

## Script description

```text
├── scripts
    ├── run_distribute_train.sh         # bash script for Ascend distribute train
    ├── run_distribute_train_gpu.sh     # bash script for GPU distribute train
    ├── run_eval.sh                     # bash Ascend eval script
    ├── run_eval_gpu.sh                 #  bash GPU eval script
    ├── run_standalone_train.sh         # bash script for Ascend train
    ├── run_standalone_train_gpu.sh     # bash script for GPU train
    ├── run_infer_310.sh                # Ascend 310 infer
├── src
│   ├── config.py                   # config file
│   ├── dataset
│   │   ├── autoaugment
│   │   │   ├── aug.py              # AutoAugment strategy
│   │   │   ├── aug_test.py         # AutoAugment strategy test
│   │   │   ├── ops
│   │   │   │   ├── crop.py         # RandomCrop operator
│   │   │   │   ├── cutout.py       # RandomCutout operator
│   │   │   │   ├── effect.py       # Image special effects operator
│   │   │   │   ├── enhance.py      # Image enhancement operator
│   │   │   │   ├── ops_test.py     # Operator testing and visualization
│   │   │   │   └── transform.py    #  Image transformation operator
│   │   │   └── third_party
│   │   │       └── policies.py     # AutoAugment searched good strategys
│   │   └── cifar10.py              # Cifar10 dataset processing
│   │   └── svhn_dataset.py              # SVHN dataset processing
│   ├── network
│   │   └── wrn.py                  # Wide-ResNet definition
│   ├── optim
│   │   └── lr.py                   # cosine learning rate definition
│   └── utils                       # initialize log format, etc.
├── test.py                         #  test script
├── train.py                        #  train script
├── export.py                       # export
├── svhn_preprocess.py              # preprocess svhn dataset
├── preprocess.py                   # preprocess cifar10 dataset
├── postprocess.py                  # postprocess cifar10 dataset
├── mindspore_hub_conf.py           # MindSpore Hub configuration
├── README_CN.md                    # documentation in Chinese
├── README.md                       # documentation in English
├── requirements.txt                # requirements

```

## Script parameters

You can configure training parameters, data set path and other parameters in [src/config.py](./src/config.py).

```python

# Set to mute logs with lower levels.
self.log_level = logging.INFO

# Random seed.
self.seed = 1

# Type of device(s) where the model would be deployed to.
# Choices: ['Ascend', 'GPU', 'CPU']
self.device_target = 'Ascend'

# The model to use. Choices: ['wrn']
self.net = 'wrn'

# The dataset to train or test against. Choices: ['cifar10']
self.dataset = 'cifar10'
# The number of classes.
self.class_num = 10
# Path to the folder where the intended dataset is stored.
self.dataset_path = './cifar-10-batches-bin'

# Batch size for both training mode and testing mode.
self.batch_size = 128

# Indicates training or testing mode.
self.training = training

# Testing parameters.
if not self.training:
    # The checkpoint to load and test against.
    # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
    self.checkpoint_path = None

# Training parameters.
if self.training:
    # Whether to apply auto-augment or not.
    self.augment = True

    # The number of device(s) to be used for training.
    self.device_num = 1
    # Whether to train the model in a distributed mode or not.
    self.run_distribute = False
    # The pre-trained checkpoint to load and train from.
    # Example: './checkpoint/train_wrn_cifar10-200_390.ckpt'
    self.pre_trained = None

    # Number of epochs to train.
    self.epoch_size = 200  # 200 for CIFAR10, 50 for SVHN
    # Momentum factor.
    self.momentum = 0.9
    # L2 penalty.
    self.weight_decay = 5e-4
    # Learning rate decaying mode. Choices: ['cosine']
    self.lr_decay_mode = 'cosine'
    # The starting learning rate.
    self.lr_init = 0.1
    # The maximum learning rate.
    self.lr_max = 0.1
    # The number of warmup epochs. Note that during the warmup period,
    # the learning rate grows from `lr_init` to `lr_max` linearly.
    self.warmup_epochs = 5
    # Loss scaling for mixed-precision training.
    self.loss_scale = 1024

    # Create a checkpoint per `save_checkpoint_epochs` epochs.
    self.save_checkpoint_epochs = 5
    # The maximum number of checkpoints to keep.
    self.keep_checkpoint_max = 10
    # The folder path to save checkpoints.
    self.save_checkpoint_path = './checkpoint'
```

## Script usage

### AutoAugment operator usage

Similar to [src/dataset/cifar10.py](./src/dataset/cifar10.py)，
in order to use the AutoAugment operator, you first need
to introduce a Augmentclass:

```python

# Developers need to copy the "src/dataset/autoaugment/"
# folder completely to the current directory, or use a soft link
from autoaugment import Augment
```

The AutoAugment operator is compatible with the MindSpore data set,
so you can directly use it as the transformation
operator of the data set:

```python

dataset = dataset.map(operations=[Augment(mean=MEAN, std=STD)],
                      input_columns='image', num_parallel_workers=8)
```

The parameters supported by AutoAugment are as follows:

```python
Args:
    index (int or None): If index is not None, the indexed policy would
        always be used. Otherwise, a policy would be randomly chosen from
        the policies set for each image.
    policies (policies found by AutoAugment or None): A set of policies
        to sample from. When the given policies is None, good policies found
        on cifar10 would be used.
    enable_basic (bool): Whether to apply basic augmentations after
                         auto-augment or not. Note that basic augmentations
                         include RandomFlip, RandomCrop, and RandomCutout.
    from_pil (bool): Whether the image passed to the operator is already a
                     PIL image.
    as_pil (bool): Whether the returned image should be kept as a PIL image.
    mean, std (list): Per-channel mean and std used to normalize the output
                      image. Only applicable when as_pil is False.
```

### Training script usage

Use AutoAugment operator to augment the data set and perform model training:

```text

# python train.py -h
usage: train.py [-h] [--device_target {Ascend,GPU,CPU}] [--dataset {cifar10}]
                [--dataset_path DATASET_PATH] [--augment AUGMENT]
                [--device_num DEVICE_NUM] [--run_distribute RUN_DISTRIBUTE]
                [--lr_max LR_MAX] [--pre_trained PRE_TRAINED]
                [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                [--lr_init LR_INIT] [--epoch_size EPOCHS_NUMBER]

AutoAugment for image classification.

optional arguments:
  -h, --help            show this help message and exit
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
  --dataset {cifar10, svhn}   The dataset to train or test against.
  --dataset_path DATASET_PATH
                        Path to the folder where the intended dataset is
                        stored.
  --augment AUGMENT     Whether to apply auto-augment or not.
  --device_num DEVICE_NUM
                        The number of device(s) to be used for training.
  --run_distribute RUN_DISTRIBUTE
                        Whether to train the model in distributed mode or not.
  --lr_max LR_MAX       The maximum learning rate.
  --pre_trained PRE_TRAINED
                        The pre-trained checkpoint to load and train from.
                        Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt
  --save_checkpoint_path SAVE_CHECKPOINT_PATH
                        The folder path to save checkpoints.
  --lr_init             Initial learning rate in the beginning of warmup.
  --epoch_size          Number of epochs
```

### Evaluation script usage

Evaluate the accuracy of the trained model:

```text

# python test.py -h
usage: test.py [-h] [--device_target {Ascend,GPU,CPU}] [--dataset {cifar10}]
               [--dataset_path DATASET_PATH]
               [--checkpoint_path CHECKPOINT_PATH]

AutoAugment for image classification.

optional arguments:
  -h, --help            show this help message and exit
  --device_target {Ascend,GPU,CPU}
                        Type of device(s) where the model would be deployed
                        to.
  --dataset {cifar10}   The dataset to train or test against.
  --dataset_path DATASET_PATH
                        Path to the folder where the intended dataset is
                        stored.
  --checkpoint_path CHECKPOINT_PATH
                        The checkpoint to load and test against.
                        Example: ./checkpoint/train_wrn_cifar10-200_390.ckpt
```

### Export script usage

Export the trained model to AIR, ONNX or MINDIR format:

```text

# python export.py -h
usage: export.py [-h] [--device_id DEVICE_ID] --checkpoint_path
                 CHECKPOINT_PATH [--file_name FILE_NAME]
                 [--file_format {AIR,ONNX,MINDIR}]
                 [--device_target {Ascend,GPU,CPU}]

WRN with AutoAugment export.

optional arguments:
  -h, --help            show this help message and exit
  --device_id DEVICE_ID
                        Device id.
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint file path.
  --file_name FILE_NAME
                        Output file name.
  --file_format {AIR,ONNX,MINDIR}
                        Export format.
  --device_target {Ascend,GPU,CPU}
                        Device target.
```

### Inference script usage

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Transfer the MINDIR model exported by Ascend 910 to the Ascend 310 server,
and run the run_infer_310 script:

```bash
#Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` mindir file path
- `DATASET_PATH` Inference data set path
- `NEED_PREPROCESS` Indicates whether the data set needs to be pre-processed.
    You can choose from ```y``` or ```n```.
    If you choose ```y```, the cifar10 data set will be processed into bin format.
- `DEVICE_ID` Optional, the default value is 0.

### cifar10 result

The inference result is saved in the current path of script execution,
and you can see the following accuracy
calculation results in acc.log.

```text
'acc': 0.976
```

## Model description

### Ascend performance

| parameter           | Single card Ascend 910    | Eight cards Ascend 910                |
|:--------------------|:--------------------------|:--------------------------------------|
| resource            | Ascend 910                | Ascend 910                            |
| Upload date         | 2021.06.21                | 2021.06.24                            |
| MindSpore version   | 1.2.0                     | 1.2.0                                 |
| Training data set   | Cifar10                   | Cifar10                               |
| Training parameters | epoch=200, batch_size=128 | epoch=200, batch_size=128, lr_max=0.8 |
| Optimizer           | Momentum                  | Momentum                              |
| Output              | loss                      | loss                                  |
| Accuracy            | 97.42%                    | 97.39%                                |
| speed               | 97.73 ms/step             | 106.29 ms/step                        |
| Total time          | 127 min                   | 17 min                                |
| Fine-tune checkpoint| 277M (.ckpt file)         | 277M (.ckpt file)                     |
| script              | [autoaugment](./)         | [autoaugment](./)                     |

### GPU performance

| parameter           | Single card GPU           | Eight cards GPU                       |
|:--------------------|:--------------------------|:--------------------------------------|
| resource            | Nvidia Titan V            | Nvidia Titan V                        |
| Upload date         | -                         | -                                     |
| MindSpore version   | 1.5.0                     | 1.5.0                                 |
| Training data set   | SVHN                      | SVHN                                  |
| Training parameters | epoch=50, batch_size=128  | epoch=50, batch_size=128, lr_max=0.08 |
| Optimizer           | Momentum                  | Momentum                              |
| Output              | loss                      | loss                                  |
| Top-1, %            | 98.25%                    | 98.36%                                |
| speed               | 242.36 ms/step            | 351.9 ms/step                         |
| Total time          | 16 h                      | 3.1 h                                 |
| Fine-tune checkpoint| 289.8M (.ckpt file)       | 289.8M (.ckpt file)                   |
| script              | [autoaugment](./)         | [autoaugment](./)                     |

## Random case description

A random seed is set in [train.py](./train.py) to ensure the reproducibility of training.

## ModelZoo homepage

Please visit the official website [homepage](https://gitee.com/mindspore/models).
