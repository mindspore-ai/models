# Contents

<!-- TOC -->

- [Contents](#contents)
    - [Network description](#network-description)
    - [Dataset](#dataset)
        - [Flash-MNIST dataset](#flash-mnist-dataset)
    - [Environmental requirements](#environmental-requirements)
    - [Quick start](#quick-start)
    - [Script description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script parameters](#script-parameters)
        - [Training process](#training-process)
            - [training](#training)
            - [training result](#training-result)
        - [Evaluation process](#evaluation-process)
            - [evaluation](#evaluation)
            - [evaluation result](#evaluation-result)
        - [Export process](#export-process)
            - [export](#export)
        - [Inference process](#inference-process)
            - [inference](#inference)
    - [Model description](#model-description)
        - [performance](#performance)
            - [evaluation performance](#evaluation-performance)
            - [inference performance](#inference-performance)
    - [Random Description](#random-description)
    - [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

## Network description

The Attention Cluster model is the best sequence model in the ActivityNet Kinetics Challenge 2017. The model processes the extracted RGB, Flow, and Audio feature data through Attention Clusters with Shifting Operation. Shifting Operation performs L2-normalization by adding an independently learnable linear transformation to the output of each attention unit, so that each attention unit tends to learn different components of features, so that Attention Cluster can better learn data with different distributions , to improve the learning representation ability of the whole network.

For details, please refer to [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550)

## Dataset

### Flash-MNIST dataset

Suppose the main directory where the video model code base is stored is: Code\_Root

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Data set size: 52.4M, a total of 10 classes, 60,000 28*28 images
    - Training set: 60k images
    - Test set: 50,000 images
- Data format: binary file
- Original MNIST dataset directory structure:

```bash
...
└── mnist_dataset_dir
     ├── t10k-images-idx3-ubyte
     ├── t10k-labels-idx1-ubyte
     ├── train-images-idx3-ubyte
     └── train-labels-idx1-ubyte
```

Prepare the data as follows:

```bash
    bash $Code_Root/scripts/make_dataset.sh [DEVICE] [DATASET_DIR] [RESULT_DIR]
```

- The directory structure of the dataset used for model training is as follows, you can also refer to the [implementation](https://github.com/longxiang92/Flash-MNIST) in the original author's repository:

```bash
...
└─ Datasets
    ├─ feature_train.pkl
    └─ feature_test.pkl
```

## Environmental requirements

- Hardware (Ascend processor)
    - Prepare Ascend or GPU processor to build hardware environment.
- framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- Install [MindSpore](https://www.mindspore.cn/install)
- Install related dependencies pip install -r requirements.txt
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## Quick start

After the data set is prepared, the model training and evaluation operations can be performed in sequence:

- train:

Ascend:

```shell
# Single card training
bash run_standalone_train_Ascend.sh [FC] [NATT] [EPOCHS] [DATASET_DIR] [RESULT_DIR] [DEVICE_ID]

# Ascend distributed Training
bash bash run_distributed_train_Ascend.sh [RANK_TABLE][RANK_SIZE][DEVICE_START][FC][NATT][EPOCHS][DATASET_DIR][RESULT_DIR]
```

GPU:

```shell
# Single card training
bash run_standalone_train.sh [CONFIG] [DATASET_DIR]

# GPU distributed Training
run_distributed_train_gpu.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG] [DATASET_DIR]
```

- evaluate:

```shell
# evaluate
bash bash run_eval.sh bash run_eval.sh [DEVICE] [CONFIG] [CHECKPOINT_PATH] [DATASET_DIR]
```

## Script description

### Script and sample code

```text
...
└─AttentionCluster
  ├─scripts
  | ├─make_dataset.sh # Build dataset script
  | ├─run_eval.sh # Evaluate the run script
  | ├─run_distributed_train_Ascend.sh # Ascend distributed training script
  | ├─run_distributed_train_gpu.sh # GPU distributed training script
  | ├─run_standalone_train_Ascend.sh # Ascend single card training script
  | └─run_standalone_train_gpu.sh # GPU single card training script
  |
  ├─src
  | ├─datasets
  | | ├─mnist.py # Read mnist dataset
  | | ├─mnist_feature.py # Extract the features of the merged mnist image
  | | ├─mnist_flash.py # Merge multiple mnist images
  | | ├─mnist_noisy.py # Generate noisy image from original mnist image
  | | └─mnist_sampler.py # Sample from the original mnist dataset
  | |
  | ├─models
  | | └─attention_cluster.py # AttentionCluster model definition
  | |
  | └─utils
  | └─config.py # Parameter configuration
  |
  ├─default_config.yaml # config file
  ├─README_CN.md # Chinese guide
  ├─README_EN.md # English guide
  ├─requirements.txt # Dependencies
  ├─eval.py # evaluation
  ├─export.py # model export
  ├─train.py # training
  ├─postprocess.py # Calculate the accuracy
  ├─preprocess.py # Dataset conversion
  └─make_dataset.py # Build a dataset
```

### Script parameters

Training parameters can be configured in `default_config.yaml`.

```text
device: ''
device_id: 0
seed: 1

distributed: False

batch_size: 2048
epochs: 100
natt: 128
lr: 0.001
weight_decay: 0
fc: 2

data_dir: ''
result_dir: ''
ckpt: ''
```

### Training process

#### train

On Ascend device:

```shell
# Single card Ascend training
bash run_standalone_train_Ascend.sh 1 1 200 '../data' '../results' 0

# Ascend distributed Training (8P)
bash bash run_distributed_train_Ascend.sh './rank_table_8pcs.json' 8 0 1 1 200 '../data' '../results'
```

On GPU device:

```shell
# Single card GPU training
bash run_standalone_train_gpu.sh ../default_config.yaml ~/Datasets/MNIST

# Distributed GPU Training (8P)
bash bash run_distributed_train_gpu.sh 8 0,1,2,3,4,5,6,7 ../default_config.yaml ~/Datasets/MNIST
```

Among them, Ascend distributed training also needs to input the directory of the corresponding `RANK_TABLE_FILE` file into the script (such as `./rank_table_8pcs.json`), `RANK_TABLE_FILE` can be [this method](#https://gitee.com/mindspore /models/tree/master/utils/hccl_tools) generated.

#### training results

During training, the current number of training rounds, the model loss value, the running time of each round and other information will be displayed in the following form:

```text
epoch: 1 step: 1600, loss is 6.558913
epoch time: 14510.303 ms, per step time: 9.069 ms
epoch: 2 step: 1600, loss is 6.3387423
epoch time: 10240.037 ms, per step time: 6.400 ms
epoch: 3 step: 1600, loss is 6.1619167
epoch time: 10197.713 ms, per step time: 6.374 ms
epoch: 4 step: 1600, loss is 6.1232996
epoch time: 10144.251 ms, per step time: 6.340 ms
epoch: 5 step: 1600, loss is 6.1846876
epoch time: 10158.274 ms, per step time: 6.349 ms
epoch: 6 step: 1600, loss is 6.040742
epoch time: 10141.502 ms, per step time: 6.338 ms
epoch: 7 step: 1600, loss is 6.1569977
epoch time: 10206.018 ms, per step time: 6.379 ms
epoch: 8 step: 1600, loss is 6.0411224
epoch time: 10160.674 ms, per step time: 6.350 ms
epoch: 9 step: 1600, loss is 6.008434
epoch time: 10316.776 ms, per step time: 6.448 ms
……
```

### Evaluation Process

#### evaluate

After completing the training process, the evaluation process will automatically load the optimal checkpoint for the corresponding task from the checkpoints directory for model evaluation:

```shell
bash run_eval.sh 1 1 '../data' '../results/attention_cluster-200_1600.ckpt' 0
```

#### evaluation result

After evaluation, the relevant evaluation results of the model on the validation set will be displayed in the following form:

```text
result: {'top_1_accuracy': 0.85205078125, 'top_5_accuracy': 0.9771484375} ckpt: ../results/attention_cluster-200_1600.ckpt
```

### Export process

#### export

Export the checkpoint file to mindir format model:

```shell
Python export.py --fc [FC] --night [NIGHT] --ckpt [CHECKPOINT_PATH]
```

### Inference process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

#### inference

After exporting the model we can do inference, the following shows an example of performing inference using the mindir model:

```shell
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

## Model description

### Performance

#### Evaluation performance

| Parameters        | Ascend                | GPU                   |
| ----------------- | --------------------- | --------------------- |
| Resources         | Ascend 910; CPU 2.60GHz, 192 cores; memory 755G; operating system Euler2.8 | RTX 3090 24GB |
| Upload Date       | 2021-10-15            | 2022-01-11            |
| MindSpore Version | 1.3.0                 | 1.5.0                 |
| Datasets          | MNIST                 | MNIST                 |
| Optimizer         | Adam                  | Adam                  |
| Loss Function     | Softmax Cross Entropy | Softmax Cross Entropy |
| Script            | [AttentionCluster](https://gitee.com/mindspore/models/tree/master/research/cv/AttentionCluster) | [AttentionCluster](https://gitee.com/mindspore/models/tree/master/research/cv/AttentionCluster) |

Results:

| fc | natt | epochs | learning rate | weight decay | Ascend top-1 acc (%) | 1P GPU top-1 acc (%) |
| -- | ---- | ------ | ------------- | ------------ | -------------------- | -------------------- |
| 1  | 1    | 200    | 1e-4          | 0            | 34.5                 | 31.8                 |
| 1  | 2    | 200    | 1e-4          | 0            | 65.8                 | 19.4                 |
| 1  | 4    | 200    | 1e-4          | 0            | 72.7                 | 37.8                 |
| 1  | 8    | 200    | 1e-4          | 0            | 80.5                 | 65.0                 |
| 1  | 16   | 200    | 5e-4          | 1e-4         | 84.6                 | 84.3                 |
| 1  | 32   | 200    | 5e-4          | 1e-4         | 84.7                 | 85.3                 |
| 1  | 64   | 200    | 5e-4          | 1e-4         | 85.0                 | 86.0                 |
| 1  | 128  | 200    | 5e-4          | 1e-4         | 85.2                 | 86.1                 |
| 2  | 1    | 200    | 1e-4          | 0            | 54.1                 | 40.3                 |
| 2  | 2    | 200    | 1e-4          | 0            | 64.7                 | 41.3                 |
| 2  | 4    | 200    | 1e-4          | 0            | 73.8                 | 53.1                 |
| 2  | 8    | 200    | 5e-4          | 1e-4         | 76.4                 | 81.6                 |
| 2  | 16   | 200    | 5e-4          | 1e-4         | 86.6                 | 85.5                 |
| 2  | 32   | 200    | 5e-4          | 1e-4         | 87.0                 | 87.6                 |
| 2  | 64   | 200    | 5e-4          | 1e-4         | 87.3                 | 88.1                 |
| 2  | 128  | 200    | 5e-4          | 1e-4         | 87.4                 | 88.3                 |

#### Inference performance

| Parameters        | Ascend       | GPU            |
| ----------------- | ------------ |--------------- |
| model version     | fc 1; natt 8 | fc 2; natt 128 |
| Resources         | Ascend 310   | RTX 3060 12GB  |
| Upload Date       | 2021-11-10   | 2022-01-11     |
| MindSpore Version | 1.3.0        | 1.5.0          |
| Datasets          | MNIST        | MNIST          |
| Accuracy          | 80.5%        | 86.8%          |

## Random description

The random seeds are fixed by mindspore.common.set_seed() in the train.py and eval.py scripts, which can be modified in the corresponding args.

## ModelZoo Homepage

Please visit the official [home page](https://gitee.com/mindspore/models).
