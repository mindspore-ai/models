# WideResNet

## Contents

- [Contents](#contents)
- [WideResNet Description](#resnet-description)
    - [Description](#description)
    - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Running on Ascend](#running-on-ascend)
            - [Running on GPU](#running-on-gpu)
            - [Running parameter server mode training](#running-parameter-server-mode-training)
            - [Evaluation while training](#evaluation-while-training)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [ResNet18 on CIFAR-10](#resnet18-on-cifar-10)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [WideResNet Description](#contents)

### Description

Szagoruyko proposed WideResNet on the basis of ResNet, which is used to solve the problem of deep and thin network models. Only a limited number of layers have learned useful knowledge, and more layers have made little contribution to the final result. This problem is also called diminishing feature reuse. The authors of WideResNet widened the residual block, which increased the training speed by several times, and the accuracy was also significantly improved.

Just like a ResNet - WideResNet network is not a network with any particular architecture, but an example of the idea of wide residual networks. So there is a group of networks called "wideresnet". But unlike ResNet, WideResNets differ by two numbers (not just one). The first number is the number of layers, as in resnet, and the second number is the "widening factor" and shows how many times the blocks of this network are _wider_ than the same blocks in ResNet.

These is example of training WideResNet-40-10 (40 layers and 10 times wider) with CIFAR-10 dataset in MindSpore.

### Paper

1.[[paper](https://arxiv.org/pdf/1605.07146.pdf)] **Wide Residual Networks**: Sergey Zagoruyko, Nikos Komodakis,

## [Model Architecture](#contents)

The overall network architecture of WideResNet is shown below:
[paper](https://arxiv.org/pdf/1605.07146.pdf)

## [Dataset](#contents)

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：60,000 32*32 colorful images in 10 classes
    - Train：50,000 images
    - Test： 10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```bash
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# Distributed training
 usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL]
 [DATASET_PATH] is the path of the dataset.
.

# Standalone training
 usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL]
[ DATASET_PATH] is the path of the data set.


# Run evaluation example
 usage:bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
[ DATASET_PATH] is the path of the data set.
[ CHECKPOINT_PATH] The trained ckpt file.
```

- Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
.
└──WideResNet
  ├── requirements.txt
  ├── README.md
  ├── config                               # parameter configuration
    ├── wideresnet_cifar10_config_gpu.yaml
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_eval_gpu.sh                    # launch gpu evaluation
    └── cache_util.sh                      # a collection of helper functions to manage cache
  ├── src
    ├── dataset.py                         # data preprocessing
    ├── callbacks.py                       # evaluation and save callbacks
    ├── cross_entropy_smooth.py            # loss definition for ImageNet2012 dataset
    ├── generator_lr.py                    # generate learning rate for each step
    ├── wide_resnet.py                     # wide_resnet backbone  
    ├── model_utils
       └── config.py                       # parameter configuration
  ├── export.py                            # Ascend 910 export network
  ├── eval.py                              # eval net
  └── train.py                             # train net
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for WideResNet-40-10, CIFAR-10 dataset

```bash
"num_classes" : 10 ,                 # Number of data set classes
"batch_size" : 32 ,                  # Input tensor batch size
"epoch_size" : 300 ,                 # Training period size
"save_checkpoint_path" : "./" ,      # Checkpoint relative execution path Jin’s save path
"repeat_num" : 1 ,                   # number of repetitions of data set
"widen_factor" : 10 ,                # network width
"depth" : 40 ,                       # network depth
"lr_init" : 0.1 ,                    # initial learning rate
"weight_decay" : 5e-4 ,             # Weight decay
"momentum" :0.9 ,                   # Momentum optimizer
"loss_scale" : 32 ,                  # Loss level
"save_checkpoint" : False ,         # Whether to save checkpoints during training
"save_checkpoint_epochs" : 5 ,       # Period interval between two checkpoints; by default, the last check Points will be saved after the last cycle is completed
"use_label_smooth" : True ,          # label smoothing
"label_smooth_factor" : 0.1 ,        # label smoothing factor
"pretrain_epoch_size" : 0 ,          # pretrain Training period
"warmup_epochs" :5,               # Warm-up cycle
```

### [Training Process](#contents)

#### Usage

##### Running on Ascend

```bash
# Distributed training
 usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [LABEL]
[ DATASET_PATH] is the path of the dataset.


# Standalone training
 usage: bash bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [LABEL]
[ DATASET_PATH] is the path of the data set.

```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

##### Running on GPU

```bash
# distributed training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL]

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL]
```

For distributed training, a hostfile configuration needs to be created in advance.

Please follow the instructions in the link [GPU-Multi-Host](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html).

##### Evaluation while training

```bash
# distributed training GPU with evaluation example:
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [RUN_EVAL] [EVAL_DATASET_PATH]

# standalone training GPU with evaluation example:
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [RUN_EVAL] [EVAL_DATASET_PATH]

# distributed training Ascend with evaluation example:
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [LABEL] [RUN_EVAL] [EVAL_DATASET_PATH]

# standalone training Ascend with evaluation example:
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [LABEL] [RUN_EVAL] [EVAL_DATASET_PATH]
```

`RUN_EVAL` and `EVAL_DATASET_PATH` are optional arguments, setting `RUN_EVAL`=True allows you to do evaluation while training. When `RUN_EVAL` is set, `EVAL_DATASET_PATH` must also be set.
And you can also set these optional arguments: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` for python script when `RUN_EVAL` is True.

By default, a standalone cache server would be started to cache all eval images in tensor format in memory to improve the evaluation performance. Please make sure the dataset fits in memory (Around 30GB of memory required for ImageNet2012 eval dataset, 6GB of memory required for CIFAR-10 eval dataset).

Users can choose to shutdown the cache server after training or leave it alone for future usage.

### [Resume Process](#contents)

#### Usage

##### Running on GPU

```text
# distributed training
Usage：bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [PRETRAINED_CKPT_PATH]

# standalone training
Usage：bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [PRETRAINED_CKPT_PATH]
```

##### Running on Ascend

```text
# distributed training
Usage：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [PRETRAINED_CKPT_PATH]

# standalone training
Usage：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [EXPERIMENT_LABEL] [PRETRAINED_CKPT_PATH]
```

### Result

- Training WideResNet-40-10 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5, loss is 2.3153763
epoch: 1 step: 5, loss is 2.274118
epoch: 1 step: 5, loss is 2.2663743
epoch: 1 step: 5, loss is 2.324574
epoch: 1 step: 5, loss is 2.253627
epoch: 1 step: 5, loss is 2.2363935
epoch: 1 step: 5, loss is 2.3112013
epoch: 1 step: 5, loss is 2.252127
...
```

### [Evaluation Process](#contents)

#### Usage

##### Running on Ascend

```bash
# Evaluation
 Usage: bash run_eval.sh [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH]
[ DATASET_PATH] is the path of the data set.
[ CHECKPOINT_PATH] The trained ckpt file.

```

```bash
# Evaluation example
 bash run_eval.sh /cifar10  ../config/wideresnet.yaml WideResNet_best.ckpt
```

> checkpoint can be produced in training process.

##### Running on GPU

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH]
```

#### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Evaluating WideResNet-40-10 with CIFAR-10 dataset

```bash
result: {'top_1_accuracy': 0.961738782051282}
```

## [Ascend310 reasoning process](#contents)

### [Export MindIR](#contents)

```bash
python export.py --ckpt_file [CKPT_PATH] --file_format [FILE_FORMAT] --device_id [0]

[ CKPT_PATH] is the ckpt file saved after training
```

The parameter ckpt_file is required and file_formatmust be selected in ["AIR", "MINDIR"].

### [Perform inference on Ascend310](#contents)

Before performing inference, the mindir file must be export.pyexported through a script. The following shows an example of using the mindir model to perform inference.

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir file path
- `DATASET_PATH` Inference data set path
- `DEVICE_ID` Optional, the default value is 0.

### [Result](#contents)

The inference result is saved in the current path of the script execution. You can view the inference accuracy in acc.log in the current folder and the inference time in time_Result.

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

##### WideResNet on CIFAR-10

| Parameters                 | Ascend 910                                                   | GPU | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |-------------------------------------- |
| Model Version              | WideResNet-40-10                                                |  WideResNet-40-10 |WideResNet-40-10 |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G  |  GeForce RTX 3090-24G (8 pcs)        | Tesla V100-PCIE 32G；CPU：2.60GHz 52cores ；RAM：754G (8 pcs)|
| uploaded Date              | 02/25/2021 (month/day/year)                          | 10/07/2021 (month/day/year)  | 11/30/2021 (month/day/year) |
| MindSpore Version          | 1.1.1                                                      | 1.3.0 | 1.6.0.20211122 |
| Dataset                    | CIFAR-10                                                    | CIFAR-10 | CIFAR-10 |
| Training Parameters        | epoch=300, steps per epoch=195, batch_size = 32             | epoch=300, steps per epoch=781, batch_size = 64      | epoch=300, steps per epoch=195, batch_size = 32 |
| Optimizer                  | Momentum                                                         | Momentum                                   | Momentum |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                             | Softmax Cross Entropy |
| outputs                    | probability                                                 | probability               |probability               |
| Loss                       | 0.545541                                                    |  0.545572    | 0.545078 |
| Speed                      | 65.2 ms/step (8 cards)）                     | 275 ms/step（8 cards） | 147ms/step (8 cards); 95ms/step (1 card) |
| Total time                 | 70 minutes                          | 16.8 h    | about 2.4h |
| Parameters (M)             | 52.1                                                        | 52.1         |52.1         |
| Checkpoint for Fine tuning | 426.49M (.ckpt file)                                         | 428M (.ckpt file)     |  427M (.ckpt file)
| Scripts                    | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/wideresnet) | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/wideresnet) | [Link](https://gitee.com/mindspore/models/tree/master/research/cv/wideresnet) |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

## FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: What should I do if memory overflow occurs when using PYNATIVE_MODE?**

  **A**: The memory overflow is usually because PYNATIVE_MODE requires more memory. Set the batch size to 16 to reduce memory consumption and allow network training.
