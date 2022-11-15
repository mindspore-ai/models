# Contents

- [Contents](#contents)
- [Res2Net Description](#res2net-description)
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
            - [Running parameter server mode training](#running-parameter-server-mode-training)
            - [Evaluation while training](#evaluation-while-training)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Res2Net Description](#contents)

## Description

We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g. , ResNet, ResNeXt, BigLittleNet, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models.

## Paper

1.[paper](https://arxiv.org/pdf/1904.01169.pdf):Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip. "Res2Net: A New Multi-scale Backbone Architecture"，TPAMI21

# [Model Architecture](#contents)

The overall network architecture of Res2Net is show below:
[Link](https://arxiv.org/pdf/1904.01169.pdf)

# [Dataset](#contents)

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

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```bash
└─dataset
    ├─ilsvrc                # train dataset
    └─validation_preprocess # evaluate dataset
```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config directory to "config_path=/The path of config in S3/"
# (3) Set the code directory to "/path/res2net" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the config directory to "config_path=/The path of config in S3/"
# (4) Set the code directory to "/path/res2net" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──res2net
  ├── README.md
  ├── config                               # parameter configuration
    ├── res2net50_cifar10_config.yaml
    ├── res2net50_imagenet2012_Boost_config.yaml     # High performance version: The performance is improved by more than 10% and the precision decrease less than 1%
    ├── res2net50_imagenet2012_Ascend_Thor_config.yaml
    ├── res2net50_imagenet2012_config.yaml
    ├── res2net50_imagenet2012_GPU_Thor_config.yaml
    ├── res2net101_imagenet2012_config.yaml
    ├── res2net152_imagenet2012_config.yaml
    └── se-res2net50_imagenet2012_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_parameter_server_train.sh      # launch ascend parameter server training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    └── cache_util.sh                      # a collection of helper functions to manage cache
  ├── src
    ├── dataset.py                         # data preprocessing
    ├─  eval_callback.py                   # evaluation callback while training
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── res2net.py                          # res2net backbone, including res2net50 and res2net101 and se-res2net50
    ├── model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       ├──moxing_adapter.py                # moxing adapter
  ├── export.py                            # export model for inference
  ├── mindspore_hub_conf.py                # mindspore hub interface
  ├── eval.py                              # eval net
  ├── train.py                             # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for Res2Net50, CIFAR-10 dataset

```bash
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_decay_mode": "poly"           # decay mode can be selected in steps, ploy and default
"lr_init": 0.01,                  # initial learning rate
"lr_end": 0.00001,                # final learning rate
"lr_max": 0.1,                    # maximum learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for Res2Net50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 256,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 90,                 # only valid for taining, which is always 1 for inference
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "Linear",        # decay mode for generating learning rate
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 0.8,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for Res2Net101, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 120,                # epoch size for training
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for Res2Net152, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 140,                # epoch size for training
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_path":"./",      # the save path of the checkpoint relative to the execution path
"save_checkpoint_epochs": 5,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 0,               # number of warmup epoch
"lr_decay_mode": "steps"          # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1,                        # base learning rate
"lr_end": 0.0001,                 # end learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for SE-Res2Net50, ImageNet2012 dataset

```bash
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum optimizer
"weight_decay": 1e-4,             # weight decay
"epoch_size": 28 ,                # epoch size for creating learning rate
"train_epoch_size": 24            # actual train epoch size
"pretrain_epoch_size": 0,         # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus pretrain_epoch_size
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 4,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last epoch
"keep_checkpoint_max": 10,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 3,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr_init": 0.0,                   # initial learning rate
"lr_max": 0.3,                    # maximum learning rate
"lr_end": 0.0001,                 # end learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

#### Running parameter server mode training

- Parameter server training Ascend example

```bash
bash run_parameter_server_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)
```

#### Evaluation while training

```bash
# evaluation with distributed training Ascend example:
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with standalone training Ascend example:
bash run_standalone_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with distributed training GPU example:
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)

# evaluation with standalone training GPU example:
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)
```

`RUN_EVAL` and `EVAL_DATASET_PATH` are optional arguments, setting `RUN_EVAL`=True allows you to do evaluation while training. When `RUN_EVAL` is set, `EVAL_DATASET_PATH` must also be set.
And you can also set these optional arguments: `save_best_ckpt`, `eval_start_epoch`, `eval_interval` for python script when `RUN_EVAL` is True.

By default, a standalone cache server would be started to cache all eval images in tensor format in memory to improve the evaluation performance. Please make sure the dataset fits in memory (Around 30GB of memory required for ImageNet2012 eval dataset, 6GB of memory required for CIFAR-10 eval dataset).

Users can choose to shutdown the cache server after training or leave it alone for future usage.

## [Resume Process](#contents)

### Usage

#### Running on Ascend

```text
# distributed training
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]

# standalone training
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]
```

### Result

- Training Res2Net50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 625, loss is 4.3152313
epoch time: 761451.474 ms, per step time: 1218.322 ms
epoch: 2 step: 625, loss is 3.6415665
epoch time: 521396.511 ms, per step time: 834.234 ms
epoch: 3 step: 625, loss is 3.470065
epoch time: 521327.250 ms, per step time: 834.124 ms
epoch: 4 step: 625, loss is 2.975669
epoch time: 521322.602 ms, per step time: 834.116 ms
epoch: 5 step: 625, loss is 3.146403
epoch time: 522505.419 ms, per step time: 836.009 ms
epoch: 6 step: 625, loss is 2.8917725
epoch time: 521711.263 ms, per step time: 834.738 ms
epoch: 7 step: 625, loss is 2.740367
epoch time: 521320.835 ms, per step time: 834.113 ms
epoch: 8 step: 625, loss is 2.8063378
epoch time: 521324.234 ms, per step time: 834.119 ms
epoch: 9 step: 625, loss is 2.840243
epoch time: 521321.925 ms, per step time: 834.115 ms
epoch: 10 step: 625, loss is 2.6885962
epoch time: 522509.646 ms, per step time: 836.015 ms
epoch: 11 step: 625, loss is 2.7149315
epoch time: 521653.353 ms, per step time: 834.645 ms
epoch: 12 step: 625, loss is 2.776991
epoch time: 521321.051 ms, per step time: 834.114 ms
...
```

- Training Res2Net101 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
using res2net101
epoch: 1 step: 5004, loss is 4.9935384
epoch time: 1275499.301 ms, per step time: 254.896 ms
epoch: 2 step: 5004, loss is 3.5333204
epoch time: 1012995.507 ms, per step time: 202.437 ms
epoch: 3 step: 5004, loss is 3.4315405
epoch time: 1012865.244 ms, per step time: 202.411 ms
epoch: 4 step: 5004, loss is 3.6367264
epoch time: 1012851.675 ms, per step time: 202.408 ms
epoch: 5 step: 5004, loss is 2.698619
epoch time: 1015196.445 ms, per step time: 202.877 ms
epoch: 6 step: 5004, loss is 3.3733695
epoch time: 1013130.002 ms, per step time: 202.464 ms
epoch: 7 step: 5004, loss is 2.9996243
epoch time: 1012861.776 ms, per step time: 202.410 ms
epoch: 8 step: 5004, loss is 2.3628292
epoch time: 1012879.776 ms, per step time: 202.414 ms
epoch: 9 step: 5004, loss is 2.7138257
epoch time: 1012875.267 ms, per step time: 202.413 ms
epoch: 10 step: 5004, loss is 2.6544142
epoch time: 1015161.856 ms, per step time: 202.870 ms
epoch: 11 step: 5004, loss is 2.447403
epoch time: 1013002.904 ms, per step time: 202.439 ms
epoch: 12 step: 5004, loss is 2.955596
epoch time: 1012869.543 ms, per step time: 202.412 ms
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
# evaluation
Usage: bash run_eval.sh [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH]
```

```bash
# evaluation example
bash run_eval.sh res2net50 cifar10 ~/cifar10-10-verify-bin ~/res2net50_cifar10/train_parallel0/res2net-90_195.ckpt --config_path /.yaml
```

> checkpoint can be produced in training process.

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
```

- The `checkpoint_file_path` parameter is required.
- `file_format` should be in ["AIR", "MINDIR"]
- `config_path` is the path of configuration yaml file, default is `res2net50_cifar10_config.yaml`.

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./res2net'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./res2net'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/res2net" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
For ImageNet2012, current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [CONFIG_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NET_TYPE` can choose from [res2net50, res2net152, res2net101, se-res2net50].
- `DATASET` can choose from [cifar10, imagenet].
- `DATA_PATH` is the path of test dataset.
- `CONFIG_PATH` is the config file of `NET_TYPE`.
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### Res2Net50 on ImageNet2012

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | Res2Net50                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 12/20/2021 (month/day/year)  ；                        |
| MindSpore Version          | 1.5.0                                                       |
| Dataset                    | ImageNet2012                                                    |
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Speed                      | 219ms/step（8pcs）                     |
| Parameters (M)             | 25.5                                                         |

#### Res2Net101 on ImageNet2012

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | Res2Net101                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 12/20/2021 (month/day/year)                          |
| MindSpore Version          | 1.5.0                                                       |
| Dataset                    | ImageNet2012                                                 |
| Training Parameters        | epoch=120, steps per epoch=5004, batch_size = 32             |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.6                                                         |
| Speed                      | 61.7ms/step（8pcs）                     |
| Parameters (M)             | 45                                                         |

#### Res2Net152 on ImageNet2012

| Parameters | Ascend 910  |
|---|---|
| Model Version  | Res2Net152  |
| Resource  |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date  | 12/20/2021 (month/day/year) |
| MindSpore Version  | 1.0.1 |
| Dataset  |  ImageNet2012 |
| Training Parameters   | epoch=140, steps per epoch=5004, batch_size = 32  |
| Optimizer  | Momentum  |
| Loss Function    |Softmax Cross Entropy |
| outputs  | probability |
| Loss | 1.7 |
| Speed|88.71ms/step（8pcs） |
| Parameters(M)   | 60 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: How to use `boost` to get the best performance?**

  **A**: We provide the `boost_level` in the `Model` interface, when you set it to `O1` or `O2` mode, the network will automatically speed up. The high-performance mode has been fully verified on res2net50, you can use the `res2net50_imagenet2012_Boost_config.yaml` to experience this mode. Meanwhile, in `O1` or `O2` mode, it is recommended to set the following environment variables: ` export ENV_FUSION_CLEAR=1; export DATASET_ENABLE_NUMA=True; export ENV_SINGLE_EVAL=1; export SKT_ENABLE=1;`.