# Contents

- [NASNet Description](#nasnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NASNet Description](#contents)

[Paper](https://arxiv.org/abs/1707.07012): Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. 2017.

# [Model architecture](#contents)

The overall network architecture of NASNet is show below:

[Link](https://arxiv.org/abs/1707.07012v4)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2M colorful images in 1000 classes
    - Train: 120G, 1.2M images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
└─nasnet
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train_for_ascend.sh   # launch standalone training with Ascend platform(1p)
    ├─run_distribute_train_for_ascend.sh   # launch distributed training with Ascend platform(8p)
    ├─run_standalone_train_for_gpu.sh      # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh      # launch distributed training with gpu platform(8p)
    └─run_eval_for_ascend                  # launch evaluating with Ascend platform
    └─run_eval_for_gpu.sh                  # launch evaluating with gpu platform
  ├─src
    ├─config.py                            # parameter configuration
    ├─dataset.py                           # data preprocessing
    ├─loss.py                              # Customized CrossEntropy loss function
    ├─lr_generator.py                      # learning rate generator
├─nasnet_a_mobile.py                       # network definition
├─eval.py                                  # eval net
├─export.py                                # convert checkpoint
└─train.py                                 # train net  

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```python
'random_seed': 1,                # fix random seed
'rank': 0,                       # local rank of distributed
'group_size': 1,                 # world size of distributed
'work_nums': 8,                  # number of workers to read the data
'epoch_size': 600,               # total epoch numbers
'keep_checkpoint_max': 30,       # max numbers to keep checkpoints
'ckpt_path': './',               # save checkpoint path
'is_save_on_master': 0           # save checkpoint on rank0, distributed parameters
'train_batch_size': 32,          # input batch size for trainning
'val_batch_size': 32,            # input batch size for validating
'image_size' : 224,              # the size of one image
'num_classes': 1000,             # dataset class numbers
'label_smooth_factor': 0.1,      # label smoothing factor
'aux_factor': 0.4,               # loss factor of aux logit
'lr_init': 0.04*8,               # initiate learning rate
'lr_decay_rate': 0.97,           # decay rate of learning rate
'num_epoch_per_decay': 2.4,      # decay epoch number
'weight_decay': 0.00004,         # weight decay
'momentum': 0.9,                 # momentum
'opt_eps': 1.0,                  # epsilon
'rmsprop_decay': 0.9,            # rmsprop decay
'loss_scale': 1,                 # loss scale
'cutout': True,                  # whether to cutout the input data for training
'coutout_length': 56,              # the length of cutout when cutout is True
```

```python
'random_seed': 1,                # fix random seed
'rank': 0,                       # local rank of distributed
'group_size': 1,                 # world size of distributed
'work_nums': 8,                  # number of workers to read the data
'epoch_size': 600,               # total epoch numbers
'keep_checkpoint_max': 100,      # max numbers to keep checkpoints
'ckpt_path': './checkpoint/',    # save checkpoint path
'is_save_on_master': 0           # save checkpoint on rank0, distributed parameters
'train_batch_size': 32,          # input batch size for trainning
'val_batch_size': 32,            # input batch size for validating
'image_size' : 224,              # the size of one image
'num_classes': 1000,             # dataset class numbers
'label_smooth_factor': 0.1,      # label smoothing factor
'aux_factor': 0.4,               # loss factor of aux logit
'lr_init': 0.04*8,               # initiate learning rate
'lr_decay_rate': 0.97,           # decay rate of learning rate
'num_epoch_per_decay': 2.4,      # decay epoch number
'weight_decay': 0.00004,         # weight decay
'momentum': 0.9,                 # momentum
'opt_eps': 1.0,                  # epsilon
'rmsprop_decay': 0.9,            # rmsprop decay
'loss_scale': 1,                 # loss scale
'cutout': False,                 # whether to cutout the input data for training
'coutout_length': 56,            # the length of cutout when cutout is True
```

## [Training Process](#contents)

### Usage

```bash
# distribute training (8p)
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
bash run_distribute_train_for_gpu.sh [DATASET_PATH]
# standalone training
bash run_standalone_train_for_ascend.sh [DEVICE_ID] [DATASET_PATH]
bash run_standalone_train_for_gpu.sh [DEVICE_ID] [DATASET_PATH]
```

### Launch

```bash
# distributed training example(8p)
bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset
bash /run_distribute_train_for_gpu.sh /dataset
# standalone training example
bash run_standalone_train_for_ascend.sh 0 /dataset
bash run_standalone_train_for_gpu.sh 0 /dataset
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
bash run_eval_for_ascend.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
bash run_eval_for_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
```

### Launch

```bash
# Evaluation with checkpoint
bash run_eval_for_ascend.sh 0 /dataset ./ckpt_0/nasnet-a-mobile-rank0-248_10009.ckpt
bash run_eval_for_gpu.sh 0 /dataset ./ckpt_0/nasnet-a-mobile-rank0-248_10009.ckpt
```

### Result

Evaluation result will be stored in the ./eval path. Under this, you can find result like the followings in  `eval.log`.

acc=74.39%(TOP1,Ascend)
acc=73.5%(TOP1,GPU)

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

The checkpoint_file_path parameter is required,
`PLATFORM` should be in ["Ascend", "GPU", "CPU"]
`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH` should be the filename of the MINDIR model.
- `DATASET_NAME` should be imagenet2012.
- `DATASET_PATH` should be the path of the val in imaagenet2012 dataset.
- `NEED_PREPROCESS` can be y or n.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.
Top1 acc: 0.74376
Top5 acc: 0.91598

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                    | GPU                           |
| -------------------------- | ----------------------------- |-------------------------------|
| Model Version              | NASNet                        | NASNet                        |
| Resource                   | Ascend 910                    | NV SMX2 V100-32G              |
| uploaded Date              | 11/01/2021 (month/day/year)   | 09/24/2020                    |
| MindSpore Version          | 1.3.0                         | 1.0.0                         |
| Dataset                    | ImageNet                      | ImageNet                      |
| Training Parameters        | src/config.py                 | src/config.py                 |
| Optimizer                  | RMSProp                       | RMSProp                       |
| Loss Function              | CrossEntropy_Val              | CrossEntropy_Val              |
| Loss                       | 1.9617                        | 1.8965                        |
| Total time                 | 403 h 8ps                     | 144 h 8ps                     |
| Checkpoint for Fine tuning | 89 M(.ckpt file)              | 89 M(.ckpt file)              |

### Inference Performance

| Parameters                 | Ascend 910                    | GPU                           |
| -------------------------- | ----------------------------- |-------------------------------|
| Model Version              | NASNet                        | NASNet                        |
| Resource                   | Ascend 910                    | NV SMX2 V100-32G              |
| uploaded Date              | 11/01/2021 (month/day/year)   | 09/24/2020                    |
| MindSpore Version          | 1.3.0                         | 1.0.0                         |
| Dataset                    | ImageNet                      | ImageNet                      |
| batch_size                 | 32                            | 32                            |
| outputs                    | probability                   | probability                   |
| Accuracy                   | acc=74.39%(TOP1)              | acc=73.5%(TOP1)               |

# [ModelZoo Homepage](#contents)

Note: This model will be move to the `/models/research/` directory in r1.8.

Please check the official [homepage](https://gitee.com/mindspore/models).
