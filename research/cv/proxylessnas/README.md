# Contents

- [ProxylessNas-Mobile Description](#proxylessnas-mobile-description)
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

# [ProxylessNas Description](#contents)

[Paper](https://arxiv.org/abs/1812.00332v2): Han Cai, Ligeng Zhu, Song Han. ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. 2019.

# [Model architecture](#contents)

The overall network architecture of ProxylessNas-Mobile is show below:

[Link](https://arxiv.org/abs/1812.00332v2)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2M colorful images in 1000 classes
    - Train: 120G, 1.2M images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
.
└─proxylessnas
  ├─README.md
  ├─scripts
    ├─run_standalone_train_for_ascend.sh   # launch standalone training with Ascend platform(1p)
    ├─run_distribute_train_for_ascend.sh   # launch distributed training with Ascend platform(8p)
    ├─run_standalone_train_for_gpu.sh      # launch standalone training with gpu platform(1p)
    ├─run_distribute_train_for_gpu.sh      # launch distributed training with gpu platform(8p)
    ├─run_eval_for_ascend                  # launch evaluating with Ascend platform
    └─run_eval_for_gpu.sh                  # launch evaluating with gpu platform
  ├─src
    ├─ model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       └──moxing_adapter.py                # moxing adapter
    ├─dataset.py                           # data preprocessing
    ├─lr_generator.py                      # learning rate generator
    └─proxylessnas_mobile.py               # network definition
  ├─default_config.yaml                    # parameter configuration
  ├─export.py                              # convert checkpoint
  ├─eval.py                                # eval net
  └─train.py                               # train net

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in default_config.yaml.

```text
'random_seed': 1,                          # fix random seed
'rank': 0,                                 # local rank of distributed
'group_size': 1,                           # world size of distributed
'work_nums': 8,                            # number of workers to read the data
'epoch_size': 360,                         # total epoch numbers
'keep_checkpoint_max': 80,                 # max numbers to keep checkpoints
'save_ckpt_path': './',                    # save checkpoint path
'train_batch_size': 131,                   # input batch size for training
'val_batch_size': 125,                     # input batch size for evaluation
'num_classes': 1000,                       # dataset class numbers
'lr_init': 0.2,                            # initiate learning rate
'weight_decay': 0.0001,                    # weight decay
'momentum': 0.9,                           # momentum
'cutout': True,                            # whether to cutout the input data for training
'coutout_length': 56,                      # the length of cutout when cutout is True
```

## [Training Process](#contents)

### Usage

```bash
# distribute training foar Ascend(8p)
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
# standalone training for Ascend
bash run_standalone_train_for_ascend.sh [DEVICE_ID] [DATASET_PATH]
```

### Launch

```bash
# distributed training example(8p) for Ascend
bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset
# standalone training example for for Ascend
bash run_standalone_train_for_ascend.sh 0 /dataset
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

```bash
# Evaluation
bash run_eval_for_ascend.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
```

### Launch

```bash
# Evaluation with checkpoint
bash run_eval_for_ascend.sh 0 /dataset ./train_parallel3/ckpt_3/proxylessnas-mobile-rank3-328_1223.ckpt
```

### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.
acc=75.04%(TOP1)

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --device_target [PLATFORM] --checkpoint [CHECKPOINT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
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
Top1 acc:  0.75088
Top5 acc:  0.92174

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                    |
| -------------------------- | ----------------------------- |
| Model Version              | ProxylessNas-Mobile           |
| Resource                   | Ascend 910                    |
| uploaded Date              | 10/23/2021 (month/day/year)   |
| MindSpore Version          | 1.3.0                         |
| Dataset                    | ImageNet                      |
| Training Parameters        | default_config.yaml           |
| Optimizer                  | SGD                           |
| Loss Function              | CrossEntropySmooth            |
| Loss                       | 1.9691                        |
| Total time                 | 16.7 h 8ps                    |
| Checkpoint for Fine tuning | 48.5 M(.ckpt file)            |

### Inference Performance

| Parameters                 | Ascend 910                    |
| -------------------------- | ----------------------------- |
| Model Version              | ProxylessNas-Mobile           |
| Resource                   | Ascend 910                    |
| uploaded Date              | 10/23/2021 (month/day/year)   |
| MindSpore Version          | 1.3.0                         |
| Dataset                    | ImageNet                      |
| batch_size                 | 125                           |
| outputs                    | probability                   |
| Accuracy                   | acc=75.04%(TOP1)              |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
