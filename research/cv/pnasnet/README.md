# Contents

- [PNASNet Description](#pnasnet-description)
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

# [PNASNet Description](#contents)

[Paper](https://arxiv.org/abs/1712.00559v3): Chenxi Liu, etc. Progressive Neural Architecture Search. 2018.

# [Model architecture](#contents)

The overall network architecture of PNASNet is show below:

[Link](https://arxiv.org/abs/1712.00559v3)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 1.2M colorful images in 1000 classes
    - Train: 120G, 1.2M images
    - Test: 5G, 50000 images

- Dataset structure:

  ```bash
  └─dataset
    ├─train                              # folder of training set
    ├─val                                # folder of validation set
  ```

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
└─pnasnet
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
    ├─CrossEntropySmooth.py                # Customized Crossentropy loss function
    ├─dataset.py                           # data preprocessing
    ├─lr_generator.py                      # learning rate generator
    └─pnasnet_mobile.py                  # network definition
  ├─default_config.yaml                    # parameter configuration
  ├─export.py                              # convert checkpoint
  ├─eval.py                                # eval net
  └─train.py                               # train net

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in default_config.yaml.

```default_config.yaml
'random_seed': 1,                          # fix random seed
'rank': 0,                                 # local rank of distributed
'group_size': 1,                           # world size of distributed
'work_nums': 8,                            # number of workers to read the data
'epoch_size': 600,                         # total epoch numbers
'keep_checkpoint_max': 5,                 # max numbers to keep checkpoints
'checkpoint_path': './checkpoint/',        # save checkpoint path
'train_batch_size': 32,                    # input batch size for training
'val_batch_size': 125,                     # input batch size for evaluation
'num_classes': 1000,                       # dataset class numbers
'aux_factor': 0.4,                         # loss factor of aux logit
'lr_init': 0.04*8,                         # initiate learning rate
'lr_decay_rate': 0.97,                     # decay rate of learning rate
'num_epoch_per_decay': 2.4,                # decay epoch number
'weight_decay': 0.00004,                   # weight decay
'momentum': 0.9,                           # momentum
'opt_eps': 1.0,                            # epsilon
'rmsprop_decay': 0.9,                      # rmsprop decay
'loss_scale': 1,                           # loss scale
'cutout': True,                            # whether to cutout the input data for training
'coutout_leng': 56,                        # the length of cutout when cutout is True
```

## [Training Process](#contents)

### Training

- running on Ascend

  ```bash
  # Ascend standalone training
  bash run_standalone_train_for_ascend.sh [DEVICE_ID] [DATASET_PATH]
  ```

  ```bash
  # standalone training example for Ascend
  bash run_standalone_train_for_ascend.sh 0 /dataset/train
  ```

- running on GPU

  ```bash
  # GPU standalone training
  bash run_standalone_train_for_gpu.sh [DEVICE_ID] [DATASET_PATH]
  ```

  ```bash
  # standalone training example for GPU
  bash run_standalone_train_for_gpu.sh 0 /dataset/train
  ```

### Distributed Training

- running on Ascend

  ```bash
  # Ascend distributed training
  bash run_distribute_train_for_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
  ```

  ```bash
  # distributed training example(8p) for Ascend
  bash run_distribute_train_for_ascend.sh /home/hccl_8p_01234567.json /dataset/train
  ```

- running on GPU

  ```bash
  # GPU distributed training
  bash run_distribute_train_for_gpu.sh [DATASET_PATH]
  ```

  ```bash
  # distributed training example(8p) for GPU
  bash run_distribute_train_for_gpu.sh /dataset/train
  ```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

- running on Ascend

  ```bash
  # Evaluation
  bash run_eval_for_ascend.sh [DATASET_PATH] [CHECKPOINT]
  ```

  ```bash
  # Evaluation with checkpoint
   bash run_eval_for_ascend.sh /dataset/val ./checkpoint/pnasnet-a-mobile-rank0-600_10009.ckpt
  ```

- running on GPU

  ```bash
  # Evaluation
  bash run_eval_for_gpu.sh [DEVICE_ID] [DATASET_PATH] [CHECKPOINT]
  ```

  ```bash
  # Evaluation with checkpoint
  bash run_eval_for_gpu.sh 0 /dataset/val ./checkpoint/pnasnet-a-mobile-rank0-600_10009.ckpt
  ```

### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

- running on Ascend

  acc=74.5%(TOP1)

- running on GPU

  acc=74.3%(TOP1)
## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --device_target [PLATFORM] --checkpoint [CHECKPOINT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

The checkpoint_file_path parameter is required, `PLATFORM` should be in ["Ascend", "GPU", "CPU"]`FILE_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model. Current batch_Size can only be set to 1.

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
Top1 acc:  0.74484
Top5 acc:  0.91976

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                    | GPU                           |
| -------------------------- | ----------------------------- | ----------------------------- |
| Model Version              | PNASNet                       | PNASNet                       |
| Resource                   | Ascend 910                    | Tesla V100-PCIE               |
| uploaded Date              | 11/07/2021 (month/day/year)   | 12/22/2021 (month/day/year)   |
| MindSpore Version          | 1.2.0                         | 1.5.0                         |
| Dataset                    | ImageNet                      | ImageNet                      |
| Training Parameters        | default_config.yaml           | default_config.yaml           |
| Optimizer                  | RMSProp                       | RMSProp                       |
| Loss Function              | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| Loss                       | 1.0660                        | 1.9632                        |
| Total time                 | 576 h 8ps                     | 193 h 8ps                     |
| Checkpoint for Fine tuning | 97 M(.ckpt file)              | 91M(.ckpt file)               |

### Inference Performance

| Parameters        | Ascend 910                  | GPU                         |
| ----------------- | --------------------------- | --------------------------- |
| Model Version     | PNASNet                     | PNASNet                     |
| Resource          | Ascend 910                  | Tesla V100-PCIE             |
| uploaded Date     | 11/07/2021 (month/day/year) | 11/22/2021 (month/day/year) |
| MindSpore Version | 1.2.0                       | 1.5.0                       |
| Dataset           | ImageNet                    | ImageNet                    |
| batch_size        | 125                         | 125                         |
| outputs           | probability                 | probability                 |
| Accuracy          | acc=74.5%(TOP1)             | acc=74.3%(TOP1)             |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).