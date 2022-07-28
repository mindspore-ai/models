# Content

- [Content](#content)
- [HRNet_cls Description](#hrnet_cls-description)
- [Dataset](#dataset)
- [Environmental Requirements](#environmental-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Start](#start)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Start](#start-1)
        - [Result](#result-1)
    - [Model Export](#model-export)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)
- [Description of Random Cases](#description-of-random-cases)
- [ModelZoo Homepage](#modelzoo-homepage)

# HRNet_cls Description

HRNet is a versatile CV backbone network that can be used in the feature extraction stage of various CV tasks such as
image classification, semantic segmentation, and facial recognition.
The network maintains a high-resolution representation by concatenating convolutions from high to low resolution
throughout processing, and produces strong-resolution representations
by iteratively fusing different convolutions into parallel branches.

The following is an example of MindSpore using the ImageNet dataset to train HRNetW48 to complete
the image classification task. W48 indicates that the network width
(the number of channels of the feature map of the first branch) is 48.

[Paper](https://arxiv.org/pdf/1908.07919.pdf): Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao.

# Dataset

Dataset used: [ImageNet](http://www.image-net.org/)

- Dataset size: 146G, 1330k colored images of 1000 classes
    - Train: 140G, 1280k images
    - Val: 6G, 50k images
- Data format: RGB
    - Note: data prepares in src/dataset.py.

# Environmental Requirements

- Hardware（Ascend or GPU）
    - Use Ascend or GPU to build the hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Script Description

## Script and Sample Code

```text
└─ HRNet_cls
  ├─ README.md
  ├─ README_CN.md
  ├─ requirements.txt
  ├─ scripts
  │  ├─ run_distribute_train.sh             # script for distribute Ascend training
  │  ├─ run_distribute_train_gpu.sh         # script for distribute GPU training
  │  ├─ run_eval.sh                         # script for Ascend eval
  │  ├─ run_eval_gpu.sh                     # script for GPU eval
  │  ├─ run_standalone_train.sh             # script for single Ascend training
  │  └─ run_standalone_train_gpu.sh         # script for single GPU training
  ├─ src
  │  ├─ model_utils
  │  │ └─ moxing_adapter.py                 # modelarts training adaptation script
  │  ├─ callback.py                         # training and eval callbacks
  │  ├─ cls_hrnet.py                        # HRNet_cls model
  │  ├─ config.py                           # configuration parameters
  │  ├─ dataset.py                          # dataset script
  │  ├─ loss.py                             # loss function
  │  └─ utils.py                            # util functions
  ├─ eval.py                                # evaluation script
  ├─ export.py                              # model export script
  └─ train.py                               # training script
```

## Script Parameters

Parameters used during model training and evaluation can be set in config.py:

```python
'train_url': None,                      # training output path
'train_path': None,                     # training output path
'data_url': None,                       # training dataset path
'data_path': None,                      # training dataset path
'checkpoint_url': None,                 # checkpoint path
'checkpoint_path': None,                # checkpoint path
'eval_data_url': None,                  # inference dataset path
'eval_data_path': None,                 # inference dataset path
'eval_interval': 10,                    # interval for inference during training
'modelarts': False,                     # whether to use modelarts
'run_distribute': False,                # do distribute training
'device_target': 'Ascend',              # target device platform
'begin_epoch': 0,                       # start training from epoch
'end_epoch': 120,                       # end training at epoch
'total_epoch': 120,                     # total training epochs
'dataset': 'imagenet',                  # dataset name
'num_classes': 1000,                    # number of dataset classes
'batchsize': 16,                        # batch size per one device
'input_size': 224,                      # image input size
'lr_scheme': 'linear',                  # learning rate decay scheme
'lr': 0.01,                             # maximum learning rate
'lr_init': 0.0001,                      # initial learning rate
'lr_end': 0.00001,                      # final learning rate
'warmup_epochs': 2,                     # number of warmup epochs
'use_label_smooth': True,               # whether to use label smooth
'label_smooth_factor': 0.1,             # label smoothing factor
'conv_init': 'TruncatedNormal',         # distribution rule for conv init
'dense_init': 'RandomNormal',           # distribution rule for linear init
'optimizer': 'rmsprop',                 # optimizer
'loss_scale': 1024,                     # loss scale
'opt_momentum': 0.9,                    # optimizer momentum
'wd': 1e-5,                             # optimizer weight decay
'eps': 0.001                            # epsilon
'save_ckpt': True,                      # whether to save the ckpt file
'save_checkpoint_epochs': 1,            # save ckpt every chosen epoch
'keep_checkpoint_max': 10,              # save last max saved epochs
'model': {...}                          # HRNet model structure parameters
```

## Training Process

After installing MindSpore through the official website, you can follow the steps below for training and evaluation,
in particular, before training, you need to install `requirements.txt` by following command `pip install -r requirements.txt`.

### Start

```bash
# Training example
# Ascend
# Standalone training
bash scripts/run_standalone_train.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH](optional) [BEGIN_EPOCH](optional) [EVAL_DATASET_PATH](optional)
# Distribute training
bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [TRAIN_OUTPUT_PATH] [CHECKPOINT_PATH](optional) [BEGIN_EPOCH](optional) [EVAL_DATASET_PATH](optional)
# GPU
# Standalone training
bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]
# Distribute training
bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [TRAIN_OUTPUT_PATH]
```

- DATASET_PATH - path to the training part of the dataset.
- TRAIN_OUTPUT_PATH - path to the training logs and checkpoint dir.
- CHECKPOINT_PATH - path to the checkpoint (if you resume training from stopped train) (optional).
- BEGIN_EPOCH - resume training from selected epoch (option)
- EVAL_DATASET_PATH - path to the test part of the dataset. If specified, validation will runs during training (optional).

### Result

Checkpoint files will be stored under the custom path and training logs will be logged to `log`.
An example of the training log section is as follows:

```text
epoch: [ 1/120], epoch time: 2404040.882, steps: 10009, per step time: 240.188, avg loss: 4.093, lr:[0.005]
epoch: [ 2/120], epoch time: 827142.272, steps: 10009, per step time: 82.640, avg loss: 4.234, lr:[0.010]
epoch: [ 3/120], epoch time: 825985.514, steps: 10009, per step time: 82.524, avg loss: 3.057, lr:[0.010]
epoch: [ 4/120], epoch time: 825988.881, steps: 10009, per step time: 82.525, avg loss: 3.093, lr:[0.010]
```

## Evaluation Process

### Start

```bash
# Evaluation example
# Ascend
bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
# GPU
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

- DATASET_PATH - path to the test part of the dataset.
- CHECKPOINT_PATH - path to the trained model checkpoint.

### Result

Evaluation results can be viewed in `eval_log`.

```text
{'Loss': 1.9160713648223877, 'Top_1_Acc': 0.79358, 'Top_5_Acc': 0.9456}
```

## Model Export

You can convert the trained model to the chosen format ("AIR", "ONNX", "MINDIR") by following command:

```bash
python export.py --checkpoint_path [CHECKPOINT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --device_target [DEVICE_TARGET]
```

- CHECKPOINT_PATH - Path to the trained model checkpoint.
- FILE_NAME - Path and name of the output file.
- FILE_FORMAT - Output converted model format ("AIR", "ONNX", "MINDIR").
- DEVICE_TARGET - Target device platform ("Ascend", "GPU").

# Model Description

## Training Performance

| Parameter                | Ascend                    | GPU                                                  |
|--------------------------| --------------------------| -----------------------------------------------------|
| Model                    | HRNet                     | HRNet                                                |
| Model version            | W48-cls                   | W48-cls                                              |
| Operating environment    | HUAWEI CLOUD Modelarts    | Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| Upload date              | 2021-11-21                | 2022-05-31                                           |
| Dataset                  | imagenet                  | imagenet                                             |
| Training parameters      | src/config.py             | src/config.py                                        |
| Optimizer                | RMSProp                   | RMSProp                                              |
| Loss function            | CrossEntropySmooth        | CrossEntropySmooth                                   |
| Training duration (8p)   | 28.7h                     | 55.8h                                                |
| Parameter quantity (M)   | 296M                      | 296M                                                 |
| Script                   | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_cls) | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_cls) |

## Inference Performance

| Parameter                | Ascend                    | GPU                                                  |
|--------------------------| --------------------------| -----------------------------------------------------|
| Model                    | HRNet                     | HRNet                                                |
| Model version            | W48-cls                   | W48-cls                                              |
| Operating environment    | HUAWEI CLOUD Modelarts    | Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| Upload date              | 2021-11-21                | 2022-05-31                                           |
| Dataset                  | imagenet (val 50k images) | imagenet (val 50k images)                            |
| Inference parameters     | batch_size=16             | batch_size=64                                        |
| Inference duration       | 5min                      | 5min                                                 |
| Metric (8p)              | Top1[79.4%]               | Top1[79.3%]                                          |

# Description of Random Cases

We set the random seed in the `train.py` script.

# ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models)。
