# Contents

- [Contents](#contents)
- [ResNeXt152 Description](#resnext152-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Usage](#usage)
            - [Launch](#launch)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Launch](#launch-1)
            - [Result](#result)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
        - [Usage](#usage-2)
        - [result](#result-1)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
            - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNeXt152 Description](#contents)

ResNeXt is a simple, highly modularized network architecture for image classification. It designs results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set in ResNeXt. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.

[Paper](https://arxiv.org/abs/1611.05431):  Xie S, Girshick R, Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks. 2016.

# [Model architecture](#contents)

The overall network architecture of ResNeXt is show below:

[Link](https://arxiv.org/abs/1611.05431)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
    - Train: 120G, 1281167 images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend  processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─resnext152_64x4d
  ├─ascend310_infer                   # 310 inference code
    ├─inc
      ├─utils.h                       # Tool library header file
    ├─src
      ├─build.sh                      # Run script
      ├─CMakeLists.txt                # cmake file
      ├─main_preprocess.cc            # pre process
      ├─main.cc                       # the entry of main function
      ├─utils.cc                      # Tool library function implementation
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training for ascend(1p)
    ├─run_standalone_train_gpu.sh     # launch standalone training for gpu (1p)
    ├─run_distribute_train.sh         # launch distributed training for ascend(8p)
    ├─run_distribute_train_gpu.sh     # launch distributed training for gpu (8p)
    ├─run_eval.sh                     # launch evaluate
    └─run_eval_gpu.sh                 # launch evaluating for gpu
  ├─src
    ├─backbone
      ├─_init_.py                     # initialize
      ├─resnet.py                     # resnext152 backbone
    ├─model_utils
      ├─config.py                     # Related parameters
      ├─device_adapter.py             # Device adapter for ModelArts
      ├─local_adapter.py              # Local adapter
      ├─moxing_adapter.py             # Moxing adapter for ModelArts
    ├─utils
      ├─_init_.py                     # initialize
      ├─auto_mixed_precision.py       # Mixed precision
      ├─cunstom_op.py                 # network operation
      ├─logging.py                    # print log
      ├─optimizers_init_.py           # get parameters
      ├─sampler.py                    # distributed sampler
      ├─var_init_.py                  # calculate gain value
    ├─_init_.py                       # initialize
    ├─config.py                       # parameter configuration
    ├─crossentropy.py                 # CrossEntropy loss function
    ├─dataset.py                      # data preprocessing
    ├─eval_callback.py                # Inference during training
    ├─head.py                         # common head
    ├─image_classification.py         # get ResNet
    ├─metric.py                       # Inference
    ├─linear_warmup.py                # linear warmup learning rate
    ├─warmup_cosine_annealing.py      # learning rate each step
    ├─warmup_step_lr.py               # warmup step learning rate
  ├─create_imagenet2012_label.py      # create label
  ├─default_config.yaml               # parameters
  ├─eval.py                           # eval net
  ├─export.py                         # export mindir script
  ├─postprocess.py                    # 310 post-processing
  ├─train.py                          # train net
  ├─requirements.txt                  # Required python libraries
  ├─README.md                         # Documentation in English
  ├─README_CN.md                      # Documentation in Chinese
```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```config
"image_size": '224,224'                   # image size
"num_classes": 1000,                      # dataset class number
"per_batch_size": 128,                    # batch size of input tensor
"lr": 0.05,                               # base learning rate
"lr_scheduler": 'cosine_annealing',       # learning rate mode
"lr_epochs": '30,60,90,120',              # epoch of lr changing
"lr_gamma": 0.1,                          # decrease lr by a factor of exponential lr_scheduler
"eta_min": 0,                             # eta_min in cosine_annealing scheduler
"T_max": 150,                             # T-max in cosine_annealing scheduler
"max_epoch": 150,                         # max epoch num to train the model
"warmup_epochs" : 1,                      # warmup epoch
"weight_decay": 0.0001,                   # weight decay
"momentum": 0.9,                          # momentum
"is_dynamic_loss_scale": 0,               # dynamic loss scale
"loss_scale": 1024,                       # loss scale
"label_smooth": 1,                        # label_smooth
"label_smooth_factor": 0.1,               # label_smooth_factor
"ckpt_interval": 2000,                    # ckpt_interval
"ckpt_path": 'outputs/',                  # checkpoint save location
"is_save_on_master": 1,
"rank": 0,                                # local rank of distributed
"group_size": 1                           # world size of distributed
```

For GPU training we modify the following parameters:

```python
"per_batch_size": 16,                   # batch size of input tensor (32 for 1P, 16 for 8P)
"lr": 0.05,                             # base learning rate (0.0125 for 1P, 0.05 for 8P)
```

## [Training Process](#contents)

### Usage

You can start training by running the python script:

```script
python train.py --data_dir ~/imagenet/train/ --platform Ascend --is_distributed 0
```

> platform can be "Ascend" or "GPU"

or shell script:

```script
Ascend:
    # distribute training example(8p)
    bash run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # standalone training
    bash run_standalone_train.sh DEVICE_ID DATA_PATH

GPU:
    # distribute training example(8p)
    bash run_distribute_train_gpu.sh DATA_DIR
    # standalone training
    bash run_standalone_train_gpu.sh DATA_DIR
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

You can start evaluation by running the following python script:

```script
python eval.py --data_dir ~/imagenet/val/ --platform Ascend --pretrained resnext.ckpt
```

> platform can be "Ascend" or "GPU"

or shell script:

```script
Ascend or GPU:
    bash run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH PLATFORM

Separate script for a GPU:
    bash run_eval_gpu.sh DATA_DIR PATH_CHECKPOINT
```

PLATFORM is Ascend or GPU, default is Ascend.

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

```log
acc=80.08%(TOP1)
acc=94.71%(TOP5)
```

Example for the GPU evaluation:

```text
...
[DATE/TIME]:INFO:load model /path/to/checkpoints/ckpt_0/0-148_10009.ckpt success
[DATE/TIME]:INFO:Inference Performance: 218.14 img/sec
[DATE/TIME]:INFO:before results=[[39666], [46445], [49984]]
[DATE/TIME]:INFO:after results=[[39666] [46445] [49984]]
[DATE/TIME]:INFO:after allreduce eval: top1_correct=39666, tot=49984,acc=79.36%(TOP1)
[DATE/TIME]:INFO:after allreduce eval: top5_correct=46445, tot=49984,acc=92.92%(TOP5)
```

## [Model Export](#contents)

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be in ["AIR", "ONNX", "MINDIR"]

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the mindir file must be exported by export.py. Currently, only batchsize 1 is supported.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

```shell
Total data: 50000, top1 accuracy: 0.79174, top5 accuracy: 0.94178.
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | ResNeXt152                                    | ResNeXt152                                   |
| -------------------------- | --------------------------------------------- | -------------------------------------------- |
| Resource                   | Ascend 910, cpu:2.60GHz 192cores, memory:755G | 8x V100, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| uploaded Date              | 06/30/2021                                    | 06/30/2021                                   |
| MindSpore Version          | 1.2                                           | 1.5.0 (docker build, CUDA 11.1)              |
| Dataset                    | ImageNet                                      | ImageNet                                     |
| Training Parameters        | src/config.py                                 | src/config.py; lr=0.05, per_batch_size=16    |
| Optimizer                  | Momentum                                      | Momentum                                     |
| Loss Function              | SoftmaxCrossEntropy                           | SoftmaxCrossEntropy                          |
| Loss                       | 1.28923                                       | 2.172222                                     |
| Accuracy                   | 80.08%(TOP1)                                  | 79.36%(TOP1) (148 epoch, early stopping)     |
| Total time                 | 7.8 h 8ps                                     | 2 days 45 minutes (8P, processes)            |
| Checkpoint for Fine tuning | 192 M(.ckpt file)                             | -                                            |

#### Inference Performance

| Parameters        |                  |                  |                  |
| ----------------- | ---------------- | ---------------- | ---------------- |
| Resource          | Ascend 910       | GPU V100         | Ascend 310       |
| uploaded Date     | 06/20/2021       | 2021-10-27       | 2021-10-27       |
| MindSpore Version | 1.2              | 1.5.0, CUDA 11.1 | 1.3.0            |
| Dataset           | ImageNet, 1.2W   | ImageNet, 1.2W   | ImageNet, 1.2W   |
| batch_size        | 1                | 32               | 1                |
| outputs           | probability      | probability      | probability      |
| Accuracy          | acc=80.08%(TOP1) | acc=79.36%(TOP1) | acc=79.34%(TOP1) |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
