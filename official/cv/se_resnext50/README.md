# Contents

- [SE_ResNeXt50 Description](#se_resnext50-description)
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
    - [Evaluation Process](#evaluation-process)
    - [Model Export](#model-export)
    - [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SE_ResNeXt50 Description](#contents)

SE-ResNeXt50 is a variant of ResNeXt50 which reference [paper 1](https://arxiv.org/abs/1709.01507) below, ResNeXt50 is a simple, highly modularized network architecture for image classification. It designs results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set in ResNeXt50. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.ResNeXt50 reference [paper 2](https://arxiv.org/abs/1611.05431) below.

[paper1](https://arxiv.org/abs/1709.01507)：Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu."Squeeze-and-Excitation Networks"

[paper2](https://arxiv.org/abs/1611.05431)：Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He." Aggregated Residual Transformations for Deep Neural Networks"

# [Model architecture](#contents)

The overall network architecture of SE_ResNeXt50 is show below:

[Link](https://arxiv.org/abs/1709.01507)

# [Dataset](#contents)

Dataset used: [imagenet2012](http://www.image-net.org/)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
- Train: 120G, 1281167 images
- Test: 5G, 50000 images
- Data format: RGB images
- Note: Data will be processed in src/dataset.py

# [Features](#contents)

## [Mixed Precision](#contents)

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.5/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.

For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend）
- Prepare hardware environment with Ascend processor.
- Framework
- [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r1.3/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```bash
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the code directory to "/path/SE_ResNeXt50" on the website UI interface.
# (3) Set the startup file to "train.py" on the website UI interface.
# (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (5) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the code directory to "/path/se_resnext50" on the website UI interface.
# (4) Set the startup file to "eval.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

# [Script description](#contents)

## [Script and sample code](#contents)

```python
.
└─SE_ResNeXt50
  ├─README.md
  ├─README_CN.md
  ├─scripts
    ├─run_standalone_train.sh         # launch standalone training for ascend(1p)
    ├─run_distribute_train.sh         # launch distributed training for ascend(8p)
    └─run_eval.sh                     # launch evaluating
  ├─src
    ├─backbone
      ├─_init_.py                     # initialize
      ├─resnet.py                     # SE_ResNeXt50 backbone
    ├─utils
      ├─_init_.py                     # initialize
      ├─cunstom_op.py                 # network operation
      ├─logging.py                    # print log
      ├─optimizers_init_.py           # get parameters
      ├─sampler.py                    # distributed sampler
      ├─var_init_.py                  # calculate gain value
    ├─_init_.py                       # initialize
    ├─config.py                       # parameter configuration
    ├─crossentropy.py                 # CrossEntropy loss function
    ├─dataset.py                      # data preprocessing
    ├─head.py                         # common head
    ├─image_classification.py         # get resnet
    ├─linear_warmup.py                # linear warmup learning rate
    ├─warmup_cosine_annealing.py      # learning rate each step
    ├─warmup_step_lr.py               # warmup step learning rate
  ├── model_utils
    ├──config.py                      # parameter configuration
    ├──device_adapter.py              # device adapter
    ├──local_adapter.py               # local adapter
    ├──moxing_adapter.py              # moxing adapter
  ├── default_config.yaml             # parameter configuration
  ├─eval.py                           # eval net
  ├──train.py                         # train net
  ├──export.py                        # export mindir script
  ├──mindspore_hub_conf.py            #  mindspore hub interface

```

## [Script Parameters](#contents)

Parameters for both training and evaluating can be set in config.py.

```python
image_size: [224,224]                         # image size
num_classes: 1000                             # dataset class number
batch_size: 1                                 # batch size of input
lr: 0.05                                      # base learning rate
lr_scheduler: "cosine_annealing"              # learning rate mode
lr_epochs: [30,60,90,120]                     # epoch of lr changing
lr_gamma: 0.1                                 # decrease lr by a factor of exponential
eta_min: 0                                    # eta_min in cosine_annealing scheduler
T_max: 150                                    # T-max in cosine_annealing scheduler
max_epoch: 150                                # max epoch num to train the model
warmup_epochs: 1                              # warmup epoch
weight_decay: 0.0001                          # weight decay
momentum: 0.9                                 # momentum
is_dynamic_loss_scale: 0                      # dynamic loss scale
loss_scale: 1024                              # loss scale
label_smooth: 1                               # label_smooth
label_smooth_factor: 0.                       # label_smooth_factor
per_batch_size: 128                           # batch size of input tensor
ckpt_interval: 2000                           # ckpt_interval
ckpt_save_max: 5                              # max of checkpoint save
is_save_on_master: 1
rank_save_ckpt_flag: 0                        # local rank of distributed
outputs_dir: ""                               # output path
log_path: "./output_log"                      # log path
```

## [Training Process](#contents)

### Usage

You can start training by python script:

```bash
python train.py --data_dir ~/imagenet/train/ --device_target Ascend --run_distribute 0
```

or shell script:

```bash
Ascend:
    # distribute training example(8p)
    sh run_distribute_train.sh RANK_TABLE_FILE DATA_PATH
    # standalone training
    sh run_standalone_train.sh DEVICE_ID DATA_PATH
```

#### Launch

```bash
# distributed training example(8p) for Ascend
sh run_distribute_train.sh RANK_TABLE_FILE /dataset/train
# standalone training example for Ascend
sh run_standalone_train.sh 0 /dataset/train
```

You can find checkpoint file together with result in log.

## [Evaluation Process](#contents)

### Usage

You can start executing by python script:

Before execution, modify the configuration item run_distribute of default config.yaml to False.

```bash
python eval.py --data_path ~/imagenet/val/ --device_target Ascend --checkpoint_file_path  se_resnext50.ckpt
```

or shell script:

```bash
# Evaluation
sh run_eval.sh DEVICE_ID DATA_PATH PRETRAINED_CKPT_PATH DEVICE_TARGET
```

DEVICE_TARGET is Ascend, default is Ascend.

#### Launch

```bash
# Evaluation with checkpoint
sh run_eval.sh 0 /opt/npu/datasets/classification/val /se_resnext50.ckpt Ascend
```

#### Result

Evaluation result will be stored in the scripts path. Under this, you can find result like the followings in log.

```log
acc=78.81%(TOP1)
acc=94.40%(TOP5)
```

## [Model Export](#contents)

```bash
python export.py --device_target [DEVICE_TARGET] --checkpoint_file_path [CKPT_PATH] --file_format [EXPORT_FORMAT]
```

The `checkpoint_file_path` parameter is required.
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"].

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./resnext50'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./SE_ResNeXt50'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the config_path="/path/yaml file" on the website UI interface.
# (3) Set the code directory to "/path/SE_ResNeXt50" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Usage

Before performing inference, the mindir file must be exported by export.py. Currently, only batchsize 1 is supported.

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

```log
Total data:50000, top1 accuracy:0.79174, top5 accuracy:0.94492.
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | SE_ResNeXt50                                                |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8 |
| uploaded Date              | 08/31/2021                                                  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | ImageNet2012                                                |
| Training Parameters        | default_config.yaml                                         |
| Optimizer                  | Momentum                                                    |
| Loss Function              | SoftmaxCrossEntropy                                         |
| Loss                       | 1.4159617                                                   |
| Accuracy                   | 78%(TOP1)                                                   |
| Total time                 | 10 h (8ps)                                                  |
| Checkpoint for Fine tuning | 212 M(.ckpt file)                                           |

#### Inference Performance

| Parameters        |                         |              |
| ----------------- | ----------------------- | ------------ |
| Resource          | Ascend 910; OS Euler2.8 | Ascend 310   |
| uploaded Date     | 08/31/2021              | 08/31/2021   |
| MindSpore Version | 1.3.0                   | 1.3.0        |
| Dataset           | ImageNet2012            | ImageNet2012 |
| batch_size        | 128                     | 1            |
| outputs           | probability             | probability  |
| Accuracy          | acc=78.61%(TOP1)        |              |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
