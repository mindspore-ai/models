# Contents

- [Contents](#contents)
- [ResNet Description](#resnet-description)
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
            - [Running on GPU](#running-on-gpu)
            - [Running parameter server mode training](#running-parameter-server-mode-training)
            - [Evaluation while training](#evaluation-while-training)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Apply algorithm in MindSpore Golden Stick](#apply-algorithm-in-mindspore-golden-stick)
    - [Training Process](#training-process-1)
        - [Running on GPU](#running-on-gpu-2)
    - [Evaluation Process](#evaluation-process-1)
        - [Running on GPU](#running-on-gpu-3)
        - [Result](#result-3)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [ResNet18 on CIFAR-10](#resnet18-on-cifar-10)
            - [ResNet18 on ImageNet2012](#resnet18-on-imagenet2012)
            - [ResNet50 on CIFAR-10](#resnet50-on-cifar-10)
            - [ResNet50 on ImageNet2012](#resnet50-on-imagenet2012)
            - [ResNet34 on ImageNet2012](#resnet34-on-imagenet2012)
            - [ResNet101 on ImageNet2012](#resnet101-on-imagenet2012)
            - [SE-ResNet50 on ImageNet2012](#se-resnet50-on-imagenet2012)
        - [Inference Performance](#inference-performance)
            - [ResNet18 on CIFAR-10](#resnet18-on-cifar-10-1)
            - [ResNet18 on ImageNet2012](#resnet18-on-imagenet2012-1)
            - [ResNet34 on ImageNet2012](#resnet34-on-imagenet2012-1)
            - [ResNet50 on CIFAR-10](#resnet50-on-cifar-10-1)
            - [ResNet50 on ImageNet2012](#resnet50-on-imagenet2012-1)
            - [ResNet101 on ImageNet2012](#resnet101-on-imagenet2012-1)
            - [SE-ResNet50 on ImageNet2012](#se-resnet50-on-imagenet2012-1)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ResNet Description](#contents)

## Description

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

These are examples of training ResNet18/ResNet50/ResNet101/ResNet152/SE-ResNet50 with CIFAR-10/ImageNet2012 dataset in MindSpore.ResNet50 and ResNet101 can reference [paper 1](https://arxiv.org/pdf/1512.03385.pdf) below, and SE-ResNet50 is a variant of ResNet50 which reference  [paper 2](https://arxiv.org/abs/1709.01507) and [paper 3](https://arxiv.org/abs/1812.01187) below, Training SE-ResNet50 for just 24 epochs using 8 Ascend 910, we can reach top-1 accuracy of 75.9%.(Training ResNet101 with dataset CIFAR-10 and SE-ResNet50 with CIFAR-10 is not supported yet.)

## Paper

1.[paper](https://arxiv.org/pdf/1512.03385.pdf):Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

2.[paper](https://arxiv.org/abs/1709.01507):Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. "Squeeze-and-Excitation Networks"

3.[paper](https://arxiv.org/abs/1812.01187):Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li. "Bag of Tricks for Image Classification with Convolutional Neural Networks"

# [Model Architecture](#contents)

The overall network architecture of ResNet is show below:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

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
    ├─train                 # train dataset
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

> - <font size=2>During training, if CIFAR-10 dataset is used, DATASET_PATH={CIFAR-10 directory}/cifar-10-batches-bin;</font>
>   <font size=2>If you are using ImageNet2012 dataset, DATASET_PATH={ImageNet2012 directory}/train</font>
> - <font size=2>During evaluating and inferring, if CIFAR-10 dataset is used, DATASET_PATH={CIFAR-10 directory}/cifar-10-verify-bin;</font>
>   <font size=2>If you are using ImageNet2012 dataset, DATASET_PATH={ImageNet2012 directory}/validation_preprocess</font>

- Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

- Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# infer example
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]

# gpu benchmark example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)
```

- Running on CPU

```bash
# standalone training example
python train.py --device_target=CPU --data_path=[DATASET_PATH]  --config_path [CONFIG_PATH] --pre_trained=[CHECKPOINT_PATH](optional)

# infer example
python eval.py --data_path=[DATASET_PATH] --checkpoint_file_path=[CHECKPOINT_PATH] --config_path [CONFIG_PATH]  --device_target=CPU
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) Add "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'" on the website UI interface.
# (2) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Set the code directory to "/path/resnet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Add "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'" on the website UI interface.
# (2) Copy or upload your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (4) Set the code directory to "/path/resnet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──resnet
  ├── README.md
  ├── config                               # parameter configuration
    ├── resnet18_cifar10_config.yaml
    ├── resnet18_cifar10_config_gpu.yaml
    ├── resnet18_imagenet2012_config.yaml
    ├── resnet18_imagenet2012_config_gpu.yaml
    ├── resnet34_cifar10_config_gpu.yaml
    ├── resnet34_imagenet2012_config.yaml
    ├── resnet34_imagenet2012_config_gpu.yaml
    ├── resnet50_cifar10_config.yaml
    ├── resnet50_imagenet2012_Boost_config.yaml     # High performance version: The performance is improved by more than 10% and the precision decrease less than 1%, the current configuration file supports 8 pcs.
    ├── resnet50_imagenet2012_Ascend_Thor_config.yaml
    ├── resnet50_imagenet2012_config.yaml
    ├── resnet50_imagenet2012_GPU_Thor_config.yaml
    ├── resnet101_imagenet2012_config.yaml
    ├── resnet152_cifar10_config_gpu.yaml
    ├── resnet152_imagenet2012_config.yaml
    ├── resnet152_imagenet2012_config_gpu.yaml
    ├── resnet_benchmark_GPU.yaml
    └── se-resnet50_imagenet2012_config.yaml
  ├── scripts
    ├── run_distribute_train.sh            # launch ascend distributed training(8 pcs)
    ├── run_parameter_server_train.sh      # launch ascend parameter server training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train.sh            # launch ascend standalone training(1 pcs)
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_parameter_server_train_gpu.sh  # launch gpu parameter server training(8 pcs)
    ├── run_eval_gpu.sh                    # launch gpu evaluation
    ├── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
    ├── run_gpu_resnet_benchmark.sh        # launch gpu benchmark train for resnet50 with imagenet2012
    |── run_eval_gpu_resnet_benckmark.sh   # launch gpu benchmark eval for resnet50 with imagenet2012
    └── cache_util.sh                      # a collection of helper functions to manage cache
  ├── src
    ├── dataset.py                         # data preprocessing
    ├─  callback.py                        # evaluation callback while training
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── logger.py                          # logger function
    ├── resnet.py                          # resnet backbone, including resnet50 and resnet101 and se-resnet50
    └── resnet_gpu_benchmark.py            # resnet50 for GPU benchmark
    ├─  util.py                            # define basic function
    ├── model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       ├──moxing_adapter.py                # moxing adapter
  ├── export.py                            # export model for inference
  ├── mindspore_hub_conf.py                # mindspore hub interface
  ├── eval.py                              # eval net
  ├── train.py                             # train net
  └── gpu_resent_benchmark.py              # GPU benchmark for resnet50
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for ResNet18 and ResNet50, CIFAR-10 dataset

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

- Config for ResNet18 and ResNet50, ImageNet2012 dataset

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
"warmup_epochs": 2,               # number of warmup epoch
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

- Config for ResNet34, ImageNet2012 dataset

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
"keep_checkpoint_max": 1,        # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 2,               # number of warmup epoch
"optimizer": 'Momentum',          # optimizer
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0,                     # initial learning rate
"lr_max": 1.0,                    # maximum learning rate
"lr_end": 0.0,                    # minimum learning rate
```

- Config for ResNet101, ImageNet2012 dataset

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
"warmup_epochs": 2,               # number of warmup epoch
"lr_decay_mode": "cosine"         # decay mode for generating learning rate
"use_label_smooth": True,         # label_smooth
"label_smooth_factor": 0.1,       # label_smooth_factor
"lr": 0.1                         # base learning rate
"save_graphs": False,             # save graph results
"save_graphs_path": "./graphs",   # save graph results path
"has_trained_epoch":0,            # epoch size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to epoch_size minus has_trained_epoch
"has_trained_step":0,             # step size that model has been trained before loading pretrained checkpoint, actual training epoch size is equal to step_size minus has_trained_step
```

- Config for ResNet152, ImageNet2012 dataset

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
"warmup_epochs": 2,               # number of warmup epoch
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

- Config for SE-ResNet50, ImageNet2012 dataset

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
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

#### Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)

# infer example
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
[CONFIG_PATH]

# gpu benchmark training example
bash run_gpu_resnet_benchmark.sh [DATASET_PATH] [BATCH_SIZE](optional) [DTYPE](optional) [DEVICE_NUM](optional) [SAVE_CKPT](optional) [SAVE_PATH](optional)

# gpu benchmark infer example
bash run_eval_gpu_resnet_benchmark.sh [DATASET_PATH] [CKPT_PATH] [BATCH_SIZE](optional) [DTYPE](optional)
```

For distributed training, a hostfile configuration needs to be created in advance.

Please follow the instructions in the link [GPU-Multi-Host](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_gpu.html).

#### Running parameter server mode training

- Parameter server training Ascend example

```bash
bash run_parameter_server_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)
```

- Parameter server training GPU example

```bash
bash run_parameter_server_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)
```

#### Evaluation while training

```bash
# evaluation with distributed training Ascend example:
cd scripts/
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# example of reasoning during distributed breakpoint training:
cd scripts/
bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH] [RESUME_CKPT]

# evaluation with standalone training Ascend example:
cd scripts/
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# example of reasoning during single machine breakpoint training:
cd scripts/
bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH] [RESUME_CKPT]

# evaluation with distributed training GPU example:
cd scripts/
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]

# evaluation with standalone training GPU example:
cd scripts/
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL] [EVAL_DATASET_PATH]
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
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]

# standalone training
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT]
```

### Result

- Training ResNet18 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
2023-02-17 14:27:29,405:INFO:epoch: [1/90] loss: 1.082604, epoch time: 40.559 s, per step time: 207.995 ms
2023-02-17 14:27:31,711:INFO:epoch: [2/90] loss: 1.045892, epoch time: 2.413 s, per step time: 12.377 ms
2023-02-17 14:27:34,012:INFO:epoch: [3/90] loss: 0.729006, epoch time: 2.486 s, per step time: 12.750 ms
2023-02-17 14:27:36,326:INFO:epoch: [4/90] loss: 0.766412, epoch time: 2.443 s, per step time: 12.529 ms
2023-02-17 14:27:39,646:INFO:epoch: [5/90] loss: 0.655058, epoch time: 2.851 s, per step time: 14.621 ms
...
```

- Training ResNet18 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
2023-02-17 15:30:06,405:INFO:epoch: [1/90] loss: 5.023574, epoch time: 154.658 s, per step time: 247.453 ms
2023-02-17 15:31:45,711:INFO:epoch: [2/90] loss: 4.253309, epoch time: 99.524 s, per step time: 159.239 ms
2023-02-17 15:33:18,012:INFO:epoch: [3/90] loss: 3.703176, epoch time: 92.655 s, per step time: 148.248 ms
2023-02-17 15:34:34,326:INFO:epoch: [4/90] loss: 3.458283, epoch time: 76.299 s, per step time: 122.078 ms
2023-02-17 15:35:59,646:INFO:epoch: [5/90] loss: 3.603806, epoch time: 84.435 s, per step time: 135.097 ms
...
```

- Training ResNet34 with ImageNet2012 dataset

```text
# 分布式训练结果（8P）
2023-02-20 09:47:10,405:INFO:epoch: [1/90] loss: 5.044510, epoch time: 139.308 s, per step time: 222.893 ms
2023-02-20 09:48:30,711:INFO:epoch: [2/90] loss: 4.194771, epoch time: 79.498 s, per step time: 127.196 ms
2023-02-20 09:49:53,012:INFO:epoch: [3/90] loss: 3.736507, epoch time: 83.387 s, per step time: 133.419 ms
2023-02-20 09:51:17,326:INFO:epoch: [4/90] loss: 3.417167, epoch time: 83.253 s, per step time: 133.204 ms
2023-02-20 09:52:41,646:INFO:epoch: [5/90] loss: 3.444441, epoch time: 83.931 s, per step time: 134.290 ms
...
```

- Training ResNet50 with CIFAR-10 dataset

```bash
# distribute training result(8 pcs)
2023-02-20 10:14:13,405:INFO:epoch: [1/90] loss: 1.519848, epoch time: 63.275 s, per step time: 324.489 ms
2023-02-20 10:14:16,711:INFO:epoch: [2/90] loss: 1.497206, epoch time: 3.305 s, per step time: 16.950 ms
2023-02-20 10:14:19,012:INFO:epoch: [3/90] loss: 1.097057, epoch time: 3.315 s, per step time: 17.002 ms
2023-02-20 10:14:23,326:INFO:epoch: [4/90] loss: 0.852322, epoch time: 3.322 s, per step time: 17.036 ms
2023-02-20 10:14:27,646:INFO:epoch: [5/90] loss: 0.896606, epoch time: 4.432 s, per step time: 22.730 ms
...
```

- Training ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
2023-02-20 10:01:18,405:INFO:epoch: [1/90] loss: 5.282135, epoch time: 183.647 s, per step time: 588.613 ms
2023-02-20 10:03:02,711:INFO:epoch: [2/90] loss: 4.446517, epoch time: 103.711 s, per step time: 332.408 ms
2023-02-20 10:04:41,012:INFO:epoch: [3/90] loss: 3.916948, epoch time: 99.554 s, per step time: 319.804 ms
2023-02-20 10:06:15,326:INFO:epoch: [4/90] loss: 3.510729, epoch time: 94.192 s, per step time: 301.897 ms
2023-02-20 10:07:43,646:INFO:epoch: [5/90] loss: 3.402662, epoch time: 87.943 s, per step time: 281.867 ms
...
```

- Training ResNet101 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
2023-02-20 10:52:57,405:INFO:epoch: [1/90] loss: 5.139862, epoch time: 218.528 s, per step time: 43.671 ms
2023-02-20 10:55:18,711:INFO:epoch: [2/90] loss: 4.252709, epoch time: 140.305 s, per step time: 28.039 ms
2023-02-20 10:57:38,012:INFO:epoch: [3/90] loss: 4.101140, epoch time: 140.267 s, per step time: 28.031 ms
2023-02-20 10:59:58,326:INFO:epoch: [4/90] loss: 3.468216, epoch time: 140.142 s, per step time: 28.006 ms
2023-02-20 11:02:20,646:INFO:epoch: [5/90] loss: 3.155962, epoch time: 140.167 s, per step time: 28.411 ms
...
```

- Training ResNet152 with ImageNet2012 dataset

```bash
# 分布式训练结果（8P）
2023-02-20 11:29:43,405:INFO:epoch: [1/90] loss: 4.546348, epoch time: 308.530 s, per step time: 61.657 ms
2023-02-20 11:33:08,711:INFO:epoch: [2/90] loss: 4.020557, epoch time: 205.175 s, per step time: 41.002 ms
2023-02-20 11:36:34,012:INFO:epoch: [3/90] loss: 3.691725, epoch time: 205.198 s, per step time: 41.007 ms
2023-02-20 11:39:59,326:INFO:epoch: [4/90] loss: 3.230466, epoch time: 205.363 s, per step time: 41.040 ms
2023-02-20 11:43:27,646:INFO:epoch: [5/90] loss: 2.961051, epoch time: 208.493 s, per step time: 41.665 ms
...
```

- Training SE-ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
2023-02-20 11:57:34,405:INFO:epoch: [1/90] loss: 4.478792, epoch time: 185.971 s, per step time: 37.164 ms
2023-02-20 11:59:22,711:INFO:epoch: [2/90] loss: 4.082346, epoch time: 107.408 s, per step time: 21.464 ms
2023-02-20 12:01:09,012:INFO:epoch: [3/90] loss: 4.116436, epoch time: 107.551 s, per step time: 21.493 ms
2023-02-20 12:02:58,326:INFO:epoch: [4/90] loss: 3.494506, epoch time: 108.719 s, per step time: 21.726 ms
2023-02-20 12:04:45,646:INFO:epoch: [5/90] loss: 3.412843, epoch time: 107.505 s, per step time: 21.484 ms
...
```

- GPU Benchmark of ResNet50 with ImageNet2012 dataset

```bash
# ========START RESNET50 GPU BENCHMARK========
epoch: [0/1] step: [20/5004], loss is 6.940182 Epoch time: 12416.098 ms, fps: 412 img/sec.
epoch: [0/1] step: [40/5004], loss is 7.078993Epoch time: 3438.972 ms, fps: 1488 img/sec.
epoch: [0/1] step: [60/5004], loss is 7.559594Epoch time: 3431.516 ms, fps: 1492 img/sec.
epoch: [0/1] step: [80/5004], loss is 6.920937Epoch time: 3435.777 ms, fps: 1490 img/sec.
epoch: [0/1] step: [100/5004], loss is 6.814013Epoch time: 3437.154 ms, fps: 1489 img/sec.
...
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
# evaluation
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# evaluation example
bash run_eval.sh ~/cifar10-10-verify-bin /resnet50_cifar10/train_parallel0/resnet-90_195.ckpt config/resnet50_cifar10_config.yaml
```

> checkpoint can be produced in training process.

#### Running on GPU

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Evaluating ResNet18 with CIFAR-10 dataset

```bash
result: {'top_5_accuracy': 0.9988420294494239, 'top_1_accuracy': 0.9369917221518} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet18 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.89609375, 'top_1_accuracy': 0.7056089743589744} ckpt=train_parallel0/resnet-90_625.ckpt
```

- Evaluating ResNet50 with CIFAR-10 dataset

```bash
result: {'top_5_accuracy': 0.99879807699230679, 'top_1_accuracy': 0.9372996794891795} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet50 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.930090206185567, 'top_1_accuracy': 0.764074581185567} ckpt=train_parallel0/resnet-90_625.ckpt
```

- Evaluating ResNet34 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9166866987179487, 'top_1_accuracy': 0.7379497051282051} ckpt=train_parallel0/resnet-90_625.ckpt
```

- Evaluating ResNet101 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9429417413572343, 'top_1_accuracy': 0.7853513124199744} ckpt=train_parallel0/resnet-120_5004.ckpt
```

- Evaluating ResNet152 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9438420294494239, 'top_1_accuracy': 0.78817221518} ckpt= resnet152-140_5004.ckpt
```

- Evaluating SE-ResNet50 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9342589628681178, 'top_1_accuracy': 0.768065781049936} ckpt=train_parallel0/resnet-24_5004.ckpt

```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH] --batch_size 1
```

The checkpoint_file_path parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Add "config_path='/path_to_code/config/resnet50_imagenet2021_config.yaml'" on the website UI interface.
# (2) Upload or copy your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./resnet'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./resnet'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (4) Set the code directory to "/path/resnet" on the website UI interface.
# (5) Set the startup file to "export.py" on the website UI interface.
# (6) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET]  [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
```

- `NET_TYPE` can choose from [resnet18, se-resnet50, resnet34, resnet50, resnet101, resnet152].
- `DATASET` can choose from [cifar10, imagenet].
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

- Evaluating ResNet18 with CIFAR-10 dataset

```bash
Total data: 10000, top1 accuracy: 0.94.26, top5 accuracy: 0.9987.
```

- Evaluating ResNet18 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.70668, top5 accuracy: 0.89698.
```

- Evaluating ResNet34 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.7342.
```

- Evaluating ResNet50 with CIFAR-10 dataset

```bash
Total data: 10000, top1 accuracy: 0.9310, top5 accuracy: 0.9980.
```

- Evaluating ResNet50 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.7696, top5 accuracy: 0.93432.
```

- Evaluating ResNet101 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.7871, top5 accuracy: 0.94354.
```

- Evaluating ResNet152 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.78625, top5 accuracy: 0.94358.
```

- Evaluating SE-ResNet50 with ImageNet2012 dataset

```bash
Total data: 50000, top1 accuracy: 0.76844, top5 accuracy: 0.93522.
```

# Apply algorithm in MindSpore Golden Stick

MindSpore Golden Stick is a compression algorithm set for MindSpore. We usually apply algorithm in Golden Stick before training for smaller model size, lower power consuming or faster inference process.

MindSpore Golden Stick provides SimQAT and SCOP algorithm for ResNet50. SimQAT is a quantization-aware training algorithm that trains the quantization parameters of certain layers in the network by introducing fake-quantization nodes, so that the model can perform inference with less power consumption or higher performance during the deployment phase. SCOP algorithm is a reliable pruning algorithm, which reduces the influence of all potential irrelevant factors by constructing a scientific control mechanism, and effectively deletes nodes in proportion, thereby realizing the miniaturization of the model.

MindSpore Golden Stick provides SLB algorithm for ResNet18. SLB is provided by Huawei Noah's Ark Lab. SLB is a quantization algorithm with low-bit weight searching, it regards the discrete weights in an arbitrary quantized neural network as searchable variables, and utilize a differential method to search them accurately. In particular, each weight is represented as a probability distribution over the discrete value set. The probabilities are optimized during training and the values with the highest probability are selected to establish the desired quantized network. SLB have more advantage when quantize with low-bit compared with SimQAT.

MindSpore Golden Stick provides UniPruning algorithm for ResNet18/34/50/101/152 and other ResNet-like and VGG-like models.  UniPruning is provided by Intelligent Systems and Data Science Technology center of Huawei Moscow Research Center. UniPruning is a soft-pruning algorithm. It measures relative importance of channels in a hardware-friendly manner. Particularly, it groups channels in groups of size G, where each channel's importance is measured as a L2 norm of its weights multiplied by consecutive BatchNorm's gamma and the absolute group importance is given as the median of channel importances. The relative importance criteria of a channel group G group of a layer L is measured as the highest median of the layer L divided by the median of group G. The higher the relative importance of a group, the less a group contributes to the layer output. During training UniPruning algorithm every N epochs searches for channel groups with the highest relative criteria network-wise and zeroes channels in that groups until reaching target sparsity, which is gives as a % of parameters to prune. To obtain pruned model after training, pruning mask and zeroed weights from the last UniPruning step are used to physically prune the network.

## Training Process

| **Algorithm**  | SimQAT | SCOP | SLB | UniPruning |
| --------- | ------ | --- | ---- | ---- |
| **supported backend**  | GPU | GPU、Ascend | GPU | GPU, Ascend |
| **support pretrain** | yes | must provide pretrained ckpt | don't need and can't load pretrained ckpt | pretrained ckpt optional |
| **support continue-train** | yes | yes | yes | yes |
| **support distribute train** | yes | yes | yes | yes |

- `pretrain` means training the network without applying algorithm. `pretrained ckpt` is loaded when training network with algorithm applied.
- `continue-train` means stop the training process after applying algorithm and continue training process from checkpoint file of previous training process.

### Running on GPU

```text
# distributed training
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'train.py'.
bash run_distribute_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# distributed training example, apply SimQAT and train from beginning
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset

# distributed training example, apply SimQAT and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# distributed training example, apply SimQAT and train from pretrained checkpoint
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt

# standalone training
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'train.py'.
bash run_standalone_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# standalone training example, apply SimQAT and train from beginning
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset

# standalone training example, apply SimQAT and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# standalone training example, apply SimQAT and train from pretrained checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml /path/to/dataset PRETRAINED /path/to/pretrained_ckpt

# Just replace PYTHON_PATH and CONFIG_FILE for applying different algorithm, take standalone training ResNet18 with SLB algorithm applied as an example
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/slb/ ../quantization/slb/resnet18_cifar10_config.yaml /path/to/dataset
Or if we want to train ResNet50 distributively with SCOP algorithm applied
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../pruner/scop/ ../pruner/scop/resnet50_cifar10_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt

# For UniPruning on GPU set config.device_target = 'GPU'

# standalone training example, apply UniPruning and train from pretrained checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../pruner/uni_pruning/ ../pruner/uni_pruning/resnet50_config.yaml /path/to/dataset FP32 ./checkpoint/resnet-90.ckpt

# distributed training example, apply UniPruning and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_distribute_train_gpu.sh ../pruner/uni_pruning/ ../pruner/uni_pruning/resnet50_config.yaml /path/to/dataset FP32 ./checkpoint/resnet-90.ckpt
```

### Running on Ascend

```text
# For UniPruning on Ascend config.device_target = 'Ascend'

# distributed training example, apply UniPruning and train from pretrained checkpoint
bash scripts/run_distribute_train.sh /path/to/rank_table_file pruner/uni_pruning/ pruner/uni_pruning/resnet50_config.yaml /path/to/dataset FP32 ./checkpoint/resnet-90.ckpt

# standalone training example, apply UniPruning and train from pretrained checkpoint
bash scripts/run_standalone_train.sh pruner/uni_pruning/ /path/to/rank_table_file  pruner/uni_pruning/resnet50_config.yaml /path/to/dataset FP32 ./checkpoint/resnet-90.ckpt
```

## Evaluation Process

### Running on GPU

```text
# evaluation
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'eval.py'.
bash run_eval_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]

# evaluation example
cd ./golden_stick/scripts/
bash run_eval_gpu.sh ../quantization/simqat/ ../quantization/simqat/resnet50_cifar10_config.yaml ./cifar10/train/ ./checkpoint/resnet-90.ckpt

# Just replace PYTHON_PATH CONFIG_FILE for applying different algorithm, take SLB algorithm as an example
bash run_eval_gpu.sh ../quantization/slb/ ../quantization/slb/resnet18_cifar10_config.yaml ./cifar10/train/ ./checkpoint/resnet-100.ckpt
```

```text
# Obtain pruned model from UniPruning training:
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'export.py'.
bash run_export.sh [PYTHON_PATH] [CONFIG_FILE] [CHECKPOINT_PATH] [MASK_PATH]

# .JSON pruning masks are saved during training in experiment directory

# evaluation example
cd ./golden_stick/scripts/
bash run_export.sh ../pruner/uni_pruning/ ../pruner/uni_pruning/resnet50_config.yaml ./checkpoint/resnet-90.ckpt ./checkpoint/mask.json

```

### Running on Ascend

```text
#Evaluation for UniPruning consists of : loading pretrained checkpoint, physically pruning model according to the pruning mask that we got from train procedure and evaluation.
#At the end the pruned model is also exported as .MINDIR and .AIR for inference deployment

!!! # To get pruned model, config.mask_path (pruning mask) should be set:
    # Pruning masks are saved as .json during training in the os.path.join(config.output_dir, config.exp_name)

bash scripts/run_eval.sh pruner/uni_pruning pruner/uni_pruning/resnet50_config.yaml /path/to/val/dataset /path/to/checkpoint
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Apply SimQAT on ResNet50, and evaluating with CIFAR-10 dataset:

```text
result:{'top_1_accuracy': 0.9354967948717948, 'top_5_accuracy': 0.9981971153846154} ckpt=~/resnet50_cifar10/train_parallel0/resnet-180_195.ckpt
```

- Apply SimQAT on ResNet50, and evaluating with ImageNet2012 dataset:

```text
result:{'top_1_accuracy': 0.7254057298335468, 'top_5_accuracy': 0.9312684058898848} ckpt=~/resnet50_imagenet2012/train_parallel0/resnet-180_6672.ckpt
```

- Apply SCOP on ResNet50, and evaluating with CIFAR-10 dataset:

```text
result:{'top_1_accuracy': 0.9273838141025641} prune_rate=0.45 ckpt=~/resnet50_cifar10/train_parallel0/resnet-400_390.ckpt
```

- Apply SLB on ResNet18 with W4, and evaluating with CIFAR-10 dataset. W4 means quantize weight with 4bit:

```text
result:{'top_1_accuracy': 0.9534254807692307, 'top_5_accuracy': 0.9969951923076923} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W4, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W4 means quantize weight with 4bit:

```text
result:{'top_1_accuracy': 0.9537259230480767, 'top_5_accuracy': 0.9970251907601913} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W4A8, and evaluating with CIFAR-10 dataset. W4 means quantize weight with 4bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9493423482907600, 'top_5_accuracy': 0.9965192030237169} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W4A8, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W4 means quantize weight with 4bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9502425480769207, 'top_5_accuracy': 0.99679551926923707} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W2, and evaluating with CIFAR-10 dataset. W2 means quantize weight with 2bit:

```text
result:{'top_1_accuracy': 0.9503205128205128, 'top_5_accuracy': 0.9966947115384616} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W2, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W2 means quantize weight with 2bit:

```text
result:{'top_1_accuracy': 0.9509508250132057, 'top_5_accuracy': 0.9967347384161105} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W2A8, and evaluating with CIFAR-10 dataset. W2 means quantize weight with 2bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9463205184161728, 'top_5_accuracy': 0.9963947115384616} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W2A8, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W2 means quantize weight with 2bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9473382052115128, 'top_5_accuracy': 0.9964718041530417} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W1, and evaluating with CIFAR-10 dataset. W1 means quantize weight with 1bit:

```text
result:{'top_1_accuracy': 0.9485176282051282, 'top_5_accuracy': 0.9965945512820513} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W1, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W1 means quantize weight with 1bit:

```text
result:{'top_1_accuracy': 0.9491012820516176, 'top_5_accuracy': 0.9966351282059453} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W1A8, and evaluating with CIFAR-10 dataset. W1 means quantize weight with 1bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9450068910250512, 'top_5_accuracy': 0.9962450312382200} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W1A8, enable BatchNorm calibration and evaluating with CIFAR-10 dataset. W1 means quantize weight with 1bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.9466145833333334, 'top_5_accuracy': 0.9964050320512820} ckpt=~/resnet18_cifar10/train_parallel0/resnet-100_195.ckpt
```

- Apply SLB on ResNet18 with W4, and evaluating with ImageNet2012 dataset. W4 means quantize weight with 4bit:

```text
result:{'top_1_accuracy': 0.6858173076923076, 'top_5_accuracy': 0.8850560897435897} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W4, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W4 means quantize weight with 4bit:

```text
result:{'top_1_accuracy': 0.6865184294871795, 'top_5_accuracy': 0.8856570512820513} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W4A8, and evaluating with ImageNet2012 dataset. W4 means quantize weight with 4bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6809975961503861, 'top_5_accuracy': 0.8819477163043847} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W4A8, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W4 means quantize weight with 4bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6816538461538406, 'top_5_accuracy': 0.8826121794871795} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W2, and evaluating with ImageNet2012 dataset. W2 means quantize weight with 2bit:

```text
result:{'top_1_accuracy': 0.6840144230769231, 'top_5_accuracy': 0.8825320512820513} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W2, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W2 means quantize weight with 2bit:

```text
result:{'top_1_accuracy': 0.6841746794871795, 'top_5_accuracy': 0.8840344551282051} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W2A8, and evaluating with ImageNet2012 dataset. W2 means quantize weight with 2bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6791516410250210, 'top_5_accuracy': 0.8808693910256410} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W2A8, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W2 means quantize weight with 2bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6805694500104102, 'top_5_accuracy': 0.8814763916410150} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W1, and evaluating with ImageNet2012 dataset. W1 means quantize weight with 1bit:

```text
result:{'top_1_accuracy': 0.6652945112820795, 'top_5_accuracy': 0.8690705128205128} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W1, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W1 means quantize weight with 1bit:

```text
result:{'top_1_accuracy': 0.6675184294871795, 'top_5_accuracy': 0.8707516025641026} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W1A8, and evaluating with ImageNet2012 dataset. W1 means quantize weight with 1bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6589927884615384, 'top_5_accuracy': 0.8664262820512820} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply SLB on ResNet18 with W1A8, enable BatchNorm calibration and evaluating with ImageNet2012 dataset. W1 means quantize weight with 1bit, A8 means quantize activation with 8bit:

```text
result:{'top_1_accuracy': 0.6609142628205128, 'top_5_accuracy': 0.8670873397435898} ckpt=~/resnet18_imagenet2012/train_parallel0/resnet-100_834.ckpt
```

- Apply UniPruning on ResNet50 with 15% target sparsity on ImageNet2012 dataset:

```text
result:{'top_1_accuracy': 0.7622}, Parameters pruned = 15%, Ascend310 acceleration = 31%
```

- Apply UniPruning on ResNet50 with 25% target sparsity on ImageNet2012 dataset:

```text
result:{'top_1_accuracy': 0.7582}, Parameters pruned = 25%, Ascend310 acceleration = 35%
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### ResNet18 on CIFAR-10

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| Model Version              | ResNet18                                                |  ResNet18 |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  PCIE V100-32G        |
| uploaded Date              | 02/25/2021 (month/day/year)                          | 07/23/2021 (month/day/year)  |
| MindSpore Version          | 1.1.1                                                       | 1.3.0 |
| Dataset                    | CIFAR-10                                                    | CIFAR-10 |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32             | epoch=90, steps per epoch=195, batch_size = 32      |
| Optimizer                  | Momentum                                                         | Momentum                                   |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                             |
| outputs                    | probability                                                 | probability               |
| Loss                       | 0.0002519517                                                    |  0.0015517382    |
| Speed                      | 13 ms/step（8pcs）                     | 29 ms/step（8pcs） |
| Total time                 | 4 mins                          | 11 minds    |
| Parameters (M)             | 11.2                                                        | 11.2          |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                         | 85.4 (.ckpt file)     |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet18 on ImageNet2012

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | -------------------------------------- | -------------------------------------- |
| Model Version              | ResNet18                                                | ResNet18     |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  | PCIE V100-32G   |
| uploaded Date              | 02/25/2021 (month/day/year)  ；                        | 07/23/2021 (month/day/year)  |
| MindSpore Version          | 1.1.1                                                       | 1.3.0 |
| Dataset                    | ImageNet2012                                                    | ImageNet2012 |
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             | epoch=90, steps per epoch=625, batch_size = 256             |
| Optimizer                  | Momentum                                                         | Momentum  |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy    |
| outputs                    | probability                                                 | probability              |
| Loss                       | 2.15702                                                   | 2.168664 |
| Speed                      | 110ms/step（8pcs）  (may need to set_numa_enbale in dataset.py)                    | 107 ms/step（8pcs）                |
| Total time                 | 110 mins                        | 130 mins            |
| Parameters (M)             | 11.7                                                       | 11.7 |
| Checkpoint for Fine tuning | 90M (.ckpt file)                                         |  90M (.ckpt file)                                         |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet50 on CIFAR-10

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet50-v1.5                                                |ResNet50-v1.5|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  | GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 07/05/2021 (month/day/year)                          | 07/05/2021 (month/day/year)
| MindSpore Version          | 1.3.0                                                       |1.3.0         |
| Dataset                    | CIFAR-10                                                    | CIFAR-10
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32             |epoch=90, steps per epoch=195, batch_size = 32  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 0.000356                                                    | 0.000716  |
| Speed                      | 18.4ms/step（8pcs）                     |69ms/step（8pcs）|
| Total time                 | 6 mins                          | 20.2 mins|
| Parameters (M)             | 25.5                                                         | 25.5 |
| Checkpoint for Fine tuning | 179.7M (.ckpt file)                                         |179.7M (.ckpt file)|
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet50 on ImageNet2012

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet50-v1.5                                                |ResNet50-v1.5|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 07/05/2021 (month/day/year)  ；                        | 07/05/2021 (month/day/year)
| MindSpore Version          | 1.3.0                                                       |1.3.0         |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             |epoch=90, steps per epoch=626, batch_size = 256  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 1.8464266                                                    | 1.9023  |
| Speed                      | 118ms/step（8pcs）                     |270ms/step（8pcs）|
| Total time                 | 114 mins                          | 260 mins|
| Parameters (M)             | 25.5                                                         | 25.5 |
| Checkpoint for Fine tuning | 197M (.ckpt file)                                         |197M (.ckpt file)     |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet34 on ImageNet2012

| Parameters                 | Ascend 910                                                   |
| -------------------------- | -------------------------------------- |
| Model Version              | ResNet50-v1.5                                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 07/05/2020 (month/day/year)  ；                        |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | ImageNet2012                                                    |
| Training Parameters        | epoch=90, steps per epoch=626, batch_size = 256             |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.9575993                                                    |
| Speed                      | 111ms/step（8pcs）                     |
| Total time                 | 112 mins                          |
| Parameters (M)             | 20.79                                                         |
| Checkpoint for Fine tuning | 166M (.ckpt file)                                         |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet101 on ImageNet2012

| Parameters                 | Ascend 910                                                   |   GPU |
| -------------------------- | -------------------------------------- |---------------------------------- |
| Model Version              | ResNet101                                                |ResNet101|
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  GPU(Tesla V100 SXM2)，CPU 2.1GHz 24cores，Memory 128G
| uploaded Date              | 07/05/2021 (month/day/year)                          | 07/05/2021 (month/day/year)
| MindSpore Version          | 1.3.0                                                       |1.3.0         |
| Dataset                    | ImageNet2012                                                    | ImageNet2012|
| Training Parameters        | epoch=120, steps per epoch=5004, batch_size = 32             |epoch=120, steps per epoch=5004, batch_size = 32  |
| Optimizer                  | Momentum                                                         |Momentum|
| Loss Function              | Softmax Cross Entropy                                       |Softmax Cross Entropy           |
| outputs                    | probability                                                 |  probability          |
| Loss                       | 1.6453942                                                    | 1.7023412  |
| Speed                      | 30.3ms/step（8pcs）                     |108.6ms/step（8pcs）|
| Total time                 | 301 mins                          | 1100 mins|
| Parameters (M)             | 44.6                                                        | 44.6 |
| Checkpoint for Fine tuning | 343M (.ckpt file)                                         |343M (.ckpt file)     |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

#### ResNet152 on ImageNet2012

| Parameters | Ascend 910  |
|---|---|
| Model Version  | ResNet152  |
| Resource  |  Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date  | 02/10/2021 (month/day/year) |
| MindSpore Version  | 1.0.1 |
| Dataset  |  ImageNet2012 |
| Training Parameters   | epoch=140, steps per epoch=5004, batch_size = 32  |
| Optimizer  | Momentum  |
| Loss Function    |Softmax Cross Entropy |
| outputs  | probability |
| Loss | 1.7375104  |
| Speed|47.47ms/step（8pcs） |
| Total time   |  577 mins |
| Parameters(M)   | 60.19 |
| Checkpoint for Fine tuning | 462M（.ckpt file）  |
| config  | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config)  |

#### SE-ResNet50 on ImageNet2012

| Parameters                 | Ascend 910
| -------------------------- | ------------------------------------------------------------------------ |
| Model Version              | SE-ResNet50                                               |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 07/05/2021 (month/day/year)                         |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | ImageNet2012                                                |
| Training Parameters        | epoch=24, steps per epoch=5004, batch_size = 32             |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 1.754404                                                    |
| Speed                      | 24.6ms/step（8pcs）                     |
| Total time                 | 49.3 mins                                                  |
| Parameters (M)             | 25.5                                                         |
| Checkpoint for Fine tuning | 215.9M (.ckpt file)                                         |
| config                    | [Link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet/config) |

### Inference Performance

#### ResNet18 on CIFAR-10

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18               |
| Resource            | Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.mindir file)             |

#### ResNet18 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18               |
| Resource            | Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                      |
| Dataset             | ImageNet2012                |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | 70.53%                      |
| Model for inference | 45M (.mindir file)             |

#### ResNet34 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18               |
| Resource            | Ascend 910; OS Euler2.8                   |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                      |
| Dataset             | ImageNet2012                |
| batch_size          | 256                         |
| outputs             | probability                 |
| Accuracy            | 73.67%                      |
| Model for inference | 70M (.mindir file)             |

#### ResNet50 on CIFAR-10

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet50-v1.5               | ResNet50-v1.5               |
| Resource            | Ascend 910; OS Euler2.8                   | GPU                         |
| Uploaded Date       | 07/05/2021 (month/day/year) | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | CIFAR-10                    | CIFAR-10                    |
| batch_size          | 32                          | 32                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 91.44%                      | 91.37%                      |
| Model for inference | 91M (.mindir file)         |  |

#### ResNet50 on ImageNet2012

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet50-v1.5               | ResNet50-v1.5               |
| Resource            | Ascend 910; OS Euler2.8                | GPU                         |
| Uploaded Date       | 07/05/2021 (month/day/year) | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | ImageNet2012                | ImageNet2012                |
| batch_size          | 256                         | 256                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 76.70%                      | 76.74%                      |
| Model for inference | 98M (.mindir file)         |  |

#### ResNet101 on ImageNet2012

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ResNet101                   | ResNet101                    |
| Resource            | Ascend 910; OS Euler2.8     | GPU                         |
| Uploaded Date       | 07/05/2021 (month/day/year) | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.3.0                       |
| Dataset             | ImageNet2012                | ImageNet2012                |
| batch_size          | 32                          | 32                          |
| outputs             | probability                 | probability                 |
| Accuracy            | 78.53%                      | 78.64%                      |
| Model for inference | 171M (.mindir file)         |  |

#### ResNet152 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet152                   |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 09/01/2021 (month/day/year) |
| MindSpore Version   | 1.4.0                       |
| Dataset             | ImageNet2012                |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 78.60%                      |
| Model for inference | 236M (.mindir file)            |

#### SE-ResNet50 on ImageNet2012

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | SE-ResNet50                 |
| Resource            | Ascend 910; OS Euler2.8             |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.3.0                       |
| Dataset             | ImageNet2012                |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 76.80%                      |
| Model for inference | 109M (.mindir file)            |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: How to use `boost` to get the best performance?**

  **A**: We provide the `boost_level` in the `Model` interface, when you set it to `O1` or `O2` mode, the network will automatically speed up. The high-performance mode has been fully verified on resnet50, you can use the `resnet50_imagenet2012_Boost_config.yaml` to experience this mode. Meanwhile, in `O1` or `O2` mode, it is recommended to set the following environment variables: ` export ENV_FUSION_CLEAR=1; export DATASET_ENABLE_NUMA=True; export ENV_SINGLE_EVAL=1; export SKT_ENABLE=1;`.

- **Q: How to use to preprocess imagenet2012 dataset?**

  **A**: Suggested reference:https://bbs.huaweicloud.com/forum/thread-134093-1-1.html