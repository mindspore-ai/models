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
            - [Evaluation while training](#evaluation-while-training)
    - [Resume Process](#resume-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-2)
            - [Running on Ascend](#running-on-ascend-2)
            - [Running on GPU](#running-on-gpu)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [ResNet50 on ImageNet2012](#resnet50-on-imagenet2012)
        - [Inference Performance](#inference-performance)
            - [ResNet50 on ImageNet2012](#resnet50-on-imagenet2012-1)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)
- [FAQ](#faq)

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

The [mixed precision](https://www.mindspore.cn/tutorials/experts/en/master/others/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
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
Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

- Running on GPU

```bash
# distributed training example
bash run_distribute_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training example
bash run_standalone_train_gpu.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

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
    ├── resnet50_imagenet2012_Boost_config.yaml     # High performance version: The performance is improved by more than 10% and the precision decrease less than 1%
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
    ├─  eval_callback.py                   # evaluation callback while training
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── lr_generator.py                    # generate learning rate for each step
    ├── resnet.py                          # resnet backbone, including resnet50 and resnet101 and se-resnet50
    └── resnet_gpu_benchmark.py            # resnet50 for GPU benchmark
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
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]

# standalone training
用法：bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH]
```

### Result

- Training ResNet50 with ImageNet2012 dataset

```bash
# distribute training result(8 pcs)
epoch: 1 step: 5004, loss is 4.8995576
epoch: 2 step: 5004, loss is 3.9235563
epoch: 3 step: 5004, loss is 3.833077
epoch: 4 step: 5004, loss is 3.2795618
epoch: 5 step: 5004, loss is 3.1978393
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
result: {'acc': 0.9363061543521088} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet18 with ImageNet2012 dataset

```bash
result: {'acc': 0.7053685897435897} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet50 with CIFAR-10 dataset

```bash
result: {'acc': 0.91446314102564111} ckpt=~/resnet50_cifar10/train_parallel0/resnet-90_195.ckpt
```

- Evaluating ResNet50 with ImageNet2012 dataset

```bash
result: {'acc': 0.7671054737516005} ckpt=train_parallel0/resnet-90_5004.ckpt
```

- Evaluating ResNet34 with ImageNet2012 dataset

```bash
result: {'top_1_accuracy': 0.736758814102564} ckpt=train_parallel0/resnet-90_625.ckpt
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

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --config_path [CONFIG_PATH]
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

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

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
| config                    | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/resnet/config) | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/resnet/config) |

### Inference Performance

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

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

Refer to the [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) for some common question.

- **Q: How to use `boost` to get the best performance?**

  **A**: We provide the `boost_level` in the `Model` interface, when you set it to `O1` or `O2` mode, the network will automatically speed up. The high-performance mode has been fully verified on resnet50, you can use the `resnet50_imagenet2012_Boost_config.yaml` to experience this mode. Meanwhile, in `O1` or `O2` mode, it is recommended to set the following environment variables: ` export ENV_FUSION_CLEAR=1; export DATASET_ENABLE_NUMA=True; export ENV_SINGLE_EVAL=1; export SKT_ENABLE=1;`.
