# Contents

- [Contents](#contents)
- [SNN Description](#snn-description)
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
            - [Running parameter server mode training](#running-parameter-server-mode-training)
            - [Evaluation while training](#evaluation-while-training)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
        - [Result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [LeNet on CIFAR-10](#lenet-on-cifar-10)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SNN Description](#contents)

## Description

SNN (Spiking neural networks) was proposed by Kaushik Roy and other two authors.

These are examples of training LeNet/ResNet50 with CIFAR-10 dataset.

## Paper

1.[paper](https://www.nature.com/articles/s41586-019-1677-2):Kaushik Roy, Akhilesh Jaiswal, Priyadarshini Panda. "Towards spike-based machine intelligence with neuromorphic computing"

# [Model Architecture](#contents)

The overall network architecture of SNN is show below:
[Link](https://www.nature.com/articles/s41586-019-1677-2)

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

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

> - <font size=2>During training, if CIFAR-10 dataset is used, DATASET_PATH={CIFAR-10 directory}/cifar-10-batches-bin;</font>
> - <font size=2>During evaluating and inferring, if CIFAR-10 dataset is used, DATASET_PATH={CIFAR-10 directory}/cifar-10-verify-bin;</font>

- Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train_ascend.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

```bash
# infer example
python eval.py --data_path=[DATASET_PATH] --ckpt_path=[CHECKPOINT_PATH] --config_path [CONFIG_PATH]
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```text
# run distributed training on modelarts example
# (1) Add "config_path='/path_to_code/config/snn_lenet_cifar10_config.yaml'" on the website UI interface.
# (2) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Set the code directory to "/path/snn" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Add "config_path='/path_to_code/config/snn_lenet_cifar10_config.yaml'" on the website UI interface.
# (2) Copy or upload your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (4) Set the code directory to "/path/snn" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└──snn
  ├── README.md
  ├── config                               # parameter configuration
    ├── snn_lenet_cifar10_config.yaml
    ├── snn_resnet50_cifar10_config.yaml
  ├── scripts
    ├── run_distribute_train_ascend.sh     # launch ascend distributed training(8 pcs)
    ├── run_eval.sh                        # launch ascend evaluation
    ├── run_standalone_train_ascend.sh     # launch ascend standalone training(1 pcs)
  ├── src
    ├── dataset.py                         # data preprocessing
    ├── ifnode.py                          # ifnode cell for snn
    ├── lr_generator.py                    # generate learning rate for each step
    ├── snn_lenet.py                       # lenet_snn for ascend benchmark
    ├── snn_resnet.py                      # resnet50_snn for ascend benchmark
    ├── model_utils
       ├──config.py                        # parameter configuration
       ├──device_adapter.py                # device adapter
       ├──local_adapter.py                 # local adapter
       ├──moxing_adapter.py                # moxing adapter
  ├── eval.py                              # eval net
  └── train.py                             # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for LeNet and ResNet50, CIFAR-10 dataset

```text
"class_num": 10,                  # dataset class num
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1024,               # loss scale
"momentum": 0.9,                  # momentum
"weight_decay": 1e-4,             # weight decay
"epoch_size": 5,                  # only valid for taining, which is always 1 for inference
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 5,         # only keep the last keep_checkpoint_max checkpoint
"warmup_epochs": 5,               # number of warmup epoch
"lr_init": 0.001,                 # initial learning rate
"save_graphs": False,             # save graph results
```

## [Training Process](#contents)

### Usage

#### Running on Ascend

```bash
# distributed training
Usage: bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train_ascend.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# run evaluation example
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccn_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

If you want to change device_id for standalone training, you can set environment variable `export DEVICE_ID=x` or set `device_id=x` in context.

## [Resume Process](#contents)

### Usage

#### Running on Ascend

```text
# distributed training
Usage: bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# standalone training
Usage: bash run_standalone_train_ascend.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)
```

### Result

- Training LeNet with CIFAR-10 dataset

```text
# training result on GRAPH mode(1 pcs)
epoch: 2, step: 250, loss is loss:0.090041, epoch time: 60457.911ms, per step time: 241.832ms
epoch: 3, step: 250, loss is loss:0.088589, epoch time: 60414.800ms, per step time: 241.659ms
epoch: 4, step: 250, loss is loss:0.078889, epoch time: 60488.454ms, per step time: 241.954ms
epoch: 5, step: 250, loss is loss:0.072030, epoch time: 60465.275ms, per step time: 241.861ms
```

- Training ResNet50 with CIFAR-10 dataset

```text
# training result on GRAPH mode(1 pcs)
epoch: 2, step: 1562, loss is loss:2.575632, epoch time: 106556.557ms, per step time: 68.218ms
epoch: 3, step: 1562, loss is loss:2.307327, epoch time: 106474.296ms, per step time: 68.165ms
epoch: 4, step: 1562, loss is loss:2.308245, epoch time: 106434.503ms, per step time: 68.140ms
epoch: 5, step: 1562, loss is loss:2.309555, epoch time: 108063.124ms, per step time: 69.183ms
```

## [Evaluation Process](#contents)

### Usage

#### Running on Ascend

```bash
# evaluation
Usage: bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Evaluating LeNet with CIFAR-10 dataset

```bash
result: {'acc': 59.5400 %} ckpt=~/snn/train/output/checkpoint/lenet-5_250.ckpt
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### LeNet on CIFAR-10

| Parameters                 | Ascend 910                                                   |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | LeNet                                                        |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |
| uploaded Date              | 06/30/2022 (month/day/year)                                  |
| MindSpore Version          | 1.8.0                                                        |
| Dataset                    | CIFAR-10                                                     |
| Training Parameters        | epoch=5, steps per epoch=250, batch_size = 200               |
| Optimizer                  | Adam                                                         |
| Loss Function              | MSE                                                          |
| outputs                    | probability                                                  |
| Loss                       | 0.072030                                                     |
| Speed                      | 241 ms/step（1pcs)                                           |
| Total time                 | 7 mins                                                       |
| Checkpoint for Fine tuning | 1.3M (.ckpt file)                                            |
| Accuracy                   | 59.54%                                                       |
| config                     | [Link](https://gitee.com/mindspore/models/tree/master/community/cv/snn/config)|

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

