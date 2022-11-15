# Contents

- [Contents](#contents)
- [PCB Description](#pcb-description)
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
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on Ascend](#running-on-ascend-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [Result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [PCB on Market-1501](#pcb-on-market-1501)
            - [PCB on DukeMTMC-reID](#pcb-on-dukemtmc-reid)
            - [PCB on CUHK03](#pcb-on-cuhk03)
            - [PCB-RPP on Market-1501](#pcb-rpp-on-market-1501)
            - [PCB-RPP on DukeMTMC-reID](#pcb-rpp-on-dukemtmc-reid)
            - [PCB-RPP on CUHK03](#pcb-rpp-on-cuhk03)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# PCB Description

## Description

PCB (Part-level Convolutional Baseline) is a classic model for Person Re-Identification. Given an image input, it outputs a convolutional descriptor consisting of several part-level features. Moreover, a refined part pooling (RPP) method is proposed to re-assign the outliers incurred by the uniform partition strategy to the parts they are closest to, resulting in refined parts
with enhanced within-part consistency.

## Paper

1.[paper](https://arxiv.org/pdf/1711.09349.pdf)：Yifan Sun, Liang Zheng, Yi Yang, Qi Tian, Shengjin Wang."Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)"

# Model Architecture

The overall network architecture of PCB is shown below:
[Link](https://arxiv.org/pdf/1711.09349.pdf)

# Dataset

Dataset used: [Market-1501](<http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html>)
[Download Link](https://pan.baidu.com/s/1qWEcLFQ?_at_=1640837580475)

- Dataset size：
    - Training Set：12936 RGB images containing 751 pedestrians
    - Test Set：

        -query set: 3368 RGB images containing 750 pedestrians

        -gallery set：15913 RGB images containing 751 pedestrians

- Data format：PNG
    - Note：Data will be processed in src/datasets/market.py
- Download the dataset, the directory structure is as follows:

```text
├─ Market-1501
 │
 ├─bounding_box_test
 │
 └─bounding_box_train
 │
 └─gt_bbx
 │
 └─gt_query
 │
 └─query
```

Dataset used: [DukeMTMC-reID](http://vision.cs.duke.edu/DukeMTMC/)

- Dataset size：
    - training set：16522 RGB images containing 702 pedestrians
    - test set：

        -query set: 2228 RGB images containing 702 pedestrians

        -gallery set：17661 RGB images containing 1110 pedestrians

- Data format：PNG
    - Note：Data will be processed in src/datasets/duke.py
- Download the dataset, the directory structure is as follows:

```text
├─ DukeMTMC-reID
 │
 ├─bounding_box_test
 │
 └─bounding_box_train
 │
 └─query
```

Dataset used：[CUHK03](<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>)
[Download Link](https://pan.baidu.com/s/1o8txURK)

The paper adopts [the new training/testing protocol](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
The new protocol splits the CUHK03 dataset into training set and testing set similar to that of Market-1501
[Download link of the new training/testing protocol split for CUHK03](https://drive.google.com/file/d/0B7TOZKXmIjU3OUhfd3BPaVRHZVE/view?resourcekey=0-hU4gyE6hFsBgizIh9DFqtA)

- Dataset size of CUHK03 in the new protocol
    - Training set：7365 RGB images containing 767 pedestrians
    - Test set:

        -query set: 1400 RGB images containing 700 pedestrians

        -gallery set：5332 RGB images containing 700 pedestrians

- Data format：PNG
    - Note：Data will be processed in src/datasets/cuhk03.py

- Note: After downloading the CUHK03 dataset and the new training/testing protocol split, please organize the cuhk03.mat (in original dataset) and cuhk03_new_protocol_config_detected.mat (in the new training/testing protocol split) as follows.

```text
├─ CUHK03
 │
 ├─ cuhk03.mat
 │
 └─ cuhk03_new_protocol_config_detected.mat
```

- Pretrained Resnet50 checkpoint file [Download Link](https://gitee.com/starseekerX/PCB_pretrained_checkpoint/blob/master/pretrained_resnet50.ckpt)

# Features

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/docs/programming_guide/en/r1.6/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# Environment Requirements

- Hardware（Ascend）
    - Prepare hardware environment with Ascend
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# Quick Start

After installing MindSpore via the official website, you can start training and evaluation as follows:

> - <font size=2>During training, evaluating and inferring, if Market-1501 dataset is used, DATASET_PATH={Market-1501 directory};if DukeMTMC-reID dataset is used, DATASET_PATH={DukeMTMC-reID directory};if CUHK03 dataset is used, DATASET_PATH={CUHK03 directory} </font>

- Running on Ascend

```bash
# standalone training
# Usage:
bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# Example:
bash run_standalone_train.sh PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# Distributed training
# Usage:
bash run_distribute_train.sh [RANK_TABLE_FILE] [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)

# Example:
bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# Evaluation
# Usage:
bash run_eval.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]

# Example:
bash run_eval.sh PCB market ../../Datasets/Market-1501 ../config/eval_PCB_market.yaml ./output/checkpoint/PCB/market/train/PCB-60_202.ckpt True
```

# Script Description

## Script and Sample Code

```text
.
└──PCB
  ├── README.md
  ├── config                               # parameter
    ├── base_config.yaml
    ├── train_PCB_market.yaml
    ├── train_PCB_duke.yaml
    ├── train_PCB_cuhk03
      ├── train_PCB.yaml
      ├── finetune_PCB.yaml
    ├── train_RPP_market
      ├── train_PCB.yaml
      ├── train_RPP.yaml
    ├── train_RPP_duke
      ├── train_PCB.yaml
      ├── train_RPP.yaml
    ├── train_RPP_cuhk03
      ├── train_PCB.yaml
      ├── train_RPP.yaml
      ├── finetune_RPP.yaml
    ├── eval_PCB_market.yaml
    ├── eval_PCB_duke.yaml
    ├── eval_PCB_cuhk03.yaml
    ├── eval_RPP_market.yaml
    ├── eval_RPP_duke.yaml
    ├── eval_RPP_cuhk03.yaml
    ├── infer_310_config.yaml
  ├── scripts
    ├── run_standalone_train.sh            # launch standalone training
    ├── run_distribute_eval.sh             # launch distributed training
    ├── run_eval.sh                        # launch evaluation
    ├── run_infer_310.sh                   # launch 310 inference
  ├── src
    ├── dataset.py                         # dataset preprocessing
    ├── eval_callback.py                   # evaluation callback while training
    ├── eval_utils.py                      # evaluation utility function (calculate CMC, mAP)
    ├── meter.py
    ├── logging.py
    ├── lr_generator.py                    # generate learning rate for each step
    ├── pcb.py                             # PCB
    ├── rpp.py                             # PCB+RPP
    ├── resnet.py                          # resnet50 backbone
    ├── datasets                           # Utility function for processing datasets
       ├── market.py
       ├── duke.py
       ├── cuhk03.py
    ├── model_utils
       ├── config.py                       # parameter configuration
       ├── device_adapter.py               # device adapter
       ├── local_adapter.py                # local adapter
       └── moxing_adapter.py               # moxing adapter
  ├── eval.py                              # eval net
  └── train.py                             # train net
  └── export.py                            # export model for inference
  └── preprocess.py                        # preprocess input data for 310 inference
  └── postprocess.py                       # postprocess result from 310 inference and calculate metrics
```

## Script Parameters

Parameters for both training, evaluation, model exportation and inference can be set in config file.

- Config for training PCB on Market-1501 dataset

```text
enable_modelarts: False                    # whether to use modelarts or not
data_url: ""
train_url: ""
checkpoint_url: ""
run_distribute: False                      # whether to run distributed training or not
enable_profiling: False
dataset_path: "/cache/dataset/"            # dataset path
output_path: "/cache/output/"              # output path
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
log_save_path: "./log/PCB/market/train"    # path for saving log
checkpoint_save_path: "./checkpoint/PCB/market/train"   # path for saving checkpoint
checkpoint_file_path: "/cache/load_checkpoint/pretrained_resnet50.ckpt"   # path for loading checkpoint

mindrecord_dir: "./MindRecord"             # path for saving MindRecord file
dataset_name: "market"                     # short name of the dataset
batch_size: 64                             # batch size of input tensor
num_parallel_workers: 4
device_num: 1                              # number of the available devices

model_name: "PCB"                          # short name of the model
learning_rate: 0.1                         # learning rate
lr_mult: 0.1                               # multiple for controlling the backbone learning rate
decay_rate: 0.1                            # decay rate of the learning rate
momentum: 0.9                              # momentum of SGD
weight_decay: 5e-4                         # weight decay of SGD
nesterov: True

mode_name: "GRAPH"                         # whether to use GRAPTH mode or use PYNATIVE mode
sink_mode: True                            # whether to start data sink mode
seed: 37                                   # random seed
epoch_size: 60                             # epoch size for training
decay_epoch_size: 40                       # interval size for learning rate decay
warmup_epoch_size: 1                       # epoch size for warmup

save_checkpoint: True                      # whether to save checkpoint
save_checkpoint_epochs: 60                 # epoch size for saving checkpoint
keep_checkpoint_max: 15                    # maximum number of the saved checkpoints

run_eval: False                            # whether to run evaluation during training
eval_interval: 15                          # evaluation interval size
eval_start_epoch: 60                       # start epoch for evaluation
use_G_feature: True                        # whether to use G feature or use H feature for evaluation
```

- Config for evaluating PCB on Market-1501 dataset

```text
enable_modelarts: False
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"
output_path: "/cache/output/"
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
log_save_path: "./log/PCB/market/eval"
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"

mindrecord_dir: "./MindRecord"
dataset_name: "market"
batch_size: 64                             # batch size of the input tensor
num_parallel_workers: 4

model_name: "PCB"
use_G_feature: True                        # whether to use G feature or use H feature for evaluation
```

- Config for exporting PCB and running 310 inference

```text
enable_modelarts: False
data_url: ""
train_url: ""
checkpoint_url: ""
enable_profiling: False
dataset_path: "/cache/dataset/"
output_path: "/cache/output/"
load_path: "/cache/load_checkpoint/"
device_target: "Ascend"
checkpoint_file_path: "/cache/load_checkpoint/PCB-60_202.ckpt"
batch_size: 1                              # Currently, only support batch size==1
model_name: "PCB"
use_G_feature: True                        # whether to use G feature or use H feature for evaluation

device_id: 0
image_height: 384                          # height of the input image
image_width: 128                           # width of the input image
file_name: "export_PCB_market_G"           # name of the exported model
file_format: "MINDIR"                      # file format of the exported model

preprocess_result_path: "./preprocess_Result"  #path for saving preprocess result

query_prediction_path: "./query_result_files"  #path for saving inference result of the query set
gallery_prediction_path: "./gallery_result_files"  #path for saving inference result of the gallery set
```

## Training Process

### Usage

#### Running on Ascend

```bash
# Standalone training
# Usage:
bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH] (optional)
# MODEL_NAME should be in ['PCB', 'RPP'], DATASET_NAME should be in ['market', 'duke', 'cuhk03'].

# Examples:

# 1. Training PCB on Market-1501

bash run_standalone_train.sh PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 2. Training PCB on DukeMTMC-reID

bash run_standalone_train.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/train_PCB_duke.yaml ../../pretrained_resnet50.ckpt

# 3. Training PCB on CUHK03

bash run_standalone_train.sh PCB cuhk03 ../../Datasets/CUHK03 ../config/train_PCB_cuhk03 ../../pretrained_resnet50.ckpt

# 4. Training PCB+RPP on Market-1501

bash run_standalone_train.sh RPP market ../../Datasets/Market-1501 ../config/train_RPP_market ../../pretrained_resnet50.ckpt

# 5. Training PCB+RPP on DukeMTMC-reID

bash run_standalone_train.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/train_RPP_duke ../../pretrained_resnet50.ckpt

# 6. Training PCB+RPP on CUHK03

bash run_standalone_train.sh RPP cuhk03 ../../Datasets/CUHK03 ../config/train_RPP_cuhk03 ../../pretrained_resnet50.ckpt


# Distributed training
# Usage:
bash run_distribute_train.sh [RANK_TABLE_FILE] [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH] (optional)
# MODEL_NAME should be in ['PCB', 'RPP'], DATASET_NAME should be in ['market', 'duke', 'cuhk03'].

# Examples:

# 1. Training PCB on Market-1501

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB market ../../Datasets/Market-1501 ../config/train_PCB_market.yaml ../../pretrained_resnet50.ckpt

# 2. Training PCB on DukeMTMC-reID

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB duke ../../Datasets/DukeMTMC-reID ../config/train_PCB_duke.yaml ../../pretrained_resnet50.ckpt

# 3. Training PCB on CUHK03

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json PCB cuhk03 ../../Datasets/CUHK03 ../config/train_PCB_cuhk03 ../../pretrained_resnet50.ckpt

# 4. Training PCB+RPP on Market-1501

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP market ../../Datasets/Market-1501 ../config/train_RPP_market ../../pretrained_resnet50.ckpt

# 5. Training PCB+RPP on DukeMTMC-reID

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP duke ../../Datasets/DukeMTMC-reID ../config/train_RPP_duke ../../pretrained_resnet50.ckpt

# 6. Training PCB+RPP on CUHK03

bash run_distribute_train.sh ../hccl_8p_01234567_127.0.0.1.json RPP cuhk03 ../../Datasets/CUHK03 ../config/train_RPP_cuhk03 ../../pretrained_resnet50.ckpt
```

For distributed training, a hccl configuration file with JSON format needs to be created in advance.

Please follow the instructions in the link [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

Training result will be stored in the "output" directory. specifically, the training log will be stored in "./output/log/{MODEL_NAME}/{DATASET_NAME}/train" and the checkpoints will be stored in "./output/checkpoint/{MODEL_NAME}/{DATASET_NAME}/train."

#### Modelarts Training Job

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start standalone training and distributed training as follows:

```text
# run standalone training on modelarts example
# (1) Add "config_path='/path_to_code/config/train_PCB_market.yaml'" on the website UI interface.
# (2) First, Perform a or b.
#       a. Set "enable_modelarts=True" in yaml file.
#          Set "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" in yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" in yaml file.
#          Set other parameters you need in yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
#          Add other parameters you need on the website UI interface.
# (3) Set the code directory to "/path/PCB" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run distributed training on modelarts example
# (1) Add "config_path='/path_to_code/config/train_PCB_market.yaml'" on the website UI interface.
# (2) First, Perform a or b.
#       a. Set "enable_modelarts=True" in yaml file.
#          Set "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" in yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" in yaml file.
#          Set "run_distribute=True" in yaml file.
#          Set "device_num = {number of the available devices}" in yaml file.
#          Set other parameters you need in yaml file.。
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
#          Add "run_distribute=True" on the website UI interface.
#          Add "device_num = {number of the available devices}" on the website UI interface.
#          Add other parameters you need on the website UI interface.
# (3) Set the code directory to "/path/PCB" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

### Result

- Training PCB on Market-1501

```log
# standalone training result
epoch: 1 step: 202, loss is 28.947758
epoch time: 88804.387 ms, per step time: 439.626 ms
epoch: 2 step: 202, loss is 18.160383
epoch time: 35282.132 ms, per step time: 174.664 ms
epoch: 3 step: 202, loss is 14.728483
epoch time: 35331.460 ms, per step time: 174.908 ms
...
```

- Training PCB on DukeMTMC-reID

```log
# standalone training result
epoch: 1 step: 258, loss is 23.912783
epoch time: 100480.371 ms, per step time: 389.459 ms
epoch: 2 step: 258, loss is 13.815624
epoch time: 33952.824 ms, per step time: 131.600 ms
epoch: 3 step: 258, loss is 9.111069
epoch time: 33952.491 ms, per step time: 131.599 ms
...
```

- Training PCB on CUHK03

```log
# standalone training result
epoch: 1 step: 115, loss is 34.977722
epoch time: 87867.500 ms, per step time: 764.065 ms
epoch: 2 step: 115, loss is 24.710325
epoch time: 15645.867 ms, per step time: 136.051 ms
epoch: 3 step: 115, loss is 16.847214
epoch time: 15694.620 ms, per step time: 136.475 ms
...
```

- Training PCB+RPP on Market-1501

```log
# standalone training result
epoch: 1 step: 202, loss is 28.807777
epoch time: 90390.587 ms, per step time: 447.478 ms
epoch: 2 step: 202, loss is 18.29936
epoch time: 35274.752 ms, per step time: 174.627 ms
epoch: 3 step: 202, loss is 14.982595
epoch time: 35277.650 ms, per step time: 174.642 ms
...
```

- Training PCB+RPP on DukeMTMC-reID

```log
# standalone training result
epoch: 1 step: 258, loss is 23.096334
epoch time: 96244.296 ms, per step time: 373.040 ms
epoch: 2 step: 258, loss is 13.114418
epoch time: 33972.328 ms, per step time: 131.676 ms
epoch: 3 step: 258, loss is 8.97956
epoch time: 33965.507 ms, per step time: 131.649 ms
...
```

- Training PCB+RPP on CUHK03

```log
# standalone training result
epoch: 1 step: 115, loss is 37.5888
epoch time: 68445.567 ms, per step time: 595.179 ms
epoch: 2 step: 115, loss is 26.582499
epoch time: 15640.461 ms, per step time: 136.004 ms
epoch: 3 step: 115, loss is 17.900295
epoch time: 15637.023 ms, per step time: 135.974 ms
...
```

## Evaluation Process

### Usage

#### Running on Ascend

```bash
# Usage:
bash run_eval.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [CHECKPOINT_PATH] [USE_G_FEATURE]
# MODEL_NAME should be in ['PCB', 'RPP'], DATASET_NAME should be in ['market', 'duke', 'cuhk03']. USE_G_FEATURE is a boolean parameter. If USE_G_FEATURE==True, G feature will be used for evaluation. If USE_G_FEATURE==False, H feature will be used for evaluation.

# Examples:

# 1. Evaluating PCB on Market-1501 using G feature

bash run_eval.sh PCB market ../../Datasets/Market-1501 ../config/eval_PCB_market.yaml ./output/checkpoint/PCB/market/train/PCB-60_202.ckpt True

# 2. Evaluating PCB on DukeMTMC-reID using G feature

bash run_eval.sh PCB duke ../../Datasets/DukeMTMC-reID ../config/eval_PCB_duke.yaml ./output/checkpoint/PCB/duke/train/PCB-60_258.ckpt True

# 3. Evaluating PCB on CUHK03 using G feature

bash run_eval.sh PCB cuhk03 ../../Datasets/CUHK03 ../config/eval_PCB_cuhk03.yaml ./output/checkpoint/PCB/cuhk03/train/PCB_1-45_115.ckpt True

# 4. Evaluating PCB+RPP on Market-1501 using G feature

bash run_eval.sh RPP market ../../Datasets/Market-1501 ../config/eval_RPP_market.yaml ./output/checkpoint/RPP/market/train/RPP_1-10_202.ckpt True

# 5. Evaluating PCB+RPP on DukeMTMC-reID using G feature

bash run_eval.sh RPP duke ../../Datasets/DukeMTMC-reID ../config/eval_RPP_duke.yaml ./output/checkpoint/RPP/duke/train/RPP-40_258.ckpt True

# 6. Evaluating PCB+RPP on CUHK03 using G feature

bash run_eval.sh RPP cuhk03 ../../Datasets/CUHK03 ../config/eval_RPP_cuhk03.yaml ./output/checkpoint/RPP/cuhk03/train/RPP_1-10_115.ckpt True
```

Evaluation results will be stored in "output/log/{MODEL_NAME}/{DATASET_NAME}/eval". You can get metrics info in the log.txt.

#### Modelarts Training Job

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start standalone training and distributed training as follows:

```text
# run evaluation on modelarts example
# (1) Add "config_path='/path_to_code/config/eval_PCB_market.yaml'" on the website UI interface.
# (2) Copy or upload your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" in yaml file.
#          Set "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" in yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" in yaml file.
#          Set "use_G_feature = True (False)" in yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
#          Add "use_G_feature = True (False)" on the website UI interface.
# (4) Set the code directory to "/path/PCB" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

### Result

- Evaluating PCB on Market-1501 using G feature

```log
Mean AP: 78.5%
CMC Scores      market
  top-1          93.0%
  top-5          97.4%
  top-10         98.4%
```

- Evaluating PCB on Market-1501 using H feature

```log
Mean AP: 77.9%
CMC Scores      market
  top-1          93.0%
  top-5          97.1%
  top-10         98.1%
```

- Evaluating PCB on DukeMTMC-reID using G feature

```log
Mean AP: 69.8%
CMC Scores        duke
  top-1          84.2%
  top-5          92.4%
  top-10         94.1%
```

- Evaluating PCB on DukeMTMC-reID using H feature

```log
Mean AP: 68.6%
CMC Scores        duke
  top-1          84.2%
  top-5          91.6%
  top-10         93.9%
```

- Evaluating PCB on CUHK03 using G feature

```log
Mean AP: 55.1%
CMC Scores      cuhk03
  top-1          61.1%
  top-5          79.5%
  top-10         86.1%
```

- Evaluating PCB on CUHK03 using H feature

```log
Mean AP: 55.4%
CMC Scores      cuhk03
  top-1          61.9%
  top-5          79.9%
  top-10         85.9%
```

- Evaluating PCB+RPP on Market-1501 using G feature

```log
Mean AP: 81.7%
CMC Scores      market
  top-1          93.8%
  top-5          97.5%
  top-10         98.6%
```

- Evaluating PCB+RPP on Market-1501 using H feature

```log
Mean AP: 81.0%
CMC Scores      market
  top-1          93.6%
  top-5          97.3%
  top-10         98.5%
```

- Evaluating PCB+RPP on DukeMTMC-reID using G feature

```log
Mean AP: 71.4%
CMC Scores        duke
  top-1          85.0%
  top-5          92.6%
  top-10         94.4%
```

- Evaluating PCB+RPP on DukeMTMC-reID using H feature

```log
Mean AP: 70.2%
CMC Scores        duke
  top-1          85.1%
  top-5          92.1%
  top-10         94.0%
```

- Evaluating PCB+RPP on CUHK03 using G feature

```log
Mean AP: 58.6%
CMC Scores      cuhk03
  top-1          63.9%
  top-5          81.2%
  top-10         87.1%
```

- Evaluating PCB+RPP on CUHK03 using H feature

```log
Mean AP: 59.1%
CMC Scores      cuhk03
  top-1          65.0%
  top-5          81.4%
  top-10         86.9%
```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

Export MindIR on local

```shell
python export.py --model_name [MODEL_NAME] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --checkpoint_file_path [CKPT_PATH] --use_G_feature [USE_G_FEATURE] --config_path [CONFIG_PATH]
```

model_name should be in ["PCB", "RPP"].
file_name refers to the name of the exported model.
file_format only supports "MINDIR".
checkpoint_file_path refers to the checkpoint to load.
USE_G_FEATURE is a boolean parameter. If USE_G_FEATURE==True, G feature will be used for evaluation. If USE_G_FEATURE==False, H feature will be used for evaluation. Different feature types will lead to different architectures of the exported model.
config_path refers to the path of infer_310_config.yaml.

```shell
# Example:
# 1. Export PCB trained on Market-1501 using G feature
python export.py --model_name "PCB" --file_name "PCB_market_G" --file_format MINDIR --checkpoint_file_path ../PCB_market.ckpt --use_G_feature True --config_path ./config/infer_310_config.yaml
```

Export mindir on ModelArts

```text
# (1) Add "config_path='/path_to_code/config/infer_310_config.yaml'" on the website UI interface.
# (2) Upload or copy your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" in infer_310_config.yaml
#          Set "checkpoint_file_path='/cache/load_checkpoint/model.ckpt" in infer_310_config.yaml
#          Set "checkpoint_url=/The path of checkpoint in S3/" in infer_310_config.yaml
#          Set "model_name='PCB'" in infer_310_config.yaml
#          Set "file_name='PCB_market_G'" in infer_310_config.yaml
#          Set "file_format='MINDIR'" in infer_310_config.yaml
#          Set "use_G_feature=True" in infer_310_config.yaml
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/load_checkpoint/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
#          Add "model_name='PCB'" on the website UI interface.
#          Add "file_name='PCB_market_G'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add "use_G_feature=True" on the website UI interface.
# (4) Set the code directory to "/path/PCB" on the website UI interface.
# (5) Set the startup file to "export.py" on the website UI interface.
# (6) Set "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

### Infer on Ascend310

Before performing inference, the mindir file must bu exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size can only be set to 1.

```bash
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [USE_G_FEATURE][CONFIG_PATH] [DEVICE_ID](optional)
```

- `DATASET_NAME` should be in [market, duke, cuhk03].
- `USE_G_FEATURE` should be consistent with the export setting.
- `CONFIG_PATH` refers to the path of infer_310_config.yaml
- `DEVICE_ID` (optional), default 0.

```bash
# Example:
# 1. Evaluating PCB on Market-1501 with G feature
bash run_infer_310.sh  ../../mindir/PCB_market_G.mindir market ../../Datasets/Market-1501 True ../config/infer_310_config.yaml
```

### Result

- Evaluating PCB on Market-1501 with G feature

```log
Mean AP: 78.5%
  top-1          93.1%
  top-5          97.4%
  top-10         98.4%
```

- Evaluating PCB on Market-1501 with H feature

```log
Mean AP: 77.9%
  top-1          92.9%
  top-5          97.1%
  top-10         98.1%
```

- Evaluating PCB on DukeMTMC-reID with G feature

```log
Mean AP: 69.8%
  top-1          84.2%
  top-5          92.4%
  top-10         94.1%
```

- Evaluating PCB on DukeMTMC-reID with H feature

```log
Mean AP: 68.6%
  top-1          84.2%
  top-5          91.5%
  top-10         93.9%
```

- Evaluating PCB on CUHK03 with G feature

```log
Mean AP: 55.1%
  top-1          60.9%
  top-5          79.4%
  top-10         86.1%
```

- Evaluating PCB on CUHK03 with H feature

```log
Mean AP: 55.3%
  top-1          61.7%
  top-5          80.1%
  top-10         86.0%
```

Inference results of PCB+RPP are not available as Ascend 310 currently doesn't support the AvgPool3D operator.

# Model Description

## Performance

### Evaluation Performance

#### PCB on Market-1501

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | Market-1501
| Training parameters        | epoch=60, steps per epoch=202, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.05631405
| Speed                      | 175 ms/step（1p）
| Total time                 | 37 min
| Parameters(M)             | 27.2

#### PCB on DukeMTMC-reID

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | DukeMTMC-reID
| Training parameters        | epoch=60, steps per epoch=258, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.095855206
| Speed                      | 132 ms/step（1p）
| Total time                 | 36 min
| Parameters(M)             | 27.2

#### PCB on CUHK03

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | CUHK03
| Training parameters        | epoch=85, steps per epoch=115, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.094226934
| Speed                      | 137 ms/step（1p）
| Total time                 | 25 min
| Parameters(M)             | 27.2

#### PCB-RPP on Market-1501

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB-RPP
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | Market-1501
| Training parameters        | epoch=75, steps per epoch=202, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.04336106
| Speed                      | 307 ms/step（1p）
| Total time                 | 72 min
| Parameters(M)             | 27.2

#### PCB-RPP on DukeMTMC-reID

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB-RPP
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | DukeMTMC-reID
| Training parameters        | epoch=60, steps per epoch=258, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.03547495
| Speed                      | 264 ms/step（1p）
| Total time                 | 59 min
| Parameters(M)             | 27.2

#### PCB-RPP on CUHK03

| Parameter                 | Ascend 910
| -------------------------- | --------------------------------------
| Model              | PCB-RPP
| Resource                   | Ascend 910; CPU 2.60GHz, 24 cores; Memory 96G; OS Euler2.8
| MindSpore version          | 1.3.0
| Dataset                    | CUHK03
| Training parameters        | epoch=95, steps per epoch=115, batch_size = 64
| Optimizer                  | SGD
| Loss function              | Softmax Cross Entropy
| Output                    | Probability
| Loss                       | 0.083887264
| Speed                      | 268 ms/step（1p）
| Total time                 | 59 min
| Parameters(M)             | 27.2

# Description of Random Situation

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# ModelZoo Homepage

 Please check the official [homepage](https://gitee.com/mindspore/models).
