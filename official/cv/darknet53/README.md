# Contents

- [Contents](#contents)
- [Darknet53 Description](#darknet53-description)
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
            - [Running on GPU](#running-on-gpu)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
            - [Darknet53 on ImageNet2012](#Darknet53-on-imagenet2012)
        - [Inference Performance](#inference-performance)
            - [Darknet53 on ImageNet2012](#Darknet53-on-imagenet2012-1)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Darknet53 Description](#contents)

## Description

"Darknet53" is a backbone for Yolov3 model.

## Paper

[paper](https://arxiv.org/abs/1804.02767):Joseph Redmon, Ali Farhadi. "YOLOv3: An Incremental Improvement"

# [Model Architecture](#contents)

The overall network architecture of Net is show below:
[Link](https://arxiv.org/abs/1804.02767)

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/experts/en/master/others/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on GPU

```bash
# distributed training
Usage:
bash run_distribute_train_gpu.sh darknet53 imagenet2012 [DATASET_PATH]

# standalone training
Usage:
bash run_standalone_train_gpu.sh darknet53 imagenet2012 [DATASET_PATH]

# run evaluation example
Usage:
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└──darknet53
  ├── README.md
  ├── scripts
    ├── run_distribute_train_gpu.sh        # launch gpu distributed training(8 pcs)
    ├── run_eval_gpu.sh                    # launch gpu evaluation
    └── run_standalone_train_gpu.sh        # launch gpu standalone training(1 pcs)
  ├── src
    ├── config.py                          # parameter configuration
    ├── CrossEntropySmooth.py              # loss definition for ImageNet2012 dataset
    ├── dataset.py                         # data preprocessing
    ├── lr_generator.py                    # generate learning rate for each step
    ├── callbacks.py                       # callbacks for training
    └── darknet53.py                       # darknet53
  ├── export.py                            # export model for inference
  ├── eval.py                              # eval net
  └── train.py                             # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in imagenet_config.yaml

- Config for Darknet53, ImageNet2012 dataset

```bash
device_target: "GPU"
run_distributed: True
device_num: ""
net: "darknet53"
dataset: "imagenet2012"
dataset_path: ""
pre_trained: ""
class_num: 1000
epoch_num: 150
pretrained_epoch_num: 0
batch_size: 256
momentum: 0.9
loss_scale: 1024
weight_decay: 0.0001
use_label_smooth: True
label_smooth_factor: 0.1
lr: 0.1
lr_scheduler: "cosine_annealing"
eta_min: 0
T_max: 150
warmup_epochs: 5
checkpoint_path: ""
save_checkpoint: True
save_checkpoint_epochs: 5
keep_checkpoint_max: 90
save_checkpoint_path: "./"
summary_dir: "./summary_dir"
```

## [Training Process](#contents)

### Usage

#### Mindinsight support

You can open http://127.0.0.1:8080 to view trainig process (training loss and validation accuracy). More info you can find in the link [MindInsight](https://www.mindspore.cn/mindinsight/en)

#### Running on GPU

```bash
# distributed training
Usage:
bash run_distribute_train_gpu.sh darknet53 imagenet2012 [DATASET_PATH]

# standalone training
Usage:
bash run_standalone_train_gpu.sh darknet53 imagenet2012 [DATASET_PATH]
```

Training result will be stored in the example path, whose folder name begins with "train" or "train_parallel". Under this, you can find checkpoint file together with result like the following in log.

## [Evaluation Process](#contents)

### Usage

#### Running on GPU

```bash
bash run_eval_gpu.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

### Result

- Evaluating Darknet53 with ImageNet2012 dataset

```bash
result: {'top_5_accuracy': 0.9247195512820513, 'top_1_accuracy': 0.7572916666666667}
```

## [Inference Process](#contents)

### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --batch_size [BATCH_SIZE] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]
`BATCH_SIZE` current batch_size can only be set to 1.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### Darknet53 on ImageNet2012

| Parameter | Darknet53 |
| -------------------------- | ---------------------------------------------------------- |
| Resource | GPU: 8xRTX3090 24G; CPU: Intel(R) Xeon(R) Gold 6226R; RAM: 252G |
| Upload date | 2021-11-03 |
| MindSpore version | 1.3.0 |
| Dataset | ImageNet |
| Training parameters | imagenet_config.yaml |
| Optimizer | Momentum |
| Loss function | SoftmaxCrossEntropy |
| Output | ckpt file |
| Final loss | 1.33 |
| Top1 Accuracy |75.7%|
| Total time | 135h |
| Parameters (M) | batch_size=256, epoch=150 |

### Inference Performance

#### Darknet53 on ImageNet2012

| Parameter | Darknet53 |
| -------------------------- | ----------------------------- |
| Resource |  GPU: 8xRTX3090 24G; CPU: Intel(R) Xeon(R) Gold 6226R; RAM: 252G  |
| Upload date | 2021-11-03 |
| MindSpore version | 1.3.0 |
| Dataset | ImageNet |
| Batch size | 256（1card） |
| Output | Probability |
| Accuracy | ACC1[75.7%] ACC5[92.5%]|
| Total time | 4m |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
