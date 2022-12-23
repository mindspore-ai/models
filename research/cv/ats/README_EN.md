
# Contents

<!-- TOC -->

- [Contents](#Contents)
- [Description](#Description)
    - [Paper](#Paper)
    - [Brief Description](#Brief-Description)
- [Model Architecture](#Model-Architecture)
- [Dataset](#Dataset)
    - [Dataset Description](#Dataset-Description)
    - [Download Link](#Download-Link)
    - [Dataset Contents](#Dataset-Contents)
- [Environmental Requirements](#Environmental-Requirements)
- [Quick Start](#Quick-Start)
- [Script Description](#Script-Description)
    - [Script Architecture Description](#Script-Architecture-Description)
- [Hyper-Parameters](#Hyper-Parameters)
    - [Network Training Hyper-Parameters](#Network-Training-Hyper-Parameters)
    - [KD Hyper-Parameters](#KD-Hyper-Parameters)
- [Training Procedure](#Training-Procedure)
    - [Usage](#Usage)
    - [Result](#Result)
- [KD Procedure](#KD-Procedure)
    - [Usage](#Usage)
    - [Result](#Result)
- [Random Seed Description](#Random-Seed-Description)
- [ModelZoo Homepage](#ModelZoo-Homepage)

<!-- /TOC -->

# Description

## Paper

NeurIPS 2022 Paper: Asymmetric Temperature Scaling Makes Larger Networks Teach Well Again

## Brief Description

Knowledge distillation could transfer the knowledge of a well-performed teacher to a weaker student. However, a larger teacher may not teach better students. This is counter-intuitive. We point out that the over-confidence of the larger teacher could provide less discriminative information among wrong classes. We propose Asymmetric Temperature Scaling (ATS) that applies different temperatures to correct and wrong classes' logits.

# Model Architecture

- Teacher networks are ResNet14/ResNet110. Student network is VGG8. Dataset is CIFAR100.
- Using Temperature Scaling, i.e., softmax with tp_tau=t_tau, the performance of ResNet110->VGG8 is lower than ResNet14->VGG8.
- Using Asymmetric Temperature Scaling (ATS), i.e., softmax with tp_tau>t_tau, could improve the performance of ResNet110->VGG8.

# Dataset

## Dataset Description

CIFAR100 is a standard image classification benchmark, containing 5w training samples and 1w test samples. Each image is 32x32 with RGB channels.

## Download Link

[https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

## Dataset Contents

- --cifar-100-python
- |--train
- |--test
- |--meta
- |--file.txt~

# Environmental Requirements

- Windows 10
- CPU
- python3.7.3
- anaconda: Anaconda3-2020.02-Windows-x86
- python other pkgs: 'requirements.txt'
- Console emulator on Windows：cmder, the download link is [cmder](https://cmder.app/)

# Quick Start

In the directory of `scripts`, open the `cmder` and run the following scripts, which contain the command lines like `python xxx.py`:

- train teacher networks: `bash run_teacher.sh`
- train student networks: `bash run_student.sh`
- train student with KD+TS: `bash run_ts.sh`
- train student with KD+ATS: `bash run_ats.sh`
- Export MindIR models: `bash run_export.sh`

# Script Description

## Script Architecture Description

```text
└──ats
  ├── README.md
  ├── scripts
    ├── run_teacher.sh                     # train teacher networks
    ├── run_student.sh                     # train student networks
    ├── run_ts.sh                          # KD+TS
    ├── run_ts.sh                          # KD+ATS
    └── run_export.sh                      # MindIR models
  ├── src
    ├── data.py                            # load data
    ├── vgg.py                             # VGG networks
    ├── resnet.py                          # ResNet networks
    ├── classify.py                        # class of Classify
    ├── distill.py                         # class of Distill
    └── utils.py                           # helper functions
  ├── run_classify.py                      # main of Classify
  ├── run_distill.py                       # main of Distill
  ├── requirements.txt                     # other python pkgs
  └── export.py                            # MindIR models
```

# Hyper-Parameters

## Network Training Hyper-Parameters

```python run_classify.py
"dataset": 'cifar100',                     # dataset name
"data_dir": './data',                      # dataset dir
"download": True,                          # download data or not
"n_classes": 100,                          # number of classes
"net": 'VGG',                              # network type
"n_layer": 8,                              # depth of network
"net_name": 'VGG8',                        # name of network
"epoches": 240,                            # training epoches
"lr": 0.03,                                # learning rate
"momentum": 0.9,                           # momentum of SGD
"batch_size": 128,                         # batch size
"log_dir": './logs',                       # log save dir
"log_name": 'student-nokd.log',            # log save filename
"ckpt_dir": './ckpts',                     # model save dir
"ckpt_name": 'cifar100-VGG8.ckpt',         # model save filename
```

## KD Hyper-Parameters

```python run_classify.py
"dataset": 'cifar100',                     # dataset name
"data_dir": './data',                      # dataset dir
"download": True,                          # download data or not
"n_classes": 100,                          # number of classes
"t_net": 'ResNet',                         # teacher network type
"t_n_layer": 14,                           # teacher network depth
"t_net_name": 'ResNet14',                  # teacher network name
"net": 'VGG',                              # student network type
"n_layer": 8,                              # student network depth
"net_name": 'VGG8',                        # student network name
"kd_way": "ATS",                           # KD algorithm: TS or ATS
"tp_tau": 4.0,                             # temperature factor of teacher's correct class
"t_tau": 2.0,                              # temperature factor of teacher's wrong classes
"s_tau": 1.0,                              # temperature factor of student
"lamb": 0.5,                               # balance factor of student's loss
"epoches": 240,                            # training epoches
"lr": 0.03,                                # learning rate
"momentum": 0.9,                           # momentum of SGD
"batch_size": 128,                         # batch size
"log_dir": './logs',                       # log save dir
"log_name": 'student-kd-ats.log',          # log save filename
"ckpt_dir": './ckpts',                     # model save dir
"ckpt_name": 'cifar100-RES14-VGG8.ckpt',   # model save filename
```

# Training Procedure

## Usage

```Shell
echo "Train teacher network ResNet14 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 14 --net_name ResNet14 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet14.ckpt

echo "Train teacher network ResNet110 on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_classify.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --net ResNet --n_layer 110 --net_name ResNet110 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name teacher.log --ckpt_name cifar100-ResNet110.ckpt
```

## Result

```text
[Epoch:1] [Loss:4.603412429491678] [TrAcc:0.011067708333333334] [TeAcc:0.0107421875]
[Epoch:2] [Loss:4.6048067808151245] [TrAcc:0.010323660714285714] [TeAcc:0.008634868421052632]
...
```

# KD Procedure

## Usage

```Shell
echo "Distill konwledge of teacher network ResNet14 to student network VGG8 using TS (tp_tau = t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 14 --t_net_name ResNet14 --net VGG --n_layer 8 --net_name VGG8 --kd_way TS --lamb 0.5 --tp_tau 4.0 --t_tau 4.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet14-VGG8-TS.ckpt

echo "Distill konwledge of teacher network ResNet110 to student network VGG8 using TS (tp_tau = t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 110 --t_net_name ResNet110 --net VGG --n_layer 8 --net_name VGG8 --kd_way TS --lamb 0.5 --tp_tau 4.0 --t_tau 4.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet110-VGG8-TS.ckpt

echo "Distill konwledge of teacher network ResNet14 to student network VGG8 using ATS (tp_tau > t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 14 --t_net_name ResNet14 --net VGG --n_layer 8 --net_name VGG8 --kd_way ATS --lamb 0.5 --tp_tau 4.0 --t_tau 2.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet14-VGG8-ATS.ckpt

echo "Distill konwledge of teacher network ResNet110 to student network VGG8 using ATS (tp_tau > t_tau) on CIFAR-100 (epoches=2 for demo, change to 240 for training)"
python run_distill.py --dataset cifar100 --data_dir ./data --download True --n_classes 100 --t_net ResNet --t_n_layer 110 --t_net_name ResNet110 --net VGG --n_layer 8 --net_name VGG8 --kd_way ATS --lamb 0.5 --tp_tau 4.0 --t_tau 2.0 --s_tau 1.0 --epoches $EPOCHES --lr 0.03 --momentum 0.9 --batch_size 128 --ckpt_dir ./ckpts --log_dir ./logs --log_name student-kd-ts.log --ckpt_name cifar100-ResNet110-VGG8-ATS.ckpt
```

## Result

```text
[Epoch:1] [Loss:4.607254505157471] [TrAcc:0.008831521739130434] [TeAcc:0.021577380952380952]
[Epoch:2] [Loss:4.593390011787415] [TrAcc:0.03515625] [TeAcc:0.0390625]
...
```

# Random Seed Description

Random seeds can be set. Without random seeds, the standard deviation of convergence performance of the final model is about 0.5%.

# ModelZoo Homepage

Please see ModelZoo [Homepage](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo)。
