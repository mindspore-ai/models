# Contents

- [NTS-Net Description](#NTS-Net-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Knowledge Distillation Process](#knowledge-distillation-process)
    - [Prediction Process](#prediction-process)
    - [Evaluation with cityscape dataset](#evaluation-with-cityscape-dataset)
    - [Export MindIR](#export-mindir)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NTS-Net Description](#contents)

NTS-Net for Navigator-Teacher-Scrutinizer Network, consists of a Navigator agent, a Teacher agent and a Scrutinizer agent. In consideration of intrinsic consistency between informativeness of the regions and their probability being ground-truth class, NTS-Net designs a novel training paradigm, which enables Navigator to detect most informative regions under the guidance from Teacher. After that, the Scrutinizer scrutinizes the proposed regions from Navigator and makes predictions
[Paper](https://arxiv.org/abs/1809.00287): Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, and L. Wang, Learning to navigate for fine-grained classification, in Proceedings of the European Conference on Computer Vision (ECCV), 2018.

# [Model Architecture](#contents)

NTS-Net consists of a Navigator agent, a Teacher agent and a Scrutinizer agent. The Navigator navigates the model to focus on the most informative regions: for each region in the image, Navigator predicts how informative the region is, and the predictions are used to propose the most informative regions. The Teacher evaluates the regions proposed by Navigator and provides feedbacks: for each proposed region, the Teacher evaluates its probability belonging to ground-truth class; the confidence evaluations guide the Navigator to propose more informative regions with a novel ordering-consistent loss function. The Scrutinizer scrutinizes proposed regions from Navigator and makes fine-grained classifications: each proposed region is enlarged to the same size and the Scrutinizer extracts features therein; the features of regions and of the whole image are jointly processed to make fine-grained classifications.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [Caltech-UCSD Birds-200-2011](<http://www.vision.caltech.edu/datasets/cub_200_2011/>)

Please download the datasets [CUB_200_2011.tgz] and unzip it, then put all training images into a directory named "train", put all testing images into a directory named "test".

The directory structure is as follows, you need to split the dataset by yourself followed by "train_test_split.txt" in the original dataset:

```path
├─resnet50.ckpt
└─cub_200_2011
  ├─train
  └─test
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─ntsnet
  ├─README.md                             # README
  ├─scripts                               # shell script
    ├─run_standalone_train_ascend.sh             # training in standalone mode(1pcs)
    ├─run_distribute_train_ascend.sh             # training in parallel mode(8 pcs)
    └─run_eval_ascend.sh                         # evaluation
  ├─src
    ├─config_gpu.py                       # network configuration
    ├─dataset_gpu.py                      # dataset utils
    ├─lr_generator_gpu.py                 # leanring rate generator
    ├─config.py                           # network configuration
    ├─dataset.py                          # dataset utils
    ├─lr_generator.py                     # leanring rate generator
    ├─network.py                          # network define for ntsnet
    └─resnet.py                           # resnet.py
  ├─mindspore_hub_conf.py                 # mindspore hub interface
  ├─export.py                             # script to export MINDIR model
  ├─eval.py                               # evaluation scripts
  ├─train.py                              # training scripts
  ├─eval_gpu.py                               # evaluation scripts
  └─train_gpu.py                              # training scripts
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: bash run_train_ascend.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]

# standalone training
Usage: bash run_standalone_train_ascend.sh [DATA_URL] [TRAIN_URL]
```

### [Parameters Configuration](#contents)

```txt
"img_width": 448,           # width of the input images
"img_height": 448,          # height of the input images

# anchor
"size": [48, 96, 192],                                                  #anchor base size
"scale": [1, 2 ** (1. / 3.), 2 ** (2. / 3.)],                           #anchor base scale
"aspect_ratio": [0.667, 1, 1.5],                                        #anchor base aspect_ratio
"stride": [32, 64, 128],                                                #anchor base stride

# resnet
"resnet_block": [3, 4, 6, 3],                                            # block number in each layer
"resnet_in_channels": [64, 256, 512, 1024],                              # in channel size for each layer
"resnet_out_channels": [256, 512, 1024, 2048],                           # out channel size for each layer

# LR
"base_lr": 0.001,                                                              # base learning rate
"base_step": 58633,                                                            # bsae step in lr generator
"total_epoch": 200,                                                            # total epoch in lr generator
"warmup_step": 4,                                                              # warmp up step in lr generator
"sgd_momentum": 0.9,                                                           # momentum in optimizer

# train
"batch_size": 8, # 16 for gpu
"weight_decay": 1e-4,
"epoch_size": 200,                                                             # total epoch size
"save_checkpoint": True,                                                       # whether save checkpoint or not
"save_checkpoint_epochs": 1,                                                   # save checkpoint interval
"num_classes": 200,
"lr_scheduler": "cosine",                                                      # lr_scheduler, support cosine or step
"optimizer": "momentum"
```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate, output filename and network hyperparameters. Click [here](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.
- Get ResNet50 pretrained model from [Mindspore Hub](https://www.mindspore.cn/resources/hub/details?MindSpore/ascend/v1.2/resnet50_v1.2_imagenet2012)

### [Training](#content)

- Run `run_standalone_train_ascend.sh` for non-distributed training of NTS-Net model.

```bash
# standalone training in ascend
bash run_standalone_train_ascend.sh [DATA_URL] [TRAIN_URL] [DEVICE_ID(optional)]
```

- Run `run_standalone_train_gpu.sh` for non-distributed training of NTS-Net model in GPU.

```bash
# standalone training in gpu
bash run_standalone_train_gpu.sh [DATA_URL] [TRAIN_URL] [DEVICE_ID(optional)]
```

### [Distributed Training](#content)

- Run `run_distribute_train_ascend.sh` for distributed training of NTS-Net model in Ascend.

```bash
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]
```

- Run `run_distribute_train_gpu.sh` for distributed training of NTS-Net model in GPU.

```bash
bash run_distribute_train_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATA_URL] [TRAIN_URL]
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools).
2. As for PRETRAINED_MODEL, it should be a trained ResNet50 checkpoint, name the pretraied weight to resnet50.ckpt and put it in dataset directory. See [Training Process](#Training Process)

### [Training Result](#content)

Training result will be stored in train_url path. You can find checkpoint file together with result like the following in loss.log.

```bash
# distribute training result(8p)
epoch: 1 step: 750 ,loss: 30.88018
epoch: 2 step: 750 ,loss: 26.73352
epoch: 3 step: 750 ,loss: 22.76208
epoch: 4 step: 750 ,loss: 20.52259
epoch: 5 step: 750 ,loss: 19.34843
epoch: 6 step: 750 ,loss: 17.74093
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval_ascend.sh` for evaluation.

```bash
# infer on Ascend
sh run_eval_ascend.sh [DATA_URL] [TRAIN_URL] [CKPT_FILENAME] [DEVICE_ID(optional)]
```

- Run `run_eval_gpu.sh` for evaluation.

```bash
# infer on GPU
sh run_eval_gpu.sh [DATA_URL] [TRAIN_URL] [CKPT_FILENAME] [DEVICE_ID(optional)]
```

### [Evaluation result](#content)

Inference result will be stored in the train_url path. Under this, you can find result like the following in eval.log.

```bash
ckpt file name: ntsnet-112_750.ckpt
accuracy: 0.876
```

## Model Export

### [Export MindIR](#contents)

when export mindir file in Ascend 910, the cropAndResize operator differs from 310 and 910. Specifically, 310 requires an input shape (N,C,H,W) while 910 requires an input shape (N,H,W,C). You need to invalid the CropAndResize Validator check in 910 mindspore environment to export successfully.

```shell
python export.py --ckpt_file [CKPT_PATH] --train_url [TRAIN_URL]
```

- `ckpt_file` Checkpoint file name.
- `train_url` should be Directory contains checkpoint file.

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` The absolute path of ntsnet.mindir.
- `DATASET_PATH` The CUB_200_2011 dataset test directory.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                      | Telsa V100-PCIE                                                      |
| -------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| Model Version              | V1                                                          | V1                                                        |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             | TeslaA100; CPU 2.3GHz, 40cores; Memory 377G               |
| uploaded Date              | 16/04/2021 (day/month/year)                                 | 05/10/2021 (day/month/year)                               |
| MindSpore Version          | 1.1.1                                                       | 1.5.0rc1                                                  |
| Dataset                    | cub200-2011                                                 | cub200-2011                                               |
| Training Parameters        | epoch=200,  batch_size = 8                                  | epoch=200,  batch_size = 16                               |
| Optimizer                  | SGD                                                         | Momentum                                                  |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                                     |
| Output                     | predict class                                               | predict class                                             |
| Loss                       | 10.9852                                                     | 12.195317                                                 |
| Speed                      | 1pc: 130 ms/step;  8pcs: 138 ms/step                        | 1pc: 480 ms/step;                                         |
| Total time                 | 8pcs: 5.93 hours                                            |                                                           |
| Parameters                 | 87.6                                                        | 87.5                                                      |
| Checkpoint for Fine tuning | 333.07M(.ckpt file)                                         | 222.03(.ckpt file)                                        |
| Scripts                    | [ntsnet script](https://gitee.com/mindspore/models/tree/r2.0/research/cv/ntsnet) | [ntsnet script](https://gitee.com/mindspore/models/tree/r2.0/research/cv/ntsnet)|

# [Description of Random Situation](#contents)

We use random seed in train.py and eval.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# FAQ

First refer to [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) to find some common public questions.

- **Q: What to do if memory overflow occurs when using PYNATIVE_MODE？** **A**:Memory overflow is usually because PYNATIVE_MODE requires more memory. Setting the batch size to 2 reduces memory consumption and can be used for network training.
