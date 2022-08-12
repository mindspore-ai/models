# Content

[查看中文](./README_CN.md)

<!-- TOC -->

- [Content](#content)
- [STPM Description](#stpm-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Pretrained Model](#pretrained-model)
    - [Training Process](#training-process)
        - [Usage](#usage)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-2)
        - [Result](#result-2)
    - [Export mindir model](#export-mindir-model)
    - [Inference Process](#inference-process)
        - [Usage](#usage-3)
        - [Result](#result-3)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random State](#description-of-random-state)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# STPM Model

The model is generally divided into two networks, the teacher and the student network. The teacher network is a network pretrained on the image classification task, and the student network has the same architecture. A test image or pixel has a high anomaly score if its features in the two networks are significantly different. Hierarchical feature alignment between the two networks enables it to detect anomalies of different sizes in one forward pass.

[Paper](https://arxiv.org/pdf/2103.04257v2.pdf)： Wang G ,  Han S ,  Ding E , et al. Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection[J].  2021.

# Dataset

Dataset used：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

- Dataset size：4.9G，15 classes、5354 images(700x700-1024x1024)
    - Train：3629 images
    - Val：1725 images

# Environment Requirements

- Hardware: Ascend/GPU
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information about MindSpore, please check the resources below:
    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Ascend

```shell
# single card training
cd scripts
bash run_standalone_train.sh  /path/dataset /path/backbone_ckpt category 1
# train all mvtec datasets on single card
cd scripts
bash run_all_mvtec.sh  /path/dataset /path/backbone_ckpt 1

# eval
bash run_eval.sh  /path/dataset /path/ckpt category 1
```

- GPU

```python
# train
bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]

# eval
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

## Script Description

## Script and Sample Code

```text
├── STPM  
 ├── README.md
 ├── README_CN.md
 ├── ascend_310_infer              // 310 inference
   ├── inc
     └── utils.h
   ├── src
     ├── main.cc
     └── utils.cc
   ├── build.sh
   └── Cmakelists.txt
 ├── scripts
   ├── run_310_infer.sh            // 310 inference
   ├── run_standalone_train.sh     // train on Ascend
   ├── run_standalone_train_gpu.sh
   ├── run_all_mvtec.sh            // Perform training and evaluation on all mvtec datasets on Ascend
   ├── run_eval_gpu.sh
   └── run_eval.sh                 // Eval on Ascend
 ├──src
   ├── loss.py
   ├── dataset.py
   ├── callbacks.py
   ├── resnet.py
   ├── stpm.py
   ├── pth2ckpt.py
   └── utils.py
 ├── eval.py
 ├── train.py
 ├── preprocess.py                  // 310 inference
 ├── postprocess.py                 // 310 inference
 └── requirements.txt
```

## Script Parameters

```text
main parameters in train.py and eval.py:

-- modelarts：whether to use the modelarts platform to train. Choose from True, False. Default is False.
-- device_target: Choose from [Ascend, GPU]. Default is Ascend.
-- device_id：Device ID used to train or evaluate the dataset. This parameter is ignored when using train.sh for distributed training.
-- train_url：output path for checkpoints
-- pre_ckpt_path：path to resnet18 pretrained checkpoints
-- save_sample：whether to save the picture during inference
-- save_sample_path：path to save the inference image
-- dataset_path：path to dataset
-- ckpt_path：path to checkpoint
-- eval_url：path to test dataset
```

## pretrained model

The pretrained model is ResNet18. Load pretrained model into the teacher network,
while the student network does not use a pretrained model. Pretrained model can be obtained in the following ways:

- Download ResNet18 from modelzoo and train on ImageNet2012 to get a pretrained model. Since the number of categories set by ImageNet2012 in modelzoo is 1001. At this point, you need to change the parameter num_class to 1001 when training inference.
- Download pytorch's ResNet18 pretrained model and convert to mindspore format via src/pth2ckpt.py script.
- Download ready checkpoints from [here](https://download.mindspore.cn/model_zoo/r1.3/resnet18_ascend_v130_imagenet2012_official_cv_bs256_acc70.64/).

## Training Process

### Usage

- Ascend

```shell
bash scripts/run_standalone_train.sh [DATASET_PATH] [BACKONE_PATH] [CATEGORY] [DEVICE_ID]
# For all mvtec dataset you can execute the following command, DEVICE_NUM is the number of cards to be executed, and the 15 datasets under mvtec will run independently on each card.
bash scripts/run_all_mvtec.sh [DATASET_PATH] [BACKONE_PATH] [DEVICE_NUM]
```

- GPU

```shell
bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [BACKONE_PATH] [CATEGORY] [DEVICE_ID]
# For all mvtec dataset you can execute the following command, DEVICE_NUM is the number of cards to be executed, and the 15 datasets under mvtec will run independently on each card.
bash scripts/run_all_mvtec.sh [DATASET_PATH] [BACKONE_PATH] [DEVICE_NUM]
```

The above shell script will run the training in the background. The results can be viewed through the `train.log` file.

## Evaluation process

### Usage

- Ascend

```shell
bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CATEGORY] [DEVICE_ID]
```

- GPU

```shell
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

### Result

The above python command will run in the background and you can view the results in `eval.log` file. The accuracy of the test dataset is as follows:

|  Category  | pixel-level | image-level |
| :--------: | :---------: | :---------: |
| bottle     | 0.987       | 1.000  |
| cable      | 0.959       | 0.983  |
| capsule    | 0.984       | 0.868  |
| carpet     | 0.988       | 0.998  |
| grid       | 0.990       | 0.997  |
| hazelnut   | 0.989       | 1.000  |
| leather    | 0.994       | 1.000  |
| metal_nut  | 0.976       | 1.000  |
| pill       | 0.973       | 0.962  |
| screw      | 0.962       | 0.921  |
| tile       | 0.965       | 0.978  |
| toothbrush | 0.985       | 0.911  |
| transistor | 0.829       | 0.942  |
| wood       | 0.964       | 0.992  |
| zipper     | 0.981       | 0.910  |
| mean       | 0.968       | 0.964  |

## Export mindir model

```python
python export.py --ckpt_file [CKPT_PATH] --category [FILE_NAME] --file_format [FILE_FORMAT]
```

Argument `ckpt_file` is required, `EXPORT_FORMAT` choose from ["AIR", "MINDIR"].

# Inference Process

## Usage

Before performing inference, the mindir file needs to be exported via `export.py`.

```shell
# Ascend310 inference
bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

`DEVICE_TARGET` choose from：['GPU', 'CPU', 'Ascend']，`NEED_PREPROCESS` Indicates whether the data needs to be preprocessed. The optional value range is:'y' or 'n'，choose ‘y’，`DEVICE_ID` optional, default is 0.

### Result

The inference result is saved in the current path, and the final accuracy result can be seen in `acc.log`.

```text
category:  zipper
Total pixel-level auc-roc score :  0.980967986777201
Total image-level auc-roc score :  0.909926470588235
```

# Model Description

## Performance

### Training Performance

| Parameter     | Ascend                                          | GPU |
| ------------- | ----------------------------------------------- | --- |
| Model         | STPM                                            | STPM |
| Environment   | Ascend 910; CPU: 2.60GHz, 192 cores; memory, 755G | Ubuntu 18.04.6, Tesla V100 1p, CPU 2.90GHz, 64cores, RAM 252GB |
| Upload Date   | 2021-12-25                                      | 2022-03-01 |
| MindSpore version | 1.5.0                                       | 1.5.0 |
| Dataset       | MVTec AD                                        | MVTec AD (zipper) |
| Training parameters | lr=0.4, epochs=100                        | lr=0.4, epochs=100 |
| Optimizer     | SGD                                             | SGD |
| Loss func     | MSELoss                                         | MSELoss |
| Output        | probability                                     | probability |
| Loss          | 2.6                                             | 2.46 |
| Speed         |                                                 | 860 ms/step |
| Total time    | 1 card: 0.6h                                    | 0.5h |
| ROC-AUC (pixel-level) | 0.981                                   | 0.9874 |

# Description of Random State

The initial parameters of the network are all initialized at random.

# ModelZoo Homepage  

Please check the official [homepage](https://gitee.com/mindspore/models).
