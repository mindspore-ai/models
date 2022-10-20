# contents

<!-- TOC -->

- [contents](#contents)
    - [Model description](#model-description)
        - [Overview](#overview)
        - [paper](#paper)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Download the dataset, and divide the training set and test set](#download-the-dataset-and-divide-the-training-set-and-test-set)
    - [Environmental requirements](#environmental-requirements)
    - [Quick start](#quick-start)
    - [Script description](#script-description)
        - [Script and sample code](#script-and-sample-code)
        - [Script parameters](#script-parameters)
            - [Usage](#usage)
            - [result](#result)
        - [Usage](#usage)
            - [result](#result)
    - [VGG16 on AVA_Dataset](#vgg16-on-ava_dataset)
    - [Random description](#random-description)
    - [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

## [Model description](#contents)

## [Overview](#contents)

NIMA - is the network for automatically learned quality assessment for images. It predicts human opinion scores using Earth Moving Distance loss.
In this repository, we present the implementation for the AVA dataset using the VGG16 network as backbone.

## [paper](#contents)

1. [Thesis](https://arxiv.org/abs/1709.05424): H. Talebi and P. Milanfar, "NIMA: Neural Image Assessment"
2. [Backbone paper](https://arxiv.org/abs/1409.1556): K. Simonyan, A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"

# [Model architecture](#contents)

The overall network architecture of VGG16 is as follows:
[Link](https://arxiv.org/pdf/1512.03385.pdf)

Pre-trained model:
[Link](https://download.mindspore.cn/model_zoo/r1.3/vgg16_bn_ascend_v130_imagenet2012_official_cv_bs64_top1acc74.33__top5acc92.1/)

## [Dataset](#contents)

## [Download the dataset, and divide the training set and test set](#contents)

Dataset used: [AVA_Dataset](<https://github.com/mtobeiyf/ava_downloader/tree/master/AVA_dataset>)

Use label: [AVA.txt](https://github.com/mtobeiyf/ava_downloader/blob/master/AVA_dataset/AVA.txt)

Prepare the data, execute the following python command to divide the dataset

```bash
python ./src/dividing_label.py --config_path=~/config_single_gpu.yaml
#Change configuration files: data_path, label_path, val_label_path, train_label_path, val_data_path
```

- Data set size: 255,502 color images
    - Training set: 229,905 images
    - Test set: 25,597 images
- Data format: JEPG image

## [Environmental requirements](#contents)

- Hardware (GPU)
    - Prepare GPU processor to build the hardware environment.
    - Links
        - [MindSpore](https://www.mindspore.cn/install/en)
- For details, please refer to the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [Quick start](#contents)

After installing MindSpore through the official website, you can follow the steps below to train and evaluate:

1. Prepare dataset [Download the dataset, and divide the training set and test set](#download-the-dataset-and-divide-the-training-set-and-test-set)
2. Specify path to pretrained model (`checkpoint_path`) in yaml config.
3. Run training and evaluation scripts:

```bash
# Run training example
bash ./scripts/run_standalone_train_gpu.sh 0

# Distributed training
bash ./scripts/run_distributed_train_gpu.sh 8

# Run evaluation example
bash ./scripts/run_eval_gpu.sh

```

## [Script description](#contents)

## [Script and sample code](#contents)

```text
.
├──nima_vgg16
  ├──scripts
    ├──run_distributed_train_gpu.sh # training on multiple gpus
    ├──run_eval_gpu.sh              # evaluate on single gpu
    ├──run_standalone_train_gpu.sh  # training on single gpu
  ├──src
    ├──utils
      ├──var_init.py                # Initialization for vgg backbone
    ├──MyCallBack.py                # callback
    ├──MyDataset.py                 # Data processing
    ├──MyMetric.py                  # Loss and metrics
    ├──config.py                    # Parameter configuration
    ├──dividing_label.py            # dividing data set
    ├──vgg.py                       # Main network architecture
  ├── README.md                     # Related description
  ├──eval.py                        # Evaluation script
  ├──export.py                      # Export the checkpoint file to mindir
  ├──train.py                       # training script

```

## [Script parameters](#contents)

```text
"batch_norm": True                 # whether to use batch norm
"batch_size": 32                   # training batch size
"bf_crop_size": 256                # Picture size before cropping
"checkpoint_path": "./vgg16.ckpt"  # The absolute path of the pre-training model
"ckpt_save_dir": "./ckpt/"         # model save path
"device_target": "GPU"             # Run device
"enable_modelarts": False          # Whether to use modelarts for training, the default is False
"epoch_size": 50                   # Total number of training epochs
"has_bias": False                  # whether to use bias
"has_dropout": True                # whether to use Dropout regularization
"image_size": 224                  # The actual size of the image sent to the network
"initialize_mode": "KaimingNormal" # weights initialization type
"is_distributed": False            # whether distributed training, the default is False
"keep_checkpoint_max": 10          # Save the maximum number of checkpoints
"learning_rate": 0.000125          # Learning rate
"momentum": 0.95                   # momentum
"num_parallel_workers": 8          # Numbers of parallel workers for data loading.
"output_path": "./"                # modelarts When training, copy the ckpt_save_dir file to the bucket
"pad_mode": "pad"                  # padding mode for vgg16
"padding": 1                       # padding border size
"train_data_path": "AVA_train.txt" # Absolute path of training set
"val_data_path": "AVA_test.txt"    # Absolute path of test set
"weight_decay": 0.001              # weight decay value

```

> For training we use the VGG16 model trained on the ImageNet2012 dataset.
>[Link](https://download.mindspore.cn/model_zoo/r1.3/vgg16_bn_ascend_v130_imagenet2012_official_cv_bs64_top1acc74.33__top5acc92.1/)

### [Usage](#contents)

```bash
# Stand-alone training
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID]
```

After running the above python command, you can view the result through the `log.txt` file

```bash
# Distributed training
bash scripts/run_distribute_train_gpu.sh [NUM_DEVICES]
```

### [result](#contents)

```text
# Distributed training results (8P)
epoch: 50 step: 1306, loss is 0.061359227
epoch: 50 step: 1306, loss is 0.057158004
epoch: 50 step: 1306, loss is 0.055451274
epoch: 50 step: 1306, loss is 0.055351827
epoch: 50 step: 1306, loss is 0.059732627
epoch: 50 step: 1306, loss is 0.05215319
epoch: 50 step: 1306, loss is 0.044236004
...

mse: 0.3903769649642374
deal imgs is: 25597
SRCC: 0.6302367589512725

```

## [Evaluation process](#contents)

### [Usage](#contents)

```text
# Run evaluation example
Usage: python eval.py
```

Change the `data_path`, `val_data_path`, `ckpt_file` in the configuration file `config_single_gpu.yaml`

### result

Evaluation result will be printed in std-out.

```text
SRCC: 0.63
```

#### [VGG16 on AVA_Dataset](#contents)

| Parameters | GPU |
| ----------------------  | ----------------------------------------------------------- |
| Model version           | VGG16                                                       |
| Resources               |  Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz 8x NVIDIA TITAN V|
| Upload Date             | 6.12.2021                                                   |
| MindSpore version       | 1.5.0rc1                                                    |
| Dataset                 | AVA_Dataset                                                 |
| Training parameters     | epoch=50, steps per epoch=7184, batch_size = 32             |
| Optimizer               | SGD                                                         |
| Loss function           | EmdLoss (bulldozer distance)                                |
| Output                  | Probability Distribution                                    |
| Loss                    | 0.0548                                                      |
| Speed                   | 577 ms/step                                                 |
| Total time              | 10h                                                         |
| Parameters (M)          | 138                                                         |
| Fine-tuning checkpoints | 1GB (.ckpt file)                                            |
| Configuration file      | /cv/nima_vgg16/config_dist_8_gpu.yaml                       |

## [Random description](#contents)

Random.seed(10) is set in `dividing_label.py`, and set_seed(10) is also set in `train.py`.

## [ModelZoo homepage](#contents)

Please visit the official website [homepage](https://gitee.com/mindspore/models)
