# Contents

- [Contents](#contents)
    - [Transformer Description](#transformer-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Dataset Preparation](#dataset-preparation)
        - [Running scripts](#running-scripts)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
            - [Training Script Parameters](#training-script-parameters)
            - [Running Options](#running-options)
            - [Network Parameters](#network-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [AlignedReID Description](#contents)

AlignedReID generates a single global feature as the final output of the input image, and use the L2 distance as the similarity metric.
However, the global feature is learned jointly with local features in the learning stage.

[Paper](https://arxiv.org/abs/1711.08184):  Zhang, Luo, et al. “AlignedReID: Surpassing Human-Level Performance in Person Re-Identification”. arXiv preprint arXiv:1711.08184, 2017.

## [Model Architecture](#contents)

Model with backbone Resnet50 extract a feature map, which is the output of the last convolution layer (C×H×W).
A global feature (a C-d vector) is extracted by directly applying global pooling on the feature map. For the local features, a horizontal pooling, which is a global pooling in the horizontal direction,
is first applied to extract a local feature for each row, and a 1 × 1 convolution is then applied to reduce the channel number from C to c.
In this way, each local feature (a c-d vector) represents a horizontal part of the image for a person.
As a result, a person image is represented by a global feature and H local features.

## [Dataset](#contents)

Market1501 dataset is used to train and test model.  
Market1501 contains 32,668 images of 1,501 labeled persons of six camera views.
There are 751 identities in the training set and 750 identities in the testing set.
In the original study on this proposed dataset, the author also uses mAP as the evaluation criteria to test the algorithms.  

Data structure:

```shell
Market-1501-v15.09.15  
├── bounding_box_test [19733 entries]
├── bounding_box_train [12937 entries]
├── gt_bbox [25260 entries]
├── gt_query [6736 entries]
├── query [3369 entries]
└── readme.txt
```

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

### [Dataset Preparation](#contents)

Dataset must be prepared by huanghoujing's [script](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch/blob/master/script/dataset/transform_market1501.py)
 or can be downloaded from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4).

Prepared data structure:

```shell
market1501  
├── images  
│  ├── 00000000_0001_00000000.jpg
│  ├── 00000000_0001_00000001.jpg
│  ├── ...
│  ├── 00001501_0006_00000002.jpg
│  ├── 00001501_0006_00000003.jpg
│  └── 00001501_0006_00000004.jpg
├── ori_to_new_im_name.pkl  
└── partitions.pkl
```

### [Running scripts](#contents)

Model uses pre-trained backbone ResNet50 trained on ImageNet2012. [Link](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44/)

After dataset preparation, you can start training and evaluation as follows:

(Note that you must specify dataset path and path to char dictionary in `configs/market1501_config.yml` and `configs/market1501_multigpu_config.yml`)

```bash
# run training example
bash scripts/run_standalone_train_gpu.sh 0

# run distributed training example
bash scripts/run_distribute_train_gpu.sh 8

# run evaluation example
bash scripts/run_eval_gpu.sh /your/path/checkpoint_file
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```shell
AlignedReID
├── configs
│   ├── market1501_config.yml # Standalone training/evaluation config
│   └── market1501_multigpu_config.yml # Distributed training config
├── metric_utils
│   ├── __init__.py
│   ├── distance.py # Distance evaluation
│   ├── metric.py # Evaluation metrics
│   └── re_ranking.py # Re-ranking function
├── model_utils
│   ├── __init__.py
│   ├── local_adapter.py # Environment variables parser
│   ├── config.py # Config parser
├── scripts
│   ├── run_distribute_train_gpu.sh # Use the Market1501 data set to start GPU distributed training (8 cards)
│   ├── run_eval_gpu.sh # Use the Market1501 data set to start single GPU evaluation
│   └── run_standalone_train_gpu.sh # Use the Market1501 data set to start single GPU training
├── src
│   ├── __init__.py
│   ├── aligned_reid.py # Aligned ReID network structure
│   ├── callbacks.py # Logging to file callback
│   ├── dataset.py #  Data preprocessing
│   ├── loss.py # Aligned ReID loss definition
│   ├── lr_schedule.py # Learning rate scheduler
│   ├── resnet.py # ResNet 50 network structure
│   └── triplet_loss.py # Triplet loss definition
├── eval.py # Evaluate the network
├── train.py # Train the network
└── README.md
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  --config_path CONFIG_PATH [--distribute DISTRIBUTE] [--device_target DEVICE]
                 [--max_epoch N] [--start_decay_epoch EPOCH]
                 [--ids_per_batch IDS_NUM] [--ims_per_id IMS_NUM]
                 [--is_save_on_master SAVE_FLAG] [--is_print_on_master PRINT_FLAG]
                 [--pre_trained MODEL_PATH] [--pre_trained_backbone BACKBONE_PATH]
                 [--ckpt_path SAVE_CHECKPOINT_PATH] [--train_log_path SAVE_LOGS_PATH]
                 [--keep_checkpoint_max CKPT_NUM] [--ckpt_interval CKPT_STEP]
                 [--log_interval LOG_STEP]

options:
    --config_path              path to .yml config file
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --device_target            target device ("GPU" | "CPU")
    --max_epoch                epoch size: N, default is 300 (600 for distributed)
    --start_decay_epoch        epoch to start exponential decay, default is 151 (301 for distributed)
    --ids_per_batch            number of person in batch, default is 32 (8 for distributed)
    --ims_per_id               number of images for person in epoch
    --is_save_on_master        save checkpoint only from main thread for distributed
    --is_print_on_master       print loss logs only from main thread for distributed
    --pre_trained              path to pretrained model checkpoint
    --pre_trained_backbone     path to resnet50 backbone model checkpoint
    --data_dir                 path to dateset images
    --partitions_file          path to prepared pkl file with meta information
    --ckpt_path                path to save checkpoint files: PATH
    --train_log_path           path to save loss logs
    --keep_checkpoint_max      number for saving checkpoint files: N, default is 2
    --ckpt_interval            number of epochs between checkpoints, default 1
    --log_interval             logging batch interval, default 8 (4 for distributed)
```

#### Running Options

```text
default_config.yaml:
    weight_decay                    optimizer weight decay
    global_loss_margin              margin for global MarginRankingLoss
    local_loss_margin               margin for local MarginRankingLoss
    g_loss_weight                   weight of global loss
    l_loss_weight                   weight of local loss
    id_loss_weight                  weight of identity loss
```

#### Network Parameters

```text
Parameters for dataset and network (Training/Evaluation):
    image_size                      size of input image
    image_mean                      image mean value for normalization
    image_std                       image std value for normalization

Parameters for learning rate:
    lr_init                         initial learning rate
```

## [Training Process](#contents)

- Set options in `market1501_config.yaml` or `market1501_multigpu_config.yaml`,
  including paths, learning rate and network hyperparameters.

- Run `run_standalone_train_gpu.sh` for non-distributed training of AlignedReID model.

    ```bash
    bash scripts/run_standalone_train_gpu.sh DEVICE_ID
    ```

- Run `run_distribute_train_gpu.sh` for distributed training of AlignedReID model.

    ```bash
    bash scripts/run_distribute_train_gpu.sh DEVICE_NUM
    ```

## [Evaluation Process](#contents)

- Set options in `market1501_config.yaml`.

- Run `bash scripts/run_eval_gpu.sh` for evaluation of AlignedReID model.

    ```bash
    bash scripts/run_eval_gpu.sh CKPT_PATH
    ```

- Calculate Character Error Rate

    ```bash
    python evaluate_cer.py
    ```

## Inference Process

### [Export MindIR](#contents)

```text
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]

options:
    --config_path              path to .yml config file
    --ckpt_file                checkpoint file
    --file_name                output file name
    --file_format              output file format, choices in ['MINDIR']
```

The ckpt_file and config_path parameters are required,
`FILE_FORMAT` should be in "MINDIR"

### result

Inference result will be shown in the terminal

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU                                                            |
| -------------------------- | -------------------------------------------------------------- |
| Resource                   | 1x V100-PCIE                                           |
| uploaded Date              | 11/10/2021 (month/day/year)                                    |
| MindSpore Version          | 1.5.0rc1                                                       |
| Dataset                    | Market1501                                                     |
| Training Parameters        | max_epoch=300, ids_per_batch=32, start_decay_epoch=151         |
| Optimizer                  | Adam                                                           |
| Loss Function              | ReIDLoss                                                       |
| Speed                      | 330ms/step (1pcs)                                              |
| Loss                       | 0.001                                                          |
| Params (M)                 | 23.5                                                           |
| Checkpoint for inference   | 304Mb (.ckpt file)                                             |
| Scripts                    | [AlignedReID scripts](scripts) |

#### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | 1x V100-PCIE           |
| Uploaded Date       | 11/10/2021 (month/day/year) |
| MindSpore Version   | 1.5.0rc1                    |
| Dataset             | Market1501                  |
| batch_size          | 512                         |
| outputs             | mAP, Rank-1 without and after re-ranking |
| Accuracy            | mAP: 71.83%, rank-1: 86.94%. **After re-rank**: mAP: 86.59%, rank-1: 90.32% |

## [Description of Random Situation](#contents)

There are four random situations:

- Shuffle of the persons in the dataset.
- Select random images for persons
- Random flip images.
- Initialization of some model weights.

Some seeds have already been set in train.py and dataset.py to avoid the randomness of dataset shuffle and weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
