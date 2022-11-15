# Contents

- [Contents](#contents)
    - [Model Description](#model-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
        - [Market1501](#market1501)
        - [DukeMTMC-reID](#dukemtmc-reid)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
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
            - [Market1501 Training Performance](#market1501-training-performance)
            - [Market1501 Evaluation Performance](#market1501-evaluation-performance)
            - [DukeMTMC-reID Training Performance](#dukemtmc-reid-training-performance)
            - [DukeMTMC-reID Evaluation Performance](#dukemtmc-evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [Model Description](#contents)

ReID Strong Baseline proposes a novel neck structure named as batch normalization neck (BNNeck).
BNNeck adds a batch normalization layer after global pooling layer to separate metric and classification
 losses into two different feature spaces because we observe they are inconsistent in one embedding space.
Extended experiments show that BNNeck can boost the baseline.

[Paper](https://arxiv.org/abs/1906.08332): Luo H., Jiang W., et al. “A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification”. CVPRW2019, Oral.

## [Model Architecture](#contents)

Model uses ResNet50 as backbone. BNNeck adds a BN layer after features and before classifier FC layers.
The BN and FC layers are initialized through Kaiming initialization.

## [Dataset](#contents)

### [Market1501](#contents)

[Market1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset is used to train and test model.  
Market1501 contains 32,668 images of 1,501 labeled persons of six camera views.
There are 751 identities in the training set and 750 identities in the testing set.
In the original study on this proposed dataset, the author also uses mAP as the evaluation criteria to test the algorithms.  

Data structure:

```text
Market-1501-v15.09.15  
├── bounding_box_test [19733 entries]
├── bounding_box_train [12937 entries]
├── gt_bbox [25260 entries]
├── gt_query [6736 entries]
├── query [3369 entries]
└── readme.txt
```

### [DukeMTMC-reID](#contents)

[DukeMTMC-reID](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) is a new large-scale person ReID dataset and collects 36,411 images from 1,404 identities of eight camera views.
The training set has 16,522 images from 702 identities, and the testing set has 19,889 images from other 702 identities

Data structure:

```text
DukeMTMC-reID
├── bounding_box_test [17661 entries]
├── bounding_box_train [16522 entries]
├── CITATION.txt
├── LICENSE_DukeMTMC-reID.txt
├── LICENSE_DukeMTMC.txt
├── query [2228 entries]
└── README.md
```

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

Model uses pre-trained backbone ResNet50 trained on ImageNet2012. [Link](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44/)

After dataset preparation, you can start training and evaluation as follows:

(Note that you must specify dataset path in `configs/market1501_config.yml`)

### [Running on Ascend](#contents)

```bash
# run distributed training example
bash scripts/run_distribute_train_ascend.sh ./configs/market1501_config.yml /path/to/dataset/ /path/to/output/ /path/to/pretrained_resnet50.ckpt rank_table_8pcs.json 8

# run evaluation example
bash scripts/run_eval_ascend.sh ./configs/market1501_config.yml /your/path/checkpoint_file /path/to/dataset/
```

### [Running on GPU](#contents)

```bash
# run training example
bash scripts/run_standalone_train_gpu.sh  ./configs/market1501_config.yml 0 /path/to/dataset/ /path/to/output/ /path/to/pretrained_resnet50.pth

# run distributed training example
bash scripts/run_distribute_train_gpu.sh ./configs/market1501_config.yml 8 /path/to/dataset/ /path/to/output/ /path/to/pretrained_resnet50.pth

# run evaluation example
bash scripts/run_eval_gpu.sh ./configs/market1501_config.yml /your/path/checkpoint_file /path/to/dataset/
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
ReIDStrongBaseline
├── ascend310_infer  # application for 310 inference
├── configs
│   ├── dukemtmc_config.yml  # Training/evaluation config on DukeMTMC dataset
│   └── market1501_config.yml  # Training/evaluation config on Market1501 dataset
├── model_utils
│   ├── __init__.py
│   ├── config.py # Config parser
│   ├── device_adapter.py # Device adapter for ModelArts
│   ├── local_adapter.py # Environment variables parser
│   └── moxing_adapter.py # Moxing adapter for ModelArts
├── scripts
│   ├── run_distribute_train_Ascend.sh  # Start multi Ascend training
│   ├── run_distribute_train_gpu.sh  # Start multi GPU training
│   ├── run_eval_Ascend.sh  # Start single Ascend evaluation
│   ├── run_eval_gpu.sh # Start single GPU evaluation
│   ├── run_infer_310.sh  # Start 310 inference
│   └── run_standalone_train_gpu.sh  # Start single GPU training
├── src
│   ├── callbacks.py # Logging to file callbacks
│   ├── center_loss.py  # Center  Loss definition
│   ├── dataset.py #  Data preprocessing
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── bases.py  # Base dataset loader
│   │   ├── dukemtmcreid.py  # DukeMTMC dataset loader
│   │   └── market1501.py  # Market1501 dataset loader
│   ├── __init__.py
│   ├── loss.py  # Losses definition
│   ├── lr_schedule.py # Learning rate scheduler
│   ├── metric_utils.py # Evaluation functions
│   ├── model
│   │   ├── __init__.py
│   │   ├── cell_wrapper.py # Model wrappers
│   │   ├── resnet.py # ResNet 50 network structure
│   │   └── strong_reid.py # ReID Strong Baseline network structure
│   ├── sampler.py # ReID sampler definition
│   └── triplet_loss.py  # Triplet  Loss definition
├── eval.py # Evaluate the network
├── export.py # Export the network
├── postprogress.py # post process for 310 inference
├── train.py # Train the network
├── requirements.txt # Required libraries
└── README.md
```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  --config_path CONFIG_PATH [--distribute DISTRIBUTE] [--device_target DEVICE]
                 [--max_epoch N] [--start_decay_epoch EPOCH]
                 [--ids_per_batch IDS_NUM] [--ims_per_id IMS_NUM]
                 [--is_save_on_master SAVE_FLAG] [--is_print_on_master PRINT_FLAG]
                 [--pre_trained MODEL_PATH] [--pre_trained_backbone BACKBONE_PATH] [--data_dir DATA_DIR]
                 [--ckpt_path SAVE_CHECKPOINT_PATH] [--train_log_path SAVE_LOGS_PATH]
                 [--keep_checkpoint_max CKPT_NUM] [--ckpt_interval CKPT_STEP]
                 [--log_interval LOG_STEP]

options:
    --config_path              path to .yml config file
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --device_target            target device ("Ascend" | "GPU" | "CPU")
    --max_epoch                epoch size: N, default is 120
    --start_decay_epoch        epoch to decay, default is '40,70'
    --ids_per_batch            number of person in batch, default is 16 (8 for distributed)
    --ims_per_id               number of images for person in epoch
    --is_save_on_master        save checkpoint only from main thread for distributed
    --is_print_on_master       print loss logs only from main thread for distributed
    --pre_trained              path to pretrained model checkpoint
    --pre_trained_backbone     path to resnet50 backbone model checkpoint
    --data_dir                 path to dateset images
    --ckpt_path                path to save checkpoint files: PATH
    --train_log_path           path to save loss logs
    --keep_checkpoint_max      number for saving checkpoint files: N, default is 2
    --ckpt_interval            number of epochs between checkpoints, default 1
    --log_interval             logging batch interval, default 32
```

#### Running Options

```text
default_config.yaml:
    weight_decay                    optimizer weight decay
    global_loss_margin              margin for global MarginRankingLoss
    local_loss_margin               margin for local MarginRankingLoss
    center_loss_weight              weight of center loss
    crossentropy_loss_weight        weight of identity loss
```

#### Network Parameters

```text
Parameters for dataset and network (Training/Evaluation):
    image_size                      size of input image
    image_mean                      image mean value for normalization
    image_std                       image std value for normalization
    padding                         image padding before random crop

Parameters for learning rate:
    lr_init                         initial network learning rate
    lr_cri                          center loss learning rate
    warmup_factor                   warmup scale
    warmup_epoch                    warmup number of epochs
```

## [Training Process](#contents)

- Set options in `configs/market1501_config.yaml` or `configs/dukemtmc_config.yaml`,
  including paths, learning rate and network hyperparameters.

### Usage

#### on Ascend

- Run `run_distribute_train_Ascend.sh` for distributed training of ReID Strong Baseline model.
- The `RANK_TABLE_FILE` is placed under `scripts/`

    ```bash
    bash scripts/run_distribute_train_Ascend.sh CONFIG_PATH DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50 RANK_TABLE_FILE RANK_SIZE
    ```

#### on GPU

- Run `run_standalone_train_gpu.sh` for non-distributed training of  model.

    ```bash
    bash scripts/run_standalone_train_gpu.sh CONFIG_PATH DEVICE_ID DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50
    ```

- Run `run_distribute_train_gpu.sh` for distributed training of ReID Strong Baseline model.

    ```bash
    bash scripts/run_distribute_train_gpu.sh CONFIG_PATH DEVICE_NUM DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50
    ```

## [Evaluation Process](#contents)

- Set options in `market1501_config.yaml`.

### Usage

#### on Ascend

- Run `bash scripts/run_eval_Ascend.sh` for evaluation of ReID Strong Baseline model.

    ```bash
    bash scripts/run_eval_Ascend.sh CONFIG_PATH CKPT_PATH DATA_DIR
    ```

#### on GPU

- Run `bash scripts/run_eval_gpu.sh` for evaluation of ReID Strong Baseline model.

    ```bash
    bash scripts/run_eval_gpu.sh CONFIG_PATH CKPT_PATH DATA_DIR
    ```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

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

#### Market1501 Training Performance

| Parameters                 | Ascend                      | GPU                                                       |
| -------------------------- | --------------------------- |---------------------------------------------------------- |
| Resource                   | 8x Ascend 910 32G      |8x Tesla V100-PCIE 32G                                          |
| uploaded Date              | 04/21/2022 (month/day/year) |03/11/2022 (month/day/year)                                |
| MindSpore Version          | 1.3.0                       |1.5.0                                                      |
| Dataset                    | Market1501                  |Market1501                                                 |
| Training Parameters        | max_epoch=120, ids_per_batch=8, start_decay_epoch=151, lr_init=0.0014, lr_cri=1.0, decay_epochs='40,70' |max_epoch=120, ids_per_batch=8, start_decay_epoch=151, lr_init=0.0014, lr_cri=1.0, decay_epochs='40,70' |
| Optimizer                  | Adam, SGD                   |Adam, SGD                                                  |
| Loss Function              | Triplet, Smooth Identity, Center |Triplet, Smooth Identity, Center                      |
| Speed                      | 536ms/step (8pcs) |182ms/step (8pcs)                                                    |
| Loss                       | 0.28                        |0.24                                                       |
| Params (M)                 | 24.8                        |25.1                                                      |
| Checkpoint for inference   | 305Mb (.ckpt file)          |319Mb (.ckpt file)                                        |
| Scripts                    | [ReID Strong Baseline scripts](scripts) |[ReID Strong Baseline scripts](scripts)        |

#### Market1501 Evaluation Performance

| Parameters          | Ascend                        | GPU                         |
| ------------------- | ----------------------------- | --------------------------- |
| Resource            | 1x Ascend 910 32G             | 1x Tesla V100-PCIE 32G      |
| Uploaded Date       | 04/21/2022 (month/day/year)   | 03/11/2022 (month/day/year) |
| MindSpore Version   | 1.3.0                         | 1.5.0                       |
| Dataset             | Market1501                    | Market1501                  |
| batch_size          | 32                            | 32                          |
| outputs             | mAP, Rank-1                   | mAP, Rank-1                 |
| Accuracy            | mAP: 86.85%, rank-1: 94.36%   | mAP: 86.99%, rank-1: 94.48% |

#### DukeMTMC-reID Training Performance

| Parameters                 |  Ascend                     | GPU                                         |
| -------------------------- |---------------------------- | ------------------------------------------- |
| Resource                   | 8x Ascend 910 32G           | 8x Tesla V100-PCIE 32G                      |
| uploaded Date              | 04/21/2022 (month/day/year) | 03/11/2022 (month/day/year)                 |
| MindSpore Version          | 1.3.0                       | 1.5.0                                       |
| Dataset                    | DukeMTMC-reID               | DukeMTMC-reID                               |
| Training Parameters        | max_epoch=120, ids_per_batch=8, start_decay_epoch=151, lr_init=0.0014, lr_cri=1.0, decay_epochs='40,70'| max_epoch=120, ids_per_batch=8, start_decay_epoch=151, lr_init=0.0014, lr_cri=1.0, decay_epochs='40,70' |
| Optimizer                  | Adam, SGD                   | Adam, SGD                                   |
| Loss Function              | Triplet, Smooth Identity, Center | Triplet, Smooth Identity, Center       |
| Speed                      | 524ms/step (8pcs)           | 180ms/step (8pcs)                           |
| Loss                       | 0.27                        | 0.24                                        |
| Params (M)                 | 24.8                        |25.1                                                      |
| Checkpoint for inference   | 302Mb (.ckpt file)          | 319Mb (.ckpt file)                          |
| Scripts                    | [ReID Strong Baseline scripts](scripts)| [ReID Strong Baseline scripts](scripts) |

#### DukeMTMC-reID Evaluation Performance

| Parameters          | Ascend                      | GPU                         |
| ------------------- | --------------------------- | --------------------------- |
| Resource            | 1x Ascend 910 32G           | 1x Tesla V100-PCIE 32G      |
| Uploaded Date       | 04/21/2022 (month/day/year) | 03/11/2022 (month/day/year) |
| MindSpore Version   | 1.3.0                       | 1.5.0                       |
| Dataset             | DukeMTMC-reID               | DukeMTMC-reID               |
| batch_size          | 32                          | 32                          |
| outputs             | mAP, Rank-1                 | mAP, Rank-1                 |
| Accuracy            | mAP: 76.58%, rank-1: 87.43% | mAP: 76.68%, rank-1: 87.34% |

## [Description of Random Situation](#contents)

There are six random situations:

- Shuffle of the persons in the dataset.
- Select random images for persons.
- Random flip images.
- Random crop images.
- Random erasing images.
- Initialization of some model weights.

Some seeds have already been set in train.py and sampler.py to avoid the randomness of dataset shuffle and weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
