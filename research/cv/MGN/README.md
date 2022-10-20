# Contents

- [Contents](#contents)
    - [MGN Description](#mgn-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
        - [Running scripts](#running-scripts)
        - [Run on Modelarts](#Run-on-Modelarts)
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
        - [Infer on Ascend310](#Infer-on-Ascend310)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [MGN Description](#contents)

Multiple Granularity Network (MGN), a multi-branch deep network architecture consisting of one branch for global feature representations and two branches for local feature representations.
Instead of learning on semantic regions, model uniformly partition the images into several stripes, and vary the number of parts in different local branches to obtain local feature representations with multiple granularities.

[Paper](https://arxiv.org/abs/1804.01438v1): Wang G., Yuan Y., et al. “Learning discriminative features with multiple granularities for person re-identification”. Proceedings of the 26th ACM international conference on Multimedia, 274-282.

## [Model Architecture](#contents)

The backbone of the network is ResNet50 which helps to achieve competitive performances in some Re-ID systems.
The most obvious modification different from the original version is to divide the subsequent part after res_conv4_1 block into three independent branches, sharing the similar architecture with the original ResNet-50.
In the upper branch, model employs down-sampling with a stride-2 convolution layer in res_conv5_1 block, following a global max-pooling (GMP) operation on the corresponding output feature map and a 1×1 convolution layer with batch normalization and ReLU to reduce 2048-dim features to 256-dim. This branch learns the global feature representations without any partition information, so we name this branch as the Global Branch.
The middle and lower branches both share the similar network architecture with Global Branch. The difference is that model employs no down-sampling operations in res_conv5_1 block, and output feature maps in each branch are uniformly split into several stripes in horizontal orientation, on which we independently perform the same following operations as Global Branch to learn local feature representations.

## [Dataset](#contents)

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

## [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

### [Running scripts](#contents)

Model uses pre-trained backbone ResNet50 trained on ImageNet2012. [Link](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44/)

After dataset preparation, you can start training and evaluation as follows:

(Note that you must specify dataset path and path to char dictionary in `configs/market1501_config.yml`)

```bash
# run training example
bash scripts/run_standalone_train_gpu.sh 0 /path/to/market1501/ /path/to/output/ /path/to/pretrined_resnet50.pth

# run distributed training example
bash scripts/run_distribute_train_gpu.sh 8 /path/to/market1501/ /path/to/output/ /path/to/pretrined_resnet50.pth

# run evaluation example
bash scripts/run_eval.sh /your/path/dataset /your/path/checkpoint_file GPU
```

### [Run on Modelarts](#contents)

If you want to run in modelarts, please check the official documentation of modelarts, and you can start training and evaluation as follows:

```text
# Train 8p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "pre_trained_backbone='/cache/data/resnet50.ckpt'" on default_config.yaml file.
#          Set "ckpt_path='/cache/train'" on default_config.yaml file if load pretrain.
#          Set "train_log_path='/cache/data/logs/'" on default_config.yaml file if load pretrain.
#          Set "log_path='/cache/data/logs/'" on default_config.yaml file if load pretrain.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "pre_trained_backbone='/cache/data/resnet50.ckpt'" on the website UI interface.
#          Add "ckpt_path='/cache/train'" on the website UI interface if load pretrain.
#          Add "train_log_path='/cache/data/logs/'" on the website UI interface if load pretrain.
#          Add "log_path='/cache/data/logs/'" on the website UI interface if load pretrain.
#          Add other parameters on the website UI interface.
# (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/MGN" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
#
# Train 1p on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "pre_trained_backbone='/cache/data/resnet50.ckpt'" on default_config.yaml file.
#          Set "ckpt_path='/cache/train'" on default_config.yaml file if load pretrain.
#          Set "train_log_path='/cache/data/logs/'" on default_config.yaml file if load pretrain.
#          Set "log_path='/cache/data/logs/'" on default_config.yaml file if load pretrain.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "pre_trained_backbone='/cache/data/resnet50.ckpt'" on the website UI interface.
#          Add "ckpt_path='/cache/train'" on the website UI interface if load pretrain.
#          Add "train_log_path='/cache/data/logs/'" on the website UI interface if load pretrain.
#          Add "log_path='/cache/data/logs/'" on the website UI interface if load pretrain.
#          Add other parameters on the website UI interface.
# (2) (option) Upload or copy your pretrained model to S3 bucket if load pretrain.
# (3) Upload a zip dataset to S3 bucket. (you could also upload the origin dataset, but it can be so slow.)
# (4) Set the code directory to "/path/MGN" on the website UI interface.
# (5) Set the startup file to "train.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
MGN
├── configs
│   └── market1501_config.yml  # Training/evaluation config
├── metric_utils
│   ├── __init__.py
│   ├── functions.py # Evaluation functions
│   └── re_ranking.py # Re-ranking function
├── model_utils
│   ├── __init__.py
│   ├── config.py # Config parser
│   ├── device_adapter.py # Device adapter for ModelArts
│   ├── local_adapter.py # Environment variables parser
│   └── moxing_adapter.py # Moxing adapter for ModelArts
├── scripts
│   ├── run_distribute_train_ascend.sh # Use the Market1501 data set to start Ascend distributed training (8 cards)
│   ├── run_distribute_train_gpu.sh # Use the Market1501 data set to start GPU distributed training (8 cards)
│   ├── run_eval.sh # Use the Market1501 data set to start single GPU or Ascend evaluation
│   ├── run_infer_310.sh # Ascend 310 infer
│   ├── run_standalone_train_gpu.sh # Use the Market1501 data set to start single GPU training
│   └── run_standalone_train_gpu.sh # Use the Market1501 data set to start single Ascend training
├── src
│   ├── __init__.py
│   ├── callbacks.py # Logging to file callbacks
│   ├── dataset.py #  Data preprocessing
│   ├── loss.py # MGN loss definition
│   ├── lr_schedule.py # Learning rate scheduler
│   ├── mgn.py # MGN network structure
│   ├── MGN_Callback.py # MGN network callback files
│   ├── resnet.py # ResNet 50 network structure
│   ├── sampler.py # Sampler definition
│   └── triplet_loss.py # Triplet loss definition
├── eval.py # Evaluate the network
├── export.py # Export the network
├── postprocess.py # Ascend 310 infer postprocess
├── preprocess.py # Ascend 310 infer preprocess
├── train.py # Train the network
└── README.md

```

### [Script Parameters](#contents)

#### Training Script Parameters

```text
usage: train.py  --config_path CONFIG_PATH [--distribute DISTRIBUTE] [--device_target DEVICE]
                 [--max_epoch N] [--decay_epochs EPOCHS]
                 [--ids_per_batch IDS_NUM] [--ims_per_id IMS_NUM]
                 [--is_save_on_master SAVE_FLAG] [--is_print_on_master PRINT_FLAG]
                 [--pre_trained MODEL_PATH] [--pre_trained_backbone BACKBONE_PATH]
                 [--data_dir DATA_PATH] [--ckpt_path SAVE_CHECKPOINT_PATH] [--train_log_path SAVE_LOGS_PATH]
                 [--keep_checkpoint_max CKPT_NUM] [--ckpt_interval CKPT_STEP]
                 [--log_interval LOG_STEP]

options:
    --config_path              path to .yml config file
    --distribute               pre_training by several devices: "true"(training by more than 1 device) | "false", default is "false"
    --device_target            target device ("GPU" | "CPU")
    --max_epoch                epoch size: N, default is 400 (800 for distributed)
    --decay_epochs             epoch for learning rate decay, default is '384,128' ('640,760' for distributed)
    --ids_per_batch            number of person in batch, default is 12 (6 for distributed)
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
    --log_interval             logging batch interval, default 8 (4 for distributed)
```

#### Running Options

```text
default_config.yaml:
    weight_decay                    optimizer weight decay
    global_loss_margin              margin for global MarginRankingLoss
    g_loss_weight                   weight of global loss
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

- Set options in `market1501_config.yaml`,
  including paths, learning rate and network hyperparameters.

- Run `run_standalone_train_gpu.sh` for non-distributed training of MGN model in GPU.

    ```bash
    bash scripts/run_standalone_train_gpu.sh DEVICE_ID DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50
    ```

- Run `run_distribute_train_gpu.sh` for distributed training of MGN model in GPU.

    ```bash
    bash scripts/run_distribute_train_gpu.sh DEVICE_NUM DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50
    ```

- Run `run_standalone_train_ascend.sh` for non-distributed training of MGN model in Ascend.

    ```bash
    bash scripts/run_standalone_train_ascend.sh DEVICE_ID DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50
    ```

- Run `run_distribute_train_ascend.sh` for distributed training of MGN model in Ascend.

    ```bash
    bash scripts/run_distribute_train_ascend.sh DEVICE_NUM DATA_DIR OUTPUT_PATH PRETRAINED_RESNET50 HCCL_JSON
    ```

## [Evaluation Process](#contents)

- Set options in `market1501_config.yaml`.

- Run `bash scripts/run_eval.sh` for evaluation of MGN model.

    ```bash
    bash scripts/run_eval.sh DATA_DIR CKPT_PATH DEVICE_TARGET
    ```

## Inference Process

### [Export MindIR](#contents)

```text
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]

options:
    --config_path              path to .yml config file
    --ckpt_file                checkpoint file
    --file_name                output file name
    --file_format              output file format, choices in ['MINDIR', 'AIR']
```

The ckpt_file and config_path parameters are required,
`FILE_FORMAT` should be in "MINDIR"

### Infer on Ascend310

Before performing inference, the mindir file must be exported by export.py script. We only provide an example of inference using MINDIR model.

```text
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

### result

Inference result will be shown in the terminal

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU                                                            | Ascend                                                         |
| -------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Resource                   | Tesla V100-PCIE 32G                                            | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8           |
| uploaded Date              | 12/17/2021 (month/day/year)                                    | 4/5/2021 (month/day/year)                                      |
| MindSpore Version          | 1.5.0                                                          | 1.6.1                                                      |
| Dataset                    | Market1501                                                     | Market1501                                                     |
| Training Parameters        | max_epoch=400, ids_per_batch=12, decay_epochs='320,380', lr_init=0.00015 | max_epoch=400, ids_per_batch=12, decay_epochs='320,380', lr_init=0.00015 |
| Optimizer                  | Adam                                                           | Adam                                                           |
| Loss Function              | ReIDLoss                                                       | ReIDLoss                                                       |
| Speed                      | 401ms/step (1pcs), 368ms/step (8pcs)                           | 346ms/step (1pcs), 304ms/step (8pcs)                           |
| Loss                       | 0.456                                                          | 0.492                                                          |
| Params (M)                 | 69                                                             | 69                                                             |
| Checkpoint for inference   | 846Mb (.ckpt file)                                             | 807Mb (.ckpt file)                                             |
| Scripts                    | [MGN scripts](scripts) | [MGN scripts](scripts) |

#### Evaluation Performance

| Parameters          | GPU                         | Ascend                      |
| ------------------- | --------------------------- | --------------------------- |
| Resource            | Tesla V100-PCIE 32G         | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8         |
| Uploaded Date       | 12/17/2021 (month/day/year) | 4/5/2022 (month/day/year)   |
| MindSpore Version   | 1.5.0                       | 1.6.1                       |
| Dataset             | Market1501                  | Market1501                  |
| batch_size          | 32                          | 32                          |
| outputs             | mAP, Rank-1                 | mAP, Rank-1                 |
| Accuracy            | mAP: 93.78%, rank-1: 95.31%.| mAP: 90.60%, rank-1: 93.76%.|

## [Description of Random Situation](#contents)

There are five random situations:

- Shuffle of the persons in the dataset.
- Select random images for persons.
- Random flip images.
- Random erase image augmentation.
- Initialization of some model weights.

Some seeds have already been set in train.py and sampler.py to avoid the randomness of dataset shuffle and weight initialization.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
