# Contents

- [3D-LWNet Description](#3d-lwnet-description)
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
        - [Running on GPU](#running-on-gpu)
            - [Usage](#usage)
            - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Running on GPU](#running-on-gpu-1)
            - [Usage](#usage-1)
            - [Result](#result-1)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [3D-LWNet on Indian Pines](#3d-lwnet-on-indian-pines)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
        - [3D-LWNet on Salinas](#3d-lwnet-on-salinas)
            - [Training Performance](#training-performance-1)
            - [Evaluation Performance](#evaluation-performance-1)
        - [3D-LWNet on WHU_Hi_HongHu dataset](#3d-lwnet-on-whu_hi_honghu-dataset)
            - [Training Performance](#training-performance-2)
            - [Evaluation Performance](#evaluation-performance-2)
- [ModelZoo Homepage](#modelzoo-homepage)

# [3D-LWNet Description](#contents)

## Description

3D-LWNet(3-D lightweight convolutional neural network) has a deeper network structure, less parameters, and lower computation cost, resulting in better hyperspectral image (HSI) classification. It was proposed by Haokui Zhang and other five authors.

## Paper

[paper](https://arxiv.org/abs/2012.03439):Haokui Zhang, Ying Li, Yenan Jiang, Peng Wang, Qiang Shen, Chunhua Shen. "Hyperspectral Classification Based on Lightweight 3-D-CNN With Transfer Learning"

# [Model Architecture](#contents)

The framework of the HSI classification is shown in Fig. 1. It consists of three parts, including samples extraction,3-D-LWNet, and classification result. The structure of HSI is 3-D, so it is intuitive to implement a 3-D model for classification. In sample extraction, we extract S × S × L-sized cube as a sample and each cube is extracted from a neighborhood window centered around a pixel. S and L are the spatial size and the number of spectral bands, respectively. The label of each sample is that of the pixel located in the center of this cube.

# [Dataset](#contents)

Dataset used:

1. [Indian Pines](<https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines>)

- Dataset size：10249 samples in 16 classes
    - Train：1412 samples
    - Val：353 samples
    - Test：6223 samples
- Data format：mat files
    - Note：Data will be processed in data_preprocess.py

| Class                      | Train samples | Validation samples | Test samples |
|----------------------------|---------------|--------------------|--------------|
| Alfalfa                    | 24            | 6                  | 16           |
| Corn-notill                | 120           | 30                 | 1198         |
| Corn-mintill               | 120           | 30                 | 232          |
| Corn                       | 80            | 20                 | 5            |
| Grass-pasture              | 120           | 30                 | 139          |
| Grass-trees                | 120           | 30                 | 580          |
| Grass-pasture-mowed        | 16            | 4                  | 8            |
| Hay-windrowed              | 120           | 30                 | 130          |
| Oats                       | 12            | 3                  | 5            |
| Soybean-notill             | 120           | 30                 | 675          |
| Soybean-mintill            | 120           | 30                 | 2032         |
| Soybean-clean              | 120           | 30                 | 263          |
| Wheat                      | 120           | 30                 | 55           |
| Woods                      | 120           | 30                 | 793          |
| Buildings-Grass-Trees      | 40            | 10                 | 49           |
| Stone-Steel-Towers         | 40            | 10                 | 43           |

2. [Salinas](<https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines>)

- Dataset size：54129 samples in 16 classes
    - Train：3984 samples
    - Val：996 samples
    - Test：49149 samples
- Data format：mat files
    - Note：Data will be processed in data_preprocess.py

| Class                     | Train samples | Validation samples | Test samples |
|---------------------------|---------------|--------------------|--------------|
| Brocoli_green_weeds_1     | 160           | 40                 | 1809         |
| Brocoli_green_weeds_2     | 288           | 72                 | 3366         |
| Fallow                    | 144           | 36                 | 1796         |
| Fallow_rough_plow         | 80            | 20                 | 1294         |
| Fallow_smooth             | 160           | 40                 | 2478         |
| Stubble                   | 288           | 72                 | 3599         |
| Celery                    | 256           | 64                 | 3259         |
| Grapes_untrained          | 800           | 200                | 10271        |
| Soil_vinyard_develop      | 480           | 120                | 5603         |
| Corn_senesced_green_weeds | 256           | 64                 | 2958         |
| Lettuce_romaine_4wk       | 80            | 20                 | 968          |
| Lettuce_romaine_5wk       | 128           | 32                 | 1767         |
| Lettuce_romaine_6wk       | 64            | 16                 | 836          |
| Lettuce_romaine_7wk       | 80            | 20                 | 970          |
| Vinyard_untrained         | 576           | 144                | 6548         |
| Vinyard_vertical_trellis  | 144           | 36                 | 1627         |

3. [WHU_Hi_HongHu dataset](<http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm>)

- Dataset size：386693 samples in 22 classes
    - Train：16080 samples
    - Val：4320 samples
    - Test：366493 samples
- Data format：mat files
    - Note：Data will be processed in data_preprocess.py

| Class                    | Train samples | Validation samples | Test samples |
|--------------------------|---------------|--------------------|--------------|
| Red roof                 | 960           | 240                | 12841        |
| Road                     | 320           | 80                 | 3112         |
| Bare soil                | 1280          | 320                | 20221        |
| Cotton                   | 2000          | 800                | 160485       |
| Cotton firewood          | 480           | 120                | 5818         |
| Rape                     | 1600          | 400                | 12557        |
| Chinese cabbage          | 1280          | 320                | 22503        |
| Pakchoi                  | 320           | 80                 | 3654         |
| Cabbage                  | 800           | 200                | 9819         |
| Tuber mustard            | 800           | 200                | 11394        |
| Brassica parachinensis   | 800           | 200                | 10015        |
| Brassica chinensis       | 640           | 160                | 8154         |
| Small Brassica chinensis | 1280          | 320                | 20907        |
| Lactuca sativa           | 640           | 160                | 6556         |
| Celtuce                  | 160           | 40                 | 802          |
| Film covered lettuce     | 640           | 160                | 6462         |
| Romaine lettuce          | 320           | 80                 | 2610         |
| Carrot                   | 320           | 80                 | 2817         |
| White radish             | 640           | 160                | 7912         |
| Garlic sprout            | 320           | 80                 | 3086         |
| Broad bean               | 160           | 40                 | 1128         |
| Tree                     | 320           | 80                 | 3640         |

  Download the dataset, the directory structure is as follows:

```bash
└──cache
  ├── data
     ├── data_mat
        ├── WHU_Hi_HongHu
           ├── WHU_Hi_HongHu.mat
           ├── WHU_Hi_HongHu_gt.mat
        ├── Indian
           ├── Indian_pines_corrected.mat
           ├── Indian_pines_gt.mat
        ├── Salinas
           ├── Salinas_corrected.mat
           ├── Salinas_gt.mat  
```

# [Environment Requirements](#contents)

- Hardware（Ascend/CPU）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Running on GPU

```bash
# run training example
# need set config_path in config.py file and set data_path in yaml file
python train.py --config_path [CONFIG_PATH] \
                --device_target GPU \
                --data_path [DATA_PATH]> train.log 2>&1 &
OR
bash scripts/run_train_gpu.sh [DATASET] [DATA_PATH]

# run evaluation example
# need set config_path in config.py file and set data_path, checkpoint_file_path in yaml file
python eval.py --config_path [CONFIG_PATH] \
               --device_target GPU \
               --ckpt_file [CKPT_FILE] \
               --data_path [DATA_PATH] > eval.log 2>&1 &
OR
bash scripts/run_eval_gpu.sh [CKPT_FILE] [DATASET] [DATA_PATH]
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```text
# run distributed training on modelarts example
# (1) Add "config_path='/path_to_code/config/Indian_pines.yaml'" on the website UI interface.
# (2) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (3) Set the code directory to "/path/3D-LWNet" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Add "config_path='/path_to_code/config/Indian_pines.yaml'" on the website UI interface.
# (2) Copy or upload your trained model to S3 bucket.
# (3) Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (4) Set the code directory to "/path/3D-LWNet" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└──3D-LWNet
  ├── README.md                             # descriptions about all the models
  ├── config                                # parameter configuration
    ├── Indian_pines.yaml
    ├── Salinas.yaml
    ├── WHU_Hi_HongHu.yaml
  ├── scripts
    ├── run_eval_gpu.sh                     # shell script for evaluation on GPU
    ├── run_train_gpu.sh                    # shell script for training on GPU
  ├── src
    ├── data_preprocess.py                  # Processing dataset
    ├── loss.py                             # contrastive_loss
    ├── models_lw_3D.py                     # 3D-LWNet network
    ├── model_utils
       ├──config.py                         # parameter configuration
       ├──device_adapter.py                 # device adapter
       ├──local_adapter.py                  # local adapter
       ├──moxing_adapter.py                 # moxing adapter
  ├── eval.py                               # eval net
  └── train.py                              # train net
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config file.

- Config for 3D-LWNet, Indian pines dataset

```text
"class_num": 16,                          # dataset class num
"batch_size": 64,                         # batch size of input tensor
"momentum": 0.9,                          # momentum
"weight_decay": 1e-5,                     # weight decay
"epoch_size": 30,                         # only valid for training, which is always 1 for inference
"save_checkpoint": True,                  # whether save checkpoint or not
"save_checkpoint_epochs": 3,              # the epoch interval between two checkpoints. By default, the last checkpoint will be saved after the last step
"keep_checkpoint_max": 10,                # only keep the last keep_checkpoint_max checkpoint
"warm": 0.1,                              # exponential decay function decay rate
"low_begin_epoch": 10,                    # number of warmup epoch
"model_name": "LWNet_3",                  # model name
"dataset_name": "Indian",                 # dataset name
"dataset_HSI": "indian_pines_corrected",  # name of hyperspectral data in data set dictionary
"dataset_gt": "indian_pines_gt",          # name of hyperspectral data label on dataset dictionary
"window_size": 27,                        # the size of the dataset padding process
"lr": 0.1,                                # initial learning rate
"save_graphs": False,                     # save graph results
"pre_trained": False                      # whether training based on the pre-trained model
```

## [Training Process](#contents)

### Running on GPU

#### Usage

```bash
# `CONFIG_PATH` `DATA_PATH` `DATASET` parameters need to be passed externally or modified yaml file
# `DATASET` must choose from ['Indian', 'Salinas', 'WHU']"
python train.py --config_path [CONFIG_PATH] \
                --device_target GPU \
                --data_path [DATA_PATH]> train.log 2>&1 &
OR
bash scripts/run_train_gpu.sh [DATASET] [DATA_PATH]
```

The python command above will run in the background, you can view the results through the file `train.log`.

#### Result

After training, you'll get some checkpoint files in `ckpt`. The loss value will be achieved as follows:

- Training 3D-LWNet with Indian Pines dataset

```python
# grep "loss is " train.log
epoch: 1 step: 22, loss is 2.301417112350464
epoch time: 36273.490 ms, per step time: 1648.795 ms
...
the best epoch is 21 best acc is 0.990625
```

- Training 3D-LWNet with Salinas dataset

```python
# grep "loss is " train.log
epoch: 1 step: 62, loss is 0.7641910314559937
epoch time: 66338.922 ms, per step time: 1069.983 ms
...
the best epoch is 6 best acc is 0.9989583333333333
```

- Training 3D-LWNet with WHU_Hi_HongHu dataset

```python
# grep "loss is " train.log
epoch: 1 step: 251, loss is 0.45947250723838806
epoch time: 196228.092 ms, per step time: 781.785 ms
...
the best epoch is 27 best acc is 0.9990671641791045
```

The model checkpoint will be saved in the `ckpt` directory.

## [Evaluation Process](#contents)

### Running on GPU

#### Usage

```bash
# `CONFIG_PATH` `CKPT_FILE` `DATA_PATH` `DATASET` parameters need to be passed externally or modified yaml file
# `DATASET` must choose from ['Indian', 'Salinas', 'WHU']"
python eval.py --config_path [CONFIG_PATH] \
               --device_target GPU \
               --ckpt_file [CKPT_FILE] \
               --data_path [DATA_PATH] > eval.log 2>&1 &
OR
bash scripts/run_eval_gpu.sh [CKPT_FILE] [DATASET] [DATA_PATH]
```

> checkpoint can be produced in training process.

#### Result

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

- Evaluating 3D-LWNet with Indian Pines dataset

```python
# grep "OA,AA,K: " eval.log
OA: 0.9849, AA: 0.9914, K: 0.9824
```

- Evaluating 3D-LWNet with Salinas dataset

```python
# grep "OA,AA,K: " eval.log
OA: 0.9977, AA: 0.9970, K: 0.9974
```

- Evaluating 3D-LWNet with WHU_Hi_HongHu dataset

```python
# grep "OA,AA,K: " eval.log
OA: 0.9985, AA: 0.9979, K: 0.9981
```

# [Model Description](#contents)

## [Performance](#contents)

### 3D-LWNet on Indian Pines

#### Training Performance

| Parameters          | GPU                                                                      |
|---------------------|--------------------------------------------------------------------------|
| Model Version       | 3D-LWNet                                                                 |
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7       |
| uploaded Date       | 8/15/2022 (month/day/year)                                               |
| MindSpore Version   | 1.6.1                                                                    |
| Dataset             | Indian Pines                                                             |
| Training Parameters | epoch=30, steps=22, batch_size=64                                        |
| Optimizer           | Momentum                                                                 |
| Loss Function       | NLLLoss                                                                  |
| outputs             | probability                                                              |
| Loss                | 0.0089                                                                   |
| Speed               | 1pc: 534.267 ms/step                                                     |
| Total time          | 1pc: 762s                                                                |
| Scripts             | [3D-LWNet script](https://gitee.com/zhangzzp_zzp/lwnet-master-mindspore) |

#### Evaluation Performance

| Parameters          | GPU                                                                |
|---------------------|--------------------------------------------------------------------|
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7 |
| Uploaded Date       | 8/15/2022 (month/day/year)                                         |
| MindSpore Version   | 1.6.1                                                              |
| Dataset             | Indian Pines                                                       |
| batch_size          | 64                                                                 |
| outputs             | OA, AA, K                                                          |
| OA                  | 0.9849                                                             |
| AA                  | 0.9914                                                             |
| K                   | 0.9824                                                             |
| Model for inference | 360.16M (.ckpt file)                                               |

### 3D-LWNet on Salinas

#### Training Performance

| Parameters          | GPU                                                                      |
|---------------------|--------------------------------------------------------------------------|
| Model Version       | 3D-LWNet                                                                 |
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7       |
| uploaded Date       | 8/15/2022 (month/day/year)                                               |
| MindSpore Version   | 1.6.1                                                                    |
| Dataset             | Salinas                                                                  |
| Training Parameters | epoch=6, steps=62, batch_size=64                                         |
| Optimizer           | Momentum                                                                 |
| Loss Function       | NLLLoss                                                                  |
| outputs             | probability                                                              |
| Loss                | 0.0337                                                                   |
| Speed               | 1pc: 722.904 ms/step                                                     |
| Total time          | 1pc: 2755s                                                               |
| Scripts             | [3D-LWNet script](https://gitee.com/zhangzzp_zzp/lwnet-master-mindspore) |

#### Evaluation Performance

| Parameters          | GPU                                                                |
|---------------------|--------------------------------------------------------------------|
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7 |
| Uploaded Date       | 8/15/2022 (month/day/year)                                         |
| MindSpore Version   | 1.6.1                                                              |
| Dataset             | Salinas                                                            |
| batch_size          | 64                                                                 |
| outputs             | OA, AA, K                                                          |
| OA                  | 0.9977                                                             |
| AA                  | 0.9970                                                             |
| K                   | 0.9974                                                             |
| Model for inference | 360.17M (.ckpt file)                                               |

### 3D-LWNet on WHU_Hi_HongHu dataset

#### Training Performance

| Parameters          | GPU                                                                      |
|---------------------|--------------------------------------------------------------------------|
| Model Version       | 3D-LWNet                                                                 |
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7       |
| uploaded Date       | 8/15/2022 (month/day/year)                                               |
| MindSpore Version   | 1.6.1                                                                    |
| Dataset             | WHU_Hi_HongHu dataset                                                    |
| Training Parameters | epoch=30, steps=251, batch_size=64                                       |
| Optimizer           | Momentum                                                                 |
| Loss Function       | NLLLoss                                                                  |
| outputs             | probability                                                              |
| Loss                | 0.0007                                                                   |
| Speed               | 1pc: 684.375 ms/step                                                     |
| Total time          | 1pc: 16401s                                                              |
| Scripts             | [3D-LWNet script](https://gitee.com/zhangzzp_zzp/lwnet-master-mindspore) |

#### Evaluation Performance

| Parameters          | GPU                                                                |
|---------------------|--------------------------------------------------------------------|
| Resource            | H3C UniServer R5300 G3; CPU 3GHz; 49cores; Memory 384G; OS CentOS7 |
| Uploaded Date       | 8/15/2022 (month/day/year)                                         |
| MindSpore Version   | 1.6.1                                                              |
| Dataset             | WHU_Hi_HongHu dataset                                              |
| batch_size          | 64                                                                 |
| outputs             | OA, AA, K                                                          |
| OA                  | 0.9985                                                              |
| AA                  | 0.9979                                                              |
| K                   | 0.9981                                                              |
| Model for inference | 360.20M (.ckpt file)                                               |

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
