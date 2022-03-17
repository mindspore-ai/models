# Contents

<!-- TOC -->

- [Contents](#contents)
- [StNet description](#stnet-description)
- [Environment requirements](#environment-requirements)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [script parameters](#script-parameters)
    - [training process](#training-process)
    - [evaluation process](#evaluation-process)
    - [export process](#Export-process)
    - [inference process](#inference-process)
- [Model description](#model-description)
    - [training performance](#training-performance)
    - [evaluation performance](#evaluation-performance)
- [Random description](#random-description)
- [Model structure description](#model-structure-description)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# StNet description

StNet is a video spatio-temporal joint modeling network framework that takes into account both local spatio-temporal connections and global spatio-temporal connections. It concatenates consecutive N frames of images in the video into a 3N-channel hypergraph, and then uses 2D convolution to perform local spatio-temporal connections on the hypergraph. In order to establish the global spatio-temporal correlation, StNet introduces a module that performs temporal convolution on multiple local spatio-temporal feature maps, and uses the temporal Xception module to further model the video feature sequence and mine the implied temporal dependencies.

[Original paper](https://arxiv.org/abs/1811.01549): "StNet: Local and Global Spatial-Temporal Modeling for Action Recognition."

# Environment requirements

- Hardware
    - Use the Ascend processor to build the hardware environment.
    - Or use Nvidia GPU
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/en/r1.7/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/r1.7/index.html)

# Dataset

**Important notice:** for now there are support for the following dataset configurations:

- Kinetics400 training for Ascend;
- UCF101 training for GPU.

The datasets must be preprocessed as described below.

## Kinetics400

[**Kinetics400 dataset**](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics):

Considering that some of the videos downloaded from YouTube are invalid or unusable, the filtered dataset labels are provided as follows
Dataset label link: [[HERE]](https://pan.baidu.com/s/1e58v4nrwzfYT459EZ3L_PA). The extraction code is: ms13.

After downloading, you can get the following directory:

  ```text
  └── data
      ├── train_mp4 .................................training set
      └── val_mp4 ............................. Validation set
  ```

- mp4 file preprocessing

In order to improve the data reading speed, the mp4 file is deframed and pickled in advance, and the dataloader reads the data from the pkl file of the video (this method consumes more storage space). The content packaged in the pkl file is (video-id,[frame1, frame2,...,frameN],label).

Create directories train_pkl and val_pkl under the data directory

  ```bash
  cd $Code_Root/data

  mkdir train_pkl && mkdir val_pkl
  ```

Enter the $Code_Root/src directory and use the video2pkl.py script for data conversion. First you need to download the file lists of the train and validation datasets.

First generate the dataset label file required for preprocessing

  ```bash
  python generate_label.py kinetics-400_train.csv kinetics400_label.txt
  ```

Then execute the following program (this script depends on the ffmpeg library, please install ffmpeg in advance):

  ```bash
  python video2pkl.py kinetics-400_train.csv $Source_dir $Target_dir 8 #Take 8 processes as an example
  ```

For train data,

  ```bash
  Source_dir = $Code_Root/data/train_mp4

  Target_dir = $Code_Root/data/train_pkl
  ```

For val data,

  ```bash
  Source_dir = $Code_Root/data/val_mp4

  Target_dir = $Code_Root/data/val_pkl
  ```

This will decode and save the mp4 file as a pkl file.

## UCF101

[**UCF101 dataset**](https://www.kaggle.com/pevogam/ucf101)

The UCF101 action recognition dataset contains 13,320 videos in 101 categories, all in avi format. The dataset consists of real user-uploaded videos that contain camera motion and cluttered backgrounds. However, the data set is not divided into training set and validation set, and users need to divide it by themselves.

After downloading the dataset via [here](https://www.kaggle.com/pevogam/ucf101), we get a classified video dataset, but the dataset is not further split into training and validation sets. Please organize the dataset in the following directory structure:

  ```text
  ├── xxx/ucf101/ // Dataset root directory
      │ ├──ApplyEyeMakeup
      │ ├──ApplyLipstick
      │ ├──....
  ```

Then set the parameters in the default_config.yaml file, and finally execute ```src/preprocess_ucf101.py``` and wait for the execution to complete. Example of running the script:

  ```bash
  # run with '-h' argument to get help
  python src/preprocess_ucf101.py -h

  # example
  python src/preprocess_ucf101.py --source_data_dir ~/Datasets/UCF101  --output_dir ~/Datasets/UCF101_img
  ```

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation. For distributed training, you need to create an hccl configuration file in JSON format in advance. Please follow the instructions in the link below:
 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- Before start, you should configure following parameters in ```default_config.yaml``` file to train on UCF101 or Kinetics400:

  ```text
  dataset_type: "ucf101"  # or "kinetics400"

  class_num: 101  # set 400 for kinetics of 101 for ucf
  avgpool_kernel_size: 4  # set 7 for kinetics or 4 for ucf
  ```

- Dataset preprocessing

  Dataset should be pre-processed as described in [Dataset](#dataset) section.

- Pre-trained models

  StNet model uses ResNet-50 network trained on ImageNet as backbone. You can run the Resnet training script in [official model zoo](https://gitee.com/mindspore/models/tree/master/official/cv/resnet) to get the model weight file or download trained checkpoint from [here](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs32_acc77.06/). The pre-training file name should be resnet50.ckpt.

- Ascend processor environment to run

  For now only training on Kinetics400 is supported on Ascend.

  ```python
  # Add pre-trained resnet50 parameters during training, set save parameters and summary location
  pre_res50:Code_Root/data/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76.ckpt
  checkpoint_path summary_dir
  # During offline training
  run_online = False
  ```

  ```bash
  # run the training example
  python train.py --target='Ascend' --device_id=0 --dataset_path='' --run_distribute False --resume=''(optional)

  # run the distributed training example
  bash scripts/run_distribute_train_ascend.sh [DATASET_PATH] [rank_table_PATH] [PRETRAINED_CKPT_PATH](optional)

  # run the evaluation example
  bash scripts/run_eval.sh [DEVICE_TARGET] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

  For distributed training, you need to create an hccl configuration file in JSON format in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- Train on ModelArts (if you want to run on modelarts, you can refer to the following document [modelarts](https://support.huaweicloud.com/modelarts/))

  ```text
  # (1) Select upload code to S3 bucket
  # Select the code directory /s3_path_to_code/StNet/
  # Select the startup file/s3_path_to_code/StNet/train.py
  # (2) Set parameters in config.py
  # run_online = True
  # data_url = [Location of dataset in S3 bucket]
  # local_data_url = [Location of dataset on cloud]
  # pre_url = [location of pretrained resnet50, resume, train and val labels in S3 bucket]
  # pre_res50_art_load_path = [location of pre-trained resnet50 on cloud]
  # best_acc_art_load_path = [location of pretrained model on cloud] or [not set]
  # load_path = [Location of pretrained resnet50, resume, train and val labels on cloud]
  # train_url = [the location of the result output in the S3 bucket]
  # output_path = [the location of the result output on the cloud]
  # local_train_list = [Location of training set labels on cloud]
  # local_val_list = [Location of validation set labels on cloud]
  # [additional parameters] = [parameter value]
  # (3) Upload the Kinetics-400 dataset to the S3 bucket. Since the processed dataset is too large, it is recommended to split the training set into 16 compressed files and the validation set into 2 compressed files, and configure "training data" set" path
  # (4) Set "training output file path", "job log path", etc. on the web page
  # (5) Select an 8-card machine and create a training job
  ```

- Train on GPU:

  For now only training on UCF101 is supported on GPU.

  ```bash
  # run the training example
  bash scripts/run_standalone_train_gpu.sh [CONFIG] [DATASET] [PRETRAINED_CKPT_PATH](optional)

  # run the distributed training example
  bash scripts/run_distribute_train_gpu.sh [NUM_DEVICES] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG] [DATASET] [PRETRAINED_CKPT_PATH](optional)

  # run the evaluation example
  bash scripts/run_eval.sh [DEVICE_TARGET] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

# Script description

## Script and sample code

```text
├── scripts
    ├── run_distribute_train_ascend.sh // ascend distributed training script
    ├── run_distribute_train_gpu.sh // gpu distributed training script
    ├── run_eval.sh // verification script
    └── run_standalone_train_gpu.sh // gpu standalone training script
├── src
    ├── model_utils
        └── moxing_adapter.py // model_art transfer file
    ├── eval_callback.py // Verify callback script
    ├── config.py // Configuration parameter script
    ├── dataset.py // dataset script
    ├── Stnet_Res_model.py // model script
    ├── generate_label.py // preprocessing to generate label script
    ├── preprocess_ucf101.py.py // preprocessing ucf101 dataset
    └── video2pkl.py // preprocessing video to pkl format script
├── README_CN.md // chinese description file
├── README_EN.md // english description file
├── eval.py // test script
├── requirements.txt  // pip requirements
└── train.py // training script
```

## script parameters

- Both training parameters and evaluation parameters can be configured in ```default_config.yaml```.

  ````text
  dataset_type: "ucf101"  # or "kinetics400"
  batch_size: 16  # batch size of input tensor
  num_epochs: 80  # This value applies to training only; fixed to 1 when applied to inference
  class_num: 101  # number of dataset classes
  T: 7  # number of fragments
  N:5  # Fragment length
  mode: 'GRAPH'  # training mode
  resume: ''  # pretrained model
  momentum: 0.9  # momentum
  lr: 0.001  # initial learning rate
  ````

## training process

To start training from scratch, **you need to load the ResNet50 weights trained on ImageNet as initialization parameters**, please download this [model parameters](https://download.mindspore.cn/model_zoo/r1.3/resnet50_ascend_v130_imagenet2012_official_cv_bs32_acc77.06/), save the parameters in code root directory below.
You can download the released model and use `--resume` to specify the weight storage path for finetune and other development

### Single card training

- Ascend processor environment to run

  For now only training on Kinetics400 is supported on Ascend.

  ```bash
  python train.py --target='Ascend' --device_id=0 --dataset_path='' --run_distribute False --resume=''(optional)
  ```

- GPU environment

  For now only training on UCF101 is supported on GPU.

  ```bash
  # standalone training
  bash scripts/run_standalone_train_gpu.sh [CONFIG] [DATASET] [PRETRAINED_CKPT_PATH](optional)

  # for example
  bash scripts/run_standalone_train_gpu.sh default_config.yaml ~/Datasets/UCF101_img resnet50_ascend_v130_imagenet2012_official_cv_bs32_acc77.06.ckpt

  ```

### Distributed training

- Ascend processor environment to run

  For now only training on Kinetics400 is supported on Ascend.

  ```bash
  bash scripts/run_distribute_train_ascend.sh [DATASET_PATH] [rank_table_PATH] [PRETRAINED_CKPT_PATH](optional)
  ```

  The above shell script will run distribution training in the background. You can view the results through the train_parallel[X]/log file.

- GPU environment

  For now only training on UCF101 is supported on GPU.

  ```bash
  # distributed training
  bash scripts/run_distribute_train_gpu.sh [NUM_DEVICES] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [CONFIG] [DATASET] [PRETRAINED_CKPT_PATH](optional)

  # for example
  bash scripts/run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7 default_config.yaml ~/Datasets/UCF101_img resnet50.ckpt
  ```

## Evaluation process

### evaluate

- Evaluate while running in the Ascend or GPU environment (you must set DEVICE_TARGET parameter)

  Before running the following command, check the checkpoint path used for evaluation. Please set the checkpoint path to an absolute full path, such as "username/stnet/best_acc.ckpt".

  ```bash
  # evaluation script
  bash scripts/run_eval.sh [DEVICE_TARGET] [DATASET_PATH] [CHECKPOINT_PATH]

  # for example
  bash scripts/run_eval.sh GPU ~/Datasets/UCF101_img train/ckpt_0/stnet-90_83.ckpt
  ```

  The above python command will run in the background and you can view the result through the eval.log file. The accuracy of the test dataset is as follows:

  ```text
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.69}
  ```

## Export process

### export

Before exporting, you need to modify the configuration file config.py corresponding to the dataset
The configuration items that need to be modified are batch_size and ckpt_file.

  ```bash
  python export.py --resume [CONFIG_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

## Inference process

### Inference

Before we can run inference we need to export the model first. Air models can only be exported on the Ascend 910 environment, mindir can be exported on any environment. batch_size only supports 1.

- Inference using Kinetics-400 dataset on Shengteng 310

  Before executing the following command, we need to modify the configuration file of confi.py. Modified items include batch_size.

  The results of inference are saved in the current directory, and results similar to the following can be found in the acc.log log file.

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
  after allreduce eval: top1_correct=9252, tot=10000, acc=92.52%
  ```

Four parameters are required:

-MINDIR_PATH: Absolute path to the MINDIR file

-DATA_PATH: Unprocessed eval dataset path, if NEED_PREPROCESS is N, this parameter can be filled with ""

-NEED_PREPROCESS: Whether to process the eval dataset

-DEVICE_ID: The chip number used on the 310

# Model description

## Performance

### Training performance

#### Train StNet on Kinetics-400 (8 cards)

| Parameters          | Ascend                                           | GPU                                 |
| ------------------- | ------------------------------------------------ | ----------------------------------- |
| Model Version       | StNet ResNet50                                   | StNet ResNet50                      |
| Resource            | Ascend 910; CPU: 2.60GHz, 192 cores; RAM: 755 GB | GPU: 8x RTX3090 24GB; RAM: 252 GB   |
| uploaded Date       | 2021-12-17                                       | 2022-03-17                          |
| MindSpore Version   | 1.3.0                                            | 1.6.0                               |
| Dataset             | Kinetics-400                                     | UCF101                              |
| Training Parameters | See default_config.yaml for details              | See default_config.yaml for details |
| Optimizer           | Momentum                                         | Momentum                            |
| Loss Function       | CrossEntropySmooth                               | CrossEntropySmooth                  |
| Speed               | 921 ms/step (0 cards)                            | 1262 ms/step (8 cards)              |
| Total time          | 33h 59 min 48s (0 cal)                           | 2 h 29 m 55 s (8 cards)             |
| Top1 accuracy       | 69.16%                                           | 97.76%                              |
| Parameters          | 301                                              | 371 MB                              |

### Evaluate performance

#### Evaluation on Kinetics-400

the performance of Top1 is 69.16%.

### Evaluation on UCF101

the performance of Top1 is 97.76%.

# Random description

In dataset.py, we set the seed inside the "create_dataset" function, while also using the random seed from train.py.

# Model structure description

In TemporalXception, since MindSpore 1.3 version does not support packet convolution, it is commented out. If the version is 1.5, you can try.

# ModelZoo homepage  

Please visit the official website [home page](https://gitee.com/mindspore/models).
