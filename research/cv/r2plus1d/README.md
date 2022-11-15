# R(2+1)D

<!-- TOC -->

- [R(2+1)D](#r21d)
- [R(2+1)D Introduction](#r21d-introduction)
- [Model structure](#model-structure)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Quickstart](#quick-start)
- [Script description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Training](#training)
    - [Evaluation process](#evaluation-process)
        - [Evaluate](#evaluate)
- [Mindir reasoning](#mindir-inference)
    - [Export Mindir](#export-mindir)
    - [Perform Inference on Ascend310](#perform-inference-on-ascend310)
    - [Result](#result)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Evaluate performance](#evaluating-performance)
- [Random Situation Description](#random-situation-description)
- [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

## R(2+1)D introduction

​ R(2+1)D: A Closer Look at Spatiotemporal Convolutions for Action Recognition; from Facebook Research & Dartmouth Colledge, through in-depth exploration of various convolution blocks in action recognition, a new network for video action recognition is proposed Structure: R(2+1)D. The inspiration for this paper is that 2D convolution on a single frame of video can still achieve SOTA results close to 3D spatiotemporal convolution methods. The paper shows that decomposing 3D convolutional filters into separate spatiotemporal components can significantly improve accuracy, so a new spatiotemporal convolution block "R(2+1)D" is designed, which produces a CNN that can reach Sports-1M, Comparable or better results to the best performance on Kinetics, UCF101 and HMDB51.

[Paper](https://arxiv.org/abs/1711.11248v1): Du T , Wang H , Torresani L , et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition[C]// IEEE/CVF Conference on Computer Vision and Pattern Recognition. 0.

## Model structure

The most important structure of this article is (2+1)D convolution, which decomposes 3-dimensional space-time convolution into 2-dimensional space convolution and 1-dimensional time convolution, then the size of the convolution kernel becomes N'×1×d ×d + M×t×1*1, the hyperparameter M determines the number of subspaces where the signal is projected between the spatiotemporal convolution. After decomposition, there will be an additional nonlinear operation between the two subconvolutions. Compared with the original 3-dimensional convolution with the same parameter amount, the nonlinear operation is doubled, which is equivalent to expanding the network. Moreover, the space-time decomposition also decomposes the optimization process. In fact, it was found before that the 3-dimensional space-time convolution twists spatial information and dynamic information together, which is not easy to optimize, while the 2+1-dimensional convolution is easier to optimize, and the loss will lower.

The 5-dimensional data (NCTHW) input by the network goes through five (2+1)D convolution operations in turn, then goes through a spatiotemporal information pooling layer, and finally goes through a fully connected layer to get the final video action classification result.

## Dataset

- [Pre-training dataset Kinetics400](https://deepmind.com/research/open-source/kinetics)

   The Kinetics400 dataset includes 400 human action categories, each with at least 400 video clips, each taken from a different Youtube video, lasting about ten seconds. There are three types of video suffixes: 'mp4', 'webm', 'mkv'.

  The official website has labels for the training set and validation set, but the labeling for the test set has not been published.

- [Transfer Learning Dataset UCF101](https://www.kaggle.com/pevogam/ucf101)

  The UCF101 action recognition dataset contains 13,320 videos in 101 categories, all in avi format. The dataset consists of real user-uploaded videos that contain camera motion and cluttered backgrounds. However, the data set is not divided into training set and validation set, and users need to divide it by themselves.

  Data content structure:

  ```text
  .
  └─ucf101                                     // contains 101 file folder
    ├── ApplyEyeMakeup                        // contains 145 videos
    │   ├── v_ApplyEyeMakeup_g01_c01.avi     // video file
    │   ├── v_ApplyEyeMakeup_g01_c02.avi     // video file
    │    ...
    ├── ApplyLipstick                         // contains 114 image files
    │   ├── v_ApplyLipstick_g01_c01.avi      // video file
    │   ├── v_ApplyLipstick_g01_c02.avi      // video file
    │    ...
    ...
  ```

## Environmental requirements

- Hardware (Ascend/ModelArts)
    - Prepare Ascend or ModelArts to build the hardware environment.
- frame
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## Quick start

After installing MindSpore through the official website, you can perform training and evaluation as follows **(please perform dataset preprocessing and pretraining model conversion steps before training and evaluation)**.

Configure paths to datasets and other parameters in configuration file (```default_config.yaml```).

- Ascend environment:

  ```bash
  ###Parameter configuration please modify the default_config.yaml file

  #Start training:
  #Run the Ascend single-card training script through the python command line.
  python train.py --is_distributed=0 --device_target=Ascend > train_ascend_log.txt 2>&1 &

  #Start Ascend single-card training through the bash command.
  bash scripts/run_train_ascend.sh [device_id]
  # for example:
  bash scripts/run_train_ascend.sh 0

  #Ascend distributed training.
  bash scripts/run_distribute_train_ascend.sh [rank_size] [rank_start_id] [rank_table_file]
  # for example:
  bash scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json

  # run evaluation script:
  bash scripts/run_eval.sh [device_target] [device_id] [checkpoint]
  # for example:
  bash scripts/run_eval.sh Ascend 0 ../r2plus1d_best.ckpt
  ```

  Ascend training: generate [RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

- GPU environment:

  ```bash
  # start GPU standalone train
  bash scripts/run_train_gpu.sh [device_id]
  # for example
  bash scripts/run_train_gpu.sh 0

  # start GPU distributed training
  bash scripts/run_distribute_train_gpu.sh [num_devices] [cuda_visible_devices(0,1,2,3,4,5,6,7)]
  # for example
  bash scripts/run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7

  # run evaluation script
  bash scripts/run_eval.sh [device_target] [device_id] [checkpoint]
  # for example
  bash scripts/run_eval.sh GPU 0 ../r2plus1d_best.ckpt
  ```

## Script description

### Script and sample code

```text
├── model_zoo
    ├── README.md // Description file for all models
    ├── R(2+1)D
        ├── README_CN.md // R(2+1)D description file
        ├── ascend310_infer // 310 inference main code folder
        | ├── CMakeLists.txt // CMake settings file
        | ├── build.sh // Compile startup script
        | ├── inc
        | | ├── utils.h // utility header file
        | ├── src
        | | ├── main.cc // Inference code file
        | | ├── utils.cc // Utilities file
        ├── scripts
        │ ├──run_distribute_train_ascend.sh // Ascend 8 card training script
        │ ├──run_distribute_train_gpu.sh // GPU multiple card training script
        │ ├──run_eval.sh // Evaluation startup script
        │ ├──run_train_ascend.sh // Ascend single card training startup script
        │ ├──run_train_gpu.sh // GPU single card training startup script
        | ├──run_infer_310.sh // 310 inference startup script
        ├── src
        │ ├──config.py // Configuration loading file
        │ ├──dataset.py // dataset processing
        │ ├──models.py // Model structure
        │ ├──logger.py // log print file
        │ ├──utils.py // Utilities
        ├── default_config.yaml // Default configuration information, including training, inference, model freezing, etc.
        ├── train.py // training script
        ├── eval.py // inference script
        ├── dataset_process.py // dataset preprocessing script
        ├── export.py // A script to freeze the weight file into a format such as MINDIR
        ├── transfer_pth.py // Script to convert pth weight file to ckpt weight
        ├── postprocess.py // 310 precision calculation script
        ├── preprocess.py // 310 preprocessing script
```

### Script parameters

```text
Model training, inference, freezing and other operations and the parameters of the model deployment environment are configured in the default_config.yaml file.
The key parameters (transfer learning for the UCF101 dataset) default to the following:
num_classes: 101
layer_num: 34
epochs: 30
batch_size: 8
lr: 0.001
momentum: 0.9
weight_decay: 0.0005
```

### Dataset Preprocessing

Since the obtained original dataset is in video format (the input of network processing is in image format), and probably has not yet been classified, dataset preprocessing is required before model training and inference are performed. The dataset preprocessing script is "dataset_process.py", and its parameters are set in the default_config.yaml file

Configuration, this section only involves the three parameters "splited", "source_data_dir", and "output_dir".

Note: The decord package required for dataset preprocessing cannot be installed on aarch64 architecture machines, please use x86 architecture machines.

- Preprocessing of Kinetics400 dataset

  After downloading the dataset through the [official website](https://deepmind.com/research/open-source/kinetics), we will get an unclassified training set and a classified validation set, please use the official script for the training set Categorize and then organize the dataset in the following directory structure:

  ```text
  ├── xxx/kinetics400/ // Dataset root directory
      │ ├──train // training set
      │ │ ├── abseiling
      │ │ ├── air_drumming
      │ │ ├── ....
      │ ├──val // validation set
      │ │ ├── abseiling
      │ │ ├── air_drumming
      │ │ ├── ....
  ```

  Then set the "splited", "source_data_dir", and "output_dir" parameters in the default_config.yaml file to 1, "xxx/kinetics400/", "./kinetics400_img" in turn, and finally execute python dataset_process.py and wait for the execution to complete. .

- Preprocessing of UCF101 dataset

  After downloading the dataset via [here](https://www.kaggle.com/pevogam/ucf101), we get a classified video dataset, but the dataset is not further split into training and validation sets . Please organize the dataset in the following directory structure:

  ```text
  ├── xxx/ucf101/ // Dataset root directory
      │ ├──ApplyEyeMakeup
      │ ├──ApplyLipstick
      │ ├──....
  ```

  Then set the "splited", "source_data_dir", and "output_dir" parameters in the default_config.yaml file to 0, "xxx/ucf101/", "./ucf101_img" in turn, and finally execute python dataset_process.py and wait for the execution to complete. .

### Pretrained model conversion

Please download from [here](https://cv.gluon.ai/model_zoo/action_recognition.html) the pretrained model [r2plus1d_v1_resnet34_kinetics400.ckpt](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/torch/models/r2plus1d_v1_resnet34_kinetics400-5102fd17.pth) and then execute the transfer_pth.py script to convert the model to the format required by mindspore.

```bash
#Please put the downloaded pre-training model r2plus1d_v1_resnet34_kinetics400-5102fd17.pth in the same directory as transfer_pth.py, and then execute the following command
python transfer_pth.py
```

### Training process

#### Training

##### ModelArts environment running

  Taking the UCF101 data set as an example, after the aforementioned "data set preprocessing" step, we will get a data set with the following structure:

  ```text
  ├── xxx/ucf101/ // Dataset root directory
      │ ├──train // training set
      │ │ ├── ApplyEyeMakeup
      │ │ ├── ApplyLipstick
      │ │ ├── ....
      │ ├──val // validation set
      │ │ ├── ApplyEyeMakeup
      │ │ ├── ApplyLipstick
      │ │ ├── ....
  ```

  Please directly compress the root directory "xxx/ucf101/" into a compressed package in zip format, name it ucf101_img.zip, and upload it to the OBS bucket.

  Your OBS buckets should be organized as follows:

  ```text
  ├── xxx/R2plus1D/ // root directory
  │ ├──code // code directory
  │ │ ├──README_CN.md // R(2+1)D description file
  │ │ ├──scripts
  │ │ │ ├──run_distribute_train_ascend.sh // Ascend 8 card training script
  │ │ │ ├──run_distribute_train_gpu.sh // GPU multiple card training script
  │ │ │ ├──run_eval.sh // Evaluation startup script
  │ │ │ ├──run_train_ascend.sh // Ascend single card training startup script
  │ │ │ ├──run_train_gpu.sh // GPU single card training startup script
  │ │ | ├──run_infer_310.sh // 310 inference startup script
  │ │ ├──src
  │ │ │ ├──config.py // Configuration loading file
  │ │ │ ├──dataset.py // dataset processing
  │ │ │ ├──models.py // Model structure
  │ │ │ ├──logger.py // log print file
  │ │ │ ├──utils.py // Utilities
  │ │ ├──default_config.yaml // Default configuration information, including training, inference, model freezing, etc.
  │ │ ├──train.py // training script
  │ │ ├──eval.py // inference script
  │ │ ├──dataset_process.py // dataset preprocessing script
  │ │ ├──export.py // A script to freeze the weight file into a format such as MINDIR
  │ │ ├──transfer_pth.py // Script to convert pth weight file to ckpt weight
  │ ├──dataset
  │ │ ├──ucf101_img.zip // dataset
  │ ├──pretrain
  │ │ ├──r2plus1d_v1_resnet34_kinetics400.ckpt // Converted pre-training model
  │ ├──outputs // Store training logs and other files
  ```

  Then go to ModelArts console ---> Training Management ---> Training Job ---> Create, the specific parameters are as follows:

  Algorithm source: Common framework

  ​ AI Engine Ascend-Powered-Engine MindSpore-1.3.0-c78-python3.7-euleros2.8-aarch64
  ​ Code directory /xxx/R2plus1D/code
  ​ Startup file /xxx/R2plus1D/code/train.py

  Data source: Data storage location

  ​ Data storage location: /xxx/R2plus1D/dataset

  Training output location: /xxx/R2plus1D/outputs

  Operating parameters:

  ​ train_url: /xxx/R2plus1D/outputs

  ​ data_url: /xxx/R2plus1D/dataset

  ​ use_modelarts: 1

  ​ outer_path: obs://xxx/R2plus1D/outputs

  ​ dataset_root_path: obs://xxx/R2plus1D/dataset

  ​ pack_file_name: ucf101_img.zip

  ​ pretrain_path: obs://xxx/R2plus1D/pretrain/

  ​ ckpt_name: r2plus1d_v1_resnet34_kinetics400.ckpt

  ​ is_distributed: 0

  Job log path: /xxx/R2plus1D/outputs

  Then select the specification: "Ascend : 1 * Ascend-910 CPU: 24 cores 256GiB" to create a training task.

  If you want to create an 8-card training task, just set "is_distributed" in the above running parameters to 1, then select the specification "Ascend: 8 * Ascend-910 CPU: 192 cores 2048GiB" to create a training task.

##### Ascend processor environment to run

  ```bash
  ### Parameter configuration please modify the default_config.yaml file

  #Start Ascend single-card training through the bash command.
  bash ./scripts/run_train_ascend.sh device_id
  #e.g.
  bash ./scripts/run_train_ascend.sh 0

  #Ascend Doka Training
  bash ./scripts/run_distribute_train_ascend.sh rank_size rank_start_id rank_table_file
  #e.g.
  bash ./scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json
  #Ascend and more
  #Card training will create an ascend_work_space folder in the code root directory, and run it independently in this working directory, saving relevant training information.
  ```

  After the training is complete, you can find the saved weight file in the directory specified by the output_path parameter. The partial loss convergence during the training process is as follows (8 cards in parallel):

  ```text
  ###Parameter configuration please modify the default_config.yaml file
  ......
  epoch time: 214265.706 ms, per step time: 1290.757 ms
  epoch: 20 step: 6, loss is 0.004272789
  epoch: 20 step: 16, loss is 0.062011678
  epoch: 20 step: 26, loss is 1.2200212
  epoch: 20 step: 36, loss is 0.20649293
  epoch: 20 step: 46, loss is 0.110879965
  epoch: 20 step: 56, loss is 0.019843677
  epoch: 20 step: 66, loss is 0.0016696296
  epoch: 20 step: 76, loss is 0.028821332
  epoch: 20 step: 86, loss is 0.022604007
  epoch: 20 step: 96, loss is 0.050388362
  epoch: 20 step: 106, loss is 0.03981915
  epoch: 20 step: 116, loss is 1.77048
  epoch: 20 step: 126, loss is 0.46865237
  epoch: 20 step: 136, loss is 0.006930205
  epoch: 20 step: 146, loss is 0.01725213
  epoch: 20 step: 156, loss is 0.15757804
  epoch: 20 step: 166, loss is 0.77281004
  epoch time: 185916.069 ms, per step time: 1119.976 ms
  [WARNING] MD(19783,fffdd97fa1e0,python):2021-12-02-16:04:30.120.203 [mindspore/ccsrc/minddata/dataset/engine/datasetops/device_queue_op.cc:725] DetectPerBatchTime] Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset pipeline, which might result in `GetNext` timeout problem. You may test dataset processing performance(with creating dataset iterator) and optimize it.
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.416 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.498 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.525 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.546 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.566 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] DEVICE(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.585 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[Equal] reduce precision from int64 to int32
  [WARNING] SESSION(19783,fffe99ffb1e0,python):2021-12-02-16:04:37.417.736 [mindspore/ccsrc/backend/session/ascend_session.cc:1205] SelectKernel] There has 1 node/nodes used reduce precision to selected the kernel!
  2021-12-02 16:09:50,480 :INFO: epoch: 20, accuracy: 97.52252
  2021-12-02 16:09:54,156 :INFO: update best result: 97.52252
  2021-12-02 16:10:47,968 :INFO: update best checkpoint at: ./output/2021-12-02_time_14_41_28/0_best_map.ckpt
  epoch: 21 step: 10, loss is 0.30040452
  epoch: 21 step: 20, loss is 0.04393909
  epoch: 21 step: 30, loss is 0.26733813
  epoch: 21 step: 40, loss is 0.35622913
  epoch: 21 step: 50, loss is 0.14869432
  epoch: 21 step: 60, loss is 0.45824617
  epoch: 21 step: 70, loss is 0.031756364
  epoch: 21 step: 80, loss is 0.07024868
  epoch: 21 step: 90, loss is 0.3892364
  epoch: 21 step: 100, loss is 3.364152
  epoch: 21 step: 110, loss is 0.48548156
  epoch: 21 step: 120, loss is 2.4292169
  epoch: 21 step: 130, loss is 0.24383453
  epoch: 21 step: 140, loss is 0.31997812
  epoch: 21 step: 150, loss is 0.0057518715
  epoch: 21 step: 160, loss is 0.009464129
  epoch time: 62759.866 ms, per step time: 378.071 ms
  ......
  ```

##### GPU environment to run

  ```bash
  ### Parameter configuration please modify the default_config.yaml file

  # start GPU standalone train:
  bash scripts/run_train_gpu.sh [device_id]
  # for example
  bash scripts/run_train_gpu.sh 0

  # start GPU distributed training:
  bash scripts/run_distribute_train_gpu.sh [num_devices] [cuda_visible_devices(0,1,2,3,4,5,6,7)]
  # for example:
  bash scripts/run_distribute_train_gpu.sh 8 0,1,2,3,4,5,6,7
  ```

### Evaluation Process

#### evaluate

Before running the following command, please check that the weights file path for inference evaluation is correct.

- Ascend processor environment to run

  ```bash
  ### Please modify the default_config.yaml file for parameter configuration
  # run evaluation script:
  bash scripts/run_eval.sh [device_target] [device_id] [checkpoint]
  # for example:
  bash scripts/run_eval.sh Ascend 0 ../r2plus1d_best.ckpt
  ```

  After the run is complete, you can find the inference run log in the directory specified by output_path. Some reasoning logs are as follows:

  ```text
  2021-12-01 15:41:35,434:INFO:Args:
  2021-12-01 15:41:35,434:INFO:-->use_modelarts: 0
  2021-12-01 15:41:35,434:INFO:-->data_url:
  2021-12-01 15:41:35,434:INFO:-->train_url:
  2021-12-01 15:41:35,434:INFO:--> outer_path: s3://output/
  2021-12-01 15:41:35,434:INFO:-->num_classes: 101
  2021-12-01 15:41:35,434:INFO:-->layer_num: 34
  2021-12-01 15:41:35,434:INFO:--> epochs: 30
  2021-12-01 15:41:35,435:INFO:--> batch_size: 8
  2021-12-01 15:41:35,435:INFO:-->lr: 0.001
  2021-12-01 15:41:35,435:INFO:--> momentum: 0.9
  2021-12-01 15:41:35,435:INFO:-->weight_decay: 0.0005
  2021-12-01 15:41:35,435:INFO:-->dataset_root_path: /opt/npu/data/R2p1D
  /dataset/ucf101_img/
  2021-12-01 15:41:35,435:INFO:--> dataset_name: ucf101
  2021-12-01 15:41:35,435:INFO:--> val_mode: val
  2021-12-01 15:41:35,435:INFO:-->pack_file_name:
  2021-12-01 15:41:35,435:INFO:--> eval_while_train: 1
  2021-12-01 15:41:35,435:INFO:--> eval_steps: 1
  2021-12-01 15:41:35,435:INFO:--> eval_start_epoch: 20
  2021-12-01 15:41:35,435:INFO:-->save_every: 1
  2021-12-01 15:41:35,436:INFO:-->is_save_on_master: 1
  2021-12-01 15:41:35,436:INFO:-->ckpt_save_max: 5
  2021-12-01 15:41:35,436:INFO:--> output_path: ./output/
  2021-12-01 15:41:35,436:INFO:--> pretrain_path: /opt/npu/data/R2p1D/code_check/
  2021-12-01 15:41:35,436:INFO:-->ckpt_name: r2plus1d_v1_resnet34_kinetics400.ckpt
  2021-12-01 15:41:35,436:INFO:--> resume_path:
  2021-12-01 15:41:35,436:INFO:--> resume_name:
  2021-12-01 15:41:35,436:INFO:--> resume_epoch: 0
  2021-12-01 15:41:35,436:INFO:--> eval_ckpt_path: /opt/npu/data/R2p1D/code_check/
  2021-12-01 15:41:35,436:INFO:--> eval_ckpt_name: r2plus1d_best_map.ckpt
  2021-12-01 15:41:35,436:INFO:--> export_batch_size: 1
  2021-12-01 15:41:35,436:INFO:-->image_height: 112
  2021-12-01 15:41:35,437:INFO:-->image_width: 112
  2021-12-01 15:41:35,437:INFO:-->ckpt_file: ./r2plus1d_best_map.ckpt
  2021-12-01 15:41:35,437:INFO:-->file_name: r2plus1d
  2021-12-01 15:41:35,437:INFO:--> file_format: MINDIR
  2021-12-01 15:41:35,437:INFO:-->source_data_dir: /data/dataset/UCF-101/
  2021-12-01 15:41:35,437:INFO:--> output_dir: ../dataset/ucf101_img/
  2021-12-01 15:41:35,437:INFO:-->splited: 0
  2021-12-01 15:41:35,437:INFO:-->device_target: Ascend
  2021-12-01 15:41:35,437:INFO:-->is_distributed: 0
  2021-12-01 15:41:35,437:INFO:-->rank: 0
  2021-12-01 15:41:35,437:INFO:-->group_size: 1
  2021-12-01 15:41:35,437:INFO:-->config_path: /opt/npu/data/R2p1D/code_check/src/../default_config.yaml
  2021-12-01 15:41:35,438:INFO:--> save_dir: ./output/2021-12-01_time_15_41_35
  2021-12-01 15:41:35,438:INFO:--> logger: <LOGGER R2plus1D (NOTSET)>
  2021-12-01 15:41:35,438:INFO:
  2021-12-01 15:41:37,742:INFO:load validation weights from /opt/npu/data/R2p1D/code_check/r2plus1d_best_map.ckpt
  2021-12-01 15:41:46,801:INFO:loaded validation weights from /opt/npu/data/R2p1D/code_check/r2plus1d_best_map.ckpt
  2021-12-01 15:41:48,838:INFO:cfg.steps_per_epoch: 333
  [WARNING] DEVICE(11897,fffe56ffd1e0,python):2021-12-01-15:42:03.243.372 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[OneHot] reduce precision from int64 to int32
  [WARNING] DEVICE(11897,fffe56ffd1e0,python):2021-12-01-15:42:03.243.467 [mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc:284] TagRaiseReduce] Node:[OneHot] reduce precision from int64 to int32
  [WARNING] SESSION(11897,fffe56ffd1e0,python):2021-12-01-15:42:06.867.318 [mindspore/ccsrc/backend/session/ascend_session.cc:1806] SelectKernel] There are 1 node/nodes used reduce precision to selected the kernel!
  2021-12-01 15:44:53,346:INFO:Final Accuracy: {'top_1_accuracy': 0.9786036036036037, 'top_5_accuracy': 0.9981231231231231}
  2021-12-01 15:44:53,347:INFO:validation finished....
  2021-12-01 15:44:53,456:INFO:All task finished!
  ```

- GPU environment to run

  ```bash
  ### Please modify the default_config.yaml file for parameter configuration
  # run evaluation script:
  bash scripts/run_eval.sh [device_target] [device_id] [checkpoint]
  # for example:
  bash scripts/run_eval.sh GPU 0 ../r2plus1d_best.ckpt
  ```

## Mindir Inference

### Export Mindir

```bash
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
#e.g.
python export.py --ckpt_file ./r2plus1d_best_map.ckpt --file_name r2plus1d --file_format MINDIR
```

- `ckpt_file` is required.
- `file_format` must be selected from ["AIR", "MINDIR"].

### Perform inference on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported via the `export.py` script. The following shows an example of performing inference using the mindir model.

```bash
# Ascend310 inference
bash run_infer_310.sh [model_path] [data_path] [out_image_path]
e.g. bash run_infer_310.sh ../r2plus1d.mindir ../dataset/ ../outputs
```

- `model_path` The path where the mindir file is located.
- `data_path` The path where the dataset is located (this path can only contain the preprocessed val directory).
- `out_image_path` The directory where the preprocessed data is stored, it will be automatically created if it does not exist.

### result

The inference results are saved under the ascend310_infer folder, and you can see the following precision calculation results in acc.log.

```text
Accuracy: 0.9774774774774775
```

## model description

### Performance

#### Evaluating performance

Validation for R(2+1)D

| Parameters | Ascend | GPU |
| -------------------------------------- | -----------------------------------------| -----------------------------------------|
| Resource | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G | 8x RTX 3090 24GB |
| uploaded Date | 11/27/2021 (month/day/year) | 03/02/2022 (month/day/year) |
| MindSpore Version | 1.5.0 | 1.6.0 |
| Dataset | UCF101 | UCF101 |
| Training Parameters | num_classes=101, layer_num=34, epochs=30, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005 | num_classes=101, layer_num=34, epochs=30, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005 |
| Optimizer | SGD | SGD |
| Loss Function | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| outputs | The probability that the input video belongs to each category | The probability that the input video belongs to each category |
| Loss| 0.2231637 | 0.2231637 |
| Accuracy | top_1=0.9786, top_5=0.9981 | top_1=0.9771, top_5=0.9969 |
| Total time | 8p: 1h58m (without validation), 1p: 3h (without validation) | 8p: 1h24m (without validation) |
| Checkpoint for Fine tuning | 8p: 706MB(.ckpt file) | 8p: 706MB(.ckpt file) |
| Scripts | [R(2+1)D Script](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d) | [R(2+1)D Script](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d) |

## random situation description

Random seeds are set in train.py , eval.py and preprocess.py .

## ModelZoo homepage

Please visit the official website [home page](https://gitee.com/mindspore/models).