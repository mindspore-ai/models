# Contents

- [Contents](#contents)
    - [C3D Description](#c3d-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
            - [Training on Ascend](#training-on-ascend)
        - [Distributed Training](#distributed-training)
            - [Distributed training on Ascend](#distributed-training-on-ascend)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
            - [Evaluating on Ascend](#training-on-ascend)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [C3D Description](#contents)

C3D model is widely used for 3D vision task. The construct of C3D network is similar to the common 2D ConvNets, the main difference is that C3D use 3D operations like Conv3D while 2D ConvNets are anentirely 2D architecture. To know more information about C3D network, you can read the original paper Learning Spatiotemporal Features with 3D Convolutional Networks.

## [Model Architecture](#contents)

C3D net has 8 convolution, 5 max-pooling, and 2 fully connected layers, followed by a softmax output layer. All 3D convolution kernels are 3 × 3 × 3 with stride 1 in both spatial and temporal dimensions. The 3D pooling layers are denoted from pool1 to pool5. All pooling kernels are 2 × 2 × 2, except for pool1 is 1 × 2 × 2. Each fully connected layer has 4096 output units.

## [Dataset](#contents)

Dataset used: [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

- Description: HMDB51 is collected from various sources, mostly from movies, and a small proportion from public databases such as the Prelinger archive, YouTube and Google videos. The dataset contains 6849 clips divided into 51 action categories, each containing a minimum of 101 clips.

- Dataset size：6849 videos
    - Note：Use the official Train/Test Splits([test_train_splits](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar)).
- Data format：rar
    - Note：Data will be processed in dataset_preprocess.py
- Data Content Structure

```text
.
└─hmdb-51                                // contains 51 file folder
  ├── brush_hair                        // contains 107 videos
  │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi      // video file
  │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1.avi      // video file
  │    ...
  ├── cartwheel                         // contains 107 image files
  │   ├── (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi       // video file
  │   ├── Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.avi   // video file
  │    ...
  ...
```

Dataset used: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

- Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

- Dataset size：13320 videos
    - Note：Use the official Train/Test Splits([UCF101TrainTestSplits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)).
- Data format：rar
    - Note：Data will be processed in dataset_preprocess.py
- Data Content Structure

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

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Processing raw data files

```text
# Convert video into image.
bash run_dataset_preprocess.sh HMDB51 [RAR_FILE_PATH] 1
# or
bash run_dataset_preprocess.sh UCF101 [RAR_FILE_PATH] 1
# for example: bash run_dataset_preprocess.sh HMDB51 /Data/hmdb51/hmdb51_org.rar 1
```

- Configure the dataset name and class number of the dataset in the YAML file

```text
dataset:         [DATASET_NAME]  # Name of dataset, 'HMDB51' or 'UCF101'
num_classes:     [CLASS_NUMBER]  # Number of total classes in the dataset, 51 for HMDB51 and 101 for UCF101
# for example:
# dataset:       HMDB51
# num_classes:   51
# or
# dataset:       UCF101
# num_classes:   101
```

- Configure the path of the processed dataset in the YAML file

```text
json_path:   [JSON_FILE_PATH]        # Path to the json file for the given dataset
img_path:    [IMAGE_FILE_PATH]       # Path to the image file for the given dataset
# for example
# json_path: /Data/hmdb-51_json/
# img_path: /Data/hmdb-51_img/
```

- Download pretrained model

```text
mkdir ./pretrained_model
# download C3D pretrained file
wget -O ./pretrained_model/c3d-pretrained.pth https://download.mindspore.cn/thirdparty/c3d_pretrained.pth
# download C3D Mean file
wget -O ./pretrained_model/sport1m_train16_128_mean.npy https://download.mindspore.cn/thirdparty/sport1m_train16_128_mean.npy
```

- Convert pretrained model (from pytorch to mindspore, both Pytorch and Mindspore must be installed.)

```text
# Convert pytorch pretrained model file to mindspore file.
bash run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
# for example: bash run_ckpt_convert.sh /home/usr/c3d/pretrain_model/c3d_pretrained.pth /home/usr/c3d/pretrain_model/c3d_pretrained_ms.ckpt
```

Refer to `default_config.yaml`. We support some parameter configurations for quick start.

- Run on Ascend

```bash
cd scripts
# run training example
bash run_standalone_train_ascend.sh
# run distributed training example
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]
# run evaluation example
bash run_standalone_eval_ascend.sh [CKPT_FILE_PATH]
```

- Run on GPU

```bash
cd scripts
# run training example
bash run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]
# run distributed training example
bash run_distribute_train_gpu.sh [CONFIG_PATH]
# run evaluation example
bash run_standalone_eval_gpu.sh [CKPT_FILE_PATH] [CONFIG_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└─c3d
  ├── README.md                           // descriptions about C3D
  ├── scripts
  │   ├──run_dataset_preprocess.sh       // shell script for preprocessing dataset
  │   ├──run_ckpt_convert.sh             // shell script for converting pytorch ckpt file to pickle file on GPU
  │   ├──run_distribute_train_ascend.sh  // shell script for distributed training on Ascend
  │   ├──run_distribute_train_gpu.sh  // shell script for distributed training on GPU
  │   ├──run_infer_310.sh                // shell script for inference on 310
  │   ├──run_standalone_train_ascend.sh  // shell script for training on Ascend
  │   ├──run_standalone_train_gpu.sh  // shell script for training on GPU
  │   ├──run_standalone_eval_ascend.sh   // shell script for testing on Ascend
  │   ├──run_standalone_eval_gpu.sh   // shell script for testing on GPU
  ├── src
  │   ├──dataset.py                    // creating dataset
  │   ├──evalcallback.py               // evalcallback
  │   ├──lr_schedule.py                // learning rate scheduler
  │   ├──transform.py                  // handle dataset
  │   ├──loss.py                       // loss
  │   ├──utils.py                      // General components (callback function)
  │   ├──c3d_model.py                  // Unet3D model
          ├── model_utils
          │   ├──config.py                    // parameter configuration
          │   ├──device_adapter.py            // device adapter
          │   ├──local_adapter.py             // local adapter
          │   ├──moxing_adapter.py            // moxing adapter
          ├── tools
          │   ├──ckpt_convert.py          // convert pytorch ckpt file to pickle file
          │   ├──dataset_preprocess.py    // preprocess dataset
  ├── default_config.yaml               // parameter configuration
  ├── default_config_gpu.yaml           // parameter configuration on gpu
  ├── requirements.txt                  // requirements configuration
  ├── export.py                         // convert mindspore ckpt file to MINDIR file
  ├── train.py                          // evaluation script
  ├── eval.py                           // training script
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in default_config.yaml

- config for C3D, HMDB51 dataset

```text
# ==============================================================================
# Device
device_target:     "Ascend"

# Dataset Setup
clip_length:       16                    # Number of frames within a clip
clip_offset:       0                     # Frame offset between beginning of video and clip (1st clip only)
clip_stride:       1                     # Frame offset between successive clips, must be >= 1
crop_shape:        [112,112]             # (Height, Width) of frame
crop_type:         Random                # Type of cropping operation (Random, Central and None)
final_shape:       [112,112]             # (Height, Width) of input to be given to CNN
num_clips:         -1                    # Number clips to be generated from a video (<0: uniform sampling, 0: Divide entire video into clips, >0: Defines number of clips)
random_offset:     0                     # Boolean switch to generate a clip length sized clip from a video
resize_shape:      [128,171]             # (Height, Width) to resize original data
subtract_mean:     ''                    # Subtract mean (R,G,B) from all frames during preprocessing
proprocess:        default               # String argument to select preprocessing type

# Experiment Setup
model:             C3D                   # Name of model to be loaded
acc_metric:        Accuracy              # Accuracy metric
batch_size:        16                    # Numbers of videos in a mini-batch
dataset:           HMDB51                # Name of dataset, 'HMDB51' or 'UCF101'
epoch:             30                    # Total number of epochs
gamma:             0.1                   # Multiplier with which to change learning rate
json_path:         ./JSON/hmdb51/        # Path to the json file for the given dataset
img_path:          ./Data/hmdb-51_img/   # Path to the img file for the given dataset
sport1m_mean_file_path: ./pretrained_model/sport1m_train16_128_mean.npy # Sport1m data distribution information
num_classes:       51                    # Number of total classes in the dataset, 51 for HMDB51 and 101 for UCF101
loss_type:         M_XENTROPY            # Loss function
lr:                1e-4                  # Learning rate
milestones:        [10, 20]              # Epoch values to change learning rate
momentum:          0.9                   # Momentum value in optimizer
loss_scale:        1.0
num_workers:       8                     # Number of CPU worker used to load data
opt:               sgd                   # Name of optimizer
pre_trained:       1                     # Load pretrained network
save_dir:          ./results             # Path to results directory
seed:              666                   # Seed for reproducibility
weight_decay:      0.0005                # Weight decay

# Train Setup
is_distributed:    0                     # Distributed training or not
is_save_on_master: 1                     # Only save ckpt on master device
device_id:         0                     # Device id
ckpt_path:         ./result/ckpt         # Ckpt save path
ckpt_interval:     1                     # Saving ckpt interval
keep_checkpoint_max: 40                  # Max ckpt file number

# Modelarts Setup
enable_modelarts:  0                     # Is training on modelarts
result_url:        ''                    # Result save path
data_url:          ''                    # Data path
train_url:         ''                    # Train path

# Export Setup
ckpt_file:         './C3D.ckpt'          # Mindspore ckpt file path
mindir_file_name:  './C3D'               # Save file path
file_format:       'mindir'              # Save file format
# ==============================================================================

```

Parameters for both training and evaluation can be set in default_config_gpu.yaml

- config for C3D, UCF101 dataset

```text
# ==============================================================================
# Device
device_target:     "GPU"

# Dataset Setup
clip_length:       16                    # Number of frames within a clip
clip_offset:       0                     # Frame offset between beginning of video and clip (1st clip only)
clip_stride:       1                     # Frame offset between successive clips, must be >= 1
crop_shape:        [112,112]             # (Height, Width) of frame
crop_type:         Random                # Type of cropping operation (Random, Central and None)
final_shape:       [112,112]             # (Height, Width) of input to be given to CNN
num_clips:         -1                    # Number clips to be generated from a video (<0: uniform sampling, 0: Divide entire video into clips, >0: Defines number of clips)
random_offset:     0                     # Boolean switch to generate a clip length sized clip from a video
resize_shape:      [128,171]             # (Height, Width) to resize original data
subtract_mean:     ''                    # Subtract mean (R,G,B) from all frames during preprocessing

# Experiment Setup
model:             C3D                   # Name of model to be loaded
acc_metric:        Accuracy              # Accuracy metric
batch_size:        8                    # Numbers of videos in a mini-batch
dataset:           UCF101                # Name of dataset
epoch:             150                   # Total number of epochs
gamma:             0.1                   # Multiplier with which to change learning rate
json_path:         ./ucf101/JSON         # Path to the json file for the given dataset
img_path:          ./ucf101/UCF-101_img  # Path to the img file for the given dataset
num_classes:       101                   # Number of total classes in the dataset
loss_type:         M_XENTROPY            # Loss function
lr:                0.00025                  # Learning rate
milestones:        [15, 30, 75]              # Epoch values to change learning rate
momentum:          0.9                   # Momentum value in optimizer
loss_scale:        1.0
num_workers:       4                     # Number of CPU worker used to load data
opt:               sgd                   # Name of optimizer
pre_trained:       1                     # Load pretrained network
save_dir:          ./results             # Path to results directory
seed:              666                   # Seed for reproducibility
weight_decay:      0.0005                # Weight decay
sport1m_mean_file_path: ./pretrained_model/sport1m_train16_128_mean.npy   # Path to sports1m pretrained model
pre_trained_ckpt_path: ./pretrained_model/c3d-pretrained.ckpt   # Path to c3d pretrained model

# Train Setup
is_distributed:    0                     # Distributed training or not
is_evalcallback:   1                     # Using evalcallback or not
is_save_on_master: 1                     # Only save ckpt on master device
device_id:         0                     # Device id
ckpt_path:         ./result/ckpt         # Ckpt save path
ckpt_interval:     1                     # Saving ckpt interval
keep_checkpoint_max: 3                 # Max ckpt file number

# Modelarts Setup
enable_modelarts:  0                     # Is training on modelarts
result_url:        ''                    # Result save path
data_url:          ''                    # Data path
train_url:         ''                    # Train path

# Export Setup
ckpt_file:         './C3D.ckpt'          # Mindspore ckpt file path
mindir_file_name:  './C3D'               # Save file path
file_format:       'mindir'              # Save file format
# ==============================================================================

```

## [Training Process](#contents)

### Training

#### Training on Ascend

```text
# enter scripts directory
cd scripts
# training
bash run_standalone_train_ascend.sh
```

The python command above will run in the background, you can view the results through the file `eval.log`.

After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

- train.log for HMDB51

```shell
epoch: 1 step: 223, loss is 2.8705792
epoch time: 74139.530 ms, per step time: 332.464 ms
epoch: 2 step: 223, loss is 1.8403366
epoch time: 60084.907 ms, per step time: 269.439 ms
epoch: 3 step: 223, loss is 1.4866445
epoch time: 61095.684 ms, per step time: 273.972 ms
...
epoch: 29 step: 223, loss is 0.3037338
epoch time: 60436.915 ms, per step time: 271.018 ms
epoch: 30 step: 223, loss is 0.2176594
epoch time: 60130.695 ms, per step time: 269.644 ms
```

- train.log for UCF101

```shell
epoch: 1 step: 596, loss is 0.53118783
epoch time: 170693.634 ms, per step time: 286.399 ms
epoch: 2 step: 596, loss is 0.51934457
epoch time: 150388.783 ms, per step time: 252.330 ms
epoch: 3 step: 596, loss is 0.07241724
epoch time: 151548.857 ms, per step time: 254.277 ms
...
epoch: 29 step: 596, loss is 0.034661677
epoch time: 150932.542 ms, per step time: 253.243 ms
epoch: 30 step: 596, loss is 0.0048465515
epoch time: 150760.797 ms, per step time: 252.954 ms
```

#### Training on GPU

> Notes:If you occur a problem with the information:
> “Bad performance attention, it takes more than 25 seconds to fetch and send a batch of data into device, which might result `GetNext` timeout problem.“
> Please change the Parameter "dataset_sink_mode" to False

```text
# enter scripts directory
cd scripts
# training
bash run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]
```

The above shell script will run distribute training in the background. You can view the results through the file `./train[X].log`. The loss value will be achieved as follows:

- train.log for HMDB51

```shell
epoch: 1 step: 446, loss is 2.5606058
epoch time: 233965.570 ms, per step time: 298.655 ms
epoch: 2 step: 446, loss is 2.526373
epoch time: 217318.708 ms, per step time: 251.523 ms
epoch: 3 step: 446, loss is 1.2750342
epoch time: 218027.155 ms, per step time: 232.700 ms
...
epoch: 29 step: 446, loss is 0.5408877
epoch time: 217762.113 ms, per step time: 220.512 ms
epoch: 30 step: 446, loss is 0.52792186
epoch time: 218174.035 ms, per step time: 228.359 ms
```

- train.log for UCF101

```shell
epoch: 1 step: 1192, loss is 0.8381556
epoch time: 593197.024 ms, per step time: 301.297 ms
epoch: 2 step: 1192, loss is 0.5701107
epoch time: 576058.976 ms, per step time: 260.542 ms
epoch: 3 step: 1192, loss is 0.1724325
epoch time: 578041.281 ms, per step time: 235.868 ms
...
epoch: 99 step: 1192, loss is 6.3519354e-05
epoch time: 573493.252 ms, per step time: 225.237 ms
epoch: 100 step: 1192, loss is 4.852382e-05
epoch time: 575237.743 ms, per step time: 229.164 ms
```

### Distributed Training

#### Distributed training on Ascend

> Notes:
> RANK_TABLE_FILE can refer to [Link](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) , and the device_ip can be got as [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools). For large models like InceptionV4, it's better to export an external environment variable `export HCCL_CONNECT_TIMEOUT=600` to extend hccl connection checking time from the default 120 seconds to 600 seconds. Otherwise, the connection could be timeout since compiling time increases with the growth of model size.
>

```text
# enter scripts directory
cd scripts
# distributed training
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]
```

The above shell script will run distribute training in the background. You can view the results through the file `./train[X].log`. The loss value will be achieved as follows:

- train0.log for HMDB51

```shell
epoch: 1 step: 223, loss is 3.1376421
epoch time: 103888.471 ms, per step time: 465.868 ms
epoch: 2 step: 223, loss is 1.4189289
epoch time: 13111.264 ms, per step time: 58.795 ms
epoch: 3 step: 223, loss is 3.3972843
epoch time: 13140.217 ms, per step time: 58.925 ms
...
epoch: 29 step: 223, loss is 0.099318035
epoch time: 13094.187 ms, per step time: 58.718 ms
epoch: 30 step: 223, loss is 0.00515177
epoch time: 13101.518 ms, per step time: 58.751 ms
```

- train0.log for UCF101

```shell
epoch: 1 step: 596, loss is 0.51830626
epoch time: 82401.300 ms, per step time: 138.257 ms
epoch: 2 step: 596, loss is 0.5527372
epoch time: 30820.129 ms, per step time: 51.712 ms
epoch: 3 step: 596, loss is 0.007791209
epoch time: 30809.803 ms, per step time: 51.694 ms
...
epoch: 29 step: 596, loss is 7.510604e-05
epoch time: 30809.334 ms, per step time: 51.694 ms
epoch: 30 step: 596, loss is 0.13138217
epoch time: 30819.966 ms, per step time: 51.711 ms
```

#### Distributed training on GPU

```text
# enter scripts directory
cd scripts
# distributed training
vim run_distribute_train_gpu.sh to set start_device_id
bash run_distribute_train_gpu.sh [CONFIG_PATH]
```

- train_distributed.log for HMDB51

```shell
epoch: 1 step: 55, loss is 3.084101915359497
epoch: 1 step: 55, loss is 3.001408100128174
epoch: 1 step: 55, loss is 2.5211687088012695
epoch: 1 step: 55, loss is 2.9400177001953125
epoch: 1 step: 55, loss is 3.598146677017212
epoch: 1 step: 55, loss is 2.37894344329834
epoch: 1 step: 55, loss is 1.7159693241119385
epoch: 1 step: 55, loss is 2.2993032932281494
epoch time: 24400.126 ms, per step time: 443.639 ms
epoch time: 25218.947 ms, per step time: 458.526 ms
epoch time: 24354.300 ms, per step time: 442.805 ms
epoch time: 25144.561 ms, per step time: 457.174 ms
epoch time: 21726.022 ms, per step time: 395.019 ms
epoch time: 24929.812 ms, per step time: 453.269 ms
epoch time: 25210.959 ms, per step time: 458.381 ms
epoch time: 24913.004 ms, per step time: 452.964 ms
epoch: 2 step: 55, loss is 2.177777051925659
epoch: 2 step: 55, loss is 2.5466907024383545
epoch: 2 step: 55, loss is 1.9837493896484375
epoch: 2 step: 55, loss is 1.4902374744415283
epoch: 2 step: 55, loss is 2.3385510444641113
epoch: 2 step: 55, loss is 1.3293176889419556
epoch: 2 step: 55, loss is 1.8483796119689941
epoch: 2 step: 55, loss is 0.8703243136405945
epoch time: 23577.773 ms, per step time: 428.687 ms
epoch time: 23325.131 ms, per step time: 424.093 ms
epoch time: 23254.383 ms, per step time: 422.807 ms
epoch time: 22038.587 ms, per step time: 400.702 ms
epoch time: 21795.474 ms, per step time: 396.281 ms
epoch time: 22424.821 ms, per step time: 407.724 ms
epoch time: 23331.409 ms, per step time: 424.207 ms
epoch time: 22178.912 ms, per step time: 403.253 ms
...
```

- train_distributed.log for UCF101

```shell
epoch: 1 step: 149, loss is 0.97137051820755
epoch: 1 step: 149, loss is 1.1462825536727905
epoch: 1 step: 149, loss is 1.484191656112671
epoch: 1 step: 149, loss is 0.639738142490387
epoch: 1 step: 149, loss is 1.1133722066879272
epoch: 1 step: 149, loss is 1.5043989419937134
epoch: 1 step: 149, loss is 1.2063453197479248
epoch: 1 step: 149, loss is 1.3174564838409424
epoch time: 183002.444 ms, per step time: 1228.204 ms
epoch time: 183388.214 ms, per step time: 1230.793 ms
epoch time: 183560.571 ms, per step time: 1231.950 ms
epoch time: 183881.357 ms, per step time: 1234.103 ms
epoch time: 184225.004 ms, per step time: 1236.409 ms
epoch time: 184383.710 ms, per step time: 1237.475 ms
epoch time: 184501.011 ms, per step time: 1238.262 ms
epoch time: 184885.520 ms, per step time: 1240.842 ms
epoch: 2 step: 149, loss is 0.10039880871772766
epoch: 2 step: 149, loss is 0.5981963276863098
epoch: 2 step: 149, loss is 0.4604840576648712
epoch: 2 step: 149, loss is 0.215419739484787
epoch: 2 step: 149, loss is 0.2556331753730774
epoch: 2 step: 149, loss is 0.03653889149427414
epoch: 2 step: 149, loss is 1.4467300176620483
epoch: 2 step: 149, loss is 1.0422033071517944
epoch time: 53143.686 ms, per step time: 356.669 ms
epoch time: 52175.739 ms, per step time: 350.173 ms
epoch time: 54300.036 ms, per step time: 364.430 ms
epoch time: 53026.808 ms, per step time: 355.885 ms
epoch time: 52941.203 ms, per step time: 355.310 ms
epoch time: 53144.090 ms, per step time: 356.672 ms
epoch time: 53896.009 ms, per step time: 361.718 ms
epoch time: 53584.895 ms, per step time: 359.630 ms
...
```

## [Evaluation Process](#contents)

### Evaluation

#### Evaluating on Ascend

- evaluation on dataset when running on Ascend

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "username/ckpt_0/c3d-hmdb51-0-30_223.ckpt".

```text
# enter scripts directory
cd scripts
# eval
bash run_standalone_eval_ascend.sh [CKPT_FILE_PATH]
```

The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

- eval.log for HMDB51

```text
start create network
pre_trained model: username/ckpt_0/c3d-hmdb51-0-30_223.ckpt
setep: 1/96, acc: 0.3125
setep: 21/96, acc: 0.375
setep: 41/96, acc: 0.9375
setep: 61/96, acc: 0.6875
setep: 81/96, acc: 0.4375
eval result: top_1 50.196%
```

- eval.log for UCF101

```text
start create network
pre_trained model: username/ckpt_0/c3d-ucf101-0-30_596.ckpt
setep: 1/237, acc: 0.625
setep: 21/237, acc: 1.0
setep: 41/237, acc: 0.5625
setep: 61/237, acc: 1.0
setep: 81/237, acc: 0.6875
setep: 101/237, acc: 1.0
setep: 121/237, acc: 0.5625
setep: 141/237, acc: 0.5
setep: 161/237, acc: 1.0
setep: 181/237, acc: 1.0
setep: 201/237, acc: 0.75
setep: 221/237, acc: 1.0
eval result: top_1 79.381%
```

#### Evaluating on GPU

- evaluation on dataset when running on GPU

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "./results/xxxx-xx-xx_time_xx_xx_xx/ckpt_0/0-30_223.ckpt".

```text
# enter scripts directory
cd scripts
# eval
bash run_standalone_eval_gpu.sh [CKPT_FILE_PATH] [CONFIG_PATH]
```

- eval.log for HMDB51

```text
start create network
pre_trained model:./results/2021-10-21_time_11_16_48/ckpt_0/0-30_223.ckpt
setep: 1/96, acc: 0.6875
setep: 21/96, acc: 0.6875
setep: 41/96, acc: 0.4375
setep: 61/96, acc: 0.3125
setep: 81/96, acc: 0.75
eval result: top_1 50.327%
```

- eval.log for UCF101

```text
start create network
pre_trained model: ./results/2021-11-02_time_07_30_42/ckpt_0/0-85_223.ckpt
setep: 1/237, acc: 0.75
setep: 21/237, acc: 1.0
setep: 41/237, acc: 0.625
setep: 61/237, acc: 1.0
setep: 81/237, acc: 0.875
setep: 101/237, acc: 1.0
setep: 121/237, acc: 0.9375
setep: 141/237, acc: 0.5625
setep: 161/237, acc: 1.0
setep: 181/237, acc: 1.0
setep: 201/237, acc: 0.5625
setep: 221/237, acc: 1.0
eval result: top_1 80.412%
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --mindir_file_name [FILE_NAME] --file_format [FILE_FORMAT] --num_classes [NUM_CLASSES] --batch_size [BATCH_SIZE]
```

- `ckpt_file` parameter is mandotory.
- `file_format` should be in ["AIR", "MINDIR"].
- `NUM_CLASSES` Number of total classes in the dataset, 51 for HMDB51 and 101 for UCF101.
- `BATCH_SIZE` Since currently mindir does not support dynamic shapes, this network only supports inference with batch_size of 1.

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET` must be 'HMDB51' or 'UCF101'.
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

- C3D for HMDB51

| Parameters          | Ascend                                                     | GPU                                                   |
| ------------------- | ---------------------------------------------------------  |-------------------------------------------------------|
| Model Version       | C3D                                                        | C3D                                                   |
| Resource            | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8| V100                                                  |
| uploaded Date       | 09/22/2021 (month/day/year)                                | 11/06/2021 (month/day/year)                           |
| MindSpore Version   | 1.2.0                                                      | 1.5.0                                                 |
| Dataset             | HMDB51                                                     | HMDB51                                                |
| Training Parameters | epoch = 30,  batch_size = 16                               | epoch = 150,  batch_size = 8                          |
| Optimizer           | SGD                                                        | SGD                                                   |
| Loss Function       | Max_SoftmaxCrossEntropyWithLogits                          | Max_SoftmaxCrossEntropyWithLogits                     |
| Speed               | 1pc: 270.694ms/step                                        | 1pc:226.299ms/step                                    |
| Top_1               | 1pc: 49.78%                                                | 1pc:50.131%                                           |
| Total time          | 1pc: 0.32hours                                             | 1pc:1.3hours                                          |
| Parameters (M)      | 78                                                         | 78

- C3D for UCF101

| Parameters          | Ascend                                                     | GPU                                                       |
| ------------------- | ---------------------------------------------------------  |---------------------------------------------------------  |
| Model Version       | C3D                                                        | C3D                                                       |
| Resource            | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8| V100                                                      |
| uploaded Date       | 09/22/2021 (month/day/year)                                | 11/06/2021 (month/day/year)                               |
| MindSpore Version   | 1.2.0                                                      | 1.5.0                                                     |
| Dataset             | UCF101                                                     | UCF101                                                    |
| Training Parameters | epoch = 30, batch_size = 16                                | epoch = 150,  batch_size = 8                              |
| Optimizer           | SGD                                                        | SGD                                                       |
| Loss Function       | Max_SoftmaxCrossEntropyWithLogits                          | Max_SoftmaxCrossEntropyWithLogits                         |
| Speed               | 1pc: 253.372ms/step                                        | 1pc:237.128ms/step                                            |
| Top_1               | 1pc: 80.33%                                                | 1pc:80.138%                                               |
| Total time          | 1pc: 1.31hours                                             | 1pc:4hours                                                |
| Parameters (M)      | 78                                                         | 78

# [Description of Random Situation](#contents)

We set random seed to 666 in default_config.yaml and default_config_gpu.yaml

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
